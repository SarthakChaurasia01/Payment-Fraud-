"""PaymentShield AI - transaction fraud screening and investigation workbench.

Educational project built on simulated PaySim mobile-money data.

The app loads the saved model at models/fraud_detection_model.pkl, validates
transaction fields, prepares the exact seven training features, and reports the
model's own estimated fraud probability. It never retrains, replaces, or
adjusts the model or its outputs.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import importlib
import io
import platform
import re
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitAPIException

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME = "PaymentShield AI"
TAGLINE = "Transaction intelligence. Risk visibility. Human-led investigation."

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.pkl"

RAW_COLUMNS: tuple[str, ...] = (
    "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
)
NUMERIC_COLUMNS: tuple[str, ...] = RAW_COLUMNS[1:]
SUPPORTED_TYPES: tuple[str, ...] = ("CASH_OUT", "TRANSFER")
# Exact feature order used when the saved model was trained.
FEATURE_ORDER: tuple[str, ...] = (
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "type_CASH_OUT", "type_TRANSFER",
)
ROW_ID = "row_number"
SCORE_COLUMNS: tuple[str, ...] = ("fraud_probability", "fraud_probability_pct", "risk_band", "decision")
ERROR_COLUMNS: tuple[str, ...] = (ROW_ID, "field", "issue", "value")

# Leave as None to derive the fraud class from model.classes_ (recommended).
# Set it (for example to 1) only after confirming in your notebook which target
# label means "fraud". The app never guesses from a column position.
CONFIRMED_FRAUD_LABEL: Any = None

MAX_BATCH_ROWS = 250_000
PREDICT_CHUNK = 25_000
DISPLAY_ROW_CAP = 5_000
SCATTER_CAP = 5_000
DEFAULT_THRESHOLD = 0.50
NOTE_MAX_CHARS = 2_000

# Risk bands are interface categories only. They are fixed probability ranges
# and are not validated estimates of real-world fraud likelihood.
BAND_NAMES: tuple[str, ...] = ("Low", "Moderate", "High", "Critical")
BAND_EDGES: tuple[float, ...] = (0.25, 0.50, 0.80)
BAND_RANGES: dict[str, str] = {
    "Low": "below 25%", "Moderate": "25% to under 50%", "High": "50% to under 80%", "Critical": "80% and above",
}

DECISION_FLAGGED = "Flagged for review"
DECISION_CLEAR = "Not flagged"

STATUS_PENDING = "Pending review"
STATUS_UNDER = "Under review"
STATUS_CLEARED = "Reviewed \u2014 no concern identified"
STATUS_ESCALATED = "Escalated for further investigation"
REVIEW_STATUSES: tuple[str, ...] = (STATUS_PENDING, STATUS_UNDER, STATUS_CLEARED, STATUS_ESCALATED)
OPEN_STATUSES: tuple[str, ...] = (STATUS_PENDING, STATUS_UNDER)

ISSUE_MISSING = "Missing value"
ISSUE_NOT_NUMBER = "Not a valid number"
ISSUE_NONFINITE = "Non-finite number"
ISSUE_NEGATIVE = "Negative value"
ISSUE_TYPE = "Unsupported transaction type"

PAGES: tuple[str, ...] = (
    "Overview", "Batch screening", "Investigation", "Single transaction",
    "Review queue", "Analytics", "Exports", "About",
)

C = {
    "bg": "#0A0F1F", "panel": "#10172B", "raise": "#16203A", "line": "#23304E",
    "text": "#E8ECF6", "muted": "#93A0BD", "accent": "#3CC8F0", "grid": "#1B2745",
    "Low": "#4CC9A7", "Moderate": "#E6B94A", "High": "#F28C4B", "Critical": "#F0566A",
    "flag": "#F0566A", "clear": "#5B77B5",
}
FONT_STACK = '"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,"Helvetica Neue",Roboto,Arial,sans-serif'

# ---------------------------------------------------------------------------
# Model loading and verification
# ---------------------------------------------------------------------------


class ModelProblem(Exception):
    """A model-file problem with a plain-language explanation and a fix."""

    def __init__(self, title: str, detail: str, fix: str) -> None:
        super().__init__(title)
        self.title, self.detail, self.fix = title, detail, fix


class PredictionError(RuntimeError):
    """Raised when the saved model cannot produce valid probabilities."""


@dataclass(frozen=True)
class ModelInfo:
    model: Any
    type_name: str
    classes: tuple
    fraud_label: Any
    fraud_index: int
    label_basis: str
    n_features: Optional[int]
    feature_names: Optional[tuple]
    size_kb: float
    versions: dict
    load_warnings: tuple


@dataclass(frozen=True)
class ModelLoad:
    info: Optional[ModelInfo] = None
    problem: Optional[ModelProblem] = None


_FRAUD_NAMES = {"fraud", "fraudulent", "isfraud", "is_fraud", "true", "yes"}
_LEGIT_NAMES = {"legit", "legitimate", "not_fraud", "notfraud", "non_fraud", "nonfraud", "normal", "genuine", "false", "no"}


def _label_key(label: Any) -> Any:
    """Normalise a class label so 1, 1.0, True and '1' compare equal."""
    if isinstance(label, (bool, np.bool_)):
        return int(label)
    if isinstance(label, (int, np.integer)):
        return int(label)
    if isinstance(label, (float, np.floating)) and float(label).is_integer():
        return int(label)
    if isinstance(label, str):
        text = label.strip().lower()
        return int(text) if text.lstrip("-").isdigit() else text
    return repr(label)


def resolve_fraud_label(classes: Sequence[Any]) -> tuple[Any, int, str]:
    """Return (fraud_label, column_index, basis). Raises ModelProblem if unknown.

    The column index is found by looking the label up in ``classes``; it is never
    assumed from a position.
    """
    labels = list(classes)
    if len(labels) != 2:
        raise ModelProblem(
            "The model is not a two-class classifier",
            f"model.classes_ contains {len(labels)} labels: {labels!r}.",
            "PaymentShield AI needs a binary classifier (legitimate vs fraud). Re-check which file is saved at models/fraud_detection_model.pkl.",
        )
    keys = [_label_key(label) for label in labels]
    if CONFIRMED_FRAUD_LABEL is not None:
        hits = [i for i, key in enumerate(keys) if key == _label_key(CONFIRMED_FRAUD_LABEL)]
        if len(hits) != 1:
            raise ModelProblem(
                "CONFIRMED_FRAUD_LABEL does not match the model's classes",
                f"CONFIRMED_FRAUD_LABEL = {CONFIRMED_FRAUD_LABEL!r}, but model.classes_ = {labels!r}.",
                "Set CONFIRMED_FRAUD_LABEL in app.py to one of the model's class labels, or set it back to None.",
            )
        return labels[hits[0]], hits[0], "Confirmed explicitly through CONFIRMED_FRAUD_LABEL in app.py."
    if set(keys) == {0, 1}:
        index = keys.index(1)
        return labels[index], index, (
            "Class labels are {0, 1}. Label 1 is the PaySim isFraud flag (1 = fraud). The probability column is "
            "found by looking this label up in model.classes_, not by assuming a column position."
        )
    fraud_hits = [i for i, key in enumerate(keys) if key in _FRAUD_NAMES]
    legit_hits = [i for i, key in enumerate(keys) if key in _LEGIT_NAMES]
    if len(fraud_hits) == 1 and len(legit_hits) == 1:
        index = fraud_hits[0]
        return labels[index], index, f"The class label {labels[index]!r} is recognised by name as the fraud class."
    raise ModelProblem(
        "The fraud class could not be identified",
        f"model.classes_ = {labels!r}. These labels do not clearly say which one means fraud.",
        "Open your notebook, confirm how the target column was encoded, then set CONFIRMED_FRAUD_LABEL near the top of app.py "
        "(for example CONFIRMED_FRAUD_LABEL = 1). Until then the app will not score transactions.",
    )


def _estimator_candidates(model: Any) -> list[Any]:
    candidates = [model]
    steps = getattr(model, "steps", None)
    if steps:
        candidates.append(steps[-1][1])
    return candidates


def _expected_feature_count(model: Any) -> Optional[int]:
    for target in _estimator_candidates(model):
        count = getattr(target, "n_features_in_", None)
        if isinstance(count, (int, np.integer)):
            return int(count)
        get_booster = getattr(target, "get_booster", None)
        if callable(get_booster):
            try:
                return int(get_booster().num_features())
            except Exception:  # booster may not be fitted or may be incompatible
                continue
    return None


def _stored_feature_names(model: Any) -> Optional[tuple]:
    for target in _estimator_candidates(model):
        names = getattr(target, "feature_names_in_", None)
        if names is None:
            get_booster = getattr(target, "get_booster", None)
            if callable(get_booster):
                try:
                    names = get_booster().feature_names
                except Exception:
                    names = None
        if names is not None and len(names) > 0:
            names = tuple(str(n) for n in names)
            if all(re.fullmatch(r"f\d+", n) for n in names):
                return None  # generic placeholder names carry no information
            return names
    return None


def _installed_versions() -> dict:
    versions = {
        "python": platform.python_version(), "streamlit": st.__version__, "pandas": pd.__version__,
        "numpy": np.__version__, "joblib": joblib.__version__,
    }
    for name in ("xgboost", "sklearn"):
        try:
            versions[name] = str(importlib.import_module(name).__version__)
        except Exception:
            versions[name] = "not installed"
    return versions


def load_model_from_path(path: Path) -> ModelLoad:
    """Load and verify the saved model. Never substitutes another model."""
    try:
        return ModelLoad(info=_load_and_verify(path))
    except ModelProblem as problem:
        return ModelLoad(problem=problem)


def _load_and_verify(path: Path) -> ModelInfo:
    if not path.is_file():
        raise ModelProblem(
            "Model file not found",
            "Expected the saved model at models/fraud_detection_model.pkl (relative to app.py).",
            "Copy your existing fraud_detection_model.pkl into the models folder next to app.py, then reload.",
        )
    with path.open("rb") as handle:
        head = handle.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise ModelProblem(
            "The model file is a Git LFS pointer, not the model",
            "The file contains a small text pointer instead of the real model data.",
            "Commit the model as a regular file (not through Git LFS), or fetch the LFS object before running the app.",
        )
    # Only this repository's own model file is ever deserialised. Pickle files
    # can run code when loaded, so uploaded files are never passed to joblib.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = joblib.load(path)
        except ModuleNotFoundError as exc:
            raise ModelProblem(
                "A package required by the model is not installed",
                f"{type(exc).__name__}: {exc}",
                "Install the missing package (for an XGBoost model: python -m pip install xgboost scikit-learn) "
                "and use the same versions as in your training environment.",
            ) from exc
        except Exception as exc:  # pickle can raise many different errors
            raise ModelProblem(
                "The model file could not be loaded",
                f"{type(exc).__name__}: {exc}",
                "Common causes: an incomplete or corrupted file, or installed package versions that differ from the "
                "ones used when the model was saved. Check xgboost and scikit-learn versions in your training environment.",
            ) from exc
    load_warnings = tuple(f"{w.category.__name__}: {w.message}" for w in caught)

    if isinstance(model, (dict, list, tuple)) or not hasattr(model, "predict"):
        keys = f" Dictionary keys: {sorted(map(str, model.keys()))}." if isinstance(model, dict) else ""
        raise ModelProblem(
            "The file does not contain a classifier object",
            f"Loaded object of type {type(model).__name__}.{keys}",
            "Save the trained classifier itself (for example joblib.dump(model, path)), not a wrapper container.",
        )
    if not callable(getattr(model, "predict_proba", None)):
        raise ModelProblem(
            "The model cannot produce probabilities",
            f"{type(model).__name__} has no predict_proba method.",
            "PaymentShield AI reports probabilities and will not invent them. Use a classifier that supports predict_proba.",
        )
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ModelProblem(
            "The model does not expose class labels",
            "model.classes_ is missing, so the fraud class cannot be verified.",
            "Re-save the fitted classifier with a recent library version, or confirm the label encoding in your notebook.",
        )
    fraud_label, fraud_index, basis = resolve_fraud_label(list(np.asarray(classes).tolist()))

    n_features = _expected_feature_count(model)
    if n_features is not None and n_features != len(FEATURE_ORDER):
        raise ModelProblem(
            "Feature count mismatch",
            f"The model expects {n_features} features, but this app prepares {len(FEATURE_ORDER)}: {list(FEATURE_ORDER)}.",
            "Confirm the feature list used in training. The app does not change feature definitions to fit a different model.",
        )
    names = _stored_feature_names(model)
    if names is not None and names != FEATURE_ORDER:
        raise ModelProblem(
            "Feature names or order differ from training",
            f"Model feature names: {list(names)}. App feature order: {list(FEATURE_ORDER)}.",
            "Check the training notebook. If the model really was trained in a different order, update FEATURE_ORDER in app.py deliberately.",
        )

    info = ModelInfo(
        model=model, type_name=f"{type(model).__module__}.{type(model).__name__}",
        classes=tuple(np.asarray(classes).tolist()), fraud_label=fraud_label, fraud_index=fraud_index,
        label_basis=basis, n_features=n_features, feature_names=names,
        size_kb=path.stat().st_size / 1024, versions=_installed_versions(), load_warnings=load_warnings,
    )
    # Compatibility probe: confirms predict_proba runs in this environment.
    # Its output is discarded and never shown as a result.
    probe = pd.DataFrame({
        "type": ["TRANSFER", "CASH_OUT"], "amount": [1.0, 1.0], "oldbalanceOrg": [1.0, 1.0],
        "newbalanceOrig": [0.0, 0.0], "oldbalanceDest": [0.0, 0.0], "newbalanceDest": [1.0, 1.0],
    })
    try:
        score_features(info, prepare_features(probe))
    except PredictionError as exc:
        raise ModelProblem(
            "The model loaded but failed a compatibility check",
            str(exc),
            "This is usually a version mismatch. Install the xgboost / scikit-learn versions used in training.",
        ) from exc
    return info


@st.cache_resource(show_spinner="Loading saved model")
def _cached_load(path_str: str, mtime_ns: int, size: int) -> ModelLoad:
    return load_model_from_path(Path(path_str))


def get_model() -> ModelLoad:
    try:
        stat = MODEL_PATH.stat()
        key = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        key = (0, 0)
    return _cached_load(str(MODEL_PATH), *key)


def retry_model_load() -> None:
    _cached_load.clear()


# ---------------------------------------------------------------------------
# Validation and feature preparation (shared by batch and single scoring)
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    fatal: Optional[str]
    rows_read: int
    column_map: dict
    clean: pd.DataFrame
    errors: pd.DataFrame

    @property
    def rows_valid(self) -> int:
        return len(self.clean)

    @property
    def rows_rejected(self) -> int:
        return int(self.errors[ROW_ID].nunique()) if len(self.errors) else 0


def _empty_errors() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype="object") for name in ERROR_COLUMNS})


def _fatal(message: str, rows_read: int = 0) -> ValidationResult:
    return ValidationResult(message, rows_read, {}, pd.DataFrame(columns=list(RAW_COLUMNS)), _empty_errors())


def _blank_mask(series: pd.Series) -> pd.Series:
    return (series.isna() | series.fillna("").str.strip().eq("")).astype(bool)


def validate_transactions(df: pd.DataFrame) -> ValidationResult:
    """Validate the six raw fields. Rows with problems are reported, never dropped silently."""
    if df is None or df.shape[1] == 0:
        return _fatal("The file has no columns. Check that it is a comma-separated CSV with a header row.")
    df = df.reset_index(drop=True)
    rows_read = len(df)
    if rows_read == 0:
        return _fatal("The file has a header but no transaction rows.")
    if rows_read > MAX_BATCH_ROWS:
        return _fatal(f"The file has {rows_read:,} rows. The limit is {MAX_BATCH_ROWS:,} rows per batch. Split the file and screen the parts separately.", rows_read)

    lookup: dict[str, list] = {}
    for col in df.columns:
        lookup.setdefault(str(col).strip().lower(), []).append(col)
    column_map: dict = {}
    missing, duplicated = [], []
    for canonical in RAW_COLUMNS:
        found = lookup.get(canonical.lower(), [])
        if not found:
            missing.append(canonical)
        elif len(found) > 1:
            duplicated.append(canonical)
        else:
            column_map[canonical] = found[0]
    if missing:
        return _fatal(
            f"Missing required column(s): {', '.join(missing)}. Columns found: {', '.join(map(str, df.columns))}. "
            "Header names are matched ignoring case and surrounding spaces. PaySim spells them oldbalanceOrg and newbalanceOrig.",
            rows_read,
        )
    if duplicated:
        return _fatal(f"These required columns appear more than once: {', '.join(duplicated)}. Keep one of each.", rows_read)

    work = pd.DataFrame({c: df[column_map[c]].astype("string") for c in RAW_COLUMNS})
    positions = np.arange(rows_read)
    frames: list[pd.DataFrame] = []

    def add_issue(mask: pd.Series, field: str, issue: str) -> None:
        selected = mask.to_numpy(dtype=bool)
        if selected.any():
            frames.append(pd.DataFrame({
                ROW_ID: positions[selected] + 1, "field": field, "issue": issue,
                "value": work[field].fillna("").to_numpy()[selected].astype(str),
            }))

    type_text = work["type"].fillna("").str.strip().str.upper()
    type_blank = _blank_mask(work["type"])
    add_issue(type_blank, "type", ISSUE_MISSING)
    add_issue(~type_blank & ~type_text.isin(list(SUPPORTED_TYPES)), "type", ISSUE_TYPE)

    numbers: dict[str, pd.Series] = {}
    for col in NUMERIC_COLUMNS:
        text = work[col].fillna("").str.strip().astype(object)
        values = pd.to_numeric(text, errors="coerce").astype("float64")
        blank = _blank_mask(work[col])
        finite = np.isfinite(values)
        add_issue(blank, col, ISSUE_MISSING)
        add_issue(~blank & values.isna(), col, ISSUE_NOT_NUMBER)
        add_issue(~blank & ~values.isna() & ~finite, col, ISSUE_NONFINITE)
        add_issue(finite & (values < 0), col, ISSUE_NEGATIVE)
        numbers[col] = values

    errors = _empty_errors()
    valid = np.ones(rows_read, dtype=bool)
    if frames:
        errors = pd.concat(frames, ignore_index=True).sort_values([ROW_ID, "field"], kind="stable").reset_index(drop=True)
        valid[errors[ROW_ID].to_numpy() - 1] = False

    clean = pd.DataFrame({"type": type_text.astype(object)})
    for col in NUMERIC_COLUMNS:
        clean[col] = numbers[col]
    clean = clean.loc[valid].copy()
    clean["type"] = clean["type"].astype(str)
    return ValidationResult(None, rows_read, column_map, clean, errors)


def prepare_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Build the seven model features in the exact training order."""
    absent = [c for c in RAW_COLUMNS if c not in clean.columns]
    if absent:
        raise ValueError(f"Cannot prepare features, missing columns: {absent}")
    type_text = clean["type"].astype(str).str.strip().str.upper()
    unsupported = sorted(set(type_text) - set(SUPPORTED_TYPES))
    if unsupported:
        raise ValueError(f"Unsupported transaction type(s) reached feature preparation: {unsupported}")
    features = pd.DataFrame(index=clean.index)
    for col in NUMERIC_COLUMNS:
        features[col] = clean[col].astype("float64")
    features["type_CASH_OUT"] = (type_text == "CASH_OUT").astype("int64")
    features["type_TRANSFER"] = (type_text == "TRANSFER").astype("int64")
    return features.loc[:, list(FEATURE_ORDER)]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _model_input(features: pd.DataFrame, feature_names: Optional[tuple]) -> Any:
    ordered = features.loc[:, list(FEATURE_ORDER)]
    return ordered if feature_names is not None else ordered.to_numpy(dtype=np.float64)


def score_features(
    info: ModelInfo, features: pd.DataFrame, progress: Optional[Callable[[float], None]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fraud probability per row, raw predict_proba preview of the first rows)."""
    total = len(features)
    result = np.empty(total, dtype=np.float64)
    preview: Optional[np.ndarray] = None
    for start in range(0, total, PREDICT_CHUNK):
        end = min(start + PREDICT_CHUNK, total)
        try:
            proba = np.asarray(info.model.predict_proba(_model_input(features.iloc[start:end], info.feature_names)))
        except Exception as exc:
            raise PredictionError(
                f"The saved model failed while predicting ({type(exc).__name__}): {exc}. "
                "If this mentions features or dtypes, compare the diagnostics panel with your training setup."
            ) from exc
        if proba.ndim != 2 or proba.shape != (end - start, len(info.classes)):
            raise PredictionError(f"Unexpected predict_proba shape {proba.shape}; expected {(end - start, len(info.classes))}.")
        column = proba[:, info.fraud_index].astype(np.float64)
        if not np.all(np.isfinite(column)) or column.min() < -1e-6 or column.max() > 1 + 1e-6:
            raise PredictionError("The model returned probabilities outside the 0 to 1 range or non-finite values.")
        result[start:end] = np.clip(column, 0.0, 1.0)
        if preview is None:
            preview = proba[:5].copy()
        if progress is not None:
            progress(end / total)
    if preview is None:
        preview = np.empty((0, len(info.classes)))
    # Rounding to 10 places removes float32-to-float64 conversion noise and is
    # far below float32 resolution, so it does not alter the model's output.
    return np.round(result, 10), preview


def band_for(probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(BAND_NAMES, dtype=object)[np.digitize(probabilities, BAND_EDGES)]


def build_scored_frame(raw: pd.DataFrame, validation: ValidationResult, probabilities: np.ndarray) -> pd.DataFrame:
    """Original fields of valid rows plus probability, percent and risk band."""
    raw = raw.reset_index(drop=True)
    kept = raw.loc[validation.clean.index].copy()
    kept = kept.rename(columns={original: canonical for canonical, original in validation.column_map.items()})
    reserved = {ROW_ID, *SCORE_COLUMNS}
    kept = kept.rename(columns={c: f"{c}_input" for c in kept.columns if c in reserved})
    for col in RAW_COLUMNS:
        kept[col] = validation.clean[col].to_numpy()
    extras = [c for c in kept.columns if c not in RAW_COLUMNS]
    out = kept.loc[:, list(RAW_COLUMNS) + extras].reset_index(drop=True)
    out.insert(0, ROW_ID, validation.clean.index.to_numpy() + 1)
    out["fraud_probability"] = probabilities
    out["fraud_probability_pct"] = np.round(probabilities * 100, 4)
    out["risk_band"] = band_for(probabilities)
    return out


def apply_decisions(base: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add the threshold-based decision. Probabilities are never touched."""
    out = base.copy()
    out["decision"] = np.where(out["fraud_probability"].to_numpy() >= threshold, DECISION_FLAGGED, DECISION_CLEAR)
    return out


def balance_observations(row: dict) -> dict:
    amount, o_old, o_new = float(row["amount"]), float(row["oldbalanceOrg"]), float(row["newbalanceOrig"])
    d_old, d_new = float(row["oldbalanceDest"]), float(row["newbalanceDest"])
    return {
        "origin_change": o_new - o_old, "dest_change": d_new - d_old,
        "origin_gap": amount + (o_new - o_old), "dest_gap": (d_new - d_old) - amount,
        "origin_after_zero": o_new == 0.0, "dest_before_zero": d_old == 0.0,
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt_num(value: float) -> str:
    return f"{value:,.2f}"


def fmt_prob(p: float) -> str:
    if p <= 0:
        return "0.00%"
    if p * 100 < 0.01:
        return "<0.01%"
    return f"{p * 100:.2f}%"


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def call_stretch(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call a Streamlit element at full width across old and new Streamlit APIs."""
    for extra in ({"width": "stretch"}, {"use_container_width": True}):
        try:
            return fn(*args, **kwargs, **extra)
        except (TypeError, StreamlitAPIException):
            continue
    return fn(*args, **kwargs)


def csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise spreadsheet formula injection in text cells before export."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype) in ("string", "str"):
            out[col] = out[col].map(
                lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r") else v
            )
    return out


@st.cache_data(show_spinner=False, max_entries=12)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return csv_safe(df).to_csv(index=False).encode("utf-8")


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Styling and reusable UI pieces
# ---------------------------------------------------------------------------
CSS = """<style>
:root{--bg:#0A0F1F;--panel:#10172B;--raise:#16203A;--line:#23304E;--text:#E8ECF6;--muted:#93A0BD;--accent:#3CC8F0;
--low:#4CC9A7;--mod:#E6B94A;--high:#F28C4B;--crit:#F0566A;}
html,body,.stApp,.stApp button,.stApp input,.stApp textarea,.stApp select{font-family:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,"Helvetica Neue",Roboto,Arial,sans-serif;}
.stApp{background:var(--bg);}
[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1240px;}
[data-testid="stSidebar"]{background:#0C1326;border-right:1px solid var(--line);}
[data-testid="stSidebar"] .block-container{padding-top:1.2rem;}
h1,h2,h3{letter-spacing:-0.01em;}
.ps-brand{display:flex;align-items:center;gap:12px;margin-bottom:4px;}
.ps-mark{width:30px;height:34px;background:var(--accent);clip-path:polygon(50% 0%,100% 14%,100% 55%,50% 100%,0% 55%,0% 14%);display:flex;align-items:center;justify-content:center;}
.ps-mark i{display:block;width:16px;height:19px;background:#0A0F1F;clip-path:polygon(50% 0%,100% 14%,100% 55%,50% 100%,0% 55%,0% 14%);}
.ps-name{font-size:1.12rem;font-weight:650;color:var(--text);letter-spacing:-0.015em;}
.ps-tag{color:var(--muted);font-size:.78rem;line-height:1.4;margin:2px 0 14px 0;}
.ps-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:22px;}
.ps-head h1{font-size:1.9rem;font-weight:650;margin:0;padding:0;color:var(--text);line-height:1.2;}
.ps-head p{color:var(--muted);margin:6px 0 0 0;font-size:.95rem;max-width:62ch;}
.ps-badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.76rem;font-weight:550;border:1px solid var(--line);color:var(--muted);background:var(--panel);white-space:nowrap;}
.ps-badge.accent{color:var(--accent);border-color:rgba(60,200,240,.4);background:rgba(60,200,240,.08);}
.ps-badge.Low{color:var(--low);border-color:rgba(76,201,167,.4);background:rgba(76,201,167,.08);}
.ps-badge.Moderate{color:var(--mod);border-color:rgba(230,185,74,.4);background:rgba(230,185,74,.08);}
.ps-badge.High{color:var(--high);border-color:rgba(242,140,75,.4);background:rgba(242,140,75,.08);}
.ps-badge.Critical,.ps-badge.flag{color:var(--crit);border-color:rgba(240,86,106,.45);background:rgba(240,86,106,.1);}
.ps-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;min-height:112px;}
.ps-card .lbl{color:var(--muted);font-size:.82rem;font-weight:500;}
.ps-card .val{color:var(--text);font-size:1.7rem;font-weight:620;letter-spacing:-0.02em;font-variant-numeric:tabular-nums;line-height:1.2;margin-top:8px;}
.ps-card .sub{color:var(--muted);font-size:.8rem;margin-top:6px;line-height:1.4;}
.ps-panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:12px;}
.ps-panel h4{margin:0 0 8px 0;font-size:1rem;font-weight:620;color:var(--text);}
.ps-panel p,.ps-panel li{color:var(--muted);font-size:.9rem;line-height:1.55;}
.ps-panel .kv{display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px solid var(--line);font-size:.88rem;}
.ps-panel .kv:last-child{border-bottom:none;}
.ps-panel .kv span{color:var(--muted);} .ps-panel .kv b{color:var(--text);font-weight:560;text-align:right;font-variant-numeric:tabular-nums;}
.ps-empty{border:1px dashed #33436A;border-radius:14px;padding:38px 28px;text-align:center;background:rgba(16,23,43,.6);margin-bottom:14px;}
.ps-empty h3{margin:0 0 8px 0;font-size:1.25rem;color:var(--text);}
.ps-empty p{margin:0 auto;color:var(--muted);max-width:56ch;line-height:1.55;}
.ps-note{border-left:3px solid var(--accent);background:rgba(60,200,240,.06);padding:10px 14px;border-radius:0 8px 8px 0;color:var(--muted);font-size:.86rem;line-height:1.5;margin:8px 0 14px 0;}
.ps-ledger{display:grid;grid-template-columns:1fr 150px 1fr;gap:0;align-items:stretch;margin:6px 0 18px 0;}
.ps-acct{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;}
.ps-acct .t{color:var(--text);font-weight:600;font-size:.92rem;margin-bottom:8px;}
.ps-acct .r{display:flex;justify-content:space-between;padding:4px 0;font-size:.88rem;color:var(--muted);}
.ps-acct .r b{color:var(--text);font-weight:560;font-variant-numeric:tabular-nums;}
.ps-acct .r.d{border-top:1px solid var(--line);margin-top:4px;padding-top:8px;}
.ps-flow{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 10px;text-align:center;}
.ps-flow .amt{color:var(--accent);font-weight:620;font-size:.98rem;font-variant-numeric:tabular-nums;}
.ps-flow .ln{position:relative;height:2px;background:var(--accent);width:100%;margin:10px 0;opacity:.7;}
.ps-flow .ln:after{content:"";position:absolute;right:-1px;top:-4px;border-left:8px solid var(--accent);border-top:5px solid transparent;border-bottom:5px solid transparent;}
.ps-flow .typ{color:var(--muted);font-size:.78rem;}
.ps-meter{margin:6px 0 4px 0;padding-top:26px;}
.ps-track{position:relative;height:12px;border-radius:6px;background:var(--raise);overflow:visible;}
.ps-seg{position:absolute;top:0;height:12px;opacity:.28;}
.ps-fill{position:absolute;left:0;top:0;height:12px;border-radius:6px 0 0 6px;background:var(--accent);opacity:.9;}
.ps-dot{position:absolute;top:-3px;width:18px;height:18px;margin-left:-9px;border-radius:50%;background:var(--text);border:3px solid var(--bg);box-shadow:0 0 0 2px var(--accent);}
.ps-thr{position:absolute;top:-8px;width:2px;height:28px;margin-left:-1px;background:var(--crit);}
.ps-thr-lbl{position:absolute;top:-30px;transform:translateX(-50%);font-size:.75rem;color:var(--crit);white-space:nowrap;font-weight:560;}
.ps-scale{position:relative;height:20px;margin-top:10px;}
.ps-scale span{position:absolute;transform:translateX(-50%);font-size:.72rem;color:var(--muted);font-variant-numeric:tabular-nums;}
.ps-pipe{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0 6px 0;}
.ps-step{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:.84rem;color:var(--text);}
.ps-sep{color:var(--accent);font-size:1rem;}
.ps-footer{margin-top:42px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:.78rem;line-height:1.5;}
@media (max-width:900px){.ps-ledger{grid-template-columns:1fr;gap:10px;}.ps-flow{padding:8px 0;}.ps-head{flex-direction:column;}}
</style>"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(text: str, kind: str = "", tip: str = "") -> str:
    title = f' title="{esc(tip)}"' if tip else ""
    return f'<span class="ps-badge {esc(kind)}"{title}>{esc(text)}</span>'


def estimate_badge() -> str:
    return badge("Model-based estimate", "accent", "A probability produced by the saved model. It is not a verified fraud rate.")


def page_header(title: str, subtitle: str, show_badge: bool = True) -> None:
    right = estimate_badge() if show_badge else ""
    html_block(f'<div class="ps-head"><div><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div><div>{right}</div></div>')


def card_html(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="ps-card"><div class="lbl">{esc(label)}</div><div class="val">{esc(value)}</div>'
            f'<div class="sub">{esc(sub)}</div></div>')


def metric_row(items: Sequence[tuple[str, str, str]]) -> None:
    for column, (label, value, sub) in zip(st.columns(len(items)), items):
        column.markdown(card_html(label, value, sub), unsafe_allow_html=True)


def note(text: str) -> None:
    html_block(f'<div class="ps-note">{esc(text)}</div>')


def empty_state(title: str, body: str, button: str, target: str, key: str) -> None:
    html_block(f'<div class="ps-empty"><h3>{esc(title)}</h3><p>{esc(body)}</p></div>')
    left, middle, right = st.columns([2, 1, 2])
    call_stretch(middle.button, button, type="primary", on_click=goto, args=(target,), key=key)


def kv_panel(title: str, pairs: Sequence[tuple[str, str]]) -> None:
    rows = "".join(f'<div class="kv"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in pairs)
    html_block(f'<div class="ps-panel"><h4>{esc(title)}</h4>{rows}</div>')


def meter_html(p: float, threshold: float) -> str:
    """Probability marker on a 0-100% track with the review threshold shown."""
    pos, thr = max(0.0, min(1.0, p)) * 100, max(0.0, min(1.0, threshold)) * 100
    edges = [0.0, *[e * 100 for e in BAND_EDGES], 100.0]
    segments = "".join(
        f'<span class="ps-seg" style="left:{edges[i]:.2f}%;width:{edges[i + 1] - edges[i]:.2f}%;background:{C[name]}"></span>'
        for i, name in enumerate(BAND_NAMES)
    )
    ticks = "".join(
        f'<span style="left:{v:.0f}%">{v:.0f}%</span>' for v in [0.0, *[e * 100 for e in BAND_EDGES], 100.0]
    )
    return (
        '<div class="ps-meter"><div class="ps-track">' + segments +
        f'<span class="ps-fill" style="width:{pos:.3f}%"></span>'
        f'<span class="ps-thr" style="left:{thr:.2f}%"></span>'
        f'<span class="ps-thr-lbl" style="left:{thr:.2f}%">Threshold {thr:.0f}%</span>'
        f'<span class="ps-dot" style="left:{pos:.3f}%"></span></div>'
        f'<div class="ps-scale">{ticks}</div></div>'
    )


def ledger_html(row: dict) -> str:
    obs = balance_observations(row)

    def acct(title: str, before: float, after: float, change: float) -> str:
        return (f'<div class="ps-acct"><div class="t">{title}</div>'
                f'<div class="r"><span>Balance before</span><b>{fmt_num(before)}</b></div>'
                f'<div class="r"><span>Balance after</span><b>{fmt_num(after)}</b></div>'
                f'<div class="r d"><span>Change</span><b>{change:+,.2f}</b></div></div>')

    return (
        '<div class="ps-ledger">'
        + acct("Origin account", float(row["oldbalanceOrg"]), float(row["newbalanceOrig"]), obs["origin_change"])
        + f'<div class="ps-flow"><div class="amt">{fmt_num(float(row["amount"]))}</div><div class="ln"></div>'
          f'<div class="typ">{esc(row["type"])}</div></div>'
        + acct("Destination account", float(row["oldbalanceDest"]), float(row["newbalanceDest"]), obs["dest_change"])
        + '</div>'
    )


def decision_badge(decision: str) -> str:
    return badge(decision, "flag" if decision == DECISION_FLAGGED else "")


def render_footer() -> None:
    html_block(
        '<div class="ps-footer">PaymentShield AI is an educational project built on simulated PaySim data. '
        'It is not connected to any bank or payment network and must not be used for real financial decisions. '
        'Flagged transactions need human review; unflagged transactions are not confirmed safe.</div>'
    )


def style_figure(fig: go.Figure, height: int = 340, title: str = "", legend: bool = False) -> go.Figure:
    layout: dict = dict(
        height=height, margin=dict(l=8, r=8, t=44 if title else 14, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], family=FONT_STACK, size=12), showlegend=legend,
        hoverlabel=dict(bgcolor=C["raise"], bordercolor=C["line"], font=dict(color=C["text"])),
    )
    if title:
        layout["title"] = dict(text=title, x=0, xanchor="left", font=dict(size=14))
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=C["grid"], zerolinecolor=C["grid"], linecolor=C["line"])
    fig.update_yaxes(gridcolor=C["grid"], zerolinecolor=C["grid"], linecolor=C["line"])
    return fig


def show_figure(fig: go.Figure, key: str) -> None:
    call_stretch(st.plotly_chart, fig, key=key)


# ---------------------------------------------------------------------------
# Session state and workflow helpers
# ---------------------------------------------------------------------------
SINGLE_DEFAULTS = {
    "single_type": "TRANSFER", "single_amount": 0.0, "single_oldorg": 0.0,
    "single_neworg": 0.0, "single_olddest": 0.0, "single_newdest": 0.0,
}


def init_state() -> None:
    defaults: dict = {
        "nav": "Overview", "threshold": DEFAULT_THRESHOLD, "batch": None, "review": {},
        "single": None, "upload_cache": None, "selected_row": None, "decorated": None,
        **SINGLE_DEFAULTS,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def goto(page: str, row: Optional[int] = None) -> None:
    st.session_state["nav"] = page
    if row is not None:
        st.session_state["selected_row"] = int(row)
        st.session_state["wb_pick"] = int(row)


def reset_single() -> None:
    for key, value in SINGLE_DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["single"] = None


def clear_batch() -> None:
    st.session_state.update(batch=None, review={}, decorated=None, selected_row=None, upload_cache=None)
    st.session_state.pop("wb_pick", None)


def current_scored() -> Optional[pd.DataFrame]:
    """Scored batch with decisions for the current threshold (no model re-run)."""
    batch = st.session_state.get("batch")
    if batch is None:
        return None
    threshold = float(st.session_state["threshold"])
    cache = st.session_state.get("decorated")
    if cache and cache["batch_id"] == batch["id"] and cache["threshold"] == threshold:
        return cache["df"]
    df = apply_decisions(batch["base"], threshold)
    st.session_state["decorated"] = {"batch_id": batch["id"], "threshold": threshold, "df": df}
    return df


def get_row(df: pd.DataFrame, row_number: int) -> Optional[pd.Series]:
    hits = np.flatnonzero(df[ROW_ID].to_numpy() == row_number)
    return df.iloc[hits[0]] if hits.size else None


def review_entry(row_number: int) -> dict:
    return st.session_state["review"].get(int(row_number), {})


def status_for(row_number: int, flagged: bool) -> str:
    stored = review_entry(row_number).get("status")
    if stored:
        return stored
    return STATUS_PENDING if flagged else "Not in queue"


def save_review(row_number: int, status: Optional[str] = None, note_text: Optional[str] = None) -> None:
    entry = dict(review_entry(row_number))
    if status is not None:
        entry["status"] = status
    if note_text is not None:
        entry["note"] = note_text.strip()[:NOTE_MAX_CHARS]
    entry["updated_at"] = now_stamp()
    st.session_state["review"][int(row_number)] = entry


def with_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flagged = (out["decision"] == DECISION_FLAGGED).to_numpy()
    out["review_status"] = [status_for(r, f) for r, f in zip(out[ROW_ID].tolist(), flagged)]
    out["review_note"] = [review_entry(r).get("note", "") for r in out[ROW_ID].tolist()]
    return out


def row_dict(series: pd.Series) -> dict:
    return {c: series[c] for c in RAW_COLUMNS}


def table_config(df: pd.DataFrame) -> dict:
    cfg: dict = {}
    for col in ("amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"):
        if col in df.columns:
            cfg[col] = st.column_config.NumberColumn(col, format="%.2f")
    if "fraud_probability_pct" in df.columns:
        cfg["fraud_probability_pct"] = st.column_config.ProgressColumn(
            "Probability (%)", min_value=0.0, max_value=100.0, format="%.2f",
            help="Model-based estimate, not a verified fraud rate.")
    if "fraud_probability" in df.columns:
        cfg["fraud_probability"] = st.column_config.NumberColumn("fraud_probability", format="%.6f")
    return cfg


def show_table(df: pd.DataFrame, key: str, **kwargs: Any) -> Any:
    shown = df.head(DISPLAY_ROW_CAP)
    if len(df) > DISPLAY_ROW_CAP:
        st.caption(f"Showing the first {DISPLAY_ROW_CAP:,} of {len(df):,} rows. Use the download for the full set.")
    return call_stretch(st.dataframe, shown, hide_index=True, column_config=table_config(shown), key=key, **kwargs)


def render_diagnostics(load: ModelLoad, diag: Optional[dict] = None, key: str = "diag") -> None:
    with st.expander("Technical diagnostics"):
        if load.info is None:
            st.write("The model is not loaded, so there is nothing to show yet.")
            if load.problem is not None:
                st.code(f"{load.problem.title}\n{load.problem.detail}")
            return
        info = load.info
        rows = [
            ("Model class", info.type_name),
            ("Class labels (model.classes_)", repr(list(info.classes))),
            ("Identified fraud class", repr(info.fraud_label)),
            ("Fraud probability column index", str(info.fraud_index)),
            ("How the fraud class was identified", info.label_basis),
            ("Expected feature count", str(info.n_features) if info.n_features is not None else "Not reported by the model"),
            ("Feature names stored in model", ", ".join(info.feature_names) if info.feature_names else "None stored"),
            ("Prepared feature order", ", ".join(FEATURE_ORDER)),
            ("Model file size", f"{info.size_kb:,.1f} KB"),
            ("Installed versions", ", ".join(f"{k} {v}" for k, v in info.versions.items())),
        ]
        st.table(pd.DataFrame(rows, columns=["Property", "Value"]).set_index("Property"))
        if info.load_warnings:
            st.warning("Warnings raised while loading the model:\n\n" + "\n\n".join(f"- {w}" for w in info.load_warnings))
        if diag:
            st.markdown("**Prepared features (first rows scored)**")
            st.dataframe(diag["features_preview"], hide_index=True)
            st.markdown("**Raw predict_proba output (first rows scored)**")
            labels = [f"P(class {c!r})" for c in info.classes]
            st.dataframe(pd.DataFrame(diag["raw_preview"], columns=labels), hide_index=True)
        else:
            st.caption("Score a batch or a single transaction to see prepared features and raw model output here.")


# ---------------------------------------------------------------------------
# Sidebar and shared page chrome
# ---------------------------------------------------------------------------


def render_sidebar(load: ModelLoad) -> None:
    with st.sidebar:
        html_block(
            f'<div class="ps-brand"><div class="ps-mark"><i></i></div><div class="ps-name">{APP_NAME}</div></div>'
            f'<div class="ps-tag">{esc(TAGLINE)}</div>'
        )
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        st.slider(
            "Review threshold", min_value=0.01, max_value=0.99, step=0.01, format="%.2f", key="threshold",
            help="Transactions with an estimated fraud probability at or above this value are flagged for human review. "
                 "Changing it changes decisions only. The model's probabilities stay exactly the same.",
        )
        st.caption("Configurable demo setting, not a validated operating policy.")
        batch = st.session_state.get("batch")
        if batch is not None:
            p = batch["base"]["fraud_probability"].to_numpy()
            flagged = int((p >= float(st.session_state["threshold"])).sum())
            st.caption(f"At this threshold, {flagged:,} of {len(p):,} scored transactions are flagged.")
        st.divider()
        if load.info is not None:
            html_block(badge("Model loaded", "Low"))
            st.caption(f"{load.info.type_name.split('.')[-1]}, fraud class {load.info.fraud_label!r}")
            if load.info.load_warnings:
                st.caption("Version warnings were raised while loading. See About, Technical diagnostics.")
        else:
            html_block(badge("Model unavailable", "flag"))


def render_model_banner(load: ModelLoad) -> None:
    if load.problem is None:
        return
    problem = load.problem
    st.error(f"**{problem.title}**\n\n{problem.detail}\n\n**What to do:** {problem.fix}")
    st.button("Retry loading model", on_click=retry_model_load, key="retry_model")


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def page_overview(load: ModelLoad) -> None:
    page_header("Overview", "Where the current batch stands at the selected review threshold.")
    df = current_scored()
    if df is None:
        empty_state(
            "No batch screened yet",
            "Upload a CSV with the six transaction fields and the saved model will score it. "
            "Nothing is shown here until you do, because this app keeps no historical data.",
            "Go to batch screening", "Batch screening", "ov_empty",
        )
        return
    batch = st.session_state["batch"]
    n = len(df)
    flagged = int((df["decision"] == DECISION_FLAGGED).sum())
    top = df.loc[df["fraud_probability"].idxmax()]
    rejected = batch["rows_read"] - n
    metric_row([
        ("Transactions scored", f"{n:,}", f"{batch['rows_read']:,} rows read, {rejected:,} rejected"),
        ("Flagged for review", f"{flagged:,}", f"{flagged / n:.2%} of scored transactions"),
        ("Mean estimated probability", fmt_prob(float(df["fraud_probability"].mean())), "Average across the batch"),
        ("Highest estimated probability", fmt_prob(float(top["fraud_probability"])), f"Row {int(top[ROW_ID])}"),
        ("Review threshold", f"{float(st.session_state['threshold']) * 100:.0f}%", "Flag at or above this probability"),
    ])
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        counts = df["risk_band"].value_counts().reindex(BAND_NAMES, fill_value=0)
        fig = go.Figure(go.Bar(
            x=counts.to_numpy(), y=list(BAND_NAMES), orientation="h", text=[f"{v:,}" for v in counts.to_numpy()],
            textposition="outside", cliponaxis=False, marker=dict(color=[C[b] for b in BAND_NAMES]),
            hovertemplate="%{y}: %{x:,} transactions<extra></extra>",
        ))
        style_figure(fig, 300, "Transactions by risk band")
        fig.update_yaxes(autorange="reversed")
        show_figure(fig, "ov_bands")
        note("Risk bands are fixed probability ranges used to organise the interface. "
             "They are not validated estimates of real-world fraud likelihood, and they do not move with the threshold.")
    with right:
        type_counts = df["type"].value_counts()
        kv_panel("Latest screening", [
            ("File", str(batch["source"])), ("Scored at", batch["scored_at"]),
            ("Rows read", f"{batch['rows_read']:,}"), ("Rows scored", f"{n:,}"), ("Rows rejected", f"{rejected:,}"),
            *[(t, f"{int(type_counts.get(t, 0)):,} transactions") for t in SUPPORTED_TYPES],
        ])
        st.button("Open the highest-scoring transaction", on_click=goto, args=("Investigation", int(top[ROW_ID])), key="ov_open_top")
    st.markdown("##### Highest-scoring transactions")
    cols = [ROW_ID, "type", "amount", "fraud_probability_pct", "risk_band", "decision"]
    show_table(df.nlargest(5, "fraud_probability")[cols], "ov_top")


# ---------------------------------------------------------------------------
# Batch screening
# ---------------------------------------------------------------------------


def read_csv_upload(uploaded: Any) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    data = uploaded.getvalue()
    if not data.strip():
        return None, "The file is empty. Upload a CSV that has a header row and at least one transaction."
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, skipinitialspace=True, encoding=encoding)
            return df, None
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return None, "The file has no readable content. Check that it is a comma-separated CSV with a header row."
        except pd.errors.ParserError as exc:
            return None, f"The CSV could not be parsed: {exc}"
    return None, "The file could not be decoded as text. Save it as UTF-8 CSV and try again."


def load_upload(uploaded: Any) -> dict:
    key = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
    cache = st.session_state.get("upload_cache")
    if cache and cache["key"] == key:
        return cache
    raw, error = read_csv_upload(uploaded)
    validation = validate_transactions(raw) if raw is not None else None
    cache = {"key": key, "name": uploaded.name, "raw": raw, "error": error, "validation": validation}
    st.session_state["upload_cache"] = cache
    return cache


def render_validation(validation: ValidationResult) -> None:
    metric_row([
        ("Rows read", f"{validation.rows_read:,}", "Rows found in the file"),
        ("Ready to score", f"{validation.rows_valid:,}", "Passed every check"),
        ("Rejected", f"{validation.rows_rejected:,}", "Not scored, listed below"),
    ])
    if validation.rows_rejected == 0:
        st.success("Every row passed validation.")
        return
    errors = validation.errors
    st.warning(f"{validation.rows_rejected:,} row(s) were not scored. Nothing was dropped silently; every reason is listed below.")
    summary = errors.groupby(["field", "issue"]).size().reset_index(name="rows").sort_values("rows", ascending=False)
    call_stretch(st.dataframe, summary, hide_index=True)
    unsupported = errors[errors["issue"] == ISSUE_TYPE]["value"].value_counts()
    if len(unsupported):
        listing = ", ".join(f"{v} ({c:,})" for v, c in unsupported.head(8).items())
        note(f"Unsupported transaction types found: {listing}. The saved model was trained only on "
             f"{' and '.join(SUPPORTED_TYPES)}. Other PaySim types are rejected rather than scored by a model that never saw them.")
    with st.expander("Rejected rows"):
        show_table(errors, "val_errors")
        st.download_button("Download validation errors (CSV)", to_csv_bytes(errors), "validation_errors.csv", "text/csv", key="dl_val_errors")


def run_screening(info: ModelInfo, cache: dict) -> None:
    validation: ValidationResult = cache["validation"]
    bar = st.progress(0.0, text="Preparing features")
    try:
        features = prepare_features(validation.clean)
        bar.progress(0.05, text="Scoring with the saved model")
        probabilities, preview = score_features(
            info, features, progress=lambda f: bar.progress(min(1.0, 0.05 + 0.95 * f), text=f"Scoring with the saved model ({f:.0%})"))
    except (PredictionError, ValueError) as exc:
        bar.empty()
        st.error(f"Screening stopped. {exc}")
        return
    bar.empty()
    base = build_scored_frame(cache["raw"], validation, probabilities)
    st.session_state["batch"] = {
        "id": uuid.uuid4().hex, "source": cache["name"], "scored_at": now_stamp(), "base": base,
        "errors": validation.errors, "rows_read": validation.rows_read,
        "diag": {"features_preview": features.head(5).reset_index(drop=True), "raw_preview": preview},
    }
    st.session_state.update(review={}, decorated=None, selected_row=None)
    st.session_state.pop("wb_pick", None)
    st.success(f"Scored {len(base):,} transactions with the saved model.")


def page_batch(load: ModelLoad) -> None:
    page_header("Batch screening", "Upload transactions, check them, and score them with the saved model.")
    info = load.info
    left, right = st.columns([3, 2])
    with right:
        kv_panel("Required columns", [
            ("type", "CASH_OUT or TRANSFER"), ("amount", "number, 0 or more"), ("oldbalanceOrg", "number, 0 or more"),
            ("newbalanceOrig", "number, 0 or more"), ("oldbalanceDest", "number, 0 or more"), ("newbalanceDest", "number, 0 or more"),
        ])
        st.download_button("Download CSV template", ",".join(RAW_COLUMNS) + "\n", "paymentshield_template.csv", "text/csv", key="dl_template")
    with left:
        uploaded = st.file_uploader("Transactions CSV", type=["csv"], key="batch_upload",
                                    help="Extra columns are kept in the output. Column names are matched ignoring case.")
        if st.session_state.get("batch") is not None:
            note("Screening a new file replaces the current batch and clears the review notes held in this session.")
    if uploaded is not None:
        cache = load_upload(uploaded)
        if cache["error"]:
            st.error(cache["error"])
        elif cache["validation"].fatal:
            st.error(cache["validation"].fatal)
        else:
            st.markdown("##### Preview")
            call_stretch(st.dataframe, cache["raw"].head(10), hide_index=True)
            render_validation(cache["validation"])
            ready = cache["validation"].rows_valid > 0 and info is not None
            if cache["validation"].rows_valid == 0:
                st.error("No rows are ready to score. Fix the issues above and upload the file again.")
            if st.button("Run screening", type="primary", disabled=not ready, key="run_screening"):
                run_screening(info, cache)
    if st.session_state.get("batch") is not None:
        render_batch_results(load)


def render_batch_results(load: ModelLoad) -> None:
    df = current_scored()
    batch = st.session_state["batch"]
    st.divider()
    head, action = st.columns([4, 1])
    head.markdown(f"#### Results for {batch['source']}")
    action.button("Clear batch", on_click=clear_batch, key="clear_batch")
    flagged = int((df["decision"] == DECISION_FLAGGED).sum())
    metric_row([
        ("Scored", f"{len(df):,}", batch["scored_at"]),
        ("Flagged for review", f"{flagged:,}", f"At threshold {float(st.session_state['threshold']):.2f}"),
        ("Highest estimated probability", fmt_prob(float(df["fraud_probability"].max())), "Model-based estimate"),
    ])
    if flagged == 0:
        st.info("No transactions reach the current threshold. These are the model's genuine scores. "
                "Lower the threshold in the sidebar, or open Technical diagnostics to inspect the raw output.")
    st.markdown("##### Filter and sort")
    f1, f2, f3 = st.columns(3)
    decisions = f1.multiselect("Decision", [DECISION_FLAGGED, DECISION_CLEAR], default=[DECISION_FLAGGED, DECISION_CLEAR], key="bf_decision")
    bands = f2.multiselect("Risk band", list(BAND_NAMES), default=list(BAND_NAMES), key="bf_band")
    present = sorted(df["type"].unique().tolist())
    types = f3.multiselect("Transaction type", present, default=present, key="bf_type")
    s1, s2, s3 = st.columns([2, 2, 1])
    low, high = s1.slider("Estimated probability range (%)", 0.0, 100.0, (0.0, 100.0), step=0.1, key="bf_range")
    sort_label = s2.selectbox("Sort by", ["Fraud probability", "Amount", "Row number"], key="bf_sort")
    descending = s3.selectbox("Order", ["Descending", "Ascending"], key="bf_order") == "Descending"
    pct = df["fraud_probability_pct"]
    mask = df["decision"].isin(decisions) & df["risk_band"].isin(bands) & df["type"].isin(types) & (pct >= low) & (pct <= high)
    sort_col = {"Fraud probability": "fraud_probability", "Amount": "amount", "Row number": ROW_ID}[sort_label]
    view = df[mask].sort_values(sort_col, ascending=not descending, kind="stable")
    st.caption(f"{len(view):,} of {len(df):,} scored transactions match the filters.")
    show_table(view, "batch_results")
    st.download_button("Download scored results (CSV)", to_csv_bytes(df), "scored_transactions.csv", "text/csv", key="dl_scored_batch")
    render_diagnostics(load, batch["diag"], "diag_batch")


# ---------------------------------------------------------------------------
# Investigation workbench
# ---------------------------------------------------------------------------


def lookup_row() -> None:
    value, df = st.session_state.get("wb_lookup"), current_scored()
    if value is None or df is None:
        return
    if get_row(df, int(value)) is not None:
        st.session_state["wb_lookup_msg"] = ""
        goto("Investigation", int(value))
    else:
        st.session_state["wb_lookup_msg"] = f"Row {int(value)} is not among the scored transactions. It may have been rejected during validation."


def page_investigation(load: ModelLoad) -> None:
    page_header("Investigation workbench", "Inspect one scored transaction and record a human review note.")
    df = current_scored()
    if df is None:
        empty_state("Nothing to investigate yet", "Screen a batch first, then open any transaction here.",
                    "Go to batch screening", "Batch screening", "wb_empty")
        return
    options = df.nlargest(500, "fraud_probability")[ROW_ID].astype(int).tolist()
    pick = st.session_state.get("wb_pick")
    if pick is not None and pick not in options and get_row(df, pick) is not None:
        options.append(int(pick))
    if st.session_state.get("wb_pick") not in options:
        st.session_state["wb_pick"] = options[0]
    summary = df[df[ROW_ID].isin(options)].set_index(ROW_ID)
    labels = {rn: f"Row {rn}, {summary.at[rn, 'type']}, amount {fmt_num(float(summary.at[rn, 'amount']))}, {fmt_prob(float(summary.at[rn, 'fraud_probability']))}"
              for rn in options}
    a, b = st.columns([3, 1])
    row_number = a.selectbox("Transaction (highest estimated probability first)", options, key="wb_pick", format_func=lambda r: labels[r])
    b.number_input("Look up any row number", min_value=1, step=1, value=None, key="wb_lookup", on_change=lookup_row)
    if st.session_state.get("wb_lookup_msg"):
        st.warning(st.session_state["wb_lookup_msg"])
    st.session_state["selected_row"] = int(row_number)
    row = get_row(df, int(row_number))
    data = row_dict(row)
    p, band, decision = float(row["fraud_probability"]), str(row["risk_band"]), str(row["decision"])
    flagged = decision == DECISION_FLAGGED
    threshold = float(st.session_state["threshold"])

    main_col, side_col = st.columns([3, 2])
    with main_col:
        html_block(ledger_html(data))
        html_block(f'<div class="ps-panel"><h4>Estimated fraud probability: {esc(fmt_prob(p))}</h4>{meter_html(p, threshold)}</div>')
    with side_col:
        kv_panel("Review summary", [
            ("Transaction type", str(row["type"])), ("Row number", str(int(row_number))),
            ("Estimated probability", fmt_prob(p)), ("Risk band", band), ("Decision", decision),
            ("Configured threshold", f"{threshold:.2f}"), ("Review status", status_for(int(row_number), flagged)),
        ])
    obs = balance_observations(data)
    st.markdown("##### Balance movements")
    o1, o2 = st.columns(2)
    with o1:
        kv_panel("Origin account", [
            ("Balance change", f"{obs['origin_change']:+,.2f}"),
            ("Amount minus origin decrease", f"{obs['origin_gap']:,.2f}"),
            ("Balance after is zero", "Yes" if obs["origin_after_zero"] else "No"),
        ])
    with o2:
        kv_panel("Destination account", [
            ("Balance change", f"{obs['dest_change']:+,.2f}"),
            ("Destination increase minus amount", f"{obs['dest_gap']:,.2f}"),
            ("Balance before is zero", "Yes" if obs["dest_before_zero"] else "No"),
        ])
    note("These figures are plain arithmetic on the transaction fields. They are not an explanation of the model's score, "
         "and in simulated data balances often do not reconcile. Review this transaction's balance movements and record what you find.")

    st.markdown("##### Investigation notes")
    entry = review_entry(int(row_number))
    stored_status = entry.get("status")
    with st.form(f"notes_form_{int(row_number)}"):
        status = st.selectbox("Review status", list(REVIEW_STATUSES),
                              index=list(REVIEW_STATUSES).index(stored_status) if stored_status in REVIEW_STATUSES else 0)
        text = st.text_area("Note", value=entry.get("note", ""), max_chars=NOTE_MAX_CHARS, height=120,
                            placeholder="What did you check, and what did you conclude?")
        saved = st.form_submit_button("Save note and status", type="primary")
    if saved:
        keep_status = None if (not flagged and not stored_status and status == STATUS_PENDING) else status
        save_review(int(row_number), status=keep_status, note_text=text)
        st.success("Saved for this session.")
    if entry.get("updated_at"):
        st.caption(f"Last updated {entry['updated_at']}")
    st.caption("Notes and statuses are held in this browser session only. They are lost when the session ends or the app restarts, "
               "and are cleared when a new file is screened. Download the review log from Exports to keep them.")
    st.button("Open the review queue", on_click=goto, args=("Review queue",), key="wb_to_queue")


# ---------------------------------------------------------------------------
# Single transaction scoring
# ---------------------------------------------------------------------------


def page_single(load: ModelLoad) -> None:
    page_header("Single transaction", "Score one transaction with the same checks and model used for batches.")
    info = load.info
    with st.form("single_form"):
        c1, c2 = st.columns(2)
        c1.selectbox("Transaction type", list(SUPPORTED_TYPES), key="single_type")
        c2.number_input("Amount", min_value=0.0, step=100.0, format="%.2f", key="single_amount")
        c3, c4 = st.columns(2)
        c3.number_input("Origin balance before", min_value=0.0, step=100.0, format="%.2f", key="single_oldorg")
        c4.number_input("Origin balance after", min_value=0.0, step=100.0, format="%.2f", key="single_neworg")
        c5, c6 = st.columns(2)
        c5.number_input("Destination balance before", min_value=0.0, step=100.0, format="%.2f", key="single_olddest")
        c6.number_input("Destination balance after", min_value=0.0, step=100.0, format="%.2f", key="single_newdest")
        submitted = st.form_submit_button("Score transaction", type="primary", disabled=info is None)
    st.button("Reset form", on_click=reset_single, key="single_reset")
    if submitted and info is not None:
        frame = pd.DataFrame([{
            "type": st.session_state["single_type"], "amount": st.session_state["single_amount"],
            "oldbalanceOrg": st.session_state["single_oldorg"], "newbalanceOrig": st.session_state["single_neworg"],
            "oldbalanceDest": st.session_state["single_olddest"], "newbalanceDest": st.session_state["single_newdest"],
        }])
        validation = validate_transactions(frame)
        if validation.fatal or validation.rows_rejected:
            st.error(validation.fatal or "; ".join(f"{r.field}: {r.issue}" for r in validation.errors.itertuples()))
        else:
            try:
                features = prepare_features(validation.clean)
                probabilities, preview = score_features(info, features)
            except (PredictionError, ValueError) as exc:
                st.error(f"Scoring stopped. {exc}")
            else:
                st.session_state["single"] = {
                    "row": {c: validation.clean.iloc[0][c] for c in RAW_COLUMNS}, "p": float(probabilities[0]),
                    "diag": {"features_preview": features.reset_index(drop=True), "raw_preview": preview}, "at": now_stamp(),
                }
    result = st.session_state.get("single")
    if result is None:
        note("Enter the six values and choose Score transaction. The result appears here.")
        return
    threshold = float(st.session_state["threshold"])
    p = result["p"]
    flagged = p >= threshold
    band = str(band_for(np.array([p]))[0])
    st.divider()
    left, right = st.columns([3, 2])
    with left:
        html_block(ledger_html(result["row"]))
        html_block(f'<div class="ps-panel"><h4>Estimated fraud probability: {esc(fmt_prob(p))}</h4>{meter_html(p, threshold)}</div>')
    with right:
        kv_panel("Screening result", [
            ("Estimated probability", fmt_prob(p)), ("Risk band", band),
            ("Decision", DECISION_FLAGGED if flagged else DECISION_CLEAR), ("Configured threshold", f"{threshold:.2f}"),
        ])
        obs = balance_observations(result["row"])
        kv_panel("Balance changes", [
            ("Origin balance change", f"{obs['origin_change']:+,.2f}"), ("Destination balance change", f"{obs['dest_change']:+,.2f}"),
        ])
    if flagged:
        st.warning("Flagged for review. The estimated probability is at or above your threshold, so a person should look at this transaction. "
                   "That does not mean it is fraudulent.")
    else:
        st.info("Not flagged. The estimated probability is below your threshold. That does not mean the transaction is safe.")
    note("The probability is the saved model's estimate for these six inputs. It is not a verified fraud rate, it does not explain why the "
         "model scored this way, and the PaySim training data is simulated. Changing the threshold changes the decision, never the probability.")
    render_diagnostics(load, result["diag"], "diag_single")


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------


def apply_queue_status(row_number: int) -> None:
    save_review(row_number, status=st.session_state["queue_status"])


def page_queue(load: ModelLoad) -> None:
    page_header("Review queue", "Flagged transactions, highest estimated probability first.")
    df = current_scored()
    if df is None:
        empty_state("The queue is empty", "Screen a batch to fill the review queue.", "Go to batch screening", "Batch screening", "q_empty")
        return
    flagged = df[df["decision"] == DECISION_FLAGGED].sort_values("fraud_probability", ascending=False, kind="stable")
    if flagged.empty:
        st.info(f"No transactions reach the threshold of {float(st.session_state['threshold']):.2f}. Lower it in the sidebar to review more.")
        return
    queue = with_review_columns(flagged)
    counts = queue["review_status"].value_counts()
    metric_row([(s, f"{int(counts.get(s, 0)):,}", "Human workflow status") for s in REVIEW_STATUSES])
    chosen = st.multiselect("Show statuses", list(REVIEW_STATUSES), default=list(REVIEW_STATUSES), key="q_filter")
    queue = queue[queue["review_status"].isin(chosen)]
    if queue.empty:
        st.info("No queue items match the selected statuses.")
        return
    cols = [ROW_ID, "type", "amount", "fraud_probability_pct", "risk_band", "review_status", "review_note"]
    view = queue[cols].head(DISPLAY_ROW_CAP).reset_index(drop=True)
    st.caption("Select a row to change its status or open it in the workbench. Statuses are set by people, not predicted by the model.")
    event = show_table(queue[cols], "queue_table", on_select="rerun", selection_mode="single-row")
    selected = list(event.selection.rows) if event is not None and hasattr(event, "selection") else []
    if selected and selected[0] < len(view):
        row_number = int(view.iloc[selected[0]][ROW_ID])
        a, b, c = st.columns([3, 1, 1])
        a.selectbox(f"Status for row {row_number}", list(REVIEW_STATUSES), key="queue_status")
        b.write("")
        b.button("Apply status", on_click=apply_queue_status, args=(row_number,), key="q_apply")
        c.write("")
        c.button("Open in workbench", on_click=goto, args=("Investigation", row_number), key="q_open")
    note("Statuses and notes are stored in this session only and are not a durable case-management system.")


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


def scatter_sample(df: pd.DataFrame) -> pd.DataFrame:
    flagged = df[df["decision"] == DECISION_FLAGGED]
    clear = df[df["decision"] != DECISION_FLAGGED]
    if len(flagged) > SCATTER_CAP // 2:
        flagged = flagged.sample(SCATTER_CAP // 2, random_state=7)
    room = SCATTER_CAP - len(flagged)
    if len(clear) > room:
        clear = clear.sample(room, random_state=7)
    return pd.concat([flagged, clear])


def page_analytics(load: ModelLoad) -> None:
    page_header("Analytics", "Charts built from the scored batch. Probabilities are estimates, not verified fraud rates.")
    df = current_scored()
    if df is None:
        empty_state("No data to chart", "Screen a batch to see distributions and breakdowns.", "Go to batch screening", "Batch screening", "an_empty")
        return
    threshold = float(st.session_state["threshold"])
    p_pct = df["fraud_probability"].to_numpy() * 100
    log_counts = st.toggle("Use a log scale for counts", value=True, key="an_log",
                           help="Fraud scores are usually concentrated near zero, so a log scale keeps small groups visible.")
    c1, c2 = st.columns(2)
    with c1:
        counts, edges = np.histogram(p_pct, bins=np.linspace(0, 100, 21))
        fig = go.Figure(go.Bar(
            x=(edges[:-1] + edges[1:]) / 2, y=counts, width=4.6, marker=dict(color=C["accent"]),
            customdata=np.column_stack([edges[:-1], edges[1:]]),
            hovertemplate="%{customdata[0]:.0f}% to %{customdata[1]:.0f}%<br>%{y:,} transactions<extra></extra>",
        ))
        style_figure(fig, 340, "Estimated probability distribution")
        fig.update_xaxes(title=dict(text="Estimated fraud probability (%)"), range=[0, 100])
        fig.update_yaxes(title=dict(text="Transactions"), type="log" if log_counts else "linear")
        fig.add_vline(x=threshold * 100, line_width=1, line_dash="dash", line_color=C["flag"])
        show_figure(fig, "an_hist")
        st.caption("The dashed line marks the review threshold. This chart shows model scores, not model accuracy.")
    with c2:
        flagged_n = int((df["decision"] == DECISION_FLAGGED).sum())
        values = [flagged_n, len(df) - flagged_n]
        fig = go.Figure(go.Bar(x=[DECISION_FLAGGED, DECISION_CLEAR], y=values, text=[f"{v:,}" for v in values], textposition="outside",
                               cliponaxis=False, marker=dict(color=[C["flag"], C["clear"]]),
                               hovertemplate="%{x}: %{y:,}<extra></extra>"))
        style_figure(fig, 340, "Flagged and not flagged")
        fig.update_yaxes(title=dict(text="Transactions"), type="log" if log_counts and min(values) > 0 else "linear")
        show_figure(fig, "an_flag")
        st.caption("A flagged transaction needs a human look. It is not confirmed fraud.")
    c3, c4 = st.columns(2)
    type_counts = df["type"].value_counts().reindex(SUPPORTED_TYPES, fill_value=0)
    with c3:
        fig = go.Figure(go.Bar(x=list(SUPPORTED_TYPES), y=type_counts.to_numpy(), text=[f"{v:,}" for v in type_counts.to_numpy()],
                               textposition="outside", cliponaxis=False, marker=dict(color=C["accent"]),
                               hovertemplate="%{x}: %{y:,}<extra></extra>"))
        style_figure(fig, 320, "Transactions by type")
        fig.update_yaxes(title=dict(text="Transactions"))
        show_figure(fig, "an_type_count")
    with c4:
        means = df.groupby("type")["fraud_probability"].mean().reindex(SUPPORTED_TYPES).dropna() * 100
        fig = go.Figure(go.Bar(x=means.index.tolist(), y=means.to_numpy(), text=[f"{v:.3f}%" for v in means.to_numpy()],
                               textposition="outside", cliponaxis=False, marker=dict(color=C["accent"]),
                               hovertemplate="%{x}: %{y:.4f}%<extra></extra>"))
        style_figure(fig, 320, "Average estimated probability by type")
        fig.update_yaxes(title=dict(text="Mean estimated probability (%)"))
        show_figure(fig, "an_type_mean")
    c5, c6 = st.columns(2)
    with c5:
        if len(df) < 2:
            st.info("The amount chart needs at least two transactions.")
        else:
            sample = scatter_sample(df)
            log_amount = st.toggle("Log scale for amount", value=True, key="an_log_amount")
            fig = go.Figure()
            for name, color in ((DECISION_FLAGGED, C["flag"]), (DECISION_CLEAR, C["clear"])):
                part = sample[sample["decision"] == name]
                fig.add_trace(go.Scatter(
                    x=part["amount"], y=part["fraud_probability_pct"], mode="markers", name=name,
                    marker=dict(color=color, size=6, opacity=0.65), customdata=part[ROW_ID],
                    hovertemplate="Row %{customdata}<br>Amount %{x:,.2f}<br>Probability %{y:.3f}%<extra></extra>",
                ))
            style_figure(fig, 340, "Amount and estimated probability", legend=True)
            fig.update_xaxes(title=dict(text="Amount"), type="log" if log_amount else "linear")
            fig.update_yaxes(title=dict(text="Estimated probability (%)"))
            show_figure(fig, "an_scatter")
            zero_note = " Zero amounts cannot be drawn on a log axis." if log_amount and (df["amount"] == 0).any() else ""
            st.caption(f"Showing {len(sample):,} of {len(df):,} transactions (all flagged ones first, up to half the cap).{zero_note}")
    with c6:
        band_counts = df["risk_band"].value_counts().reindex(BAND_NAMES, fill_value=0)
        fig = go.Figure(go.Bar(x=list(BAND_NAMES), y=band_counts.to_numpy(), text=[f"{v:,}" for v in band_counts.to_numpy()],
                               textposition="outside", cliponaxis=False, marker=dict(color=[C[b] for b in BAND_NAMES]),
                               hovertemplate="%{x}: %{y:,}<extra></extra>"))
        style_figure(fig, 340, "Risk-band breakdown")
        fig.update_yaxes(title=dict(text="Transactions"), type="log" if log_counts and band_counts.min() > 0 else "linear")
        show_figure(fig, "an_bands")
        st.caption("Bands: " + "; ".join(f"{b} {BAND_RANGES[b]}" for b in BAND_NAMES) + ". Interface categories only.")


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def export_card(title: str, description: str, df: pd.DataFrame, filename: str, key: str) -> None:
    left, right = st.columns([4, 1])
    with left:
        html_block(f'<div class="ps-panel"><h4>{esc(title)}</h4><p>{esc(description)}</p>'
                   f'<p><b style="color:var(--text)">{len(df):,}</b> rows, {len(df.columns)} columns</p></div>')
    with right:
        st.write("")
        st.download_button("Download CSV", to_csv_bytes(df), filename, "text/csv", key=key, disabled=df.empty)


def page_exports(load: ModelLoad) -> None:
    page_header("Export center", "Download results as CSV. Text cells that start with a formula character are prefixed with an apostrophe for spreadsheet safety.", show_badge=False)
    df = current_scored()
    if df is None:
        empty_state("Nothing to export", "Screen a batch to create downloads.", "Go to batch screening", "Batch screening", "ex_empty")
        return
    batch = st.session_state["batch"]
    flagged = with_review_columns(df[df["decision"] == DECISION_FLAGGED].sort_values("fraud_probability", ascending=False, kind="stable"))
    awaiting = flagged[flagged["review_status"].isin(OPEN_STATUSES)]
    export_card("Full scored results", "Every scored transaction with probability, risk band and decision.", df, "scored_transactions.csv", "dl_full")
    export_card("Flagged transactions", "Transactions at or above the current threshold, with review status and note.", flagged, "flagged_transactions.csv", "dl_flagged")
    export_card("Awaiting review", "Flagged transactions that are pending or under review.", awaiting, "awaiting_review.csv", "dl_awaiting")
    export_card("Validation errors", "Rows that were rejected before scoring, with the field and the reason.", batch["errors"], "validation_errors.csv", "dl_errors")
    reviewed = with_review_columns(df)
    reviewed = reviewed[reviewed[ROW_ID].isin(list(st.session_state["review"].keys()))]
    export_card("Review log", "Transactions that have a saved status or note in this session. Download before the session ends.", reviewed, "review_log.csv", "dl_review_log")


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------


def page_about(load: ModelLoad) -> None:
    page_header("About PaymentShield AI", "What it does, how the saved model is used, and what it cannot tell you.", show_badge=False)
    steps = ["CSV or manual input", "Validation", "Feature preparation", "Saved model", "Estimated probability", "Threshold-based review", "Investigation and export"]
    flow = '<span class="ps-sep">&rarr;</span>'.join(f'<span class="ps-step">{esc(s)}</span>' for s in steps)
    html_block(f'<div class="ps-pipe">{flow}</div>')
    feature_list = ", ".join(f"`{name}`" for name in FEATURE_ORDER)
    type_list = " and ".join(f"`{name}`" for name in SUPPORTED_TYPES)
    st.markdown(f"""
**What it does.** PaymentShield AI scores transactions with a fraud model you already trained, lets a person adjust a review threshold,
and supports a simple investigation workflow: queue, notes, statuses and exports.

**How the saved model is used.** The app loads `models/fraud_detection_model.pkl` as it is. It never retrains, replaces or adjusts the model.
Each transaction becomes seven features in the training order ({feature_list}). The fraud probability is read from the
column that matches the fraud class label in `model.classes_`, not from an assumed position.

**Supported transaction types.** {type_list}. The model was trained on these two only, so other types are rejected, not guessed.

**About the data.** PaySim is a simulated mobile-money dataset. Patterns in it are not evidence about real customers, banks or fraud rates.

**Why people stay in the loop.** A probability is an estimate. A flagged transaction is not proven fraudulent, and an unflagged one is not proven safe.
The review threshold is a demo setting, not a validated policy, and it changes decisions without changing any probability.

**What the model cannot guarantee.** This app reports no accuracy, precision or recall, because that requires labelled test data and a separate evaluation.
Score charts show how the model scored the uploaded file, nothing more. There is no banking integration and no real-time payment interception.

**Where review work is kept.** Statuses and notes live in Streamlit session state only. They are not saved to a database and disappear when the session ends.
""")
    html_block('<div class="ps-panel"><h4>Risk bands</h4><p>Fixed probability ranges used to organise the interface. They are not validated estimates of real-world fraud likelihood.</p>'
              + "".join(f'<div class="kv"><span>{badge(b, b)}</span><b>{esc(BAND_RANGES[b])}</b></div>' for b in BAND_NAMES) + '</div>')
    render_diagnostics(load, (st.session_state.get("batch") or {}).get("diag"), "diag_about")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
PAGE_FUNCTIONS: dict[str, Callable[[ModelLoad], None]] = {
    "Overview": page_overview, "Batch screening": page_batch, "Investigation": page_investigation,
    "Single transaction": page_single, "Review queue": page_queue, "Analytics": page_analytics,
    "Exports": page_exports, "About": page_about,
}


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="\U0001F6E1\uFE0F", layout="wide", initial_sidebar_state="expanded")
    init_state()
    inject_css()
    load = get_model()
    render_sidebar(load)
    render_model_banner(load)
    PAGE_FUNCTIONS[st.session_state["nav"]](load)
    render_footer()


if __name__ == "__main__":
    main()
