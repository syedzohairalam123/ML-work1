"""
Repair notebook dependency chain:

1. SHAP cell (Section 8a) currently writes these local names:
   - shap_vals_class1   (sampled test SHAP, n_samples x n_features)
   - primary_drivers    (per-sample string driver from sampled SHAP)
   Later Section 10 then uses shap_vals_class1 for the FULL df, which:
     - is the wrong row count if X_shap was sampled, and
     - fails to attach primary_driver before reason/action/playbook creation.

Fix:
- Recompute SHAP on the full X_test once.
- Normalize SHAP robustly for the installed SHAP/sklearn output shape.
- Attach primary_driver (+ direction + value) to X_test by index.
- Rebuild Section 10 playbook from X_test-based attribution where that is
  the intended methodology, and keep the downstream global playbook creation
  consistent with available columns.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

NOTEBOOK_PATH = Path("work/notebooks/capstone.ipynb")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def find_code_cell(nb: dict, marker: str) -> dict | None:
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        if marker in "".join(cell["source"]):
            return cell
    return None


def source_as_text(cell: dict) -> str:
    return "".join(cell["source"])


def set_source(cell: dict, text: str) -> None:
    # preserve notebook source format: list of lines, last line without trailing newline
    lines = text.splitlines()
    if not lines:
        cell["source"] = []
        return
    cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]


def make_shap_cell() -> str:
    return r'''# ═══════════════════════════════════════════════════════════════════════════════
# Section 8a: SHAP Global + Local Explainability
# ═══════════════════════════════════════════════════════════════════════════════

# Use the best non-stacking model for SHAP (TreeExplainer requires tree model)
shap_model_name = best_model_name
if shap_model_name == 'stacking_ensemble':
    shap_model_name = 'gradient_boosting'
shap_model = trained_models[shap_model_name]

# ------------------------------------------------------------------
# Robust SHAP normalization for binary classifiers under the installed
# SHAP/sklearn versions.
#
# TreeExplainer.shap_values for a binary sklearn tree model can return:
#   - list[class0, class1]            (older style)
#   - ndarray (n_samples, n_features) (some configs)
#   - ndarray (n_samples, n_features, n_classes)
# We normalize to a single 2D array: (n_samples, n_features) for the
# positive/decay class.
# ------------------------------------------------------------------

_raw = shap_model.shap_values(X_test)
_raw_arr = np.asarray(_raw)

if hasattr(shap_model, "classes_"):
    classes = list(shap_model.classes_)
else:
    classes = [0, 1]

# Decide which axis, if any, corresponds to class output.
# Heuristic:
#   - If _raw is a list, treat as [class0, class1]
#   - If 3D array, assume last axis is class
#   - Otherwise assume already (n_samples, n_features)
if isinstance(_raw, list):
    if len(_raw) == 2:
        # Pick the positive class array by matching class label if possible
        if 1 in classes:
            pos_idx = classes.index(1)
        else:
            pos_idx = 1
        shap_matrix = np.asarray(_raw[pos_idx])
    else:
        shap_matrix = np.asarray(_raw[0])
elif _raw_arr.ndim == 3:
    # (n_samples, n_features, n_classes)
    if 1 in classes:
        class_axis_idx = classes.index(1)
    else:
        class_axis_idx = 1
    shap_matrix = np.asarray(_raw)[:, :, class_axis_idx]
else:
    shap_matrix = _raw_arr

# Validation
shap_matrix = np.asarray(shap_matrix, dtype=float)

if shap_matrix.ndim != 2:
    raise ValueError(
        f"SHAP normalization produced an unexpected number of dimensions: "
        f"{shap_matrix.ndim}. Expected 2D (n_samples, n_features)."
    )

n_samples, n_features = shap_matrix.shape
if n_samples != len(X_test):
    raise ValueError(
        f"SHAP sample count mismatch: SHAP has {n_samples} rows but X_test has "
        f"{len(X_test)} rows. SHAP must be computed on X_test for test-set explanations."
    )
if n_features != len(FEATURES):
    raise ValueError(
        f"SHAP feature count mismatch: SHAP has {n_features} features but FEATURES has "
        f"{len(FEATURES)} features."
    )

# Diagnostics (can be reduced later, kept here for debug clarity)
print("\n🔍 SHAP diagnostics")
print("  SHAP raw type:", type(_raw))
print("  SHAP normalized shape:", shap_matrix.shape)
print("  X_test shape:", X_test.shape)
print("  FEATURES count:", len(FEATURES))

# ------------------------------------------------------------------
# Global feature importance
# ------------------------------------------------------------------
feature_names = np.asarray(FEATURES, dtype=object)
mean_abs_shap = np.abs(shap_matrix).mean(axis=0)

shap_importance = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap,
}).sort_values('mean_abs_shap', ascending=False)

print("\n📊 SHAP Global Feature Importance (Top 10):")
print(shap_importance.head(10).to_string(index=False))

# ------------------------------------------------------------------
# Primary driver extraction
#
# For each TEST sample, compute:
#   - feature with max absolute SHAP contribution
#   - that feature's signed SHAP value
#   - direction of push: toward_decay vs toward_stability
#
# Output arrays are 1D and index-aligned with X_test.
# ------------------------------------------------------------------

top_feature_indices = np.argmax(np.abs(shap_matrix), axis=1)
primary_drivers = np.asarray(feature_names, dtype=object)[top_feature_indices]
primary_driver_values = shap_matrix[np.arange(n_samples), top_feature_indices]

# Direction: positive SHAP for a feature pushes model output toward class 1 (decay)
# if class 1 is the positive/decay class in this model.
if 1 in classes:
    positive_class_is_decay = True
else:
    positive_class_is_decay = False

if positive_class_is_decay:
    primary_driver_directions = np.where(
        primary_driver_values >= 0, "toward_decay", "toward_stability"
    )
else:
    # If class 1 is not the decay class, flip interpretation
    primary_driver_directions = np.where(
        primary_driver_values >= 0, "toward_stability", "toward_decay"
    )

# Attach to X_test by index. Do NOT assign test SHAP rows to full df.
test_results = X_test.copy()
test_results["primary_driver"] = primary_drivers
test_results["primary_driver_shap_value"] = primary_driver_values
test_results["primary_driver_direction"] = primary_driver_directions

print("\n🔍 Primary driver diagnostics")
print("  primary_drivers shape:", primary_drivers.shape)
print("  primary_drivers sample (first 5):")
print(pd.Series(primary_drivers).head(5).to_string(index=False))
print("  primary_driver_direction sample (first 5):")
print(pd.Series(primary_driver_directions).head(5).to_string(index=False))

# Visualization: SHAP Summary
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Bar plot
ax = axes[0]
top_10 = shap_importance.head(10)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_10)))
ax.barh(range(len(top_10)), top_10['mean_abs_shap'].values, color=colors, alpha=0.85)
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10['feature'].values)
ax.invert_yaxis()
ax.set_xlabel('Mean |SHAP Value|')
ax.set_title('SHAP Feature Importance (Global)', fontweight='bold')

# Primary driver distribution
ax = axes[1]
driver_counts = pd.Series(primary_drivers).value_counts().head(8)
colors2 = plt.cm.Set2(np.linspace(0, 1, len(driver_counts)))
ax.pie(
    driver_counts.values,
    labels=driver_counts.index,
    autopct='%1.1f%%',
    colors=colors2,
    startangle=90,
    textprops={'fontsize': 9},
)
ax.set_title('Primary Driver Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('work/figures/shap_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
'''


def make_playbook_cell() -> str:
    return r'''# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: SHAP-Driven Dynamic Action Playbook with Uncertainty
# ═══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------
# Pre-flight validation
# ------------------------------------------------------------------
_required_playbook_prereqs = [
    "primary_driver",
    "primary_driver_shap_value",
    "primary_driver_direction",
    "priority_score",
    "historical_ctr",
]
_missing_prereqs = [c for c in _required_playbook_prereqs if c not in X_test.columns]
if _missing_prereqs:
    raise ValueError(
        "Playbook prerequisites missing from test-results frame. "
        f"Missing columns: {_missing_prereqs}."
    )

# ------------------------------------------------------------------
# Global model outputs (used for full-dataset ranking metrics)
# ------------------------------------------------------------------
full_probs = best_model.predict_proba(X)[:, 1]
df["decay_probability"] = full_probs

# Conformal prediction sets for full dataset
cal_probs_for_full = best_model.predict_proba(X_cal)[:, 1]
scores_for_full = np.where(y_cal == 1, 1 - cal_probs_for_full, cal_probs_for_full)
q_full = np.quantile(
    scores_for_full,
    np.minimum(np.ceil(0.9 * (len(scores_for_full) + 1)) / len(scores_for_full), 1.0),
)

def get_conformal_set_size(prob, q_threshold):
    """Compute prediction set size for a single probability."""
    in_class1 = prob >= (1 - q_threshold)
    in_class0 = (1 - prob) >= (1 - q_threshold)
    return int(in_class1) + int(in_class0)

df["conformal_set_size"] = df["decay_probability"].apply(
    lambda p: get_conformal_set_size(p, q_full)
)

# ------------------------------------------------------------------
# Priority score (full dataset)
# ------------------------------------------------------------------
df["confidence_factor"] = df.apply(
    lambda row: 1.0 if row["conformal_set_size"] == 1
    else (0.75 if row["decay_probability"] > 0.7 else 0.5),
    axis=1,
)

df["priority_score"] = (
    df["decay_probability"]
    * np.log10(df["total_impressions"] + 1)
    * df["confidence_factor"]
).round(4)

# ------------------------------------------------------------------
# Build an explicit test-set editorial playbook from SHAP attribution
# ------------------------------------------------------------------
_test_playbook = test_results.copy()

# Bring in identifying/context columns from df by matching index,
# so the playbook keeps content identifiers without fabricating them.
for _col in ["content_hash_id", "client_hash_id", "historical_ctr"]:
    if _col in df.columns and _col not in _test_playbook.columns:
        _test_playbook[_col] = df.loc[_test_playbook.index, _col]

def assign_dynamic_action(row):
    """SHAP-driven action assignment with confidence bands.

    This function is defensive: it validates required inputs before use.
    """
    required = ["primary_driver", "priority_score", "historical_ctr"]
    missing = [k for k in required if k not in row.index]
    if missing:
        raise ValueError(
            "assign_dynamic_action prerequisites missing. "
            f"Missing fields: {missing}."
        )

    if row["priority_score"] < 1.0:
        return ("MONITOR", "No immediate intervention — trend monitoring recommended")

    prefix = "CRITICAL: " if row["priority_score"] >= 2.0 else "REVIEW: "

    pd_feature = str(row["primary_driver"])

    if pd_feature in ["smoothed_ctr", "engagement_proxy", "ctr_percentile"]:
        return (
            prefix + "Optimize Meta Titles & Snippets",
            "Click deficit is primary driver — title tag, meta description, and snippet optimization",
        )
    elif pd_feature in [
        "avg_position",
        "position_volatility",
        "position_stability",
        "position_spread",
    ]:
        return (
            prefix + "Deep Content Refresh & Structural Audit",
            "Position instability is primary driver — content refresh, structural review, factual updates",
        )
    elif pd_feature in ["total_impressions", "log_impressions", "impression_density"]:
        return (
            prefix + "Traffic Volume Review",
            "Impression decline pattern — review content relevance and search intent alignment",
        )
    else:
        return (
            prefix + "General Editorial Audit",
            "Multi-factor review — comprehensive content assessment recommended",
        )

_test_playbook["recommended_action"] = _test_playbook.apply(
    lambda row: assign_dynamic_action(row)[0], axis=1
)
_test_playbook["action_detail"] = _test_playbook.apply(
    lambda row: assign_dynamic_action(row)[1], axis=1
)

# Rank by priority
playbook = (
    _test_playbook.sort_values("priority_score", ascending=False)
    .reset_index(drop=True)
)
playbook["rank"] = playbook.index + 1

# ------------------------------------------------------------------
# Validation before export
# ------------------------------------------------------------------
assert "playbook" in globals(), "playbook was not created"
assert isinstance(playbook, pd.DataFrame), "playbook is not a DataFrame"
assert not playbook.empty, "playbook is empty"

_required_playbook_cols = [
    "content_hash_id",
    "priority_score",
    "reason_code",
    "primary_driver",
    "historical_ctr",
    "recommended_action",
]
_missing_cols = [c for c in _required_playbook_cols if c not in playbook.columns]

if _missing_cols:
    raise ValueError(
        "Playbook missing required columns: {_missing_cols}. "
        f"Current columns: {playbook.columns.tolist()}"
    )

# Add a reason_code column derived from primary driver so downstream export has it
def reason_code_from_driver(row):
    pd_feature = str(row.get("primary_driver", ""))
    direction = str(row.get("primary_driver_direction", ""))
    if pd_feature in ["smoothed_ctr", "engagement_proxy", "ctr_percentile"]:
        base = "Click-efficiency deficit"
    elif pd_feature in [
        "avg_position",
        "position_volatility",
        "position_stability",
        "position_spread",
    ]:
        base = "Position instability"
    elif pd_feature in ["total_impressions", "log_impressions", "impression_density"]:
        base = "Impression decline"
    else:
        base = "Multi-factor review"
    if direction == "toward_decay":
        return f"{base} (SHAP pushes toward decay)"
    return f"{base} (SHAP pushes toward stability)"

playbook["reason_code"] = playbook.apply(reason_code_from_driver, axis=1)

# Display top 15
print("\n" + "=" * 100)
print("  TOP 15 EDITORIAL PRIORITIES (SHAP-DRIVEN + CONFORMAL UNCERTAINTY)")
print("=" * 100)
display_cols = [
    "rank",
    "content_hash_id",
    "priority_score",
    "decay_probability",
    "conformal_set_size",
    "primary_driver",
    "primary_driver_direction",
    "recommended_action",
    "reason_code",
    "historical_ctr",
    "total_impressions",
    "smoothed_ctr",
]
print(playbook[display_cols].head(15).to_string(index=False))
print("=" * 100)

print("\n📊 PLAYBOOK SUMMARY")
print(f"  Total pages evaluated: {len(playbook):,}")
print(f"  Pages with decay probability > 0.5: {(playbook['decay_probability'] > 0.5).sum():,}")
print(f"\n  Action distribution:")
for action, count in playbook["recommended_action"].value_counts().items():
    pct = count / len(playbook) * 100
    print(f"    {action}: {count:,} ({pct:.1f}%)")
print(f"\n  SHAP primary driver distribution:")
for driver, count in playbook["primary_driver"].value_counts().head(6).items():
    print(f"    {driver}: {count:,}")
print(f"\n  Primary driver direction distribution:")
for direction, count in playbook["primary_driver_direction"].value_counts().items():
    print(f"    {direction}: {count:,}")
print(f"\n  Confidence distribution:")
for level, count in playbook["conformal_set_size"].value_counts().sort_index().items():
    label = "High (set=1)" if level == 1 else "Medium/Low (set=2)"
    print(f"    {label}: {count:,}")print("\n🔍 Final playbook shape:", playbook.shape)
print("   Playbook columns:", playbook.columns.tolist())
'''


def make_playbook_cell():
    return make_playbook_cell_inner()


def make_playbook_cell_inner() -> str:
    return r'''# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: SHAP-Driven Dynamic Action Playbook with Uncertainty
# ═══════════════════════════════════════════════════════════════════════════════

# ------------------------------------------------------------------
# Pre-flight validation
# ------------------------------------------------------------------
_required_playbook_prereqs = [
    "primary_driver",
    "primary_driver_shap_value",
    "primary_driver_direction",
    "priority_score",
    "historical_ctr",
]
_missing_prereqs = [c for c in _required_playbook_prereqs if c not in X_test.columns]
if _missing_prereqs:
    raise ValueError(
        "Playbook prerequisites missing from test-results frame. "
        f"Missing columns: {_missing_prereqs}."
    )

# ------------------------------------------------------------------
# Global model outputs (used for full-dataset ranking metrics)
# ------------------------------------------------------------------
full_probs = best_model.predict_proba(X)[:, 1]
df["decay_probability"] = full_probs

# Conformal prediction sets for full dataset
cal_probs_for_full = best_model.predict_proba(X_cal)[:, 1]
scores_for_full = np.where(y_cal == 1, 1 - cal_probs_for_full, cal_probs_for_full)
q_full = np.quantile(
    scores_for_full,
    np.minimum(np.ceil(0.9 * (len(scores_for_full) + 1)) / len(scores_for_full), 1.0),
)

def get_conformal_set_size(prob, q_threshold):
    """Compute prediction set size for a single probability."""
    in_class1 = prob >= (1 - q_threshold)
    in_class0 = (1 - prob) >= (1 - q_threshold)
    return int(in_class1) + int(in_class0)

df["conformal_set_size"] = df["decay_probability"].apply(
    lambda p: get_conformal_set_size(p, q_full)
)

# ------------------------------------------------------------------
# Priority score (full dataset)
# ------------------------------------------------------------------
df["confidence_factor"] = df.apply(
    lambda row: 1.0 if row["conformal_set_size"] == 1
    else (0.75 if row["decay_probability"] > 0.7 else 0.5),
    axis=1,
)

df["priority_score"] = (
    df["decay_probability"]
    * np.log10(df["total_impressions"] + 1)
    * df["confidence_factor"]
).round(4)

# ------------------------------------------------------------------
# Build an explicit test-set editorial playbook from SHAP attribution
# ------------------------------------------------------------------
_test_playbook = test_results.copy()

# Bring in identifying/context columns from df by matching index,
# so the playbook keeps content identifiers without fabricating them.
for _col in ["content_hash_id", "client_hash_id", "historical_ctr"]:
    if _col in df.columns and _col not in _test_playbook.columns:
        _test_playbook[_col] = df.loc[_test_playbook.index, _col]

def assign_dynamic_action(row):
    """SHAP-driven action assignment with confidence bands.

    This function is defensive: it validates required inputs before use.
    """
    required = ["primary_driver", "priority_score", "historical_ctr"]
    missing = [k for k in required if k not in row.index]
    if missing:
        raise ValueError(
            "assign_dynamic_action prerequisites missing. "
            f"Missing fields: {missing}."
        )

    if row["priority_score"] < 1.0:
        return ("MONITOR", "No immediate intervention — trend monitoring recommended")

    prefix = "CRITICAL: " if row["priority_score"] >= 2.0 else "REVIEW: "

    pd_feature = str(row["primary_driver"])

    if pd_feature in ["smoothed_ctr", "engagement_proxy", "ctr_percentile"]:
        return (
            prefix + "Optimize Meta Titles & Snippets",
            "Click deficit is primary driver — title tag, meta description, and snippet optimization",
        )
    elif pd_feature in [
        "avg_position",
        "position_volatility",
        "position_stability",
        "position_spread",
    ]:
        return (
            prefix + "Deep Content Refresh & Structural Audit",
            "Position instability is primary driver — content refresh, structural review, factual updates",
        )
    elif pd_feature in ["total_impressions", "log_impressions", "impression_density"]:
        return (
            prefix + "Traffic Volume Review",
            "Impression decline pattern — review content relevance and search intent alignment",
        )
    else:
        return (
            prefix + "General Editorial Audit",
            "Multi-factor review — comprehensive content assessment recommended",
        )

_test_playbook["recommended_action"] = _test_playbook.apply(
    lambda row: assign_dynamic_action(row)[0], axis=1
)
_test_playbook["action_detail"] = _test_playbook.apply(
    lambda row: assign_dynamic_action(row)[1], axis=1
)

# Rank by priority
playbook = (
    _test_playbook.sort_values("priority_score", ascending=False)
    .reset_index(drop=True)
)
playbook["rank"] = playbook.index + 1

# ------------------------------------------------------------------
# Validation before export
# ------------------------------------------------------------------
assert "playbook" in globals(), "playbook was not created"
assert isinstance(playbook, pd.DataFrame), "playbook is not a DataFrame"
assert not playbook.empty, "playbook is empty"

_required_playbook_cols = [
    "content_hash_id",
    "priority_score",
    "reason_code",
    "primary_driver",
    "historical_ctr",
    "recommended_action",
]
_missing_cols = [c for c in _required_playbook_cols if c not in playbook.columns]

if _missing_cols:
    raise ValueError(
        "Playbook missing required columns: {_missing_cols}. "
        f"Current columns: {playbook.columns.tolist()}"
    )

# Add a reason_code column derived from primary driver so downstream export has it
def reason_code_from_driver(row):
    pd_feature = str(row.get("primary_driver", ""))
    direction = str(row.get("primary_driver_direction", ""))
    if pd_feature in ["smoothed_ctr", "engagement_proxy", "ctr_percentile"]:
        base = "Click-efficiency deficit"
    elif pd_feature in [
        "avg_position",
        "position_volatility",
        "position_stability",
        "position_spread",
    ]:
        base = "Position instability"
    elif pd_feature in ["total_impressions", "log_impressions", "impression_density"]:
        base = "Impression decline"
    else:
        base = "Multi-factor review"
    if direction == "toward_decay":
        return f"{base} (SHAP pushes toward decay)"
    return f"{base} (SHAP pushes toward stability)"

playbook["reason_code"] = playbook.apply(reason_code_from_driver, axis=1)

# Display top 15
print("\n" + "=" * 100)
print("  TOP 15 EDITORIAL PRIORITIES (SHAP-DRIVEN + CONFORMAL UNCERTAINTY)")
print("=" * 100)
display_cols = [
    "rank",
    "content_hash_id",
    "priority_score",
    "decay_probability",
    "conformal_set_size",
    "primary_driver",
    "primary_driver_direction",
    "recommended_action",
    "reason_code",
    "historical_ctr",
    "total_impressions",
    "smoothed_ctr",
]
print(playbook[display_cols].head(15).to_string(index=False))
print("=" * 100)

print("\n📊 PLAYBOOK SUMMARY")
print(f"  Total pages evaluated: {len(playbook):,}")
print(f"  Pages with decay probability > 0.5: {(playbook['decay_probability'] > 0.5).sum():,}")
print(f"\n  Action distribution:")
for action, count in playbook["recommended_action"].value_counts().items():
    pct = count / len(playbook) * 100
    print(f"    {action}: {count:,} ({pct:.1f}%)")
print(f"\n  SHAP primary driver distribution:")
for driver, count in playbook["primary_driver"].value_counts().head(6).items():
    print(f"    {driver}: {count:,}")
print(f"\n  Primary driver direction distribution:")
for direction, count in playbook["primary_driver_direction"].value_counts().items():
    print(f"    {direction}: {count:,}")
print(f"\n  Confidence distribution:")
for level, count in playbook["conformal_set_size"].value_counts().sort_index().items():
    label = "High (set=1)" if level == 1 else "Medium/Low (set=2)"
    print(f"    {label}: {count:,}")

print("\n🔍 Final playbook shape:", playbook.shape)
print("   Playbook columns:", playbook.columns.tolist())
'''


def patch_notebook(nb: dict) -> None:
    shap_cell = find_code_cell(nb, "Section 8a: SHAP Global")
    if shap_cell is None:
        raise RuntimeError("Could not find SHAP cell (Section 8a)")
    set_source(shap_cell, make_shap_cell())

    playbook_cell = find_code_cell(nb, "Section 10: SHAP-Driven Dynamic Action Playbook")
    if playbook_cell is None:
        raise RuntimeError("Could not find playbook cell (Section 10)")
    set_source(playbook_cell, make_playbook_cell())


def main() -> None:
    nb = load(NOTEBOOK_PATH)
    patch_notebook(nb)
    save(NOTEBOOK_PATH, nb)
    print("Patched capstone.ipynb:")
    print("- Replaced Section 8a SHAP cell with robust normalization + test-set primary driver")
    print("- Replaced Section 10 playbook cell with prerequisite validation + reason_code + explicit playbook build")


if __name__ == "__main__":
    main()
