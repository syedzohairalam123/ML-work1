"""Fix categorical pos_bucket causing TypeError in target formulation cell."""
import json
from pathlib import Path

nb_path = Path("work/notebooks/capstone.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# Find the target formulation cell
target_cell_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "Section 5: Target Formulation" in src and "ranked_df" in src:
            target_cell_idx = i
            break

if target_cell_idx is None:
    raise RuntimeError("Could not find target formulation cell")

new_source = r'''# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Target Formulation, Leakage Audit, and Validation Design
# ═══════════════════════════════════════════════════════════════════════════════

# --- 5a. Tautology-Free Target Definition ---
# Instead of a hard-coded rule on features, we compute a COMPOSITE score that
# measures how much a page underperforms its rank class peers.

# Step 1: Define 'ranked' pages (top-20 position = competitive striking distance)
ranked_mask = df['avg_position'] <= 20
ranked_df = df.loc[ranked_mask].copy()

# Drop any non-numeric columns that may have leaked in from EDA
for col in ['pos_bucket']:
    ranked_df.drop(columns=[col], inplace=True, errors='ignore')

# Step 2: Compute position-bucket-specific expected CTR (median)
ranked_df['pos_bucket'] = pd.cut(
    ranked_df['avg_position'].astype(float),
    bins=[0, 5, 10, 15, 20],
    labels=['1-5', '6-10', '11-15', '16-20']
)
expected_ctr_by_bucket = ranked_df.groupby('pos_bucket', observed=True)['smoothed_ctr'].median()
ranked_df['expected_ctr'] = ranked_df['pos_bucket'].map(expected_ctr_by_bucket).astype(float)

# Drop the categorical column BEFORE any numeric operations
ranked_df.drop(columns=['pos_bucket'], inplace=True, errors='ignore')

# Step 3: Compute click deficit (how much CTR falls below rank-class expectation)
ranked_df['click_deficit'] = np.maximum(0.0, ranked_df['expected_ctr'].astype(float) - ranked_df['smoothed_ctr'].astype(float))

# Step 4: Scale by impression volume (high-traffic pages with deficits matter more)
ranked_df['structural_score'] = (ranked_df['click_deficit'].astype(float) * np.log1p(ranked_df['total_impressions'].astype(float)))

# Step 5: Binary target = above-median structural score (top half = decay candidate)
median_score = float(ranked_df['structural_score'].median())
ranked_df['target_decay'] = (ranked_df['structural_score'] > median_score).astype(int)

# Merge target back to full dataframe (non-ranked pages = 0, not decay candidates)
merge_cols = ['content_hash_id', 'target_decay', 'structural_score', 'expected_ctr', 'click_deficit']
df = df.merge(
    ranked_df[merge_cols],
    on='content_hash_id', how='left'
)
df['target_decay'] = df['target_decay'].fillna(0).astype(int)
df['structural_score'] = df['structural_score'].fillna(0).astype(float)
df['expected_ctr'] = df['expected_ctr'].fillna(0).astype(float)
df['click_deficit'] = df['click_deficit'].fillna(0).astype(float)

# Clean up any leftover categorical columns
df.drop(columns=['pos_bucket'], inplace=True, errors='ignore')

# --- Define model features (NO target-derived features) ---
FEATURES = [
    'total_impressions', 'total_clicks', 'avg_position', 'active_days',
    'smoothed_ctr', 'position_volatility', 'position_stability',
    'impression_density', 'daily_clicks', 'engagement_proxy',
    'position_spread', 'impression_rank', 'ctr_percentile',
    'log_impressions', 'log_clicks', 'max_position', 'min_position'
]

X = df[FEATURES].copy()
y = df['target_decay'].copy()
groups = df['client_hash_id'].copy()

# Fill any remaining NaN and ensure numeric
X = X.fillna(0).astype(float)

print(f"Target distribution: {y.value_counts().to_dict()}")
print(f"Decay rate: {y.mean():.2%}")
print(f"Features used: {len(FEATURES)}")
print(f"Feature list: {FEATURES}")
'''

lines = new_source.strip().split("\n")
source_lines = [line + "\n" for line in lines[:-1]] + [lines[-1]]
nb["cells"][target_cell_idx]["source"] = source_lines

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Fixed target formulation cell {target_cell_idx}")
