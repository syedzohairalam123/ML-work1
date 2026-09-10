"""Patch capstone.ipynb to add synthetic data fallback when HF_TOKEN is unavailable."""
import json
from pathlib import Path

nb_path = Path("work/notebooks/capstone.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# Find the data extraction cell (Section 4)
data_cell_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "Section 4: Data Extraction" in src and "feature_query" in src:
            data_cell_idx = i
            break

if data_cell_idx is None:
    raise RuntimeError("Could not find Section 4 data extraction cell")

# Build source as list of strings (each ending with \n except last)
raw = r'''# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Data Extraction & Advanced Feature Engineering Engine
# ═══════════════════════════════════════════════════════════════════════════════
#
# When HF_TOKEN is set (Colab or env), loads real FlyRank data.
# When HF_TOKEN is missing, generates realistic synthetic data for local testing.

# --- Authentication ---
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    HF_TOKEN = os.environ.get('HF_TOKEN')

if HF_TOKEN:
    print("HF_TOKEN found - connecting to FlyRank warehouse...")
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{HF_TOKEN}')")
    REL = 'hf://datasets/FlyRank/internship-warehouse'

    feature_query = f"""
        SELECT
            client_hash_id,
            content_hash_id,
            SUM(gsc_impressions) AS total_impressions,
            SUM(gsc_clicks) AS total_clicks,
            AVG(gsc_avg_position) AS avg_position,
            COUNT(DISTINCT report_date) AS active_days,
            ROUND((SUM(gsc_clicks) + 1)::FLOAT / (SUM(gsc_impressions) + 10), 6) AS smoothed_ctr,
            ROUND(SUM(gsc_clicks)::FLOAT / NULLIF(SUM(gsc_impressions), 0), 6) AS raw_ctr,
            MAX(gsc_avg_position) AS max_position,
            MIN(gsc_avg_position) AS min_position,
            STDDEV(gsc_avg_position) AS position_volatility,
            SUM(gsc_impressions) / NULLIF(COUNT(DISTINCT report_date), 0) AS impression_density,
            SUM(gsc_clicks) / NULLIF(COUNT(DISTINCT report_date), 0) AS daily_clicks,
            ROUND(
                (SUM(gsc_clicks)::FLOAT / NULLIF(SUM(gsc_impressions), 0)) /
                NULLIF(1.0 / NULLIF(AVG(gsc_avg_position), 0), 0),
                6
            ) AS engagement_proxy,
            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY gsc_avg_position) -
            PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY gsc_avg_position) AS position_spread
        FROM read_parquet('{REL}/fact_content_daily_performance/month=2026-03/*.parquet')
        GROUP BY 1, 2
        HAVING SUM(gsc_impressions) >= 50 AND COUNT(DISTINCT report_date) >= 5
    """

    df = con.sql(feature_query).df()
    print(f"Live data loaded: {len(df):,} entities from FlyRank warehouse")

else:
    print("No HF_TOKEN found - generating synthetic dataset for local testing...")
    print("(Set HF_TOKEN env var or Colab secret to use real FlyRank data)")

    np.random.seed(42)
    N_CLIENTS = 45
    N_CONTENT_PER_CLIENT = np.random.randint(800, 4000, N_CLIENTS)
    N_TOTAL = int(N_CONTENT_PER_CLIENT.sum())

    client_ids = [f"client_{hashlib.md5(str(i).encode()).hexdigest()[:16]}" for i in range(N_CLIENTS)]
    content_ids = [f"content_{hashlib.md5(str(i).encode()).hexdigest()[:16]}" for i in range(N_TOTAL)]

    client_col = np.repeat(client_ids, N_CONTENT_PER_CLIENT)
    content_col = np.array(content_ids)

    avg_position = np.random.lognormal(mean=2.0, sigma=0.8, size=N_TOTAL).clip(1, 120)
    total_impressions = np.random.lognormal(mean=6.5, sigma=1.5, size=N_TOTAL).clip(10, 500000).astype(int)
    active_days = np.random.choice(range(5, 32), size=N_TOTAL)

    base_ctr = 0.15 / (avg_position ** 0.5)
    noise = np.random.lognormal(mean=-3, sigma=1.0, size=N_TOTAL)
    smoothed_ctr = (base_ctr + noise * 0.005).clip(0.0001, 0.4)
    total_clicks = (smoothed_ctr * total_impressions).astype(int).clip(0, None)
    smoothed_ctr = ((total_clicks + 1) / (total_impressions + 10))

    position_volatility = np.random.exponential(2.0, size=N_TOTAL).clip(0, 30)
    max_position = (avg_position + position_volatility * np.random.uniform(0.3, 1.5, N_TOTAL)).clip(1, 150)
    min_position = (avg_position - position_volatility * np.random.uniform(0.3, 1.0, N_TOTAL)).clip(0.5, avg_position)
    impression_density = total_impressions / active_days
    daily_clicks = total_clicks / active_days
    engagement_proxy = smoothed_ctr * avg_position
    position_spread = np.random.exponential(3.0, size=N_TOTAL).clip(0, 50)

    df = pd.DataFrame({
        'client_hash_id': client_col,
        'content_hash_id': content_col,
        'total_impressions': total_impressions,
        'total_clicks': total_clicks,
        'avg_position': np.round(avg_position, 4),
        'active_days': active_days,
        'smoothed_ctr': np.round(smoothed_ctr, 6),
        'raw_ctr': np.round(smoothed_ctr * np.random.uniform(0.9, 1.1, N_TOTAL), 6),
        'max_position': np.round(max_position, 4),
        'min_position': np.round(min_position, 4),
        'position_volatility': np.round(position_volatility, 4),
        'impression_density': np.round(impression_density, 2),
        'daily_clicks': np.round(daily_clicks, 4),
        'engagement_proxy': np.round(engagement_proxy, 6),
        'position_spread': np.round(position_spread, 4),
    })
    print(f"Synthetic data generated: {len(df):,} entities across {N_CLIENTS} clients")

# --- Feature Post-Processing ---
df['position_volatility'] = df['position_volatility'].fillna(0.0)
df['position_spread'] = df['position_spread'].fillna(0.0)
df['engagement_proxy'] = df['engagement_proxy'].fillna(0.0)
df['smoothed_ctr'] = df['smoothed_ctr'].fillna(0.001)

df['position_stability'] = 1.0 / (1.0 + df['position_volatility'].fillna(0))
df['impression_rank'] = df['total_impressions'].rank(pct=True)
df['ctr_percentile'] = df['smoothed_ctr'].rank(pct=True)

df['log_impressions'] = np.log1p(df['total_impressions'])
df['log_clicks'] = np.log1p(df['total_clicks'])

print(f"\nData ready: {len(df):,} content entities")
print(f"Unique clients: {df['client_hash_id'].nunique():,}")
print(f"Feature matrix shape: {df.shape}")
df.head(3)
'''

# Convert to notebook source format: list of strings, each ending with \n except last
lines = raw.strip().split("\n")
source_lines = [line + "\n" for line in lines[:-1]] + [lines[-1]]

nb["cells"][data_cell_idx]["source"] = source_lines

# Write back
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Patched cell {data_cell_idx} with synthetic data fallback")
print(f"Total cells: {len(nb['cells'])}")
