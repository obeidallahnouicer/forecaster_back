from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
csv_path = project_root / "forecast-summary.csv"
print(f"Loading CSV from: {csv_path}")

df = pd.read_csv(csv_path)
# coerce trend_pct numeric
if 'trend_pct' in df.columns:
    df['trend_pct_num'] = pd.to_numeric(df['trend_pct'], errors='coerce')
    zeros = df[df['trend_pct_num'] == 0]
    print(f"Total rows: {len(df)}, rows with trend_pct == 0: {len(zeros)}")
    if not zeros.empty:
        print(zeros[['ref_article','trend_pct','trend_pct_num','data_points']].head(50).to_string(index=False))
    else:
        print('No exact zeros found in numeric coercion.')
else:
    print('trend_pct column not present')
