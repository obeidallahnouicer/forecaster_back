from pathlib import Path
import pandas as pd
import json

project_root = Path(__file__).resolve().parent.parent
csv_path = project_root / "forecast-summary.csv"
print(f"Loading CSV from: {csv_path}")

df = pd.read_csv(csv_path)
# show columns
print("Columns:", df.columns.tolist()[:50])

from rag_chatbot.dataset_analyzer import DatasetAnalyzer, MIN_DATA_POINTS_FOR_RELIABILITY

an = DatasetAnalyzer(df)
print("MIN_DATA_POINTS_FOR_RELIABILITY =", MIN_DATA_POINTS_FOR_RELIABILITY)

stable = an.find_most_stable_product()
print("find_most_stable_product ->", stable)

# show top 10 sorted by abs(trend_pct), data_points desc, avg_forecast desc like our logic
candidates = an.df[an.df['trend_pct'].notna()].copy()
# ensure numeric
candidates['trend_pct'] = pd.to_numeric(candidates['trend_pct'], errors='coerce')
if 'data_points' in candidates.columns:
    candidates['data_points'] = pd.to_numeric(candidates['data_points'], errors='coerce').fillna(0).astype(int)
else:
    candidates['data_points'] = 0
if 'avg_forecast' in candidates.columns:
    candidates['avg_forecast'] = pd.to_numeric(candidates['avg_forecast'], errors='coerce').fillna(0)
else:
    candidates['avg_forecast'] = 0
candidates['abs_trend'] = candidates['trend_pct'].abs()

sorted_cand = candidates.sort_values(by=['abs_trend','data_points','avg_forecast'], ascending=[True, False, False])
print('\nTop 20 candidates:')
print(sorted_cand[['ref_article','trend_pct','abs_trend','data_points','avg_forecast']].head(20).to_string(index=False))

# show any exact zeros
zeros = candidates[candidates['trend_pct']==0]
print(f"\nNumber of exact zero trend_pct: {len(zeros)}")
print(zeros[['ref_article','trend_pct','data_points','avg_forecast']].head(20).to_string(index=False))
