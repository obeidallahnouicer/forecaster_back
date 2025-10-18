import logging
from sales_forecaster import SalesForecaster
from rag_chatbot import data_loader, retriever
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
print('Load data...')
df = data_loader.load_and_preprocess()
print('Rows:', len(df))

sf = SalesForecaster(df, cache_dir='cache')
print('Preparing...')
sf.prepare_data()
print('Articles:', len(sf.grouped_data[sf.ref_col].unique()))

print('Running forecasts...')
df_all = sf.forecast_all_articles(force_recompute=True)
print('Forecasts computed:', 0 if df_all is None else len(df_all))

# list files
cache_forecasts = Path('cache') / 'forecasts'
files = sorted([p for p in cache_forecasts.glob('*.csv')])
print('Written files:', len(files))
for p in files[:20]:
    print('-', p.name)

# show top 10
if df_all is not None and not df_all.empty:
    top10 = df_all.sort_values('avg_forecast', ascending=False).head(10)
    print('\nTop 10 products by avg_forecast:')
    print(top10[['ref_article','designation','avg_forecast','next_year']].to_string(index=False))
    # show sample CSV for top product
    top0 = top10.iloc[0]['ref_article']
    safe = str(top0).replace('/', '_').replace(' ', '_')
    candidates = list(cache_forecasts.glob(f"{safe}*.csv"))
    if candidates:
        p = candidates[0]
        print('\nSample CSV for top product:', p)
        print(p.read_text(encoding='utf-8')[:1000])
else:
    print('No forecasts produced')
