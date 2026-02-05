# /// script
# requires-python = ">=3.11"
# dependencies = ["yfinance>=0.2.36", "pandas>=2.0"]
# ///
import yfinance as yf
import pandas as pd

df = yf.download('SPY', start='2020-01-01', progress=False, auto_adjust=True)
print(f'DataFrame shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print(f'Column type: {type(df.columns)}')
if isinstance(df.columns, pd.MultiIndex):
    print(f'MultiIndex levels: {df.columns.names}')
print(df.head().to_string())
