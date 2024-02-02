import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# Define the tickers and fetch data
tickers = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'SPY', '^VIX']
end_date = pd.to_datetime('today')
start_date = end_date - pd.DateOffset(years=3)
data = yf.download(tickers, start=start_date, end=end_date)

# Calculate daily portfolio returns and VIX daily returns
daily_returns = data['Adj Close'].dropna().pct_change().dropna()
portfolio_daily_returns = daily_returns[tickers[:-1]].mean(axis=1)
vix_daily_returns = daily_returns[tickers[-1]]

# Resample for quarterly data points
portfolio_quarterly_returns = portfolio_daily_returns.resample('QE').mean()
vix_quarterly_returns = vix_daily_returns.resample('QE').mean()

# Align and shift VIX returns to avoid look-ahead bias
aligned_vix = portfolio_daily_returns.shift(1).dropna()
aligned_portfolio = vix_daily_returns[aligned_vix.index]

# Combine the features and target into one DataFrame
combined_df = pd.DataFrame({
    'features': aligned_vix,
    'target': aligned_portfolio
})

# Split the combined DataFrame and dates
combined_train, combined_test, dates_train, dates_test = train_test_split(
    combined_df, combined_df.index, test_size=0.2, random_state=42
)

# Extract X_train, X_test, y_train, y_test
X_train = combined_train['features'].values.reshape(-1, 1)
y_train = combined_train['target'].values
X_test = combined_test['features'].values.reshape(-1, 1)
y_test = combined_test['target'].values

# Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train.ravel())
predictions = rf.predict(X_test)

# Calculate the daily absolute difference and apply EWMA
daily_abs_diff = np.abs(portfolio_daily_returns - rf.predict(vix_daily_returns.values.reshape(-1, 1)))
ewma_abs_diff = daily_abs_diff.ewm(span=7, adjust=False).mean()

# Convert test dates for matplotlib
np_test_dates = mdates.date2num(dates_test.to_pydatetime())

# Plotting scatter for actual and predicted returns
plt.figure(figsize=(14, 7))
plt.scatter(np_test_dates, y_test, color='blue', label='Actual Quarterly Returns', alpha=0.6)
plt.scatter(np_test_dates, predictions, color='red', label='Predicted Quarterly Returns', alpha=0.6)

# Smooth line of EWMA of daily absolute difference
daily_dates = mdates.date2num(portfolio_daily_returns.index.to_pydatetime())
plt.plot(daily_dates, ewma_abs_diff, color='black', label='EWMA of Absolute Difference', linewidth=2)

# Formatting the plot
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)
plt.ylabel('Returns (%)')
plt.title('Quarterly Returns and EWMA of Absolute Difference')
plt.legend()

# Convert y-axis labels to percentage format
fmt = '%.2f%%'  # Format you want the ticks, e.g. '40%'
yticks = plt.gca().get_yticks()
plt.gca().set_yticklabels([fmt % (y * 100) for y in yticks])

plt.tight_layout()
plt.show()
