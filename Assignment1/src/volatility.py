import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import yfinance as yf

# Define color scheme
COLORS = {
    'primary': 'black',
    'secondary': 'red',
    'tertiary': 'green',
    'quaternary': 'blue'
}

# Define consistent y-axis limits for volatility plots
VOLATILITY_YLIM = (0, 100)  # 0% to 100% annualized volatility

def download_stock_data(ticker, start_date, end_date, auto_adjust=True):
    """Download historical stock data using yfinance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=auto_adjust)
        if data.empty:
            raise ValueError(f"No data available for ticker {ticker}")
        return data
    except Exception as e:
        raise ValueError(f"Error downloading data for {ticker}: {str(e)}")

def calculate_returns(prices):
    """Calculate percentage returns from price series."""
    return prices.pct_change().dropna()

def calculate_drift(returns):
    """Calculate the drift (μ) from returns."""
    return float(np.mean(returns))

def calculate_classical_volatility(returns, mu=None, annualize=False):
    """Calculate classical volatility estimator."""
    if mu is None:
        mu = calculate_drift(returns)
    vol = float(np.sqrt(((returns - mu)**2).sum() / (len(returns) - 1)).iloc[0])
    if annualize:
        vol *= np.sqrt(252)
    return vol

def calculate_parkinson_volatility(high, low, annualize=False):
    """Calculate Parkinson volatility estimator."""
    ln_hl = np.log(high / low)
    T = len(high)
    sigma = float(np.sqrt((ln_hl**2).sum() / (4 * T * np.log(2))).iloc[0])
    if annualize:
        sigma *= np.sqrt(252)
    return sigma

def calculate_garman_klass_volatility(high, low, open_price, close, annualize=False):
    """Calculate Garman-Klass volatility estimator."""
    ln_hl = np.log(high / low)
    ln_co = np.log(close / open_price)
    T = len(high)
    result = float(np.sqrt(
        ((ln_hl**2).sum() / (2 * T)) -
        ((2 * np.log(2) - 1) / T) * (ln_co**2).sum()
    ).iloc[0]
    ).iloc[0]
    sigma = float(result.iloc[0] if hasattr(result, 'iloc') else result)
    if annualize:
        sigma *= np.sqrt(252)
    return sigma

def calculate_rolling_volatility(returns, window=30, annualize=True):
    """Calculate rolling volatility."""
    rolling_std = returns.rolling(window=window).std()
    if annualize:
        rolling_std *= np.sqrt(252)
    return rolling_std * 100  # Convert to percentage

def calculate_rolling_parkinson(high, low, window=30, annualize=True):
    """Calculate rolling Parkinson volatility estimator."""
    ln_hl = np.log(high / low)
    rolling_parkinson = np.sqrt(
        ln_hl.rolling(window=window).apply(
            lambda x: (x**2).sum() / (4 * window * np.log(2))
        )
    )
    if annualize:
        rolling_parkinson *= np.sqrt(252)
    return rolling_parkinson * 100  # Convert to percentage

def calculate_rolling_garman_klass(data, window=30, annualize=True):
    """Calculate rolling Garman-Klass volatility estimator."""
    def gk_estimator(window_data):
        hl = np.log(window_data['High'] / window_data['Low'])
        co = np.log(window_data['Close'] / window_data['Open'])
        return np.sqrt(
            ((hl**2).sum() / (2 * len(window_data))) - 
            ((2 * np.log(2) - 1) / len(window_data)) * (co**2).sum()
        )
    
    rolling_gk = pd.Series(index=data.index, dtype=float)
    for i in range(window, len(data)):
        window_data = data.iloc[i-window:i]
        rolling_gk.iloc[i] = gk_estimator(window_data)
    
    if annualize:
        rolling_gk *= np.sqrt(252)
    return rolling_gk * 100  # Convert to percentage

def calculate_realized_variance_proper(returns, m):
    """Calculate time-series average of realized variance.
    
    Args:
        returns (Series): Series of returns
        m (int): Number of samples per day
        
    Returns:
        float: Time-series average of realized variance
    """
    # Calculate trading day duration in minutes (assuming 6.5 hours trading day)
    trading_minutes = 6.5 * 60
    min_per_sample = int(trading_minutes/m)  # Minutes per sample
    freq = f'{min_per_sample}T'  # Convert to minutes
    
    # Resample returns to m samples per day
    resampled_returns = returns.resample(freq).sum().dropna()
    
    # Group by date to calculate daily realized variance
    daily_rv = resampled_returns.groupby(resampled_returns.index.date).apply(
        lambda x: (x**2).sum() / min_per_sample  # Scale by minutes per sample
    )
    
    # Calculate time-series average
    avg_rv = daily_rv.mean()
    
    return avg_rv

def plot_volatility_signature(returns, figsize=(8, 5)):
    """Create volatility signature plot.
    
    The plot shows how the time-series average of realized variance
    varies with the number of samples per day (m).
    
    Args:
        returns (Series): Series of returns
        figsize (tuple): Figure size
        
    Returns:
        Figure: Matplotlib figure
    """
    # Define number of samples per day and corresponding minutes
    # From 390 min (daily) to 5 min intervals in 6.5h trading day
    samples = [1, 2, 4, 8, 13, 26, 39, 78]  # Samples per day
    minutes = [int(390/m) for m in samples]  # Minutes per sample
    
    realized_vols = []
    for m in samples:
        # Calculate realized variance
        rv = calculate_realized_variance_proper(returns, m)
        # Convert to annualized volatility in percentage
        vol = np.sqrt(rv * 252) * 100
        realized_vols.append(vol)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.plot(minutes, realized_vols, color=COLORS['primary'], marker='o')
    ax.set_title('Volatility Signature Plot', fontsize=16)
    ax.set_xlabel('Minutes per Sample', fontsize=15)
    ax.set_ylabel('Average Annualized Volatility (%)', fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, ls='dashed')
    
    # Add x-axis ticks for all minute points
    ax.set_xticks(minutes)
    ax.set_xticklabels(minutes, rotation=45)
    
    plt.tight_layout()
    return fig

def plot_rolling_volatility(rolling_std, rolling_parkinson, rolling_gk, ticker, figsize=(8, 5)):
    """Plot rolling volatility estimators."""
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.plot(rolling_std.index, rolling_std, label='Classical', color=COLORS['primary'])
    ax.plot(rolling_parkinson.index, rolling_parkinson, label='Parkinson', color=COLORS['secondary'])
    ax.plot(rolling_gk.index, rolling_gk, label='Garman-Klass', color=COLORS['tertiary'])
    ax.set_title(f'{ticker} Rolling Volatility Estimators', fontsize=16)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, ls='dashed')
    ax.set_ylim(VOLATILITY_YLIM)
    return fig

def plot_realized_vs_implied(realized_vols, implied_vol, figsize=(12, 6)):
    """Plot realized vs implied volatility comparison."""
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # Align data on common dates
    common_dates = pd.Index([])
    for vol in realized_vols.values():
        if not vol.empty:
            if common_dates.empty:
                common_dates = vol.index
            else:
                common_dates = common_dates.intersection(vol.index)
    
    if not implied_vol.empty:
        common_dates = common_dates.intersection(implied_vol.index)
    
    # Plot realized volatility estimators
    for name, vol in realized_vols.items():
        if not vol.empty:
            color = {
                'Classical': COLORS['primary'],
                'Parkinson': COLORS['secondary'],
                'Garman-Klass': COLORS['tertiary']
            }[name]
            ax.plot(common_dates, vol[common_dates], label=name, color=color)
    
    # Plot implied volatility
    if not implied_vol.empty:
        ax.plot(common_dates, implied_vol[common_dates], 
                label='Implied', color=COLORS['quaternary'], linewidth=2)
    
    ax.set_title('Realized vs Implied Volatility Comparison', fontsize=16)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel('Annualized Volatility (%)', fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, ls='dashed')
    ax.set_ylim(VOLATILITY_YLIM)
    return fig

def create_volatility_comparison_df(vol_classical, vol_parkinson, vol_gk, annualize=True):
    """Create a comparison DataFrame of volatility estimators."""
    comparison_df = pd.DataFrame({
        'Estimator': ['Classical', 'Parkinson', 'Garman-Klass'],
        'Value': [vol_classical, vol_parkinson, vol_gk]
    })
    
    if annualize:
        comparison_df['Annualized (%)'] = [
            vol_classical * np.sqrt(252) * 100,
            vol_parkinson * np.sqrt(252) * 100,
            vol_gk * np.sqrt(252) * 100
        ]
    return comparison_df

def download_vix_data(start_date, end_date):
    """Download VIX data for the specified time period."""
    try:
        vix_data = yf.download("^VIX", start=start_date, end=end_date)
        if vix_data.empty:
            raise ValueError("No VIX data available")
        return vix_data
    except Exception as e:
        raise ValueError(f"Error downloading VIX data: {str(e)}")


def get_option_chain(symbol, days_to_expiry=30):
    """
    Load call and put option chains for the given symbol and select options
    with expiry closest to the target days_to_expiry.

    Returns:
        calls_df, puts_df, expiry_date
    """
    try:
        calls = pd.read_csv('./data/call_data.csv')
        puts = pd.read_csv('./data/put_data.csv')

        calls['expiry'] = pd.to_datetime(calls['lastTradeDate']).dt.date
        puts['expiry'] = pd.to_datetime(puts['lastTradeDate']).dt.date

        # Find common expiry closest to target
        today = datetime.date.today()
        target_date = today + datetime.timedelta(days=days_to_expiry)
        common_expiries = set(calls['expiry']) & set(puts['expiry'])

        if not common_expiries:
            raise ValueError("No matching expiries found.")

        closest_expiry = min(common_expiries, key=lambda x: abs(x - target_date))

        # Filter to the selected expiry
        calls = calls[calls['expiry'] == closest_expiry]
        puts = puts[puts['expiry'] == closest_expiry]

        calls["mid"] = (calls["bid"] + calls["ask"]) / 2
        puts["mid"] = (puts["bid"] + puts["ask"]) / 2

        return calls, puts, closest_expiry

    except Exception as e:
        raise ValueError(f"Error loading option chain: {e}")
    

def calculate_forward_price(S_t, r, tau):
    """Calculate forward price.
    
    Args:
        S_t (float): Current spot price
        r (float): Risk-free rate
        tau (float): Time to expiration in years
        
    Returns:
        float: Forward price
    """
    return S_t * np.exp(r * tau)

def filter_otm_options(puts_df, calls_df, forward_price):
    """Filter out-of-the-money options.
    
    Args:
        puts_df (DataFrame): Put options data
        calls_df (DataFrame): Call options data
        forward_price (float): Forward price
        
    Returns:
        tuple: (OTM puts DataFrame, OTM calls DataFrame)
    """
    otm_puts = puts_df[puts_df['strike'] < forward_price].copy()
    otm_calls = calls_df[calls_df['strike'] > forward_price].copy()
    return otm_puts, otm_calls

def calculate_vix(otm_puts, otm_calls, r, tau):
    """Calculate VIX index using CBOE methodology.
    
    Args:
        otm_puts (DataFrame): OTM put options
        otm_calls (DataFrame): OTM call options
        r (float): Risk-free rate
        tau (float): Time to expiration in years
        
    Returns:
        float: Estimated VIX value
    """
    otm_puts = otm_puts.sort_values("strike", ascending=False)
    otm_calls = otm_calls.sort_values("strike", ascending=True)
    
    puts_contrib = 0
    for i, row in otm_puts.iterrows():
        K = row["strike"]
        Q = row["mid"]  # Use mid-price, never use Last-price
        next_K = otm_puts.loc[otm_puts["strike"] < K, "strike"].max() if not otm_puts[otm_puts["strike"] < K].empty else K
        delta_K = K - next_K
        puts_contrib += delta_K / (K**2) * np.exp(r*tau) * Q
    
    calls_contrib = 0
    for i, row in otm_calls.iterrows():
        K = row["strike"]
        Q = row["mid"]
        next_K = otm_calls.loc[otm_calls["strike"] > K, "strike"].min() if not otm_calls[otm_calls["strike"] > K].empty else K
        delta_K = next_K - K
        calls_contrib += delta_K / (K**2) * np.exp(r*tau) * Q
    
    var = (2 / tau) * (puts_contrib + calls_contrib)
    vix = 100 * np.sqrt(var)
    return vix

def calculate_correlation(x, y):
    """Calculate correlation coefficient between two series."""
    return float(x.corr(y))

def test_cointegration(x, y):
    """Perform cointegration test between two series.
    
    Args:
        x (Series): First time series
        y (Series): Second time series
        
    Returns:
        tuple: Cointegration test results (test statistic, p-value)
    """
    return coint(x, y)

def perform_regression_analysis(X, y, add_constant=True):
    """Perform regression and return results."""
    X = pd.DataFrame(X)
    if add_constant:
        X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    return results

def analyze_spx_vix_relationship(spx_returns, vix_data):
    """Analyze relationship between SPX returns and VIX."""
    # Align data
    common_idx = spx_returns.index.intersection(vix_data.index)
    spx_returns = spx_returns.loc[common_idx].dropna()
    vix_data = vix_data.loc[common_idx].dropna()

    vix_changes = vix_data.pct_change().dropna()

    common_idx = spx_returns.index.intersection(vix_changes.index)
    spx_returns = spx_returns[common_idx]
    vix_changes = vix_changes[common_idx]

    results = perform_regression_analysis(vix_changes, spx_returns)
    corr = spx_returns.corr(vix_changes)
    return results, corr

def analyze_returns_volatility_relationship(spx_returns, realized_vol):
    """Analyze relationship between SPX returns and realized volatility."""
    common_idx = spx_returns.index.intersection(realized_vol.index)
    spx_returns = spx_returns.loc[common_idx].dropna()
    realized_vol = realized_vol.loc[common_idx].dropna()

    vol_changes = realized_vol.pct_change().dropna()

    common_idx = spx_returns.index.intersection(vol_changes.index)
    spx_returns = spx_returns[common_idx]
    vol_changes = vol_changes[common_idx]

    results = perform_regression_analysis(vol_changes, spx_returns)
    corr = spx_returns.corr(vol_changes)
    return results, corr

def format_regression_results(results, precision=4):
    """Generate LaTeX-formatted regression results."""
    coef = results.params
    std_err = results.bse
    t_stat = results.tvalues
    p_val = results.pvalues
    r2 = results.rsquared
    adj_r2 = results.rsquared_adj

    latex = "\\begin{tabular}{lcccc}\n"
    latex += "\\hline\n"
    latex += "Variable & Coefficient & Std. Error & t-stat & p-value \\\\\n"
    latex += "\\hline\n"

    for var in coef.index:
        latex += f"{var} & {coef[var]:.{precision}f} & {std_err[var]:.{precision}f} & "
        latex += f"{t_stat[var]:.{precision}f} & {p_val[var]:.{precision}f} \\\\\n"

    latex += "\\hline\n"
    latex += f"$R^2$ & \\multicolumn{{4}}{{c}}{{{r2:.{precision}f}}} \\\\\n"
    latex += f"Adj. $R^2$ & \\multicolumn{{4}}{{c}}{{{adj_r2:.{precision}f}}} \\\\\n"
    latex += "\\hline\n"
    latex += "\\end{tabular}"
    
    return latex

def format_correlation_matrix(df, precision=4):
    """Format a LaTeX-style correlation matrix."""
    corr = df.corr().round(precision)
    latex = "\\begin{tabular}{" + "c" * (len(corr.columns)+1) + "}\n\\hline\n"
    latex += " & " + " & ".join(corr.columns) + " \\\\\n\\hline\n"
    for row in corr.index:
        latex += row + " & " + " & ".join(map(str, corr.loc[row])) + " \\\\\n"
    latex += "\\hline\n\\end{tabular}"
    return latex