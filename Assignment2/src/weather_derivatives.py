import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.tsa.ar_model import ar_select_order

class TemperatureModel:
    # Temperature model with seasonal mean and stochastic components
    
    def __init__(self, a, b, a1, b1, kappa, V, U, c1, d1, c2, d2):
        self.a = a          # constant term in μ̄(t)
        self.b = b          # linear trend in μ̄(t)
        self.a1 = a1        # cosine amplitude
        self.b1 = b1        # sine amplitude
        self.kappa = kappa  # mean reversion rate
        
        # volatility params
        self.V = V    # base volatility
        self.U = U    # linear trend in volatility
        self.c1 = c1  # first harmonic sine
        self.d1 = d1  # first harmonic cosine
        self.c2 = c2  # second harmonic sine
        self.d2 = d2  # second harmonic cosine
        
        self.omega = 2.0 * np.pi / 365.25  # annual frequency

    def mu_bar(self, t_days: np.ndarray) -> np.ndarray:
        """Calculate long-term mean temperature for vector of day offsets."""
        return (
            self.a
            + self.b * t_days
            + self.a1 * np.cos(self.omega * t_days)
            + self.b1 * np.sin(self.omega * t_days)
        )

    def sigma_t(self, t_days: np.ndarray) -> np.ndarray:
        # Calculate seasonal volatility σ(t)
        return (
            self.V
            + self.U * t_days
            + self.c1 * np.sin(self.omega * t_days)
            + self.d1 * np.cos(self.omega * t_days)
            + self.c2 * np.sin(2 * self.omega * t_days)
            + self.d2 * np.cos(2 * self.omega * t_days)
        )

    def simulate_paths(self, n_paths: int,
                      start: str = "2024-12-01",
                      end: str = "2025-02-28",
                      seed: int = 42,
                      sigma_scale: float = 1.0,
                      antithetic: bool = False) -> pd.DataFrame:
        """
        Simulate n_paths daily temperatures using Euler-Maruyama scheme.
        If antithetic=True, generates correlated pairs of paths for variance reduction.
        """
        if antithetic and n_paths % 2 != 0:
            raise ValueError("n_paths must be even for antithetic sampling")
            
        half = n_paths // 2 if antithetic else None
        
        # set up simulation grid
        rng = np.random.default_rng(seed) 
        dates = pd.date_range(start=start, end=end, freq="D", tz=None)
        n_days = len(dates)
        t = np.arange(n_days)
        
        # get deterministic components
        mu = self.mu_bar(t)
        sig = np.maximum(sigma_scale * self.sigma_t(t), 1e-3) 
        temps = np.empty((n_days, n_paths), dtype=np.float32)
        temps[0, :] = mu[0]  
        
        if antithetic:
            # do antithetic sampling for variance reduction
            Z = rng.standard_normal(size=(n_days - 1, half), dtype=np.float32)
            Z_full = np.concatenate([Z, -Z], axis=1)  # pair up paths
            
            for i in range(1, n_days):
                dt = 1.0  
                d_mu = mu[i] - mu[i-1]
                mean_revert = self.kappa * (mu[i-1] - temps[i-1, :])
                temps[i, :] = (
                    temps[i-1, :]
                    + d_mu
                    + mean_revert * dt
                    + sig[i-1] * np.sqrt(dt) * Z_full[i-1, :]
                )
        else:
            # standard Monte Carlo
            for i in range(1, n_days):
                dt = 1.0
                d_mu = mu[i] - mu[i-1]
                mean_revert = self.kappa * (mu[i-1] - temps[i-1, :])
                dW = rng.standard_normal(n_paths) * np.sqrt(dt)
                temps[i, :] = (
                    temps[i-1, :]
                    + d_mu
                    + mean_revert * dt
                    + sig[i-1] * dW
                )
        
        return pd.DataFrame(temps, index=dates)


# helper function for HDD calculation
def hdd_series(temp_df: pd.DataFrame, base_temp: float = 18.0) -> pd.Series:
    """Calculate Heating Degree Days per path over the DataFrame index."""
    daily_hdd = np.maximum(base_temp - temp_df, 0.0)
    return daily_hdd.sum(axis=0)


def price_hdd_call(model: TemperatureModel,
                   n_paths: int = 100_000,
                   strike: float = 800.0,
                   cap: float = 4000.0,
                   notional: float = 20.0,
                   r: float = 0.02,
                   start: str = "2024-12-01",
                   end: str = "2025-02-28",
                   seed: int = 123,
                   antithetic: bool = False,
                   sigma_scale: float = 1.0) -> tuple[float, float]:
    """Price a capped HDD call option using Monte Carlo."""
    paths = model.simulate_paths(n_paths, start, end, seed, sigma_scale, antithetic)
    H = hdd_series(paths)
    
    payoff = np.minimum(notional * np.maximum(H - strike, 0.0), cap)
    discount = np.exp(-r * ((pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25))
    pv = discount * payoff
    price = pv.mean()
    stderr = pv.std(ddof=1) / np.sqrt(n_paths)
    
    # for debugging/reporting
    hit_cap = (payoff == cap).mean()
    return price, stderr, hit_cap

def price_hdd_put(model: TemperatureModel,
                  n_paths: int = 100_000,
                  strike: float = 800.0,
                  floor: float = 2000.0,
                  notional: float = 20.0,
                  r: float = 0.02,
                  start: str = "2024-12-01",
                  end: str = "2025-02-28",
                  seed: int = 123,
                  antithetic: bool = False,
                  sigma_scale: float = 1.0) -> tuple[float, float]:
    """Price a floored HDD put option using Monte Carlo sim."""
    paths = model.simulate_paths(n_paths, start, end, seed, sigma_scale, antithetic)
    H = hdd_series(paths)
    
    payoff = np.minimum(notional * np.maximum(strike - H, 0.0), floor)
    discount = np.exp(-r * ((pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25))
    pv = discount * payoff
    price = pv.mean()
    stderr = pv.std(ddof=1) / np.sqrt(n_paths)
    
    hit_floor = (payoff == floor).mean()
    return price, stderr, hit_floor


def price_hdd_collar(model: TemperatureModel,
                     n_paths: int = 100_000,
                     strike_call: float = 800.0,
                     cap_call: float = 4000.0,
                     strike_put: float = 750.0,
                     floor_put: float = 2000.0,
                     notional: float = 20.0,
                     **kwargs) -> tuple[float, float]:
    
    """Price a zero-premium HDD collar (long call + short put)."""
    price_c, se_c, hit_cap = price_hdd_call(
        model, n_paths=n_paths,
        strike=strike_call, cap=cap_call,
        notional=notional, **kwargs
    )
    
    price_p, se_p, hit_floor = price_hdd_put(
        model, n_paths=n_paths,
        strike=strike_put, floor=floor_put,
        notional=notional, **kwargs
    )
    
    # simple subtraction since it's long call - short put
    price_collar = price_c - price_p
    se = np.sqrt(se_c**2 + se_p**2) 
    return price_collar, se, hit_cap, hit_floor


def calibrate_from_data(df: pd.DataFrame, temp_col: str = "T_mean") -> TemperatureModel:
    """Fit model parameters from historical data."""
    # time axis for fitting seasonality
    t0 = df.index[0]
    x = (df.index - t0).days.to_numpy()
    omega = 2*np.pi/365.25
    
    # fit seasonal mean first
    def mu_bar_fit(x, a, b, a1, b1):
        return a + b*x + a1*np.cos(omega*x) + b1*np.sin(omega*x)
    
    # decent initial guesses
    init = [df[temp_col].mean(), 0.0, 7.0, 0.0]
    pars, _ = curve_fit(mu_bar_fit, x, df[temp_col], p0=init, maxfev=10000)
    a, b, a1, b1 = pars
    
    # get residuals for AR fit
    fitted = pd.Series(mu_bar_fit(x, *pars), index=df.index)
    residuals = (df[temp_col] - fitted).asfreq("D")
    
    # fit AR model to get kappa 
    res_ar = sm.tsa.ARIMA(residuals, order=(3, 0, 0), freq="D").fit()
    gamma1 = res_ar.arparams[0]
    kappa = 1 - gamma1
    
    # seasonal vol
    abs_res = np.abs(res_ar.resid.dropna())
    t_sigma = (abs_res.index - abs_res.index[0]).days.values
    
    X_sig = np.column_stack([
        np.ones_like(t_sigma),
        t_sigma,
        np.sin(omega * t_sigma), np.cos(omega * t_sigma),
        np.sin(2 * omega * t_sigma), np.cos(2 * omega * t_sigma)
    ])
    
    beta_sig, *_ = np.linalg.lstsq(X_sig, abs_res.values, rcond=None)
    V, U, c1, d1, c2, d2 = beta_sig
    
    return TemperatureModel(a, b, a1, b1, kappa, V, U, c1, d1, c2, d2)