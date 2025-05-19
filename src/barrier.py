import numpy as np
from scipy.stats import norm
import mpmath as mp
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.interpolate import interp1d  # Added missing import


def analytical_price_barrier(S, K, B, T, r, sigma):
    """
    Calculates the analytical price of a down-and-out European call barrier option.

    Parameters:
        S (float): Current price of the underlying asset.
        K (float): Strike price of the option.
        B (float): Barrier level. If the asset price reaches or exceeds this level, the option is knocked out.
        T (float): Time to maturity
        r (float): Risk-free interest rate
        sigma (float): Volatility of the underlying asset

    Returns:
        float: The analytical price of the down-and-out European call barrier option. Returns 0.0 if the option is already knocked out (S >= B).
    """

    def delta(z, sign=1):
        return (np.log(z) + (r + sign*0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))

    # If knocked out, return 0
    if S >= B:
        return 0.0
    
    # Re-occuring constants
    tau, lamb, discount = T, 2 * r / sigma**2 + 1, np.exp(-r * T)
    SK, SB, BS = S/K, S/B, B/S
    B2SK = B**2/(K*S)

    # Calculate four terms
    t1 = S*(norm.cdf(delta(SK)) - norm.cdf(delta(SB)))
    t2 = S*(BS)**lamb*(norm.cdf(delta(B2SK)) - norm.cdf(delta(BS)))
    t3 = discount*K*(norm.cdf(delta(SK, -1)) - norm.cdf(delta(SB, -1)))
    t4 = discount*K*(SB)**(2-lamb)*(norm.cdf(delta(B2SK, -1)) - norm.cdf(delta(BS, -1)))
    return t1 - t2 - t3 + t4

def monte_carlo_price_barrier(S0, K, B, T, r, sigma, m, N=200_000, seed=None):
    """
    Monte Carlo pricer for a down-and-out European call barrier option.

    Parameters:
        S0 (float): Initial asset price.
        K (float): Strike price.
        B (float): Barrier level (option knocked out if S >= B).
        T (float): Time to maturity.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        m (int): Number of time steps.
        N (int): Number of simulated paths.
        seed (int, optional): Random seed.

    Returns:
        tuple: (price, stderr) - Estimated option price and standard error.
    """
    dt = T / m
    discount = np.exp(-r * T)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, m))
    drift = (r - 0.5 * sigma ** 2) * dt
    diff = sigma * np.sqrt(dt)
    logS = np.cumsum(drift + diff * Z, axis=1)
    S_paths = S0 * np.exp(logS)
    if m > 1:
        max_S = np.maximum.accumulate(S_paths, axis=1)
    else:
        max_S = S_paths
    knocked = max_S[:, -1] >= B
    St = S_paths[:, -1]
    payoffs = np.where(~knocked, np.maximum(St - K, 0.0), 0.0)
    price = discount * payoffs.mean()
    stderr = (discount * payoffs).std(ddof=1) / np.sqrt(N)
    return price, stderr

def adjusted_barrier(
    B: float,
    sigma: float,
    T: float,
    m: int
) -> float:
    """
    Compute the effective (shifted) barrier for discrete monitoring.

    Cm(H) ≈ C(H * exp(β₁ · σ · √(T/m)))

    Parameters
    ----------
    B      : float
        Original barrier level.
    sigma  : float
        Volatility of the underlying.
    T      : float
        Time to maturity.
    m      : int
        Number of discrete monitoring points.


    Returns
    -------
    B_eff  : float
        Adjusted barrier level.
    """
    beta1 = -mp.zeta(0.5) / mp.sqrt(2 * mp.pi)

    # build adjusted barrier
    return B * np.exp(float(beta1 * sigma * np.sqrt(T / m)))


def barrier_call_BTCS(K, B, r, sigma, T, Nx, Nt):
    """
    BTCS for up-and-out barrier call.
    
    Parameters:
        K (float): Strike price.
        B (float): Barrier level.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        T (float): Time to maturity.
        Nx (int): Number of price grid points.
        Nt (int): Number of time grid points.
        
    Returns:
        tuple: (S_grid, T_grid, V) - Price grid, time grid, and option values.
    """
    # BTCS for up-and-out barrier call
    def BTCS_scheme(K, x, tau, a, b):
        Nx_len = len(x) - 1
        Nt_len = len(tau) - 1
        dx = x[1] - x[0]
        dt = tau[1] - tau[0]
        lam = dt / dx**2

        main = (1 + 2 * lam) * np.ones(Nx_len - 1)
        off = -lam * np.ones(Nx_len - 2)
        A = diags([off, main, off], offsets=[-1, 0, 1], format='csc')
        lu = splu(A)

        y = np.zeros((Nt_len + 1, Nx_len + 1))
        payoff = np.maximum(np.exp(x) - 1, 0)
        y[0, :] = (payoff) * np.exp(a * x)

        y[0,  0]       = 0.0      # S→0 boundary (payoff is anyway zero, but set for clarity)
        y[0, -1]       = 0.0      # S=B barrier

        for n in range(Nt_len):
            rhs = y[n, 1:Nx_len].copy()
            y[n+1, 1:Nx_len] = lu.solve(rhs)
            y[n+1, 0] = 0.0
            y[n+1, Nx_len] = 0.0

        return y

    x_min = np.log(1e-4)
    x_max = np.log(B / K)
    x = np.linspace(x_min, x_max, Nx + 1)
    tau = np.linspace(0, 0.5 * sigma**2 * T, Nt + 1)
    q = 2 * r / sigma**2
    a = 0.5 * (q - 1)
    b = 0.25 * (q + 1)**2

    y = BTCS_scheme(K, x, tau, a, b)
    T_grid = T - 2 * tau / (sigma**2)
    S_grid = K * np.exp(x)
    V = np.zeros_like(y)
    for i in range(len(tau)):
        V[i, :] = K * np.exp(-a * x - b * tau[i]) * y[i, :]

    return S_grid, T_grid, V


def BTCS_delta_barrier(S_grid, V, S):
    """
    Calculates delta numerically from the BTCS price array using finite differences.

    Parameters:
        S_grid (numpy.ndarray): Grid of underlying prices.
        V (numpy.ndarray): Option prices corresponding to the grid.
        S (float): Current underlying asset price for which delta is required.

    Returns:
        float: Delta of the option at the given underlying price.
    """
    # For barrier option prices, we need to use the final time slice
    V_final = V[-1, :]
    
    # Interpolate the BTCS solution
    V_interp = interp1d(S_grid, V_final, kind='cubic', fill_value="extrapolate")

    # Central finite differences
    h = 1e-4
    price_up = V_interp(S + h)
    price_down = V_interp(S - h)

    return (price_up - price_down) / (2 * h)
    