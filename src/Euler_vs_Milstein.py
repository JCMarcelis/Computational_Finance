import numpy as np

def heston_euler_vs_milstein(M, S0, V0, K, T, dt, r, rho, kappa, theta, xi, seed):
    """Calculates the Asian option price + confidence intervals using Monte Carlo simulations of the Heston model
    with Euler and Milstein discretization.

    Args:
        M (int): number of Monte Carlo paths
        S0 (float): initial underlying price
        V0 (float): initial variance
        K (float): strike price
        T (float): total time of the option
        dt (float): time increment
        r (float): risk-free interest rate
        rho (float): correlation between Brownian motions
        kappa (float): rate at which V reverts to theta
        theta (float): long-run average variance
        xi (float): vol-of-vol
        seed (int): seed for reproducibility

    Returns:
        float: option prices and confidence intervals for both discretization schemes 
    """    
    N = int(T/dt)
    rng = np.random.default_rng(seed)

    S_eul = np.full(M, S0)
    V_eul = np.full(M, V0)
    all_paths_S_eul = np.zeros(M)

    S_mil = np.full(M, S0)
    V_mil = np.full(M, V0)
    all_paths_S_mil = np.zeros(M)

    for i in range(N):
        # Sum all MC paths
        all_paths_S_eul += S_eul
        all_paths_S_mil += S_mil

        # Create correlated random variables
        Z_V = rng.standard_normal(M)
        Z2 = rng.standard_normal(M)
        Z_S = rho*Z_V + np.sqrt(1-rho**2) * Z2

        # Update variance and stock for Euler
        V_prev_eul = np.maximum(0, V_eul)
        V_eul = V_eul + kappa * (theta - V_prev_eul) * dt + xi * np.sqrt(V_prev_eul * dt) * Z_V
        S_eul = S_eul * np.exp((r - 0.5 * V_prev_eul) * dt + np.sqrt(V_prev_eul * dt) * Z_S)
        
        # Update variance and stock for Milstein
        V_prev_mil = np.maximum(0, V_mil)
        V_mil = V_mil + kappa * (theta - V_prev_mil) * dt + xi * np.sqrt(V_prev_mil * dt) * Z_V + 0.25 * xi**2 * dt * (Z_V**2 - 1)
        S_mil = S_mil * np.exp((r - 0.5 * V_prev_mil) * dt + np.sqrt(V_prev_mil * dt) * Z_S
                               + 0.5 * V_prev_mil * dt * (Z_S**2 - 1))

    # Calculate arithmetic average
    avg_paths_eul = all_paths_S_eul / N
    avg_paths_mil = all_paths_S_mil / N

    # Calculate discounted payoff
    payoff_eul = np.exp(-r*T) * np.maximum(avg_paths_eul - K, 0)
    payoff_mil = np.exp(-r*T) * np.maximum(avg_paths_mil - K, 0)

    # Calculate means and confidence intervals (p=95%)
    opt_price_eul = np.mean(payoff_eul)
    std_price_eul = 1.96 * np.std(payoff_eul) / np.sqrt(M)
    opt_price_mil = np.mean(payoff_mil)
    std_price_mil = 1.96 * np.std(payoff_mil) / np.sqrt(M)

    return opt_price_eul, std_price_eul, opt_price_mil, std_price_mil

def gbm_benchmark(M, S0, K, T, dt, r, sigma, seed):
    """Calculates Asian option using Geometric Brownian motion

    Args:
        M (int): number of Monte Carlo paths
        S0 (float): initial underlying price
        K (float): strike price
        T (float): total time of the option
        dt (float): time increment
        r (float): risk-free interest rate
        seed (int): seed for reproducibility
        sigma (float): constant volatility

    Returns:
        floats: option price and confidence intervals
    """    
    N = int(T / dt)
    rng = np.random.default_rng(seed)
    
    S = np.full(M, S0)
    path_sums = np.zeros(M)
    
    for i in range(N):
        path_sums += S
        Z = rng.standard_normal(M)
        S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    A = path_sums / N
    payoffs = np.exp(-r * T) * np.maximum(A - K, 0)
    opt_price = np.mean(payoffs)
    std_price = 1.96 * np.std(payoffs) / np.sqrt(M)
    
    return opt_price, std_price
