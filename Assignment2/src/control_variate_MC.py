import numpy as np
from scipy.stats import norm

def analytical_price(S0, r, T, K, sigma, N):
    """Calculates the analytical geometric asian option price.

    Args:
        S0 (float): initial underlying price
        r (float): risk-free interest rate
        T (float): total time of the option
        K (float): strike price
        sigma (float): constant volatility
        N (int): number of mesh points

    Returns:
        float: analytical geometric asian option price
    """    
    sigma_tilde = sigma * np.sqrt((2*N+ 1) / (6*(N+1)))
    r_tilde = ((r - 0.5 * sigma**2) + sigma_tilde**2) / 2

    d1 = (np.log(S0/K) + (r_tilde + 0.5*sigma_tilde**2) * T) / (np.sqrt(T)*sigma_tilde)
    d2 = (np.log(S0/K) + (r_tilde - 0.5*sigma_tilde**2) * T) / (np.sqrt(T)*sigma_tilde)

    return S0 * np.exp((r_tilde - r)*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def optimal_c(X, Y):
    """Calculates the optimal control variate coefficient

    Args:
        X (array): payoffs under the GBM model
        Y (array): payoffs under the Heseton model

    Returns:
        float: optimal c
    """    
    cov = np.sum((Y - np.mean(Y))*(X - np.mean(X)))
    var = np.sum((X - np.mean(X))**2)
    return cov/var

def control_variate_MC(M, S0, V0, K, T, dt, r, rho, kappa, theta, xi, sigma, c, seed):
    """Calculates the asian option price using plain Monte Carlo and control variate Monte Carlo

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
        c (float): control variate coefficient
        seed (int): seed for reproducibility

    Returns:
        dict: contains option prices with standard error and variance, and the payoffs
    """    
    N = int(T/dt)
    rng = np.random.default_rng(seed)

    S_hest = np.full(M, S0)
    V_hest = np.full(M, V0)
    all_paths_S_hest = np.zeros(M)

    S_gbm = np.full(M, S0)
    all_paths_S_gbm = np.zeros(M)

    for i in range(N):
        # Sum all MC paths
        all_paths_S_hest += S_hest
        all_paths_S_gbm += np.log(S_gbm)

        # Create correlated random variables
        Z_V = rng.standard_normal(M)
        Z2 = rng.standard_normal(M)
        Z_S = rho*Z_V + np.sqrt(1-rho**2) * Z2

        # Update variance and stock for Heston (Milstein)
        V_prev = np.maximum(0, V_hest)
        V_hest = V_hest + kappa * (theta - V_prev) * dt + xi * np.sqrt(V_prev * dt) * Z_V + 0.25 * xi**2 * dt * (Z_V**2 - 1)
        S_hest = S_hest * np.exp((r - 0.5 * V_prev) * dt + np.sqrt(V_prev * dt) * Z_S
                               + 0.5 * V_prev * dt * (Z_S**2 - 1))
        
        # Update GBM
        S_gbm = S_gbm * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_S)

    # Calculate arithmetic average
    avg_paths_hest = all_paths_S_hest / N
    avg_paths_gbm = np.exp(all_paths_S_gbm / N)

    # Calculate discounted payoff
    Y = np.exp(-r*T) * np.maximum(avg_paths_hest - K, 0)
    X = np.exp(-r*T) * np.maximum(avg_paths_gbm - K, 0)

    # Plain Monte Carlo price
    plain_price = np.mean(Y)
    plain_CI = np.std(Y) / np.sqrt(M)
    plain_var = np.var(Y)

    # Control Variate estimator
    analy_price = analytical_price(S0, r, T, K, sigma, N)
    cv_paths = Y + c * (analy_price - X)
    cv_price = np.mean(cv_paths)
    cv_CI = np.std(cv_paths) / np.sqrt(M)
    cv_var = np.var(cv_paths)

    return {
        'plain': [plain_price, plain_CI, plain_var],
        'control_var': [cv_price, cv_CI, cv_var],
        'analy_price': analy_price,
        'payoffs': [Y, X]
    }


def heston_sensitivity(M, S0, V0, K, T, dt, r, rho, kappa, theta, xi, sigma, c, seed, parameter_to_vary='xi'):
    """Runs Monte Carlo simulations of the Heston model with and without control variates, the user can vary xi, rho, or K.

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
        c (float): control variate coefficient
        seed (int): seed for reproducibility
        parameter_to_vary (str, optional): Heston parameters to vary. Defaults to 'xi'.

    Returns:
        arrays: contains prices, standard errors
    """    
    plain_prices = np.zeros(5)
    plain_stderrs = np.zeros(5)
    cv_prices = np.zeros(5)
    cv_stderrs = np.zeros(5)

    if parameter_to_vary == 'xi':
        xis = np.linspace(0.1, 1, 5)
        for i, xi in enumerate(xis):
            results = control_variate_MC(M, S0, V0, K, T, dt, r, rho, kappa, theta, xi, sigma, c, seed)
            plain_prices[i] = results['plain'][0]
            plain_stderrs[i] = results['plain'][1]
            cv_prices[i] = results['control_var'][0]
            cv_stderrs[i] = results['control_var'][1]
        return plain_prices, plain_stderrs, cv_prices, cv_stderrs, xis
    
    if parameter_to_vary == 'rho':
        rhos = np.linspace(-0.9, 0.5, 5)
        for i, rho in enumerate(rhos):
            results = control_variate_MC(M, S0, V0, K, T, dt, r, rho, kappa, theta, xi, sigma, c, seed)
            plain_prices[i] = results['plain'][0]
            plain_stderrs[i] = results['plain'][1]
            cv_prices[i] = results['control_var'][0]
            cv_stderrs[i] = results['control_var'][1]
        return plain_prices, plain_stderrs, cv_prices, cv_stderrs, rhos

    if parameter_to_vary == 'K':
        ks = np.linspace(95, 105, 5)
        for i, K in enumerate(ks):
            results = control_variate_MC(M, S0, V0, K, T, dt, r, rho, kappa, theta, xi, sigma, c, seed)
            plain_prices[i] = results['plain'][0]
            plain_stderrs[i] = results['plain'][1]
            cv_prices[i] = results['control_var'][0]
            cv_stderrs[i] = results['control_var'][1]
        return plain_prices, plain_stderrs, cv_prices, cv_stderrs, ks
