import numpy as np
from scipy.stats import norm

def euler_BS(S0, r, sigma, T, dt, rng):
    """Calculates at price path using the Black-Scholes model with an Euler Scheme

    Args:
        S0 (int): initial stock price
        r (float): interest rate
        sigma (float): volatility
        T (int): total time
        dt (float): time increment
        rng (function): seeded random number generator

    Returns:
        array: price path
    """    
    N = int(T/dt)
    S = np.zeros(N)
    S[0] = S0

    for t in range(1, N):
        Z = rng.normal()
        S[t] = S[t-1] + r*S[t-1]*dt + sigma*S[t-1]*np.sqrt(dt)*Z

    return S

def delta_call(S, K, r, sigma, T):
    """Calculates option delta

    Args:
        S (float): price at time t
        K (int): strike price
        r (float): interest rate
        sigma (float): volatility
        T (float): remaining time of the option

    Returns:
        float: option delta
    """    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)


def bs_option_price(S, K, r, sigma, T):
    """Calculats the price of a European option

    Args:
        S (float): initial stock price
        K (int): strike price
        r (float): interest rate
        sigma (float): volatility
        T (float): remaining time of the option

    Returns:
        float: price of the option
    """    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def hedging_sim(S0, r, sigma, sigma_delta, T, dt, K, rng):
    """Performs a hedging simulations for one price path

    Args:
        S0 (float): initial stock price
        r (float): interest rate
        sigma (float): volatility of price path
        sigma_delta (float): volatility of option delta
        T (int): total time
        dt (float): time increment
        K (int): strike price
        rng (function): seeded random number generator

    Returns:
        array: array containing hedging errors
    """    
    # simulate price path
    S = euler_BS(S0, r, sigma, T, dt, rng)

    # initialise
    remaining_T = T
    delta_start = delta_call(S[0], K, r, sigma_delta, remaining_T)
    option_price = bs_option_price(S[0], K, r, sigma, remaining_T)
    stock_pos = delta_start
    cash = option_price - stock_pos*S[0]

    N = int(T/dt)
    for t in range(1, N):
        remaining_T -= dt
        cash *= np.exp(r*dt) # interest 

        # adjust hedge
        new_delta = delta_call(S[t], K, r, sigma_delta, remaining_T)
        stock_pos_change = new_delta - delta_start
        cash -= stock_pos_change * S[t]
        delta_start = new_delta

    # calculate hedge error
    payoff = -max(S[-1] - K, 0)
    stock_value = delta_start * S[-1]
    hedge_pnl = payoff + stock_value + cash
        
    return hedge_pnl


def multiple_hedge_sims(num_runs, S0, r, sigma, sigma_delta, T, dt, K, seed):
    """Performs many runs of the hedging simulation to obtain an average with confindence intervals

    Args:
        num_runs (int): number of simulations
        S0 (float): initial stock price
        r (float): interest rate
        sigma (float): volatility of price path
        sigma_delta (float): volatility of option delta
        T (int): total time
        dt (float): time increment
        K (int): strike price
        rng (function): seeded random number generator

    Returns:
        floats: mean + CI of the hedging error for some parameter combination
    """    
    results = [hedging_sim(S0, r, sigma, sigma_delta, T, dt, K, np.random.default_rng(seed+i)) for i in range(num_runs)]
    mean_res = np.mean(results)
    CI_res = 1.96 * np.std(results) / np.sqrt(num_runs)
    return mean_res, CI_res