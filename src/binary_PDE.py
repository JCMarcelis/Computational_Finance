import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.stats import norm
from scipy.interpolate import interp1d

def closed_form_price(T, r, sigma, K, S0):
    """Calculates the binary option price using the analytical expression

    Args:
        T (float): remaining time of the option
        r (float): risk-free interest rate
        sigma (float): constant volatility
        K (float): strike price
        S0 (float): initial underlying asset price

    Returns:
        float: analytical binary option price
    """    
    d_minus = (np.log(S0/K) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return np.exp(-r * T) * norm.cdf(d_minus)


def MC_binary_price(num_paths, T, r, sigma, K, S0, seed):
    """Calculates a Monte Carlo estimate of the binary option price

    Args:
        num_paths (int): number of Monte Carlo sample paths
        T (float): remaining time of the option
        r (float): risk-free interest rate
        sigma (float): constant volatility
        K (float): strike price
        S0 (float): initial underlying asset price
        seed (int): seed for reproducibility

    Returns:
        floats: Monte Carlo estimate and the p=95% confidence intervals
    """    
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(num_paths)

    S_gbm = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    S_gbm[S_gbm < K] = 0
    S_gbm[S_gbm >= K] = 1
    disc_payoffs = np.exp(-r*T) * S_gbm
    MC_price = np.mean(disc_payoffs)
    MC_CI = 1.96 * np.std(disc_payoffs) / np.sqrt(num_paths)

    return MC_price, MC_CI


def BTCS_scheme(K, x, tau, q, a, b):
    """Calculates the binary option price from black-scholes equation in the heat equation form 
       using the backward-time central-space discretization scheme. The variable transformations 
       are listed in the accompanying report.

    Args:
        K (float): strike price
        x (array): values for the stock price in transformed variables
        tau (array): time steps in transformed variables
        q (float): coefficient resulting from transformations
        a (float): coefficient resulting from transformations
        b (float): coefficient resulting from transformations

    Returns:
        array: array of transformed option prices at different times and underlying prices
    """    
    Nx  = len(x)-1
    Nt  = len(tau)-1
    dx  = x[1] - x[0]
    dt  = tau[1] - tau[0]
    lamda = dt / dx**2

    # Create sparse matrix
    main_diag = (1 + 2*lamda)*np.ones(Nx-1)
    off_diags = -lamda *np.ones(Nx-2)
    A = diags([off_diags, main_diag, off_diags], [-1,0,1], format="csc")
    lu = splu(A)

    y = np.zeros((Nt+1, Nx+1))

    # initial condition at tau=0
    payoff = (x >= 0)
    y[0,:] = (payoff/K) * np.exp(a*x)

    # time steps
    for n in range(Nt):
        # boundary at right
        y_bnd = (1/K)*np.exp(a*x[-1] + (b - q)*tau[n+1])

        # RHS vector
        rhs = y[n,1:Nx].copy()
        rhs[-1] += lamda * y_bnd

        # solve matrix eq
        y[n+1,1:Nx] = lu.solve(rhs)

        # set BC's
        y[n+1,0]   = 0
        y[n+1,Nx]  = y_bnd

    return y


def CN_scheme(K, x, tau, q, a, b):
    """Calculates the binary option price from black-scholes equation in the heat equation form 
       using the Crank-Nicolson discretization scheme. The variable transformations 
       are listed in the accompanying report.

    Args:
        K (float): strike price
        x (array): values for the stock price in transformed variables
        tau (array): time steps in transformed variables
        q (float): coefficient resulting from transformations
        a (float): coefficient resulting from transformations
        b (float): coefficient resulting from transformations

    Returns:
        array: array of transformed option prices at different times and underlying prices
    """    
    Nx  = len(x)-1
    Nt  = len(tau)-1
    dx  = x[1] - x[0]
    dt  = tau[1] - tau[0]
    lamda = dt / dx**2

    # Create sparse matrix A1
    main_diag_1 = (1 + lamda)*np.ones(Nx-1)
    off_diags_1 = -0.5*lamda *np.ones(Nx-2)
    A1 = diags([off_diags_1, main_diag_1, off_diags_1], [-1,0,1], format="csc")
    lu_A1 = splu(A1)

    # Create sparse matrix A2
    main_diag_2 = (1 - lamda)*np.ones(Nx-1)
    off_diags_2 = 0.5*lamda *np.ones(Nx-2)
    A2 = diags([off_diags_2, main_diag_2, off_diags_2], [-1,0,1], format="csc")

    y = np.zeros((Nt+1, Nx+1))

    # initial condition at tau=0
    payoff = (x >= 0)
    y[0,:] = (payoff/K) * np.exp(a*x)

    # time steps
    for n in range(Nt):
        # boundary at right
        y_bnd_old = (1/K)*np.exp(a*x[-1] + (b - q)*tau[n])
        y_bnd_new = (1/K)*np.exp(a*x[-1] + (b - q)*tau[n+1])

        # RHS vector
        rhs = A2.dot(y[n,1:Nx])
        rhs[-1] += 0.5 * lamda * y_bnd_old

        # Use k1 (y_bnd_new) on RHS:
        rhs_new = rhs.copy()
        rhs_new[-1] += 0.5 * lamda * y_bnd_new

        # solve with new RHS
        y[n+1,1:Nx] = lu_A1.solve(rhs_new)

        # set BC's
        y[n+1,0]   = 0
        y[n+1,Nx]  = y_bnd_new

    return y


def plot_option_surface(y, tau, x, T, K, sigma, a, b):
    """Plots the option surface by transforming x, tau, and y to S, t, and C

    Args:
        y (array): option prices in transformed variables
        x (array): values for the stock price in transformed variables
        tau (array): time steps in transformed variables
        T (float): total time of the option
        K (float): strike price
        sigma (float): constant volatility
        a (float): coefficient resulting from transformations
        b (float): coefficient resulting from transformations
    """    
    TAU, X = np.meshgrid(tau, x, indexing='ij')
    S = K * np.exp(X)
    t = T - 2*TAU/sigma**2
    C = K * y  * np.exp(-a*X - b*TAU)

    fig = plt.figure(figsize=(10, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(t, S, C, cmap='viridis', edgecolor='none')
    ax.set_xlabel(r'$t$', fontsize=16)
    ax.set_ylabel(r'$S$', fontsize=16)
    ax.set_zlabel(r'$C_d(t,S)$', fontsize=16)
    ax.set_title('Binary Call Option Price Surface', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    cbar = fig.colorbar(surf, shrink=0.4, aspect=10, pad=0.1)
    cbar.set_label('Binary Option Price', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()
    return  


def sens_analysis(x, tau, Nx, r, sigma, K_sing, parameter_list, parameter_to_vary='sigma'):
    """Sensitivity anaylsis for the parameters K and sigma.

    Args:
        x (array): values for the stock price in transformed variables
        tau (array): time steps in transformed variables
        Nx (int): number of steps in x
        r (float): risk-free interest rate
        sigma (float): constant volatility
        K_sing (float): original strike price
        parameter_list (array): different values for the parameter of choice
        parameter_to_vary (str, optional): parameter that changes. Defaults to 'sigma'.

    Returns:
        arrays: values for option surface and S, and values for K or sigma
    """    
    C_surface = np.zeros((len(parameter_list), Nx))

    if parameter_to_vary == 'sigma':
        for i, sig in enumerate(parameter_list):
            q = 2 * r / sig**2
            a = 0.5 * (q - 1)
            b = 0.25 * (q + 1)**2

            y_grid = CN_scheme(K_sing, x, tau, q, a, b)

            tau_final = tau[-1]
            C_grid = K_sing * y_grid[-1, :] * np.exp(-a*x - b*tau_final)
            C_surface[i, :] = C_grid

        SIG, X = np.meshgrid(parameter_list, x, indexing='ij')
        S = K_sing * np.exp(X)

        return SIG, S, C_surface

    elif parameter_to_vary == 'K':
        for i, K in enumerate(parameter_list):
            q = 2*r/sigma**2
            a = 0.5*(q - 1)
            b = 0.25*(q + 1)**2

            y_grid = CN_scheme(K, x, tau, q, a, b)
            tau_final = tau[-1]

            C_grid = K * y_grid[-1, :] * np.exp(-a*x - b*tau_final)
            C_surface[i, :] = C_grid

        K_mesh, X_mesh = np.meshgrid(parameter_list, x, indexing='ij')
        S_mesh = K_mesh * np.exp(X_mesh)

        return K_mesh, S_mesh, C_surface
    
    else:
        raise ValueError("Choose sigma or K")


def digital_delta_analytic(r, sigma, S, K, tau):
    """Calculates the binary option delta using the analytical expression

    Args:
        r (float): risk-free interest rate
        sigma (float): constant volatility
        S (float): underlying asset price
        K (float): strike price
        tau (float): time until maturity

    Returns:
        float: binary option delta
    """    
    d_minus = (np.log(S/K) + (r - 0.5*sigma**2)*tau) / (sigma * np.sqrt(tau))
    return np.exp(-r*tau) * norm.pdf(d_minus) / (sigma * np.sqrt(tau) * S)
