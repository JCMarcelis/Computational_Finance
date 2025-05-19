import numpy as np
from scipy.optimize import minimize, Bounds, brentq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import os
from scipy.integrate import quad

def load_market_data(date):
    raw = np.load("data/raw_ivol_surfaces.npy", allow_pickle=True).item()
    interp = np.load("data/interp_ivol_surfaces.npy", allow_pickle=True).item()

    return {
            'strikes': raw[date]['strikes'],
            'tenors': raw[date]['tenors'],
            'vols': raw[date]['vols']}

def heston_cf(phi_complex, T, S0, r, kappa, theta, sigma, rho, v0):
    if T < 1e-9:
        return np.exp(1j * phi_complex * np.log(S0 if S0 > 0 else 1e-100)) # Avoid log(0)

    i = 1j
    x0 = np.log(S0)
    xi = kappa - sigma * rho * phi_complex * i
    
    d_squared_term = phi_complex**2 + phi_complex * i 
    d_val_squared = xi**2 + sigma**2 * d_squared_term
    d_val = np.sqrt(d_val_squared)

    exp_comp1_drift = phi_complex * (x0 + r * T) * i
    small_dT_threshold = 1e-7

    if np.abs(d_val * T) < small_dT_threshold:
        C_prime_for_kappa_theta = xi * T
        # D_prime_for_v0 corresponds to sigma^2 * D_Heston(T, phi_complex) where D_Heston is coeff of v0
        D_prime_for_v0 = sigma**2 * (phi_complex*i - phi_complex**2) * T / 2.0
    else:
        den_g2 = xi + d_val
        if np.isclose(np.abs(den_g2), 0.0): return np.nan
        g2 = (xi - d_val) / den_g2
        exp_neg_dT = np.exp(-d_val * T)
        
        A_log = 1 - g2 * exp_neg_dT
        B_log = 1 - g2
        if np.isclose(np.abs(B_log), 0.0): return np.nan
        
        log_argument = A_log / B_log
        if np.isclose(np.abs(log_argument),0.0) or \
           (hasattr(log_argument, 'real') and log_argument.real <= 0 and np.isclose(log_argument.imag,0.0)):
             return np.nan
        log_val_calc = np.log(log_argument)
        if np.isnan(log_val_calc).any(): return np.nan
        C_prime_for_kappa_theta = (xi - d_val) * T - 2 * log_val_calc
        
        den_D_frac = A_log 
        if np.isclose(np.abs(den_D_frac), 0.0): return np.nan
        D_prime_for_v0 = (xi - d_val) * (1 - exp_neg_dT) / den_D_frac

    if np.isclose(sigma, 0.0):
        exp_comp2_kappa_theta = 0.0 if np.isclose(kappa*theta,0.0) or np.isclose(np.abs(C_prime_for_kappa_theta),0.0) else np.nan
        exp_comp3_v0 = 0.0 if np.isclose(v0,0.0) or np.isclose(np.abs(D_prime_for_v0),0.0) else np.nan
    else:
        exp_comp2_kappa_theta = (kappa * theta / sigma**2) * C_prime_for_kappa_theta
        exp_comp3_v0 = (v0 / sigma**2) * D_prime_for_v0
    
    final_exponent = exp_comp1_drift + exp_comp2_kappa_theta + exp_comp3_v0
    if np.isnan(final_exponent).any(): return np.nan
        
    return np.exp(final_exponent)

def Pj(j_index, K_strike, T_maturity, S0, r, kappa, theta, sigma, rho, v0):
    i_complex = 1j
    def integrand(u_integrand):
        phi_arg = u_integrand - i_complex if j_index == 1 else u_integrand
        cf_val = heston_cf(phi_arg, T_maturity, S0, r, kappa, theta, sigma, rho, v0)
        if np.isnan(cf_val).any(): return np.nan

        if np.isclose(u_integrand, 0.0): return 0.0 
        
        denom = (i_complex * u_integrand * S0 * np.exp(r * T_maturity)) if j_index == 1 else (i_complex * u_integrand)
        if np.isclose(np.abs(denom), 0.0): return 0.0

        term = np.exp(-i_complex * u_integrand * np.log(K_strike)) * cf_val / denom
        return np.real(term)

    try:
        integral_value, _ = quad(integrand, 1e-8, 200, limit=200, epsabs=1e-9, epsrel=1e-9)
    except Exception:
        return np.nan
    return np.nan if np.isnan(integral_value) else (0.5 + (1 / np.pi) * integral_value)

def heston_call_price(S0, K_strike, r, T_maturity, kappa, theta, sigma, rho, v0):
    P1 = Pj(1, K_strike, T_maturity, S0, r, kappa, theta, sigma, rho, v0)
    P2 = Pj(2, K_strike, T_maturity, S0, r, kappa, theta, sigma, rho, v0)

    if np.isnan(P1) or np.isnan(P2):
        return np.nan
    price = S0 * P1 - K_strike * np.exp(-r * T_maturity) * P2
    return max(0, price) if not np.isnan(price) else np.nan

def calculate_implied_volatility(price, S, K, r, T):
    if price < (max(0.0, S - K * np.exp(-r*T)) - 1e-7) or price > (S + 1e-7) : # allow for small numerical error from max(0,price)
         return np.nan
    try:
        objective = lambda sigma_iv: (S * norm.cdf((np.log(S/K) + (r + 0.5*sigma_iv**2)*T) / (sigma_iv*np.sqrt(T) + 1e-12)) - \
                                      K * np.exp(-r*T) * norm.cdf((np.log(S/K) + (r + 0.5*sigma_iv**2)*T) / (sigma_iv*np.sqrt(T) + 1e-12) - sigma_iv*np.sqrt(T))) - price
        return brentq(objective, 1e-6, 5.0, xtol=1e-7, rtol=1e-7, maxiter=100)
    except (ValueError, RuntimeError): # Brentq might fail if objective doesn't change sign or other issues
        return np.nan

def calibrate_heston(market_data, S0_market, r_market, 
                                initial_guess_params=[2.0, 0.05, 0.20, -0.7, 0.04]):
    eps_b = 1e-5 
    param_bounds = Bounds(
        [eps_b, eps_b, eps_b, -1 + eps_b, eps_b], 
        [15.0, 1.0, 1.0, 1 - eps_b, 1.0]        
    )    
    
    strikes_grid = market_data['strikes']
    tenors_vec = market_data['tenors']
    market_vols_grid = market_data['vols']
    num_strike_levels, num_tenors = market_vols_grid.shape

    def objective_function(params_to_opt):
        kappa, theta, sigma_opt, rho_opt, v0_opt = params_to_opt
        
        # Minimal check for obviously bad params (often handled by bounds)
        if sigma_opt <= 1e-4 or v0_opt <= 1e-6 or kappa <= 1e-4 or theta <= 1e-4: return 1e10

        total_squared_error = 0.0
        points_processed = 0

        for i in range(num_strike_levels):
            for j in range(num_tenors):
                K_val, T_val, market_iv_val = float(strikes_grid[i, j]), float(tenors_vec[j]), market_vols_grid[i, j]
                if np.isnan(market_iv_val): continue

                model_price = heston_call_price(S0_market, K_val, r_market, T_val,
                                                              kappa, theta, sigma_opt, rho_opt, v0_opt)
                
                model_iv_for_error = 0.0 # Default error target if IV calc fails
                if np.isnan(model_price) or model_price < 0:
                    model_iv_for_error = 3.0 # Penalize bad Heston prices
                else:
                    epsilon_iv = 1e-9
                    min_bs_price = max(0.0, S0_market - K_val * np.exp(-r_market * T_val)) + epsilon_iv
                    max_bs_price = S0_market - epsilon_iv
                    clamped_model_price = np.clip(model_price, min_bs_price, max_bs_price) if min_bs_price < max_bs_price else min_bs_price
                    
                    calculated_model_iv = calculate_implied_volatility(clamped_model_price, S0_market, K_val, r_market, T_val)
                    
                    if np.isnan(calculated_model_iv):
                        model_iv_for_error = 3.0 if clamped_model_price >= max_bs_price else 1e-4
                    else:
                        model_iv_for_error = calculated_model_iv
                
                total_squared_error += (model_iv_for_error - market_iv_val)**2
                points_processed += 1
        
        if points_processed == 0: return 1e15
        mse = total_squared_error / points_processed

        feller_penalty_val = 0.0
        if 2 * kappa * theta < sigma_opt**2:
            feller_penalty_val = (sigma_opt**2 - 2 * kappa * theta)**2 * 100
            
        return mse + feller_penalty_val

    initial_guess_clipped = np.clip(initial_guess_params, param_bounds.lb, param_bounds.ub)

    optim_result = minimize(objective_function, initial_guess_clipped, method='L-BFGS-B',
                            bounds=param_bounds, options={'disp':True, 'maxiter': 100, 'ftol': 1e-7, 'gtol': 1e-5, 'eps':1e-8})
    
    # Print MSE and RMSE
    mse = optim_result.fun
    rmse = np.sqrt(mse)
    print(f"\nCalibration Results:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    return optim_result.x, optim_result.fun, optim_result

def plot_volatility_surface(strikes_input, tenors_input, vols_surface, title_str, ax_plot=None):
    if ax_plot is None:
        fig = plt.figure(figsize=(10, 8))
        ax_plot = fig.add_subplot(111, projection='3d')
    
    if strikes_input.ndim == 2:
        unique_strikes = np.unique(strikes_input[:, 0]) 
    else: 
        unique_strikes = strikes_input
        
    if tenors_input.ndim == 2: 
        unique_tenors = np.unique(tenors_input[0, :])
    else: 
        unique_tenors = tenors_input

    X_mesh, Y_mesh = np.meshgrid(unique_strikes, unique_tenors)
    
    ax_plot.plot_surface(X_mesh, Y_mesh, vols_surface.T, cmap='viridis', rstride=1, cstride=1, alpha=0.9, linewidth=0.1, edgecolor='black')
    
    ax_plot.set_xlabel('Strike (K)', fontsize=14)
    ax_plot.set_ylabel('Maturity (T)', fontsize=14)
    ax_plot.set_zlabel('Implied Volatility', fontsize=14)
    ax_plot.set_title(title_str, fontsize=16)
    
    ax_plot.tick_params(axis='both', which='major', labelsize=12)
    
    return ax_plot

def analyze_and_plot(market_data, S0_market, r_market, calibrated_params_array, date_str):
    plt.rcParams.update({'font.size': 14})  # Increase default font size
    fig = plt.figure(figsize=(17, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    unique_strikes_for_plot = market_data['strikes'][:, 0] 
    unique_tenors_for_plot = market_data['tenors']

    plot_volatility_surface(
        unique_strikes_for_plot, 
        unique_tenors_for_plot, 
        market_data['vols'], 
        'Market Implied Volatility Surface', 
        ax1
    )
    
    kappa_cal, theta_cal, sigma_cal, rho_cal, v0_cal = calibrated_params_array
    
    num_strike_levels = market_data['strikes'].shape[0]
    num_tenors = len(market_data['tenors'])
    model_vols_surface = np.zeros((num_strike_levels, num_tenors))

    for i in range(num_strike_levels):
        for j in range(num_tenors):
            K_val = float(market_data['strikes'][i, j]) 
            T_val = float(market_data['tenors'][j])
            
            # Ensure heston_call_price is the correct, working function name
            model_price = heston_call_price(S0_market, K_val, r_market, T_val,
                                            kappa_cal, theta_cal, sigma_cal, rho_cal, v0_cal)
            
            if np.isnan(model_price) or model_price < 0:
                model_vols_surface[i, j] = np.nan 
            else:
                epsilon_iv = 1e-9
                min_bs_price = max(0.0, S0_market - K_val * np.exp(-r_market * T_val)) + epsilon_iv
                max_bs_price = S0_market - epsilon_iv
                
                clamped_model_price = model_price 
                if min_bs_price < max_bs_price: 
                    clamped_model_price = np.clip(model_price, min_bs_price, max_bs_price)
                
                # Ensure calculate_implied_volatility is the correct, working function name
                model_vols_surface[i, j] = calculate_implied_volatility(clamped_model_price, S0_market, K_val, r_market, T_val)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_volatility_surface(
        unique_strikes_for_plot, 
        unique_tenors_for_plot, 
        model_vols_surface, 
        'Calibrated Model Implied Volatility Surface', 
        ax2
    )
    
    fig.suptitle(f'Heston Calibration Results: {date_str}\n' + 
                 f'κ={kappa_cal:.2f}, θ={theta_cal:.3f}, σ={sigma_cal:.3f}, ρ={rho_cal:.3f}, v₀={v0_cal:.3f}',
                 fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.savefig(f'calibration_result_{date_str.replace(" ", "_").replace("/", "-")}.png', dpi=300)
    
    return model_vols_surface