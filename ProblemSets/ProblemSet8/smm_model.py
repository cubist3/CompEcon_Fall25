import numpy as np
import numpy.linalg as lin
import scipy.optimize as opt
from scipy.stats import norm
import time

###############################################################################
# 1. params
###############################################################################
DELTA = 0.15 # depreciation
BETA = 0.95  # discount
PHI_1 = 0.0  # lin. cost
P = 1.0  # price of capital

###############################################################################
# 2. target moments set up (Cooper & Ejarque 2003, Table 3)
###############################################################################
# [a1, a2, sc(I/K), std(pi/K), mean(Q), Ext.Frac]
TARGET_MOMENTS = np.array([0.03, 0.24, 0.40, 0.25, 3.00, 0.25])
MOMENT_NAMES = ['a1', 'a2', 'sc(I/K)', 'std(π/K)', 'mean(Q)', 'Ext.frac']

###############################################################################
# 3. computational params
###############################################################################
# explaination of those below is in the latex file, pdf (the section 4)
N_A = 7
N_K = 60           
K_MIN = 1.0
K_MAX = 350.0

# put some noises
N_FIRMS = 500      
N_PERIODS = 30     
BURN_IN = 10       
VFI_TOL = 1e-4
VFI_MAX_ITER = 500

###############################################################################
# 4. Tauchen method
###############################################################################
def tauchen(n_states, rho, sigma, m=3):
    sigma_z = sigma / np.sqrt(1 - rho**2)
    z_grid = np.linspace(-m*sigma_z, m*sigma_z, n_states)
    d = z_grid[1] - z_grid[0]
    
    trans_matrix = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            up = z_grid[j] + d/2 - rho * z_grid[i]
            lo = z_grid[j] - d/2 - rho * z_grid[i]
            if j == 0:
                trans_matrix[i, j] = norm.cdf(up / sigma)
            elif j == n_states - 1:
                trans_matrix[i, j] = 1 - norm.cdf(lo / sigma)
            else:
                trans_matrix[i, j] = norm.cdf(up / sigma) - norm.cdf(lo / sigma)
                
    trans_matrix = trans_matrix / trans_matrix.sum(axis=1, keepdims=True)
    A_grid = np.exp(z_grid)
    return z_grid, A_grid, trans_matrix

###############################################################################
# 5. VFI
###############################################################################
def solve_model(alpha, gamma, rho, sigma, phi_0, verbose=False):
    # grids
    K_grid = np.linspace(K_MIN, K_MAX, N_K)
    z_grid, A_grid, trans_matrix = tauchen(N_A, rho, sigma)
    
    # pre-compute profits: pi = A * K^alpha
    # dim shape: (N_K, N_A)
    profit_matrix = A_grid[None, :] * (K_grid[:, None] ** alpha)
    
    # cost of external finance (FC)
    # the paper scales this by mean capital, but for VFI we use a level
    # approx. mean_K roughly as median of grid to keep VFI static
    phi_cost = phi_0 * np.mean(K_grid) 

    # VFI (value function iteration)
    V = profit_matrix / (1 - BETA) # initial guess
    
    # pre-calculate adjustment costs and Inv for all (K, K') pairs
    # vectorisation - computational speed
    # dim shape: (N_K_current, N_K_next)
    I_mat = K_grid[None, :] - (1 - DELTA) * K_grid[:, None]
    Inv_rate = I_mat / K_grid[:, None]
    Adj_cost = (gamma / 2.0) * (Inv_rate**2) * K_grid[:, None]
    
    # handle the external finance constraint 
    # some of vectorisation
    
    for it in range(VFI_MAX_ITER):
        V_new = np.zeros_like(V)
        
        # exp.value: (N_K, N_A)
        EV = V @ trans_matrix.T 
        
        for i_a in range(N_A):
            pi_col = profit_matrix[:, i_a]
            max_internal = pi_col + (1 - DELTA) * K_grid
            
            # EV_col is value of landing in K' tomorrow given current A
            EV_col = EV[:, i_a] 
            
            # total return matrix for this specific A: (N_K_current, N_K_next)
            # Flow utility = Profit - Investment - AdjCost - (ExternalCost if needed)
            
            # base flow:
            Flow = pi_col[:, None] - I_mat - Adj_cost
            
            # external finance constraint: K' > internal_Funds
            # K_grid[None, :] is K', max_internal[:, None] is current funds
            Is_External = K_grid[None, :] > max_internal[:, None]
            
            # aply cost
            Flow[Is_External] -= phi_cost
            
            # total Value
            Tv = Flow + BETA * EV_col[None, :]
            
            # max over K' (columns)
            V_new[:, i_a] = np.max(Tv, axis=1)
            
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < VFI_TOL:
            break
            
    # policy function
    policy_K_idx = np.zeros((N_K, N_A), dtype=int)
    
    EV = V @ trans_matrix.T
    for i_a in range(N_A):
        pi_col = profit_matrix[:, i_a]
        max_internal = pi_col + (1 - DELTA) * K_grid
        Flow = pi_col[:, None] - I_mat - Adj_cost
        Is_External = K_grid[None, :] > max_internal[:, None]
        Flow[Is_External] -= phi_cost
        Tv = Flow + BETA * EV[:, i_a][None, :]
        policy_K_idx[:, i_a] = np.argmax(Tv, axis=1)
        
    return {
        'V': V, 'pol_idx': policy_K_idx, 
        'K_grid': K_grid, 'A_grid': A_grid, 
        'trans': trans_matrix, 'alpha': alpha
    }

###############################################################################
# 6. simulation
###############################################################################
def simulate_panel(model, seed_A, seed_noise):
    pol_idx = model['pol_idx']
    K_grid = model['K_grid']
    A_grid = model['A_grid']
    trans_cdf = np.cumsum(model['trans'], axis=1)
    
    n_firms, total_periods = seed_A.shape
    
    K_idx = np.zeros((n_firms, total_periods), dtype=int)
    A_idx = np.zeros((n_firms, total_periods), dtype=int)
    
    # initialize at steady state approx
    K_idx[:, 0] = N_K // 3
    A_idx[:, 0] = N_A // 2
    
    for t in range(total_periods - 1):
        # update A
        for i in range(n_firms):
            # (1) update A
            a_prev = A_idx[i, t]
            rand_a = seed_A[i, t+1]
            # Fast way to find transition
            a_next = np.searchsorted(trans_cdf[a_prev], rand_a)
            A_idx[i, t+1] = min(a_next, N_A-1)
            
            # (2) update K
            k_prev = K_idx[i, t]
            k_next_idx = pol_idx[k_prev, a_prev]
            K_idx[i, t+1] = k_next_idx

    # drop Burn-in
    K_idx = K_idx[:, BURN_IN:]
    A_idx = A_idx[:, BURN_IN:]
    
    K_val = K_grid[K_idx]
    A_val = A_grid[A_idx]
    
    # calculate flows
    # I = K' - (1-d)K
    I_val = K_val[:, 1:] - (1 - DELTA) * K_val[:, :-1]
    
    # align K and A for t (drop last period of K/A for flows)
    K_curr = K_val[:, :-1]
    A_curr = A_val[:, :-1]
    
    Profits = A_curr * (K_curr ** model['alpha'])
    Inv_Rate = I_val / K_curr
    Profit_Rate = Profits / K_curr
    
    # external finance check
    # need K_prime to compare against internal funds
    K_next = K_val[:, 1:]
    Max_Internal = Profits + (1-DELTA)*K_curr
    External = K_next > Max_Internal
    
    # AVG Q = V / K
    # calculate V at current state
    V = model['V']
    Avg_Q = np.zeros_like(K_curr)
    for i in range(n_firms):
        for t in range(K_curr.shape[1]):
            Avg_Q[i,t] = V[K_idx[i,t], A_idx[i,t]] / K_curr[i,t]
            
    return {
        'Inv_Rate': Inv_Rate, 'Profit_Rate': Profit_Rate, 
        'Q': Avg_Q, 'I': I_val, 'External': External
    }

###############################################################################
# 7. moments
###############################################################################
def get_moments(sim_data):
    ik = sim_data['Inv_Rate']
    pk = sim_data['Profit_Rate']
    q = sim_data['Q']
    
    # (1) reg I/K = a_i + a1*Q + a2*(Pi/K)
    # FE, fixed effects -> demean variables
    ik_demean = ik - np.mean(ik, axis=1, keepdims=True)
    pk_demean = pk - np.mean(pk, axis=1, keepdims=True)
    q_demean = q - np.mean(q, axis=1, keepdims=True)
    
    Y = ik_demean.flatten()
    X = np.column_stack([q_demean.flatten(), pk_demean.flatten()])
    
    try:
        beta = lin.lstsq(X, Y, rcond=None)[0]
        a1, a2 = beta[0], beta[1]
    except:
        a1, a2 = 0, 0
        
    # (2) serial corr. of I/K
    sc = 0.0
    valid_firms = 0
    for i in range(ik.shape[0]):
        # simple correlation of series with itself lagged
        c = np.corrcoef(ik[i, 1:], ik[i, :-1])[0,1]
        if np.isfinite(c):
            sc += c
            valid_firms += 1
    sc = sc / max(valid_firms, 1)
    
    # (3) stdev of profits
    std_pi = np.std(pk)
    
    # (4) mean Q
    mean_q = np.mean(q)
    
    # (5) external Fraction (Sum of external I / Total I)
    # filter for positive investment to match literature conventions often used
    I = sim_data['I']
    Ext = sim_data['External']
    
    mask = I > 0
    tot_inv = np.sum(I[mask])
    ext_inv = np.sum(I[mask & Ext])
    
    ext_frac = ext_inv / tot_inv if tot_inv > 0 else 0.0
    
    return np.array([a1, a2, sc, std_pi, mean_q, ext_frac])

# =============================================================================
# 8. obj function
# =============================================================================
def criterion(params, seed_A, seed_noise, W, simple_diff=False):
    alpha, gamma, rho, sigma, phi0 = params
    
    # Hard Bounds
    if alpha < 0.2 or alpha > 0.95: return 1e10
    if gamma < 0.0: return 1e10
    if rho < 0.0 or rho > 0.99: return 1e10
    if sigma < 0.01 or sigma > 2.0: return 1e10
    if phi0 < 0.0: return 1e10
    
    try:
        model = solve_model(alpha, gamma, rho, sigma, phi0)
        data = simulate_panel(model, seed_A, seed_noise)
        sim_moments = get_moments(data)
        
        diff = TARGET_MOMENTS - sim_moments
        
        # Percentage difference for better scaling?...
        if simple_diff:
            loss = np.sum(diff**2)
        else:
            loss = diff.T @ W @ diff
            
        return loss
    except Exception as e:
        return 1e10
