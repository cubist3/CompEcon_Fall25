
###############################################################################
# SECTION 9: MAIN EXECUTION
###############################################################################
if __name__ == "__main__":
    print("Starting SMM Estimation (Robust Version)...")
    np.random.seed(123)
    # dim_shape: (Firms, Periods + Burn)
    SEED_A = np.random.rand(N_FIRMS, N_PERIODS + BURN_IN) 
    SEED_NOISE = np.random.rand(N_FIRMS, N_PERIODS + BURN_IN) # not used in VFI, but kept for structure
    
    # inital guess
    # [alpha, gamma, rho, sigma, phi_0]
    # Cooper_Ejarque approx values
    theta_init = np.array([0.65, 0.50, 0.85, 0.50, 0.001]) 
    
    # first stage (identity matrix)
    W_eye = np.eye(len(TARGET_MOMENTS))
    
    print("Stage 1: Minimization (Nelder-Mead)...")
    # Using Nelder-Mead because discrete grids make the objective function non-smooth
    res1 = opt.minimize(criterion, theta_init, 
                        args=(SEED_A, SEED_NOISE, W_eye),
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'disp': True})
    
    theta_1 = res1.x
    print(f"Stage 1 Estimates: {theta_1}")
    
    # optimal weighting matrix
    print("Calculating Weighting Matrix...")
    n_sims = 20 
    moments_store = np.zeros((n_sims, len(TARGET_MOMENTS)))
    
    model_fixed = solve_model(*theta_1)
    
    for s in range(n_sims):
        s_A = np.random.rand(N_FIRMS, N_PERIODS + BURN_IN)
        s_N = np.random.rand(N_FIRMS, N_PERIODS + BURN_IN)
        
        # just re-simulate, not solving again the model (assuming theta_1 is true param)
        # SMM variance includes parameter uncertainty, but 
        # for weighting matrix W, we estimate V_moments(theta_1).
        d = simulate_panel(model_fixed, s_A, s_N)
        moments_store[s, :] = get_moments(d)
        
    # COV of moments
    Omega = np.cov(moments_store.T)
    
    # Pseudo-Inverse to handle near-singularities
    W_opt = lin.pinv(Omega)
    
    # second stage
    print("Stage 2: Minimization (Nelder-Mead)...")
    res2 = opt.minimize(criterion, theta_1,
                        args=(SEED_A, SEED_NOISE, W_opt),
                        method='Nelder-Mead',
                        options={'maxiter': 300, 'disp': True})
    
    theta_final = res2.x
    print(f"Final Estimates: {theta_final}")
    
    # SEs
    print("Calculating Standard Errors...")
    
    # numerical Jacobian
    h = 1e-4
    J = np.zeros((len(TARGET_MOMENTS), len(theta_final)))
    
    # central difference
    # must solve model inside here
    orig_moments = get_moments(simulate_panel(solve_model(*theta_final), SEED_A, SEED_NOISE))
    
    for i in range(len(theta_final)):
        t_plus = theta_final.copy(); t_plus[i] += h
        t_minus = theta_final.copy(); t_minus[i] -= h
        
        try:
            m_plus = get_moments(simulate_panel(solve_model(*t_plus), SEED_A, SEED_NOISE))
            m_minus = get_moments(simulate_panel(solve_model(*t_minus), SEED_A, SEED_NOISE))
            J[:, i] = (m_plus - m_minus) / (2*h)
        except:
            J[:, i] = 0.0
            
    # SMM Variance Formula: V = (1 + 1/M) (J' W J)^-1
    
    inv_term = lin.pinv(J.T @ W_opt @ J)
    V_theta = (1 + 1/n_sims) * inv_term
    SE = np.sqrt(np.diag(V_theta))

    ###############################################################################
    # printing the output
    ###############################################################################
    
    print("\n" + "="*40)
    print("REPLICATION RESULTS (Table 3)")
    print("="*40)
    names = ['alpha', 'gamma', 'rho', 'sigma', 'phi_0']
    print(f"{'Param':<10} {'Est':<10} {'SE':<10}")
    for i in range(5):
        print(f"{names[i]:<10} {theta_final[i]:.4f}     {SE[i]:.4f}")
        
    print("-" * 40)
    final_mom = get_moments(simulate_panel(solve_model(*theta_final), SEED_A, SEED_NOISE))
    print(f"{'Moment':<10} {'Target':<10} {'Model':<10}")
    for i in range(6):
        print(f"{MOMENT_NAMES[i]:<10} {TARGET_MOMENTS[i]:.4f}     {final_mom[i]:.4f}")
    
    J_stat = res2.fun 
    print(f"\nJ-Stat: {J_stat:.4f}")
    
    # Save for LaTeX
    np.savez('smm_results.npz', params=theta_final, se=SE, moments=final_mom)