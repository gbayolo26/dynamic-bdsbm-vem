 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:26:04 2023

@author: gabriela
"""

import numpy as np

def f_compute_lambda_mu(df, N_l, tau, b):
    """
    This function computes the maximum-likelihood estimates of the birth rate
    `lambda` and death rate `mu` for a continuous-time birth–death process.
    Also the log-likelihood contribution of the birth–death process evaluated at
        (`lambda_est`, `mu_est`).
    """
   
    integral_N = N_l[1:,].dot(np.diff(tau))
    num_births = np.sum(b == 1) 
    num_deaths = np.sum(b == -1) 
    
    lambda_est = num_births/integral_N
    mu_est = num_deaths/integral_N
    
    if lambda_est == 0:
       log_lambda = 0
       
    else :
       log_lambda = np.log(lambda_est)
       
    if mu_est == 0:
       log_mu = 0   
       
    else :
       log_mu = np.log(mu_est)
          
    term2 = num_births*log_lambda  + num_deaths*log_mu - (lambda_est + mu_est)*integral_N
    
    return lambda_est, mu_est, term2

def poisson_binomial_pmf(p):
    """
    Compute the probability mass function (PMF) of the Poisson Binomial distribution.
    Uses the recursive formula for exact computation.
    """
    n = len(p)
    pmf = np.zeros(n + 1)
    pmf[0] = 1  # P(X=0)
    
    for pi in p:
        pmf[1:] = pmf[1:] * (1 - pi) + pmf[:-1] * pi
        pmf[0] *= (1 - pi)
    
    return pmf

def compute_delta_gamma_init(df, K, delta_0, t0=0.0, smooth=1e-9):
    """
    Initialize community memebership probabilities for all individuals.
    
    - Individuals alive at t0: given by delta_0 (shape: N0 x K).
    - Newborns at time t: delta_i,k = sum_{alive before t} delta_j,k / (#alive before t),
      with a small smoothing to avoid exact zeros.

    Assumption: individuals present at t0 correspond to indices 0..N0-1.
    """
    tb = df["t_birth"].to_numpy(float)
    td = df["t_death"].to_numpy(float)
    N = len(df)

    N0 = delta_0.shape[0]
    delta = np.zeros((N, K), dtype=float)
    delta[:N0] = delta_0

    # alive set at t0
    alive = np.zeros(N, dtype=bool)
    alive[:N0] = True

    # expected alive mass per community among alive individuals
    n_k = delta[:N0].sum(axis=0)

    # process birth times > t0 in chronological order (grouped)
    birth_times = np.unique(tb[tb > t0])
    birth_times.sort()

    for t in birth_times:
        # remove those who died before (or at) time t
        dying = alive & (td <= t)
        if np.any(dying):
            n_k -= delta[dying].sum(axis=0)
            alive[dying] = False

        newborn = np.where(tb == t)[0]
        if newborn.size == 0:
            continue

        total_alive = n_k.sum()
        # p_k = (sum alive delta_k) / (#alive), smoothing to avoid zeros
        p = (n_k + smooth) / (total_alive + K * smooth)

        delta[newborn] = p
        alive[newborn] = True
        n_k += newborn.size * p

    return delta


def compute_delta_gamma_init_old(df, K, N_l, tau, b, delta_0, gamma_mar_0):
    """
    Initialize community memebership probabilities for all individuals.

    The first N0 rows are set to `delta_0`. The remaining (N - N0) rows are drawn
    at random and row-normalized to form valid probability vectors.
    """
    rng = np.random.default_rng(seed = 1)    
    N = len(df) 
    n0 = delta_0.shape[0]
    rest = rng.random((N - n0, K))
    rest /= rest.sum(axis=1, keepdims=True)
    new_delta = np.vstack([delta_0, rest])
    
    return new_delta

def H_q(S1, S, df, beta, delta, pi, gamma, gamma_mar, b, term2):
    """
    Compute the ELBO.

    - term1: expected SBM log-likelihood using interaction counts (S1) and exposure (S)
    - term2: birth–death log-likelihood contribution (passed as input)
    - term3: birth-related combinatorial/transition term involving gamma and gamma_mar
    - term4: prior/mixing term at time 0 using beta (only for nodes alive at t=0)
    - term5: entropy term of delta for nodes alive at t=0
    - term6: entropy term of gamma 

    Returns
    -------
    H_q_value : float
        Value of H(q).
    term1_icl_soft : float
        Soft-ICL-like term.
    """
    
    V_t0 = list(df[df['t_birth']<=0]['id'])
    times = np.array( list(range(len(b))))
    times_births = times[b== 1]
    log_n = np.log(np.arange(1, len(df)+1))
    
    # first term: sum_{i<j} sum_{k1, k2} delta(i, k1) delta(j, k2) sum_{l in Delta_ij} log(phi(e^l_ij, pi_k1k2))        
    term1 = 0.5 * np.sum(delta.dot(np.log((pi/(1-pi)))).dot(delta.transpose())*S1 + delta.dot(np.log(1-pi)).dot(delta.transpose())*S)
        
    # third term: sum_{l in T_B} sum_k sum_{n=1}^{N_l} gamma(l, k, n, n+1) gamma_mar(l-1, k, n) log(n)
    term3 = np.sum((1-gamma[times_births, : , 1:])*gamma_mar[times_births, : , 1:]*log_n[None, None, :])
    
    # fourth term: sum_{i in V_{t0}} sum_k delta(i, k) log(beta_k)
    term4 = np.sum(delta[V_t0].dot(np.log(beta)))
    
    #fifth term: sum_{i in V_{t0}} sum_k delta(i, k) log(delta(i, k))    
    d0 = delta[V_t0]
    d0_safe = np.where(d0 == 0, 1.0, d0)   # 0 -> 1 so that log(1)=0
    term5 = np.sum(d0 * np.log(d0_safe))
    
    #sixth term: sum_{l=1}^M sum_k sum_{n=1}^{N_l} gamma(l, k, n, n) gamma_mar(l-1, k, n) log(gamma(l, k, n, n))    
    safe_log_1 = np.zeros_like(gamma)
    mask = gamma > 0
    safe_log_1[mask] = np.log(gamma[mask])
    new_gamma_2 = 1 - gamma
    safe_log_2 = np.zeros_like(new_gamma_2)
    mask = new_gamma_2 > 0
    safe_log_2[mask] = np.log(new_gamma_2[mask])
    A = gamma*gamma_mar*safe_log_1+new_gamma_2*gamma_mar*safe_log_2
    #A[np.isnan(A)] = 0
    term6 =  np.sum(A)
    
    H_q_value = term1 + term2 + term3 + term4 - term5 - term6
    
    term1_icl_soft = term1 + term3 + term4
    
    return H_q_value, term1_icl_soft 

def ICL(S1, S, df, b, delta, pi, gamma_mar, pen):
    """
    Compute the Integrated Completed Likelihood (ICL) criterion.
    """
    idx = np.argmax(delta, axis=1)
    Z = np.zeros_like(delta, dtype=int)
    Z[np.arange(delta.shape[0]), idx] = 1                  
    num_pi = Z.T.dot(S1).dot(Z)  
    den_pi = Z.T.dot(S).dot(Z)                
    pi = num_pi/den_pi
    pi[np.isnan(pi)] = 0
    #pi = np.clip(pi, eps, 1 - eps)            
    beta = Z.mean(axis = 0)  
    #beta = np.clip(beta, eps, 1 - eps)                     
    N = len(df)
    times = np.array( list(range(len(b))))
    times_births = times[b== 1]
    
    # first term: 
    term1 = 0.5 * np.sum(Z.dot(np.log(pi/(1-pi))).dot(Z.transpose())*S1 + Z.dot(np.log(1-pi)).dot(Z.transpose())*S)
    
    # second term:
    k_idx = np.argmax(gamma_mar, axis=1)
    L = np.zeros_like(gamma_mar, dtype=np.uint8)
    t_idx = np.arange(gamma_mar.shape[0])[:, None]       # shape (T,1)
    n_idx = np.arange(gamma_mar.shape[2])[None, :]       # shape (1,N)
    t = times_births[0:-1]
    L[t_idx, k_idx, n_idx] = 1    # coloca 1 en (t, k*, n)
    term2 = np.sum((L[t + 1, : , 2:N])*L[t, : , 1:N-1]*np.log(np.arange(N - 2 )+1))
    
    # third term
    term3 = np.sum(Z.dot(np.log(beta)))
   
    ICL_value = term1 + term2 + term3 - pen
    
    return ICL_value

def ICL_effective(S1, S, df, b, times_births, M, N0, delta, pi, gamma_mar):
    """
    Compute the Integrated Completed Likelihood (ICL) criterion.
    """
    
    eps = 1e-30
    N_nodes, K = delta.shape
    
    idx = np.argmax(delta, axis=1)       # MAP per individual
    Z_full = np.zeros_like(delta, dtype=int)
    Z_full[np.arange(N_nodes), idx] = 1  # Z_full: (N, K)

    # number of individuals by class
    n_k_full = Z_full.sum(axis=0)        # (K,)

    # non empty class
    mask = n_k_full > 0
    K_effective = int(mask.sum())        

    # Z for non empty class
    Z = Z_full[:, mask]     # (N, K_eff)

    num_pi = Z.T @ S1 @ Z   
    den_pi = Z.T @ S  @ Z  
       
    pi_hat = np.divide(num_pi,
                   den_pi,
                   out=np.zeros_like(num_pi, dtype=float),
                   where=den_pi > 0)

    # to avoid 0 and 1
    pi_hat = np.clip(pi_hat, eps, 1.0 - eps)
        
    n_k = Z.sum(axis=0)               
    beta = n_k / float(N_nodes)       
    
    logit_pi = np.log(pi_hat / (1.0 - pi_hat))
    log1m_pi = np.log(1.0 - pi_hat)
    
    # first term: 
    term1 = 0.5 * np.sum( (Z @ logit_pi @ Z.T) * S1 + (Z @ log1m_pi @ Z.T) *S)
            
    # second term:
    k_idx = np.argmax(gamma_mar, axis=1)
    L = np.zeros_like(gamma_mar, dtype=np.uint8)
    t_idx = np.arange(gamma_mar.shape[0])[:, None]       # shape (T,1)
    n_idx = np.arange(gamma_mar.shape[2])[None, :]       # shape (1,N)
    t = times_births[0:-1]
    L[t_idx, k_idx, n_idx] = 1    # coloca 1 en (t, k*, n)
    term2 = np.sum((L[t + 1, : , 2:N_nodes])*L[t, : , 1:N_nodes-1]*np.log(np.arange(N_nodes - 2 )+1))
    
    # third term
    term3 = np.sum(Z @ np.log(beta))  
   
    pen = 0.25 * K_effective * (K_effective + 1) * np.log(max(M/2, 1.0)) + 0.5*(K_effective - 1) * np.log(N0)

    ICL_value = term1 + term2 + term3 - pen
    
    return ICL_value

