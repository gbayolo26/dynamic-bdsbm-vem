#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 18:00:46 2025

@author: gabriela
"""

import numpy as np
from scipy.optimize import fsolve
from .utils import f_compute_lambda_mu, compute_delta_gamma_init, H_q, ICL, ICL_effective, poisson_binomial_pmf

class DynamicBDSBM_VEM:
    
    """
    Variational EM (VEM) inference for a dynamic Birth–Death Stochastic Block Model (BD-SBM).

    This class implements a variational EM procedure to infer latent community
    memberships and model parameters from dynamic interaction data where the
    node set evolves over time according to a birth–death process.

     The model is observed through:
    - `interaction_counts` (S1): a (N × N) interaction-count matrix where
      S1[i, j] is the total number of observed interactions between individuals i and j
      over the whole study interval.
    - `alive_matrix` (S): a (N × N) exposure / co-aliveness matrix where
      S[i, j] is the number of observation times at which individuals i and j are
      simultaneously alive/active (i.e., the number of opportunities for them to interact).
    - `df_births_deaths`: individual birth and death times defining the evolving node set.

    The VEM algorithm estimates:
    - birth–death parameters (lambda, mu),
    - dynamic SBM parameters (pi, beta),    
    - variational distribution over communities memberships (delta) and auxiliary probabilities
    (gamma, gamma_mar), and monitors convergence using the ELBO.
    """
    
    def __init__(self, interaction_counts, alive_matrix, df_births_deaths, delta_0, max_iters, tol):
        
        """
        Initialize the dynamic BD-SBM VEM estimator.

        Parameters
        ----------
        interaction_counts :  array-like, shape (N, N)
            Pairwise interaction counts over the whole study interval. Entry (i, j) is the 
            total number of observed interactions between individuals i and j.
        alive_matrix : array-like, shape (N, N)
            Pairwise co-aliveness (exposure) matrix. Entry (i, j) is the number of
            observation times at which individuals i and j are simultaneously alive/active
            (i.e., the number of opportunities for them to interact).
        df_births_deaths : pandas.DataFrame
            Per-individual information including at least columns:
            - 't_birth': birth time of the individual
            - 't_death': death time of the individual
        delta_0 : array-like, shape (N0, K) or shape(N, K)
            Initial variational distribution over communities memberships.
        max_iters : int
            Maximum number of VEM iterations.
        tol : float
            Convergence tolerance for the stopping criterion.
        """
        
        # Number of communities
        self.K = delta_0.shape[1] 

        # Birth/death dataframe        
        self.df = df_births_deaths  

        # Sorted births and deaths times     
        self.tau = np.sort(np.unique(np.concatenate((self.df['t_birth'], self.df['t_death']))))[:-1]     
        
        # Event-type vector along `tau`: +1 birth, -1 death
        b = np.full(len(self.tau), -1, dtype=int)
        birth_times = self.df["t_birth"].to_numpy(dtype=float)
        mask = np.isin(self.tau, birth_times)
        b[mask] = 1
        b = b[1:]
        self.b = b                              
        
        # Number of individuals present at time 0 (t_birth <= 0)
        self.N0 = len(self.df[self.df['t_birth']<=0]) 
        
        # Total number of individuals in the dataset
        self.N = len(df_births_deaths)    

        # Number of alive individuals after each event time (aligned with `b`)            
        self.N_l = self.N0 + np.cumsum(np.concatenate(([0], b))) 
        
        self.delta_0 = delta_0        
        self.max_iters = max_iters
        self.tol = tol          
        self.S1 = interaction_counts
        self.S = alive_matrix     
        self.M = np.sum(self.S)
        
        # Indices of event times (aligned with b) 
        times = np.array( list(range(len(self.b))))      
        
        # Indices corresponding to birth events
        self.times_births = times[self.b== 1]
        
        # Pre-computed penalty term used in model selection 
        self.pen = 0.5*(0.5 * self.K * (self.K+1) * np.log(max(self.M/2, 1.0)) + (self.K - 1) * np.log(self.N0))
                     
        
    def expectation_step(self, delta, pi, beta):
        """
        Perform the VEM E-step for the dynamic BD-SBM.
        
        This E-step updates the variational distributions given the current model
        parameters. It (i) updates the node probabilities `delta`, and (ii) propagates 
        variational quantities (`gamma`, `gamma_mar`) along the birth–death event 
        timeline to account for the varying number of alive individuals per community.
    
        Parameters
        ----------
        delta : array-like, shape (N, K)
            Current community membership probabilities matrix.
            Row i corresponds to individual i and sums to 1 across communities.
        pi : array-like, shape (K, K)
            Current SBM connection probability matrix.
        beta : array-like, shape (K,) 
            Current vector of community probabilities at time t_0.
    
        Returns
        -------
        new_delta : array-like, shape (N, K)
            Updated of community membership probabilities matrix.
        gamma : array-like, shape (T, K, N+1)
            Variational transition probabilities of the the community-sizes over the 
            event timeline. For each event index l, community k, and count n,
            `gamma[l, k, n]` denotes the transition probability to move from n to n 
            individuals in community k at time tau_l+1. 
            Implementation note: the original quantity can be viewed as γ(l, k, n, n),
            which is stored here as a 3D tensor for convenience. The indexing is shifted:
            since there is no γ at the initial time, `gamma[l]` corresponds to the update
            from event time τ_l to τ_{l+1}. The first column (n = 0) is fixed to 1 for all
            l and k.
        gamma_mar : array-like, shape (T, K, N+1)
            Marginal distributions of community sizes over the event timeline.
            `gamma_mar[l, k, n]` is the probability that, at event index l, there are n
            alive individuals in community k. It is initialized at time 0 using a
            Poisson–binomial pmf and then updated recursively at each birth/death event.
        """
    
        V = list(range(self.N + 1))
        V_t0 = list(range(self.N0))
        zeros = np.zeros((self.K, 1))
        
        #to upgrade delta for i in V_t_0
        A = (np.log((pi/(1-pi)))).dot(delta.transpose()).dot(self.S1) +(np.log(1-pi)).dot(delta.transpose()).dot(self.S)
        At = A.transpose()
        maxi = np.max(At, axis=1)      
        #maxi = np.mean(At, axis=1)      
        res = At - maxi[:, np.newaxis]        
        new_delta = np.exp(res)*beta
        new_delta = new_delta.transpose()*(1/ np.sum(new_delta, axis = 1)) # Normalize 
        new_delta = new_delta.transpose()
        new_delta_0 = new_delta[V_t0]                 
        
        vector_log = np.log(np.arange(1, self.N))  # Log of the numbers from 1 to N
       
        At_ = res[len(V_t0):, :]
        B = At_[:, :, np.newaxis] + vector_log    
        new_B = np.exp(B)
        
        # to initialize gamma_mar
        gamma_mar = np.zeros((len(self.b), self.K, self.N+1))   
        
        # to compute gamma_mar at time 0
        for k in range(self.K):            
            gamma_mar[0][k][V_t0 + [len(V_t0)]] = poisson_binomial_pmf(new_delta_0[:, k])
        
        # to initialize gamma
        gamma = np.zeros((len(self.b), self.K, self.N))
        ones_column = np.ones((gamma.shape[0], gamma.shape[1], 1))
        gamma = np.concatenate((ones_column, gamma), axis=2)
    
        idx = 0
        zeros = np.zeros((self.K, 1))
        error = 1e-10  
          
        #to upgrade gamma, gamma_mar, delta for each tau_l
        for l in range(len(self.b)-1):
           
            v = 1 - V / self.N_l[l]
            v[v<0] = 0    
           
            if l in self.times_births:
                
                alpha0 = np.sqrt(1/np.max(new_B[idx]))
                
                if alpha0 == 0:
                   alpha0 = 1e-16
                
                def system(alpha):
                   
                    gamma[l, :, 1:self.N_l[l]] = 1 / (1 + ((alpha)**2)*new_B[idx, :, 0:self.N_l[l]-1])
                    eq = np.sum(gamma[l] * gamma_mar[l]) - self.K + 1 
            
                    return eq
                
                alpha1 = fsolve(system, alpha0)      
                residual = system(alpha1)
                
                if abs(residual) > error:
                   
                   multipliers=(0.1, 10, 100, 1000)
                   best_alpha = alpha1
                   best_residual = residual
                    
                   for m in multipliers:
                      
                      new_alpha0 = alpha0 * m
                      new_alpha = fsolve(system, new_alpha0)      
                      residual = system(new_alpha)
                      
                      if np.isfinite(new_alpha) and np.isfinite(residual) and abs(residual) < abs(best_residual):
                         best_alpha, best_residual = new_alpha, residual
                   
                   # final sync : make gamma consistent with the chosen alpha
                   gamma[l, :, 1:self.N_l[l]] = 1 / (1 + ((best_alpha)**2)*new_B[idx, :, 0:self.N_l[l]-1])                  
                
                idx = idx + 1
        
                i = self.df[self.df['t_birth'] == self.tau[l+1]]['id']
                new_delta[i] = np.sum((1- gamma[l])*gamma_mar[l], axis=1)
                        
                if l< len(self.b)-1:     
                   m1 = gamma[l]*gamma_mar[l]
                   m2 = np.hstack((zeros, (1-gamma[l])*gamma_mar[l],zeros))    
                   n = m1 + m2[:, int((1-self.b[l])):self.N + 1 + int((1-self.b[l]))]                
                   gamma_mar[l + 1] = n
                    
            else: 
                        
                i = self.df[self.df['t_death'] == self.tau[l+1]]['id']                               
                                
                r = np.arange(1, self.N_l[l])
                               
                for k in range(self.K):
                    
                    if new_delta[i,k] == 0:
                        
                        gamma[l, k, 1:self.N_l[l]] = 1
                                            
                    else :
                       
                        alpha0 = 1
                                                
                        def system(alpha):
                   
                            gamma[l, k, 1:self.N_l[l]] = 1 / (1 + ((alpha)**2)*r)
                            eq = np.sum(gamma[l,k,:] * gamma_mar[l,k,:]) - 1 + new_delta[i,k] 
            
                            return eq

                        alpha1 = fsolve(system, alpha0)                   
                        residual = system(alpha1)
                        
                        if abs(residual) > error:
                           
                           multipliers=(0.1, 10, 100, 1000, 10000, 100000)
                           best_alpha = alpha1
                           best_residual = residual
                           
                           for m in multipliers:
                              
                              new_alpha0 = alpha0 * m
                              new_alpha = fsolve(system, new_alpha0)      
                              residual = system(new_alpha)
                              
                              if np.isfinite(new_alpha) and np.isfinite(residual) and abs(residual) < abs(best_residual):
                                 best_alpha, best_residual = new_alpha, residual
                          
                           # final sync : make gamma consistent with the chosen alpha
                           gamma[l, k, 1:self.N_l[l]] = 1 / (1 + ((best_alpha)**2)*r)        
                                        
                if l< len(self.b)-1:  
                                        
                    m1 = gamma[l]*gamma_mar[l]
                    m2 = np.hstack((zeros, (1-gamma[l])*gamma_mar[l],zeros))    
                    n = m1 + m2[:, int((1-self.b[l])):self.N + 1 + int((1-self.b[l]))]                
                    gamma_mar[l + 1] = n
                               
                    if np.any(gamma_mar[l+1] < 0):
                       A = np.asarray(gamma_mar[l+1], dtype=float)
                       mask = A < 0 
                       masked = np.where(mask, A, -np.inf)
                       idx_flat = np.argmax(masked)              
                       pos = np.unravel_index(idx_flat, A.shape) 
                       val = A[pos]
                       
                       if abs(val) < error:
                          gamma_mar[l+1] = np.maximum(gamma_mar[l+1], 0, out=gamma_mar[l+1])       
                                   
        return  new_delta, gamma, gamma_mar  
            
    def maximization_step(self, delta):
        
        """
        Perform the VEM M-step for the BD-SBM.
        """                
        eps=1e-20
        
        num_pi = delta.T @ self.S1 @ delta
        den_pi = delta.T @ self.S  @ delta
        pi = num_pi / den_pi
    
        if np.any(den_pi == 0):
            
            print("Warning: den_pi has zeros in", np.argwhere(den_pi == 0).tolist())
            pi_new = pi.copy()  # fallback: keep previous pi
            mask = den_pi > 0
            pi_new[mask] = num_pi[mask] / den_pi[mask]
        
            # stability for logs later
            pi_new = np.clip(pi_new, eps, 1 - eps)
            
        else :
            pi_new = pi
            
        beta = delta[0:self.N0].mean(axis = 0)
        
        return  pi_new, beta
        
    def fit(self ):
        """
        Fit the model using Variational Expectation-Maximization (VEM).
        """
        lambda_est, mu_est, term2 = f_compute_lambda_mu(self.df, self.N_l, self.tau, self.b)
              
        # Initialization of delta
        if self.delta_0.shape[0] == self.N0:    
            delta = compute_delta_gamma_init(self.df, self.K, self.delta_0)
                                    
        elif self.delta_0.shape[0] == self.N: 
            delta = self.delta_0
            self.delta_0 = self.delta_0[0:self.N0]
                   
        # Initialization of pi and beta 
        pi, beta = self.maximization_step( delta)
        
        elbo_values = []
        icl_soft_values = []
        
        for i in range(self.max_iters):
            
            # E-step: 
            delta, gamma, gamma_mar = self.expectation_step( delta, pi, beta)
            
            # M-step: 
            pi, beta = self.maximization_step( delta)
            
            # to compute the ELBO and the ICL variational
            current_elbo, icl_soft_term = H_q(self.S1, self.S, self.df, beta, delta, pi, gamma, gamma_mar, self.b, term2)
            elbo_values.append(current_elbo)
            icl_soft_values.append(icl_soft_term - self.pen) 
                        
            #print()
            #print(f"Iteration {i+1}, ELBO: {current_elbo}")
        
            # Convergence criterion
            if i > 0 and np.abs(elbo_values[-1] - elbo_values[-2]) < self.tol:
                print("Convergence reached")
                break
        
        #icl = ICL(self.S1, self.S, self.df, self.b, delta, pi, gamma_mar, self.pen)
        icl = ICL_effective(self.S1, self.S, self.df, self.b, self.times_births, self.M, self.N0, delta, pi, gamma_mar)
        
        return pi, beta, lambda_est, mu_est, delta, gamma, gamma_mar, elbo_values, icl, icl_soft_values

