# dynamic-bdsbm-vem
## Variational EM inference for a dynamic Birth–Death Stochastic Block Model

This repository provides a **Variational EM (VEM)** implementation for a **dynamic Birth–Death Stochastic Block Model (BD-SBM)**, designed for settings where:
- individuals **arrive and depart** over time (following a birth–death process),
- interactions are observed through **aggregated pairwise counts** over the study period,
- exposure is captured by a pairwise **co-aliveness (alive) matrix**.
This repository contains a Variational EM (VEM) implementation for a **Dynamic Birth–Death Stochastic Block Model (BD-SBM)**, designed to infer community structure when individuals **arrive and depart** over time and interactions are observed through **aggregated pairwise counts**.

Main components:

- Data generation (birth–death process + SBM interactions)  
  ([code here](./BDSBM/data_generation.py))

- VEM inference for the dynamic BD-SBM (dense interaction counts)  
  ([code here](./BDSBM/Dynamic_BDSBM_VEM.py))

- VEM inference for the dynamic BD-SBM (sparse CSR interaction counts)  
  ([code here](./BDSBM/Dynamic_BDSBM_VEM_sparse.py))

- Utility functions (ELBO/ICL computation, Poison binomial distribution function, etc.)  
  ([code here](./BDSBM/utils.py))

A complete end-to-end example (data generation → model fitting → model selection → plots) is provided in:  
- Demo notebook  
  ([notebook here](./notebooks/00_dynamic_birth_death_sbm_vem.ipynb))

For more information, see:  
- Paper / preprint: ([link here](<YOUR_LINK>))

We evaluate the method through simulations where:
- a birth–death process generates individual lifetimes and community memberships,
- an SBM with connection matrix `pi` generates interactions at observation times,
- VEM jointly estimates `(pi, beta, lambda, mu)` and the variational distributions `(delta, gamma, gamma_mar)`,
- model selection can be performed using ICL and its variational (soft) counterpart.


### Usage
The [Dynamic Birth-Death SBM (VEM) notebook](https://github.com/gbayolo26/dynamic-bdsbm-vem/blob/main/Dynamic_Birth-Death_SBM_VEM.ipynb) contains an example of how to run the model

### Maintainers
Gabriela Bayolo Soler (gabriela.bayolo-soler@utc.fr)

<img src="https://github.com/gbayolo26/risk_estimation/assets/79975920/eedd8e6e-6cea-4327-bef3-54fe0256ff06" width="180" height="50">

### Licence
[Apache-2.0 licence](https://github.com/gbayolo26/risk_estimation/blob/main/LICENSE)
