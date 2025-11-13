import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import csv, sys
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import os
from flask import Flask, request, jsonify
from functools import wraps
import threading
import sys
import logging

from scipy.stats import norm, gamma as scipy_gamma

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPU available: {physical_devices}")
    except RuntimeError as e:
        print(f"Error in configuring GPU: {e}")
else:
    print("WARNING: NO GPU DETECTED")




class GNN:
    def __init__(self, hyperparams=None, hidden_units=-1):

        if hidden_units<0 :
            raise ValueError("You should pass the number of hidden nodes to use")
        self.HIDDEN_UNITS=hidden_units

        if hyperparams is None:
            self.hyperparams = {
                'd_beta1': 3, 'd_beta2': 3, 'c_beta': 15, 'e_beta': 20.0,
                'd_gamma1': 2, 'd_gamma2': 5, 'c_gamma': 20.0, 'a': 4, 'b': 1.5,
                'm_0_beta_means': [3.046, 0.297, 7.0]  # [mean_D, mean_μ, mean_λ]
            }
        else:
            self.hyperparams = hyperparams
        
        self.precision_beta_dist = tfd.Gamma(
            concentration=self.hyperparams['d_beta1']/2,
            rate=self.hyperparams['d_beta2']/2
        )
        
        self.precision_gamma_dist = tfd.Gamma(
            concentration=self.hyperparams['d_gamma1']/2,
            rate=self.hyperparams['d_gamma2']/2
        )

        self.precision_sigma_dist= tfd.Gamma(
            concentration=self.hyperparams['a']/2,
            rate=self.hyperparams['b']/2
        )
        
        self.mcmc_samples = []
        self.is_trained = False

    @staticmethod
    @tf.function 
    def gompertz_function(t, N0, D, mu, lambda_p):
        # Gompertz function with learned params D, mu, lambda
        e = tf.constant(np.e, dtype=tf.float32)  # Euler's number
        D_safe = tf.maximum(D, 1e-6)
        exponent = 1.0 + (mu * e * (lambda_p - t)) / D_safe
        return N0 + D_safe * tf.exp(-tf.exp(exponent))

    def sample_new_hierarchical_parameters(self):
        """Samples the hierarchical parameters according to the prior distributions
        """

        # m_0β (super mean) is the mean value around which the means of the beta weights (m_iβ) are centered.
        # m_iβ (i app. 1,2,3) are the means for each of the three beta weights which estimates rispectively D, μ, λ.
        # m_γ (vector of size = number of inputs/ number of external environmental factors) is the mean for the gamma weights (those which connect the inputs to the hidden layer).

        # Step 0: σ² ~ 1/G(a/2, b/2)
        precision_sigma = self.precision_sigma_dist.sample()
        sigma_squared = 1.0 / precision_sigma # a and b controls the variance of the error model 


        # Step 1: σ²_γ ~ 1/G(d_γ1/2, d_γ2/2)
        precision_gamma = self.precision_gamma_dist.sample()   # 1/variance, high precision = γ weights close to mean, low precision = more spread out
         # precision_gamma è uno scalare
        sigma_gamma_squared = 1.0 / precision_gamma # is the variance of the γ weights
        
        # Step 2: m_γ|σ²_γ ~ N(0, σ²_γ/c_γ * I)
        m_gamma_dist = tfd.Normal(0.0, tf.sqrt(sigma_gamma_squared / self.hyperparams["c_gamma"])) # c_gamma controls the variance of m_gamma, where a higher value means a smaller variance, so m_gamma close to 0.
        m_gamma = m_gamma_dist.sample([3])  # Per [T, pH, NaCl]


        # Step 3: σ²_β ~ 1/G(d_β1/2, d_β2/2) 
        precision_beta = self.precision_beta_dist.sample()  # 1/variance, high precision = β weights close to mean, low precision = more spread out
        sigma_beta_squared = 1.0 / precision_beta # is the variance of the β weights
        
        # Step 4: m_0β|σ²_β ~ N(m_0_beta_means, σ²_β/e_β)

        
        m_0_beta = []
        for i in range(3):
            m_0_beta_dist = tfd.Normal(
                self.hyperparams['m_0_beta_means'][i],  # Media specifica per parametro
                tf.sqrt(sigma_beta_squared / self.hyperparams["e_beta"])
            )
            m_0_beta.append(m_0_beta_dist.sample())

        m_0_beta = tf.stack(m_0_beta)

        # m_0_beta_dist = tfd.Normal(0.0, np.sqrt(sigma_beta_squared / self.hyperparams["e_beta"]))  # e_beta controls the variance of m_0beta, where a higher value means a smaller variance, so m_0β close to 0.
        # m_0_beta = m_0_beta_dist.sample()
        

        # Step 5: m_iβ|σ²_β ~ N(m_0β, σ²_β/c_β) per i=1,2,3 (D, μ, λ)
        m_i_beta = []
        for i in range(3):
            m_i_beta_dist = tfd.Normal(
                m_0_beta[i], 
                tf.sqrt(sigma_beta_squared / self.hyperparams["c_beta"])
            )
            m_i_beta.append(m_i_beta_dist.sample())

        m_i_beta = tf.stack(m_i_beta)
        #m_i_beta_dist = tfd.Normal(m_0_beta, np.sqrt(sigma_beta_squared / self.hyperparams["c_beta"])) # c_beta controls the variance of m_ibeta (i=1,2,3), where a higher value means a smaller variance, so m_ibeta close to m_0β.
        #m_i_beta = m_i_beta_dist.sample([3])  # Per [D, μ, λ]
        
        return {
            'sigma_beta_squared': sigma_beta_squared,
            'sigma_gamma_squared': sigma_gamma_squared,
            'm_0_beta': m_0_beta,
            'm_gamma': m_gamma,
            'm_i_beta': m_i_beta,
            'precision_beta': precision_beta,
            'precision_gamma': precision_gamma,
            'sigma_squared' : sigma_squared,
            'precision_sigma' : precision_sigma
        }
    
    def sample_new_network_weights(self, hierarchical_params):
        """Samples the network weights according to the prior distributions
        """
        
        # β_ik|m_iβ, σ²_β ~ N(m_iβ, σ²_β)
        beta_weights = []
        for i in range(3):  # Per D, μ, λ
            beta_i_dist = tfd.Normal(
                hierarchical_params['m_i_beta'][i], 
                tf.sqrt(hierarchical_params['sigma_beta_squared'])
            )
            beta_i = beta_i_dist.sample([self.HIDDEN_UNITS])
            beta_weights.append(beta_i)
        
        # γ_k|m_γ, σ²_γ ~ N(m_γ, σ²_γ I)
        gamma_k_dist = tfd.Normal(
        hierarchical_params['m_gamma'],  # [3]
        tf.sqrt(hierarchical_params['sigma_gamma_squared'])
        )
        # Sample [HIDDEN_UNITS, 3]
        gamma_weights = gamma_k_dist.sample([self.HIDDEN_UNITS])
        
        
        return {
            'beta': tf.stack(beta_weights),  # [3, HIDDEN_UNITS]
            'gamma': gamma_weights  # [HIDDEN_UNITS, 3]
        }
    
    @tf.function
    def forward_pass(self, environmental_data, weights):
        """Forward pass between the neural network, using the current weights and estimating the Gompertz parameters
        
        Network architecture:
        INPUT (env data) → HIDDEN (sigmoid activation) → OUTPUT (parametri Gompertz)
        
        The Sigmoid activation function in the hidden layer introduces non linearity
        """
        # environmental_data: [3] = [Duty, Temperature, Frequency]
        # weights['gamma']: [HIDDEN_UNITS, 3]
        # weights['beta']: [3, HIDDEN_UNITS]

        is_single = (len(tf.shape(environmental_data)) == 1) #if single prediction or batch training error calculation

        # if batch calculation (training)
        env_data = tf.reshape(environmental_data, [-1, 3])  # [N, 3] o [1, 3]
        
        # input → hidden
        hidden_inputs = tf.matmul(env_data, weights['gamma'], transpose_b=True)  # [N, HIDDEN_UNITS]
        hidden_activations = tf.nn.sigmoid(hidden_inputs)  # [N, HIDDEN_UNITS]

        # hidden → Gompertz params
        outputs = tf.matmul(hidden_activations, weights['beta'], transpose_b=True) # [N, 3]
        D = outputs[:, 0]
        mu = outputs[:, 1]
        lambda_p = outputs[:, 2]

        
        # Se used for prediction, returns a scalar (scalar input -> scalar output)
        if is_single:
            return D[0], mu[0], lambda_p[0]

        return D, mu, lambda_p

    @tf.function
    def likelihood_probability_tf(self, env_data, times, N_0s, N_observed, weights, sigma_squared):
        """Calculates log-likelihood
        
        Teoria della likelihood: P(data|θ) = Π_i P(y_i|x_i,θ)
        In log-space: log P(data|θ) = Σ_i log P(y_i|x_i,θ)
        """
        # calculates log P(dati|parametri) for all the data points, assuming independence between data points
        # usage of logaritmis to avoid numerical underflow when multiplying many small probabilities

        # neural network prediction (batch)
        D, mu, lambda_p = self.forward_pass(env_data, weights)

        # Gompertz prediction, using gnn predicted params (batch)
        gompertz_pred = self.gompertz_function(times, N_0s, D, mu, lambda_p)  # [N]
        
        # errors between observed and predicted data
        errors = N_observed - gompertz_pred  # [N]
        
        # error model Likelihood vectorized for each prediction => a different std. dev.
        """
            εₜᵢⱼ|gₜᵢⱼ, σ, v ∼ N(0, σ²g(tᵢⱼ)ᵛ) , v=0.5
        """
        std_dev = tf.sqrt(sigma_squared * tf.pow(gompertz_pred, 0.5))
        error_dist = tfd.Normal(loc=0.0, scale=std_dev)
        log_point_likelihoods = error_dist.log_prob(errors)  # P(εₜᵢⱼ|gₜᵢⱼ, σ, v), the PDF value of the error given the observed correct value - predicted gompertz value
        # so how likely is to observe this error given the predicted value and the error model, the more the error is close to 0, the more likely is to observe it
        # so the more the predicted value is close to the observed value, the more likely is to observe this error, the more the likelihood is high, the more the current weights is good to explain the observed data

        # Total likelihood
        # P(data_point_1 & data_point_2 & ... & data_point_n|weights) = Π[P(εₜᵢⱼ|gₜᵢⱼ, σ, v)] for all data points, cause we assume indipendence between data points
        # Sum instead product, because in logaritmic domain
        log_total_likelihood = tf.reduce_sum(log_point_likelihoods)

        return log_total_likelihood

    def likelihood_probability(self, observed_data, weights, hierarchical_params):
        """Wrapper which calls the tf optimized function, this is necessary because you cannot use "numpy" functions inside a tf optimized function"""
     
        env_data = tf.constant([dp['environmental'] for dp in observed_data], dtype=tf.float32)  # [N, 3]
        times = tf.constant([dp['time'] for dp in observed_data], dtype=tf.float32)  # [N]
        N_0s = tf.constant([dp['initial_pop'] for dp in observed_data], dtype=tf.float32)  # [N]
        N_observed = tf.constant([dp['observed_concentration'] for dp in observed_data], dtype=tf.float32)  # [N]

        return self.likelihood_probability_tf(
            env_data, times, N_0s, N_observed, weights, 
            tf.cast(hierarchical_params['sigma_squared'], tf.float32)
        ).numpy()
    
    @tf.function
    def prior_probability_tf(self, beta_weights, gamma_weights, hierarchical_params):
        """Calculates the prior
        """
        log_total_prior = tf.constant(0.0, dtype=tf.float32)
        
        # Prior on β weights
        for i in range(3):
            beta_i = beta_weights[i]  # [HIDDEN_UNITS]
            beta_prior_dist = tfd.Normal(
                tf.cast(hierarchical_params['m_i_beta'][i], tf.float32),
                tf.cast(tf.sqrt(hierarchical_params['sigma_beta_squared']), tf.float32)
            )
            log_probs = beta_prior_dist.log_prob(beta_i)  # [HIDDEN_UNITS]
            
            log_total_prior += tf.reduce_sum(log_probs)

        # P(β_ik|m_iβ, σ²_β) # the PDF value of the beta weight given the hierarchical parameters, 
        # so how likely is to observe this beta weight given the hierarchical parameters, the more the beta weight is close to the mean m_iβ, 
        # the more likely is to observe it, so the more the prior is high, the more the current beta weights are good according to the prior knowledge
        # Π[P(β_ik| hyperparams)] for all i,k (i outputs and k hidden units), cause we assume indipendence between weights
        
        
        # Prior on γ weights
        gamma_flat = tf.reshape(gamma_weights, [-1])  # [HIDDEN_UNITS * 3]
        # Order: [γ_00, γ_01, γ_02, γ_10, γ_11, γ_12, ...]
        m_gamma_expanded = tf.tile(hierarchical_params['m_gamma'], [self.HIDDEN_UNITS])
        # Order: [m_γ[0], m_γ[1], m_γ[2], m_γ[0], m_γ[1], m_γ[2], ...]
        
        gamma_prior_dist = tfd.Normal(
            tf.cast(m_gamma_expanded, tf.float32),
            tf.cast(tf.sqrt(hierarchical_params['sigma_gamma_squared']), tf.float32)
        )
        log_probs = gamma_prior_dist.log_prob(gamma_flat)
        
        if tf.reduce_any(tf.math.is_nan(log_probs)) or tf.reduce_any(tf.math.is_inf(log_probs)):
            return tf.constant(-np.inf, dtype=tf.float32)
        
        log_total_prior += tf.reduce_sum(log_probs)
        # P(γ_kj|m_γ, σ²_γ I) # the PDF value of the gamma weight given the hierarchical parameters,
        # so how likely is to observe this gamma weight given the hierarchical parameters, the more the gamma weight is close to the mean m_γ,
        # the more likely is to observe it, so the more the prior is high, the more the current gamma weights are good according to the prior knowledge
        # Π[P(γ_kj| hyperparams)] for all k,j (k hidden units and j inputs), cause we assume indipendence between weights
        
        # Prior on precision hiperparameters
        log_total_prior += self.precision_beta_dist.log_prob(hierarchical_params['precision_beta'])
        log_total_prior += self.precision_gamma_dist.log_prob(hierarchical_params['precision_gamma'])
        log_total_prior += self.precision_sigma_dist.log_prob(hierarchical_params['precision_sigma'])

        # Prior on state hyperparams m_iβ | m_0β, σ²_β

        # Prior on m_iβ | m_0β, σ²_β, i=1,2,3 (number of outputs)
        for i in range(3):
            m_i_beta_dist = tfd.Normal(
                tf.cast(hierarchical_params['m_0_beta'][i], tf.float32),
                tf.cast(tf.sqrt(hierarchical_params['sigma_beta_squared'] / self.hyperparams['c_beta']), tf.float32)
            )
            log_prob = m_i_beta_dist.log_prob(hierarchical_params['m_i_beta'][i])
            log_total_prior += log_prob      

        # Prior on m_0β | σ²_β
        for i in range(3):
            m_0_beta_dist = tfd.Normal(
                tf.cast(self.hyperparams['m_0_beta_means'][i], tf.float32),  # Media specifica
                tf.cast(tf.sqrt(hierarchical_params['sigma_beta_squared'] / self.hyperparams['e_beta']), tf.float32)
            )
            log_prob = m_0_beta_dist.log_prob(hierarchical_params['m_0_beta'][i])
            log_total_prior += log_prob 

        # Prior on m_γ | σ²_γ a vector of size = number of inputs
        m_gamma_dist = tfd.Normal(
            0.0,
            tf.cast(tf.sqrt(hierarchical_params['sigma_gamma_squared'] / self.hyperparams['c_gamma']), tf.float32)
        )
        log_probs = m_gamma_dist.log_prob(hierarchical_params['m_gamma'])  # [3]
        log_total_prior += tf.reduce_sum(log_probs)

        return log_total_prior

    def prior_probability(self, weights, hierarchical_params):
        """Wrapper which calls the tf optimized function, this is necessary because you cannot use "numpy" functions inside a tf optimized function"""
        # calculates log P(parametri) wrt weights and hyperparams
        return self.prior_probability_tf(
            weights['beta'], 
            weights['gamma'], 
            hierarchical_params
        ).numpy()
    
    def posterior_probability(self, weights, hierarchical_params, observed_data):
        """log P(parametri|dati) ∝ log P(dati|parametri) + log P(parametri)
        
        Bayes theorem in log-space:
        log P(θ|D) = log P(D|θ) + log P(θ)
                """
        likelihood = self.likelihood_probability(observed_data, weights, hierarchical_params)
        prior = self.prior_probability(weights, hierarchical_params)
        return likelihood + prior
    
    def propose_new_state(self, current_weights, current_hierarchical_params, step_size=0.02, iteration=0):
        """Proposes a new state by perturbing the current weights and hierarchical parameters
        
        Theory of MCMC proposal:
        - Random Walk Metropolis: q(θ'|θ) = N(θ, σ²I)
        - Adaptive step size: diminuishing the step size at each iteration for convergence
        """
        
        adaptive_step = step_size * (1.0 + 5.0 / (1.0 + iteration / 10000))

        # proposes new weights by adding a small gaussian noise to the current weights
        new_weights = {
            'beta': current_weights['beta'] + tf.random.normal(current_weights['beta'].shape, 0, adaptive_step, dtype=tf.float32),
            'gamma': current_weights['gamma'] + tf.random.normal(current_weights['gamma'].shape, 0, adaptive_step, dtype=tf.float32)
        }
        
        # proposes new hierarchical parameters by adding a small gaussian noise to the current hierarchical parameters
        new_hierarchical_params = copy.deepcopy(current_hierarchical_params)
        new_hierarchical_params['m_0_beta'] = (
            current_hierarchical_params['m_0_beta'] + 
            tf.random.normal([3], 0, adaptive_step, dtype=tf.float32)
        )
        new_hierarchical_params['m_gamma'] = current_hierarchical_params['m_gamma'] + tf.random.normal([3], 0, adaptive_step, dtype=tf.float32) # 3 environmental inputs
        new_hierarchical_params['m_i_beta'] = current_hierarchical_params['m_i_beta'] + tf.random.normal([3], 0, adaptive_step, dtype=tf.float32) # 3 outputs D, μ, λ
        
        new_hierarchical_params['precision_beta'] = tf.exp(tf.math.log(tf.maximum(current_hierarchical_params['precision_beta'], 1e-10)) + tf.random.normal([], 0, adaptive_step, dtype=tf.float32))
        new_hierarchical_params['precision_gamma'] = tf.exp(tf.math.log(tf.maximum(current_hierarchical_params['precision_gamma'], 1e-10)) + tf.random.normal([], 0, adaptive_step, dtype=tf.float32))
        new_hierarchical_params['precision_sigma'] = tf.exp(tf.math.log(tf.maximum(current_hierarchical_params['precision_sigma'], 1e-10)) + tf.random.normal([], 0, adaptive_step, dtype=tf.float32))
        new_hierarchical_params['precision_beta'] = tf.clip_by_value(new_hierarchical_params['precision_beta'], 1e-6, 1e6)
        new_hierarchical_params['precision_gamma'] = tf.clip_by_value(new_hierarchical_params['precision_gamma'], 1e-6, 1e6)
        new_hierarchical_params['precision_sigma'] = tf.clip_by_value(new_hierarchical_params['precision_sigma'], 1e-6, 1e6)

        
        new_hierarchical_params['sigma_beta_squared'] = 1.0 / new_hierarchical_params['precision_beta']
        new_hierarchical_params['sigma_gamma_squared'] = 1.0 / new_hierarchical_params['precision_gamma']
        new_hierarchical_params['sigma_squared'] = 1.0 / new_hierarchical_params['precision_sigma']

        
        return new_weights, new_hierarchical_params

    def mcmc_inference(self, observed_data, num_samples, burn_in, thin):
        """Executes MCMC inference to sample from the posterior distribution of the model parameters
        
        Metropolis-Hastings MCMC theory:
        1. Propose a new state θ' ~ q(·|θ), where the state θ' consists in new sampled weights and hyerarchical params
        2. Calculates acceptance ratio: α = min(1, P(θ'|D)/P(θ|D))
        3. Accepts with probability α, otherwise mantain θ (the old state)
        3.b A new state is accepted, with a smaller probability, even if its log posterior is worse wrt the current state one, in order to avoid to be stuck in local minima
        4. Ignores the first burn-in iterations (initial exploration), saves the samples every 'thin' iterations, avoiding high correlation between samples
        
        This operations produces posterior samples P(θ|D) asintotically.
        """

        print("Starting MCMC inference...")
        
        # Initialize current state
        current_hierarchical_params = self.sample_new_hierarchical_parameters()
        current_weights = self.sample_new_network_weights(current_hierarchical_params)
        
        acceptance_count = 0
        total_proposals = 0 
        samples = []
        
        log_current_posterior = self.posterior_probability(
            current_weights, current_hierarchical_params, observed_data
        )
        
        for iteration in range(num_samples + burn_in):
            # Proposes new state
            proposed_weights, proposed_hierarchical_params = self.propose_new_state(
                current_weights, current_hierarchical_params, iteration=iteration
            )
            
            # Calculate posterior probabilities
            log_proposed_posterior = self.posterior_probability(
                proposed_weights, proposed_hierarchical_params, observed_data
            )
            
            # Metropolis-Hastings acceptance
            # Decide whether to accept the new state
            # the new state: the better it is: (higher posterior, so it is plausible wrt the prior knowledge (high prior) and 
            # it fits well the observed data (high likelihood)) , the more likely to be accepted
            total_proposals += 1

            if np.isnan(log_proposed_posterior) or np.isinf(log_proposed_posterior):
                accept = False
            elif np.isnan(log_current_posterior) or np.isinf(log_current_posterior):
                accept = False
            else:
                log_acceptance_ratio = log_proposed_posterior - log_current_posterior
                
                if log_acceptance_ratio >= 0:
                    accept = True
                else:
                    acceptance_ratio = np.exp(log_acceptance_ratio)
                    accept = (np.random.uniform() < acceptance_ratio)

            if accept:
                current_weights = proposed_weights
                current_hierarchical_params = proposed_hierarchical_params
                log_current_posterior = log_proposed_posterior
                acceptance_count += 1

            # Saves samples after burn-in and thinning
            # the first burn_in samples are discarded, because they are too random and doesn't represent the posterior distribution, 
            # then every 'thin' samples are kept, in order to keep the samples more independent
            if iteration >= burn_in and iteration % thin == 0:
                samples.append({
                    'weights': {
                        'beta': current_weights['beta'].numpy().copy(),
                        'gamma': current_weights['gamma'].numpy().copy()
                    },
                    'hierarchical_params': {k: (v.numpy() if isinstance(v, tf.Tensor) else v) 
                                           for k, v in current_hierarchical_params.items()}
                })
            
            if iteration % 500 == 0:
                acceptance_rate = acceptance_count / total_proposals if total_proposals > 0 else 0
                print(f"Iterazione {iteration}, Acceptance rate: {acceptance_rate:.3f}")
                print(f"Log posterior: {log_current_posterior:.2f}")
                print(f"Log proposed: {log_proposed_posterior:.2f}")

        
        self.mcmc_samples = samples
        self.is_trained = True
        
        print(f"MCMC completed after {num_samples + burn_in} iterations.")
        print(f"Collected samples: {len(self.mcmc_samples)}")
        
        return samples
    
    def predict_with_uncertainty(self, environmental_data, t, N_0):
        """Prediction with uncertainty estimation using the MCMC samples
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() before predict().")
        
        predictions = []
        
        env_tf = tf.constant(environmental_data, dtype=tf.float32)
        t_tf = tf.constant(t, dtype=tf.float32)
        N_0_tf = tf.constant(N_0, dtype=tf.float32)
        
        for sample in self.mcmc_samples:
            weights_tf = {
                'beta': tf.constant(sample['weights']['beta'], dtype=tf.float32),
                'gamma': tf.constant(sample['weights']['gamma'], dtype=tf.float32)
            }
            D, mu, lambda_p = self.forward_pass(env_tf, weights_tf)
            
            # Gompertz prediction
            gompertz_pred = self.gompertz_function(t_tf, N_0_tf, D, mu, lambda_p)
            predictions.append({
                'gompertz_prediction': gompertz_pred.numpy(),
                'D': D.numpy(),
                'mu': mu.numpy(), 
                'lambda': lambda_p.numpy()
            })
        
        # Statistics on predictions: mean, std, quantiles

        gompertz_preds = [p['gompertz_prediction'] for p in predictions]
        D_preds = [p['D'] for p in predictions]
        mu_preds = [p['mu'] for p in predictions]
        lambda_preds = [p['lambda'] for p in predictions]
        
        return {
            'gompertz': {
                'mean': np.mean(gompertz_preds),
                'std': np.std(gompertz_preds),
                'quantiles': np.percentile(gompertz_preds, [2.5, 25, 50, 75, 97.5]) 
            },
            'parameters': {
                'D': {'mean': np.mean(D_preds), 'std': np.std(D_preds)},
                'mu': {'mean': np.mean(mu_preds), 'std': np.std(mu_preds)},
                'lambda': {'mean': np.mean(lambda_preds), 'std': np.std(lambda_preds)}
            }
        }
    
class Model:
    def __init__(self, hyperparams=None, observed_data=[], mcmc_params=None, M=2):

        if mcmc_params is None:
            mcmc_params = {
                'num_samples': 20000,
                'burn_in': 5000,
                'thin': 25
            }

        self.observed_data=observed_data
        self.mcmc_params=mcmc_params
        
        self.gnn = GNN(hyperparams, hidden_units=M)

        self.is_trained = False

    def update_trainset_and_params(self, new_observed_data, mcmc_params=None):
        """Updates the training dataset"""
        if not mcmc_params is None:
            self.mcmc_params=mcmc_params
        self.observed_data = new_observed_data
        self.is_trained = False
        self.samples = []
    
    def fit(self):
        """
        Train the model using MCMC inference
        """
        if len(self.observed_data)==0:
            return {'success': False, 'message': 'No dataset provided'}

        try:
            # Executes MCMC inference
            self.samples = self.gnn.mcmc_inference(
                self.observed_data,
                num_samples=self.mcmc_params['num_samples'],
                burn_in=self.mcmc_params['burn_in'],
                thin=self.mcmc_params['thin'] 
            )
            self.is_trained = True
            return {'success': True, 'samples': len(self.samples)}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def predict(self, environmental_data, t, N_0):
        """
        Predicts the Gompertz function with uncertainty estimation
        """
        if not self.is_trained:
            return {'success': False, 'message': 'Untrained model. Call fit() before predict.'}
        
        try:
            result = self.gnn.predict_with_uncertainty(environmental_data, t, N_0)
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'message': str(e)}

    def save_model(self, filepath):
        """
        Saves the trained model to a file
        """
        if not self.is_trained:
            return {'success': False, 'message': 'Untrained model. Call fit() before saving.'}
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'mcmc_samples': self.gnn.mcmc_samples,
                    'hyperparams': self.gnn.hyperparams,
                    'hidden_units': self.gnn.HIDDEN_UNITS,
                    'mcmc_params': self.mcmc_params
                }, f)
            print(f"Model saved to {filepath}")
            return {'success': True, 'message': f'Model saved to {filepath}'}
        except Exception as e:
            return {'success': False, 'message': str(e)}

    
    def load_model(self, filepath):
        """
        Loads a trained model from a pickle file
        """
        try:
            print(f"Attempting to load model from: {filepath}")
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
                # Ricrea il GNN con i parametri salvati
                self.gnn = GNN(
                    hyperparams=data['hyperparams'],
                    hidden_units=data['hidden_units']
                )
                self.gnn.mcmc_samples = data['mcmc_samples']
                self.gnn.is_trained = True
                
                self.mcmc_params = data['mcmc_params']

                self.is_trained = True
                
                print(f"Model loaded from {filepath}")
                print(f"Loaded {len(self.gnn.mcmc_samples)} samples")
                return {'success': True, 'message': f'Loaded {len(self.gnn.mcmc_samples)} samples'}
                
        except Exception as e:
            print(f"Error loading model: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'message': str(e)}


def normalize(value, min_val, max_val):
    if max_val == min_val:
        raise ValueError("max_val e min_val cannot be the same value")
    return (value - min_val) / (max_val - min_val)



def divide_train_test(all_data):
    
    groups = defaultdict(list)
    observed_data = []
    test_data = []

    for data in all_data:
        key = (
            round(data['environmental'][0], 1),  # Duty
            round(data['environmental'][1], 1),  # Temperature
            round(data['environmental'][2], 1)   # Frequency
        )
        groups[key].append(data)

    np.random.seed(42)

    for key, group in groups.items():
        n_test = max(1, len(group) // 3)
        indices = np.random.permutation(len(group))
        
        for i, idx in enumerate(indices):
            if i < n_test:
                test_data.append(group[idx])
            else:
                observed_data.append(group[idx])

    print(f"Total data points: {len(all_data)}")
    print(f"Training data points: {len(observed_data)}")
    print(f"Test data points: {len(test_data)}")

    return {'observed': observed_data, 'test': test_data}


def test_train(test_data, model):
    if test_data is None or len(test_data) == 0:
        logger.info("No test data provided.")
        return 0.0

    errors = []
    absolute_errors = []
    predictions_list = []

    logger.info("\n=== TEST SET PREDICTIONS ===")
    for i, test in enumerate(test_data):
        env_data = test['environmental']
        t = test['time']
        N_0 = test['initial_pop']
        N_obs = test['observed_concentration']
        
        pred_result = model.predict(env_data, t, N_0)
        if not pred_result['success']:
            logger.info(f"Prediction error: {pred_result['message']}")
            continue
            
        prediction = pred_result['result']
        pred_mean = prediction['gompertz']['mean']
        
        percent_error = abs(pred_mean - N_obs) / N_obs * 100
        absolute_errors.append(pred_mean - N_obs)
        errors.append(percent_error)
        predictions_list.append({
            'predicted': pred_mean,
            'observed': N_obs,
            'error': percent_error
        })
        
        logger.info(f"\nTest {i+1}:")
        logger.info(f"  Predicted: {pred_mean:.4f} ± {prediction['gompertz']['std']:.4f}")
        logger.info(f"  Observed: {N_obs:.4f}")
        logger.info(f"  Absolute Error: {abs(pred_mean - N_obs):.4f}")
        logger.info(f"  Percent Error: {percent_error:.2f}%")

    logger.info("\n\n\n")
    logger.info("TEST SET SUMMARY")
    logger.info("\n\n")
    logger.info(f"Number of test points: {len(test_data)}")
    logger.info(f"Mean Absolute Percentage Error (MAPE): {np.mean(errors):.2f}%")
    logger.info(f"Median Absolute Percentage Error: {np.median(errors):.2f}%")
    logger.info(f"Std of Percentage Errors: {np.std(errors):.2f}%")
    logger.info(f"Min Error: {np.min(errors):.2f}%")
    logger.info(f"Max Error: {np.max(errors):.2f}%")
    logger.info(f"Errors < 5%: {sum(e < 5 for e in errors)}/{len(errors)}")
    logger.info(f"Errors < 10%: {sum(e < 10 for e in errors)}/{len(errors)}")

    mse = np.mean([e**2 for e in absolute_errors])
    return mse

def load_dataset(filepath):
    all_data = []
    
    print(f"DEBUG: Opening {filepath}")
    
    with open(filepath, 'r', encoding='utf-8-sig', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        
        # Leggi header
        header = next(reader)
        print(f"DEBUG: Header = {header}")
        
        for row_num, row in enumerate(reader, start=2):
            print(f"DEBUG: Row {row_num} raw = {row}")
            
            # Pulisci ogni cella
            row = [cell.strip() for cell in row]
            print(f"DEBUG: Row {row_num} cleaned = {row}")
            
            if not row or len(row) < 6:
                print(f"DEBUG: Skipping row {row_num} - not enough columns")
                continue
            
            if all(not cell for cell in row[:6]):
                print(f"DEBUG: Skipping row {row_num} - all empty")
                continue
                
            try:
                single_data={'environmental':[], 'time':0.0, 'initial_pop':0.0, 'observed_concentration':0.0}
                environmental_data=[]
                environmental_data.append(normalize(float(row[0]), 0, 1))
                environmental_data.append(normalize(float(row[1]), 18, 28))
                environmental_data.append(normalize(float(row[2]), 0, 40))
                single_data['environmental']=environmental_data
                single_data['time']= float(row[4])
                single_data['initial_pop']= float(row[3])
                single_data['observed_concentration']= float(row[5])
                all_data.append(single_data)
                print(f"DEBUG: Successfully added row {row_num}")
            except (ValueError, IndexError) as e:
                print(f'ERROR at row {row_num}: {e}, row: {row}')
                continue

    print(f"Loaded {len(all_data)} data points")
    return all_data

def append_row(filepath, data):
    """Appende una riga al file"""
    data = [str(cell).strip() for cell in data]
    
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(data)


def append_csv_to_csv(source_file, target_file):
    with open(source_file, 'r', newline='', encoding='utf-8-sig') as source:
        reader = csv.reader(source, delimiter=';')
        
        with open(target_file, 'a', newline='', encoding='utf-8') as target:
            writer = csv.writer(target, delimiter=';')
            for row in reader:
                row = [cell.strip() for cell in row]
                writer.writerow(row)

    with open(source_file, 'w', newline='', encoding='utf-8') as source:
        pass

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True 
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
# API Key
API_KEY = os.environ.get('MODEL_API_KEY', 'default_api_key')
model = Model(M=3)
is_in_training=False
current_mse=0.0


def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        print("=== AUTH CHECK ===", file=sys.stderr, flush=True)
        api_key = request.headers.get('Authorization')
        print(f"Received API key: {api_key}", file=sys.stderr, flush=True)
        print(f"Expected API key: {API_KEY}", file=sys.stderr, flush=True)

        if api_key and api_key == API_KEY:
            print("AUTH SUCCESS", file=sys.stderr, flush=True)
            return f(*args, **kwargs)
        else:
            print("AUTH FAILED", file=sys.stderr, flush=True)
            return jsonify({'error': 'Invalid or missing API key'}), 401

    return decorated_function

def convert_to_json_serializable(obj):
    """Converte ricorsivamente numpy types in tipi Python nativi"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    global model

    try:
        data = request.get_json()

        duty = float(data.get('dutyCycle', '0'))
        temperature = float(data.get('temperature', '0'))
        frequency = float(data.get('frequency', '0'))
        initial_conc = float(data.get('initialConcentration', '0'))
        time = float(data.get('timeLasted', '0'))

        env_data = []
        env_data.append(normalize(duty,0,1))
        env_data.append(normalize(temperature,18,28))
        env_data.append(normalize(frequency,0,40))


        pred_result = model.predict(env_data, time, initial_conc)
        
        if not pred_result['success']:
            return jsonify({
                'status': 'error',
                'message': pred_result['message']
            }), 400

        print(pred_result, file=sys.stderr, flush=True)
        result = convert_to_json_serializable(pred_result['result'])
        
        return jsonify({
            'status': 'success',
            'received': data,
            'result': result
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/train', methods=['POST'])
@require_api_key
def train():
    global model, is_in_training, current_mse

    def train_model_background():
        global model, is_in_training, current_mse
        try:
            all_data = load_dataset('/app/dataset_conf/dataset.csv')

            #divide_train_test_result = divide_train_test(all_data)
            #new_observed_data = divide_train_test_result['observed']
            #new_test_data = divide_train_test_result['test']
            new_observed_data = all_data
            new_test_data = []

            mcmc_params = {
                'num_samples': 300000,
                'burn_in': 100000,
                'thin': 1000
            }
            model.update_trainset_and_params(new_observed_data, mcmc_params)
            fit_result = model.fit()
            
            if not fit_result['success']:
                logger.info(f"Training failed: {fit_result['message']}")
                is_in_training = False
                return

            logger.info(f"Old Mean Square Error (MSE): {current_mse:.4f}")
            mse = test_train(new_test_data, model)  
            current_mse=mse
            logger.info(f"New Mean Square Error (MSE): {current_mse:.4f}")

            save_result = model.save_model('/app/dataset_conf/model.pkl')
            if save_result['success']:
                logger.info(save_result['message'])
            else:
                logger.info(f"Save failed: {save_result['message']}")
                
            is_in_training = False

        except Exception as e:
            logger.info(f"Training error: {e}")
            is_in_training = False


    try:
        data = request.get_json()
        if(is_in_training):
            return jsonify({
                'status': 'error',
                'received': data,
                'result': 'Training already in progress'
            }), 409

        append_csv_to_csv('/app/dataset_conf/dataset_new_rows.csv', '/app/dataset_conf/dataset.csv')
        is_in_training = True
        thread = threading.Thread(target=train_model_background, daemon=True)
        thread.start()

        return jsonify({
            'status': 'success',
            'received': data,
            'result': 'Training started in background'
        }), 202
        
    except Exception as e:
        is_in_training = False
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/addData', methods=['POST'])
@require_api_key
def addData():
    try:
        data = request.get_json()

        data_row = [
            str(data.get('dutyCycle', '0')),
            str(data.get('temperature', '0')),
            str(data.get('frequency', '0')),
            str(data.get('initialConcentration', '0')),
            str(data.get('timeLasted', '0')),
            str(data.get('observedConcentration', '0'))
        ]
        append_row('/app/dataset_conf/dataset_new_rows.csv', data_row)

        return jsonify({
            'status': 'success',
            'received': data_row,
            'result': True
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def plot_optical_density_with_uncertainty(data1, data2=None, 
                                          label1="Real growth curve", 
                                          label2="Predicted growth curve", 
                                          title="Optical density growth", 
                                          xlabel="Time", 
                                          ylabel="Optical Density",
                                          filename="optical_density_with_uncertainty.png", 
                                          save_path="/app/dataset_conf"):
    """
    Disegna il grafico della densità ottica con banda di confidenza.
    
    Parameters:
    -----------
    data1 : list of tuples
        Lista di coppie (tempo, densità_ottica) per il primo dataset (dati reali)
    data2 : list of tuples, optional
        Lista di triple (tempo, densità_ottica, std_dev) per il secondo dataset (predizioni)
        Se presente, disegna anche la banda di confidenza (mean ± 2*std)
    label1 : str
        Etichetta per il primo dataset
    label2 : str
        Etichetta per il secondo dataset
    title : str
        Titolo del grafico
    xlabel : str
        Etichetta asse x
    ylabel : str
        Etichetta asse y
    filename : str
        Nome del file da salvare
    save_path : str
        Percorso dove salvare il grafico
    
    Returns:
    --------
    str : Path completo del file salvato
    """
    
    # Estrai dati del primo dataset (dati reali)
    times1, od1 = zip(*data1)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot dati reali
    ax.plot(times1, od1, marker='o', linestyle='-', linewidth=2.5, 
            label=label1, color='steelblue', markersize=8)
    
    # Se ci sono predizioni con incertezza
    if data2 is not None:
        # Controlla se data2 contiene triple (time, density, std) o coppie (time, density)
        if len(data2[0]) == 3:
            # Triple: (time, density, std_dev)
            times2, od2, std_devs = zip(*data2)
            times2 = np.array(times2)
            od2 = np.array(od2)
            std_devs = np.array(std_devs)
            
            # Plot predizioni (linea centrale)
            ax.plot(times2, od2, marker='s', linestyle='-', linewidth=2.5, 
                   label=label2, color='darkorange', markersize=6)
            
            # Banda di confidenza al 95% (≈ mean ± 2*std)
            lower_bound = od2 - 2 * std_devs
            upper_bound = od2 + 2 * std_devs
            
            ax.fill_between(times2, lower_bound, upper_bound, 
                           alpha=0.3, color='darkorange', 
                           label='95% Confidence interval')
            
            # Banda più stretta al 68% (≈ mean ± 1*std) - opzionale
            lower_bound_1std = od2 - std_devs
            upper_bound_1std = od2 + std_devs
            
            ax.fill_between(times2, lower_bound_1std, upper_bound_1std, 
                           alpha=0.4, color='orange', 
                           label='68% Confidence interval')
            
        else:
            # Coppie: (time, density) - backward compatibility
            times2, od2 = zip(*data2)
            ax.plot(times2, od2, marker='s', linestyle='-', linewidth=2.5, 
                   label=label2, color='darkorange', markersize=6)
    
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold')
    
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {full_path}")
    
    return full_path



def plot_prior_with_final_values(model, save_path="/app/dataset_conf"):
    """
    Visualizza le distribuzioni prior con i valori finali (stima bayesiana).
    
    Per ogni parametro gerarchico:
    - Disegna la distribuzione prior (la "campana" iniziale)
    - Marca con un punto rosso la MEDIA dei campioni MCMC (stima bayesiana finale)
    
    La media rappresenta la stima bayesiana che integra su tutta l'incertezza posterior.
    
    Questo mostra: "Partivo da questa credenza (curva), la stima finale è qui (punto rosso)"
    
    Parameters:
    -----------
    model : Model
        Il modello addestrato
    save_path : str
        Percorso dove salvare i grafici
    
    Returns:
    --------
    dict : Dizionario con i path dei file salvati
    """
    
    if not model.is_trained:
        raise ValueError("Il modello deve essere addestrato")
    
    samples = model.gnn.mcmc_samples
    hyperparams = model.gnn.hyperparams
    
    # Calcola i valori finali (media dei campioni posterior - stima bayesiana)
    m_0_beta_final = [
        np.mean([s['hierarchical_params']['m_0_beta'][0] for s in samples]),
        np.mean([s['hierarchical_params']['m_0_beta'][1] for s in samples]),
        np.mean([s['hierarchical_params']['m_0_beta'][2] for s in samples])
    ]
    
    m_i_beta_final = [
        np.mean([s['hierarchical_params']['m_i_beta'][0] for s in samples]),
        np.mean([s['hierarchical_params']['m_i_beta'][1] for s in samples]),
        np.mean([s['hierarchical_params']['m_i_beta'][2] for s in samples])
    ]
    
    m_gamma_final = [
        np.mean([s['hierarchical_params']['m_gamma'][0] for s in samples]),
        np.mean([s['hierarchical_params']['m_gamma'][1] for s in samples]),
        np.mean([s['hierarchical_params']['m_gamma'][2] for s in samples])
    ]
    
    sigma_beta_sq_final = np.mean([s['hierarchical_params']['sigma_beta_squared'] for s in samples])
    sigma_gamma_sq_final = np.mean([s['hierarchical_params']['sigma_gamma_squared'] for s in samples])
    sigma_sq_final = np.mean([s['hierarchical_params']['sigma_squared'] for s in samples])
    
    saved_files = {}
    
    # ========================================================================
    # FIGURA 1: m_0β (3 parametri: D, μ, λ)
    # Prior: m_0β|σ²_β ~ N(m_0_beta_means, σ²_β/e_β)
    # ========================================================================
    
    print("\nCreating prior distributions with final values...")
    print("="*80)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    param_names = ['D (capacity)', 'μ (growth rate)', 'λ (lag phase)']
    colors = ['steelblue', 'darkorange', 'forestgreen']
    
    # Per il prior, usiamo una stima ragionevole di σ²_β
    # (potremmo usare il valore finale, o un valore dal prior di σ²_β)
    # Usiamo il valore atteso del prior di σ²_β: E[1/Gamma] ≈ d_beta2/(d_beta1-2)
    if hyperparams['d_beta1'] > 2:
        sigma_beta_sq_prior_mean = hyperparams['d_beta2'] / (hyperparams['d_beta1'] - 2)
    else:
        sigma_beta_sq_prior_mean = sigma_beta_sq_final  # fallback
    
    for idx, (name, final_val, color) in enumerate(zip(param_names, m_0_beta_final, colors)):
        ax = axes[idx]
        
        # Prior distribution: N(m_0_beta_means[idx], sqrt(σ²_β/e_β))
        prior_mean = hyperparams['m_0_beta_means'][idx]
        prior_std = np.sqrt(sigma_beta_sq_prior_mean / hyperparams['e_beta'])
        
        # Disegna la campana gaussiana
        x_range = np.linspace(prior_mean - 4*prior_std, prior_mean + 4*prior_std, 300)
        prior_pdf = norm.pdf(x_range, prior_mean, prior_std)
        
        ax.plot(x_range, prior_pdf, linewidth=3, color=color, label='Prior distribution')
        ax.fill_between(x_range, prior_pdf, alpha=0.3, color=color)
        
        # Marca il valore finale con un punto rosso
        final_pdf_value = norm.pdf(final_val, prior_mean, prior_std)
        ax.plot(final_val, final_pdf_value, 'o', color='red', markersize=12, 
               label=f'Final value: {final_val:.3f}', zorder=10)
        
        # Linea verticale al valore finale
        ax.axvline(final_val, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Marca anche la media del prior
        ax.axvline(prior_mean, color=color, linestyle=':', linewidth=2, alpha=0.5,
                  label=f'Prior mean: {prior_mean:.3f}')
        
        ax.set_xlabel(f'm_0β ({name})', fontsize=13, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=13)
        ax.set_title(f'Prior Distribution: m_0β for {name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Info box
        shift = final_val - prior_mean
        info_text = f'Prior: N({prior_mean:.3f}, {prior_std:.3f})\n'
        info_text += f'Final: {final_val:.3f}\n'
        info_text += f'Shift: {shift:+.3f}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    path1 = os.path.join(save_path, "prior_m0beta_with_final.png")
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files['m0_beta'] = path1
    print(f"✓ m_0β: {path1}")
    
    # ========================================================================
    # FIGURA 2: m_iβ (3 parametri: D, μ, λ)
    # Prior: m_iβ|m_0β, σ²_β ~ N(m_0β, σ²_β/c_β)
    # ========================================================================
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, final_val, color) in enumerate(zip(param_names, m_i_beta_final, colors)):
        ax = axes[idx]
        
        # Per il prior usiamo m_0β finale come centro
        prior_mean = m_0_beta_final[idx]
        prior_std = np.sqrt(sigma_beta_sq_prior_mean / hyperparams['c_beta'])
        
        x_range = np.linspace(prior_mean - 4*prior_std, prior_mean + 4*prior_std, 300)
        prior_pdf = norm.pdf(x_range, prior_mean, prior_std)
        
        ax.plot(x_range, prior_pdf, linewidth=3, color=color, label='Prior distribution')
        ax.fill_between(x_range, prior_pdf, alpha=0.3, color=color)
        
        final_pdf_value = norm.pdf(final_val, prior_mean, prior_std)
        ax.plot(final_val, final_pdf_value, 'o', color='red', markersize=12, 
               label=f'Final value: {final_val:.3f}', zorder=10)
        
        ax.axvline(final_val, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(prior_mean, color=color, linestyle=':', linewidth=2, alpha=0.5,
                  label=f'Prior mean: {prior_mean:.3f}')
        
        ax.set_xlabel(f'm_iβ ({name})', fontsize=13, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=13)
        ax.set_title(f'Prior Distribution: m_iβ for {name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        shift = final_val - prior_mean
        info_text = f'Prior: N({prior_mean:.3f}, {prior_std:.3f})\n'
        info_text += f'Final: {final_val:.3f}\n'
        info_text += f'Shift: {shift:+.3f}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    path2 = os.path.join(save_path, "prior_mibeta_with_final.png")
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files['mi_beta'] = path2
    print(f"✓ m_iβ: {path2}")
    
    # ========================================================================
    # FIGURA 3: m_γ (3 parametri per i 3 input)
    # Prior: m_γ|σ²_γ ~ N(0, σ²_γ/c_γ)
    # ========================================================================
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    input_names = ['Duty Cycle', 'Temperature', 'Frequency']
    
    if hyperparams['d_gamma1'] > 2:
        sigma_gamma_sq_prior_mean = hyperparams['d_gamma2'] / (hyperparams['d_gamma1'] - 2)
    else:
        sigma_gamma_sq_prior_mean = sigma_gamma_sq_final
    
    for idx, (name, final_val, color) in enumerate(zip(input_names, m_gamma_final, colors)):
        ax = axes[idx]
        
        # Prior: N(0, sqrt(σ²_γ/c_γ))
        prior_mean = 0.0
        prior_std = np.sqrt(sigma_gamma_sq_prior_mean / hyperparams['c_gamma'])
        
        x_range = np.linspace(prior_mean - 4*prior_std, prior_mean + 4*prior_std, 300)
        prior_pdf = norm.pdf(x_range, prior_mean, prior_std)
        
        ax.plot(x_range, prior_pdf, linewidth=3, color=color, label='Prior distribution')
        ax.fill_between(x_range, prior_pdf, alpha=0.3, color=color)
        
        final_pdf_value = norm.pdf(final_val, prior_mean, prior_std)
        ax.plot(final_val, final_pdf_value, 'o', color='red', markersize=12, 
               label=f'Final value: {final_val:.3f}', zorder=10)
        
        ax.axvline(final_val, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(prior_mean, color=color, linestyle=':', linewidth=2, alpha=0.5,
                  label=f'Prior mean: {prior_mean:.3f}')
        
        ax.set_xlabel(f'm_γ ({name})', fontsize=13, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=13)
        ax.set_title(f'Prior Distribution: m_γ for {name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        shift = final_val - prior_mean
        info_text = f'Prior: N({prior_mean:.3f}, {prior_std:.3f})\n'
        info_text += f'Final: {final_val:.3f}\n'
        info_text += f'Shift: {shift:+.3f}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    path3 = os.path.join(save_path, "prior_mgamma_with_final.png")
    plt.savefig(path3, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files['m_gamma'] = path3
    print(f"✓ m_γ: {path3}")
    
    # ========================================================================
    # FIGURA 4: Varianze (σ²_β, σ²_γ, σ²)
    # Prior: precision ~ Gamma(d/2, d2/2), quindi σ² = 1/precision
    # ========================================================================
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    variance_names = ['σ²_β (beta weights)', 'σ²_γ (gamma weights)', 'σ² (error model)']
    variance_finals = [sigma_beta_sq_final, sigma_gamma_sq_final, sigma_sq_final]
    variance_colors = ['purple', 'crimson', 'teal']

    prior_params = [
        (hyperparams['d_beta1'], hyperparams['d_beta2']),
        (hyperparams['d_gamma1'], hyperparams['d_gamma2']),
        (hyperparams['a'], hyperparams['b'])
    ]

    for idx, (name, final_val, color, (d1, d2)) in enumerate(zip(variance_names, variance_finals, variance_colors, prior_params)):
        ax = axes[idx]
        
        # Prior: precision ~ Gamma(d1/2, d2/2), quindi σ² = 1/precision
        precision_samples = scipy_gamma.rvs(d1/2, scale=2/d2, size=50000)
        variance_samples = 1.0 / precision_samples
        
        # KDE per curva smooth
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(variance_samples)
        
        # SOLO PER IL SECONDO GRAFICO (σ²_γ) - range fisso 0-1000
        if idx == 1:  # σ²_γ (gamma weights)
            x_min = 0
            x_max = 1000
            x_range = np.linspace(x_min, x_max, 2000)
        else:
            # PRIMO E TERZO GRAFICO: lascia come prima
            x_min = 0
            x_max = np.percentile(variance_samples, 99)
            x_range = np.linspace(x_min, x_max, 500)
        
        prior_pdf = kde(x_range)
        
        # Disegna la curva
        ax.plot(x_range, prior_pdf, linewidth=3, color=color, label='Prior distribution')
        ax.fill_between(x_range, prior_pdf, alpha=0.3, color=color)
        
        # Punto rosso al valore finale
        final_pdf_value = kde.evaluate([final_val])[0]
        ax.plot(final_val, final_pdf_value, 'o', color='red', markersize=10, 
            label=f'Final value: {final_val:.4f}', zorder=10)
        ax.axvline(final_val, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Media del prior
        prior_mean = np.mean(variance_samples)
        ax.axvline(prior_mean, color=color, linestyle=':', linewidth=2, alpha=0.5,
                label=f'Prior mean: {prior_mean:.4f}')
        
        ax.set_xlabel(name, fontsize=13, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=13)
        ax.set_title(f'Prior Distribution: {variance_names[idx].split("(")[0].strip()}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        shift = final_val - prior_mean
        info_text = f'Prior mean: {prior_mean:.4f}\n'
        info_text += f'Final: {final_val:.4f}\n'
        info_text += f'Shift: {shift:+.4f}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlim(x_min, x_max)

    plt.tight_layout()
    path4 = os.path.join(save_path, "prior_variances_with_final.png")
    plt.savefig(path4, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files['variances'] = path4
    print(f"✓ Variances: {path4}")
    
    # ========================================================================
    # FIGURA 5: Error Model (distribuzione dell'errore)
    # Il modello di errore è: ε ~ N(0, σ²×g(t)^0.5)
    # Visualizziamo la distribuzione di σ² (già fatto) e mostriamo l'effetto
    # ========================================================================
    
    # Nota: L'error_dist nel codice non è un parametro da salvare/visualizzare direttamente
    # perché dipende da σ² e dal valore predetto g(t)
    # Quello che possiamo visualizzare è come σ² (il parametro base) è cambiato
    # Questo è già fatto nella Figura 4 (terzo subplot)
    
    # Però possiamo fare un grafico separato più dettagliato per σ² con interpretazione
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    d1, d2 = hyperparams['a'], hyperparams['b']
    
    # Prior: precision ~ Gamma(a/2, b/2), quindi σ² = 1/precision
    precision_samples = scipy_gamma.rvs(d1/2, scale=2/d2, size=50000)
    variance_samples = 1.0 / precision_samples
    
    # KDE per curva smooth
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(variance_samples)
    
    # Range per il plot - SEMPRE mostra bene la forma del prior
    x_min = 0
    x_max = np.percentile(variance_samples, 95)
    
    x_range = np.linspace(x_min, x_max, 500)
    prior_pdf = kde(x_range)
    
    # Disegna la curva smooth
    ax.plot(x_range, prior_pdf, linewidth=3, color='teal', label='Prior distribution')
    ax.fill_between(x_range, prior_pdf, alpha=0.3, color='teal')
    
    # Punto rosso al valore finale
    if sigma_sq_final <= x_max:
        final_pdf_value = kde(sigma_sq_final)[0]
        ax.plot(sigma_sq_final, final_pdf_value, 'o', color='red', markersize=12, 
               label=f'Final value: {sigma_sq_final:.4f}', zorder=10)
        ax.axvline(sigma_sq_final, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
    else:
        ax.axvline(x_max * 0.98, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
        ax.annotate(f'Final: {sigma_sq_final:.4f}\n(off scale →)', 
                   xy=(x_max * 0.95, ax.get_ylim()[1] * 0.7),
                   fontsize=11, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2),
                   ha='right')
    
    prior_mean = np.mean(variance_samples)
    ax.axvline(prior_mean, color='teal', linestyle=':', linewidth=2.5, alpha=0.7,
              label=f'Prior mean: {prior_mean:.4f}')
    
    ax.set_xlabel('σ² (Error Model Variance)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title('Prior Distribution: Error Model Variance σ²\nε ~ N(0, σ²×g(t)^0.5)', 
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Limita asse x - sempre mostra bene il prior
    ax.set_xlim(x_min, x_max)
    
    shift = sigma_sq_final - prior_mean
    info_text = f'Prior: 1/Gamma({d1/2:.1f}, {d2/2:.2f})\n'
    info_text += f'Prior mean: {prior_mean:.4f}\n'
    info_text += f'Final σ²: {sigma_sq_final:.4f}\n'
    info_text += f'Shift: {shift:+.4f}\n'
    if sigma_sq_final > x_max:
        info_text += '(off scale)\n'
    info_text += f'\nError model:\nε ~ N(0, σ²×g(t)^0.5)\n'
    info_text += f'where g(t) is the\nGompertz prediction'
    
    ax.text(0.98, 0.97, info_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    
    plt.tight_layout()
    path5 = os.path.join(save_path, "prior_error_model_with_final.png")
    plt.savefig(path5, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files['error_model'] = path5
    print(f"✓ Error Model (σ²): {path5}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("PRIOR DISTRIBUTIONS WITH BAYESIAN ESTIMATES")
    print("="*80)
    print("\nInterpretazione:")
    print("  - La curva mostra la distribuzione prior (credenza iniziale)")
    print("  - Il punto rosso mostra la MEDIA POSTERIOR (stima bayesiana finale)")
    print("  - La media integra su tutta la posterior, non è solo il MAP")
    print("  - Lo 'Shift' indica quanto abbiamo imparato dai dati")
    print("\nSe il punto rosso è:")
    print("  - Vicino al centro della campana → I dati confermano il prior")
    print("  - Lontano dal centro → I dati hanno aggiornato significativamente il prior")
    print("\nFiles created:")
    for key, path in saved_files.items():
        print(f"  - {key}: {path}")
    print("\nNote:")
    print("  - L'error model completo è: ε ~ N(0, σ²×g(t)^0.5)")
    print("  - Il grafico 'error_model' mostra la distribuzione di σ² (parametro base)")
    print("="*80 + "\n")
    
    return saved_files



if __name__ == '__main__':
    print("Starting Bayesian Neural Network model...")
    
    all_data = []
    observed_data = []
    test_data = []

    if(os.path.exists('/app/dataset_conf/dataset.csv')):
        print("A dataset is present, loading it...")
        
        all_data = load_dataset('/app/dataset_conf/dataset.csv')

        divide_train_test_result = divide_train_test(all_data)
        observed_data = divide_train_test_result['observed']
        test_data = divide_train_test_result['test']
        #observed_data = all_data
        #test_data = []

  
    load_result = model.load_model('/app/dataset_conf/model.pkl')
    if load_result['success']:
        print("Model loaded successfully")
        
        mse = test_train(test_data, model)
        current_mse=mse  
        print(f"Mean Square Error (MSE): {mse:.4f}")
        data1=[(0,8.18), (17, 12.528), (20, 12.112), (24, 12.76), (37, 12.653), (48, 12.507)]

        prediction_batch=[{'environmental':[0, normalize(25.3, 18, 28), normalize(0, 0, 40)], 'time': 0, 'initial_pop': 8.18},
                          {'environmental':[0, normalize(26.2, 18, 28), normalize(0, 0, 40)], 'time': 17, 'initial_pop': 8.18},
                          {'environmental':[0, normalize(25.3, 18, 28), normalize(0, 0, 40)], 'time': 20, 'initial_pop': 8.18},
                          {'environmental':[0, normalize(25.5, 18, 28), normalize(0, 0, 40)], 'time': 24, 'initial_pop': 8.18},
                          {'environmental':[0, normalize(25.3, 18, 28), normalize(0, 0, 40)], 'time': 37, 'initial_pop': 8.18},
                          {'environmental':[0, normalize(25.3, 18, 28), normalize(0, 0, 40)], 'time': 48, 'initial_pop': 8.18}]

        results=[]

        for i, test in enumerate(prediction_batch):
            env_data = test['environmental']
            t = test['time']
            N_0 = test['initial_pop']
            
            pred_result = model.predict(env_data, t, N_0)
                
            prediction = pred_result['result']
            pred_mean = prediction['gompertz']['mean']
            results.append((t, pred_mean, prediction['gompertz']['std']))
        
        plot_optical_density_with_uncertainty(data1=data1, data2=results)
        plot_prior_with_final_values(model)

        

    elif(len(all_data)>0):
        print("Training a new model")
        
        mcmc_params = {
            'num_samples': 300000,
            'burn_in': 100000,
            'thin': 1000
        }

        model = Model(observed_data=observed_data, M=4, mcmc_params=mcmc_params)
        
        fit_result = model.fit()
        if not fit_result['success']:
            print(f"Training failed: {fit_result['message']}")
            sys.exit(1)

        mse = test_train(test_data, model)  
        current_mse=mse
        print(f"Mean Square Error (MSE): {mse:.4f}")
        

        save_result = model.save_model('/app/dataset_conf/model.pkl')
        if not save_result['success']:
            print(f"Failed to save model: {save_result['message']}")

    else:
        print("Neither a model and a dataset were found, please provide a valid dataset in '/app/dataset_conf/dataset.csv' or a trained model in '/app/dataset_conf/model.pkl'")
        
        sys.exit(1)

    print("Starting Flask server...")
    
    
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False, threaded=True)