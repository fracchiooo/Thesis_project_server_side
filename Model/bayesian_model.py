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

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers




class GNN:
    def __init__(self, hyperparams=None, hidden_units=-1):

        if hidden_units<0 :
            raise ValueError("You should pass the number of hidden nodes to use")
        self.HIDDEN_UNITS=hidden_units

        if hyperparams is None:
            # TODO set hyperparams according to prior knowledge
            # Default hyperparameters
            self.hyperparams = {
                'd_beta1': 6, 'd_beta2': 1, 'c_beta': 15.0, 'e_beta': 20.0,
                'd_gamma1': 4, 'd_gamma2': 1, 'c_gamma': 20.0, 'a': 4, 'b': 1.5,
                'm_0_beta_means': [2.0, 0.5, 12.0]  # NUOVO: [mean_D, mean_μ, mean_λ]
            }
        else:
            self.hyperparams = hyperparams
        
        # prior distributions for precision hyperparameters (always fixed)
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
    def gompertz_function(t, N0, D, mu, lambda_p):
        # Gompertz function with learned params D, mu, lambda
        e = np.exp(1) # Euler's number
        D_safe = np.maximum(D, 1e-6)
        exponent = 1.0 + (mu * e * (lambda_p - t)) / D_safe
        return N0 + D_safe * tf.exp(-tf.exp(exponent))

    @staticmethod
    def error_model(gompertz_prediction, sigma_squared, v=0.5):
        """εₜᵢⱼ|gₜᵢⱼ, σ, v ∼ N(0, σ²g(tᵢⱼ)ᵛ) , v=0.5"""
    
        std_dev= np.sqrt(sigma_squared * np.power(gompertz_prediction, v))
        return tfd.Normal(loc=0.0, scale=std_dev)


    def sample_new_hierarchical_parameters(self):
        """Samples the hierarchical parameters according to the prior distributions"""

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
        m_gamma_dist = tfd.Normal(0.0, np.sqrt(sigma_gamma_squared / self.hyperparams["c_gamma"])) # c_gamma controls the variance of m_gamma, where a higher value means a smaller variance, so m_gamma close to 0.
        m_gamma = m_gamma_dist.sample([3]).numpy()  # Per [T, pH, NaCl]


        # Step 3: σ²_β ~ 1/G(d_β1/2, d_β2/2) 
        precision_beta = self.precision_beta_dist.sample()  # 1/variance, high precision = β weights close to mean, low precision = more spread out
        sigma_beta_squared = 1.0 / precision_beta # is the variance of the β weights
        
        # Step 4: m_0β|σ²_β ~ N(m_0_beta_means, σ²_β/e_β)

        
        m_0_beta = []
        for i in range(3):
            m_0_beta_dist = tfd.Normal(
                self.hyperparams['m_0_beta_means'][i],  # Media specifica per parametro
                np.sqrt(sigma_beta_squared / self.hyperparams["e_beta"])
            )
            m_0_beta.append(m_0_beta_dist.sample().numpy())

        m_0_beta = tf.stack(m_0_beta)
        m_0_beta = np.array(m_0_beta)

        # m_0_beta_dist = tfd.Normal(0.0, np.sqrt(sigma_beta_squared / self.hyperparams["e_beta"]))  # e_beta controls the variance of m_0beta, where a higher value means a smaller variance, so m_0β close to 0.
        # m_0_beta = m_0_beta_dist.sample()
        

        # Step 5: m_iβ|σ²_β ~ N(m_0β, σ²_β/c_β) per i=1,2,3 (D, μ, λ)
        m_i_beta = []
        for i in range(3):
            if i == 1 or i == 2:  # μ e λ
                # m_μβ, m_λβ ~ TruncatedNormal([0, ∞))
                m_i_beta_dist = tfd.TruncatedNormal(
                    loc=m_0_beta[i],
                    scale=np.sqrt(sigma_beta_squared / self.hyperparams["c_beta"]),
                    low=0.0,
                    high=np.inf
                )
            else:  # D - Normal standard
                m_i_beta_dist = tfd.Normal(
                    m_0_beta[i], 
                    np.sqrt(sigma_beta_squared / self.hyperparams["c_beta"])
                )
            m_i_beta.append(m_i_beta_dist.sample().numpy())

        m_i_beta = tf.stack(m_i_beta)
        m_i_beta = np.array(m_i_beta)
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
        """Samples the network weights according to the prior distributions"""
        
        # β_ik|m_iβ, σ²_β ~ N(m_iβ, σ²_β)
        beta_weights = []
        for i in range(3):  # Per D, μ, λ
            if i == 1 or i == 2:  # μ and λ >=0
                beta_i_dist = tfd.TruncatedNormal(
                    loc=hierarchical_params['m_i_beta'][i], 
                    scale=np.sqrt(hierarchical_params['sigma_beta_squared']),
                    low=0.0,
                    high=np.inf
                )
            else:
                beta_i_dist = tfd.Normal(
                    hierarchical_params['m_i_beta'][i], 
                    np.sqrt(hierarchical_params['sigma_beta_squared'])
                )
            beta_i = beta_i_dist.sample([self.HIDDEN_UNITS]).numpy()
            beta_weights.append(beta_i)
        
        # γ_k|m_γ, σ²_γ ~ N(m_γ, σ²_γ I)

        gamma_k_dist = tfd.Normal(
        hierarchical_params['m_gamma'],  # [3]
        np.sqrt(hierarchical_params['sigma_gamma_squared'])
        )
        # Sample [HIDDEN_UNITS, 3] in una volta
        gamma_weights = gamma_k_dist.sample([self.HIDDEN_UNITS]).numpy()
        
        """
        gamma_weights = []
        for k in range(self.HIDDEN_UNITS):
            gamma_k_dist = tfd.Normal(
                hierarchical_params['m_gamma'],
                np.sqrt(hierarchical_params['sigma_gamma_squared'])
            )
            gamma_k = gamma_k_dist.sample().numpy()  # Shape [3] per [T, pH, NaCl]
            gamma_weights.append(gamma_k)
        """
        
        return {
            'beta': np.array(beta_weights),  # List of 3 vectors, each one of length num_hidden
            'gamma': gamma_weights  # List of num_hidden vectors, each one of length 3
        }
    
    def forward_pass(self, environmental_data, weights):
        """Forward pass between the neural network, using the current weights and estimating the Gompertz parameters"""
        # environmental_data: [3] = [T_norm, pH_norm, NaCl_norm]
        # weights['gamma']: [HIDDEN_UNITS, 3]
        # weights['beta']: [3, HIDDEN_UNITS]

        is_single = (np.array(environmental_data).ndim == 1) #if single prediction or batch training error calculation

        # if batch calculation (training)
        env_data = np.atleast_2d(environmental_data)  # [N, 3] o [1, 3]
        
        # input → hidden
        # env_data: [N, 3] @ gamma.T: [3, HIDDEN_UNITS] = [N, HIDDEN_UNITS]
        hidden_inputs = np.dot(env_data, weights['gamma'].T)  # [N, HIDDEN_UNITS]
        hidden_activations = tf.nn.sigmoid(hidden_inputs).numpy()  # [N, HIDDEN_UNITS]
        
        # hidden → parametri Gompertz
        # hidden_activations: [N, HIDDEN_UNITS] @ beta[i]: [HIDDEN_UNITS] = [N]
        D = np.dot(hidden_activations, weights['beta'][0])  # [N]
        mu = np.dot(hidden_activations, weights['beta'][1])  # [N]
        lambda_p = np.dot(hidden_activations, weights['beta'][2])  # [N]
        
        # Se used for prediction, returns a scalar (scalar input -> scalar output)
        if is_single:
            return D[0], mu[0], lambda_p[0]

        """
        # Layer nascosto: input → hidden
        hidden_inputs = np.dot(weights['gamma'], environmental_data)  # [HIDDEN_UNITS]
        hidden_activations = tf.nn.sigmoid(hidden_inputs).numpy()    # Sigmoid activation function
        
        # Layer output: hidden → parametri Gompertz
        D = np.dot(weights['beta'][0], hidden_activations)     # D
        mu = np.dot(weights['beta'][1], hidden_activations)    # μ
        lambda_p = np.dot(weights['beta'][2], hidden_activations)  # λ
        """
        return D, mu, lambda_p

    def likelihood_probability(self, observed_data, weights, hierarchical_params):
        # calculates log P(dati|parametri) for each data points, assuming independence between data points
        # usage of logaritmis to avoid numerical underflow when multiplying many small probabilities

        env_data = np.array([dp['environmental'] for dp in observed_data])  # [N, 3]
        times = np.array([dp['time'] for dp in observed_data], dtype=np.float32)  # [N]
        N_0s = np.array([dp['initial_pop'] for dp in observed_data], dtype=np.float32)  # [N]
        N_observed = np.array([dp['observed_concentration'] for dp in observed_data], dtype=np.float32)  # [N]

        # neural network prediction
        D, mu, lambda_p = self.forward_pass(env_data, weights)

        # Gompertz prediction, using nn predicted params
        gompertz_pred = self.gompertz_function(times, N_0s, D, mu, lambda_p)  # [N]
        
        # errors between observed and predicted data
        errors = N_observed - gompertz_pred  # [N]
        errors = tf.cast(errors, tf.float32)
        
        # error model Likelihood vectorized
        error_dist = self.error_model(gompertz_pred, tf.cast(hierarchical_params['sigma_squared'], tf.float32))
        log_point_likelihoods = error_dist.log_prob(errors).numpy()  # P(εₜᵢⱼ|gₜᵢⱼ, σ, v), the PDF value of the error given the observed correct value - predicted gompertz value
        # so how likely is to observe this error given the predicted value and the error model, the more the error is close to 0, the more likely is to observe it
        # so the more the predicted value is close to the observed value, the more likely is to observe this error, the more the likelihood is high, the more the current weights is good to explain the observed data

        # Total likelihood
        log_total_likelihood = np.sum(log_point_likelihoods)

        """ log_total_likelihood = 0.0
        
        for data_point in observed_data:
            env_data = data_point['environmental']  # [T, pH, NaCl] normalizzati
            t = data_point['time']
            N_0 = data_point['initial_pop']
            N_observed = data_point['observed_concentration']
            
            # neural network prediction
            D, mu, lambda_p = self.forward_pass(env_data, weights)
            
            # Gompertz prediction, using nn predicted params
            gompertz_pred = self.gompertz_function(t, N_0, D, mu, lambda_p)

            # error between observed and predicted
            error = N_observed - gompertz_pred
            error = tf.cast(error, tf.float32)
            
            # error model likelihood
            error_dist = self.error_model(gompertz_pred, tf.cast(hierarchical_params['sigma_squared'], tf.float32))
            log_point_likelihood = error_dist.log_prob(error).numpy() # P(εₜᵢⱼ|gₜᵢⱼ, σ, v), the PDF value of the error given the observed correct value - predicted gompertz value
            # so how likely is to observe this error given the predicted value and the error model, the more the error is close to 0, the more likely is to observe it
            # so the more the predicted value is close to the observed value, the more likely is to observe this error, the more the likelihood is high, the more the current weights is good to explain the observed data
            log_total_likelihood += log_point_likelihood # P(data_point_1 & data_point_2 & ... & data_point_n|weights) = Π[P(εₜᵢⱼ|gₜᵢⱼ, σ, v)] for all data points, cause we assume indipendence between data points
        """
        return log_total_likelihood
    
    def prior_probability(self, weights, hierarchical_params):
        # calculates log P(parametri) wrt weights and hyperparams

        log_total_prior = 0.0
        
        # Prior on β weights
        # Per ogni i, tutti i beta_i[k] hanno la STESSA distribuzione
        for i in range(3):
            beta_i = weights['beta'][i]  # [HIDDEN_UNITS]
            
            if i == 1 or i == 2:  # μ and λ
                beta_prior_dist = tfd.TruncatedNormal(
                    loc=float(hierarchical_params['m_i_beta'][i]),
                    scale=float(np.sqrt(hierarchical_params['sigma_beta_squared'])),
                    low=0.0,
                    high=np.inf
                )
            else:  # D
                beta_prior_dist = tfd.Normal(
                    float(hierarchical_params['m_i_beta'][i]),
                    float(np.sqrt(hierarchical_params['sigma_beta_squared']))
                )
            
            # Tutti i beta_i[k] valutati insieme (stessa distribuzione)
            log_probs = beta_prior_dist.log_prob(beta_i).numpy()  # [HIDDEN_UNITS]
            
            if np.any(np.isnan(log_probs)) or np.any(np.isinf(log_probs)):
                return -np.inf
            
            log_total_prior += np.sum(log_probs)

        # P(β_ik|m_iβ, σ²_β) # the PDF value of the beta weight given the hierarchical parameters, 
        # so how likely is to observe this beta weight given the hierarchical parameters, the more the beta weight is close to the mean m_iβ, 
        # the more likely is to observe it, so the more the prior is high, the more the current beta weights are good according to the prior knowledge
        # Π[P(β_ik| hyperparams)] for all i,k (i outputs and k hidden units), cause we assume indipendence between weights
        
        
        # Prior on γ weights
        # Prior on γ weights - VETTORIZZATO con broadcasting
        gamma_flat = weights['gamma'].flatten()  # [HIDDEN_UNITS * 3]
        # Ordine: [γ_00, γ_01, γ_02, γ_10, γ_11, γ_12, ...]
        m_gamma_expanded = np.tile(hierarchical_params['m_gamma'], self.HIDDEN_UNITS)
        # Ordine: [m_γ[0], m_γ[1], m_γ[2], m_γ[0], m_γ[1], m_γ[2], ...]
        
        gamma_prior_dist = tfd.Normal(
            m_gamma_expanded.astype(np.float64),
            float(np.sqrt(hierarchical_params['sigma_gamma_squared']))
        )
        
        log_probs = gamma_prior_dist.log_prob(gamma_flat).numpy()
        
        if np.any(np.isnan(log_probs)) or np.any(np.isinf(log_probs)):
            return -np.inf
        
        log_total_prior += np.sum(log_probs)
                # P(γ_kj|m_γ, σ²_γ I) # the PDF value of the gamma weight given the hierarchical parameters,
                # so how likely is to observe this gamma weight given the hierarchical parameters, the more the gamma weight is close to the mean m_γ,
                # the more likely is to observe it, so the more the prior is high, the more the current gamma weights are good according to the prior knowledge
                # Π[P(γ_kj| hyperparams)] for all k,j (k hidden units and j inputs), cause we assume indipendence between weights
        
        # Prior on precision hiperparameters
        log_total_prior += self.precision_beta_dist.log_prob(hierarchical_params['precision_beta']).numpy()
        log_total_prior += self.precision_gamma_dist.log_prob(hierarchical_params['precision_gamma']).numpy()
        log_total_prior += self.precision_sigma_dist.log_prob(hierarchical_params['precision_sigma']).numpy()

        # Prior on state hyperparams m_iβ | m_0β, σ²_β

        # Prior on m_iβ | m_0β, σ²_β, i=1,2,3 (number of outputs)
        for i in range(3):
            if i == 1 or i == 2:  # μ and λ
                m_i_beta_dist = tfd.TruncatedNormal(
                    loc=float(hierarchical_params['m_0_beta'][i]),
                    scale=float(np.sqrt(hierarchical_params['sigma_beta_squared'] / self.hyperparams['c_beta'])),
                    low=0.0,
                    high=np.inf
                )
            else:  # D
                m_i_beta_dist = tfd.Normal(
                    float(hierarchical_params['m_0_beta'][i]),
                    float(np.sqrt(hierarchical_params['sigma_beta_squared'] / self.hyperparams['c_beta']))
                )
            log_prob = m_i_beta_dist.log_prob(hierarchical_params['m_i_beta'][i]).numpy()
            log_total_prior += log_prob      

        # Prior on m_0β | σ²_β
        for i in range(3):
            m_0_beta_dist = tfd.Normal(
                float(self.hyperparams['m_0_beta_means'][i]),  # Media specifica
                float(np.sqrt(hierarchical_params['sigma_beta_squared'] / self.hyperparams['e_beta']))
            )
            log_prob = m_0_beta_dist.log_prob(hierarchical_params['m_0_beta'][i]).numpy()
            log_total_prior += log_prob 

        # Prior on m_γ | σ²_γ a vector of size = number of inputs
        m_gamma_dist = tfd.Normal(
            0.0,
            float(np.sqrt(hierarchical_params['sigma_gamma_squared'] / self.hyperparams['c_gamma']))
        )
        log_probs = m_gamma_dist.log_prob(hierarchical_params['m_gamma']).numpy()  # [3]
        log_total_prior += np.sum(log_probs)

        return log_total_prior
    

    def posterior_probability(self, weights, hierarchical_params, observed_data):
        # log P(parametri|dati) ∝ log P(dati|parametri) + log P(parametri)
        likelihood = self.likelihood_probability(observed_data, weights, hierarchical_params)
        prior = self.prior_probability(weights, hierarchical_params)
        return likelihood + prior
    

    def propose_new_state(self, current_weights, current_hierarchical_params, step_size=0.02, iteration=0):
        """Proposes a new state by perturbing the current weights and hierarchical parameters"""
        
        adaptive_step = step_size * (1.0 + 0.3 / (1.0 + iteration / 30000))

        # proposes new weights by adding a small gaussian noise to the current weights
        new_weights = {
            'beta': current_weights['beta'] + np.random.normal(0, adaptive_step, current_weights['beta'].shape),
            'gamma': current_weights['gamma'] + np.random.normal(0, adaptive_step, current_weights['gamma'].shape)
        }
        
        # proposes new hierarchical parameters by adding a small gaussian noise to the current hierarchical parameters
        new_hierarchical_params = copy.deepcopy(current_hierarchical_params)
        new_hierarchical_params['m_0_beta'] = (
            current_hierarchical_params['m_0_beta'] + 
            np.random.normal(0, adaptive_step, 3)  # 3 perturbazioni indipendenti
        )
        new_hierarchical_params['m_gamma'] += np.random.normal(0, adaptive_step, 3) # 3 environmental inputs
        new_hierarchical_params['m_i_beta'] += np.random.normal(0, adaptive_step, 3) # 3 outputs D, μ, λ
        
        
        new_hierarchical_params['precision_beta'] = np.exp(np.log(np.maximum(current_hierarchical_params['precision_beta'], 1e-10)) + np.random.normal(0, adaptive_step))
        new_hierarchical_params['precision_gamma'] = np.exp(np.log(np.maximum(current_hierarchical_params['precision_gamma'], 1e-10)) + np.random.normal(0, adaptive_step))
        new_hierarchical_params['precision_sigma'] = np.exp(np.log(np.maximum(current_hierarchical_params['precision_sigma'], 1e-10)) + np.random.normal(0, adaptive_step))
        new_hierarchical_params['precision_beta'] = np.clip(new_hierarchical_params['precision_beta'], 1e-6, 1e6)
        new_hierarchical_params['precision_gamma'] = np.clip(new_hierarchical_params['precision_gamma'], 1e-6, 1e6)
        new_hierarchical_params['precision_sigma'] = np.clip(new_hierarchical_params['precision_sigma'], 1e-6, 1e6)

        
        new_hierarchical_params['sigma_beta_squared'] = 1.0 / new_hierarchical_params['precision_beta']
        new_hierarchical_params['sigma_gamma_squared'] = 1.0 / new_hierarchical_params['precision_gamma']
        new_hierarchical_params['sigma_squared'] = 1.0 / new_hierarchical_params['precision_sigma']

        
        return new_weights, new_hierarchical_params



    def mcmc_inference(self, observed_data, num_samples, burn_in, thin):

        """
        Executes MCMC inference to sample from the posterior distribution of the model parameters.        
        observed_data: Lista of dict with keys:
                      ['environmental', 'time', 'initial_pop', 'observed_concentration']
        """
        print("Starting MCMC inference...")
        
        # Initialize current state
        current_hierarchical_params = self.sample_new_hierarchical_parameters()
        current_weights = self.sample_new_network_weights(current_hierarchical_params)
        
        acceptance_count = 0
        total_proposals = 0 
        samples = []
        
        for iteration in range(num_samples + burn_in):
            # Proposes new state
            proposed_weights, proposed_hierarchical_params = self.propose_new_state(
                current_weights, current_hierarchical_params, iteration=iteration
            )
            
            # Calculate posterior probabilities
            log_current_posterior = self.posterior_probability(
                current_weights, current_hierarchical_params, observed_data
            )
            log_proposed_posterior = self.posterior_probability(
                proposed_weights, proposed_hierarchical_params, observed_data
            )
            
            # Metropolis-Hastings acceptance
            if np.isnan(log_proposed_posterior) or np.isinf(log_proposed_posterior):
                accept = False
            elif np.isnan(log_current_posterior) or np.isinf(log_current_posterior):
                accept = False
            else:
                log_acceptance_ratio = log_proposed_posterior - log_current_posterior
                
                # Evita overflow
                if log_acceptance_ratio >= 0:
                    accept = True
                else:
                    acceptance_ratio = np.exp(log_acceptance_ratio)
                    accept = (np.random.uniform() < acceptance_ratio)

                        
            # Decide whether to accept the new state
            # the new state: the better it is: (higher posterior, so it is plausible wrt the prior knowledge (high prior) and 
            # it fits well the observed data (high likelihood)) , the more likely to be accepted
            total_proposals += 1  # AGGIUNGI QUESTO

            if accept:
                current_weights = proposed_weights
                current_hierarchical_params = proposed_hierarchical_params
                acceptance_count += 1  # AGGIUNGI QUESTO

            # Saves samples after burn-in and thinning
            # the first burn_in samples are discarded, because they are too random and doesn't represent the posterior distribution, 
            # then every 'thin' samples are kept, in order to keep the samples more independent
            if iteration >= burn_in and iteration % thin == 0:  
                samples.append({
                    'weights': {
                        'beta': current_weights['beta'].copy(),
                        'gamma': current_weights['gamma'].copy()
                    },
                    'hierarchical_params': current_hierarchical_params.copy()
                })
            
            if iteration % 500 == 0:
                acceptance_rate = acceptance_count / total_proposals if total_proposals > 0 else 0
                print(f"Iterazione {iteration}, Acceptance rate: {acceptance_rate:.3f}")
                print(f"Log posterior: {log_current_posterior:.2f}")
                print(f"Log proposed: {log_proposed_posterior:.2f}")  # AGGIUNGI QUESTO

        
        self.mcmc_samples = samples
        self.is_trained = True
        
        print(f"MCMC completed after {num_samples + burn_in} iterations.")
        print(f"Collected samples: {len(self.mcmc_samples)}")
        
        return samples
    
    def predict_with_uncertainty(self, environmental_data, t, N_0):
        """
        Prediction with uncertainty estimation using the MCMC samples.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() before predict().")
        
        predictions = []
        
        for sample in self.mcmc_samples:
            # Prediction using the current weights
            D, mu, lambda_p = self.forward_pass(environmental_data, sample['weights'])
            
            # Gompertz prediction
            gompertz_pred = self.gompertz_function(t, N_0, D, mu, lambda_p)
            predictions.append({
                'gompertz_prediction': gompertz_pred,
                'D': D,
                'mu': mu, 
                'lambda': lambda_p
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

        """
        observed_data: List of dicts with keys:
        [
            {
                'environmental': [T_norm, pH_norm, NaCl_norm],
                'time': t,
                'initial_pop': N_0, 
                'observed_concentration': N_observed
            },
            ...
        ]
        """

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

    def update_trainset(self, new_observed_data):
        """Updates the training dataset"""
        self.observed_data = new_observed_data
        self.is_trained = False
        self.samples = []
    
    def fit(self):
        """
        Train the model using MCMC inference
        """

        if len(self.observed_data)==0:
            raise ValueError("Set a valid dataset when instanciating the class")
        
        
        # Executes MCMC inference
        self.samples = self.gnn.mcmc_inference(
            observed_data,
            num_samples=self.mcmc_params['num_samples'],
            burn_in=self.mcmc_params['burn_in'],
            thin=self.mcmc_params['thin'] 
        )
        self.is_trained = True
        
        return self.samples
    
    def predict(self, environmental_data, t, N_0):
        """Predicts the Gompertz function with uncertainty estimation"""
        if not self.is_trained:
            raise ValueError("Untrained model. Call fit() before predict().")
            
        return self.gnn.predict_with_uncertainty(environmental_data, t, N_0)


    def save_model(self, filepath):
        """Saves the trained model to a file"""
        if not self.is_trained:
            raise ValueError("Untrained model. Call fit() before saving.")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'gnn_state': self.gnn,
                'samples': self.samples
            }, f)
        print(f"Model saved to {filepath}")

    
    def load_model(self, filepath):
        """Loads a trained model from a file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.gnn = data['gnn_state']
                self.samples = data['samples']
                self.is_trained = True
                print(f"Model loaded from {filepath}")
                return 0
        except Exception as e:
            print(f"Error in loading the model")
            return -1


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

    errors = []
    absolute_errors = []
    predictions_list = []

    print("\n=== TEST SET PREDICTIONS ===")
    for i, test in enumerate(test_data):
        env_data = test['environmental']
        t = test['time']
        N_0 = test['initial_pop']
        N_obs = test['observed_concentration']
        
        prediction = model.predict(env_data, t, N_0)
        pred_mean = prediction['gompertz']['mean']
        
        percent_error = abs(pred_mean - N_obs) / N_obs * 100
        absolute_errors.append(pred_mean - N_obs)
        errors.append(percent_error)
        predictions_list.append({
            'predicted': pred_mean,
            'observed': N_obs,
            'error': percent_error
        })
        
        print(f"\nTest {i+1}:")
        print(f"  Predicted: {pred_mean:.4f} ± {prediction['gompertz']['std']:.4f}")
        print(f"  Observed: {N_obs:.4f}")
        print(f"  Absolute Error: {abs(pred_mean - N_obs):.4f}")
        print(f"  Percent Error: {percent_error:.2f}%")

    print("\n\n\n")
    print("TEST SET SUMMARY")
    print("\n\n")
    print(f"Number of test points: {len(test_data)}")
    print(f"Mean Absolute Percentage Error (MAPE): {np.mean(errors):.2f}%")
    print(f"Median Absolute Percentage Error: {np.median(errors):.2f}%")
    print(f"Std of Percentage Errors: {np.std(errors):.2f}%")
    print(f"Min Error: {np.min(errors):.2f}%")
    print(f"Max Error: {np.max(errors):.2f}%")
    print(f"Errors < 5%: {sum(e < 5 for e in errors)}/{len(errors)}")
    print(f"Errors < 10%: {sum(e < 10 for e in errors)}/{len(errors)}")


    print("\n\n\n\n")
    print("\n=== WORST PREDICTIONS ===")
    worst_errors = sorted(enumerate(predictions_list), key=lambda x: x[1]['error'], reverse=True)[:7]
    for idx, pred in worst_errors:
        test = test_data[idx]
        print(f"\nTest {idx+1} (error: {pred['error']:.1f}%):")
        print(f"  Env: PH={test['environmental'][0]*4+3:.1f}, Temperature={test['environmental'][1]*3+22:.1f}, Frequency={test['environmental'][2]*40:.1f}")
        print(f"  Time: {test['time']:.1f}, N0: {test['initial_pop']:.2f}")
        print(f"  Predicted: {pred['predicted']:.2f}, Observed: {pred['observed']:.2f}")

    mse = np.mean([e**2 for e in absolute_errors])
    return mse



def load_dataset(filepath):

    # WARNING: the input environmental data must be normalized between 0 and 1
    all_data = []
    with open(filepath, newline='') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        next(reader)
        try:
            for row_num, row in enumerate(reader, start=1):  # start=1 for first data row
                data = row[:6]
                single_data={'environmental':[], 'time':0.0, 'initial_pop':0.0, 'observed_concentration':0.0}
                environmental_data=[]
                environmental_data.append(normalize(float(row[0].replace(",",".")),0,1))
                environmental_data.append(normalize(float(row[1].replace(",",".")),18,28))
                environmental_data.append(normalize(float(row[2].replace(",", ".")),0,40))
                single_data['environmental']=environmental_data
                single_data['time']= float(row[4].replace(",","."))
                single_data['initial_pop']= float(row[3].replace(",","."))
                single_data['observed_concentration']= float(row[5].replace(",","."))
                all_data.append(single_data)     
        except csv.Error as e:
            print(f'Error reading CSV file at line {reader.line_num}: {e}')
            return []

    return all_data
    

def append_row(filepath, data):
    """Appende una riga al file"""
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def append_csv_to_csv(source_file, target_file):
    with open(source_file, 'r', newline='') as source:
        reader = csv.reader(source)
        
        with open(target_file, 'a', newline='') as target:
            writer = csv.writer(target)
            for row in reader:
                writer.writerow(row)

    with open(source_file, 'w', newline='') as source:
        # Clear the source file
        pass


app = Flask(__name__)

# API Key
API_KEY = os.environ.get('MODEL_API_KEY', 'default_api_key')
rows = []
empty_row_idx = 0
dialect = csv.excel()
model = Model(M=3)
is_in_training=false
current_mse=0.0

if __name__ == '__main__':


    all_data = []
    observed_data = []
    test_data = []


    if(os.path.exists('../../dataset_conf/dataset.csv')):
        print("A dataset is present, loading it...")
        # Load dataset
        all_data = load_dataset('../../dataset_conf/dataset.csv')

        # Split dataset into training and test sets

        divide_train_test_result = divide_train_test(all_data)
        observed_data = divide_train_test_result['observed']
        test_data = divide_train_test_result['test']
  

    if(model.load_model('model.pkl')==0):
        print("Model loaded successfully")
        mse = test_train(test_data, model)
        current_mse=mse  
        print(f"Mean Square Error (MSE): {mse:.4f}")


    elif(len(all_data)>0):
        print("Training a new model")

        # set mcmc training params
        mcmc_params = {
            'num_samples': 4000,
            'burn_in': 1000,
            'thin': 1
        }

        model = Model(observed_data=observed_data, M=3, mcmc_params=mcmc_params)
        
        # Fit the model
        results = model.fit()

        # Test the model
        mse = test_train(test_data, model)  
        current_mse=mse
        print(f"Mean Square Error (MSE): {mse:.4f}")

        # Save the model
        model.save_model('model.pkl')

    else:
        print("Neither a model and a dataset were found, please provide a valid dataset in '../../dataset_conf/dataset.csv' or a trained model in './model.pkl'")
        return -1

    # Avvia il server
    app.run(host='0.0.0.0', port=5000, debug=False)


def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if api_key and api_key == API_KEY:
            return f(*args, **kwargs)
        else:
            return jsonify({'error': 'Invalid or missing API key'}), 401
    
    return decorated_function



@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
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

    
    prediction = model.predict(env_data, time, initial_conc)


    return jsonify({
        'status': 'success',
        'received': data,
        'result': prediction
    }), 200



@app.route('/train', methods=['POST'])
@require_api_key
def train():

    def train_model_background():
        try:
            all_data = load_dataset('../../dataset_conf/dataset.csv')

            # Split dataset into training and test sets

            divide_train_test_result = divide_train_test(all_data)
            new_observed_data = divide_train_test_result['observed']
            new_test_data = divide_train_test_result['test']


            model.update_trainset(new_observed_data)
            model.fit()

            print(f"Old Mean Square Error (MSE): {current_mse:.4f}")
            mse = test_train(new_test_data, model)  
            current_mse=mse
            print(f"New Mean Square Error (MSE): {current_mse:.4f}")

            # Save the model
            model.save_model('model.pkl')
            is_in_training = false

        except Exception as e:
            print(f"Training error: {e}")

    if(is_in_training):
        return jsonify({
            'status': 'error',
            'received': data,
            'result': 'Training already in progress'
        }), 409

    append_csv_to_csv('../../dataset_conf/dataset_new_rows.csv', '../../dataset_conf/dataset.csv')
    is_in_training = true
    thread = threading.Thread(target=train_model_background, daemon=True)
    thread.start()

    return jsonify({
        'status': 'success',
        'received': data,
        'result': 'Training started in background'
    }), 202


@app.route('/addData', methods=['POST'])
@require_api_key
def addData():
    data = request.get_json()

    data_row = [
        str(data.get('dutyCycle', '0')),
        str(data.get('temperature', '0')),
        str(data.get('frequency', '0')),
        str(data.get('initialConcentration', '0')),
        str(data.get('timeLasted', '0')),
        str(data.get('observedConcentration', '0'))
    ]
    append_row('../../dataset_conf/dataset_new_rows.csv', data_row)


    return jsonify({
        'status': 'success',
        'received': data_row,
        'result': True
    }), 200