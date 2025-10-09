/**
 * @file generate_alpha_binomial.h
 * @brief Header for efficient sampling routines in dynamic binomial/Bernoulli models,
 *        including logit (Metropolis-Hastings) and probit (Albert-Chib) link functions.
 * @author Michel H. Montoril
 * @date 2025-10-05
 * @version 1.3
 *
 * @details This header declares functions for adaptive MCMC and Gibbs sampling in binomial
 *          state-space models, supporting both logit and probit links:
 *          - Component-wise Metropolis-Hastings updates for local level/trend binomial models (logit link)
 *          - Gibbs sampling via data augmentation for local level/trend Bernoulli models (probit link)
 *          - Configurable adaptation threshold parameters for enhanced flexibility in MH algorithms
 *          - Utility routines for efficient latent variable sampling
 *
 * @changelog
 * - v1.3 (2025-10-05): Added declarations for probit-linked (Albert-Chib) sampling routines.
 * - v1.2 (2025-09-23): Updated function signatures to include min_deviation_threshold.
 */

#ifndef GENERATE_ALPHA_BINOMIAL_H
#define GENERATE_ALPHA_BINOMIAL_H

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        local level model with configurable adaptation threshold.
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample the
 *          level state vector theta_1 in a binomial observation model with logit link.
 *          The adaptation threshold parameter provides flexible control over when
 *          proposal variance adjustments are triggered.
 *
 * @param theta_1              Matrix of level states (vectorized B x n), input/output.
 * @param theta_01             Vector of initial level states (size B).
 * @param theta_1_updated      Sliding window matrix of acceptance indicators
 *                             (vectorized lag_update x n), output. Uses circular indexing.
 * @param alpha                Matrix of transformed probabilities (vectorized B x n), output.
 * @param prec_theta_1         Vector of level precision parameters (size B).
 * @param y                    Vector of observed binomial counts (size n).
 * @param accept_prop          Vector of acceptance proportions for each component (size n).
 * @param log_sigma            Vector of log proposal standard deviations (size n).
 * @param hat_theta_1          Temporary vector for conditional means (size n).
 * @param theta_1_new          Temporary vector for proposed values (size n).
 * @param log_accept_prob      Temporary vector for log acceptance probabilities (size n).
 * @param lag_update           Integer scalar, sliding window size for adaptation frequency.
 * @param n_trials             Number of Bernoulli trials (double).
 * @param n                    Length of the time series.
 * @param iter                 Current MCMC iteration (0-based).
 * @param max_step_size        Double scalar, maximum adaptation step size.
 * @param base_adaptation_rate Double scalar, initial adaptation rate before decay.
 * @param decay_exponent       Double scalar, exponent for diminishing adaptation schedule.
 * @param target_acceptance    Double scalar, target acceptance rate for adaptive tuning.
 * @param min_deviation_threshold Double scalar, minimum absolute deviation from target_acceptance
 *                             required to trigger log_sigma updates. Must be >= 0.
 *
 * @note Model is local level (random walk only, no trend).
 * @note Uses sliding window memory optimization.
 * @note Configurable threshold allows fine-tuned adaptation sensitivity.
 * @note Recommended threshold: 1.0/lag_update for practical applications.
 *
 * @warning min_deviation_threshold must be >= 0.0.
 * @warning lag_update must be > 0 for sliding window indexing.
 *
 * @see adapt_cwmh_parameters
 * @see cwmh_alpha_logit_binomial_locallevel
 * @since version 1.2
 */
void generate_alpha_logit_binomial_locallevel(double *theta_1,
                                              double *theta_01,
                                              double *theta_1_updated,
                                              double *alpha,
                                              double *prec_theta_1,
                                              double *y,
                                              double *accept_prop,
                                              double *log_sigma,
                                              double *hat_theta_1,
                                              double *theta_1_new,
                                              double *log_accept_prob,
                                              int     lag_update,
                                              double  n_trials,
                                              int     n,
                                              int     iter,
                                              double  max_step_size,
                                              double  base_adaptation_rate,
                                              double  decay_exponent,
                                              double  target_acceptance,
                                              double  min_deviation_threshold);

/**
 * @brief Component-wise Metropolis-Hastings sampler for theta_1 in a logit-binomial
 *        local trend model with configurable adaptation threshold.
 *
 * @details Implements an optimized component-wise Metropolis-Hastings algorithm to sample the
 *          level state vector theta_1 in a binomial observation model with local trend
 *          state-space evolution. The adaptation threshold parameter provides flexible
 *          control over when proposal variance adjustments are triggered.
 *
 * @param theta_1              Matrix of level states (vectorized B x n), input/output.
 * @param theta_2              Matrix of trend states (vectorized B x n), input only.
 * @param theta_01             Vector of initial level states (size B).
 * @param theta_02             Vector of initial trend states (size B).
 * @param theta_1_updated      Sliding window matrix of acceptance indicators
 *                             (vectorized lag_update x n), output. Uses circular indexing.
 * @param alpha                Matrix of transformed probabilities (vectorized B x n), output.
 * @param prec_theta_1         Vector of level precision parameters (size B).
 * @param y                    Vector of observed binomial counts (size n).
 * @param accept_prop          Vector of acceptance proportions for each component (size n).
 * @param log_sigma            Vector of log proposal standard deviations (size n).
 * @param hat_theta_1          Temporary vector for conditional means (size n).
 * @param theta_1_new          Temporary vector for proposed values (size n).
 * @param log_accept_prob      Temporary vector for log acceptance probabilities (size n).
 * @param lag_update           Integer scalar, sliding window size for adaptation frequency.
 * @param n_trials             Number of Bernoulli trials (double).
 * @param n                    Length of the time series.
 * @param iter                 Current MCMC iteration (0-based).
 * @param max_step_size        Double scalar, maximum adaptation step size.
 * @param base_adaptation_rate Double scalar, initial adaptation rate before decay.
 * @param decay_exponent       Double scalar, exponent for diminishing adaptation schedule.
 * @param target_acceptance    Double scalar, target acceptance rate for adaptive tuning.
 * @param min_deviation_threshold Double scalar, minimum absolute deviation from target_acceptance
 *                             required to trigger log_sigma updates. Must be >= 0.
 *
 * @note Model is local trend (random walk + trend component).
 * @note Uses sliding window memory optimization.
 * @note Configurable threshold allows fine-tuned adaptation sensitivity.
 * @note Recommended threshold: 1.0/lag_update for practical applications.
 *
 * @warning min_deviation_threshold must be >= 0.0.
 * @warning lag_update must be > 0 for sliding window indexing.
 *
 * @see adapt_cwmh_parameters
 * @see cwmh_alpha_logit_binomial
 * @since version 1.2
 */
void generate_alpha_logit_binomial(double *theta_1,
                                   double *theta_2,
                                   double *theta_01,
                                   double *theta_02,
                                   double *theta_1_updated,
                                   double *alpha,
                                   double *prec_theta_1,
                                   double *y,
                                   double *accept_prop,
                                   double *log_sigma,
                                   double *hat_theta_1,
                                   double *theta_1_new,
                                   double *log_accept_prob,
                                   int     lag_update,
                                   double  n_trials,
                                   int     n,
                                   int     iter,
                                   double  max_step_size,
                                   double  base_adaptation_rate,
                                   double  decay_exponent,
                                   double  target_acceptance,
                                   double  min_deviation_threshold);

/**
 * @brief Gibbs sampler for theta_1 in a probit-Bernoulli local level model
 *        using Albert-Chib data augmentation.
 *
 * @details Implements Gibbs sampling for the level state vector theta_1 in a
 *          Bernoulli observation model with probit link:
 *          y_t ~ Bernoulli(alpha_t),
 *          where alpha_t = Phi(theta_{t,1}) and Phi is the standard normal CDF.
 *
 *          State equation (random walk):
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1},
 *          with u_{t,1} ~ N(0, 1/prec_theta_1).
 *
 *          **Albert-Chib Data Augmentation:**
 *          Introduces latent variables v_t ~ N(theta_{t,1}, 1) such that:
 *          - y_t = 1 if v_t > 0
 *          - y_t = 0 if v_t <= 0
 *
 *          **Algorithm:**
 *          The full conditional posterior for theta_1 given latent variables v is:
 *          theta_1 | v, [...] ~ N(mu_posterior, Sigma_posterior)
 *          where Sigma_posterior^{-1} = I + prec_theta_1 * H'H (tridiagonal precision)
 *                mu_posterior = Sigma_posterior * [v + prec_theta_1 * theta_01 * e_1]
 *
 *          The sampler proceeds by drawing latent variables from truncated normals
 *          conditional on current theta_1 values, constructing the right-hand side
 *          vector for the linear system, and sampling theta_1 from its multivariate
 *          normal full conditional using generate_normal_vector.
 *
 * @param theta_1              Matrix of level states (vectorized B x n), input/output.
 * @param theta_01             Vector of initial level states (size B).
 * @param alpha                Matrix of transformed probabilities (vectorized B x n), output.
 *                             Only computed if compute_alpha is non-zero.
 * @param prec_theta_1         Vector of level precision parameters (size B).
 * @param y                    Vector of observed Bernoulli outcomes (size n).
 *                             Each y[k] must be exactly 0 or 1.
 * @param rhs_vector           Working vector for right-hand side of linear system (size n).
 * @param n                    Length of the time series.
 * @param iter                 Current MCMC iteration (0-based).
 * @param compute_alpha        Flag to control alpha transformation (0 = skip, non-zero = compute).
 *                             Set to 0 during burn-in or when probability scale values are not needed
 *                             for inference.
 *
 * @note Complexity: O(n) per iteration exploiting tridiagonal structure.
 * @note Acceptance rate is always 1.0 (Gibbs sampling).
 * @note Model assumes local level without trend component.
 * @note For maximum efficiency, set compute_alpha = 0 during burn-in or when
 *       alpha values are not required for inference.
 *
 * @warning Each y[t] must be exactly 0 or 1 (Bernoulli outcomes).
 * @warning Results are invalid if theta_01 or prec_theta_1 do not contain
 *          sufficient history (iter < 1).
 * @warning n must be > 0.
 *
 * @see Albert & Chib (1993). Bayesian Analysis of Binary and Polychotomous Response Data.
 *      JASA, 88(422), 669-679. https://doi.org/10.1080/01621459.1993.10476321
 * @see generate_normal_vector
 * @see rtruncnorm
 */
void generate_alpha_probit_bernoulli_locallevel(double *theta_1,
                                                double *theta_01,
                                                double *alpha,
                                                double *prec_theta_1,
                                                double *y,
                                                double *rhs_vector,
                                                int     n,
                                                int     iter,
                                                int     compute_alpha);

/**
 * @brief Gibbs sampler for theta_1 in a probit-Bernoulli local trend model
 *        using Albert-Chib data augmentation.
 *
 * @details Implements Gibbs sampling for the level state vector theta_1 in a
 *          Bernoulli observation model with probit link and local trend dynamics:
 *          y_t ~ Bernoulli(alpha_t),
 *          where alpha_t = Phi(theta_{t,1}) and Phi is the standard normal CDF.
 *
 *          State equations:
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},
 *          with u_{t,1} ~ N(0, 1/prec_theta_1), u_{t,2} ~ N(0, 1/prec_theta_2).
 *
 *          **Albert-Chib Data Augmentation:**
 *          Introduces latent variables v_t ~ N(theta_{t,1}, 1) such that:
 *          - y_t = 1 if v_t > 0
 *          - y_t = 0 if v_t <= 0
 *
 *          **Algorithm:**
 *          The full conditional posterior for theta_1 given latent variables v is:
 *          theta_1 | v, theta_2, [...] ~ N(mu_posterior, Sigma_posterior)
 *          where Sigma_posterior^{-1} = I + prec_theta_1 * H'H (tridiagonal precision)
 *                mu_posterior = Sigma_posterior * [v + prec_theta_1 * (theta_01 + theta_02) * e_1
 *                                                   + prec_theta_1 * H'B * theta_2]
 *
 *          The sampler proceeds by drawing latent variables from truncated normals
 *          conditional on current theta_1 values, constructing the right-hand side
 *          vector for the linear system accounting for trend contributions, and
 *          sampling theta_1 from its multivariate normal full conditional using
 *          generate_normal_vector.
 *
 * @param theta_1              Matrix of level states (vectorized B x n), input/output.
 * @param theta_2              Matrix of trend states (vectorized B x n), input only.
 *                             Must contain valid values for current iteration.
 * @param theta_01             Vector of initial level states (size B).
 * @param theta_02             Vector of initial trend states (size B).
 * @param alpha                Matrix of transformed probabilities (vectorized B x n), output.
 *                             Only computed if compute_alpha is non-zero.
 * @param prec_theta_1         Vector of level precision parameters (size B).
 * @param y                    Vector of observed Bernoulli outcomes (size n).
 *                             Each y[t] must be exactly 0 or 1.
 * @param rhs_vector           Working vector for right-hand side of linear system (size n).
 * @param n                    Length of the time series.
 * @param iter                 Current MCMC iteration (0-based).
 * @param compute_alpha        Flag to control alpha transformation (0 = skip, non-zero = compute).
 *                             Set to 0 during burn-in or when probability scale values are not
 *                             needed for inference.
 *
 * @note Complexity: O(n) per iteration exploiting tridiagonal structure.
 * @note Acceptance rate is always 1.0 (Gibbs sampling).
 * @note Model includes local trend component.
 * @note The trend theta_2 must be already sampled in the Gibbs cycle before calling this function.
 * @note For maximum efficiency, set compute_alpha = 0 during burn-in or when
 *       alpha values are not required for inference.
 *
 * @warning Each y[t] must be exactly 0 or 1 (Bernoulli outcomes).
 * @warning Results are invalid if theta_01, theta_02, or prec_theta_1 do not contain
 *          sufficient history (iter < 1).
 * @warning Theta_2 matrix must contain valid values for current iteration.
 * @warning n must be > 0.
 *
 * @see Albert & Chib (1993). Bayesian Analysis of Binary and Polychotomous Response Data.
 *      JASA, 88(422), 669-679. https://doi.org/10.1080/01621459.1993.10476321
 * @see generate_normal_vector
 * @see rtruncnorm
 * @see generate_alpha_probit_bernoulli_locallevel
 */
void generate_alpha_probit_bernoulli(double *theta_1,
                                     double *theta_2,
                                     double *theta_01,
                                     double *theta_02,
                                     double *alpha,
                                     double *prec_theta_1,
                                     double *y,
                                     double *rhs_vector,
                                     int     n,
                                     int     iter,
                                     int     compute_alpha);

#endif /* GENERATE_ALPHA_BINOMIAL_H */
