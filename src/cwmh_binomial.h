#ifndef CWMH_BINOMIAL_H
#define CWMH_BINOMIAL_H

/**
 * Orchestrates the adaptive CWMH update for theta_1 under a logit-binomial local level model.
 *
 * This routine coordinates adaptive proposal tuning with component-wise Metropolis–Hastings
 * sampling for the level state vector theta_1 when the observation equation follows:
 * y_t ~ Binomial(n_trials, alpha_t), where alpha_t = logit^{-1}(theta_{1,t}).
 *
 * The state equation is a simple random walk: theta_{1,t} = theta_{1,t-1} + u_{1,t},
 * where u_{1,t} ~ N(0, 1/prec_theta_1). This is simpler than the local trend model
 * as it excludes the trend component entirely.
 *
 * This routine glues together:
 * - Adaptive proposal tuning (log_sigma) via recent acceptance rates (accrate)
 * - Component-wise Metropolis–Hastings update for theta_1 (nonlinear observation with logit link)
 *
 * The adaptation follows a diminishing adaptation schedule and is executed periodically
 * over a sliding window of size lag_update. The actual state update is delegated to
 * CWMH_alpha_logit_binomial_locallevel, which handles boundary conditions and log-acceptance.
 *
 * Notes on iteration indexing and dependencies:
 * - CWMH_alpha_logit_binomial_locallevel internally uses prev_iter = iter - 1 for theta_01 and prec_theta_1,
 *   so this function should only be called for iter >= 1 (i.e., after an initial iteration exists).
 * - The array theta_1 is stored as vectorized (B x n) matrix by iteration blocks.
 *
 * Adaptation cadence:
 * - Performed when iter > lag_update and (iter - 1) % lag_update == 0,
 *   i.e., at iterations 1, 1 + lag_update, 1 + 2*lag_update, ...
 *
 * @param theta_1                Matrix of level states (vectorized B x n), input/output.
 * @param theta_01               Vector of initial level states (size B).
 * @param theta_1_updated        Matrix of acceptance indicators (vectorized B x n), output.
 * @param alpha                  Matrix of transformed probabilities (vectorized B x n), output.
 *                               Each alpha[t] = logit^{-1}(theta_1[t]) represents the binomial
 *                               success probability at time t.
 * @param prec_theta_1           Vector of level precision parameters (size B).
 * @param y                      Vector of observed binomial counts (size n).
 *                               Each y[k] must satisfy 0 ≤ y[k] ≤ n_trials.
 * @param accrate                Vector of acceptance rates per component (size n), output.
 * @param log_sigma              Vector of log proposal standard deviations (size n), input/output.
 * @param hat_theta_1            Temporary vector for conditional means (size n).
 * @param theta_1_new            Temporary vector for proposed values (size n).
 * @param log_accept_prob        Temporary vector for log acceptance probabilities (size n).
 * @param updated                Temporary vector for acceptance indicators (size n).
 * @param lag_update             Window length for acceptance-rate estimation and adaptation cadence.
 * @param n_trials               Number of Bernoulli trials for binomial distribution (integer).
 * @param n                      Length of the time series.
 * @param iter                   Current MCMC iteration (0-based, must satisfy iter >= 1 here).
 * @param max_step_size          Maximum per-adaptation change in log_sigma (stability cap).
 * @param base_adaptation_rate   Initial intensity of adaptation before decay.
 * @param decay_exponent         Controls decay of adaptation over iterations (e.g., 0.5 for 1/sqrt(t)).
 * @param target_acceptance      Target acceptance rate guiding proposal scale (e.g., 0.44).
 *
 * @note This function requires iter >= 1 due to internal dependencies in CWMH_alpha_logit_binomial_locallevel.
 * @note Adaptation follows a diminishing schedule: intensity ∝ base_adaptation_rate / iter^decay_exponent.
 * @note The function safely handles lag_update = 0 (no adaptation) and lag_update > 0 (periodic adaptation).
 * @note Memory layout: All matrices are stored as vectorized (B x n) blocks by iteration.
 * @note Model specification: This implements a local level model (random walk) without
 *       trend components, making it computationally simpler than the local trend version.
 */
void generate_alpha_logit_binomial_locallevel(double *theta_1,
                                              double *theta_01,
                                              double *theta_1_updated,
                                              double *alpha,
                                              double *prec_theta_1,
                                              double *y,
                                              double *accrate,
                                              double *log_sigma,
                                              double *hat_theta_1,
                                              double *theta_1_new,
                                              double *log_accept_prob,
                                              int *updated,
                                              int lag_update,
                                              double n_trials,
                                              int n,
                                              int iter,
                                              double max_step_size,
                                              double base_adaptation_rate,
                                              double decay_exponent,
                                              double target_acceptance);

//----------------------------------------------------------------------

/**
 * Orchestrates the adaptive CWMH update for theta_1 under a logit-binomial observation model.
 *
 * This routine coordinates adaptive proposal tuning with component-wise Metropolis–Hastings
 * sampling for the level state vector theta_1 when the observation equation follows:
 * y_t ~ Binomial(n_trials, alpha_t), where alpha_t = logit^{-1}(theta_{1,t}).
 *
 * The state equation is: theta_{1,t} = theta_{1,t-1} + theta_{2,t-1} + u_{1,t}.
 *
 * This routine glues together:
 * - Adaptive proposal tuning (log_sigma) via recent acceptance rates (accrate)
 * - Component-wise Metropolis–Hastings update for theta_1 (nonlinear observation with logit link)
 *
 * The adaptation follows a diminishing adaptation schedule and is executed periodically
 * over a sliding window of size lag_update. The actual state update is delegated to
 * CWMH_alpha_logit_binomial, which handles boundary conditions and log-acceptance.
 *
 * Notes on iteration indexing and dependencies:
 * - CWMH_alpha_logit_binomial internally uses prev_iter = iter - 1 for theta_01 and prec_theta_1,
 *   so this function should only be called for iter >= 1 (i.e., after an initial iteration exists).
 * - The arrays theta_1 and theta_2 are stored as vectorized (B x n) matrices by iteration blocks.
 *
 * Adaptation cadence:
 * - Performed when iter > lag_update and (iter - 1) % lag_update == 0,
 *   i.e., at iterations 1, 1 + lag_update, 1 + 2*lag_update, ...
 *
 * @param theta_1                Matrix of level states (vectorized B x n), input/output.
 * @param theta_2                Matrix of trend states (vectorized B x n), input only.
 * @param theta_01               Vector of initial level states (size B).
 * @param theta_02               Vector of initial trend states (size B).
 * @param theta_1_updated        Matrix of acceptance indicators (vectorized B x n), output.
 * @param alpha                  Matrix of transformed probabilities (vectorized B x n), output.
 *                               Each alpha[t] = logit^{-1}(theta_1[t]) represents the binomial
 *                               success probability at time t.
 * @param prec_theta_1           Vector of level precision parameters (size B).
 * @param y                      Vector of observed binomial counts (size n).
 *                               Each y[k] must satisfy 0 ≤ y[k] ≤ n_trials.
 * @param accrate                Vector of acceptance rates per component (size n), output.
 * @param log_sigma              Vector of log proposal standard deviations (size n), input/output.
 * @param hat_theta_1            Temporary vector for conditional means (size n).
 * @param theta_1_new            Temporary vector for proposed values (size n).
 * @param log_accept_prob        Temporary vector for log acceptance probabilities (size n).
 * @param updated                Temporary vector for acceptance indicators (size n).
 * @param lag_update             Window length for acceptance-rate estimation and adaptation cadence.
 * @param n_trials               Number of Bernoulli trials for binomial distribution (integer).
 * @param n                      Length of the time series.
 * @param iter                   Current MCMC iteration (0-based, must satisfy iter >= 1 here).
 * @param max_step_size          Maximum per-adaptation change in log_sigma (stability cap).
 * @param base_adaptation_rate   Initial intensity of adaptation before decay.
 * @param decay_exponent         Controls decay of adaptation over iterations (e.g., 0.5 for 1/sqrt(t)).
 * @param target_acceptance      Target acceptance rate guiding proposal scale (e.g., 0.44).
 *
 * @note This function requires iter >= 1 due to internal dependencies in CWMH_alpha_logit_binomial.
 * @note Adaptation follows a diminishing schedule: intensity ∝ base_adaptation_rate / iter^decay_exponent.
 * @note The function safely handles lag_update = 0 (no adaptation) and lag_update > 0 (periodic adaptation).
 * @note Memory layout: All matrices are stored as vectorized (B x n) blocks by iteration.
 */
void generate_alpha_logit_binomial(double *theta_1,
                                   double *theta_2,
                                   double *theta_01,
                                   double *theta_02,
                                   double *theta_1_updated,
                                   double *alpha,
                                   double *prec_theta_1,
                                   double *y,
                                   double *accrate,
                                   double *log_sigma,
                                   double *hat_theta_1,
                                   double *theta_1_new,
                                   double *log_accept_prob,
                                   int *updated,
                                   int lag_update,
                                   double n_trials,
                                   int n,
                                   int iter,
                                   double max_step_size,
                                   double base_adaptation_rate,
                                   double decay_exponent,
                                   double target_acceptance);

#endif /* CWMH_BINOMIAL_H */
