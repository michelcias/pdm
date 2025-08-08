#ifndef CWMH_ADAPTIVE_H
#define CWMH_ADAPTIVE_H

/**
 * Adapts CWMH (Component-Wise Metropolis-Hastings) parameters by updating
 * acceptance rates and log-sigma values for proposal variance tuning.
 *
 * This function implements an adaptive MCMC algorithm that automatically adjusts
 * the proposal variance (sigma) for each component of the parameter vector to
 * achieve user-specified target acceptance rates.
 *
 * The adaptation follows the diminishing adaptation principle:
 * - Uses a sliding window of recent iterations to compute acceptance rates
 * - Adjusts log_sigma based on deviation from target acceptance rate
 * - Adaptation step size decreases over iterations to ensure convergence
 *
 * @param theta_updated          Matrix of update indicators (1 if accepted, 0 if rejected).
 *                               Stored as vectorized (B x n) matrix where B = total iterations.
 * @param accrate                Output vector (size n) of acceptance rates for each component.
 * @param log_sigma              Input/output vector (size n) of log proposal standard deviations.
 *                               Will be updated based on current acceptance rates.
 * @param lag_update             Number of recent iterations to use for acceptance rate calculation.
 *                               Should be large enough for stable estimates (e.g., 50-100).
 * @param n                      Number of components in the parameter vector.
 * @param iter                   Current MCMC iteration (1-based). Must be > lag_update.
 * @param max_step_size          Maximum adaptation step size to ensure stability (e.g., 0.01).
 * @param base_adaptation_rate   Base adaptation rate controlling initial adaptation intensity.
 * @param decay_exponent         Exponent controlling decay speed (0.5 = sqrt decay, 1.0 = linear).
 * @param target_acceptance      Target acceptance rate for optimization (e.g., 0.44 for univariate,
 *                               0.234 for multivariate, 0.6 for aggressive burn-in).
 *
 * @note This function assumes iter > lag_update to have sufficient history for adaptation.
 * @note Common target acceptance rates: 0.44 (univariate MH), 0.234 (multivariate MH),
 *       0.6 (burn-in phase), 0.2-0.3 (high-dimensional problems).
 * @note Common values: max_step_size = 0.01, base_adaptation_rate = 1.0-10.0,
 *       decay_exponent = 0.3-0.8 (0.5 for classic Robbins-Monro).
 */
void adapt_cwmh_parameters(double *theta_updated,
                           double *accrate,
                           double *log_sigma,
                           int lag_update,
                           int n,
                           int iter,
                           double max_step_size,
                           double base_adaptation_rate,
                           double decay_exponent,
                           double target_acceptance);

#endif /* CWMH_ADAPTIVE_H */
