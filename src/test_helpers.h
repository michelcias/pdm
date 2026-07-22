#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <Rinternals.h>

/* Note: the scalar/length validation helpers (ensure_length,
 * require_int_scalar, require_real_scalar) are file-local helpers defined
 * with internal linkage in test_helpers.c. They are intentionally NOT
 * declared here: a `static` declaration in this shared header would create
 * an unused, undefined static symbol in every translation unit that
 * includes it (e.g. init.c), triggering -Wunused-function warnings. */

/**
 * @brief Evaluate the inverse-logit transform for a scalar value.
 *
 * @details Provides a thin wrapper around the internal @c ilogit helper so
 *          that tests can confirm numerical behaviour through R's @c .Call
 *          interface.
 *
 * @param x_  Numeric vector whose first element is transformed.
 *
 * @return A length-one numeric vector containing @f$\mathrm{logit}^{-1}(x)@f$.
 */
SEXP test_ilogit(SEXP x_);

/**
 * @brief Draw floored Gamma variates through the shared rgamma_positive helper.
 *
 * @details Exposes @c rgamma_positive so that tests can confirm the
 *          DBL_EPSILON floor applied to precision draws.
 *
 * @param shape_  Length-one numeric vector with the Gamma shape parameter.
 * @param scale_  Length-one numeric vector with the Gamma scale parameter.
 * @param n_      Length-one integer vector with the number of draws.
 *
 * @return A numeric vector of @p n_ Gamma draws, each floored at DBL_EPSILON.
 */
SEXP test_rgamma_positive(SEXP shape_, SEXP scale_, SEXP n_);

/**
 * @brief Draw a normal vector used when simulating latent states.
 *
 * @details Calls the shared random number helper to produce conditionally
 *          independent normal draws centred around @p y_ with optional shift.
 *
 * @param y_      Baseline numeric vector used as the proposal centre.
 * @param a_      Scalar location shift applied during proposal generation.
 * @param b_      Scalar precision parameter governing variability.
 * @param add_a_  Integer flag indicating whether @p a_ should be added element-wise.
 *
 * @return A numeric vector of the same length as @p y_ containing the simulated values.
 */
SEXP test_generate_normal_vector(SEXP y_,
                                 SEXP a_,
                                 SEXP b_,
                                 SEXP add_a_);

/**
 * @brief Run a single adaptive update of the CWMH proposal parameters.
 *
 * @details Exposes @c adapt_cwmh_parameters to R so that tests can verify the
 *          Robbins--Monro adaptation schedule. Returns both acceptance
 *          proportions and updated log standard deviations.
 *
 * @param theta_updated_           Matrix (stored as a vector) containing the most recent
 *                                  lagged theta values.
 * @param log_sigma_               Numeric vector with the current log standard deviations.
 * @param lag_update_              Integer scalar specifying the adaptation window length.
 * @param n_                       Integer scalar representing the dimensionality of the parameter.
 * @param iter_                    Integer scalar with the current iteration index.
 * @param max_step_size_           Double scalar limiting per-iteration sigma growth.
 * @param base_adaptation_rate_    Double scalar providing the adaptation learning rate.
 * @param decay_exponent_          Double scalar controlling how fast learning decays.
 * @param target_acceptance_       Double scalar giving the desired acceptance probability.
 * @param min_deviation_threshold_ Double scalar setting the minimum deviation threshold.
 *
 * @return A named list with updated acceptance proportions and log standard deviations.
 *
 * @note Requires @p theta_updated_ to have length @p lag_update_ * @p n_.
 */
SEXP test_adapt_cwmh_parameters(SEXP theta_updated_,
                                SEXP log_sigma_,
                                SEXP lag_update_,
                                SEXP n_,
                                SEXP iter_,
                                SEXP max_step_size_,
                                SEXP base_adaptation_rate_,
                                SEXP decay_exponent_,
                                SEXP target_acceptance_,
                                SEXP min_deviation_threshold_);

/**
 * @brief Reset the global cache used by adaptive CWMH routines between runs.
 *
 * @details Provides tests with explicit control over the shared adaptation
 *          cache to ensure deterministic behaviour.
 *
 * @return R's @c NULL value.
 */
SEXP reset_adaptation_cache_wrapper(void);

/**
 * @brief Sample the conditional data precision for the first time slice.
 *
 * @details Wraps @c generate_precision_data to validate arguments before
 *          sampling from the Gamma full conditional of the observation
 *          precision.
 *
 * @param y_        Observed data vector.
 * @param theta_1_  Latent state vector for time one.
 * @param nu_y_     Shape hyperparameter of the Gamma prior.
 * @param eta_y_    Rate hyperparameter of the Gamma prior.
 *
 * @return A length-one numeric vector containing the sampled precision.
 */
SEXP test_generate_precision_data(SEXP y_,
                                  SEXP theta_1_,
                                  SEXP nu_y_,
                                  SEXP eta_y_);

/**
 * @brief Sample the conditional precision for an interior latent state.
 *
 * @details Verifies length compatibility before sampling the Gamma
 *          conditional precision governing @f$\theta_k@f$ transitions.
 *
 * @param theta_0k_    Scalar prior mean for @f$\theta_k@f$.
 * @param theta_0kp1_  Scalar prior mean for @f$\theta_{k+1}@f$.
 * @param theta_k_     Numeric vector with the current state draws at @f$k@f$.
 * @param theta_kp1_   Numeric vector with the state draws at @f$k+1@f$.
 * @param nu_0k_       Shape hyperparameter for the precision prior.
 * @param eta_0k_      Rate hyperparameter for the precision prior.
 *
 * @return A length-one numeric vector containing the sampled precision.
 */
SEXP test_generate_precision_theta_k(SEXP theta_0k_,
                                     SEXP theta_0kp1_,
                                     SEXP theta_k_,
                                     SEXP theta_kp1_,
                                     SEXP nu_0k_,
                                     SEXP eta_0k_);

/**
 * @brief Sample the conditional precision for the final latent state.
 *
 * @details Ensures consistency between arguments before invoking the Gamma
 *          conditional sampler for @f$\theta_p@f$.
 *
 * @param theta_0p_  Scalar prior mean for @f$\theta_p@f$.
 * @param theta_p_   Numeric vector with the terminal state draws.
 * @param nu_0p_     Shape hyperparameter for the precision prior.
 * @param eta_0p_    Rate hyperparameter for the precision prior.
 *
 * @return A length-one numeric vector containing the sampled precision.
 */
SEXP test_generate_precision_theta_p(SEXP theta_0p_,
                                     SEXP theta_p_,
                                     SEXP nu_0p_,
                                     SEXP eta_0p_);

/**
 * @brief Update the Half-t scale-mixture auxiliary variable b = 1/a.
 *
 * @details Thin R-facing wrapper around ::generate_halft_aux used to unit-test
 *          the auxiliary draw of the Half-t (Half-Cauchy when df = 1) prior.
 *
 * @param prec_      Scalar current precision W^{-1} (> 0).
 * @param hc_scale_  Scalar Half-t scale hyperparameter A (> 0).
 * @param df_        Scalar Half-t degrees of freedom nu (> 0).
 *
 * @return A length-one numeric vector containing the sampled auxiliary b = 1/a.
 */
SEXP test_generate_halft_aux(SEXP prec_,
                             SEXP hc_scale_,
                             SEXP df_);

/**
 * @brief Sample the first latent state for the local level model.
 *
 * @details Performs argument coercion and length extraction prior to sampling
 *          the normal conditional posterior for @f$\theta_1@f$ in the local
 *          level model.
 *
 * @param data_          Observed data vector.
 * @param prec_data_     Scalar precision associated with the observation model.
 * @param prec_theta_1_  Scalar prior precision for the first state.
 * @param theta_01_      Prior mean for the first state.
 *
 * @return A numeric vector containing sampled values for @f$\theta_1@f$.
 */
SEXP test_generate_theta_1_locallevel(SEXP data_,
                                      SEXP prec_data_,
                                      SEXP prec_theta_1_,
                                      SEXP theta_01_);

/**
 * @brief Sample the first latent state for the dynamic binomial model.
 *
 * @details Extends the local level wrapper by incorporating the companion
 *          state @f$\theta_2@f$ when forming the conditional posterior.
 *
 * @param data_          Observed data vector.
 * @param theta_2_       Numeric vector containing draws for @f$\theta_2@f$.
 * @param prec_data_     Scalar precision associated with the observation model.
 * @param prec_theta_1_  Scalar prior precision for @f$\theta_1@f$.
 * @param theta_01_      Prior mean for @f$\theta_1@f$.
 * @param theta_02_      Prior mean for @f$\theta_2@f$.
 *
 * @return A numeric vector containing sampled values for @f$\theta_1@f$.
 */
SEXP test_generate_theta_1(SEXP data_,
                           SEXP theta_2_,
                           SEXP prec_data_,
                           SEXP prec_theta_1_,
                           SEXP theta_01_,
                           SEXP theta_02_);

/**
 * @brief Sample an interior latent state for the dynamic binomial model.
 *
 * @details Applies validation before calling the Gaussian conditional sampler
 *          for interior states in the dynamic model.
 *
 * @param theta_km1_  Numeric vector with draws at @f$k-1@f$.
 * @param theta_kp1_  Numeric vector with draws at @f$k+1@f$.
 * @param prec_km1_   Scalar precision for the backward transition.
 * @param prec_k_     Scalar precision for the forward transition.
 * @param theta_0k_   Prior mean for @f$\theta_k@f$.
 * @param theta_0kp1_ Prior mean for @f$\theta_{k+1}@f$.
 *
 * @return A numeric vector containing sampled values for @f$\theta_k@f$.
 */
SEXP test_generate_theta_k(SEXP theta_km1_,
                           SEXP theta_kp1_,
                           SEXP prec_km1_,
                           SEXP prec_k_,
                           SEXP theta_0k_,
                           SEXP theta_0kp1_);

/**
 * @brief Sample the terminal latent state for the dynamic binomial model.
 *
 * @details Finishes the state trajectory by sampling @f$\theta_p@f$ from its
 *          conditional distribution given @f$\theta_{p-1}@f$ and precision
 *          parameters.
 *
 * @param theta_pm1_  Numeric vector with draws at @f$p-1@f$.
 * @param prec_pm1_   Scalar precision for the backward transition.
 * @param prec_p_     Scalar precision for the forward transition.
 * @param theta_0p_   Prior mean for @f$\theta_p@f$.
 *
 * @return A numeric vector containing sampled values for @f$\theta_p@f$.
 */
SEXP test_generate_theta_p(SEXP theta_pm1_,
                           SEXP prec_pm1_,
                           SEXP prec_p_,
                           SEXP theta_0p_);

/**
 * @brief Sample the prior mean of the first latent state in the local level model.
 *
 * @details Handles coercion and validation before invoking the Gaussian
 *          conditional sampler for the hyper-mean @f$\theta_{0,1}@f$.
 *
 * @param theta_1_        Numeric vector of first-state draws.
 * @param prec_theta_1_   Scalar precision for the first state.
 * @param mean_theta_01_  Prior mean hyperparameter.
 * @param prec_theta_01_  Prior precision hyperparameter.
 *
 * @return A length-one numeric vector with the sampled @f$\theta_{0,1}@f$.
 */
SEXP test_generate_theta_01_locallevel(SEXP theta_1_,
                                       SEXP prec_theta_1_,
                                       SEXP mean_theta_01_,
                                       SEXP prec_theta_01_);

/**
 * @brief Sample the prior mean of the first latent state in the full dynamic model.
 *
 * @details Includes dependence on @f$\theta_2@f$ when generating the
 *          conditional posterior for the hyper-mean @f$\theta_{0,1}@f$.
 *
 * @param theta_1_        Numeric vector of first-state draws.
 * @param theta_02_       Prior mean hyperparameter for @f$\theta_2@f$.
 * @param prec_theta_1_   Scalar precision for the first state.
 * @param mean_theta_01_  Prior mean hyperparameter.
 * @param prec_theta_01_  Prior precision hyperparameter.
 *
 * @return A length-one numeric vector with the sampled @f$\theta_{0,1}@f$.
 */
SEXP test_generate_theta_01(SEXP theta_1_,
                            SEXP theta_02_,
                            SEXP prec_theta_1_,
                            SEXP mean_theta_01_,
                            SEXP prec_theta_01_);

/**
 * @brief Sample the prior mean for an interior latent state.
 *
 * @details Validates neighbour lengths before sampling the conditional normal
 *          distribution for the hyper-mean @f$\theta_{0,k}@f$.
 *
 * @param theta_km1_   Numeric vector with draws at @f$k-1@f$.
 * @param theta_k_     Numeric vector with draws at @f$k@f$.
 * @param theta_0km1_  Scalar prior mean for @f$\theta_{k-1}@f$.
 * @param theta_0kp1_  Scalar prior mean for @f$\theta_{k+1}@f$.
 * @param prec_km1_    Scalar precision for the backward transition.
 * @param prec_k_      Scalar precision for the forward transition.
 * @param mean_0k_     Scalar prior mean hyperparameter.
 * @param prec_0k_     Scalar prior precision hyperparameter.
 *
 * @return A length-one numeric vector with the sampled @f$\theta_{0,k}@f$.
 */
SEXP test_generate_theta_0k(SEXP theta_km1_,
                            SEXP theta_k_,
                            SEXP theta_0km1_,
                            SEXP theta_0kp1_,
                            SEXP prec_km1_,
                            SEXP prec_k_,
                            SEXP mean_0k_,
                            SEXP prec_0k_);

/**
 * @brief Sample the prior mean for the terminal latent state.
 *
 * @details Ensures terminal vectors align before drawing the conditional
 *          Gaussian hyper-mean @f$\theta_{0,p}@f$.
 *
 * @param theta_pm1_   Numeric vector with draws at @f$p-1@f$.
 * @param theta_p_     Numeric vector with draws at @f$p@f$.
 * @param theta_0pm1_  Scalar prior mean for @f$\theta_{p-1}@f$.
 * @param prec_pm1_    Scalar precision for the backward transition.
 * @param prec_p_      Scalar precision for the forward transition.
 * @param mean_0p_     Scalar prior mean hyperparameter.
 * @param prec_0p_     Scalar prior precision hyperparameter.
 *
 * @return A length-one numeric vector with the sampled @f$\theta_{0,p}@f$.
 */
SEXP test_generate_theta_0p(SEXP theta_pm1_,
                            SEXP theta_p_,
                            SEXP theta_0pm1_,
                            SEXP prec_pm1_,
                            SEXP prec_p_,
                            SEXP mean_0p_,
                            SEXP prec_0p_);

/**
 * @brief Run a single CWMH update for the logit-binomial local level model.
 *
 * @details Provides deterministic wrappers around the production sampler by
 *          setting small adaptation windows and exposing the resulting states
 *          and acceptance diagnostics.
 *
 * @param theta_1_in_     Numeric vector with the previous state draws.
 * @param theta_01_in_    Scalar prior mean for the initial state.
 * @param prec_theta1_in_      Scalar prior precision for the state.
 * @param y_              Observed binomial counts.
 * @param n_trials_       Scalar number of trials for the binomial likelihood.
 * @param log_sigma_in_   Numeric vector of proposal log standard deviations.
 *
 * @return A list containing updated state draws and logit-scale alphas.
 */
SEXP test_cwmh_alpha_logit_binomial_locallevel(SEXP theta_1_in_,
                                               SEXP theta_01_in_,
                                               SEXP prec_theta1_in_,
                                               SEXP y_,
                                               SEXP n_trials_,
                                               SEXP log_sigma_in_);

/**
 * @brief Run a single CWMH update for the two-parameter logit-binomial model.
 *
 * @details Mirrors the production sampler while keeping proposals fixed so
 *          that unit tests can inspect latent states and alpha values.
 *
 * @param theta_1_in_   Numeric vector with the previous state draws.
 * @param theta_2_in_   Numeric vector with the companion state draws.
 * @param theta_01_in_  Scalar prior mean for the first state.
 * @param theta_02_in_  Scalar prior mean for the second state.
 * @param prec_theta1_in_    Scalar prior precision for the first state.
 * @param y_            Observed binomial counts.
 * @param n_trials_     Scalar number of trials for the binomial likelihood.
 *
 * @return A list containing updated state draws and logit-scale alphas.
 */
SEXP test_cwmh_alpha_logit_binomial(SEXP theta_1_in_,
                                    SEXP theta_2_in_,
                                    SEXP theta_01_in_,
                                    SEXP theta_02_in_,
                                    SEXP prec_theta1_in_,
                                    SEXP y_,
                                    SEXP n_trials_);

/**
 * @brief Adaptively generate alphas for the logit-binomial local level model.
 *
 * @details Runs @c generate_alpha_logit_binomial_locallevel with predetermined
 *          tuning constants so that tests can confirm adaptive behaviour.
 *
 * @param theta_1_in_  Numeric vector with the previous state draws.
 * @param theta_01_in_ Scalar prior mean for the initial state.
 * @param prec_theta1_in_   Scalar prior precision for the state.
 * @param y_           Observed binomial counts.
 * @param n_trials_    Scalar number of trials for the binomial likelihood.
 *
 * @return A list containing updated state draws and logit-scale alphas.
 */
SEXP test_generate_alpha_logit_binomial_locallevel(SEXP theta_1_in_,
                                                   SEXP theta_01_in_,
                                                   SEXP prec_theta1_in_,
                                                   SEXP y_,
                                                   SEXP n_trials_);

/**
 * @brief Adaptively generate alphas for the two-parameter logit-binomial model.
 *
 * @details Runs @c generate_alpha_logit_binomial with predetermined tuning
 *          constants so that tests can confirm adaptive behaviour.
 *
 * @param theta_1_in_  Numeric vector with the previous state draws.
 * @param theta_2_in_  Numeric vector with the companion state draws.
 * @param theta_01_in_ Scalar prior mean for the first state.
 * @param theta_02_in_ Scalar prior mean for the second state.
 * @param prec_theta1_in_   Scalar prior precision for the first state.
 * @param y_           Observed binomial counts.
 * @param n_trials_    Scalar number of trials for the binomial likelihood.
 *
 * @return A list containing updated state draws and logit-scale alphas.
 */
SEXP test_generate_alpha_logit_binomial(SEXP theta_1_in_,
                                        SEXP theta_2_in_,
                                        SEXP theta_01_in_,
                                        SEXP theta_02_in_,
                                        SEXP prec_theta1_in_,
                                        SEXP y_,
                                        SEXP n_trials_);

/**
 * @brief Generate alphas for the probit-Bernoulli local level model.
 *
 * @details Wraps the probit alpha generator while supplying deterministic
 *          control flags for reproducible testing.
 *
 * @param theta_1_in_  Numeric vector with previous state draws.
 * @param theta_01_in_ Scalar prior mean for the initial state.
 * @param prec_theta1_in_   Scalar prior precision for the state.
 * @param y_           Observed Bernoulli outcomes.
 *
 * @return A list containing updated state draws and probit-scale alphas.
 */
SEXP test_generate_alpha_probit_bernoulli_locallevel(SEXP theta_1_in_,
                                                     SEXP theta_01_in_,
                                                     SEXP prec_theta1_in_,
                                                     SEXP y_);

/**
 * @brief Generate alphas for the two-parameter probit-Bernoulli model.
 *
 * @details Similar to the local level variant but incorporates the second
 *          state when computing probit link updates.
 *
 * @param theta_1_in_  Numeric vector with previous state draws.
 * @param theta_2_in_  Numeric vector with companion state draws.
 * @param theta_01_in_ Scalar prior mean for the first state.
 * @param theta_02_in_ Scalar prior mean for the second state.
 * @param prec_theta1_in_   Scalar prior precision for the first state.
 * @param y_           Observed Bernoulli outcomes.
 *
 * @return A list containing updated state draws and probit-scale alphas.
 */
SEXP test_generate_alpha_probit_bernoulli(SEXP theta_1_in_,
                                          SEXP theta_2_in_,
                                          SEXP theta_01_in_,
                                          SEXP theta_02_in_,
                                          SEXP prec_theta1_in_,
                                          SEXP y_);

/**
 * @brief Execute the full MCMC sampler for the logit-binomial local level model with parameter fixing.
 *
 * @details This function implements a full MCMC loop that conditionally fixes parameters during
 *          sampling based on which "true" parameters are provided. This is critical for testing
 *          the statistical correctness of the sampler by validating conditional distributions.
 *
 *          **Conditional sampling behavior:**
 *          - If theta_1_true is provided (not NULL): theta_1 is fixed to true values (not sampled)
 *          - If theta_01_true is provided (not NULL): theta_01 is fixed to true value (not sampled)
 *          - If prec_theta1_true is provided (not NULL): prec_theta1 is fixed to true value (not sampled)
 *          - Otherwise: parameter is sampled normally from its conditional posterior
 *
 *          **Sampling sequence per iteration (when not fixed):**
 *          1. theta_1, alpha | y, theta_01, prec_theta1 -> CWMH with adaptive tuning
 *          2. prec_theta1 | theta_1, theta_01 -> Gamma posterior
 *          3. theta_01 | theta_1, prec_theta1 -> Normal posterior
 *
 * @param y_                       Observed binomial counts [n].
 * @param n_trials_                Number of trials for each observation.
 * @param burnin_                  Number of burn-in iterations (discarded).
 * @param thinning_                Thinning interval for retained samples.
 * @param n_chain_                 Number of chains to simulate.
 * @param theta_1_true_            Optional: true theta_1 values [n] to fix (NULL = sample normally).
 * @param theta_01_true_           Optional: true theta_01 value to fix (NULL = sample normally).
 * @param prec_theta1_true_             Optional: true prec_theta1 value to fix (NULL = sample normally).
 * @param prior_theta01_mean_      Prior mean hyperparameter for theta_{0,1}.
 * @param prior_theta01_prec_      Prior precision hyperparameter for theta_{0,1}.
 * @param prior_prec1_shape_       Gamma shape hyperparameter for 1/W_1.
 * @param prior_prec1_rate_        Gamma rate hyperparameter for 1/W_1.
 * @param lag_update_              Adaptation window length (iterations).
 * @param max_step_size_           Maximum adaptation step size.
 * @param base_adaptation_rate_    Base adaptation rate.
 * @param decay_exponent_          Adaptation decay exponent.
 * @param target_acceptance_       Target acceptance probability.
 * @param return_log_sigma_        Logical flag: return log_sigma diagnostics.
 * @param return_accept_prop_      Logical flag: return accept_prop diagnostics.
 *
 * @return An R list mirroring the production sampler output.
 *
 * @note Validates binomial constraints: 0 <= y[i] <= n_trials for all i.
 * @note Validates parameter positivity: n_trials > 0, precisions > 0.
 * @note When parameters are fixed, corresponding posterior samples are constant.
 *
 * @warning This function is for testing only. Do not use for production inference.
 * @warning Fixed parameters must have correct dimensions matching observed data.
 */
SEXP test_mcmc_binomial_locallevel_fixed_params(SEXP y_,
                                                SEXP n_trials_,
                                                SEXP burnin_,
                                                SEXP thinning_,
                                                SEXP n_chain_,
                                                SEXP theta_1_true_,
                                                SEXP theta_01_true_,
                                                SEXP prec_theta1_true_,
                                                SEXP prior_theta01_mean_,
                                                SEXP prior_theta01_prec_,
                                                SEXP prior_prec1_shape_,
                                                SEXP prior_prec1_rate_,
                                                SEXP lag_update_,
                                                SEXP max_step_size_,
                                                SEXP base_adaptation_rate_,
                                                SEXP decay_exponent_,
                                                SEXP target_acceptance_,
                                                SEXP return_log_sigma_,
                                                SEXP return_accept_prop_);

/**
 * @brief Execute the full MCMC sampler for the probit-Bernoulli local level model with parameter fixing.
 *
 * @details This function implements a full MCMC loop that conditionally fixes parameters during
 *          sampling based on which "true" parameters are provided. This is critical for testing
 *          the statistical correctness of the sampler by validating conditional distributions.
 *
 *          **Conditional sampling behavior:**
 *          - If theta_1_true is provided (not NULL): theta_1 is fixed to true values (not sampled)
 *          - If theta_01_true is provided (not NULL): theta_01 is fixed to true value (not sampled)
 *          - If prec_theta1_true is provided (not NULL): prec_theta1 is fixed to true value (not sampled)
 *          - Otherwise: parameter is sampled normally from its conditional posterior
 *
 *          **Sampling sequence per iteration (when not fixed):**
 *          1. theta_1, alpha | y, theta_01, prec_theta1 -> Gibbs sampling via Albert-Chib augmentation
 *          2. prec_theta1 | theta_1, theta_01 -> Gamma posterior
 *          3. theta_01 | theta_1, prec_theta1 -> Normal posterior
 *
 * @param y_                  Observed Bernoulli outcomes [n] (0 or 1).
 * @param burnin_             Number of burn-in iterations (discarded).
 * @param thinning_           Thinning interval for retained samples.
 * @param n_chain_            Number of chains to simulate.
 * @param theta_1_true_       Optional: true theta_1 values [n] to fix (NULL = sample normally).
 * @param theta_01_true_      Optional: true theta_01 value to fix (NULL = sample normally).
 * @param prec_theta1_true_        Optional: true prec_theta1 value to fix (NULL = sample normally).
 * @param prior_theta01_mean_ Prior mean hyperparameter for theta_{0,1}.
 * @param prior_theta01_prec_ Prior precision hyperparameter for theta_{0,1}.
 * @param prior_prec1_shape_  Gamma shape hyperparameter for 1/W_1.
 * @param prior_prec1_rate_   Gamma rate hyperparameter for 1/W_1.
 *
 * @return An R list mirroring the production sampler output.
 *
 * @note Validates Bernoulli constraints: y[i] must be 0 or 1 for all i.
 * @note Validates parameter positivity: precisions > 0.
 * @note When parameters are fixed, corresponding posterior samples are constant.
 *
 * @warning This function is for testing only. Do not use for production inference.
 * @warning Fixed parameters must have correct dimensions matching observed data.
 */
SEXP test_mcmc_probit_bernoulli_locallevel_fixed_params(SEXP y_,
                                                        SEXP burnin_,
                                                        SEXP thinning_,
                                                        SEXP n_chain_,
                                                        SEXP theta_1_true_,
                                                        SEXP theta_01_true_,
                                                        SEXP prec_theta1_true_,
                                                        SEXP prior_theta01_mean_,
                                                        SEXP prior_theta01_prec_,
                                                        SEXP prior_prec1_shape_,
                                                        SEXP prior_prec1_rate_);

//==============================================================================
// POISSON MODEL TEST HELPERS
//==============================================================================

/**
 * @brief Test wrapper for component-wise MH in log-Poisson local level model
 *
 * @details Exposes cwmh_alpha_log_poisson_locallevel for testing with fixed
 *          adaptation parameters.
 *
 * @param theta_1_in_     Numeric vector with previous state draws [n].
 * @param theta_01_in_    Scalar prior mean for initial state.
 * @param prec_theta1_in_ Scalar prior precision for state.
 * @param y_              Observed Poisson counts [n].
 * @param log_sigma_in_   Numeric vector of proposal log standard deviations [n].
 *
 * @return A list containing updated state draws and log-scale rates (alpha).
 */
SEXP test_cwmh_alpha_log_poisson_locallevel(SEXP theta_1_in_,
                                            SEXP theta_01_in_,
                                            SEXP prec_theta1_in_,
                                            SEXP y_,
                                            SEXP log_sigma_in_);

/**
 * @brief Test wrapper for adaptive alpha generation in log-Poisson local level model
 *
 * @details Runs generate_alpha_log_poisson_locallevel with predetermined tuning
 *          constants for testing adaptive behavior.
 *
 * @param theta_1_in_     Numeric vector with previous state draws [n].
 * @param theta_01_in_    Scalar prior mean for initial state.
 * @param prec_theta1_in_ Scalar prior precision for state.
 * @param y_              Observed Poisson counts [n].
 *
 * @return A list containing updated state draws and log-scale rates.
 */
SEXP test_generate_alpha_log_poisson_locallevel(SEXP theta_1_in_,
                                                SEXP theta_01_in_,
                                                SEXP prec_theta1_in_,
                                                SEXP y_);

/**
 * @brief Test wrapper for component-wise MH in log-Poisson local trend model
 *
 * @details Exposes cwmh_alpha_log_poisson for testing with fixed adaptation
 *          parameters.
 *
 * @param theta_1_in_     Numeric vector with previous level state draws [n].
 * @param theta_2_in_     Numeric vector with current trend state draws [n].
 * @param theta_01_in_    Scalar prior mean for initial level.
 * @param theta_02_in_    Scalar prior mean for initial trend.
 * @param prec_theta1_in_ Scalar prior precision for level.
 * @param y_              Observed Poisson counts [n].
 *
 * @return A list containing updated level state draws and log-scale rates.
 */
SEXP test_cwmh_alpha_log_poisson(SEXP theta_1_in_,
                                 SEXP theta_2_in_,
                                 SEXP theta_01_in_,
                                 SEXP theta_02_in_,
                                 SEXP prec_theta1_in_,
                                 SEXP y_);

/**
 * @brief Test wrapper for adaptive alpha generation in log-Poisson local trend model
 *
 * @details Runs generate_alpha_log_poisson with predetermined tuning constants
 *          for testing adaptive behavior.
 *
 * @param theta_1_in_     Numeric vector with previous level state draws [n].
 * @param theta_2_in_     Numeric vector with current trend state draws [n].
 * @param theta_01_in_    Scalar prior mean for initial level.
 * @param theta_02_in_    Scalar prior mean for initial trend.
 * @param prec_theta1_in_ Scalar prior precision for level.
 * @param y_              Observed Poisson counts [n].
 *
 * @return A list containing updated level state draws and log-scale rates.
 */
SEXP test_generate_alpha_log_poisson(SEXP theta_1_in_,
                                     SEXP theta_2_in_,
                                     SEXP theta_01_in_,
                                     SEXP theta_02_in_,
                                     SEXP prec_theta1_in_,
                                     SEXP y_);

/**
 * @brief Execute full MCMC sampler for log-Poisson local level model with parameter fixing
 *
 * @details Implements full MCMC loop that conditionally fixes parameters during sampling
 *          based on which "true" parameters are provided.  Critical for testing statistical
 *          correctness by validating conditional distributions.
 *
 *          **Conditional sampling behavior:**
 *          - If theta_1_true provided: theta_1 fixed to true values (not sampled)
 *          - If theta_01_true provided: theta_01 fixed to true value (not sampled)
 *          - If prec_theta1_true provided:  prec_theta1 fixed to true value (not sampled)
 *          - Otherwise: parameter sampled normally from conditional posterior
 *
 *          **Sampling sequence per iteration (when not fixed):**
 *          1. theta_1, alpha | y, theta_01, prec_theta1 -> CWMH with adaptive tuning
 *          2. prec_theta1 | theta_1, theta_01 -> Gamma posterior
 *          3. theta_01 | theta_1, prec_theta1 -> Normal posterior
 *
 * @param y_                       Observed Poisson counts [n].
 * @param burnin_                  Number of burn-in iterations (discarded).
 * @param thinning_                Thinning interval for retained samples.
 * @param n_chain_                 Number of chains to simulate.
 * @param theta_1_true_            Optional:  true theta_1 values [n] to fix (NULL = sample).
 * @param theta_01_true_           Optional: true theta_01 value to fix (NULL = sample).
 * @param prec_theta1_true_        Optional: true prec_theta1 value to fix (NULL = sample).
 * @param prior_theta01_mean_      Prior mean hyperparameter for theta_{0,1}.
 * @param prior_theta01_prec_      Prior precision hyperparameter for theta_{0,1}.
 * @param prior_prec1_shape_       Gamma shape hyperparameter for 1/W_1.
 * @param prior_prec1_rate_        Gamma rate hyperparameter for 1/W_1.
 * @param lag_update_              Adaptation window length (iterations).
 * @param max_step_size_           Maximum adaptation step size.
 * @param base_adaptation_rate_    Base adaptation rate.
 * @param decay_exponent_          Adaptation decay exponent.
 * @param target_acceptance_       Target acceptance probability.
 * @param return_log_sigma_        Logical flag:  return log_sigma diagnostics.
 * @param return_accept_prop_      Logical flag: return accept_prop diagnostics.
 *
 * @return R list with posterior samples (matching production sampler output).
 *
 * @note Validates Poisson constraints:  y[i] >= 0 for all i.
 * @note Validates parameter positivity: precisions > 0.
 * @note When parameters fixed, corresponding posterior samples are constant.
 *
 * @warning For testing only.  Do not use for production inference.
 * @warning Fixed parameters must have correct dimensions matching observed data.
 */
SEXP test_mcmc_log_poisson_locallevel_fixed_params(SEXP y_,
                                                   SEXP burnin_,
                                                   SEXP thinning_,
                                                   SEXP n_chain_,
                                                   SEXP theta_1_true_,
                                                   SEXP theta_01_true_,
                                                   SEXP prec_theta1_true_,
                                                   SEXP prior_theta01_mean_,
                                                   SEXP prior_theta01_prec_,
                                                   SEXP prior_prec1_shape_,
                                                   SEXP prior_prec1_rate_,
                                                   SEXP lag_update_,
                                                   SEXP max_step_size_,
                                                   SEXP base_adaptation_rate_,
                                                   SEXP decay_exponent_,
                                                   SEXP target_acceptance_,
                                                   SEXP return_log_sigma_,
                                                   SEXP return_accept_prop_);

#endif /* TEST_HELPERS_H */
