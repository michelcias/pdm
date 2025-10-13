/**
 * @file test_helpers.c
 * @brief R-accessible wrappers for internal C functions used in testing
 * @details This module bridges internal C routines with R's .Call interface so
 *          that unit tests (e.g., via testthat) can validate numerical
 *          correctness and edge-case behaviour. Wrappers provide lightweight
 *          argument validation, handle RNG state transitions, and translate
 *          native results into SEXP objects without exposing implementation
 *          details in production interfaces.
 *
 * @author Michel H. Montoril
 * @date 2025-10-12
 * @version 1.6
 *
 * @changelog
 * - v1.6 (2025-10-12): Realigned every helper with refactored core routines.
 *   Updated signatures to reflect scalar-returning precision samplers,
 *   simplified state samplers using direct vectors, and refreshed binomial
 *   helpers for adaptive interfaces. Added dedicated validation helpers and
 *   clarified documentation to match the optimised codebase.
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <R_ext/Arith.h>
#include <R_ext/Utils.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

// Headers for the C functions exposed through this file
#include "utils.h"
#include "cwmh_adaptive.h"
#include "conditional_precision.h"
#include "conditional_state.h"
#include "conditional_theta0.h"
#include "cwmh_binomial.h"
#include "generate_alpha_binomial.h"
#include "mcmc_binomial_locallevel.h"

//==============================================================================
// INTERNAL UTILITIES
//==============================================================================

/**
 * @brief Ensure that a numeric vector matches the expected length.
 *
 * @param vec Candidate vector to validate.
 * @param expected Required length for @p vec.
 * @param arg Name of the argument being checked, used for error messages.
 * @param reference Name of the reference argument establishing the length, used for error
 *        reporting.
 */
static void ensure_length(SEXP vec, int expected, const char *arg, const char *reference) {
  if (LENGTH(vec) != expected) {
    error("%s must have length %d to match %s (got %d)",
          arg, expected, reference, LENGTH(vec));
  }
}

/**
 * @brief Coerce an object to integer and return the first scalar entry.
 *
 * @param x R object expected to hold an integer scalar.
 * @param arg Name of the argument being coerced, used for diagnostic messages.
 *
 * @return The first integer extracted from @p x.
 */
static int require_int_scalar(SEXP x, const char *arg) {
  SEXP tmp = PROTECT(coerceVector(x, INTSXP));
  if (LENGTH(tmp) < 1) {
    UNPROTECT(1);
    error("%s must contain at least one integer", arg);
  }
  int value = INTEGER(tmp)[0];
  UNPROTECT(1);
  return value;
}

/**
 * @brief Coerce an object to double and return the first scalar entry.
 *
 * @param x R object expected to hold a numeric scalar.
 * @param arg Name of the argument being coerced, used for diagnostic messages.
 *
 * @return The first double extracted from @p x.
 */
static double require_real_scalar(SEXP x, const char *arg) {
  SEXP tmp = PROTECT(coerceVector(x, REALSXP));
  if (LENGTH(tmp) < 1) {
    UNPROTECT(1);
    error("%s must contain at least one numeric value", arg);
  }
  double value = REAL(tmp)[0];
  UNPROTECT(1);
  return value;
}

//==============================================================================
// UTILITY FUNCTION WRAPPERS
//==============================================================================

/**
 * @brief Evaluate the inverse-logit transform for a scalar value.
 *
 * @param x_ Numeric vector whose first element is transformed.
 *
 * @return A length-one numeric vector containing @f$\mathrm{logit}^{-1}(x)@f$.
 */
SEXP test_ilogit(SEXP x_) {
  if (!isReal(x_) || LENGTH(x_) == 0) {
    error("Input must be a non-empty numeric vector");
  }
  double val = ilogit(REAL(x_)[0]);
  return ScalarReal(val);
}

/**
 * @brief Draw a normal vector used when simulating latent states.
 *
 * @param y_ Baseline numeric vector used as the proposal centre.
 * @param a_ Scalar location shift applied during proposal generation.
 * @param b_ Scalar precision parameter governing variability.
 * @param add_a_ Integer flag indicating whether @p a_ should be added element-wise.
 *
 * @return A numeric vector of the same length as @p y_ containing the simulated values.
 */
SEXP test_generate_normal_vector(SEXP y_, SEXP a_, SEXP b_, SEXP add_a_) {
  int protect_count = 0;
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  SEXP a = PROTECT(coerceVector(a_, REALSXP));
  protect_count++;
  SEXP b = PROTECT(coerceVector(b_, REALSXP));
  protect_count++;
  SEXP add_a = PROTECT(coerceVector(add_a_, INTSXP));
  protect_count++;

  if (LENGTH(a) < 1 || LENGTH(b) < 1 || LENGTH(add_a) < 1) {
    UNPROTECT(protect_count);
    error("Parameters 'a', 'b', and 'add_a' must be scalars");
  }

  int n = LENGTH(y);
  SEXP result = PROTECT(allocVector(REALSXP, n));
  protect_count++;

  GetRNGstate();
  generate_normal_vector(REAL(result), REAL(y), REAL(a)[0], REAL(b)[0], n, INTEGER(add_a)[0]);
  PutRNGstate();

  UNPROTECT(protect_count);
  return result;
}

//==============================================================================
// ADAPTIVE MCMC WRAPPERS
//==============================================================================

/**
 * @brief Run a single adaptive update of the CWMH proposal parameters.
 *
 * @param theta_updated_ Matrix (stored as a vector) containing the most recent
 *        lagged theta values.
 * @param log_sigma_ Numeric vector with the current log standard deviations.
 * @param lag_update_ Integer scalar specifying the adaptation window length.
 * @param n_ Integer scalar representing the dimensionality of the parameter.
 * @param iter_ Integer scalar with the current iteration index.
 * @param max_step_size_ Double scalar limiting per-iteration sigma growth.
 * @param base_adaptation_rate_ Double scalar providing the adaptation learning rate.
 * @param decay_exponent_ Double scalar controlling how fast learning decays.
 * @param target_acceptance_ Double scalar giving the desired acceptance probability.
 * @param min_deviation_threshold_ Double scalar setting the minimum deviation threshold.
 *
 * @return A named list with updated acceptance proportions and log standard deviations.
 */
SEXP test_adapt_cwmh_parameters(SEXP theta_updated_, SEXP log_sigma_,
                                SEXP lag_update_, SEXP n_, SEXP iter_,
                                SEXP max_step_size_, SEXP base_adaptation_rate_,
                                SEXP decay_exponent_, SEXP target_acceptance_,
                                SEXP min_deviation_threshold_) {

  int protect_count = 0;
  SEXP theta_updated = PROTECT(coerceVector(theta_updated_, REALSXP));
  protect_count++;
  SEXP log_sigma_in = PROTECT(coerceVector(log_sigma_, REALSXP));
  protect_count++;
  int lag_update = require_int_scalar(lag_update_, "lag_update");
  int n = require_int_scalar(n_, "n");
  int iter = require_int_scalar(iter_, "iter");
  double max_step_size = require_real_scalar(max_step_size_, "max_step_size");
  double base_adaptation_rate = require_real_scalar(base_adaptation_rate_, "base_adaptation_rate");
  double decay_exponent = require_real_scalar(decay_exponent_, "decay_exponent");
  double target_acceptance = require_real_scalar(target_acceptance_, "target_acceptance");
  double min_deviation_threshold = require_real_scalar(min_deviation_threshold_, "min_deviation_threshold");

  if (lag_update <= 0 || n <= 0) {
    UNPROTECT(protect_count);
    error("lag_update and n must be positive");
  }
  if (LENGTH(theta_updated) != lag_update * n) {
    UNPROTECT(protect_count);
    error("theta_updated must have length lag_update * n");
  }
  if (LENGTH(log_sigma_in) != n) {
    UNPROTECT(protect_count);
    error("log_sigma must have length n");
  }

  SEXP result_list = PROTECT(allocVector(VECSXP, 2));
  protect_count++;
  SEXP accept_prop_sexp = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  SEXP log_sigma_out = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  memcpy(REAL(log_sigma_out), REAL(log_sigma_in), n * sizeof(double));

  adapt_cwmh_parameters(REAL(theta_updated), REAL(accept_prop_sexp), REAL(log_sigma_out),
                        lag_update, n, iter, max_step_size, base_adaptation_rate,
                        decay_exponent, target_acceptance, min_deviation_threshold);

  SEXP names = PROTECT(allocVector(STRSXP, 2));
  protect_count++;
  SET_STRING_ELT(names, 0, mkChar("accept_prop"));
  SET_STRING_ELT(names, 1, mkChar("log_sigma"));

  SET_VECTOR_ELT(result_list, 0, accept_prop_sexp);
  SET_VECTOR_ELT(result_list, 1, log_sigma_out);
  setAttrib(result_list, R_NamesSymbol, names);

  UNPROTECT(protect_count);
  return result_list;
}

/**
 * @brief Reset the global cache used by adaptive CWMH routines between runs.
 *
 * @return R's @c NULL value.
 */
SEXP reset_adaptation_cache_wrapper(void) {
  reset_adaptation_cache();
  return R_NilValue;
}

//==============================================================================
// CONDITIONAL PRECISION WRAPPERS
//==============================================================================

/**
 * @brief Sample the conditional data precision for the first time slice.
 *
 * @param y_ Observed data vector.
 * @param theta_1_ Latent state vector for time one.
 * @param nu_y_ Shape hyperparameter of the Gamma prior.
 * @param eta_y_ Rate hyperparameter of the Gamma prior.
 *
 * @return A length-one numeric vector containing the sampled precision.
 */
SEXP test_generate_precision_data(SEXP y_, SEXP theta_1_, SEXP nu_y_, SEXP eta_y_) {
  int protect_count = 0;
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  SEXP theta_1 = PROTECT(coerceVector(theta_1_, REALSXP));
  protect_count++;
  if (LENGTH(y) != LENGTH(theta_1)) {
    UNPROTECT(protect_count);
    error("theta_1 must have the same length as y");
  }
  double nu_y = require_real_scalar(nu_y_, "nu_y");
  double eta_y = require_real_scalar(eta_y_, "eta_y");
  int n = LENGTH(y);

  GetRNGstate();
  double sample = generate_precision_data(REAL(y), REAL(theta_1), nu_y, eta_y, n);
  PutRNGstate();

  UNPROTECT(protect_count);
  return ScalarReal(sample);
}

/**
 * @brief Sample the conditional precision for an interior latent state.
 *
 * @param theta_0k_ Scalar prior mean for @f$\theta_k@f$.
 * @param theta_0kp1_ Scalar prior mean for @f$\theta_{k+1}@f$.
 * @param theta_k_ Numeric vector with the current state draws at @f$k@f$.
 * @param theta_kp1_ Numeric vector with the state draws at @f$k+1@f$.
 * @param nu_0k_ Shape hyperparameter for the precision prior.
 * @param eta_0k_ Rate hyperparameter for the precision prior.
 *
 * @return A length-one numeric vector containing the sampled precision.
 */
SEXP test_generate_precision_theta_k(SEXP theta_0k_, SEXP theta_0kp1_, SEXP theta_k_,
                                     SEXP theta_kp1_, SEXP nu_0k_, SEXP eta_0k_) {
  int protect_count = 0;
  double theta_0k = require_real_scalar(theta_0k_, "theta_0k");
  double theta_0kp1 = require_real_scalar(theta_0kp1_, "theta_0kp1");
  SEXP theta_k = PROTECT(coerceVector(theta_k_, REALSXP));
  protect_count++;
  SEXP theta_kp1 = PROTECT(coerceVector(theta_kp1_, REALSXP));
  protect_count++;
  ensure_length(theta_kp1, LENGTH(theta_k), "theta_kp1", "theta_k");
  double nu_0k = require_real_scalar(nu_0k_, "nu_0k");
  double eta_0k = require_real_scalar(eta_0k_, "eta_0k");
  int n = LENGTH(theta_k);

  GetRNGstate();
  double sample = generate_precision_theta_k(theta_0k, theta_0kp1, REAL(theta_k), REAL(theta_kp1),
                                             nu_0k, eta_0k, n);
  PutRNGstate();

  UNPROTECT(protect_count);
  return ScalarReal(sample);
}

/**
 * @brief Sample the conditional precision for the final latent state.
 *
 * @param theta_0p_ Scalar prior mean for @f$\theta_p@f$.
 * @param theta_p_ Numeric vector with the terminal state draws.
 * @param nu_0p_ Shape hyperparameter for the precision prior.
 * @param eta_0p_ Rate hyperparameter for the precision prior.
 *
 * @return A length-one numeric vector containing the sampled precision.
 */
SEXP test_generate_precision_theta_p(SEXP theta_0p_, SEXP theta_p_, SEXP nu_0p_, SEXP eta_0p_) {
  int protect_count = 0;
  double theta_0p = require_real_scalar(theta_0p_, "theta_0p");
  SEXP theta_p = PROTECT(coerceVector(theta_p_, REALSXP));
  protect_count++;
  double nu_0p = require_real_scalar(nu_0p_, "nu_0p");
  double eta_0p = require_real_scalar(eta_0p_, "eta_0p");
  int n = LENGTH(theta_p);

  GetRNGstate();
  double sample = generate_precision_theta_p(theta_0p, REAL(theta_p), nu_0p, eta_0p, n);
  PutRNGstate();

  UNPROTECT(protect_count);
  return ScalarReal(sample);
}

//==============================================================================
// CONDITIONAL STATE WRAPPERS
//==============================================================================

/**
 * @brief Sample the first latent state for the local level model.
 *
 * @param data_ Observed data vector.
 * @param prec_data_ Scalar precision associated with the observation model.
 * @param prec_theta_1_ Scalar prior precision for the first state.
 * @param theta_01_ Prior mean for the first state.
 *
 * @return A numeric vector containing sampled values for @f$\theta_1@f$.
 */
SEXP test_generate_theta_1_locallevel(SEXP data_, SEXP prec_data_, SEXP prec_theta_1_, SEXP theta_01_) {
  int protect_count = 0;
  SEXP data = PROTECT(coerceVector(data_, REALSXP));
  protect_count++;
  double prec_data = require_real_scalar(prec_data_, "prec_data");
  double prec_theta_1 = require_real_scalar(prec_theta_1_, "prec_theta_1");
  double theta_01 = require_real_scalar(theta_01_, "theta_01");
  int n = LENGTH(data);

  SEXP result = PROTECT(allocVector(REALSXP, n));
  protect_count++;

  GetRNGstate();
  generate_theta_1_locallevel(REAL(data), REAL(result), prec_data, prec_theta_1, theta_01, n);
  PutRNGstate();

  UNPROTECT(protect_count);
  return result;
}

/**
 * @brief Sample the first latent state for the dynamic binomial model.
 *
 * @param data_ Observed data vector.
 * @param theta_2_ Numeric vector containing draws for @f$\theta_2@f$.
 * @param prec_data_ Scalar precision associated with the observation model.
 * @param prec_theta_1_ Scalar prior precision for @f$\theta_1@f$.
 * @param theta_01_ Prior mean for @f$\theta_1@f$.
 * @param theta_02_ Prior mean for @f$\theta_2@f$.
 *
 * @return A numeric vector containing sampled values for @f$\theta_1@f$.
 */
SEXP test_generate_theta_1(SEXP data_, SEXP theta_2_, SEXP prec_data_, SEXP prec_theta_1_,
                           SEXP theta_01_, SEXP theta_02_) {
  int protect_count = 0;
  SEXP data = PROTECT(coerceVector(data_, REALSXP));
  protect_count++;
  SEXP theta_2 = PROTECT(coerceVector(theta_2_, REALSXP));
  protect_count++;
  ensure_length(theta_2, LENGTH(data), "theta_2", "data");
  double prec_data = require_real_scalar(prec_data_, "prec_data");
  double prec_theta_1 = require_real_scalar(prec_theta_1_, "prec_theta_1");
  double theta_01 = require_real_scalar(theta_01_, "theta_01");
  double theta_02 = require_real_scalar(theta_02_, "theta_02");
  int n = LENGTH(data);

  SEXP result = PROTECT(allocVector(REALSXP, n));
  protect_count++;

  GetRNGstate();
  generate_theta_1(REAL(data), REAL(result), REAL(theta_2), prec_data, prec_theta_1, theta_01, theta_02, n);
  PutRNGstate();

  UNPROTECT(protect_count);
  return result;
}

/**
 * @brief Sample an interior latent state for the dynamic binomial model.
 *
 * @param theta_km1_ Numeric vector with draws at @f$k-1@f$.
 * @param theta_kp1_ Numeric vector with draws at @f$k+1@f$.
 * @param prec_km1_ Scalar precision for the backward transition.
 * @param prec_k_ Scalar precision for the forward transition.
 * @param theta_0k_ Prior mean for @f$\theta_k@f$.
 * @param theta_0kp1_ Prior mean for @f$\theta_{k+1}@f$.
 *
 * @return A numeric vector containing sampled values for @f$\theta_k@f$.
 */
SEXP test_generate_theta_k(SEXP theta_km1_, SEXP theta_kp1_, SEXP prec_km1_, SEXP prec_k_,
                           SEXP theta_0k_, SEXP theta_0kp1_) {
  int protect_count = 0;
  SEXP theta_km1 = PROTECT(coerceVector(theta_km1_, REALSXP));
  protect_count++;
  SEXP theta_kp1 = PROTECT(coerceVector(theta_kp1_, REALSXP));
  protect_count++;
  ensure_length(theta_kp1, LENGTH(theta_km1), "theta_kp1", "theta_km1");
  double prec_km1 = require_real_scalar(prec_km1_, "prec_km1");
  double prec_k = require_real_scalar(prec_k_, "prec_k");
  double theta_0k = require_real_scalar(theta_0k_, "theta_0k");
  double theta_0kp1 = require_real_scalar(theta_0kp1_, "theta_0kp1");
  int n = LENGTH(theta_km1);

  SEXP result = PROTECT(allocVector(REALSXP, n));
  protect_count++;

  GetRNGstate();
  generate_theta_k(REAL(theta_km1), REAL(result), REAL(theta_kp1),
                   prec_km1, prec_k, theta_0k, theta_0kp1, n);
  PutRNGstate();

  UNPROTECT(protect_count);
  return result;
}

/**
 * @brief Sample the terminal latent state for the dynamic binomial model.
 *
 * @param theta_pm1_ Numeric vector with draws at @f$p-1@f$.
 * @param prec_pm1_ Scalar precision for the backward transition.
 * @param prec_p_ Scalar precision for the forward transition.
 * @param theta_0p_ Prior mean for @f$\theta_p@f$.
 *
 * @return A numeric vector containing sampled values for @f$\theta_p@f$.
 */
SEXP test_generate_theta_p(SEXP theta_pm1_, SEXP prec_pm1_, SEXP prec_p_, SEXP theta_0p_) {
  int protect_count = 0;
  SEXP theta_pm1 = PROTECT(coerceVector(theta_pm1_, REALSXP));
  protect_count++;
  double prec_pm1 = require_real_scalar(prec_pm1_, "prec_pm1");
  double prec_p = require_real_scalar(prec_p_, "prec_p");
  double theta_0p = require_real_scalar(theta_0p_, "theta_0p");
  int n = LENGTH(theta_pm1);

  SEXP result = PROTECT(allocVector(REALSXP, n));
  protect_count++;

  GetRNGstate();
  generate_theta_p(REAL(theta_pm1), REAL(result), prec_pm1, prec_p, theta_0p, n);
  PutRNGstate();

  UNPROTECT(protect_count);
  return result;
}

//==============================================================================
// CONDITIONAL THETA0 WRAPPERS
//==============================================================================

/**
 * @brief Sample the prior mean of the first latent state in the local level model.
 *
 * @param theta_1_ Numeric vector of first-state draws.
 * @param prec_theta_1_ Scalar precision for the first state.
 * @param mean_theta_01_ Prior mean hyperparameter.
 * @param prec_theta_01_ Prior precision hyperparameter.
 *
 * @return A length-one numeric vector with the sampled @f$\theta_{0,1}@f$.
 */
SEXP test_generate_theta_01_locallevel(SEXP theta_1_, SEXP prec_theta_1_,
                                       SEXP mean_theta_01_, SEXP prec_theta_01_) {
  int protect_count = 0;
  SEXP theta_1 = PROTECT(coerceVector(theta_1_, REALSXP));
  protect_count++;
  double prec_theta_1 = require_real_scalar(prec_theta_1_, "prec_theta_1");
  double mean_theta_01 = require_real_scalar(mean_theta_01_, "mean_theta_01");
  double prec_theta_01 = require_real_scalar(prec_theta_01_, "prec_theta_01");
  int n = LENGTH(theta_1);

  GetRNGstate();
  double sample = generate_theta_01_locallevel(REAL(theta_1), prec_theta_1,
                                               mean_theta_01, prec_theta_01, n);
  PutRNGstate();

  UNPROTECT(protect_count);
  return ScalarReal(sample);
}

/**
 * @brief Sample the prior mean of the first latent state in the full dynamic model.
 *
 * @param theta_1_ Numeric vector of first-state draws.
 * @param theta_02_ Prior mean hyperparameter for @f$\theta_2@f$.
 * @param prec_theta_1_ Scalar precision for the first state.
 * @param mean_theta_01_ Prior mean hyperparameter.
 * @param prec_theta_01_ Prior precision hyperparameter.
 *
 * @return A length-one numeric vector with the sampled @f$\theta_{0,1}@f$.
 */
SEXP test_generate_theta_01(SEXP theta_1_, SEXP theta_02_, SEXP prec_theta_1_,
                            SEXP mean_theta_01_, SEXP prec_theta_01_) {
  int protect_count = 0;
  SEXP theta_1 = PROTECT(coerceVector(theta_1_, REALSXP));
  protect_count++;
  double theta_02 = require_real_scalar(theta_02_, "theta_02");
  double prec_theta_1 = require_real_scalar(prec_theta_1_, "prec_theta_1");
  double mean_theta_01 = require_real_scalar(mean_theta_01_, "mean_theta_01");
  double prec_theta_01 = require_real_scalar(prec_theta_01_, "prec_theta_01");
  int n = LENGTH(theta_1);

  GetRNGstate();
  double sample = generate_theta_01(REAL(theta_1), theta_02, prec_theta_1,
                                    mean_theta_01, prec_theta_01, n);
  PutRNGstate();

  UNPROTECT(protect_count);
  return ScalarReal(sample);
}

/**
 * @brief Sample the prior mean for an interior latent state.
 *
 * @param theta_km1_ Numeric vector with draws at @f$k-1@f$.
 * @param theta_k_ Numeric vector with draws at @f$k@f$.
 * @param theta_0km1_ Scalar prior mean for @f$\theta_{k-1}@f$.
 * @param theta_0kp1_ Scalar prior mean for @f$\theta_{k+1}@f$.
 * @param prec_km1_ Scalar precision for the backward transition.
 * @param prec_k_ Scalar precision for the forward transition.
 * @param mean_0k_ Scalar prior mean hyperparameter.
 * @param prec_0k_ Scalar prior precision hyperparameter.
 *
 * @return A length-one numeric vector with the sampled @f$\theta_{0,k}@f$.
 */
SEXP test_generate_theta_0k(SEXP theta_km1_, SEXP theta_k_, SEXP theta_0km1_, SEXP theta_0kp1_,
                            SEXP prec_km1_, SEXP prec_k_, SEXP mean_0k_, SEXP prec_0k_) {
  int protect_count = 0;
  SEXP theta_km1 = PROTECT(coerceVector(theta_km1_, REALSXP));
  protect_count++;
  SEXP theta_k = PROTECT(coerceVector(theta_k_, REALSXP));
  protect_count++;
  ensure_length(theta_k, LENGTH(theta_km1), "theta_k", "theta_km1");
  double theta_0km1 = require_real_scalar(theta_0km1_, "theta_0km1");
  double theta_0kp1 = require_real_scalar(theta_0kp1_, "theta_0kp1");
  double prec_km1 = require_real_scalar(prec_km1_, "prec_km1");
  double prec_k = require_real_scalar(prec_k_, "prec_k");
  double mean_0k = require_real_scalar(mean_0k_, "mean_0k");
  double prec_0k = require_real_scalar(prec_0k_, "prec_0k");
  int n = LENGTH(theta_km1);

  GetRNGstate();
  double sample = generate_theta_0k(REAL(theta_km1), REAL(theta_k), theta_0km1, theta_0kp1,
                                    prec_km1, prec_k, mean_0k, prec_0k, n);
  PutRNGstate();

  UNPROTECT(protect_count);
  return ScalarReal(sample);
}

/**
 * @brief Sample the prior mean for the terminal latent state.
 *
 * @param theta_pm1_ Numeric vector with draws at @f$p-1@f$.
 * @param theta_p_ Numeric vector with draws at @f$p@f$.
 * @param theta_0pm1_ Scalar prior mean for @f$\theta_{p-1}@f$.
 * @param prec_pm1_ Scalar precision for the backward transition.
 * @param prec_p_ Scalar precision for the forward transition.
 * @param mean_0p_ Scalar prior mean hyperparameter.
 * @param prec_0p_ Scalar prior precision hyperparameter.
 *
 * @return A length-one numeric vector with the sampled @f$\theta_{0,p}@f$.
 */
SEXP test_generate_theta_0p(SEXP theta_pm1_, SEXP theta_p_, SEXP theta_0pm1_,
                            SEXP prec_pm1_, SEXP prec_p_, SEXP mean_0p_, SEXP prec_0p_) {
  int protect_count = 0;
  SEXP theta_pm1 = PROTECT(coerceVector(theta_pm1_, REALSXP));
  protect_count++;
  SEXP theta_p = PROTECT(coerceVector(theta_p_, REALSXP));
  protect_count++;
  ensure_length(theta_p, LENGTH(theta_pm1), "theta_p", "theta_pm1");
  double theta_0pm1 = require_real_scalar(theta_0pm1_, "theta_0pm1");
  double prec_pm1 = require_real_scalar(prec_pm1_, "prec_pm1");
  double prec_p = require_real_scalar(prec_p_, "prec_p");
  double mean_0p = require_real_scalar(mean_0p_, "mean_0p");
  double prec_0p = require_real_scalar(prec_0p_, "prec_0p");
  int n = LENGTH(theta_pm1);

  GetRNGstate();
  double sample = generate_theta_0p(REAL(theta_pm1), REAL(theta_p), theta_0pm1,
                                    prec_pm1, prec_p, mean_0p, prec_0p, n);
  PutRNGstate();

  UNPROTECT(protect_count);
  return ScalarReal(sample);
}

//==============================================================================
// BINOMIAL CWMH WRAPPERS
//==============================================================================

/**
 * @brief Run a single CWMH update for the logit-binomial local level model.
 *
 * @param theta_1_in_ Numeric vector with the previous state draws.
 * @param theta_01_in_ Scalar prior mean for the initial state.
 * @param prec_1_in_ Scalar prior precision for the state.
 * @param y_ Observed binomial counts.
 * @param n_trials_ Scalar number of trials for the binomial likelihood.
 * @param log_sigma_in_ Numeric vector of proposal log standard deviations.
 *
 * @return A list containing updated state draws and logit-scale alphas.
 */
SEXP test_cwmh_alpha_logit_binomial_locallevel(SEXP theta_1_in_, SEXP theta_01_in_,
                                               SEXP prec_1_in_, SEXP y_, SEXP n_trials_,
                                               SEXP log_sigma_in_) {
  const int LAG_UPDATE = 10;
  const int ITER = 1;
  const int COMPUTE_ALPHA = 1;

  int protect_count = 0;
  SEXP theta_1_prev = PROTECT(coerceVector(theta_1_in_, REALSXP));
  protect_count++;
  double theta_01_prev = require_real_scalar(theta_01_in_, "theta_01_in");
  double prec_1_prev = require_real_scalar(prec_1_in_, "prec_1_in");
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  ensure_length(y, LENGTH(theta_1_prev), "y", "theta_1_in");
  double n_trials = require_real_scalar(n_trials_, "n_trials");
  SEXP log_sigma = PROTECT(coerceVector(log_sigma_in_, REALSXP));
  protect_count++;
  ensure_length(log_sigma, LENGTH(theta_1_prev), "log_sigma", "theta_1_in");
  int n = LENGTH(theta_1_prev);

  double *theta_1_current = (double *) R_alloc(n, sizeof(double));
  double *alpha_current = (double *) R_alloc(n, sizeof(double));
  double *theta_1_updated = (double *) R_alloc(LAG_UPDATE * n, sizeof(double));
  memset(theta_1_updated, 0, LAG_UPDATE * n * sizeof(double));
  double *hat_theta_1 = (double *) R_alloc(n, sizeof(double));
  double *theta_1_new = (double *) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double *) R_alloc(n, sizeof(double));

  GetRNGstate();
  cwmh_alpha_logit_binomial_locallevel(REAL(theta_1_prev), theta_1_current, alpha_current,
                                       theta_01_prev, prec_1_prev, theta_1_updated,
                                       REAL(y), REAL(log_sigma), hat_theta_1,
                                       theta_1_new, log_accept_prob,
                                       LAG_UPDATE, n_trials, n, ITER, COMPUTE_ALPHA);
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 2));
  protect_count++;
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  memcpy(REAL(theta_1_out), theta_1_current, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_current, n * sizeof(double));

  SEXP names = PROTECT(allocVector(STRSXP, 2));
  protect_count++;
  SET_STRING_ELT(names, 0, mkChar("theta_1"));
  SET_STRING_ELT(names, 1, mkChar("alpha"));

  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);
  setAttrib(res, R_NamesSymbol, names);

  UNPROTECT(protect_count);
  return res;
}

/**
 * @brief Run a single CWMH update for the two-parameter logit-binomial model.
 *
 * @param theta_1_in_ Numeric vector with the previous state draws.
 * @param theta_2_in_ Numeric vector with the companion state draws.
 * @param theta_01_in_ Scalar prior mean for the first state.
 * @param theta_02_in_ Scalar prior mean for the second state.
 * @param prec_1_in_ Scalar prior precision for the first state.
 * @param y_ Observed binomial counts.
 * @param n_trials_ Scalar number of trials for the binomial likelihood.
 *
 * @return A list containing updated state draws and logit-scale alphas.
 */
SEXP test_cwmh_alpha_logit_binomial(SEXP theta_1_in_, SEXP theta_2_in_,
                                    SEXP theta_01_in_, SEXP theta_02_in_,
                                    SEXP prec_1_in_, SEXP y_, SEXP n_trials_) {
  const int LAG_UPDATE = 10;
  const int ITER = 1;
  const int COMPUTE_ALPHA = 1;
  const double DEFAULT_LOG_SIGMA = log(0.1);

  int protect_count = 0;
  SEXP theta_1_prev = PROTECT(coerceVector(theta_1_in_, REALSXP));
  protect_count++;
  SEXP theta_2_curr = PROTECT(coerceVector(theta_2_in_, REALSXP));
  protect_count++;
  ensure_length(theta_2_curr, LENGTH(theta_1_prev), "theta_2_in", "theta_1_in");
  double theta_01_prev = require_real_scalar(theta_01_in_, "theta_01_in");
  double theta_02_prev = require_real_scalar(theta_02_in_, "theta_02_in");
  double prec_1_prev = require_real_scalar(prec_1_in_, "prec_1_in");
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  ensure_length(y, LENGTH(theta_1_prev), "y", "theta_1_in");
  double n_trials = require_real_scalar(n_trials_, "n_trials");
  int n = LENGTH(theta_1_prev);

  double *theta_1_current = (double *) R_alloc(n, sizeof(double));
  double *alpha_current = (double *) R_alloc(n, sizeof(double));
  double *theta_1_updated = (double *) R_alloc(LAG_UPDATE * n, sizeof(double));
  memset(theta_1_updated, 0, LAG_UPDATE * n * sizeof(double));
  double *log_sigma = (double *) R_alloc(n, sizeof(double));
  for (int i = 0; i < n; i++) {
    log_sigma[i] = DEFAULT_LOG_SIGMA;
  }
  double *hat_theta_1 = (double *) R_alloc(n, sizeof(double));
  double *theta_1_new = (double *) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double *) R_alloc(n, sizeof(double));

  GetRNGstate();
  cwmh_alpha_logit_binomial(REAL(theta_1_prev), theta_1_current, REAL(theta_2_curr), alpha_current,
                            theta_01_prev, theta_02_prev, theta_1_updated,
                            prec_1_prev, REAL(y), log_sigma, hat_theta_1,
                            theta_1_new, log_accept_prob,
                            LAG_UPDATE, n_trials, n, ITER, COMPUTE_ALPHA);
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 2));
  protect_count++;
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  memcpy(REAL(theta_1_out), theta_1_current, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_current, n * sizeof(double));

  SEXP names = PROTECT(allocVector(STRSXP, 2));
  protect_count++;
  SET_STRING_ELT(names, 0, mkChar("theta_1"));
  SET_STRING_ELT(names, 1, mkChar("alpha"));
  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);
  setAttrib(res, R_NamesSymbol, names);

  UNPROTECT(protect_count);
  return res;
}

//==============================================================================
// BINOMIAL ALPHA GENERATION WRAPPERS
//==============================================================================

/**
 * @brief Adaptively generate alphas for the logit-binomial local level model.
 *
 * @param theta_1_in_ Numeric vector with the previous state draws.
 * @param theta_01_in_ Scalar prior mean for the initial state.
 * @param prec_1_in_ Scalar prior precision for the state.
 * @param y_ Observed binomial counts.
 * @param n_trials_ Scalar number of trials for the binomial likelihood.
 *
 * @return A list containing updated state draws and logit-scale alphas.
 */
SEXP test_generate_alpha_logit_binomial_locallevel(SEXP theta_1_in_, SEXP theta_01_in_,
                                                   SEXP prec_1_in_, SEXP y_, SEXP n_trials_) {
  const int LAG_UPDATE = 50;
  const int ITER = 1;
  const double MAX_STEP = 0.1;
  const double BASE_ADAPT = 1.0;
  const double DECAY = 0.5;
  const double TARGET = 0.44;
  const double MIN_DEV = 1.0 / (double)LAG_UPDATE;
  const double DEFAULT_LOG_SIGMA = log(0.1);

  int protect_count = 0;
  SEXP theta_1_prev = PROTECT(coerceVector(theta_1_in_, REALSXP));
  protect_count++;
  double theta_01_prev = require_real_scalar(theta_01_in_, "theta_01_in");
  double prec_1_prev = require_real_scalar(prec_1_in_, "prec_1_in");
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  ensure_length(y, LENGTH(theta_1_prev), "y", "theta_1_in");
  double n_trials = require_real_scalar(n_trials_, "n_trials");
  int n = LENGTH(theta_1_prev);

  double *theta_1_current = (double *) R_alloc(n, sizeof(double));
  double *alpha_current = (double *) R_alloc(n, sizeof(double));
  double *theta_1_updated = (double *) R_alloc(LAG_UPDATE * n, sizeof(double));
  memset(theta_1_updated, 0, LAG_UPDATE * n * sizeof(double));
  double *accept_prop = (double *) R_alloc(n, sizeof(double));
  double *log_sigma = (double *) R_alloc(n, sizeof(double));
  for (int i = 0; i < n; i++) {
    log_sigma[i] = DEFAULT_LOG_SIGMA;
  }
  double *hat_theta_1 = (double *) R_alloc(n, sizeof(double));
  double *theta_1_new = (double *) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double *) R_alloc(n, sizeof(double));

  GetRNGstate();
  generate_alpha_logit_binomial_locallevel(REAL(theta_1_prev), theta_1_current, alpha_current,
                                           theta_01_prev, prec_1_prev, theta_1_updated,
                                           REAL(y), accept_prop, log_sigma, hat_theta_1,
                                           theta_1_new, log_accept_prob, LAG_UPDATE, n_trials,
                                           n, ITER, MAX_STEP, BASE_ADAPT, DECAY, TARGET,
                                           MIN_DEV, 1);
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 2));
  protect_count++;
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  memcpy(REAL(theta_1_out), theta_1_current, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_current, n * sizeof(double));

  SEXP names = PROTECT(allocVector(STRSXP, 2));
  protect_count++;
  SET_STRING_ELT(names, 0, mkChar("theta_1"));
  SET_STRING_ELT(names, 1, mkChar("alpha"));
  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);
  setAttrib(res, R_NamesSymbol, names);

  UNPROTECT(protect_count);
  return res;
}

/**
 * @brief Adaptively generate alphas for the two-parameter logit-binomial model.
 *
 * @param theta_1_in_ Numeric vector with the previous state draws.
 * @param theta_2_in_ Numeric vector with the companion state draws.
 * @param theta_01_in_ Scalar prior mean for the first state.
 * @param theta_02_in_ Scalar prior mean for the second state.
 * @param prec_1_in_ Scalar prior precision for the first state.
 * @param y_ Observed binomial counts.
 * @param n_trials_ Scalar number of trials for the binomial likelihood.
 *
 * @return A list containing updated state draws and logit-scale alphas.
 */
SEXP test_generate_alpha_logit_binomial(SEXP theta_1_in_, SEXP theta_2_in_, SEXP theta_01_in_,
                                        SEXP theta_02_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_) {
  const int LAG_UPDATE = 50;
  const int ITER = 1;
  const double MAX_STEP = 0.1;
  const double BASE_ADAPT = 1.0;
  const double DECAY = 0.5;
  const double TARGET = 0.44;
  const double MIN_DEV = 1.0 / (double)LAG_UPDATE;
  const double DEFAULT_LOG_SIGMA = log(0.1);

  int protect_count = 0;
  SEXP theta_1_prev = PROTECT(coerceVector(theta_1_in_, REALSXP));
  protect_count++;
  SEXP theta_2_curr = PROTECT(coerceVector(theta_2_in_, REALSXP));
  protect_count++;
  ensure_length(theta_2_curr, LENGTH(theta_1_prev), "theta_2_in", "theta_1_in");
  double theta_01_prev = require_real_scalar(theta_01_in_, "theta_01_in");
  double theta_02_prev = require_real_scalar(theta_02_in_, "theta_02_in");
  double prec_1_prev = require_real_scalar(prec_1_in_, "prec_1_in");
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  ensure_length(y, LENGTH(theta_1_prev), "y", "theta_1_in");
  double n_trials = require_real_scalar(n_trials_, "n_trials");
  int n = LENGTH(theta_1_prev);

  double *theta_1_current = (double *) R_alloc(n, sizeof(double));
  double *alpha_current = (double *) R_alloc(n, sizeof(double));
  double *theta_1_updated = (double *) R_alloc(LAG_UPDATE * n, sizeof(double));
  memset(theta_1_updated, 0, LAG_UPDATE * n * sizeof(double));
  double *accept_prop = (double *) R_alloc(n, sizeof(double));
  double *log_sigma = (double *) R_alloc(n, sizeof(double));
  for (int i = 0; i < n; i++) {
    log_sigma[i] = DEFAULT_LOG_SIGMA;
  }
  double *hat_theta_1 = (double *) R_alloc(n, sizeof(double));
  double *theta_1_new = (double *) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double *) R_alloc(n, sizeof(double));

  GetRNGstate();
  generate_alpha_logit_binomial(REAL(theta_1_prev), theta_1_current, REAL(theta_2_curr),
                                alpha_current, theta_01_prev, theta_02_prev, prec_1_prev,
                                theta_1_updated, REAL(y), accept_prop, log_sigma,
                                hat_theta_1, theta_1_new, log_accept_prob, LAG_UPDATE,
                                n_trials, n, ITER, MAX_STEP, BASE_ADAPT, DECAY,
                                TARGET, MIN_DEV, 1);
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 2));
  protect_count++;
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  memcpy(REAL(theta_1_out), theta_1_current, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_current, n * sizeof(double));

  SEXP names = PROTECT(allocVector(STRSXP, 2));
  protect_count++;
  SET_STRING_ELT(names, 0, mkChar("theta_1"));
  SET_STRING_ELT(names, 1, mkChar("alpha"));
  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);
  setAttrib(res, R_NamesSymbol, names);

  UNPROTECT(protect_count);
  return res;
}

//==============================================================================
// PROBIT-BERNOULLI WRAPPERS
//==============================================================================

/**
 * @brief Generate alphas for the probit-Bernoulli local level model.
 *
 * @param theta_1_in_ Numeric vector with previous state draws.
 * @param theta_01_in_ Scalar prior mean for the initial state.
 * @param prec_1_in_ Scalar prior precision for the state.
 * @param y_ Observed Bernoulli outcomes.
 *
 * @return A list containing updated state draws and probit-scale alphas.
 */
SEXP test_generate_alpha_probit_bernoulli_locallevel(SEXP theta_1_in_, SEXP theta_01_in_,
                                                     SEXP prec_1_in_, SEXP y_) {
  int protect_count = 0;
  SEXP theta_1_prev = PROTECT(coerceVector(theta_1_in_, REALSXP));
  protect_count++;
  double theta_01_prev = require_real_scalar(theta_01_in_, "theta_01_in");
  double prec_1_prev = require_real_scalar(prec_1_in_, "prec_1_in");
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  ensure_length(y, LENGTH(theta_1_prev), "y", "theta_1_in");
  int n = LENGTH(theta_1_prev);

  double *theta_1_current = (double *) R_alloc(n, sizeof(double));
  double *alpha_current = (double *) R_alloc(n, sizeof(double));
  double *rhs_vector = (double *) R_alloc(n, sizeof(double));

  GetRNGstate();
  generate_alpha_probit_bernoulli_locallevel(REAL(theta_1_prev), theta_1_current,
                                             alpha_current, theta_01_prev, prec_1_prev,
                                             REAL(y), rhs_vector, n, 1);
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 2));
  protect_count++;
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  memcpy(REAL(theta_1_out), theta_1_current, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_current, n * sizeof(double));

  SEXP names = PROTECT(allocVector(STRSXP, 2));
  protect_count++;
  SET_STRING_ELT(names, 0, mkChar("theta_1"));
  SET_STRING_ELT(names, 1, mkChar("alpha"));
  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);
  setAttrib(res, R_NamesSymbol, names);

  UNPROTECT(protect_count);
  return res;
}

/**
 * @brief Generate alphas for the two-parameter probit-Bernoulli model.
 *
 * @param theta_1_in_ Numeric vector with previous state draws.
 * @param theta_2_in_ Numeric vector with companion state draws.
 * @param theta_01_in_ Scalar prior mean for the first state.
 * @param theta_02_in_ Scalar prior mean for the second state.
 * @param prec_1_in_ Scalar prior precision for the first state.
 * @param y_ Observed Bernoulli outcomes.
 *
 * @return A list containing updated state draws and probit-scale alphas.
 */
SEXP test_generate_alpha_probit_bernoulli(SEXP theta_1_in_, SEXP theta_2_in_,
                                          SEXP theta_01_in_, SEXP theta_02_in_,
                                          SEXP prec_1_in_, SEXP y_) {
  int protect_count = 0;
  SEXP theta_1_prev = PROTECT(coerceVector(theta_1_in_, REALSXP));
  protect_count++;
  SEXP theta_2_curr = PROTECT(coerceVector(theta_2_in_, REALSXP));
  protect_count++;
  ensure_length(theta_2_curr, LENGTH(theta_1_prev), "theta_2_in", "theta_1_in");
  double theta_01_prev = require_real_scalar(theta_01_in_, "theta_01_in");
  double theta_02_prev = require_real_scalar(theta_02_in_, "theta_02_in");
  double prec_1_prev = require_real_scalar(prec_1_in_, "prec_1_in");
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  ensure_length(y, LENGTH(theta_1_prev), "y", "theta_1_in");
  int n = LENGTH(theta_1_prev);

  double *theta_1_current = (double *) R_alloc(n, sizeof(double));
  double *alpha_current = (double *) R_alloc(n, sizeof(double));
  double *rhs_vector = (double *) R_alloc(n, sizeof(double));

  GetRNGstate();
  generate_alpha_probit_bernoulli(REAL(theta_1_prev), theta_1_current, alpha_current,
                                  REAL(theta_2_curr), theta_01_prev, theta_02_prev,
                                  prec_1_prev, REAL(y), rhs_vector, n, 1);
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 2));
  protect_count++;
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));
  protect_count++;
  memcpy(REAL(theta_1_out), theta_1_current, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_current, n * sizeof(double));

  SEXP names = PROTECT(allocVector(STRSXP, 2));
  protect_count++;
  SET_STRING_ELT(names, 0, mkChar("theta_1"));
  SET_STRING_ELT(names, 1, mkChar("alpha"));
  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);
  setAttrib(res, R_NamesSymbol, names);

  UNPROTECT(protect_count);
  return res;
}

//==============================================================================
// COMPLETE MCMC WRAPPERS
//==============================================================================

/**
 * @brief Execute the full MCMC sampler for the logit-binomial local level model.
 *
 * @param y_ Observed binomial counts.
 * @param n_trials_ Number of trials for each observation.
 * @param burnin_ Number of burn-in iterations.
 * @param thinning_ Thinning interval for retained samples.
 * @param n_chain_ Number of chains to simulate.
 * @param theta_1_true_ Ignored parameter retained for interface compatibility.
 * @param theta_01_true_ Ignored parameter retained for interface compatibility.
 * @param prec_1_true_ Ignored parameter retained for interface compatibility.
 * @param prior_theta01_mean_ Prior mean hyperparameter for @f$\theta_{0,1}@f$.
 * @param prior_theta01_prec_ Prior precision hyperparameter for @f$\theta_{0,1}@f$.
 * @param prior_prec1_shape_ Shape hyperparameter for the precision prior.
 * @param prior_prec1_rate_ Rate hyperparameter for the precision prior.
 * @param lag_update_ Adaptation window length.
 * @param max_step_size_ Maximum adaptation step size.
 * @param base_adaptation_rate_ Base adaptation rate.
 * @param decay_exponent_ Adaptation decay exponent.
 * @param target_acceptance_ Target acceptance probability.
 * @param return_log_sigma_ Logical flag indicating whether to return log sigmas.
 * @param return_accept_prop_ Logical flag indicating whether to return acceptance proportions.
 *
 * @return An R list mirroring the production sampler output.
 */
SEXP test_mcmc_binomial_locallevel_fixed_params(SEXP y_, SEXP n_trials_, SEXP burnin_, SEXP thinning_,
                                                SEXP n_chain_, SEXP theta_1_true_, SEXP theta_01_true_,
                                                SEXP prec_1_true_, SEXP prior_theta01_mean_,
                                                SEXP prior_theta01_prec_, SEXP prior_prec1_shape_,
                                                SEXP prior_prec1_rate_, SEXP lag_update_,
                                                SEXP max_step_size_, SEXP base_adaptation_rate_,
                                                SEXP decay_exponent_, SEXP target_acceptance_,
                                                SEXP return_log_sigma_, SEXP return_accept_prop_) {
  (void)theta_1_true_;
  (void)theta_01_true_;
  (void)prec_1_true_;

  int lag_update_int = require_int_scalar(lag_update_, "lag_update");
  SEXP min_dev = PROTECT(ScalarReal(1.0 / (double)lag_update_int));

  SEXP result = C_MCMC_logit_binomial_locallevel(y_, n_trials_, burnin_, thinning_, n_chain_,
                                                 prior_theta01_mean_, prior_theta01_prec_,
                                                 prior_prec1_shape_, prior_prec1_rate_,
                                                 lag_update_, max_step_size_, base_adaptation_rate_,
                                                 decay_exponent_, target_acceptance_,
                                                 min_dev, return_log_sigma_, return_accept_prop_);

  UNPROTECT(1);
  return result;
}

/**
 * @brief Execute the full MCMC sampler for the probit-Bernoulli local level model.
 *
 * @param y_ Observed Bernoulli outcomes.
 * @param burnin_ Number of burn-in iterations.
 * @param thinning_ Thinning interval for retained samples.
 * @param n_chain_ Number of chains to simulate.
 * @param theta_1_true_ Ignored parameter retained for interface compatibility.
 * @param theta_01_true_ Ignored parameter retained for interface compatibility.
 * @param prec_1_true_ Ignored parameter retained for interface compatibility.
 * @param prior_theta01_mean_ Prior mean hyperparameter for @f$\theta_{0,1}@f$.
 * @param prior_theta01_prec_ Prior precision hyperparameter for @f$\theta_{0,1}@f$.
 * @param prior_prec1_shape_ Shape hyperparameter for the precision prior.
 * @param prior_prec1_rate_ Rate hyperparameter for the precision prior.
 *
 * @return An R list mirroring the production sampler output.
 */
SEXP test_mcmc_probit_bernoulli_locallevel_fixed_params(SEXP y_, SEXP burnin_, SEXP thinning_,
                                                        SEXP n_chain_, SEXP theta_1_true_,
                                                        SEXP theta_01_true_, SEXP prec_1_true_,
                                                        SEXP prior_theta01_mean_, SEXP prior_theta01_prec_,
                                                        SEXP prior_prec1_shape_, SEXP prior_prec1_rate_) {
  (void)theta_1_true_;
  (void)theta_01_true_;
  (void)prec_1_true_;

  return C_MCMC_probit_bernoulli_locallevel(y_, burnin_, thinning_, n_chain_,
                                            prior_theta01_mean_, prior_theta01_prec_,
                                            prior_prec1_shape_, prior_prec1_rate_);
}
