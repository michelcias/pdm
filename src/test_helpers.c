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
#include "cwmh_poisson.h"
#include "generate_alpha_poisson.h"
#include "test_helpers.h"

//==============================================================================
// INTERNAL UTILITIES
//==============================================================================

/**
 * @brief Ensure that a numeric vector matches the expected length.
 *
 * @details This lightweight validation helper checks that wrapper inputs share
 *          consistent lengths before delegating to the core numerical
 *          routines.
 *
 * @param vec        Candidate vector to validate.
 * @param expected   Required length for @p vec.
 * @param arg        Name of the argument being checked, used for error messages.
 * @param reference  Name of the reference argument establishing the length,
 *                   used for error reporting.
 *
 * @return Nothing. Raises an R error when the supplied length is invalid.
 */
static void ensure_length(SEXP        vec,
                          int         expected,
                          const char *arg,
                          const char *reference) {
  if (LENGTH(vec) != expected) {
    error("%s must have length %d to match %s (got %d)",
          arg, expected, reference, LENGTH(vec));
  }
}

/**
 * @brief Coerce an object to integer and return the first scalar entry.
 *
 * @details Wrappers rely on this helper to normalise scalar integer
 *          arguments, ensuring consistent type handling across the module.
 *
 * @param x    R object expected to hold an integer scalar.
 * @param arg  Name of the argument being coerced, used for diagnostic messages.
 *
 * @return The first integer extracted from @p x (defaults to 0 when NULL).
 *
 * @note Accepts NULL input (returns 0 for NULL inputs).
 */
static int require_int_scalar(SEXP        x,
                              const char *arg) {
  if (x == R_NilValue) {
    return 0;
  }
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
 * @details Wrappers use this helper to promote scalar numeric arguments to
 *          double precision and to provide uniform error reporting.
 *
 * @param x    R object expected to hold a numeric scalar.
 * @param arg  Name of the argument being coerced, used for diagnostic messages.
 *
 * @return The first double extracted from @p x (defaults to 0.0 when NULL).
 *
 * @note Accepts NULL input (returns 0.0 for NULL inputs).
 */
static double require_real_scalar(SEXP        x,
                                  const char *arg) {
  if (x == R_NilValue) {
    return 0.0;
  }
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
 * @details Provides a thin wrapper around the internal @c ilogit helper so
 *          that tests can confirm numerical behaviour through R's @c .Call
 *          interface.
 *
 * @param x_  Numeric vector whose first element is transformed.
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
                                 SEXP add_a_) {
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
  generate_normal_vector(
    REAL(result),         /* result: output buffer */
    REAL(y),              /* y: baseline centre */
    REAL(a)[0],           /* a: location shift */
    REAL(b)[0],           /* b: precision parameter */
    n,                    /* n: vector length */
    INTEGER(add_a)[0]     /* add_a: apply shift flag */
  );
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

  adapt_cwmh_parameters(
    REAL(theta_updated),       /* theta_updated: lagged theta window */
    REAL(accept_prop_sexp),    /* accept_prop: output acceptance proportions */
    REAL(log_sigma_out),       /* log_sigma_out: output log sigma */
    lag_update,                /* lag_update: adaptation window length */
    n,                         /* n: parameter dimension */
    iter,                      /* iter: current iteration */
    max_step_size,             /* max_step_size: adaptation cap */
    base_adaptation_rate,      /* base_adaptation_rate: initial learning rate */
    decay_exponent,            /* decay_exponent: decay schedule */
    target_acceptance,         /* target_acceptance: desired acceptance */
    min_deviation_threshold    /* min_deviation_threshold: deviation trigger */
  );

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
 * @details Provides tests with explicit control over the shared adaptation
 *          cache to ensure deterministic behaviour.
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
                                  SEXP eta_y_) {
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
  double sample = generate_precision_data(
    REAL(y),         /* y: observed data */
    REAL(theta_1),   /* theta_1: latent state draws */
    nu_y,            /* nu_y: Gamma shape */
    eta_y,           /* eta_y: Gamma rate */
    n                /* n: number of observations */
  );
  PutRNGstate();

  UNPROTECT(protect_count);
  return ScalarReal(sample);
}

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
                                     SEXP eta_0k_) {
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
  double sample = generate_precision_theta_k(
    theta_0k,         /* theta_0k: prior mean for theta_k */
    theta_0kp1,       /* theta_0kp1: prior mean for theta_{k+1} */
    REAL(theta_k),    /* theta_k: draws at k */
    REAL(theta_kp1),  /* theta_kp1: draws at k+1 */
    nu_0k,            /* nu_0k: Gamma shape */
    eta_0k,           /* eta_0k: Gamma rate */
    n                 /* n: number of observations */
  );
  PutRNGstate();

  UNPROTECT(protect_count);
  return ScalarReal(sample);
}

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
                                     SEXP eta_0p_) {
  int protect_count = 0;
  double theta_0p = require_real_scalar(theta_0p_, "theta_0p");
  SEXP theta_p = PROTECT(coerceVector(theta_p_, REALSXP));
  protect_count++;
  double nu_0p = require_real_scalar(nu_0p_, "nu_0p");
  double eta_0p = require_real_scalar(eta_0p_, "eta_0p");
  int n = LENGTH(theta_p);

  GetRNGstate();
  double sample = generate_precision_theta_p(
    theta_0p,        /* theta_0p: prior mean */
    REAL(theta_p),   /* theta_p: terminal state draws */
    nu_0p,           /* nu_0p: Gamma shape */
    eta_0p,          /* eta_0p: Gamma rate */
    n                /* n: number of observations */
  );
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
                                      SEXP theta_01_) {
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
  generate_theta_1_locallevel(
    REAL(data),      /* data: observed sequence */
    REAL(result),    /* result: output theta_1 draws */
    prec_data,       /* prec_data: observation precision */
    prec_theta_1,    /* prec_theta_1: prior precision */
    theta_01,        /* theta_01: prior mean */
    n                /* n: number of observations */
  );
  PutRNGstate();

  UNPROTECT(protect_count);
  return result;
}

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
                           SEXP theta_02_) {
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
  generate_theta_1(
    REAL(data),     /* data: observed sequence */
    REAL(result),   /* result: output theta_1 draws */
    REAL(theta_2),  /* theta_2: companion state */
    prec_data,      /* prec_data: observation precision */
    prec_theta_1,   /* prec_theta_1: prior precision */
    theta_01,       /* theta_01: prior mean for theta_1 */
    theta_02,       /* theta_02: prior mean for theta_2 */
    n               /* n: number of observations */
  );
  PutRNGstate();

  UNPROTECT(protect_count);
  return result;
}

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
                           SEXP theta_0kp1_) {
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
  generate_theta_k(
    REAL(theta_km1),  /* theta_km1: state draws at k-1 */
    REAL(result),     /* result: output theta_k draws */
    REAL(theta_kp1),  /* theta_kp1: state draws at k+1 */
    prec_km1,         /* prec_km1: backward precision */
    prec_k,           /* prec_k: forward precision */
    theta_0k,         /* theta_0k: prior mean at k */
    theta_0kp1,       /* theta_0kp1: prior mean at k+1 */
    n                 /* n: number of observations */
  );
  PutRNGstate();

  UNPROTECT(protect_count);
  return result;
}

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
                           SEXP theta_0p_) {
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
  generate_theta_p(
    REAL(theta_pm1),  /* theta_pm1: state draws at p-1 */
    REAL(result),     /* result: output theta_p draws */
    prec_pm1,         /* prec_pm1: backward precision */
    prec_p,           /* prec_p: forward precision */
    theta_0p,         /* theta_0p: prior mean at p */
    n                 /* n: number of observations */
  );
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
                                       SEXP prec_theta_01_) {
  int protect_count = 0;
  SEXP theta_1 = PROTECT(coerceVector(theta_1_, REALSXP));
  protect_count++;
  double prec_theta_1 = require_real_scalar(prec_theta_1_, "prec_theta_1");
  double mean_theta_01 = require_real_scalar(mean_theta_01_, "mean_theta_01");
  double prec_theta_01 = require_real_scalar(prec_theta_01_, "prec_theta_01");
  int n = LENGTH(theta_1);

  GetRNGstate();
  double sample = generate_theta_01_locallevel(
    REAL(theta_1),     /* theta_1: first-state draws */
    prec_theta_1,      /* prec_theta_1: state precision */
    mean_theta_01,     /* mean_theta_01: prior mean */
    prec_theta_01,     /* prec_theta_01: prior precision */
    n                  /* n: number of observations */
  );
  PutRNGstate();

  UNPROTECT(protect_count);
  return ScalarReal(sample);
}

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
                            SEXP prec_theta_01_) {
  int protect_count = 0;
  SEXP theta_1 = PROTECT(coerceVector(theta_1_, REALSXP));
  protect_count++;
  double theta_02 = require_real_scalar(theta_02_, "theta_02");
  double prec_theta_1 = require_real_scalar(prec_theta_1_, "prec_theta_1");
  double mean_theta_01 = require_real_scalar(mean_theta_01_, "mean_theta_01");
  double prec_theta_01 = require_real_scalar(prec_theta_01_, "prec_theta_01");
  int n = LENGTH(theta_1);

  GetRNGstate();
  double sample = generate_theta_01(
    REAL(theta_1),    /* theta_1: first-state draws */
    theta_02,         /* theta_02: prior mean for theta_2 */
    prec_theta_1,     /* prec_theta_1: state precision */
    mean_theta_01,    /* mean_theta_01: prior mean */
    prec_theta_01,    /* prec_theta_01: prior precision */
    n                 /* n: number of observations */
  );
  PutRNGstate();

  UNPROTECT(protect_count);
  return ScalarReal(sample);
}

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
                            SEXP prec_0k_) {
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
  double sample = generate_theta_0k(
    REAL(theta_km1),  /* theta_km1: state draws at k-1 */
    REAL(theta_k),    /* theta_k: state draws at k */
    theta_0km1,       /* theta_0km1: prior mean at k-1 */
    theta_0kp1,       /* theta_0kp1: prior mean at k+1 */
    prec_km1,         /* prec_km1: backward precision */
    prec_k,           /* prec_k: forward precision */
    mean_0k,          /* mean_0k: prior mean hyperparameter */
    prec_0k,          /* prec_0k: prior precision hyperparameter */
    n                 /* n: number of observations */
  );
  PutRNGstate();

  UNPROTECT(protect_count);
  return ScalarReal(sample);
}

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
                            SEXP prec_0p_) {
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
  double sample = generate_theta_0p(
    REAL(theta_pm1),  /* theta_pm1: state draws at p-1 */
    REAL(theta_p),    /* theta_p: state draws at p */
    theta_0pm1,       /* theta_0pm1: prior mean at p-1 */
    prec_pm1,         /* prec_pm1: backward precision */
    prec_p,           /* prec_p: forward precision */
    mean_0p,          /* mean_0p: prior mean hyperparameter */
    prec_0p,          /* prec_0p: prior precision hyperparameter */
    n                 /* n: number of observations */
  );
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
                                               SEXP log_sigma_in_) {
  const int LAG_UPDATE = 10;
  const int ITER = 1;
  const int COMPUTE_ALPHA = 1;

  int protect_count = 0;
  SEXP theta_1_prev = PROTECT(coerceVector(theta_1_in_, REALSXP));
  protect_count++;
  double theta_01_prev = require_real_scalar(theta_01_in_, "theta_01_in");
  double prec_theta1_prev = require_real_scalar(prec_theta1_in_, "prec_theta1_in");
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
  cwmh_alpha_logit_binomial_locallevel(
    REAL(theta_1_prev),  /* theta_1_previous: previous theta_1 draws */
    theta_1_current,     /* theta_1_current: output theta_1 values */
    alpha_current,       /* alpha_current: output probabilities */
    theta_01_prev,       /* theta_01_previous: prior mean */
    prec_theta1_prev,         /* prec_theta1_previous: prior precision */
    theta_1_updated,     /* theta_1_updated: sliding window states */
    REAL(y),             /* y: observed counts */
    REAL(log_sigma),     /* log_sigma: proposal log standard deviations */
    hat_theta_1,         /* hat_theta_1: conditional means */
    theta_1_new,         /* theta_1_new: proposal buffer */
    log_accept_prob,     /* log_accept_prob: log acceptance storage */
    LAG_UPDATE,          /* lag_update: adaptation window length */
    n_trials,            /* n_trials: number of binomial trials */
    n,                   /* n: number of observations */
    ITER,                /* iter: current iteration */
    COMPUTE_ALPHA        /* compute_alpha: flag to compute alpha */
  );
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
                                    SEXP n_trials_) {
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
  double prec_theta1_prev = require_real_scalar(prec_theta1_in_, "prec_theta1_in");
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
  cwmh_alpha_logit_binomial(
    REAL(theta_1_prev),  /* theta_1_previous: previous theta_1 draws */
    theta_1_current,     /* theta_1_current: output theta_1 values */
    REAL(theta_2_curr),  /* theta_2_current: current theta_2 draws */
    alpha_current,       /* alpha_current: output probabilities */
    theta_01_prev,       /* theta_01_previous: prior mean for theta_1 */
    theta_02_prev,       /* theta_02_previous: prior mean for theta_2 */
    prec_theta1_prev,         /* prec_theta1_previous: prior precision */
    theta_1_updated,     /* theta_1_updated: sliding window states */
    REAL(y),             /* y: observed counts */
    log_sigma,           /* log_sigma: proposal log standard deviations */
    hat_theta_1,         /* hat_theta_1: conditional means */
    theta_1_new,         /* theta_1_new: proposal buffer */
    log_accept_prob,     /* log_accept_prob: log acceptance storage */
    LAG_UPDATE,          /* lag_update: adaptation window length */
    n_trials,            /* n_trials: number of binomial trials */
    n,                   /* n: number of observations */
    ITER,                /* iter: current iteration */
    COMPUTE_ALPHA        /* compute_alpha: flag to compute alpha */
  );
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
                                                   SEXP n_trials_) {
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
  double prec_theta1_prev = require_real_scalar(prec_theta1_in_, "prec_theta1_in");
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
  generate_alpha_logit_binomial_locallevel(
    REAL(theta_1_prev),              /* theta_1_previous: level states from previous iteration */
    theta_1_current,                 /* theta_1_current: output level states for current iteration */
    alpha_current,                   /* alpha_current: output probabilities */
    theta_01_prev,                   /* theta_01_previous: initial level state from previous */
    prec_theta1_prev,                     /* prec_theta1_previous: level precision from previous */
    theta_1_updated,                 /* theta_1_updated: sliding window acceptance indicators */
    REAL(y),                         /* y: observed counts */
    accept_prop,                     /* accept_prop: acceptance proportions */
    log_sigma,                       /* log_sigma: proposal log standard deviations */
    hat_theta_1,                     /* hat_theta_1: conditional means */
    theta_1_new,                     /* theta_1_new: proposal buffer */
    log_accept_prob,                 /* log_accept_prob: log acceptance storage */
    LAG_UPDATE,                      /* lag_update: adaptation window length */
    n_trials,                        /* n_trials: number of binomial trials */
    n,                               /* n: number of observations */
    ITER,                            /* iter: current iteration */
    MAX_STEP,                        /* max_step_size: adaptation step cap */
    BASE_ADAPT,                      /* base_adaptation_rate: initial adaptation rate */
    DECAY,                           /* decay_exponent: diminishing schedule */
    TARGET,                          /* target_acceptance: desired acceptance proportion */
    MIN_DEV,                         /* min_deviation_threshold: deviation trigger */
    1                                /* compute_alpha: flag to compute alpha (1 = compute) */
  );
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
 * @details Exposes the adaptive alpha generator with deterministic tuning so
 *          that regression tests can verify joint state updates.
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
                                        SEXP n_trials_) {
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
  double prec_theta1_prev = require_real_scalar(prec_theta1_in_, "prec_theta1_in");
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
  generate_alpha_logit_binomial(
    REAL(theta_1_prev),              /* theta_1_previous: level states from previous iteration */
    theta_1_current,                 /* theta_1_current: output level states for current iteration */
    alpha_current,                   /* alpha_current: output probabilities */
    REAL(theta_2_curr),              /* theta_2_current: companion state draws */
    theta_01_prev,                   /* theta_01_previous: initial level state from previous */
    theta_02_prev,                   /* theta_02_previous: second-state prior mean */
    prec_theta1_prev,                     /* prec_theta1_previous: level precision from previous */
    theta_1_updated,                 /* theta_1_updated: sliding window acceptance indicators */
    REAL(y),                         /* y: observed counts */
    accept_prop,                     /* accept_prop: acceptance proportions */
    log_sigma,                       /* log_sigma: proposal log standard deviations */
    hat_theta_1,                     /* hat_theta_1: conditional means */
    theta_1_new,                     /* theta_1_new: proposal buffer */
    log_accept_prob,                 /* log_accept_prob: log acceptance storage */
    LAG_UPDATE,                      /* lag_update: adaptation window length */
    n_trials,                        /* n_trials: number of binomial trials */
    n,                               /* n: number of observations */
    ITER,                            /* iter: current iteration */
    MAX_STEP,                        /* max_step_size: adaptation step cap */
    BASE_ADAPT,                      /* base_adaptation_rate: initial adaptation rate */
    DECAY,                           /* decay_exponent: diminishing schedule */
    TARGET,                          /* target_acceptance: desired acceptance proportion */
    MIN_DEV,                         /* min_deviation_threshold: deviation trigger */
    1                                /* compute_alpha: flag to compute alpha (1 = compute) */
  );
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
                                                     SEXP y_) {
  int protect_count = 0;
  SEXP theta_1_prev = PROTECT(coerceVector(theta_1_in_, REALSXP));
  protect_count++;
  double theta_01_prev = require_real_scalar(theta_01_in_, "theta_01_in");
  double prec_theta1_prev = require_real_scalar(prec_theta1_in_, "prec_theta1_in");
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  ensure_length(y, LENGTH(theta_1_prev), "y", "theta_1_in");
  int n = LENGTH(theta_1_prev);

  double *theta_1_current = (double *) R_alloc(n, sizeof(double));
  double *alpha_current = (double *) R_alloc(n, sizeof(double));
  double *rhs_vector = (double *) R_alloc(n, sizeof(double));

  GetRNGstate();
  generate_alpha_probit_bernoulli_locallevel(
    REAL(theta_1_prev),  /* theta_1_previous: previous theta_1 draws */
    theta_1_current,     /* theta_1_current: output theta_1 values */
    alpha_current,       /* alpha_current: output probabilities */
    theta_01_prev,       /* theta_01_previous: prior mean */
    prec_theta1_prev,         /* prec_theta1_previous: prior precision */
    REAL(y),             /* y: observed Bernoulli outcomes */
    rhs_vector,          /* rhs_vector: working buffer */
    n,                   /* n: number of observations */
    1                    /* compute_alpha: flag to compute alpha */
  );
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
                                          SEXP y_) {
  int protect_count = 0;
  SEXP theta_1_prev = PROTECT(coerceVector(theta_1_in_, REALSXP));
  protect_count++;
  SEXP theta_2_curr = PROTECT(coerceVector(theta_2_in_, REALSXP));
  protect_count++;
  ensure_length(theta_2_curr, LENGTH(theta_1_prev), "theta_2_in", "theta_1_in");
  double theta_01_prev = require_real_scalar(theta_01_in_, "theta_01_in");
  double theta_02_prev = require_real_scalar(theta_02_in_, "theta_02_in");
  double prec_theta1_prev = require_real_scalar(prec_theta1_in_, "prec_theta1_in");
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  ensure_length(y, LENGTH(theta_1_prev), "y", "theta_1_in");
  int n = LENGTH(theta_1_prev);

  double *theta_1_current = (double *) R_alloc(n, sizeof(double));
  double *alpha_current = (double *) R_alloc(n, sizeof(double));
  double *rhs_vector = (double *) R_alloc(n, sizeof(double));

  GetRNGstate();
  generate_alpha_probit_bernoulli(
    REAL(theta_1_prev),  /* theta_1_previous: previous theta_1 draws */
    theta_1_current,     /* theta_1_current: output theta_1 values */
    alpha_current,       /* alpha_current: output probabilities */
    REAL(theta_2_curr),  /* theta_2_current: companion state draws */
    theta_01_prev,       /* theta_01_previous: prior mean for theta_1 */
    theta_02_prev,       /* theta_02_previous: prior mean for theta_2 */
    prec_theta1_prev,         /* prec_theta1_previous: prior precision */
    REAL(y),             /* y: observed Bernoulli outcomes */
    rhs_vector,          /* rhs_vector: working buffer */
    n,                   /* n: number of observations */
    1                    /* compute_alpha: flag to compute alpha */
  );
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
// POISSON CWMH WRAPPERS
//==============================================================================

/**
 * @brief Run a single CWMH update for the log-Poisson local level model.
 *
 * @details Provides deterministic wrapper around production sampler by setting
 *          small adaptation windows and exposing resulting states and acceptance
 *          diagnostics.
 *
 * @param theta_1_in_     Numeric vector with previous state draws.
 * @param theta_01_in_    Scalar prior mean for initial state.
 * @param prec_theta1_in_ Scalar prior precision for state.
 * @param y_              Observed Poisson counts.
 * @param log_sigma_in_   Numeric vector of proposal log standard deviations.
 *
 * @return A list containing updated state draws and log-scale rates (alpha).
 */
SEXP test_cwmh_alpha_log_poisson_locallevel(SEXP theta_1_in_,
                                            SEXP theta_01_in_,
                                            SEXP prec_theta1_in_,
                                            SEXP y_,
                                            SEXP log_sigma_in_) {
  const int LAG_UPDATE = 10;
  const int ITER = 1;
  const int COMPUTE_ALPHA = 1;

  int protect_count = 0;
  SEXP theta_1_prev = PROTECT(coerceVector(theta_1_in_, REALSXP));
  protect_count++;
  double theta_01_prev = require_real_scalar(theta_01_in_, "theta_01_in");
  double prec_theta1_prev = require_real_scalar(prec_theta1_in_, "prec_theta1_in");
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  ensure_length(y, LENGTH(theta_1_prev), "y", "theta_1_in");
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
  cwmh_alpha_log_poisson_locallevel(
    REAL(theta_1_prev),  /* theta_1_previous */
    theta_1_current,     /* theta_1_current */
    alpha_current,       /* alpha_current */
    theta_01_prev,       /* theta_01_previous */
    prec_theta1_prev,    /* prec_theta1_previous */
    theta_1_updated,     /* theta_1_updated */
    REAL(y),             /* y */
    REAL(log_sigma),     /* log_sigma */
    hat_theta_1,         /* hat_theta_1 */
    theta_1_new,         /* theta_1_new */
    log_accept_prob,     /* log_accept_prob */
    LAG_UPDATE,          /* lag_update */
    n,                   /* n */
    ITER,                /* iter */
    COMPUTE_ALPHA        /* compute_alpha */
  );
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
 * @brief Run a single CWMH update for the two-parameter log-Poisson local trend model.
 *
 * @details Mirrors production sampler while keeping proposals fixed so that
 *          unit tests can inspect latent states and alpha values.
 *
 * @param theta_1_in_     Numeric vector with previous level state draws.
 * @param theta_2_in_     Numeric vector with current trend state draws.
 * @param theta_01_in_    Scalar prior mean for initial level.
 * @param theta_02_in_    Scalar prior mean for initial trend.
 * @param prec_theta1_in_ Scalar prior precision for level.
 * @param y_              Observed Poisson counts.
 *
 * @return A list containing updated level state draws and log-scale rates.
 */
SEXP test_cwmh_alpha_log_poisson(SEXP theta_1_in_,
                                 SEXP theta_2_in_,
                                 SEXP theta_01_in_,
                                 SEXP theta_02_in_,
                                 SEXP prec_theta1_in_,
                                 SEXP y_) {
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
  double prec_theta1_prev = require_real_scalar(prec_theta1_in_, "prec_theta1_in");
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  ensure_length(y, LENGTH(theta_1_prev), "y", "theta_1_in");
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
  cwmh_alpha_log_poisson(
    REAL(theta_1_prev),  /* theta_1_previous */
    theta_1_current,     /* theta_1_current */
    alpha_current,       /* alpha_current */
    REAL(theta_2_curr),  /* theta_2_current */
    theta_01_prev,       /* theta_01_previous */
    theta_02_prev,       /* theta_02_previous */
    prec_theta1_prev,    /* prec_theta1_previous */
    theta_1_updated,     /* theta_1_updated */
    REAL(y),             /* y */
    log_sigma,           /* log_sigma */
    hat_theta_1,         /* hat_theta_1 */
    theta_1_new,         /* theta_1_new */
    log_accept_prob,     /* log_accept_prob */
    LAG_UPDATE,          /* lag_update */
    n,                   /* n */
    ITER,                /* iter */
    COMPUTE_ALPHA        /* compute_alpha */
  );
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
// POISSON ALPHA GENERATION WRAPPERS
//==============================================================================

/**
 * @brief Adaptively generate alphas for log-Poisson local level model.
 *
 * @details Runs generate_alpha_log_poisson_locallevel with predetermined tuning
 *          constants so that tests can confirm adaptive behaviour.
 *
 * @param theta_1_in_     Numeric vector with previous state draws.
 * @param theta_01_in_    Scalar prior mean for initial state.
 * @param prec_theta1_in_ Scalar prior precision for state.
 * @param y_              Observed Poisson counts.
 *
 * @return A list containing updated state draws and log-scale rates.
 */
SEXP test_generate_alpha_log_poisson_locallevel(SEXP theta_1_in_,
                                                SEXP theta_01_in_,
                                                SEXP prec_theta1_in_,
                                                SEXP y_) {
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
  double prec_theta1_prev = require_real_scalar(prec_theta1_in_, "prec_theta1_in");
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  ensure_length(y, LENGTH(theta_1_prev), "y", "theta_1_in");
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
  generate_alpha_log_poisson_locallevel(
    REAL(theta_1_prev),   /* theta_1_previous */
    theta_1_current,      /* theta_1_current */
    alpha_current,        /* alpha_current */
    theta_01_prev,        /* theta_01_previous */
    prec_theta1_prev,     /* prec_theta1_previous */
    theta_1_updated,      /* theta_1_updated */
    REAL(y),              /* y */
    accept_prop,          /* accept_prop */
    log_sigma,            /* log_sigma */
    hat_theta_1,          /* hat_theta_1 */
    theta_1_new,          /* theta_1_new */
    log_accept_prob,      /* log_accept_prob */
    LAG_UPDATE,           /* lag_update */
    n,                    /* n */
    ITER,                 /* iter */
    MAX_STEP,             /* max_step_size */
    BASE_ADAPT,           /* base_adaptation_rate */
    DECAY,                /* decay_exponent */
    TARGET,               /* target_acceptance */
    MIN_DEV,              /* min_deviation_threshold */
    1                     /* compute_alpha */
  );
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
 * @brief Adaptively generate alphas for two-parameter log-Poisson local trend model.
 *
 * @details Exposes adaptive alpha generator with deterministic tuning so that
 *          regression tests can verify joint state updates.
 *
 * @param theta_1_in_     Numeric vector with previous level state draws.
 * @param theta_2_in_     Numeric vector with current trend state draws.
 * @param theta_01_in_    Scalar prior mean for initial level.
 * @param theta_02_in_    Scalar prior mean for initial trend.
 * @param prec_theta1_in_ Scalar prior precision for level.
 * @param y_              Observed Poisson counts.
 *
 * @return A list containing updated level state draws and log-scale rates.
 */
SEXP test_generate_alpha_log_poisson(SEXP theta_1_in_,
                                     SEXP theta_2_in_,
                                     SEXP theta_01_in_,
                                     SEXP theta_02_in_,
                                     SEXP prec_theta1_in_,
                                     SEXP y_) {
  const int LAG_UPDATE = 50;
  const int ITER = 1;
  const int COMPUTE_ALPHA = 1;
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
  double prec_theta1_prev = require_real_scalar(prec_theta1_in_, "prec_theta1_in");
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  protect_count++;
  ensure_length(y, LENGTH(theta_1_prev), "y", "theta_1_in");
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
  generate_alpha_log_poisson(
    REAL(theta_1_prev),   /* theta_1_previous */
    theta_1_current,      /* theta_1_current */
    alpha_current,        /* alpha_current */
    REAL(theta_2_curr),   /* theta_2_current */
    theta_01_prev,        /* theta_01_previous */
    theta_02_prev,        /* theta_02_previous */
    prec_theta1_prev,     /* prec_theta1_previous */
    theta_1_updated,      /* theta_1_updated */
    REAL(y),              /* y */
    accept_prop,          /* accept_prop */
    log_sigma,            /* log_sigma */
    hat_theta_1,          /* hat_theta_1 */
    theta_1_new,          /* theta_1_new */
    log_accept_prob,      /* log_accept_prob */
    LAG_UPDATE,           /* lag_update */
    n,                    /* n */
    ITER,                 /* iter */
    MAX_STEP,             /* max_step_size */
    BASE_ADAPT,           /* base_adaptation_rate */
    DECAY,                /* decay_exponent */
    TARGET,               /* target_acceptance */
    MIN_DEV,              /* min_deviation_threshold */
    COMPUTE_ALPHA         /* compute_alpha */
  );
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
// POISSON COMPLETE MCMC WRAPPERS
//==============================================================================

/**
 * @brief Execute the full MCMC sampler for the log-Poisson local level model with parameter fixing.
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
 * @param y_                       Observed Poisson counts [n].
 * @param burnin_                  Number of burn-in iterations (discarded).
 * @param thinning_                Thinning interval for retained samples.
 * @param n_chain_                 Number of chains to simulate.
 * @param theta_1_true_            Optional:  true theta_1 values [n] to fix (NULL = sample normally).
 * @param theta_01_true_           Optional: true theta_01 value to fix (NULL = sample normally).
 * @param prec_theta1_true_        Optional: true prec_theta1 value to fix (NULL = sample normally).
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
 * @return An R list mirroring the production sampler output.
 *
 * @note Validates Poisson constraints:  y[i] >= 0 for all i.
 * @note Validates parameter positivity: precisions > 0.
 * @note When parameters are fixed, corresponding posterior samples are constant.
 *
 * @warning This function is for testing only. Do not use for production inference.
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
                                                   SEXP return_accept_prop_) {

  /* ========== Parse Data Vector and Validate Length ========== */
  double   *y_ptr = REAL(y_);
  R_xlen_t  len   = LENGTH(y_);

  /* Enforce minimum sample size for numerical stability */
  if (len < 3) {
    error("test_mcmc_log_poisson_locallevel_fixed_params: sample size 'n' must be at least 3, got %lld",
          (long long) len);
  }

  /* Check for integer overflow (R limitation) */
  if (len > INT_MAX) {
    error("test_mcmc_log_poisson_locallevel_fixed_params: sample size too large (%lld > %d)",
          (long long) len, INT_MAX);
  }

  int n = (int) len;

  /* Validate Poisson constraints:  y[i] >= 0 */
  for (int t = 0; t < n; t++) {
    if (y_ptr[t] < 0) {
      error("test_mcmc_log_poisson_locallevel_fixed_params:  y[%d] = %f must be non-negative",
            t, y_ptr[t]);
    }
    /* Also check for non-integer values (optional but recommended) */
    if (y_ptr[t] != floor(y_ptr[t])) {
      error("test_mcmc_log_poisson_locallevel_fixed_params: y[%d] = %f must be an integer",
            t, y_ptr[t]);
    }
  }

  /* ========== Parse MCMC Control Parameters ========== */
  int burnin   = INTEGER(burnin_)[0];     /* Burn-in iterations to discard */
  int thinning = INTEGER(thinning_)[0];   /* Thinning interval */
  int n_chain  = INTEGER(n_chain_)[0];    /* Number of retained samples */

  /* Compute total iterations needed */
  int n_iter = burnin + (n_chain - 1) * thinning + 1;

  /* ========== Parse Prior Hyperparameters ========== */
  double mean_theta01 = REAL(prior_theta01_mean_)[0]; /* Prior mean for theta_{0,1} */
  double prec_theta01 = REAL(prior_theta01_prec_)[0]; /* Prior precision for theta_{0,1} */
  double nu_01        = REAL(prior_prec1_shape_)[0];  /* Gamma shape for 1/W_1 */
  double eta_01       = REAL(prior_prec1_rate_)[0];   /* Gamma rate for 1/W_1 */

  /* Validate prior parameter positivity */
  if (prec_theta01 <= 0) {
    error("test_mcmc_log_poisson_locallevel_fixed_params:  prior_theta01_prec must be positive, got %f",
          prec_theta01);
  }
  if (nu_01 <= 0 || eta_01 <= 0) {
    error("test_mcmc_log_poisson_locallevel_fixed_params:  Gamma prior parameters must be positive");
  }

  /* ========== Parse Adaptation Parameters ========== */
  int    lag_update              = INTEGER(lag_update_)[0];
  double max_step_size           = REAL(max_step_size_)[0];
  double base_adaptation_rate    = REAL(base_adaptation_rate_)[0];
  double decay_exponent          = REAL(decay_exponent_)[0];
  double target_acceptance       = REAL(target_acceptance_)[0];
  double min_deviation_threshold = 1.0 / (double)lag_update;

  /* ========== Parse Diagnostic Output Options ========== */
  int return_log_sigma    = LOGICAL(return_log_sigma_)[0];
  int return_accept_prop  = LOGICAL(return_accept_prop_)[0];

  /* ========== Check Which Parameters Should Be Fixed ========== */
  int fix_theta_1      = (theta_1_true_ != R_NilValue && ! Rf_isNull(theta_1_true_));
  int fix_theta_01     = (theta_01_true_ != R_NilValue && !Rf_isNull(theta_01_true_));
  int fix_prec_theta1  = (prec_theta1_true_ != R_NilValue && !Rf_isNull(prec_theta1_true_));

  /* Extract true values if provided */
  double *theta_1_true   = NULL;
  double  theta_01_true  = 0.0;
  double  prec_theta1_true = 0.0;

  int protect_count = 0;

  if (fix_theta_1) {
    SEXP theta_1_tmp = PROTECT(coerceVector(theta_1_true_, REALSXP));
    protect_count++;
    if (LENGTH(theta_1_tmp) != n) {
      UNPROTECT(protect_count);
      error("test_mcmc_log_poisson_locallevel_fixed_params: theta_1_true must have length %d, got %d",
            n, LENGTH(theta_1_tmp));
    }
    theta_1_true = REAL(theta_1_tmp);
  }

  if (fix_theta_01) {
    theta_01_true = require_real_scalar(theta_01_true_, "theta_01_true");
  }

  if (fix_prec_theta1) {
    prec_theta1_true = require_real_scalar(prec_theta1_true_, "prec_theta1_true");
    if (prec_theta1_true <= 0) {
      UNPROTECT(protect_count);
      error("test_mcmc_log_poisson_locallevel_fixed_params: prec_theta1_true must be positive, got %f",
            prec_theta1_true);
    }
  }

  /* ========== Allocate Output Storage (Retained Samples Only) ========== */
  SEXP theta_1_samples      = PROTECT(allocMatrix(REALSXP, n_chain, n));
  protect_count++;
  SEXP theta_01_samples     = PROTECT(allocVector(REALSXP, n_chain));
  protect_count++;
  SEXP prec_theta1_samples  = PROTECT(allocVector(REALSXP, n_chain));
  protect_count++;
  SEXP alpha_samples        = PROTECT(allocMatrix(REALSXP, n_chain, n));
  protect_count++;

  /* Conditional allocation for diagnostics */
  SEXP log_sigma_samples   = R_NilValue;
  SEXP accept_prop_samples = R_NilValue;
  int n_outputs = 4;  /* Base outputs:  theta_1, theta_01, prec_theta1, alpha */

  if (return_log_sigma) {
    log_sigma_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
    protect_count++;
    n_outputs++;
  }
  if (return_accept_prop) {
    accept_prop_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
    protect_count++;
    n_outputs++;
  }

  /* ========== Allocate Temporary Buffers (Memory-Efficient O(n) Storage) ========== */
  double *theta_1_current   = (double *) R_Calloc(n, double);
  double *theta_1_previous  = (double *) R_Calloc(n, double);
  double *alpha_current     = (double *) R_Calloc(n, double);

  /* Scalar parameters for current and previous iterations */
  double theta_01_current, theta_01_previous;
  double prec_theta1_current, prec_theta1_previous;

  /* Sliding window buffer for acceptance tracking */
  double *theta_1_updated  = (double *) R_Calloc(lag_update * n, double);
  memset(theta_1_updated, 0, lag_update * n * sizeof(double));

  /* Working arrays for CWMH algorithm */
  double *accept_prop      = (double *) R_Calloc(n, double);
  double *log_sigma        = (double *) R_Calloc(n, double);
  double *hat_theta_1      = (double *) R_Calloc(n, double);
  double *theta_1_new      = (double *) R_Calloc(n, double);
  double *log_accept_prob  = (double *) R_Calloc(n, double);

  /* Initialize log_sigma with reasonable starting values */
  for (int t = 0; t < n; t++) {
    log_sigma[t] = log(0.1);  /* Initial proposal sd = 0.1 */
  }

  /* ========== Initialize RNG State ========== */
  GetRNGstate();

  /* ========== Initialize Parameters (Iteration 0) ========== */
  /* Draw initial values from priors to start the Markov chain */
  theta_01_previous = fix_theta_01 ? theta_01_true : rnorm(mean_theta01, sqrt(1.0 / prec_theta01));
  prec_theta1_previous = fix_prec_theta1 ?  prec_theta1_true :  rgamma(nu_01, 1.0 / eta_01);

  /* Initialize theta_1 and alpha with efficient neutral starting values */
  if (fix_theta_1) {
    memcpy(theta_1_previous, theta_1_true, n * sizeof(double));
    for (int t = 0; t < n; t++) {
      alpha_current[t] = exp(theta_1_true[t]);
    }
  } else {
    for (int t = 0; t < n; t++) {
      theta_1_previous[t] = 0.0;   /* Zeros for state trajectory */
  alpha_current[t]    = 1.0;   /* Neutral rate for Poisson */
    }
  }

  /* ========== Main Gibbs Sampling Loop ========== */
  int chain_idx = 0;

  for (int ii = 1; ii < n_iter; ii++) {

    /* Determine whether to compute alpha transformations */
    int compute_alpha = (ii >= burnin && ((ii - burnin) % thinning) == 0) ?  1 : 0;

    /* ===== Step 1: Sample State Vector theta_1 and Rates alpha ===== */
    if (fix_theta_1) {
      /* Use fixed true values instead of sampling */
      memcpy(theta_1_current, theta_1_true, n * sizeof(double));
      if (compute_alpha) {
        for (int t = 0; t < n; t++) {
          alpha_current[t] = exp(theta_1_true[t]);
        }
      }
    } else {
      /* Sample theta_1 | y, theta_01, prec_theta1 using component-wise Metropolis-Hastings */
      generate_alpha_log_poisson_locallevel(
        theta_1_previous,              /* theta_1_previous:  level states from previous iteration */
      theta_1_current,               /* theta_1_current: output level states for current iteration */
      compute_alpha ?  alpha_current : NULL,  /* alpha_current: output rates */
      theta_01_previous,             /* theta_01_previous: initial level state from previous */
      prec_theta1_previous,          /* prec_theta1_previous: level precision from previous */
      theta_1_updated,               /* theta_1_updated: sliding window acceptance indicators */
      y_ptr,                         /* y:  observed counts */
      accept_prop,                   /* accept_prop: acceptance proportions */
      log_sigma,                     /* log_sigma: proposal log standard deviations */
      hat_theta_1,                   /* hat_theta_1: conditional means */
      theta_1_new,                   /* theta_1_new: proposal buffer */
      log_accept_prob,               /* log_accept_prob: log acceptance storage */
      lag_update,                    /* lag_update: adaptation window length */
      n,                             /* n: number of observations */
      ii,                            /* iter: current iteration */
      max_step_size,                 /* max_step_size: adaptation step cap */
      base_adaptation_rate,          /* base_adaptation_rate: initial adaptation rate */
      decay_exponent,                /* decay_exponent: diminishing schedule */
      target_acceptance,             /* target_acceptance: desired acceptance proportion */
      min_deviation_threshold,       /* min_deviation_threshold: deviation trigger */
      compute_alpha                  /* compute_alpha: flag to compute alpha (1 = compute) */
      );
    }

    /* ===== Step 2: Sample Innovation Precision 1/W_1 ===== */
    if (fix_prec_theta1) {
      /* Use fixed true value instead of sampling */
      prec_theta1_current = prec_theta1_true;
    } else {
      /* Sample 1/W_1 | theta_1, theta_01 from Gamma posterior */
      prec_theta1_current = generate_precision_theta_p(
        theta_01_previous,         /* theta_0p: initial level from previous iteration */
      theta_1_current,           /* theta_p_current: current level trajectory [n] */
      nu_01,                     /* nu_0p: prior shape parameter */
      eta_01,                    /* eta_0p:  prior rate parameter */
      n                          /* n: number of time points */
      );
    }

    /* ===== Step 3: Sample Initial State theta_{0,1} ===== */
    if (fix_theta_01) {
      /* Use fixed true value instead of sampling */
      theta_01_current = theta_01_true;
    } else {
      /* Sample theta_{0,1} | theta_1, prec_theta1 from Normal posterior */
      theta_01_current = generate_theta_01_locallevel(
        theta_1_current,           /* theta_1_current: current level trajectory [n] */
      prec_theta1_current,       /* prec_theta1: current level precision */
      mean_theta01,              /* mean_theta01: prior mean */
      prec_theta01,              /* prec_theta01: prior precision */
      n                          /* n: number of time points */
      );
    }

    /* ===== Store Post-Burn-in Samples with Thinning ===== */
    if (compute_alpha) {
      int idx = chain_idx++;

      /* Copy current theta_1 and alpha to output matrices (column-major) */
      for (int t = 0; t < n; t++) {
        REAL(theta_1_samples)[idx + t * n_chain] = theta_1_current[t];
        REAL(alpha_samples)[idx + t * n_chain] = alpha_current[t];

        /* Store diagnostics if requested */
        if (return_log_sigma) {
          REAL(log_sigma_samples)[idx + t * n_chain] = log_sigma[t];
        }
        if (return_accept_prop) {
          REAL(accept_prop_samples)[idx + t * n_chain] = accept_prop[t];
        }
      }

      /* Copy scalar parameters to output vectors */
      REAL(theta_01_samples)[idx] = theta_01_current;
      REAL(prec_theta1_samples)[idx] = prec_theta1_current;
    }

    /* ===== Update Previous Values for Next Iteration ===== */
    memcpy(theta_1_previous, theta_1_current, n * sizeof(double));
    theta_01_previous = theta_01_current;
    prec_theta1_previous = prec_theta1_current;
  }

  /* ========== Restore RNG State ========== */
  PutRNGstate();

  /* ========== Free Temporary Buffers ========== */
  R_Free(theta_1_current);
  R_Free(theta_1_previous);
  R_Free(alpha_current);
  R_Free(theta_1_updated);
  R_Free(accept_prop);
  R_Free(log_sigma);
  R_Free(hat_theta_1);
  R_Free(theta_1_new);
  R_Free(log_accept_prob);

  /* ========== Package Results into Named List ========== */
  SEXP out = PROTECT(allocVector(VECSXP, n_outputs));
  protect_count++;
  SEXP names = PROTECT(allocVector(STRSXP, n_outputs));
  protect_count++;

  int out_idx = 0;
  SET_VECTOR_ELT(out, out_idx, theta_1_samples);
  SET_STRING_ELT(names, out_idx++, mkChar("theta_1"));

  SET_VECTOR_ELT(out, out_idx, theta_01_samples);
  SET_STRING_ELT(names, out_idx++, mkChar("theta_01"));

  SET_VECTOR_ELT(out, out_idx, prec_theta1_samples);
  SET_STRING_ELT(names, out_idx++, mkChar("prec_theta1"));

  SET_VECTOR_ELT(out, out_idx, alpha_samples);
  SET_STRING_ELT(names, out_idx++, mkChar("alpha"));

  if (return_log_sigma) {
    SET_VECTOR_ELT(out, out_idx, log_sigma_samples);
    SET_STRING_ELT(names, out_idx++, mkChar("log_sigma"));
  }

  if (return_accept_prop) {
    SET_VECTOR_ELT(out, out_idx, accept_prop_samples);
    SET_STRING_ELT(names, out_idx++, mkChar("accept_prop"));
  }

  setAttrib(out, R_NamesSymbol, names);

  UNPROTECT(protect_count);
  return out;
}

//==============================================================================
// COMPLETE MCMC WRAPPERS
//==============================================================================

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
                                                SEXP return_accept_prop_) {

  /* ========== Parse Data Vector and Validate Length ========== */
  double   *y_ptr = REAL(y_);
  R_xlen_t  len   = LENGTH(y_);

  /* Enforce minimum sample size for numerical stability */
  if (len < 3) {
    error("test_mcmc_binomial_locallevel_fixed_params: sample size 'n' must be at least 3, got %lld",
          (long long) len);
  }

  /* Check for integer overflow (R limitation) */
  if (len > INT_MAX) {
    error("test_mcmc_binomial_locallevel_fixed_params: sample size too large (%lld > %d)",
          (long long) len, INT_MAX);
  }

  int n = (int) len;

  /* ========== Parse Observation Model Parameters and Validate ========== */
  double n_trials = REAL(n_trials_)[0];

  /* Validate n_trials positivity */
  if (n_trials <= 0) {
    error("test_mcmc_binomial_locallevel_fixed_params: n_trials must be positive, got %f", n_trials);
  }

  /* Validate binomial constraints: 0 <= y[i] <= n_trials */
  for (int t = 0; t < n; t++) {
    if (y_ptr[t] < 0 || y_ptr[t] > n_trials) {
      error("test_mcmc_binomial_locallevel_fixed_params: y[%d] = %f violates 0 <= y <= n_trials = %f",
            t, y_ptr[t], n_trials);
    }
  }

  /* ========== Parse MCMC Control Parameters ========== */
  int burnin   = INTEGER(burnin_)[0];     /* Burn-in iterations to discard */
  int thinning = INTEGER(thinning_)[0];   /* Thinning interval */
  int n_chain  = INTEGER(n_chain_)[0];    /* Number of retained samples */

  /* Compute total iterations needed */
  int n_iter = burnin + (n_chain - 1) * thinning + 1;

  /* ========== Parse Prior Hyperparameters ========== */
  double mean_theta01 = REAL(prior_theta01_mean_)[0]; /* Prior mean for theta_{0,1} */
  double prec_theta01 = REAL(prior_theta01_prec_)[0]; /* Prior precision for theta_{0,1} */
  double nu_01        = REAL(prior_prec1_shape_)[0];  /* Gamma shape for 1/W_1 */
  double eta_01       = REAL(prior_prec1_rate_)[0];   /* Gamma rate for 1/W_1 */

  /* Validate prior parameter positivity */
  if (prec_theta01 <= 0) {
    error("test_mcmc_binomial_locallevel_fixed_params: prior_theta01_prec must be positive, got %f",
          prec_theta01);
  }
  if (nu_01 <= 0 || eta_01 <= 0) {
    error("test_mcmc_binomial_locallevel_fixed_params: Gamma prior parameters must be positive");
  }

  /* ========== Parse Adaptation Parameters ========== */
  int    lag_update              = INTEGER(lag_update_)[0];
  double max_step_size           = REAL(max_step_size_)[0];
  double base_adaptation_rate    = REAL(base_adaptation_rate_)[0];
  double decay_exponent          = REAL(decay_exponent_)[0];
  double target_acceptance       = REAL(target_acceptance_)[0];
  double min_deviation_threshold = 1.0 / (double)lag_update;

  /* ========== Parse Diagnostic Output Options ========== */
  int return_log_sigma    = LOGICAL(return_log_sigma_)[0];
  int return_accept_prop  = LOGICAL(return_accept_prop_)[0];

  /* ========== Check Which Parameters Should Be Fixed ========== */
  int fix_theta_1      = (theta_1_true_ != R_NilValue && !Rf_isNull(theta_1_true_));
  int fix_theta_01     = (theta_01_true_ != R_NilValue && !Rf_isNull(theta_01_true_));
  int fix_prec_theta1  = (prec_theta1_true_ != R_NilValue && !Rf_isNull(prec_theta1_true_));

  /* Extract true values if provided */
  double *theta_1_true   = NULL;
  double  theta_01_true  = 0.0;
  double  prec_theta1_true = 0.0;

  int protect_count = 0;

  if (fix_theta_1) {
    SEXP theta_1_tmp = PROTECT(coerceVector(theta_1_true_, REALSXP));
    protect_count++;
    if (LENGTH(theta_1_tmp) != n) {
      UNPROTECT(protect_count);
      error("test_mcmc_binomial_locallevel_fixed_params: theta_1_true must have length %d, got %d",
            n, LENGTH(theta_1_tmp));
    }
    theta_1_true = REAL(theta_1_tmp);
  }

  if (fix_theta_01) {
    theta_01_true = require_real_scalar(theta_01_true_, "theta_01_true");
  }

  if (fix_prec_theta1) {
    prec_theta1_true = require_real_scalar(prec_theta1_true_, "prec_theta1_true");
    if (prec_theta1_true <= 0) {
      UNPROTECT(protect_count);
      error("test_mcmc_binomial_locallevel_fixed_params: prec_theta1_true must be positive, got %f",
            prec_theta1_true);
    }
  }

  /* ========== Allocate Output Storage (Retained Samples Only) ========== */
  SEXP theta_1_samples      = PROTECT(allocMatrix(REALSXP, n_chain, n));
  protect_count++;
  SEXP theta_01_samples     = PROTECT(allocVector(REALSXP, n_chain));
  protect_count++;
  SEXP prec_theta1_samples  = PROTECT(allocVector(REALSXP, n_chain));
  protect_count++;
  SEXP alpha_samples        = PROTECT(allocMatrix(REALSXP, n_chain, n));
  protect_count++;

  /* Conditional allocation for diagnostics */
  SEXP log_sigma_samples   = R_NilValue;
  SEXP accept_prop_samples = R_NilValue;
  int n_outputs = 4;  /* Base outputs: theta_1, theta_01, prec_theta1, alpha */

  if (return_log_sigma) {
    log_sigma_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
    protect_count++;
    n_outputs++;
  }
  if (return_accept_prop) {
    accept_prop_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
    protect_count++;
    n_outputs++;
  }

  /* ========== Allocate Temporary Buffers (Memory-Efficient O(n) Storage) ========== */
  double *theta_1_current   = (double *) R_Calloc(n, double);
  double *theta_1_previous  = (double *) R_Calloc(n, double);
  double *alpha_current     = (double *) R_Calloc(n, double);

  /* Scalar parameters for current and previous iterations */
  double theta_01_current, theta_01_previous;
  double prec_theta1_current, prec_theta1_previous;

  /* Sliding window buffer for acceptance tracking */
  double *theta_1_updated  = (double *) R_Calloc(lag_update * n, double);
  memset(theta_1_updated, 0, lag_update * n * sizeof(double));

  /* Working arrays for CWMH algorithm */
  double *accept_prop      = (double *) R_Calloc(n, double);
  double *log_sigma        = (double *) R_Calloc(n, double);
  double *hat_theta_1      = (double *) R_Calloc(n, double);
  double *theta_1_new      = (double *) R_Calloc(n, double);
  double *log_accept_prob  = (double *) R_Calloc(n, double);

  /* Initialize log_sigma with reasonable starting values */
  for (int t = 0; t < n; t++) {
    log_sigma[t] = log(0.1);  /* Initial proposal sd = 0.1 */
  }

  /* ========== Initialize RNG State ========== */
  GetRNGstate();

  /* ========== Initialize Parameters (Iteration 0) ========== */
  /* Draw initial values from priors to start the Markov chain */
  theta_01_previous = fix_theta_01 ? theta_01_true : rnorm(mean_theta01, sqrt(1.0 / prec_theta01));
  prec_theta1_previous = fix_prec_theta1 ? prec_theta1_true : rgamma(nu_01, 1.0 / eta_01);

  /* Initialize theta_1 and alpha with efficient neutral starting values */
  if (fix_theta_1) {
    memcpy(theta_1_previous, theta_1_true, n * sizeof(double));
    for (int t = 0; t < n; t++) {
      alpha_current[t] = ilogit(theta_1_true[t]);
    }
  } else {
    for (int t = 0; t < n; t++) {
      theta_1_previous[t] = 0.0;   /* Zeros for state trajectory */
      alpha_current[t]    = 0.5;   /* Neutral probability for success rates */
    }
  }

  /* ========== Main Gibbs Sampling Loop ========== */
  int chain_idx = 0;

  for (int ii = 1; ii < n_iter; ii++) {

    /* Determine whether to compute alpha transformations */
    int compute_alpha = (ii >= burnin && ((ii - burnin) % thinning) == 0) ? 1 : 0;

    /* ===== Step 1: Sample State Vector theta_1 and Success Probabilities alpha ===== */
    if (fix_theta_1) {
      /* Use fixed true values instead of sampling */
      memcpy(theta_1_current, theta_1_true, n * sizeof(double));
      if (compute_alpha) {
        for (int t = 0; t < n; t++) {
          alpha_current[t] = ilogit(theta_1_true[t]);
        }
      }
    } else {
      /* Sample theta_1 | y, theta_01, prec_theta1 using component-wise Metropolis-Hastings */
      generate_alpha_logit_binomial_locallevel(
        theta_1_previous,              /* theta_1_previous: level states from previous iteration */
        theta_1_current,               /* theta_1_current: output level states for current iteration */
        compute_alpha ? alpha_current : NULL,  /* alpha_current: output probabilities */
        theta_01_previous,             /* theta_01_previous: initial level state from previous */
        prec_theta1_previous,               /* prec_theta1_previous: level precision from previous */
        theta_1_updated,               /* theta_1_updated: sliding window acceptance indicators */
        y_ptr,                         /* y: observed counts */
        accept_prop,                   /* accept_prop: acceptance proportions */
        log_sigma,                     /* log_sigma: proposal log standard deviations */
        hat_theta_1,                   /* hat_theta_1: conditional means */
        theta_1_new,                   /* theta_1_new: proposal buffer */
        log_accept_prob,               /* log_accept_prob: log acceptance storage */
        lag_update,                    /* lag_update: adaptation window length */
        n_trials,                      /* n_trials: number of binomial trials */
        n,                             /* n: number of observations */
        ii,                            /* iter: current iteration */
        max_step_size,                 /* max_step_size: adaptation step cap */
        base_adaptation_rate,          /* base_adaptation_rate: initial adaptation rate */
        decay_exponent,                /* decay_exponent: diminishing schedule */
        target_acceptance,             /* target_acceptance: desired acceptance proportion */
        min_deviation_threshold,       /* min_deviation_threshold: deviation trigger */
        compute_alpha                  /* compute_alpha: flag to compute alpha (1 = compute) */
      );
    }

    /* ===== Step 2: Sample Innovation Precision 1/W_1 ===== */
    if (fix_prec_theta1) {
      /* Use fixed true value instead of sampling */
      prec_theta1_current = prec_theta1_true;
    } else {
      /* Sample 1/W_1 | theta_1, theta_01 from Gamma posterior */
      prec_theta1_current = generate_precision_theta_p(
        theta_01_previous,         /* theta_0p: initial level from previous iteration */
        theta_1_current,           /* theta_p_current: current level trajectory [n] */
        nu_01,                     /* nu_0p: prior shape parameter */
        eta_01,                    /* eta_0p: prior rate parameter */
        n                          /* n: number of time points */
      );
    }

    /* ===== Step 3: Sample Initial State theta_{0,1} ===== */
    if (fix_theta_01) {
      /* Use fixed true value instead of sampling */
      theta_01_current = theta_01_true;
    } else {
      /* Sample theta_{0,1} | theta_1, prec_theta1 from Normal posterior */
      theta_01_current = generate_theta_01_locallevel(
        theta_1_current,           /* theta_1_current: current level trajectory [n] */
        prec_theta1_current,            /* prec_theta1: current level precision */
        mean_theta01,              /* mean_theta01: prior mean */
        prec_theta01,              /* prec_theta01: prior precision */
        n                          /* n: number of time points */
      );
    }

    /* ===== Store Post-Burn-in Samples with Thinning ===== */
    if (compute_alpha) {
      int idx = chain_idx++;

      /* Copy current theta_1 and alpha to output matrices (column-major) */
      for (int t = 0; t < n; t++) {
        REAL(theta_1_samples)[idx + t * n_chain] = theta_1_current[t];
        REAL(alpha_samples)[idx + t * n_chain] = alpha_current[t];

        /* Store diagnostics if requested */
        if (return_log_sigma) {
          REAL(log_sigma_samples)[idx + t * n_chain] = log_sigma[t];
        }
        if (return_accept_prop) {
          REAL(accept_prop_samples)[idx + t * n_chain] = accept_prop[t];
        }
      }

      /* Copy scalar parameters to output vectors */
      REAL(theta_01_samples)[idx] = theta_01_current;
      REAL(prec_theta1_samples)[idx] = prec_theta1_current;
    }

    /* ===== Update Previous Values for Next Iteration ===== */
    memcpy(theta_1_previous, theta_1_current, n * sizeof(double));
    theta_01_previous = theta_01_current;
    prec_theta1_previous = prec_theta1_current;
  }

  /* ========== Restore RNG State ========== */
  PutRNGstate();

  /* ========== Free Temporary Buffers ========== */
  R_Free(theta_1_current);
  R_Free(theta_1_previous);
  R_Free(alpha_current);
  R_Free(theta_1_updated);
  R_Free(accept_prop);
  R_Free(log_sigma);
  R_Free(hat_theta_1);
  R_Free(theta_1_new);
  R_Free(log_accept_prob);

  /* ========== Package Results into Named List ========== */
  SEXP out = PROTECT(allocVector(VECSXP, n_outputs));
  protect_count++;
  SEXP names = PROTECT(allocVector(STRSXP, n_outputs));
  protect_count++;

  int out_idx = 0;
  SET_VECTOR_ELT(out, out_idx, theta_1_samples);
  SET_STRING_ELT(names, out_idx++, mkChar("theta_1"));

  SET_VECTOR_ELT(out, out_idx, theta_01_samples);
  SET_STRING_ELT(names, out_idx++, mkChar("theta_01"));

  SET_VECTOR_ELT(out, out_idx, prec_theta1_samples);
  SET_STRING_ELT(names, out_idx++, mkChar("prec_theta1"));

  SET_VECTOR_ELT(out, out_idx, alpha_samples);
  SET_STRING_ELT(names, out_idx++, mkChar("alpha"));

  if (return_log_sigma) {
    SET_VECTOR_ELT(out, out_idx, log_sigma_samples);
    SET_STRING_ELT(names, out_idx++, mkChar("log_sigma"));
  }

  if (return_accept_prop) {
    SET_VECTOR_ELT(out, out_idx, accept_prop_samples);
    SET_STRING_ELT(names, out_idx++, mkChar("accept_prop"));
  }

  setAttrib(out, R_NamesSymbol, names);

  UNPROTECT(protect_count);
  return out;
}

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
                                                        SEXP prior_prec1_rate_) {

  /* ========== Parse Data Vector and Validate Length ========== */
  double   *y_ptr = REAL(y_);
  R_xlen_t  len   = LENGTH(y_);

  /* Enforce minimum sample size for numerical stability */
  if (len < 3) {
    error("test_mcmc_probit_bernoulli_locallevel_fixed_params: sample size 'n' must be at least 3, got %lld",
          (long long) len);
  }

  /* Check for integer overflow (R limitation) */
  if (len > INT_MAX) {
    error("test_mcmc_probit_bernoulli_locallevel_fixed_params: sample size too large (%lld > %d)",
          (long long) len, INT_MAX);
  }

  int n = (int) len;

  /* Validate Bernoulli constraints: y[i] must be 0 or 1 */
  for (int t = 0; t < n; t++) {
    if (y_ptr[t] != 0.0 && y_ptr[t] != 1.0) {
      error("test_mcmc_probit_bernoulli_locallevel_fixed_params: y[%d] = %f must be 0 or 1",
            t, y_ptr[t]);
    }
  }

  /* ========== Parse MCMC Control Parameters ========== */
  int burnin   = INTEGER(burnin_)[0];     /* Burn-in iterations to discard */
  int thinning = INTEGER(thinning_)[0];   /* Thinning interval */
  int n_chain  = INTEGER(n_chain_)[0];    /* Number of retained samples */

  /* Compute total iterations needed */
  int n_iter = burnin + (n_chain - 1) * thinning + 1;

  /* ========== Parse Prior Hyperparameters ========== */
  double mean_theta01 = REAL(prior_theta01_mean_)[0]; /* Prior mean for theta_{0,1} */
  double prec_theta01 = REAL(prior_theta01_prec_)[0]; /* Prior precision for theta_{0,1} */
  double nu_01        = REAL(prior_prec1_shape_)[0];  /* Gamma shape for 1/W_1 */
  double eta_01       = REAL(prior_prec1_rate_)[0];   /* Gamma rate for 1/W_1 */

  /* Validate prior parameter positivity */
  if (prec_theta01 <= 0) {
    error("test_mcmc_probit_bernoulli_locallevel_fixed_params: prior_theta01_prec must be positive, got %f",
          prec_theta01);
  }
  if (nu_01 <= 0 || eta_01 <= 0) {
    error("test_mcmc_probit_bernoulli_locallevel_fixed_params: Gamma prior parameters must be positive");
  }

  /* ========== Check Which Parameters Should Be Fixed ========== */
  int fix_theta_1  = (theta_1_true_ != R_NilValue && !Rf_isNull(theta_1_true_));
  int fix_theta_01 = (theta_01_true_ != R_NilValue && !Rf_isNull(theta_01_true_));
  int fix_prec_theta1   = (prec_theta1_true_ != R_NilValue && !Rf_isNull(prec_theta1_true_));

  /* Extract true values if provided */
  double *theta_1_true = NULL;
  double  theta_01_true = 0.0;
  double  prec_theta1_true = 0.0;

  int protect_count = 0;

  if (fix_theta_1) {
    SEXP theta_1_tmp = PROTECT(coerceVector(theta_1_true_, REALSXP));
    protect_count++;
    if (LENGTH(theta_1_tmp) != n) {
      UNPROTECT(protect_count);
      error("test_mcmc_probit_bernoulli_locallevel_fixed_params: theta_1_true must have length %d, got %d",
            n, LENGTH(theta_1_tmp));
    }
    theta_1_true = REAL(theta_1_tmp);
  }

  if (fix_theta_01) {
    theta_01_true = require_real_scalar(theta_01_true_, "theta_01_true");
  }

  if (fix_prec_theta1) {
    prec_theta1_true = require_real_scalar(prec_theta1_true_, "prec_theta1_true");
    if (prec_theta1_true <= 0) {
      UNPROTECT(protect_count);
      error("test_mcmc_probit_bernoulli_locallevel_fixed_params: prec_theta1_true must be positive, got %f",
            prec_theta1_true);
    }
  }

  /* ========== Allocate Output Storage (Retained Samples Only) ========== */
  SEXP theta_1_samples      = PROTECT(allocMatrix(REALSXP, n_chain, n));
  protect_count++;
  SEXP theta_01_samples     = PROTECT(allocVector(REALSXP, n_chain));
  protect_count++;
  SEXP prec_theta1_samples  = PROTECT(allocVector(REALSXP, n_chain));
  protect_count++;
  SEXP alpha_samples        = PROTECT(allocMatrix(REALSXP, n_chain, n));
  protect_count++;

  int n_outputs = 4;  /* Base outputs: theta_1, theta_01, prec_theta1, alpha */

  /* ========== Allocate Temporary Buffers (Memory-Efficient O(n) Storage) ========== */
  double *theta_1_current   = (double *) R_Calloc(n, double);
  double *theta_1_previous  = (double *) R_Calloc(n, double);
  double *alpha_current     = (double *) R_Calloc(n, double);

  /* Scalar parameters for current and previous iterations */
  double theta_01_current, theta_01_previous;
  double prec_theta1_current, prec_theta1_previous;

  /* Working array for probit algorithm */
  double *rhs_vector = (double *) R_Calloc(n, double);

  /* ========== Initialize RNG State ========== */
  GetRNGstate();

  /* ========== Initialize Parameters (Iteration 0) ========== */
  /* Draw initial values from priors to start the Markov chain */
  theta_01_previous = fix_theta_01 ? theta_01_true : rnorm(mean_theta01, sqrt(1.0 / prec_theta01));
  prec_theta1_previous = fix_prec_theta1 ? prec_theta1_true : rgamma(nu_01, 1.0 / eta_01);

  /* Initialize theta_1 and alpha with efficient neutral starting values */
  if (fix_theta_1) {
    memcpy(theta_1_previous, theta_1_true, n * sizeof(double));
    for (int t = 0; t < n; t++) {
      alpha_current[t] = pnorm(theta_1_true[t], 0.0, 1.0, 1, 0);
    }
  } else {
    for (int t = 0; t < n; t++) {
      theta_1_previous[t] = 0.0;   /* Zeros for state trajectory */
      alpha_current[t]    = 0.5;   /* Neutral probability for success rates */
    }
  }

  /* ========== Main Gibbs Sampling Loop ========== */
  int chain_idx = 0;

  for (int ii = 1; ii < n_iter; ii++) {

    /* Determine whether to compute alpha transformations */
    int compute_alpha = (ii >= burnin && ((ii - burnin) % thinning) == 0) ? 1 : 0;

    /* ===== Step 1: Sample Latent Utilities, theta_1, and alpha ===== */
    if (fix_theta_1) {
      /* Use fixed true values instead of sampling */
      memcpy(theta_1_current, theta_1_true, n * sizeof(double));
      if (compute_alpha) {
        for (int t = 0; t < n; t++) {
          alpha_current[t] = pnorm(theta_1_true[t], 0.0, 1.0, 1, 0);
        }
      }
    } else {
      /* Sample via Albert-Chib augmentation */
      generate_alpha_probit_bernoulli_locallevel(
        theta_1_previous,              /* theta_1_previous: level from previous iteration [n] */
        theta_1_current,               /* theta_1_current: output for current iteration [n] */
        compute_alpha ? alpha_current : NULL,  /* alpha_current: NULL if not retained */
        theta_01_previous,             /* theta_01_previous: initial level from previous iteration */
        prec_theta1_previous,               /* prec_theta1_previous: level precision from previous iteration */
        y_ptr,                         /* y: Bernoulli observations */
        rhs_vector,                    /* rhs_vector: solver right-hand side */
        n,                             /* n: number of time points */
        compute_alpha                  /* compute_alpha: flag for alpha computation */
      );
    }

    /* ===== Step 2: Sample Innovation Precision 1/W_1 ===== */
    if (fix_prec_theta1) {
      /* Use fixed true value instead of sampling */
      prec_theta1_current = prec_theta1_true;
    } else {
      /* Sample 1/W_1 | theta_1, theta_01 from Gamma posterior */
      prec_theta1_current = generate_precision_theta_p(
        theta_01_previous,         /* theta_0p: initial level from previous iteration */
        theta_1_current,           /* theta_p_current: current level trajectory [n] */
        nu_01,                     /* nu_0p: prior shape */
        eta_01,                    /* eta_0p: prior rate */
        n                          /* n: number of time points */
      );
    }

    /* ===== Step 3: Sample Initial State theta_{0,1} ===== */
    if (fix_theta_01) {
      /* Use fixed true value instead of sampling */
      theta_01_current = theta_01_true;
    } else {
      /* Sample theta_{0,1} | theta_1, prec_theta1 from Normal posterior */
      theta_01_current = generate_theta_01_locallevel(
        theta_1_current,           /* theta_1_current: current level trajectory [n] */
        prec_theta1_current,            /* prec_theta1: current level precision */
        mean_theta01,              /* mean_theta01: prior mean */
        prec_theta01,              /* prec_theta01: prior precision */
        n                          /* n: number of time points */
      );
    }

    /* ===== Store Post-Burn-in Samples with Thinning ===== */
    if (compute_alpha) {
      int idx = chain_idx++;

      /* Copy current theta_1 and alpha to output matrices (column-major) */
      for (int t = 0; t < n; t++) {
        REAL(theta_1_samples)[idx + t * n_chain] = theta_1_current[t];
        REAL(alpha_samples)[idx + t * n_chain] = alpha_current[t];
      }

      /* Copy scalar parameters to output vectors */
      REAL(theta_01_samples)[idx] = theta_01_current;
      REAL(prec_theta1_samples)[idx] = prec_theta1_current;
    }

    /* ===== Update Previous Values for Next Iteration ===== */
    memcpy(theta_1_previous, theta_1_current, n * sizeof(double));
    theta_01_previous = theta_01_current;
    prec_theta1_previous = prec_theta1_current;
  }

  /* ========== Restore RNG State ========== */
  PutRNGstate();

  /* ========== Free Temporary Buffers ========== */
  R_Free(theta_1_current);
  R_Free(theta_1_previous);
  R_Free(alpha_current);
  R_Free(rhs_vector);

  /* ========== Package Results into Named List ========== */
  SEXP out = PROTECT(allocVector(VECSXP, n_outputs));
  protect_count++;
  SEXP names = PROTECT(allocVector(STRSXP, n_outputs));
  protect_count++;

  int out_idx = 0;
  SET_VECTOR_ELT(out, out_idx, theta_1_samples);
  SET_STRING_ELT(names, out_idx++, mkChar("theta_1"));

  SET_VECTOR_ELT(out, out_idx, theta_01_samples);
  SET_STRING_ELT(names, out_idx++, mkChar("theta_01"));

  SET_VECTOR_ELT(out, out_idx, prec_theta1_samples);
  SET_STRING_ELT(names, out_idx++, mkChar("prec_theta1"));

  SET_VECTOR_ELT(out, out_idx, alpha_samples);
  SET_STRING_ELT(names, out_idx++, mkChar("alpha"));

  setAttrib(out, R_NamesSymbol, names);

  UNPROTECT(protect_count);
  return out;
}
