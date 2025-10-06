/**
 * @file test_helpers.c
 * @brief C wrappers for testing internal C functions from R
 * @details This file contains wrapper functions that expose internal C
 *          functions to R's .Call interface, specifically for the purpose of
 *          unit testing with packages like 'testthat'. Each wrapper handles
 *          proper memory management, input validation, and R object protection.
 * @author Michel H. Montoril
 * @date 2025-10-01
 * @version 1.5
 *
 * @changelog
 * - v1.5 (2025-10-01): Code cleanup and bug fixes. Removed unused variables,
 *   improved memory management consistency, enhanced input validation, and
 *   updated documentation to reflect actual implementation. Fixed UNPROTECT
 *   counting logic and made hardcoded parameters more explicit.
 * - v1.4 (2025-09-28): Fixed function signatures to match updated
 *   generate_alpha_logit_binomial functions with min_deviation_threshold parameter.
 *   Corrected CWMH function calls to use proper lag_update parameter.
 * - v1.4 (2025-09-23): Updated test_adapt_cwmh_parameters to include
 *   min_deviation_threshold parameter for optimized adaptive MCMC testing.
 *   comparison testing. Enhanced documentation for adaptive function testing.
 * - v1.3 (2025-09-15): Modified test_cwmh_alpha_logit_binomial_locallevel
 *   to accept log_sigma as an argument to close the adaptive MCMC loop.
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>     // Include for rnorm and rgamma
#include <string.h>    // For memcpy

// Include headers for the C functions to be tested
#include "utils.h"
#include "cwmh_adaptive.h"
#include "conditional_precision.h"
#include "conditional_state.h"
#include "conditional_theta0.h"
#include "cwmh_binomial.h"
#include "generate_alpha_binomial.h"
#include "mcmc_binomial_locallevel.h"

//==============================================================================
// UTILITY FUNCTION WRAPPERS
//==============================================================================

/**
 * @brief R interface wrapper for the internal C ilogit function
 *
 * @details This function serves as a bridge to allow the internal C `ilogit`
 *          function to be called directly from R for unit testing. It takes a
 *          numeric SEXP, applies the transformation to the first element, and
 *          returns the result as a scalar SEXP. Essential for validating
 *          numerical accuracy of logistic transformations.
 *
 * @param x_ A numeric SEXP from R. Only the first element is used.
 * @return A scalar real SEXP containing the result of the ilogit transformation.
 *
 * @note Computational complexity: O(1) - single function call
 * @note Numerical stability: Inherits stability properties from ilogit implementation
 *
 * @warning Input must be a non-empty numeric vector
 * @warning No bounds checking on array access beyond length verification
 *
 * @see ilogit
 * @since version 1.0
 */
SEXP test_ilogit(SEXP x_) {
  if (!isReal(x_) || length(x_) == 0) {
    error("Input must be a non-empty numeric vector");
  }

  double val = ilogit(REAL(x_)[0]);
  return ScalarReal(val);
}

/**
 * @brief Test wrapper for the generate_normal_vector function
 *
 * @details Allows calling the C function 'generate_normal_vector' with parameters
 *          defined in R to verify its output. This is crucial for testing the
 *          correctness of the multivariate normal sampling with a tridiagonal
 *          precision matrix. Manages R's random number generator state properly.
 *
 * @param y_ SEXP: A numeric vector for 'y' (the right-hand side of the system).
 * @param a_ SEXP: A numeric scalar for 'a'.
 * @param b_ SEXP: A numeric scalar for 'b'.
 * @param add_a_ SEXP: An integer scalar for the 'add_a' flag.
 * @return A SEXP containing the generated random vector.
 *
 * @note Computational complexity: O(n) where n = length(y)
 * @note Memory access: Allocates result vector of size n
 * @note Algorithm: Uses R's random number generator with proper state management
 *
 * @warning Requires valid numeric inputs for mathematical operations
 * @warning Uses fixed iter=0 for testing purposes
 *
 * @see generate_normal_vector
 * @see GetRNGstate, PutRNGstate
 * @since version 1.0
 */
SEXP test_generate_normal_vector(SEXP y_, SEXP a_, SEXP b_, SEXP add_a_) {

  // Protect R objects from garbage collection and ensure correct types
  PROTECT(y_ = coerceVector(y_, REALSXP));
  PROTECT(a_ = coerceVector(a_, REALSXP));
  PROTECT(b_ = coerceVector(b_, REALSXP));
  PROTECT(add_a_ = coerceVector(add_a_, INTSXP));

  double *y = REAL(y_);
  int n = LENGTH(y_);
  double a = REAL(a_)[0];
  double b = REAL(b_)[0];
  int add_a = INTEGER(add_a_)[0];

  // Allocate memory for the result vector
  SEXP result_sexp = PROTECT(allocVector(REALSXP, n));
  double *result_ptr = REAL(result_sexp);

  // Manage R's random number generator state
  GetRNGstate();

  // Call the target function for testing
  generate_normal_vector(
    result_ptr, /* dest: output buffer for sampled vector */
    y,          /* y: right-hand side vector */
    a,          /* a: main diagonal precision contribution */
    b,          /* b: off-diagonal precision contribution */
    n,          /* n: vector length */
    0,          /* iter: fixed iteration index for testing */
    add_a       /* add_a: adjustment flag for last diagonal */
  );

  PutRNGstate();

  // Release protected objects
  UNPROTECT(5);

  return result_sexp;
}

//==============================================================================
// ADAPTIVE MCMC WRAPPERS
//==============================================================================

/**
 * @brief Test wrapper for the optimized adapt_cwmh_parameters function
 *
 * @details Exposes the optimized C function 'adapt_cwmh_parameters' to R for testing.
 *          This wrapper takes all necessary parameters from R, including the new
 *          min_deviation_threshold parameter, calls the adaptation function, and returns
 *          the updated 'accept_prop' and 'log_sigma' vectors in a named list.
 *          Critical for validating optimized adaptive MCMC behavior with configurable
 *          threshold sensitivity.
 *
 *          **Version 1.4 enhancements:**
 *          Updated to include min_deviation_threshold parameter, enabling comprehensive
 *          testing of threshold-based adaptation logic. This allows validation of the
 *          practical 1/lag_update threshold recommendation and comparison with other
 *          threshold values for sensitivity analysis.
 *
 * @param theta_updated_ SEXP: A numeric vector representing the acceptance history
 *                       in sliding window format (lag_update x n, row-major).
 * @param log_sigma_ SEXP: A numeric vector of the current log proposal standard deviations.
 * @param lag_update_ SEXP: An integer scalar for the sliding window size.
 * @param n_ SEXP: An integer scalar for the number of components.
 * @param iter_ SEXP: An integer scalar for the current MCMC iteration.
 * @param max_step_size_ SEXP: A numeric scalar for the maximum step size.
 * @param base_adaptation_rate_ SEXP: A numeric scalar for the base adaptation rate.
 * @param decay_exponent_ SEXP: A numeric scalar for the decay exponent.
 * @param target_acceptance_ SEXP: A numeric scalar for the target acceptance rate.
 * @param min_deviation_threshold_ SEXP: A numeric scalar for the minimum deviation
 *                       threshold to trigger adaptation updates.
 * @return A named list (VECSXP) with two elements: 'accept_prop' and 'log_sigma'.
 *
 * @note Computational complexity: O(n) for component-wise adaptation with optimizations
 * @note Memory access: Allocates vectors for accept_prop and log_sigma outputs
 * @note Algorithm: Implements optimized adaptive step size tuning with configurable threshold
 * @note Testing focus: Validates threshold-based filtering and cache performance
 *
 * @warning Requires positive adaptation parameters for numerical stability
 * @warning min_deviation_threshold must be non-negative
 * @warning Sliding window format must match C function expectations (row-major)
 *
 * @see adapt_cwmh_parameters
 * @since version 1.4
 */
SEXP test_adapt_cwmh_parameters(SEXP theta_updated_, SEXP log_sigma_,
                                SEXP lag_update_, SEXP n_, SEXP iter_,
                                SEXP max_step_size_, SEXP base_adaptation_rate_,
                                SEXP decay_exponent_, SEXP target_acceptance_,
                                SEXP min_deviation_threshold_) {

  // Protect and coerce inputs (10 parameters)
  PROTECT(theta_updated_ = coerceVector(theta_updated_, REALSXP));
  PROTECT(log_sigma_ = coerceVector(log_sigma_, REALSXP));
  PROTECT(lag_update_ = coerceVector(lag_update_, INTSXP));
  PROTECT(n_ = coerceVector(n_, INTSXP));
  PROTECT(iter_ = coerceVector(iter_, INTSXP));
  PROTECT(max_step_size_ = coerceVector(max_step_size_, REALSXP));
  PROTECT(base_adaptation_rate_ = coerceVector(base_adaptation_rate_, REALSXP));
  PROTECT(decay_exponent_ = coerceVector(decay_exponent_, REALSXP));
  PROTECT(target_acceptance_ = coerceVector(target_acceptance_, REALSXP));
  PROTECT(min_deviation_threshold_ = coerceVector(min_deviation_threshold_, REALSXP));

  // Extract C types from SEXPs
  double *theta_updated = REAL(theta_updated_);
  double *log_sigma_in = REAL(log_sigma_);
  int lag_update = INTEGER(lag_update_)[0];
  int n = INTEGER(n_)[0];
  int iter = INTEGER(iter_)[0];
  double max_step_size = REAL(max_step_size_)[0];
  double base_adaptation_rate = REAL(base_adaptation_rate_)[0];
  double decay_exponent = REAL(decay_exponent_)[0];
  double target_acceptance = REAL(target_acceptance_)[0];
  double min_deviation_threshold = REAL(min_deviation_threshold_)[0];

  // Allocate memory for outputs and the list to hold them
  SEXP result_list = PROTECT(allocVector(VECSXP, 2));
  SEXP accept_prop_sexp = PROTECT(allocVector(REALSXP, n));
  SEXP log_sigma_sexp = PROTECT(allocVector(REALSXP, n));

  double *accept_prop_out = REAL(accept_prop_sexp);
  double *log_sigma_out = REAL(log_sigma_sexp);

  // Copy input log_sigma to output log_sigma, as it is modified in place
  for(int i = 0; i < n; i++) {
    log_sigma_out[i] = log_sigma_in[i];
  }

  // Call the optimized C function
  adapt_cwmh_parameters(
    theta_updated,          /* theta_updated: sliding window acceptance history */
    accept_prop_out,        /* accept_prop: output acceptance proportions */
    log_sigma_out,          /* log_sigma: proposal log standard deviations */
    lag_update,             /* lag_update: window length */
    n,                      /* n: number of components */
    iter,                   /* iter: current iteration */
    max_step_size,          /* max_step_size: adaptation step cap */
    base_adaptation_rate,   /* base_adaptation_rate: initial adaptation rate */
    decay_exponent,         /* decay_exponent: diminishing schedule */
    target_acceptance,      /* target_acceptance: desired acceptance proportion */
    min_deviation_threshold /* min_deviation_threshold: deviation trigger */
  );

  // Set names for the list elements
  SEXP names = PROTECT(allocVector(STRSXP, 2));
  SET_STRING_ELT(names, 0, mkChar("accept_prop"));
  SET_STRING_ELT(names, 1, mkChar("log_sigma"));
  setAttrib(result_list, R_NamesSymbol, names);

  // Populate the list with the results
  SET_VECTOR_ELT(result_list, 0, accept_prop_sexp);
  SET_VECTOR_ELT(result_list, 1, log_sigma_sexp);

  // Unprotect all SEXPs (10 inputs + 1 list + 2 vectors + 1 names = 14)
  UNPROTECT(14);

  return result_list;
}

/**
 * @brief R interface wrapper for reset_adaptation_cache utility function
 *
 * @details Provides R access to the internal cache reset function for testing purposes.
 *          This wrapper enables test isolation by clearing cached values between test
 *          cases, particularly important when testing with different lag_update values.
 *
 *          **Testing importance:**
 *          Version 1.2 of cwmh_adaptive.c includes automatic cache validation for
 *          lag_update changes. However, explicit cache reset between tests ensures
 *          complete isolation and validates that the automatic mechanism works correctly.
 *
 *          **Usage in tests:**
 *          Call at the beginning of each test case that uses adapt_cwmh_parameters
 *          to ensure deterministic behavior independent of test execution order.
 *
 * @return R_NilValue (invisible NULL in R).
 *
 * @note Computational complexity: O(1) - simply resets static variables
 * @note No arguments required - function signature matches R's .Call interface
 * @note Safe to call multiple times - idempotent operation
 * @note Thread-safe as it only modifies static cache variables
 *
 * @warning Should only be used in testing contexts
 * @warning Not necessary in production code due to automatic cache validation
 *
 * @see reset_adaptation_cache
 * @see adapt_cwmh_parameters
 * @since version 1.2
 */
SEXP reset_adaptation_cache_wrapper(void) {
  reset_adaptation_cache();
  return R_NilValue;
}

//==============================================================================
// CONDITIONAL PRECISION WRAPPERS
//==============================================================================

/**
 * @brief Test wrapper for generate_precision_data from conditional_precision.c
 *
 * @details Enables testing of observation precision sampling in dynamic models.
 *          Simulates minimal MCMC history structure required by the C function
 *          and extracts the generated precision value for validation.
 *
 * @param y_ SEXP: Observed data vector
 * @param theta_1_ SEXP: State parameter vector theta_1
 * @param nu_y_ SEXP: Prior shape parameter for Gamma distribution
 * @param eta_y_ SEXP: Prior rate parameter for Gamma distribution
 * @return Scalar SEXP containing the generated precision sample
 *
 * @note Computational complexity: O(n) for residual sum calculation
 * @note Memory access: Simulates MCMC history with minimal allocation
 * @note Algorithm: Gamma-Normal conjugate updating for observation precision
 *
 * @warning Uses fixed iter=1 for testing purposes
 * @warning Minimal history simulation may not reflect full MCMC dynamics
 *
 * @see generate_precision_data
 * @since version 1.0
 */
SEXP test_generate_precision_data(SEXP y_, SEXP theta_1_, SEXP nu_y_, SEXP eta_y_) {
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  int n = LENGTH(y);
  double *y_ptr = REAL(y);

  // Simulate the MCMC history array required by the C function
  double *theta_1_post = (double *) R_alloc(2 * n, sizeof(double));
  SEXP theta_1 = PROTECT(coerceVector(theta_1_, REALSXP));
  memcpy(theta_1_post + n, REAL(theta_1), n * sizeof(double));

  SEXP nu_y = PROTECT(coerceVector(nu_y_, REALSXP));
  double nu_y_val = REAL(nu_y)[0];
  SEXP eta_y = PROTECT(coerceVector(eta_y_, REALSXP));
  double eta_y_val = REAL(eta_y)[0];

  double *prec_y_post = (double *) R_alloc(2, sizeof(double));

  GetRNGstate();
  // Call the function for iter = 1 (the second position)
  generate_precision_data(
    y_ptr,         /* y: observed data */
    theta_1_post,  /* theta_1_post: state trajectories */
    prec_y_post,   /* prec_y_post: precision storage */
    nu_y_val,      /* nu_y: prior shape */
    eta_y_val,     /* eta_y: prior rate */
    n,             /* n: number of observations */
    1              /* iter: fixed iteration index */
  );
  PutRNGstate();

  UNPROTECT(4);
  return ScalarReal(prec_y_post[1]);
}

/**
 * @brief Test wrapper for generate_precision_theta_k from conditional_precision.c
 *
 * @details Enables testing of intermediate state innovation precision sampling.
 *          Simulates MCMC history arrays and validates the Gamma-Normal conjugate
 *          updating for k-th component precision parameters.
 *
 * @param theta_0k_ SEXP: Initial value for k-th component
 * @param theta_0kp1_ SEXP: Initial value for (k+1)-th component
 * @param theta_k_ SEXP: State values for k-th component
 * @param theta_kp1_ SEXP: State values for (k+1)-th component
 * @param nu_0k_ SEXP: Prior shape parameter
 * @param eta_0k_ SEXP: Prior rate parameter
 * @return Scalar SEXP containing the generated precision sample
 *
 * @note Computational complexity: O(n) for innovation sum calculation
 * @note Memory access: Simulates vectorized MCMC history storage
 * @note Algorithm: Conjugate updating for intermediate component precision
 *
 * @warning Uses fixed iter=1 for testing purposes
 * @warning Requires proper initialization of theta arrays
 *
 * @see generate_precision_theta_k
 * @since version 1.0
 */
SEXP test_generate_precision_theta_k(SEXP theta_0k_, SEXP theta_0kp1_, SEXP theta_k_, SEXP theta_kp1_,
                                     SEXP nu_0k_, SEXP eta_0k_) {
  SEXP theta_k = PROTECT(coerceVector(theta_k_, REALSXP));
  int n = LENGTH(theta_k);

  // Simulate MCMC history arrays
  double *theta_0k_post = (double *) R_alloc(2, sizeof(double));
  SEXP theta_0k = PROTECT(coerceVector(theta_0k_, REALSXP));
  theta_0k_post[0] = REAL(theta_0k)[0];

  double *theta_0kp1_post = (double *) R_alloc(2, sizeof(double));
  SEXP theta_0kp1 = PROTECT(coerceVector(theta_0kp1_, REALSXP));
  theta_0kp1_post[1] = REAL(theta_0kp1)[0];

  double *theta_k_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_k_post + n, REAL(theta_k), n * sizeof(double));

  double *theta_kp1_post = (double *) R_alloc(2 * n, sizeof(double));
  SEXP theta_kp1 = PROTECT(coerceVector(theta_kp1_, REALSXP));
  memcpy(theta_kp1_post + n, REAL(theta_kp1), n * sizeof(double));

  SEXP nu_0k = PROTECT(coerceVector(nu_0k_, REALSXP));
  double nu_0k_val = REAL(nu_0k)[0];
  SEXP eta_0k = PROTECT(coerceVector(eta_0k_, REALSXP));
  double eta_0k_val = REAL(eta_0k)[0];

  double *prec_theta_k_post = (double *) R_alloc(2, sizeof(double));

  GetRNGstate();
  // Call for iter = 1
  generate_precision_theta_k(
    theta_0k_post,    /* theta_0k_post: initial state k */
    theta_0kp1_post,  /* theta_0kp1_post: initial state k+1 */
    theta_k_post,     /* theta_k_post: state trajectories k */
    theta_kp1_post,   /* theta_kp1_post: state trajectories k+1 */
    prec_theta_k_post,/* prec_theta_k_post: precision storage */
    nu_0k_val,        /* nu_0k: prior shape */
    eta_0k_val,       /* eta_0k: prior rate */
    n,                /* n: number of observations */
    1                 /* iter: fixed iteration index */
  );
  PutRNGstate();

  UNPROTECT(6);
  return ScalarReal(prec_theta_k_post[1]);
}

/**
 * @brief Test wrapper for generate_precision_theta_p from conditional_precision.c
 *
 * @details Enables testing of final component precision sampling in polynomial
 *          dynamic models. This boundary case follows random walk structure
 *          for the highest-order state component.
 *
 * @param theta_0p_ SEXP: Initial value for p-th component
 * @param theta_p_ SEXP: State values for p-th component
 * @param nu_0p_ SEXP: Prior shape parameter
 * @param eta_0p_ SEXP: Prior rate parameter
 * @return Scalar SEXP containing the generated precision sample
 *
 * @note Computational complexity: O(n) for random walk innovation calculation
 * @note Memory access: Simulates MCMC history for boundary component
 * @note Algorithm: Conjugate updating for random walk precision
 *
 * @warning Uses fixed iter=1 for testing purposes
 * @warning Boundary case requires different treatment than intermediate components
 *
 * @see generate_precision_theta_p
 * @since version 1.0
 */
SEXP test_generate_precision_theta_p(SEXP theta_0p_, SEXP theta_p_, SEXP nu_0p_, SEXP eta_0p_) {
  SEXP theta_p = PROTECT(coerceVector(theta_p_, REALSXP));
  int n = LENGTH(theta_p);
  double *theta_p_ptr = REAL(theta_p);

  // Simulate MCMC history arrays
  double *theta_0p_post = (double *) R_alloc(2, sizeof(double));
  SEXP theta_0p = PROTECT(coerceVector(theta_0p_, REALSXP));
  theta_0p_post[0] = REAL(theta_0p)[0];

  double *theta_p_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_p_post + n, theta_p_ptr, n * sizeof(double));

  SEXP nu_0p = PROTECT(coerceVector(nu_0p_, REALSXP));
  double nu_0p_val = REAL(nu_0p)[0];
  SEXP eta_0p = PROTECT(coerceVector(eta_0p_, REALSXP));
  double eta_0p_val = REAL(eta_0p)[0];

  double *prec_theta_p_post = (double *) R_alloc(2, sizeof(double));

  GetRNGstate();
  // Call for iter = 1
  generate_precision_theta_p(
    theta_0p_post,    /* theta_0p_post: initial state p */
    theta_p_post,     /* theta_p_post: state trajectories p */
    prec_theta_p_post,/* prec_theta_p_post: precision storage */
    nu_0p_val,        /* nu_0p: prior shape */
    eta_0p_val,       /* eta_0p: prior rate */
    n,                /* n: number of observations */
    1                 /* iter: fixed iteration index */
  );
  PutRNGstate();

  UNPROTECT(4);
  return ScalarReal(prec_theta_p_post[1]);
}

//==============================================================================
// CONDITIONAL STATE WRAPPERS
//==============================================================================

/**
 * @brief Test wrapper for generate_theta_1_locallevel from conditional_state.c
 *
 * @details Enables testing of first-order state sampling in local level models.
 *          This function validates the multivariate normal sampling with
 *          tridiagonal precision structure for dynamic state estimation.
 *
 * @param data_ SEXP: Observed data vector
 * @param prec_data_ SEXP: Data precision parameter
 * @param prec_theta_1_ SEXP: State precision parameter
 * @param theta_01_ SEXP: Initial state value
 * @return SEXP vector containing generated theta_1 samples
 *
 * @note Computational complexity: O(n) for tridiagonal system solution
 * @note Memory access: Simulates MCMC arrays for state sampling
 * @note Algorithm: Forward-backward algorithm for structured multivariate normal
 *
 * @warning Uses fixed iter=1 for testing purposes
 * @warning Requires positive precision parameters for numerical stability
 *
 * @see generate_theta_1_locallevel
 * @since version 1.0
 */
SEXP test_generate_theta_1_locallevel(SEXP data_, SEXP prec_data_, SEXP prec_theta_1_, SEXP theta_01_) {
  SEXP data = PROTECT(coerceVector(data_, REALSXP));
  int n = LENGTH(data);
  double *data_ptr = REAL(data);

  SEXP prec_data = PROTECT(coerceVector(prec_data_, REALSXP));
  double prec_data_val = REAL(prec_data)[0];
  SEXP prec_theta_1 = PROTECT(coerceVector(prec_theta_1_, REALSXP));
  double prec_theta_1_val = REAL(prec_theta_1)[0];
  SEXP theta_01 = PROTECT(coerceVector(theta_01_, REALSXP));
  double theta_01_val = REAL(theta_01)[0];

  double *theta_1_post = (double *) R_alloc(2 * n, sizeof(double));
  double *prec_data_post = (double *) R_alloc(2, sizeof(double));
  double *prec_theta_1_post = (double *) R_alloc(2, sizeof(double));
  double *theta_01_post = (double *) R_alloc(2, sizeof(double));

  prec_data_post[0] = prec_data_val;
  prec_theta_1_post[0] = prec_theta_1_val;
  theta_01_post[0] = theta_01_val;

  GetRNGstate();
  generate_theta_1_locallevel(
    data_ptr,        /* data: observed series */
    theta_1_post,    /* theta_1_post: level trajectories */
    prec_data_post,  /* prec_data_post: data precision draws */
    prec_theta_1_post,/* prec_theta_1_post: level precision draws */
    theta_01_post,   /* theta_01_post: initial level states */
    n,               /* n: number of observations */
    1                /* iter: fixed iteration index */
  );
  PutRNGstate();

  SEXP result_sexp = PROTECT(allocVector(REALSXP, n));
  memcpy(REAL(result_sexp), theta_1_post + n, n * sizeof(double));

  UNPROTECT(5);
  return result_sexp;
}

/**
 * @brief Test wrapper for generate_theta_1 from conditional_state.c
 *
 * @details Enables testing of first-order state sampling in local trend models.
 *          This function validates the conditional sampling of theta_1 given
 *          theta_2 and other model parameters in polynomial dynamic structures.
 *
 * @param data_ SEXP: Observed data vector
 * @param theta_2_ SEXP: Second-order state vector
 * @param prec_data_ SEXP: Data precision parameter
 * @param prec_theta_1_ SEXP: First-order state precision parameter
 * @param theta_01_ SEXP: Initial first-order state value
 * @param theta_02_ SEXP: Initial second-order state value
 * @return SEXP vector containing generated theta_1 samples
 *
 * @note Computational complexity: O(n) for conditional multivariate normal sampling
 * @note Memory access: Manages multiple MCMC history arrays
 * @note Algorithm: Conditional sampling in polynomial state space
 *
 * @warning Uses fixed iter=1 for testing purposes
 * @warning Requires consistent dimensionality across state vectors
 *
 * @see generate_theta_1
 * @since version 1.0
 */
SEXP test_generate_theta_1(SEXP data_, SEXP theta_2_, SEXP prec_data_, SEXP prec_theta_1_, SEXP theta_01_, SEXP theta_02_) {
  SEXP data = PROTECT(coerceVector(data_, REALSXP));
  int n = LENGTH(data);
  double *data_ptr = REAL(data);

  SEXP theta_2 = PROTECT(coerceVector(theta_2_, REALSXP));
  double *theta_2_ptr = REAL(theta_2);
  SEXP prec_data = PROTECT(coerceVector(prec_data_, REALSXP));
  double prec_data_val = REAL(prec_data)[0];
  SEXP prec_theta_1 = PROTECT(coerceVector(prec_theta_1_, REALSXP));
  double prec_theta_1_val = REAL(prec_theta_1)[0];
  SEXP theta_01 = PROTECT(coerceVector(theta_01_, REALSXP));
  double theta_01_val = REAL(theta_01)[0];
  SEXP theta_02 = PROTECT(coerceVector(theta_02_, REALSXP));
  double theta_02_val = REAL(theta_02)[0];

  double *theta_1_post = (double *) R_alloc(2 * n, sizeof(double));
  double *theta_2_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_2_post + n, theta_2_ptr, n * sizeof(double));

  double *prec_data_post = (double *) R_alloc(2, sizeof(double));
  prec_data_post[0] = prec_data_val;

  double *prec_theta_1_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_1_post[0] = prec_theta_1_val;

  double *theta_01_post = (double *) R_alloc(2, sizeof(double));
  theta_01_post[0] = theta_01_val;

  double *theta_02_post = (double *) R_alloc(2, sizeof(double));
  theta_02_post[1] = theta_02_val;

  GetRNGstate();
  generate_theta_1(
    data_ptr,        /* data: observed series */
    theta_1_post,    /* theta_1_post: level trajectories */
    theta_2_post,    /* theta_2_post: trend trajectories */
    prec_data_post,  /* prec_data_post: data precision draws */
    prec_theta_1_post,/* prec_theta_1_post: level precision draws */
    theta_01_post,   /* theta_01_post: initial level states */
    theta_02_post,   /* theta_02_post: initial trend states */
    n,               /* n: number of observations */
    1                /* iter: fixed iteration index */
  );
  PutRNGstate();

  SEXP result_sexp = PROTECT(allocVector(REALSXP, n));
  memcpy(REAL(result_sexp), theta_1_post + n, n * sizeof(double));

  UNPROTECT(7);
  return result_sexp;
}

/**
 * @brief Test wrapper for generate_theta_k from conditional_state.c
 *
 * @details Enables testing of intermediate state component sampling in polynomial
 *          dynamic models. This function validates conditional sampling of the
 *          k-th state component given adjacent components.
 *
 * @param theta_km1_ SEXP: (k-1)-th state component vector
 * @param theta_kp1_ SEXP: (k+1)-th state component vector
 * @param prec_km1_ SEXP: Precision for (k-1)-th component
 * @param prec_k_ SEXP: Precision for k-th component
 * @param theta_0k_ SEXP: Initial k-th state value
 * @param theta_0kp1_ SEXP: Initial (k+1)-th state value
 * @return SEXP vector containing generated theta_k samples
 *
 * @note Computational complexity: O(n) for conditional sampling algorithm
 * @note Memory access: Coordinates multiple state component arrays
 * @note Algorithm: Conditional multivariate normal for intermediate components
 *
 * @warning Uses fixed iter=1 for testing purposes
 * @warning Requires proper indexing for polynomial state structure
 *
 * @see generate_theta_k
 * @since version 1.0
 */
SEXP test_generate_theta_k(SEXP theta_km1_, SEXP theta_kp1_, SEXP prec_km1_, SEXP prec_k_, SEXP theta_0k_, SEXP theta_0kp1_) {
  SEXP theta_km1 = PROTECT(coerceVector(theta_km1_, REALSXP));
  int n = LENGTH(theta_km1);
  double *theta_km1_ptr = REAL(theta_km1);

  SEXP theta_kp1 = PROTECT(coerceVector(theta_kp1_, REALSXP));
  double *theta_kp1_ptr = REAL(theta_kp1);
  SEXP prec_km1 = PROTECT(coerceVector(prec_km1_, REALSXP));
  double prec_km1_val = REAL(prec_km1)[0];
  SEXP prec_k = PROTECT(coerceVector(prec_k_, REALSXP));
  double prec_k_val = REAL(prec_k)[0];
  SEXP theta_0k = PROTECT(coerceVector(theta_0k_, REALSXP));
  double theta_0k_val = REAL(theta_0k)[0];
  SEXP theta_0kp1 = PROTECT(coerceVector(theta_0kp1_, REALSXP));
  double theta_0kp1_val = REAL(theta_0kp1)[0];

  double *theta_km1_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_km1_post, theta_km1_ptr, n * sizeof(double));

  double *theta_k_post = (double *) R_alloc(2 * n, sizeof(double));

  double *theta_kp1_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_kp1_post + n, theta_kp1_ptr, n * sizeof(double));

  double *prec_theta_km1_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_km1_post[0] = prec_km1_val;

  double *prec_theta_k_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_k_post[0] = prec_k_val;

  double *theta_0k_post = (double *) R_alloc(2, sizeof(double));
  theta_0k_post[0] = theta_0k_val;

  double *theta_0kp1_post = (double *) R_alloc(2, sizeof(double));
  theta_0kp1_post[1] = theta_0kp1_val;

  GetRNGstate();
  generate_theta_k(
    theta_km1_post,   /* theta_km1_post: trajectories for component k-1 */
    theta_k_post,     /* theta_k_post: trajectories for component k */
    theta_kp1_post,   /* theta_kp1_post: trajectories for component k+1 */
    prec_theta_km1_post,/* prec_theta_km1_post: precision for component k-1 */
    prec_theta_k_post,  /* prec_theta_k_post: precision for component k */
    theta_0k_post,    /* theta_0k_post: initial state k */
    theta_0kp1_post,  /* theta_0kp1_post: initial state k+1 */
    n,                /* n: number of observations */
    1                 /* iter: fixed iteration index */
  );
  PutRNGstate();

  SEXP result_sexp = PROTECT(allocVector(REALSXP, n));
  memcpy(REAL(result_sexp), theta_k_post + n, n * sizeof(double));

  UNPROTECT(7);
  return result_sexp;
}

/**
 * @brief Test wrapper for generate_theta_p from conditional_state.c
 *
 * @details Enables testing of final state component sampling in polynomial
 *          dynamic models. This boundary case handles the highest-order
 *          state component following random walk dynamics.
 *
 * @param theta_pm1_ SEXP: (p-1)-th state component vector
 * @param prec_pm1_ SEXP: Precision for (p-1)-th component
 * @param prec_p_ SEXP: Precision for p-th component
 * @param theta_0p_ SEXP: Initial p-th state value
 * @return SEXP vector containing generated theta_p samples
 *
 * @note Computational complexity: O(n) for boundary component sampling
 * @note Memory access: Handles final component in polynomial sequence
 * @note Algorithm: Random walk sampling for highest-order component
 *
 * @warning Uses fixed iter=1 for testing purposes
 * @warning Boundary conditions differ from intermediate components
 *
 * @see generate_theta_p
 * @since version 1.0
 */
SEXP test_generate_theta_p(SEXP theta_pm1_, SEXP prec_pm1_, SEXP prec_p_, SEXP theta_0p_) {
  SEXP theta_pm1 = PROTECT(coerceVector(theta_pm1_, REALSXP));
  int n = LENGTH(theta_pm1);
  double *theta_pm1_ptr = REAL(theta_pm1);

  SEXP prec_pm1 = PROTECT(coerceVector(prec_pm1_, REALSXP));
  double prec_pm1_val = REAL(prec_pm1)[0];
  SEXP prec_p = PROTECT(coerceVector(prec_p_, REALSXP));
  double prec_p_val = REAL(prec_p)[0];
  SEXP theta_0p = PROTECT(coerceVector(theta_0p_, REALSXP));
  double theta_0p_val = REAL(theta_0p)[0];

  double *theta_pm1_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_pm1_post, theta_pm1_ptr, n * sizeof(double));

  double *theta_p_post = (double *) R_alloc(2 * n, sizeof(double));

  double *prec_theta_pm1_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_pm1_post[0] = prec_pm1_val;

  double *prec_theta_p_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_p_post[0] = prec_p_val;

  double *theta_0p_post = (double *) R_alloc(2, sizeof(double));
  theta_0p_post[0] = theta_0p_val;

  GetRNGstate();
  generate_theta_p(
    theta_pm1_post,   /* theta_pm1_post: trajectories for component p-1 */
    theta_p_post,     /* theta_p_post: trajectories for component p */
    prec_theta_pm1_post,/* prec_theta_pm1_post: precision for component p-1 */
    prec_theta_p_post,  /* prec_theta_p_post: precision for component p */
    theta_0p_post,    /* theta_0p_post: initial state p */
    n,                /* n: number of observations */
    1                 /* iter: fixed iteration index */
  );
  PutRNGstate();

  SEXP result_sexp = PROTECT(allocVector(REALSXP, n));
  memcpy(REAL(result_sexp), theta_p_post + n, n * sizeof(double));

  UNPROTECT(5);
  return result_sexp;
}

//==============================================================================
// CONDITIONAL THETA0 WRAPPERS
//==============================================================================

/**
 * @brief Test wrapper for generate_theta_01_locallevel from conditional_theta0.c
 *
 * @details Enables testing of initial state sampling in local level models.
 *          This function validates the Normal-Normal conjugate updating for
 *          the initial state parameter in dynamic models.
 *
 * @param theta_1_ SEXP: First-order state vector
 * @param prec_theta_1_ SEXP: State precision parameter
 * @param mean_theta_01_ SEXP: Prior mean for initial state
 * @param prec_theta_01_ SEXP: Prior precision for initial state
 * @return Scalar SEXP containing the generated theta_01 sample
 *
 * @note Computational complexity: O(n) for sufficient statistics computation
 * @note Memory access: Simulates MCMC history for initial state sampling
 * @note Algorithm: Normal-Normal conjugate updating
 *
 * @warning Uses fixed iter=1 for testing purposes
 * @warning Requires positive precision parameters
 *
 * @see generate_theta_01_locallevel
 * @since version 1.0
 */
SEXP test_generate_theta_01_locallevel(SEXP theta_1_, SEXP prec_theta_1_, SEXP mean_theta_01_, SEXP prec_theta_01_) {
  SEXP theta_1 = PROTECT(coerceVector(theta_1_, REALSXP));
  int n = LENGTH(theta_1);
  double *theta_1_ptr = REAL(theta_1);

  SEXP prec_theta_1 = PROTECT(coerceVector(prec_theta_1_, REALSXP));
  double prec_theta_1_val = REAL(prec_theta_1)[0];
  SEXP mean_theta_01 = PROTECT(coerceVector(mean_theta_01_, REALSXP));
  double mean_theta_01_val = REAL(mean_theta_01)[0];
  SEXP prec_theta_01 = PROTECT(coerceVector(prec_theta_01_, REALSXP));
  double prec_theta_01_val = REAL(prec_theta_01)[0];

  double *theta_1_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_1_post + n, theta_1_ptr, n * sizeof(double));

  double *prec_theta_1_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_1_post[1] = prec_theta_1_val;

  double *theta_01_post = (double *) R_alloc(2, sizeof(double));

  GetRNGstate();
  generate_theta_01_locallevel(
    theta_01_post,      /* theta_01_post: initial level states */
    theta_1_post,       /* theta_1_post: level trajectories */
    prec_theta_1_post,  /* prec_theta_1_post: level precision draws */
    mean_theta_01_val,  /* mean_theta_01: prior mean */
    prec_theta_01_val,  /* prec_theta_01: prior precision */
    n,                  /* n: number of observations */
    1                   /* iter: fixed iteration index */
  );
  PutRNGstate();

  UNPROTECT(4);
  return ScalarReal(theta_01_post[1]);
}

/**
 * @brief Test wrapper for generate_theta_01 from conditional_theta0.c
 *
 * @details Enables testing of initial first-order state sampling in local trend models.
 *          This function validates the conditional sampling of theta_01 given
 *          theta_02 and state observations in polynomial dynamic structures.
 *
 * @param theta_1_ SEXP: First-order state vector
 * @param theta_02_ SEXP: Initial second-order state value
 * @param prec_theta_1_ SEXP: State precision parameter
 * @param mean_theta_01_ SEXP: Prior mean for initial state
 * @param prec_theta_01_ SEXP: Prior precision for initial state
 * @return Scalar SEXP containing the generated theta_01 sample
 *
 * @note Computational complexity: O(n) for conditional sampling computation
 * @note Memory access: Coordinates initial state parameters
 * @note Algorithm: Conditional Normal updating in polynomial state space
 *
 * @warning Uses fixed iter=1 for testing purposes
 * @warning Requires consistent state initialization
 *
 * @see generate_theta_01
 * @since version 1.0
 */
SEXP test_generate_theta_01(SEXP theta_1_, SEXP theta_02_, SEXP prec_theta_1_, SEXP mean_theta_01_, SEXP prec_theta_01_) {
  SEXP theta_1 = PROTECT(coerceVector(theta_1_, REALSXP));
  int n = LENGTH(theta_1);
  double *theta_1_ptr = REAL(theta_1);

  SEXP theta_02 = PROTECT(coerceVector(theta_02_, REALSXP));
  double theta_02_val = REAL(theta_02)[0];
  SEXP prec_theta_1 = PROTECT(coerceVector(prec_theta_1_, REALSXP));
  double prec_theta_1_val = REAL(prec_theta_1)[0];
  SEXP mean_theta_01 = PROTECT(coerceVector(mean_theta_01_, REALSXP));
  double mean_theta_01_val = REAL(mean_theta_01)[0];
  SEXP prec_theta_01 = PROTECT(coerceVector(prec_theta_01_, REALSXP));
  double prec_theta_01_val = REAL(prec_theta_01)[0];

  double *theta_1_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_1_post + n, theta_1_ptr, n * sizeof(double));

  double *theta_02_post = (double *) R_alloc(2, sizeof(double));
  theta_02_post[1] = theta_02_val;

  double *prec_theta_1_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_1_post[1] = prec_theta_1_val;

  double *theta_01_post = (double *) R_alloc(2, sizeof(double));

  GetRNGstate();
  generate_theta_01(
    theta_01_post,      /* theta_01_post: initial level states */
    theta_02_post,      /* theta_02_post: initial trend states */
    theta_1_post,       /* theta_1_post: level trajectories */
    prec_theta_1_post,  /* prec_theta_1_post: level precision draws */
    mean_theta_01_val,  /* mean_theta_01: prior mean */
    prec_theta_01_val,  /* prec_theta_01: prior precision */
    n,                  /* n: number of observations */
    1                   /* iter: fixed iteration index */
  );
  PutRNGstate();

  UNPROTECT(5);
  return ScalarReal(theta_01_post[1]);
}

/**
 * @brief Test wrapper for generate_theta_0k from conditional_theta0.c
 *
 * @details Enables testing of intermediate initial state sampling in polynomial
 *          dynamic models. This function validates conditional sampling of the
 *          k-th initial state component given adjacent initial components.
 *
 * @param theta_km1_ SEXP: (k-1)-th state component vector
 * @param theta_k_ SEXP: k-th state component vector
 * @param theta_0km1_ SEXP: Initial (k-1)-th state value
 * @param theta_0kp1_ SEXP: Initial (k+1)-th state value
 * @param prec_km1_ SEXP: Precision for (k-1)-th component
 * @param prec_k_ SEXP: Precision for k-th component
 * @param mean_0k_ SEXP: Prior mean for initial k-th state
 * @param prec_0k_ SEXP: Prior precision for initial k-th state
 * @return Scalar SEXP containing the generated theta_0k sample
 *
 * @note Computational complexity: O(n) for sufficient statistics with state vectors
 * @note Memory access: Manages multiple initial state dependencies
 * @note Algorithm: Conditional Normal updating for initial state components
 *
 * @warning Uses fixed iter=1 for testing purposes
 * @warning Requires consistent initial state structure
 *
 * @see generate_theta_0k
 * @since version 1.0
 */
SEXP test_generate_theta_0k(SEXP theta_km1_, SEXP theta_k_, SEXP theta_0km1_, SEXP theta_0kp1_, SEXP prec_km1_, SEXP prec_k_, SEXP mean_0k_, SEXP prec_0k_) {
  SEXP theta_k = PROTECT(coerceVector(theta_k_, REALSXP));
  int n = LENGTH(theta_k);
  double *theta_k_ptr = REAL(theta_k);

  SEXP theta_km1 = PROTECT(coerceVector(theta_km1_, REALSXP));
  double *theta_km1_ptr = REAL(theta_km1);
  SEXP theta_0km1 = PROTECT(coerceVector(theta_0km1_, REALSXP));
  double theta_0km1_val = REAL(theta_0km1)[0];
  SEXP theta_0kp1 = PROTECT(coerceVector(theta_0kp1_, REALSXP));
  double theta_0kp1_val = REAL(theta_0kp1)[0];
  SEXP prec_km1 = PROTECT(coerceVector(prec_km1_, REALSXP));
  double prec_km1_val = REAL(prec_km1)[0];
  SEXP prec_k = PROTECT(coerceVector(prec_k_, REALSXP));
  double prec_k_val = REAL(prec_k)[0];
  SEXP mean_0k = PROTECT(coerceVector(mean_0k_, REALSXP));
  double mean_0k_val = REAL(mean_0k)[0];
  SEXP prec_0k = PROTECT(coerceVector(prec_0k_, REALSXP));
  double prec_0k_val = REAL(prec_0k)[0];

  double *theta_km1_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_km1_post, theta_km1_ptr, n * sizeof(double));

  double *theta_k_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_k_post + n, theta_k_ptr, n * sizeof(double));

  double *theta_0km1_post = (double *) R_alloc(2, sizeof(double));
  theta_0km1_post[0] = theta_0km1_val;

  double *theta_0kp1_post = (double *) R_alloc(2, sizeof(double));
  theta_0kp1_post[1] = theta_0kp1_val;

  double *prec_theta_km1_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_km1_post[0] = prec_km1_val;

  double *prec_theta_k_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_k_post[1] = prec_k_val;

  double *theta_0k_post = (double *) R_alloc(2, sizeof(double));

  GetRNGstate();
  generate_theta_0k(
    theta_0km1_post,   /* theta_0km1_post: initial state k-1 */
    theta_0k_post,     /* theta_0k_post: initial state k */
    theta_0kp1_post,   /* theta_0kp1_post: initial state k+1 */
    theta_km1_post,    /* theta_km1_post: trajectories for component k-1 */
    theta_k_post,      /* theta_k_post: trajectories for component k */
    prec_theta_km1_post,/* prec_theta_km1_post: precision for component k-1 */
    prec_theta_k_post,  /* prec_theta_k_post: precision for component k */
    mean_0k_val,       /* mean_theta_0k: prior mean */
    prec_0k_val,       /* prec_theta_0k: prior precision */
    n,                 /* n: number of observations */
    1                  /* iter: fixed iteration index */
  );
  PutRNGstate();

  UNPROTECT(8);
  return ScalarReal(theta_0k_post[1]);
}

/**
 * @brief Test wrapper for generate_theta_0p from conditional_theta0.c
 *
 * @details Enables testing of final initial state sampling in polynomial
 *          dynamic models. This boundary case handles the initial value for
 *          the highest-order state component.
 *
 * @param theta_pm1_ SEXP: (p-1)-th state component vector
 * @param theta_p_ SEXP: p-th state component vector
 * @param theta_0pm1_ SEXP: Initial (p-1)-th state value
 * @param prec_pm1_ SEXP: Precision for (p-1)-th component
 * @param prec_p_ SEXP: Precision for p-th component
 * @param mean_0p_ SEXP: Prior mean for initial p-th state
 * @param prec_0p_ SEXP: Prior precision for initial p-th state
 * @return Scalar SEXP containing the generated theta_0p sample
 *
 * @note Computational complexity: O(n) for boundary initial state calculation
 * @note Memory access: Handles final initial state in polynomial sequence
 * @note Algorithm: Conditional Normal updating for boundary initial state
 *
 * @warning Uses fixed iter=1 for testing purposes
 * @warning Boundary conditions affect prior-likelihood balance
 *
 * @see generate_theta_0p
 * @since version 1.0
 */
SEXP test_generate_theta_0p(SEXP theta_pm1_, SEXP theta_p_, SEXP theta_0pm1_, SEXP prec_pm1_, SEXP prec_p_, SEXP mean_0p_, SEXP prec_0p_) {
  SEXP theta_p = PROTECT(coerceVector(theta_p_, REALSXP));
  int n = LENGTH(theta_p);
  double *theta_p_ptr = REAL(theta_p);

  SEXP theta_pm1 = PROTECT(coerceVector(theta_pm1_, REALSXP));
  double *theta_pm1_ptr = REAL(theta_pm1);
  SEXP theta_0pm1 = PROTECT(coerceVector(theta_0pm1_, REALSXP));
  double theta_0pm1_val = REAL(theta_0pm1)[0];
  SEXP prec_pm1 = PROTECT(coerceVector(prec_pm1_, REALSXP));
  double prec_pm1_val = REAL(prec_pm1)[0];
  SEXP prec_p = PROTECT(coerceVector(prec_p_, REALSXP));
  double prec_p_val = REAL(prec_p)[0];
  SEXP mean_0p = PROTECT(coerceVector(mean_0p_, REALSXP));
  double mean_0p_val = REAL(mean_0p)[0];
  SEXP prec_0p = PROTECT(coerceVector(prec_0p_, REALSXP));
  double prec_0p_val = REAL(prec_0p)[0];

  double *theta_pm1_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_pm1_post, theta_pm1_ptr, n * sizeof(double));

  double *theta_p_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_p_post + n, theta_p_ptr, n * sizeof(double));

  double *theta_0pm1_post = (double *) R_alloc(2, sizeof(double));
  theta_0pm1_post[0] = theta_0pm1_val;

  double *prec_theta_pm1_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_pm1_post[0] = prec_pm1_val;

  double *prec_theta_p_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_p_post[1] = prec_p_val;

  double *theta_0p_post = (double *) R_alloc(2, sizeof(double));

  GetRNGstate();
  generate_theta_0p(
    theta_0pm1_post,   /* theta_0pm1_post: initial state p-1 */
    theta_0p_post,     /* theta_0p_post: initial state p */
    theta_pm1_post,    /* theta_pm1_post: trajectories for component p-1 */
    theta_p_post,      /* theta_p_post: trajectories for component p */
    prec_theta_pm1_post,/* prec_theta_pm1_post: precision for component p-1 */
    prec_theta_p_post,  /* prec_theta_p_post: precision for component p */
    mean_0p_val,       /* mean_theta_0p: prior mean */
    prec_0p_val,       /* prec_theta_0p: prior precision */
    n,                 /* n: number of observations */
    1                  /* iter: fixed iteration index */
  );
  PutRNGstate();

  UNPROTECT(7);
  return ScalarReal(theta_0p_post[1]);
}

//==============================================================================
// BINOMIAL CWMH WRAPPERS
//==============================================================================

/**
 * @brief Test wrapper for cwmh_alpha_logit_binomial_locallevel from cwmh_binomial.c
 *
 * @details Enables testing of Component-Wise Metropolis-Hastings algorithm for
 *          binomial local level models with logit link. This function validates
 *          the adaptive MCMC sampling of state parameters and success probabilities
 *          in non-Gaussian observation models.
 *
 *          **Version 1.5 corrections:**
 *          Updated function to match actual CWMH implementation. Uses explicit
 *          TEST_LAG_UPDATE constant for reproducible testing behavior.
 *
 * @param theta_1_in_ SEXP: Input first-order state vector
 * @param theta_01_in_ SEXP: Input initial state value
 * @param prec_1_in_ SEXP: Input state precision parameter
 * @param y_ SEXP: Binomial count observations
 * @param n_trials_ SEXP: Number of trials for binomial model
 * @param log_sigma_in_ SEXP: Input log proposal standard deviations
 * @return Named list with theta_1 and alpha components only
 *
 * @note Computational complexity: O(n) for component-wise MH updates
 * @note Memory access: Manages MCMC state arrays and working memory
 * @note Algorithm: Adaptive CWMH with logit transform for binomial likelihood
 * @note Return structure: Only theta_1 and alpha (no individual acceptance indicators)
 * @note Test parameter: Uses TEST_LAG_UPDATE = 50 for consistent testing
 *
 * @warning Uses external log_sigma input for adaptation loop closure
 * @warning Requires proper initialization of proposal variances
 *
 * @see cwmh_alpha_logit_binomial_locallevel
 * @since version 1.3
 */
SEXP test_cwmh_alpha_logit_binomial_locallevel(SEXP theta_1_in_, SEXP theta_01_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_, SEXP log_sigma_in_) {
  // Test configuration constants
  const int TEST_LAG_UPDATE = 50;
  const int TEST_ITER = 1;

  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  int n = LENGTH(y);
  double *y_ptr = REAL(y);

  SEXP n_trials = PROTECT(coerceVector(n_trials_, REALSXP));
  double n_trials_val = REAL(n_trials)[0];

  // MCMC history arrays
  double *theta_1_post = (double*) R_alloc(2 * n, sizeof(double));
  SEXP theta_1_in = PROTECT(coerceVector(theta_1_in_, REALSXP));
  memcpy(theta_1_post, REAL(theta_1_in), n * sizeof(double));

  double *theta_01_post = (double*) R_alloc(2, sizeof(double));
  SEXP theta_01_in = PROTECT(coerceVector(theta_01_in_, REALSXP));
  theta_01_post[0] = REAL(theta_01_in)[0];

  double *prec_1_post = (double*) R_alloc(2, sizeof(double));
  SEXP prec_1_in = PROTECT(coerceVector(prec_1_in_, REALSXP));
  prec_1_post[0] = REAL(prec_1_in)[0];

  double *theta_1_updated = (double*) R_alloc(2 * n, sizeof(double));
  double *alpha_post = (double*) R_alloc(2 * n, sizeof(double));

  // CWMH working arrays
  SEXP log_sigma_in = PROTECT(coerceVector(log_sigma_in_, REALSXP));
  double *log_sigma = REAL(log_sigma_in); // Use log_sigma from R
  double *hat_theta_1 = (double*) R_alloc(n, sizeof(double));
  double *theta_1_new = (double*) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double*) R_alloc(n, sizeof(double));

  GetRNGstate();
  cwmh_alpha_logit_binomial_locallevel(
    theta_1_post, theta_01_post, theta_1_updated, alpha_post, prec_1_post, y_ptr,
    log_sigma, hat_theta_1, theta_1_new, log_accept_prob,
    TEST_LAG_UPDATE, n_trials_val, n, TEST_ITER
  );
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 2));
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));

  memcpy(REAL(theta_1_out), theta_1_post + n, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_post + n, n * sizeof(double));

  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);

  SEXP nms = PROTECT(allocVector(STRSXP, 2));
  SET_STRING_ELT(nms, 0, mkChar("theta_1"));
  SET_STRING_ELT(nms, 1, mkChar("alpha"));
  setAttrib(res, R_NamesSymbol, nms);

  UNPROTECT(10);
  return res;
}

/**
 * @brief Test wrapper for cwmh_alpha_logit_binomial from cwmh_binomial.c
 *
 * @details Enables testing of Component-Wise Metropolis-Hastings algorithm for
 *          binomial local trend models with logit link. This function validates
 *          the MCMC sampling in polynomial dynamic models with non-Gaussian
 *          observation structures.
 *
 *          **Version 1.5 corrections:**
 *          Updated function to match actual CWMH implementation. Uses explicit
 *          TEST_LAG_UPDATE constant for reproducible testing behavior.
 *
 * @param theta_1_in_ SEXP: Input first-order state vector
 * @param theta_2_in_ SEXP: Input second-order state vector
 * @param theta_01_in_ SEXP: Input initial first-order state value
 * @param theta_02_in_ SEXP: Input initial second-order state value
 * @param prec_1_in_ SEXP: Input state precision parameter
 * @param y_ SEXP: Binomial count observations
 * @param n_trials_ SEXP: Number of trials for binomial model
 * @return Named list with theta_1 and alpha components only
 *
 * @note Computational complexity: O(n) for component-wise MH in trend model
 * @note Memory access: Coordinates multiple state vectors and parameters
 * @note Algorithm: CWMH for polynomial state space with binomial observations
 * @note Return structure: Only theta_1 and alpha (no individual acceptance indicators)
 * @note Test parameter: Uses TEST_LAG_UPDATE = 50 for consistent testing
 *
 * @warning Initializes log_sigma internally with default values
 * @warning Requires consistent state vector dimensions
 *
 * @see cwmh_alpha_logit_binomial
 * @since version 1.0
 */
SEXP test_cwmh_alpha_logit_binomial(SEXP theta_1_in_, SEXP theta_2_in_, SEXP theta_01_in_, SEXP theta_02_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_) {
  // Test configuration constants
  const int TEST_LAG_UPDATE = 50;
  const int TEST_ITER = 1;
  const double DEFAULT_LOG_SIGMA = log(0.1);

  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  int n = LENGTH(y);
  double *y_ptr = REAL(y);

  SEXP n_trials = PROTECT(coerceVector(n_trials_, REALSXP));
  double n_trials_val = REAL(n_trials)[0];

  // MCMC history arrays
  double *theta_1_post = (double*) R_alloc(2 * n, sizeof(double));
  SEXP theta_1_in = PROTECT(coerceVector(theta_1_in_, REALSXP));
  memcpy(theta_1_post, REAL(theta_1_in), n * sizeof(double));

  double *theta_2_post = (double*) R_alloc(2 * n, sizeof(double));
  SEXP theta_2_in = PROTECT(coerceVector(theta_2_in_, REALSXP));
  memcpy(theta_2_post + n, REAL(theta_2_in), n * sizeof(double));

  double *theta_01_post = (double*) R_alloc(2, sizeof(double));
  SEXP theta_01_in = PROTECT(coerceVector(theta_01_in_, REALSXP));
  theta_01_post[0] = REAL(theta_01_in)[0];

  double *theta_02_post = (double*) R_alloc(2, sizeof(double));
  SEXP theta_02_in = PROTECT(coerceVector(theta_02_in_, REALSXP));
  theta_02_post[0] = REAL(theta_02_in)[0];

  double *prec_1_post = (double*) R_alloc(2, sizeof(double));
  SEXP prec_1_in = PROTECT(coerceVector(prec_1_in_, REALSXP));
  prec_1_post[0] = REAL(prec_1_in)[0];

  double *theta_1_updated = (double*) R_alloc(2 * n, sizeof(double));
  double *alpha_post = (double*) R_alloc(2 * n, sizeof(double));

  // CWMH working arrays
  double *log_sigma = (double*) R_alloc(n, sizeof(double));
  for(int i = 0; i < n; i++) log_sigma[i] = DEFAULT_LOG_SIGMA;
  double *hat_theta_1 = (double*) R_alloc(n, sizeof(double));
  double *theta_1_new = (double*) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double*) R_alloc(n, sizeof(double));

  GetRNGstate();
  cwmh_alpha_logit_binomial(
    theta_1_post,     /* theta_1: level states */
    theta_2_post,     /* theta_2: trend states */
    theta_01_post,    /* theta_01: initial level states */
    theta_02_post,    /* theta_02: initial trend states */
    theta_1_updated,  /* theta_1_updated: sliding window indicators */
    alpha_post,       /* alpha: success probabilities */
    prec_1_post,      /* prec_theta_1: level precisions */
    y_ptr,            /* y: observed counts */
    log_sigma,        /* log_sigma: proposal log standard deviations */
    hat_theta_1,      /* hat_theta_1: conditional means */
    theta_1_new,      /* theta_1_new: proposal buffer */
    log_accept_prob,  /* log_accept_prob: log acceptance storage */
    TEST_LAG_UPDATE,  /* lag_update: adaptation window length */
    n_trials_val,     /* n_trials: number of binomial trials */
    n,                /* n: number of observations */
    TEST_ITER         /* iter: iteration index */
  );
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 2));
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));

  memcpy(REAL(theta_1_out), theta_1_post + n, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_post + n, n * sizeof(double));

  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);

  SEXP nms = PROTECT(allocVector(STRSXP, 2));
  SET_STRING_ELT(nms, 0, mkChar("theta_1"));
  SET_STRING_ELT(nms, 1, mkChar("alpha"));
  setAttrib(res, R_NamesSymbol, nms);

  UNPROTECT(11);
  return res;
}

//==============================================================================
// BINOMIAL ALPHA GENERATION WRAPPERS
//==============================================================================

/**
 * @brief Test wrapper for generate_alpha_logit_binomial_locallevel from generate_alpha_binomial.c
 *
 * @details Enables testing of adaptive alpha generation in binomial local level models.
 *          This function validates the complete adaptive MCMC workflow including
 *          Component-Wise Metropolis-Hastings with automatic proposal tuning
 *          for success probability parameters.
 *
 *          **Version 1.5 corrections:**
 *          Updated to use explicit test constants and proper min_deviation_threshold
 *          parameter. Simplified adaptation parameters for testing consistency.
 *
 * @param theta_1_in_ SEXP: Input first-order state vector
 * @param theta_01_in_ SEXP: Input initial state value
 * @param prec_1_in_ SEXP: Input state precision parameter
 * @param y_ SEXP: Binomial count observations
 * @param n_trials_ SEXP: Number of trials for binomial model
 * @return Named list with theta_1 and alpha components only
 *
 * @note Computational complexity: O(n) for adaptive MCMC with tuning
 * @note Memory access: Manages adaptive arrays and working memory
 * @note Algorithm: Adaptive CWMH with automatic proposal variance tuning
 * @note Return structure: Only theta_1 and alpha (no individual acceptance indicators)
 * @note Test parameters: Uses explicit constants for reproducible testing
 *
 * @warning Uses fixed adaptation parameters for testing
 * @warning Initializes proposal variances with default values
 *
 * @see generate_alpha_logit_binomial_locallevel
 * @since version 1.0
 */
SEXP test_generate_alpha_logit_binomial_locallevel(SEXP theta_1_in_, SEXP theta_01_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_) {
  // Test configuration constants
  const int TEST_LAG_UPDATE = 50;
  const int TEST_ITER = 1;
  const double TEST_MAX_STEP_SIZE = 0.1;
  const double TEST_BASE_ADAPTATION_RATE = 1.0;
  const double TEST_DECAY_EXPONENT = 0.5;
  const double TEST_TARGET_ACCEPTANCE = 0.44;
  const double TEST_MIN_DEVIATION_THRESHOLD = 0.02;  // 1/TEST_LAG_UPDATE
  const double DEFAULT_LOG_SIGMA = log(0.1);

  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  int n = LENGTH(y);
  double *y_ptr = REAL(y);

  SEXP n_trials = PROTECT(coerceVector(n_trials_, REALSXP));
  double n_trials_val = REAL(n_trials)[0];

  // MCMC history arrays
  double *theta_1_post = (double*) R_alloc(2 * n, sizeof(double));
  SEXP theta_1_in = PROTECT(coerceVector(theta_1_in_, REALSXP));
  memcpy(theta_1_post, REAL(theta_1_in), n * sizeof(double));

  double *theta_01_post = (double*) R_alloc(2, sizeof(double));
  SEXP theta_01_in = PROTECT(coerceVector(theta_01_in_, REALSXP));
  theta_01_post[0] = REAL(theta_01_in)[0];

  double *prec_1_post = (double*) R_alloc(2, sizeof(double));
  SEXP prec_1_in = PROTECT(coerceVector(prec_1_in_, REALSXP));
  prec_1_post[0] = REAL(prec_1_in)[0];

  double *theta_1_updated = (double*) R_alloc(2 * n, sizeof(double));
  double *alpha_post = (double*) R_alloc(2 * n, sizeof(double));

  // CWMH working arrays
  double *accept_prop = (double*) R_alloc(n, sizeof(double));
  double *log_sigma = (double*) R_alloc(n, sizeof(double));
  for(int i = 0; i < n; i++) log_sigma[i] = DEFAULT_LOG_SIGMA;
  double *hat_theta_1 = (double*) R_alloc(n, sizeof(double));
  double *theta_1_new = (double*) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double*) R_alloc(n, sizeof(double));

  GetRNGstate();
  generate_alpha_logit_binomial_locallevel(
    theta_1_post,               /* theta_1: level states */
    theta_01_post,              /* theta_01: initial level states */
    theta_1_updated,            /* theta_1_updated: sliding window indicators */
    alpha_post,                 /* alpha: success probabilities */
    prec_1_post,                /* prec_theta_1: level precisions */
    y_ptr,                      /* y: observed counts */
    accept_prop,                /* accept_prop: acceptance proportions */
    log_sigma,                  /* log_sigma: proposal log standard deviations */
    hat_theta_1,                /* hat_theta_1: conditional means */
    theta_1_new,                /* theta_1_new: proposal buffer */
    log_accept_prob,            /* log_accept_prob: log acceptance storage */
    TEST_LAG_UPDATE,            /* lag_update: adaptation window length */
    n_trials_val,               /* n_trials: number of binomial trials */
    n,                          /* n: number of observations */
    TEST_ITER,                  /* iter: iteration index */
    TEST_MAX_STEP_SIZE,         /* max_step_size: adaptation step cap */
    TEST_BASE_ADAPTATION_RATE,  /* base_adaptation_rate: initial adaptation rate */
    TEST_DECAY_EXPONENT,        /* decay_exponent: diminishing schedule */
    TEST_TARGET_ACCEPTANCE,     /* target_acceptance: desired acceptance proportion */
    TEST_MIN_DEVIATION_THRESHOLD/* min_deviation_threshold: deviation trigger */
  );
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 2));
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));

  memcpy(REAL(theta_1_out), theta_1_post + n, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_post + n, n * sizeof(double));

  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);

  SEXP nms = PROTECT(allocVector(STRSXP, 2));
  SET_STRING_ELT(nms, 0, mkChar("theta_1"));
  SET_STRING_ELT(nms, 1, mkChar("alpha"));
  setAttrib(res, R_NamesSymbol, nms);

  UNPROTECT(9);
  return res;
}

/**
 * @brief Test wrapper for generate_alpha_logit_binomial from generate_alpha_binomial.c
 *
 * @details Enables testing of adaptive alpha generation in binomial local trend models.
 *          This function validates the adaptive CWMH workflow for a trend model,
 *          including proposal tuning and component-wise updates.
 *
 *          **Version 1.5 corrections:**
 *          Updated to use explicit test constants and proper min_deviation_threshold
 *          parameter. Simplified adaptation parameters for testing consistency.
 *
 * @param theta_1_in_ SEXP: Input first-order state vector
 * @param theta_2_in_ SEXP: Input second-order state vector
 * @param theta_01_in_ SEXP: Input initial first-order state value
 * @param theta_02_in_ SEXP: Input initial second-order state value
 * @param prec_1_in_ SEXP: Input state precision parameter
 * @param y_ SEXP: Binomial count observations
 * @param n_trials_ SEXP: Number of trials for binomial model
 * @return Named list with theta_1 and alpha components only
 *
 * @note Computational complexity: O(n) for adaptive MCMC with tuning
 * @note Memory access: Manages adaptive arrays and working memory
 * @note Algorithm: Adaptive CWMH with automatic proposal variance tuning (trend)
 * @note Return structure: Only theta_1 and alpha (no individual acceptance indicators)
 * @note Test parameters: Uses explicit constants for reproducible testing
 *
 * @warning Uses fixed adaptation parameters for testing
 * @warning Initializes proposal variances with default values
 *
 * @see generate_alpha_logit_binomial
 * @since version 1.0
 */
SEXP test_generate_alpha_logit_binomial(SEXP theta_1_in_, SEXP theta_2_in_, SEXP theta_01_in_, SEXP theta_02_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_) {
  // Test configuration constants
  const int TEST_LAG_UPDATE = 50;
  const int TEST_ITER = 1;
  const double TEST_MAX_STEP_SIZE = 0.1;
  const double TEST_BASE_ADAPTATION_RATE = 1.0;
  const double TEST_DECAY_EXPONENT = 0.5;
  const double TEST_TARGET_ACCEPTANCE = 0.44;
  const double TEST_MIN_DEVIATION_THRESHOLD = 0.02;  // 1/TEST_LAG_UPDATE
  const double DEFAULT_LOG_SIGMA = log(0.1);

  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  int n = LENGTH(y);
  double *y_ptr = REAL(y);

  SEXP n_trials = PROTECT(coerceVector(n_trials_, REALSXP));
  double n_trials_val = REAL(n_trials)[0];

  // MCMC history arrays
  double *theta_1_post = (double*) R_alloc(2 * n, sizeof(double));
  SEXP theta_1_in = PROTECT(coerceVector(theta_1_in_, REALSXP));
  memcpy(theta_1_post, REAL(theta_1_in), n * sizeof(double));

  double *theta_2_post = (double*) R_alloc(2 * n, sizeof(double));
  SEXP theta_2_in = PROTECT(coerceVector(theta_2_in_, REALSXP));
  memcpy(theta_2_post + n, REAL(theta_2_in), n * sizeof(double));

  double *theta_01_post = (double*) R_alloc(2, sizeof(double));
  SEXP theta_01_in = PROTECT(coerceVector(theta_01_in_, REALSXP));
  theta_01_post[0] = REAL(theta_01_in)[0];

  double *theta_02_post = (double*) R_alloc(2, sizeof(double));
  SEXP theta_02_in = PROTECT(coerceVector(theta_02_in_, REALSXP));
  theta_02_post[0] = REAL(theta_02_in)[0];

  double *prec_1_post = (double*) R_alloc(2, sizeof(double));
  SEXP prec_1_in = PROTECT(coerceVector(prec_1_in_, REALSXP));
  prec_1_post[0] = REAL(prec_1_in)[0];

  double *theta_1_updated = (double*) R_alloc(2 * n, sizeof(double));
  double *alpha_post = (double*) R_alloc(2 * n, sizeof(double));

  // CWMH working arrays
  double *accept_prop = (double*) R_alloc(n, sizeof(double));
  double *log_sigma = (double*) R_alloc(n, sizeof(double));
  for(int i = 0; i < n; i++) log_sigma[i] = DEFAULT_LOG_SIGMA;
  double *hat_theta_1 = (double*) R_alloc(n, sizeof(double));
  double *theta_1_new = (double*) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double*) R_alloc(n, sizeof(double));

  GetRNGstate();
  generate_alpha_logit_binomial(
    theta_1_post,               /* theta_1: level states */
    theta_2_post,               /* theta_2: trend states */
    theta_01_post,              /* theta_01: initial level states */
    theta_02_post,              /* theta_02: initial trend states */
    theta_1_updated,            /* theta_1_updated: sliding window indicators */
    alpha_post,                 /* alpha: success probabilities */
    prec_1_post,                /* prec_theta_1: level precisions */
    y_ptr,                      /* y: observed counts */
    accept_prop,                /* accept_prop: acceptance proportions */
    log_sigma,                  /* log_sigma: proposal log standard deviations */
    hat_theta_1,                /* hat_theta_1: conditional means */
    theta_1_new,                /* theta_1_new: proposal buffer */
    log_accept_prob,            /* log_accept_prob: log acceptance storage */
    TEST_LAG_UPDATE,            /* lag_update: adaptation window length */
    n_trials_val,               /* n_trials: number of binomial trials */
    n,                          /* n: number of observations */
    TEST_ITER,                  /* iter: iteration index */
    TEST_MAX_STEP_SIZE,         /* max_step_size: adaptation step cap */
    TEST_BASE_ADAPTATION_RATE,  /* base_adaptation_rate: initial adaptation rate */
    TEST_DECAY_EXPONENT,        /* decay_exponent: diminishing schedule */
    TEST_TARGET_ACCEPTANCE,     /* target_acceptance: desired acceptance proportion */
    TEST_MIN_DEVIATION_THRESHOLD/* min_deviation_threshold: deviation trigger */
  );
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 2));
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));

  memcpy(REAL(theta_1_out), theta_1_post + n, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_post + n, n * sizeof(double));

  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);

  SEXP nms = PROTECT(allocVector(STRSXP, 2));
  SET_STRING_ELT(nms, 0, mkChar("theta_1"));
  SET_STRING_ELT(nms, 1, mkChar("alpha"));
  setAttrib(res, R_NamesSymbol, nms);

  UNPROTECT(11);
  return res;
}

/**
 * @brief Test wrapper for generate_alpha_probit_bernoulli_locallevel
 *
 * @details Exposes the Gibbs sampler for probit-linked Bernoulli local level models
 *          implemented in C to the R testing environment. The wrapper prepares the
 *          minimal two-iteration history required by the underlying sampler,
 *          performs input validation, and returns the sampled level states and
 *          corresponding probabilities for the first stored iteration.
 *
 * @param theta_1_in_ SEXP: Numeric vector with previous iteration level states.
 * @param theta_01_in_ SEXP: Numeric scalar with previous iteration initial level state.
 * @param prec_1_in_  SEXP: Numeric scalar with previous iteration level precision.
 * @param y_          SEXP: Numeric vector with Bernoulli outcomes (0/1).
 * @return Named list with elements 'theta_1' and 'alpha'.
 *
 * @note The sampler relies on Albert-Chib latent variable augmentation and uses
 *       generate_normal_vector internally for efficient Gaussian draws.
 *
 * @warning Input vectors must be finite and of matching lengths.
 */
SEXP test_generate_alpha_probit_bernoulli_locallevel(SEXP theta_1_in_, SEXP theta_01_in_,
                                                     SEXP prec_1_in_, SEXP y_) {

  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  if (LENGTH(y) == 0) {
    UNPROTECT(1);
    error("Input vector 'y' must have positive length");
  }
  int n = LENGTH(y);
  double *y_ptr = REAL(y);

  SEXP theta_1_in = PROTECT(coerceVector(theta_1_in_, REALSXP));
  if (LENGTH(theta_1_in) != n) {
    UNPROTECT(2);
    error("theta_1_in must have length %d to match y, got %d", n, LENGTH(theta_1_in));
  }

  SEXP theta_01_in = PROTECT(coerceVector(theta_01_in_, REALSXP));
  SEXP prec_1_in = PROTECT(coerceVector(prec_1_in_, REALSXP));

  double *theta_1_store = (double*) R_alloc(2 * n, sizeof(double));
  memcpy(theta_1_store, REAL(theta_1_in), n * sizeof(double));

  double *theta_01_store = (double*) R_alloc(2, sizeof(double));
  theta_01_store[0] = REAL(theta_01_in)[0];

  double *prec_1_store = (double*) R_alloc(2, sizeof(double));
  prec_1_store[0] = REAL(prec_1_in)[0];

  double *alpha_store = (double*) R_alloc(2 * n, sizeof(double));
  memset(alpha_store, 0, 2 * n * sizeof(double));

  double *v_latent = (double*) R_alloc(n, sizeof(double));
  double *rhs_vector = (double*) R_alloc(n, sizeof(double));

  GetRNGstate();
  generate_alpha_probit_bernoulli_locallevel(
    theta_1_store, /* theta_1: two-iteration storage for level state */
    theta_01_store, /* theta_01: two-iteration storage for initial level */
    alpha_store,    /* alpha: two-iteration storage for success probabilities */
    prec_1_store,   /* prec_1: two-iteration storage for level precision */
    y_ptr,          /* y: Bernoulli outcomes */
    v_latent,       /* v: latent Gaussian draws */
    rhs_vector,     /* rhs: working right-hand side vector */
    n,              /* n: series length */
    1               /* iter: index of iteration to draw */
  );
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 2));
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));

  memcpy(REAL(theta_1_out), theta_1_store + n, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_store + n, n * sizeof(double));

  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);

  SEXP nms = PROTECT(allocVector(STRSXP, 2));
  SET_STRING_ELT(nms, 0, mkChar("theta_1"));
  SET_STRING_ELT(nms, 1, mkChar("alpha"));
  setAttrib(res, R_NamesSymbol, nms);

  UNPROTECT(8);
  return res;
}

/**
 * @brief Test wrapper for generate_alpha_probit_bernoulli (local trend model)
 *
 * @details Similar to test_generate_alpha_probit_bernoulli_locallevel but additionally
 *          accepts the trend state history required by the local trend sampler.
 *          The wrapper mirrors the two-iteration storage layout expected by the
 *          underlying C routine and returns sampled level states alongside the
 *          implied Bernoulli probabilities for verification in R tests.
 *
 * @param theta_1_in_ SEXP: Numeric vector with previous iteration level states.
 * @param theta_2_in_ SEXP: Numeric vector with current iteration trend states.
 * @param theta_01_in_ SEXP: Numeric scalar with previous iteration initial level state.
 * @param theta_02_in_ SEXP: Numeric scalar with previous iteration initial trend state.
 * @param prec_1_in_  SEXP: Numeric scalar with previous iteration level precision.
 * @param y_          SEXP: Numeric vector with Bernoulli outcomes (0/1).
 * @return Named list with elements 'theta_1' and 'alpha'.
 */
SEXP test_generate_alpha_probit_bernoulli(SEXP theta_1_in_, SEXP theta_2_in_,
                                          SEXP theta_01_in_, SEXP theta_02_in_,
                                          SEXP prec_1_in_, SEXP y_) {

  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  if (LENGTH(y) == 0) {
    UNPROTECT(1);
    error("Input vector 'y' must have positive length");
  }
  int n = LENGTH(y);
  double *y_ptr = REAL(y);

  SEXP theta_1_in = PROTECT(coerceVector(theta_1_in_, REALSXP));
  if (LENGTH(theta_1_in) != n) {
    UNPROTECT(2);
    error("theta_1_in must have length %d to match y, got %d", n, LENGTH(theta_1_in));
  }

  SEXP theta_2_in = PROTECT(coerceVector(theta_2_in_, REALSXP));
  if (LENGTH(theta_2_in) != n) {
    UNPROTECT(3);
    error("theta_2_in must have length %d to match y, got %d", n, LENGTH(theta_2_in));
  }

  SEXP theta_01_in = PROTECT(coerceVector(theta_01_in_, REALSXP));
  SEXP theta_02_in = PROTECT(coerceVector(theta_02_in_, REALSXP));
  SEXP prec_1_in = PROTECT(coerceVector(prec_1_in_, REALSXP));

  double *theta_1_store = (double*) R_alloc(2 * n, sizeof(double));
  memcpy(theta_1_store, REAL(theta_1_in), n * sizeof(double));

  double *theta_2_store = (double*) R_alloc(2 * n, sizeof(double));
  const double *theta_2_src = REAL(theta_2_in);
  memcpy(theta_2_store, theta_2_src, n * sizeof(double));
  memcpy(theta_2_store + n, theta_2_src, n * sizeof(double));

  double *theta_01_store = (double*) R_alloc(2, sizeof(double));
  theta_01_store[0] = REAL(theta_01_in)[0];

  double *theta_02_store = (double*) R_alloc(2, sizeof(double));
  double theta_02_value = REAL(theta_02_in)[0];
  theta_02_store[0] = theta_02_value;
  theta_02_store[1] = theta_02_value;

  double *prec_1_store = (double*) R_alloc(2, sizeof(double));
  prec_1_store[0] = REAL(prec_1_in)[0];

  double *alpha_store = (double*) R_alloc(2 * n, sizeof(double));
  memset(alpha_store, 0, 2 * n * sizeof(double));

  double *v_latent = (double*) R_alloc(n, sizeof(double));
  double *rhs_vector = (double*) R_alloc(n, sizeof(double));

  GetRNGstate();
  generate_alpha_probit_bernoulli(
    theta_1_store, /* theta_1: two-iteration storage for level state */
    theta_2_store, /* theta_2: two-iteration storage for trend state */
    theta_01_store, /* theta_01: two-iteration storage for initial level */
    theta_02_store, /* theta_02: two-iteration storage for initial trend */
    alpha_store,   /* alpha: two-iteration storage for success probabilities */
    prec_1_store,  /* prec_1: two-iteration storage for level precision */
    y_ptr,         /* y: Bernoulli outcomes */
    v_latent,      /* v: latent Gaussian draws */
    rhs_vector,    /* rhs: working right-hand side vector */
    n,             /* n: series length */
    1              /* iter: index of iteration to draw */
  );
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 2));
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));

  memcpy(REAL(theta_1_out), theta_1_store + n, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_store + n, n * sizeof(double));

  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);

  SEXP nms = PROTECT(allocVector(STRSXP, 2));
  SET_STRING_ELT(nms, 0, mkChar("theta_1"));
  SET_STRING_ELT(nms, 1, mkChar("alpha"));
  setAttrib(res, R_NamesSymbol, nms);

  UNPROTECT(10);
  return res;
}

//==============================================================================
// FULL MCMC TEST WRAPPER
//==============================================================================

/**
 * @brief Test wrapper for complete MCMC simulation with fixed parameters and diagnostic outputs
 *
 * @details Comprehensive testing function for the full MCMC binomial local level
 *          algorithm with ability to fix specific parameters for validation.
 *          Supports selective parameter fixing (theta_1, theta_01, prec_1) by
 *          passing NULL or valid values, enabling thorough algorithm testing.
 *
 *          Includes optional diagnostic outputs (accrate, log_sigma) and always
 *          returns success probabilities (alpha) for comprehensive validation.
 *
 *          **Version 1.5 improvements:**
 *          - Enhanced input validation with meaningful error messages
 *          - Simplified UNPROTECT counting logic for better maintainability
 *          - Improved memory management with consistent allocation patterns
 *          - Added explicit constants for numerical thresholds
 *          - Enhanced sanity checks for numerical stability
 *
 * @param y_ SEXP: Binomial count observations (size n)
 * @param n_trials_ SEXP: Number of trials for binomial model (scalar)
 * @param burnin_ SEXP: Number of burn-in iterations (scalar)
 * @param thinning_ SEXP: Thinning interval for sample storage (scalar)
 * @param n_chain_ SEXP: Number of samples to store after burn-in (scalar)
 * @param theta_1_true_ SEXP: Fixed theta_1 values (NULL or size n vector)
 * @param theta_01_true_ SEXP: Fixed theta_01 value (NULL or scalar)
 * @param prec_1_true_ SEXP: Fixed precision value (NULL or scalar)
 * @param prior_theta01_mean_ SEXP: Prior mean for theta_01 (scalar)
 * @param prior_theta01_prec_ SEXP: Prior precision for theta_01 (scalar)
 * @param prior_prec1_shape_ SEXP: Prior shape for precision (scalar)
 * @param prior_prec1_rate_ SEXP: Prior rate for precision (scalar)
 * @param lag_update_ SEXP: Adaptation window size (scalar)
 * @param max_step_size_ SEXP: Maximum adaptation step size (scalar)
 * @param base_adaptation_rate_ SEXP: Base adaptation rate (scalar)
 * @param decay_exponent_ SEXP: Adaptation decay exponent (scalar)
 * @param target_acceptance_ SEXP: Target acceptance rate (scalar)
 * @param return_log_sigma_ SEXP: Logical scalar, whether to return log_sigma diagnostics
 * @param return_accept_prop_ SEXP: Logical scalar, whether to return accrate diagnostics
 * @return Named list with MCMC samples and diagnostics:
 *         - theta_1: Matrix [n_chain x n] of state samples
 *         - theta_01: Vector [n_chain] of initial state samples
 *         - prec_1: Vector [n_chain] of precision samples
 *         - alpha: Matrix [n_chain x n] of success probability samples
 *         - log_sigma: Matrix [n_chain x n] of proposal scales (if requested)
 *         - accrate: Matrix [n_chain x n] of acceptance rates (if requested)
 *
 * @note Supports flexible parameter fixing by checking for NULL values
 * @note Initializes parameters appropriately when not fixed
 * @note Manages extensive memory allocation for full MCMC simulation
 * @note Returns samples in matrix format compatible with R analysis
 * @note Validates binomial constraints: 0 <= y[i] <= n_trials
 * @note Enforces minimum sample size n >= 3 for numerical stability
 * @warning Large memory requirements for long chains
 * @warning Requires careful memory management with R_Calloc/R_Free
 * @warning Invalid precision parameters may cause numerical instability
 *
 * @complexity O(n_iter * n) for full MCMC simulation
 * @memory Allocates arrays proportional to n_iter * n for state storage
 *
 * @see mcmc_binomial_locallevel functions
 * @see C_MCMC_logit_binomial_locallevel
 * @since version 1.2
 */
SEXP test_mcmc_binomial_locallevel_fixed_params(SEXP y_, SEXP n_trials_, SEXP burnin_, SEXP thinning_, SEXP n_chain_,
                                                SEXP theta_1_true_, SEXP theta_01_true_, SEXP prec_1_true_,
                                                SEXP prior_theta01_mean_, SEXP prior_theta01_prec_,
                                                SEXP prior_prec1_shape_, SEXP prior_prec1_rate_,
                                                SEXP lag_update_, SEXP max_step_size_, SEXP base_adaptation_rate_,
                                                SEXP decay_exponent_, SEXP target_acceptance_,
                                                SEXP return_log_sigma_, SEXP return_accept_prop_) {

  // Numerical constants for validation and initialization
  const int MIN_SAMPLE_SIZE = 3;
  const double MIN_PRECISION_THRESHOLD = 1e-6;
  const double DEFAULT_INIT_SD = 0.1;
  const double LOGIT_BOUND_LIMIT = 10.0;
  const double MIN_DEVIATION_THRESHOLD_FACTOR = 1.0;  // Will be divided by lag_update

  /* Parse data vector and validate */
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  int protected_count = 1;
  int n = LENGTH(y);
  if (n < MIN_SAMPLE_SIZE) {
    UNPROTECT(protected_count);
    error("Sample size 'n' must be at least %d for numerical stability, got %d", MIN_SAMPLE_SIZE, n);
  }
  double *y_ptr = REAL(y);

  SEXP n_trials_sexp = PROTECT(coerceVector(n_trials_, REALSXP));
  protected_count++;
  double n_trials = REAL(n_trials_sexp)[0];

  /* Validate binomial constraints with improved error messages */
  for (int i = 0; i < n; i++) {
    if (y_ptr[i] < 0 || y_ptr[i] > n_trials || !isfinite(y_ptr[i])) {
      UNPROTECT(protected_count);
      error("Invalid observation y[%d] = %f: must satisfy 0 <= y <= n_trials = %f and be finite",
            i + 1, y_ptr[i], n_trials);  // R uses 1-based indexing
    }
  }

  /* Parse MCMC settings with validation */
  SEXP burnin_sexp = PROTECT(coerceVector(burnin_, INTSXP));
  protected_count++;
  int burnin = INTEGER(burnin_sexp)[0];

  SEXP thinning_sexp = PROTECT(coerceVector(thinning_, INTSXP));
  protected_count++;
  int thinning = INTEGER(thinning_sexp)[0];

  SEXP n_chain_sexp = PROTECT(coerceVector(n_chain_, INTSXP));
  protected_count++;
  int n_chain = INTEGER(n_chain_sexp)[0];

  int n_iter = burnin + (n_chain - 1) * thinning + 1;

  /* Enhanced MCMC parameter validation */
  if (burnin < 0) {
    UNPROTECT(protected_count);
    error("Burnin must be non-negative, got %d", burnin);
  }
  if (thinning <= 0) {
    UNPROTECT(protected_count);
    error("Thinning must be positive, got %d", thinning);
  }
  if (n_chain <= 0) {
    UNPROTECT(protected_count);
    error("n_chain must be positive, got %d", n_chain);
  }
  if (n_iter > 1000000) {  // Prevent excessive memory allocation
    UNPROTECT(protected_count);
    error("Total iterations (%d) exceeds safety limit of 1,000,000", n_iter);
  }

  /* Parse fixed parameter flags and values */
  int fix_theta_1 = !isNull(theta_1_true_);
  int fix_theta_01 = !isNull(theta_01_true_);
  int fix_prec_1 = !isNull(prec_1_true_);

  SEXP theta_1_true_sexp = R_NilValue;
  double *theta_1_true = NULL;
  if (fix_theta_1) {
    theta_1_true_sexp = PROTECT(coerceVector(theta_1_true_, REALSXP));
    protected_count++;
    if (LENGTH(theta_1_true_sexp) != n) {
      UNPROTECT(protected_count);
      error("theta_1_true must have length %d to match y, got %d", n, LENGTH(theta_1_true_sexp));
    }
    theta_1_true = REAL(theta_1_true_sexp);
  }

  double theta_01_true = 0.0;
  if (fix_theta_01) {
    SEXP theta_01_true_sexp = PROTECT(coerceVector(theta_01_true_, REALSXP));
    protected_count++;
    theta_01_true = REAL(theta_01_true_sexp)[0];
    if (!isfinite(theta_01_true)) {
      UNPROTECT(protected_count);
      error("theta_01_true must be finite, got %f", theta_01_true);
    }
  }

  double prec_1_true = 0.0;
  if (fix_prec_1) {
    SEXP prec_1_true_sexp = PROTECT(coerceVector(prec_1_true_, REALSXP));
    protected_count++;
    prec_1_true = REAL(prec_1_true_sexp)[0];
    if (prec_1_true <= 0 || !isfinite(prec_1_true)) {
      UNPROTECT(protected_count);
      error("prec_1_true must be positive and finite, got %f", prec_1_true);
    }
  }

  /* Parse priors with enhanced validation */
  SEXP prior_theta01_mean_sexp = PROTECT(coerceVector(prior_theta01_mean_, REALSXP));
  protected_count++;
  double prior_theta01_mean = REAL(prior_theta01_mean_sexp)[0];

  SEXP prior_theta01_prec_sexp = PROTECT(coerceVector(prior_theta01_prec_, REALSXP));
  protected_count++;
  double prior_theta01_prec = REAL(prior_theta01_prec_sexp)[0];

  SEXP prior_prec1_shape_sexp = PROTECT(coerceVector(prior_prec1_shape_, REALSXP));
  protected_count++;
  double prior_prec1_shape = REAL(prior_prec1_shape_sexp)[0];

  SEXP prior_prec1_rate_sexp = PROTECT(coerceVector(prior_prec1_rate_, REALSXP));
  protected_count++;
  double prior_prec1_rate = REAL(prior_prec1_rate_sexp)[0];

  if (prior_theta01_prec <= 0 || !isfinite(prior_theta01_prec)) {
    UNPROTECT(protected_count);
    error("Prior precision for theta_01 must be positive and finite, got %f", prior_theta01_prec);
  }
  if (prior_prec1_shape <= 0 || prior_prec1_rate <= 0 ||
      !isfinite(prior_prec1_shape) || !isfinite(prior_prec1_rate)) {
      UNPROTECT(protected_count);
      error("Prior shape (%f) and rate (%f) must be positive and finite", prior_prec1_shape, prior_prec1_rate);
  }

  /* Parse adaptation parameters */
  SEXP lag_update_sexp = PROTECT(coerceVector(lag_update_, INTSXP));
  protected_count++;
  int lag_update = INTEGER(lag_update_sexp)[0];

  SEXP max_step_size_sexp = PROTECT(coerceVector(max_step_size_, REALSXP));
  protected_count++;
  double max_step_size = REAL(max_step_size_sexp)[0];

  SEXP base_adaptation_rate_sexp = PROTECT(coerceVector(base_adaptation_rate_, REALSXP));
  protected_count++;
  double base_adaptation_rate = REAL(base_adaptation_rate_sexp)[0];

  SEXP decay_exponent_sexp = PROTECT(coerceVector(decay_exponent_, REALSXP));
  protected_count++;
  double decay_exponent = REAL(decay_exponent_sexp)[0];

  SEXP target_acceptance_sexp = PROTECT(coerceVector(target_acceptance_, REALSXP));
  protected_count++;
  double target_acceptance = REAL(target_acceptance_sexp)[0];

  /* Validate adaptation parameters */
  if (lag_update <= 0) {
    UNPROTECT(protected_count);
    error("lag_update must be positive, got %d", lag_update);
  }
  if (max_step_size <= 0 || !isfinite(max_step_size)) {
    UNPROTECT(protected_count);
    error("max_step_size must be positive and finite, got %f", max_step_size);
  }
  if (target_acceptance <= 0 || target_acceptance >= 1) {
    UNPROTECT(protected_count);
    error("target_acceptance must be in (0, 1), got %f", target_acceptance);
  }

  /* Parse diagnostic output options */
  SEXP return_log_sigma_sexp = PROTECT(coerceVector(return_log_sigma_, LGLSXP));
  protected_count++;
  int return_log_sigma = LOGICAL(return_log_sigma_sexp)[0];

  SEXP return_accept_prop_sexp = PROTECT(coerceVector(return_accept_prop_, LGLSXP));
  protected_count++;
  int return_accept_prop = LOGICAL(return_accept_prop_sexp)[0];

  /* Allocate storage for posterior samples */
  SEXP theta_1_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
  protected_count++;
  SEXP theta_01_samples = PROTECT(allocVector(REALSXP, n_chain));
  protected_count++;
  SEXP prec_1_samples = PROTECT(allocVector(REALSXP, n_chain));
  protected_count++;
  SEXP alpha_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
  protected_count++;

  /* Conditional allocation for diagnostics */
  SEXP log_sigma_samples = R_NilValue;
  SEXP accept_prop_samples = R_NilValue;
  int n_outputs = 4;  // Base outputs: theta_1, theta_01, prec_1, alpha

  if (return_log_sigma) {
    log_sigma_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
    protected_count++;
    n_outputs++;
  }
  if (return_accept_prop) {
    accept_prop_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
    protected_count++;
    n_outputs++;
  }

  /* MCMC history arrays */
  double *theta_1_post = (double*) R_Calloc(n_iter * n, double);
  double *theta_01_post = (double*) R_Calloc(n_iter, double);
  double *prec_1_post = (double*) R_Calloc(n_iter, double);
  double *alpha_post = (double*) R_Calloc(n_iter * n, double);
  double *theta_1_updated = (double*) R_Calloc(n_iter * n, double);

  /* Working arrays for CWMH algorithm */
  double *accept_prop = (double*) R_Calloc(n, double);
  double *log_sigma = (double*) R_Calloc(n, double);
  double *hat_theta_1 = (double*) R_Calloc(n, double);
  double *theta_1_new = (double*) R_Calloc(n, double);
  double *log_accept_prob = (double*) R_Calloc(n, double);

  /* Initialize log_sigma with reasonable starting values */
  for (int j = 0; j < n; j++) {
    log_sigma[j] = log(DEFAULT_INIT_SD);
  }

  GetRNGstate();

  /* Initialization (iter = 0), respecting fixed values */
  if (fix_theta_01) {
    theta_01_post[0] = theta_01_true;
  } else {
    theta_01_post[0] = rnorm(prior_theta01_mean, 1.0/sqrt(prior_theta01_prec));
    /* Truncate to avoid extreme values in logit scale */
    theta_01_post[0] = fmax2(-LOGIT_BOUND_LIMIT, fmin2(LOGIT_BOUND_LIMIT, theta_01_post[0]));
  }

  if (fix_prec_1) {
    prec_1_post[0] = prec_1_true;
  } else {
    prec_1_post[0] = rgamma(prior_prec1_shape, 1.0/prior_prec1_rate);
    /* Ensure minimum precision for numerical stability */
    prec_1_post[0] = fmax2(prec_1_post[0], MIN_PRECISION_THRESHOLD);
  }

  if (fix_theta_1) {
    memcpy(theta_1_post, theta_1_true, n * sizeof(double));
  } else {
    double init_sd = sqrt(1.0 / prec_1_post[0]);
    theta_1_post[0] = rnorm(theta_01_post[0], init_sd);
    for (int j = 1; j < n; j++) {
      theta_1_post[j] = rnorm(theta_1_post[j - 1], init_sd);
    }
  }

  /* Initialize alpha (success probabilities) */
  for (int j = 0; j < n; j++) {
    alpha_post[j] = ilogit(theta_1_post[j]);
  }

  /* Calculate min_deviation_threshold based on lag_update */
  double min_deviation_threshold = MIN_DEVIATION_THRESHOLD_FACTOR / (double)lag_update;

  /* Main MCMC loop */
  int chain_idx = 0;
  for (int ii = 1; ii < n_iter; ii++) {
    /* Check for user interruption periodically */
    if (ii % 1000 == 0) {
      R_CheckUserInterrupt();
    }

    /* Step 1: Sample theta_1 and alpha */
    if (fix_theta_1) {
      memcpy(theta_1_post + ii * n, theta_1_true, n * sizeof(double));
      /* Update alpha even if theta_1 is fixed */
      for (int j = 0; j < n; j++) {
        alpha_post[ii * n + j] = ilogit(theta_1_true[j]);
      }
    } else {
      generate_alpha_logit_binomial_locallevel(
        theta_1_post, theta_01_post, theta_1_updated, alpha_post, prec_1_post, y_ptr,
        accept_prop, log_sigma, hat_theta_1, theta_1_new, log_accept_prob,
        lag_update, n_trials, n, ii, max_step_size, base_adaptation_rate,
        decay_exponent, target_acceptance, min_deviation_threshold);
    }

    /* Step 2: Sample prec_1 */
    if (fix_prec_1) {
      prec_1_post[ii] = prec_1_true;
    } else {
      generate_precision_theta_p(theta_01_post, theta_1_post, prec_1_post,
                                 prior_prec1_shape, prior_prec1_rate, n, ii);

      /* Enhanced sanity check for precision with better error handling */
      if (prec_1_post[ii] <= 0 || !isfinite(prec_1_post[ii])) {
        warning("Invalid precision value %f at iteration %d, using previous value %f",
                prec_1_post[ii], ii, prec_1_post[ii-1]);
        prec_1_post[ii] = prec_1_post[ii-1];
      }
      /* Ensure minimum precision threshold */
      prec_1_post[ii] = fmax2(prec_1_post[ii], MIN_PRECISION_THRESHOLD);
    }

    /* Step 3: Sample theta_01 */
    if (fix_theta_01) {
      theta_01_post[ii] = theta_01_true;
    } else {
      generate_theta_01_locallevel(theta_01_post, theta_1_post, prec_1_post,
                                   prior_theta01_mean, prior_theta01_prec, n, ii);
    }

    /* Store samples after burn-in with thinning */
    if (ii >= burnin && ((ii - burnin) % thinning) == 0) {
      for (int j = 0; j < n; j++) {
        REAL(theta_1_samples)[chain_idx + j * n_chain] = theta_1_post[ii * n + j];
        REAL(alpha_samples)[chain_idx + j * n_chain] = alpha_post[ii * n + j];

        /* Store diagnostics if requested */
        if (return_log_sigma) {
          REAL(log_sigma_samples)[chain_idx + j * n_chain] = log_sigma[j];
        }
        if (return_accept_prop) {
          REAL(accept_prop_samples)[chain_idx + j * n_chain] = accept_prop[j];
        }
      }
      REAL(theta_01_samples)[chain_idx] = theta_01_post[ii];
      REAL(prec_1_samples)[chain_idx] = prec_1_post[ii];
      chain_idx++;
    }
  }

  PutRNGstate();

  /* Free memory */
  R_Free(theta_1_post); R_Free(theta_01_post); R_Free(prec_1_post);
  R_Free(alpha_post); R_Free(theta_1_updated); R_Free(accept_prop);
  R_Free(log_sigma); R_Free(hat_theta_1); R_Free(theta_1_new);
  R_Free(log_accept_prob);

  /* Package results into a named list */
  SEXP result_list = PROTECT(allocVector(VECSXP, n_outputs));
  protected_count++;
  SEXP names = PROTECT(allocVector(STRSXP, n_outputs));
  protected_count++;

  int output_idx = 0;

  /* Always include base outputs */
  SET_VECTOR_ELT(result_list, output_idx, theta_1_samples);
  SET_STRING_ELT(names, output_idx++, mkChar("theta_1"));

  SET_VECTOR_ELT(result_list, output_idx, theta_01_samples);
  SET_STRING_ELT(names, output_idx++, mkChar("theta_01"));

  SET_VECTOR_ELT(result_list, output_idx, prec_1_samples);
  SET_STRING_ELT(names, output_idx++, mkChar("prec_1"));

  SET_VECTOR_ELT(result_list, output_idx, alpha_samples);
  SET_STRING_ELT(names, output_idx++, mkChar("alpha"));

  /* Conditionally add diagnostic outputs */
  if (return_log_sigma) {
    SET_VECTOR_ELT(result_list, output_idx, log_sigma_samples);
    SET_STRING_ELT(names, output_idx++, mkChar("log_sigma"));
  }
  if (return_accept_prop) {
    SET_VECTOR_ELT(result_list, output_idx, accept_prop_samples);
    SET_STRING_ELT(names, output_idx++, mkChar("accept_prop"));
  }

  setAttrib(result_list, R_NamesSymbol, names);

  /* Simplified UNPROTECT: release all protected objects at once */
  UNPROTECT(protected_count);

  return result_list;
}

/**
 * @brief Test wrapper for probit Bernoulli local-level MCMC with selective parameter fixing
 *
 * @details Enables validation of MCMC algorithm correctness by fixing specific parameters
 *          to known true values while sampling others. Uses Albert-Chib data augmentation
 *          for efficient Gibbs sampling in Bernoulli models with probit link.
 *
 *          Model specification:
 *          y_t ~ Bernoulli(alpha_t), alpha_t = Phi(theta_{t,1})
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1}, u_{t,1} ~ N(0, 1/prec_1)
 *
 * @param y_                   SEXP Numeric vector of Bernoulli observations [0,1] (length n)
 * @param burnin_              SEXP Integer scalar, number of burn-in iterations
 * @param thinning_            SEXP Integer scalar, thinning interval
 * @param n_chain_             SEXP Integer scalar, number of retained samples
 * @param theta_1_true_        SEXP Numeric matrix [n_chain x n] or NULL (fix theta_1 if provided)
 * @param theta_01_true_       SEXP Numeric scalar or NULL (fix theta_01 if provided)
 * @param prec_1_true_         SEXP Numeric scalar or NULL (fix prec_1 if provided)
 * @param prior_theta01_mean_  SEXP Double scalar, prior mean for theta_01
 * @param prior_theta01_prec_  SEXP Double scalar, prior precision for theta_01
 * @param prior_prec1_shape_   SEXP Double scalar, prior shape for prec_1
 * @param prior_prec1_rate_    SEXP Double scalar, prior rate for prec_1
 *
 * @return Named list with MCMC samples:
 *         - theta_1: Matrix [n_chain x n] of level state samples
 *         - theta_01: Vector [n_chain] of initial state samples
 *         - prec_1: Vector [n_chain] of precision samples
 *         - alpha: Matrix [n_chain x n] of success probability samples
 *
 * @note Uses Gibbs sampling (100% acceptance rate) via Albert-Chib augmentation
 * @note Supports flexible parameter fixing by checking for NULL values
 * @note Validates Bernoulli constraints: y[i] ∈ {0,1}
 * @note Enforces minimum sample size n >= 3 for numerical stability
 *
 * @see generate_alpha_probit_bernoulli_locallevel
 * @see C_MCMC_probit_bernoulli_locallevel
 * @since version 1.3
 */
SEXP test_mcmc_probit_bernoulli_locallevel_fixed_params(SEXP y_, SEXP burnin_, SEXP thinning_, SEXP n_chain_,
                                                        SEXP theta_1_true_, SEXP theta_01_true_, SEXP prec_1_true_,
                                                        SEXP prior_theta01_mean_, SEXP prior_theta01_prec_,
                                                        SEXP prior_prec1_shape_, SEXP prior_prec1_rate_) {

  // Numerical constants for validation and initialization
  const int MIN_SAMPLE_SIZE = 3;
  const double MIN_PRECISION_THRESHOLD = 1e-6;
  const double DEFAULT_INIT_SD = 0.1;

  /* Parse data vector and validate */
  SEXP y = PROTECT(coerceVector(y_, REALSXP));
  int protected_count = 1;
  int n = LENGTH(y);
  if (n < MIN_SAMPLE_SIZE) {
    UNPROTECT(protected_count);
    error("Sample size 'n' must be at least %d for numerical stability, got %d", MIN_SAMPLE_SIZE, n);
  }
  double *y_ptr = REAL(y);

  /* Validate Bernoulli constraints */
  for (int i = 0; i < n; i++) {
    if (y_ptr[i] != 0.0 && y_ptr[i] != 1.0) {
      UNPROTECT(protected_count);
      error("Invalid Bernoulli observation y[%d] = %f: must be exactly 0 or 1", i + 1, y_ptr[i]);
    }
  }

  /* Parse MCMC parameters */
  int burnin = asInteger(burnin_);
  int thinning = asInteger(thinning_);
  int n_chain = asInteger(n_chain_);

  if (burnin < 0 || thinning < 1 || n_chain < 1) {
    UNPROTECT(protected_count);
    error("Invalid MCMC parameters: burnin >= 0, thinning >= 1, n_chain >= 1");
  }

  int n_iter = burnin + (n_chain - 1) * thinning + 1;

  /* Parse prior parameters */
  double prior_theta01_mean = asReal(prior_theta01_mean_);
  double prior_theta01_prec = asReal(prior_theta01_prec_);
  double prior_prec1_shape = asReal(prior_prec1_shape_);
  double prior_prec1_rate = asReal(prior_prec1_rate_);

  if (prior_theta01_prec <= 0 || prior_prec1_shape <= 0 || prior_prec1_rate <= 0) {
    UNPROTECT(protected_count);
    error("Prior precision parameters must be positive");
  }

  /* Check which parameters to fix */
  bool fix_theta_1 = !isNull(theta_1_true_);
  bool fix_theta_01 = !isNull(theta_01_true_);
  bool fix_prec_1 = !isNull(prec_1_true_);

  /* Allocate memory for MCMC storage */
  double *theta_1_samples = R_Calloc(n_iter * n, double);
  double *theta_01_samples = R_Calloc(n_iter, double);
  double *prec_1_samples = R_Calloc(n_iter, double);
  double *alpha_samples = R_Calloc(n_iter * n, double);

  /* Working arrays for Gibbs sampling */
  double *v_latent = R_Calloc(n, double);
  double *rhs_vector = R_Calloc(n, double);

  /* Initialize or fix parameters */
  if (fix_theta_1) {
    double *theta_1_true = REAL(theta_1_true_);
    for (int i = 0; i < n_iter * n; i++) {
      theta_1_samples[i] = theta_1_true[i % n];  // Replicate if needed
    }
  } else {
    // Initialize with small random values
    GetRNGstate();
    for (int i = 0; i < n_iter * n; i++) {
      theta_1_samples[i] = rnorm(0.0, DEFAULT_INIT_SD);
    }
    PutRNGstate();
  }

  if (fix_theta_01) {
    double theta_01_true = asReal(theta_01_true_);
    for (int i = 0; i < n_iter; i++) {
      theta_01_samples[i] = theta_01_true;
    }
  } else {
    // Initialize from prior
    GetRNGstate();
    for (int i = 0; i < n_iter; i++) {
      theta_01_samples[i] = rnorm(prior_theta01_mean, sqrt(1.0 / prior_theta01_prec));
    }
    PutRNGstate();
  }

  if (fix_prec_1) {
    double prec_1_true = asReal(prec_1_true_);
    for (int i = 0; i < n_iter; i++) {
      prec_1_samples[i] = prec_1_true;
    }
  } else {
    // Initialize from prior
    GetRNGstate();
    for (int i = 0; i < n_iter; i++) {
      prec_1_samples[i] = rgamma(prior_prec1_shape, 1.0 / prior_prec1_rate);
    }
    PutRNGstate();
  }

  /* Run MCMC iterations */
  GetRNGstate();

  for (int iter = 1; iter < n_iter; iter++) {

    // Sample theta_1 (unless fixed)
    if (!fix_theta_1) {
      generate_alpha_probit_bernoulli_locallevel(
        theta_1_samples, theta_01_samples, alpha_samples, prec_1_samples,
        y_ptr, v_latent, rhs_vector, n, iter
      );
    } else {
      // Still need to compute alpha for fixed theta_1
      int current_pos = iter * n;
      for (int t = 0; t < n; t++) {
        alpha_samples[current_pos + t] = pnorm(theta_1_samples[current_pos + t], 0.0, 1.0, 1, 0);
      }
    }

    // Sample theta_01 (unless fixed)
    if (!fix_theta_01) {
      generate_theta_01_locallevel(theta_1_samples, theta_01_samples, prec_1_samples,
                                   prior_theta01_mean, prior_theta01_prec, n, iter);
    }

    // Sample prec_1 (unless fixed)
    if (!fix_prec_1) {
      generate_precision_theta_p(theta_1_samples, theta_01_samples, prec_1_samples,
                                 prior_prec1_shape, prior_prec1_rate, n, iter);
    }
  }

  PutRNGstate();

  /* Create output matrices and vectors */
  SEXP theta_1_out = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_out = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_1_out = PROTECT(allocVector(REALSXP, n_chain));
  SEXP alpha_out = PROTECT(allocMatrix(REALSXP, n_chain, n));
  protected_count += 4;

  /* Extract thinned samples */
  for (int chain_idx = 0; chain_idx < n_chain; chain_idx++) {
    int iter_idx = burnin + chain_idx * thinning;

    // Extract theta_1 row
    for (int t = 0; t < n; t++) {
      REAL(theta_1_out)[chain_idx + t * n_chain] = theta_1_samples[iter_idx * n + t];
      REAL(alpha_out)[chain_idx + t * n_chain] = alpha_samples[iter_idx * n + t];
    }

    // Extract scalar parameters
    REAL(theta_01_out)[chain_idx] = theta_01_samples[iter_idx];
    REAL(prec_1_out)[chain_idx] = prec_1_samples[iter_idx];
  }

  /* Create output list */
  SEXP result = PROTECT(allocVector(VECSXP, 4));
  SEXP names = PROTECT(allocVector(STRSXP, 4));
  protected_count += 2;

  SET_VECTOR_ELT(result, 0, theta_1_out);
  SET_VECTOR_ELT(result, 1, theta_01_out);
  SET_VECTOR_ELT(result, 2, prec_1_out);
  SET_VECTOR_ELT(result, 3, alpha_out);

  SET_STRING_ELT(names, 0, mkChar("theta_1"));
  SET_STRING_ELT(names, 1, mkChar("theta_01"));
  SET_STRING_ELT(names, 2, mkChar("prec_1"));
  SET_STRING_ELT(names, 3, mkChar("alpha"));

  setAttrib(result, R_NamesSymbol, names);

  /* Clean up memory */
  R_Free(theta_1_samples);
  R_Free(theta_01_samples);
  R_Free(prec_1_samples);
  R_Free(alpha_samples);
  R_Free(v_latent);
  R_Free(rhs_vector);

  UNPROTECT(protected_count);
  return result;
}
