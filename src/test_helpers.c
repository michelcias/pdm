/**
 * @file test_helpers.c
 * @brief C wrappers for testing internal C functions from R
 * @details This file contains wrapper functions that expose internal C
 *          functions to R's .Call interface, specifically for the purpose of
 *          unit testing with packages like 'testthat'. Each wrapper handles
 *          proper memory management, input validation, and R object protection.
 * @author Michel H. Montoril
 * @date 2025-09-28
 * @version 1.4
 *
 * @changelog
 * - v1.4 (2025-09-28): Fixed function signatures to match updated
 *   generate_alpha_logit_binomial functions with min_deviation_threshold parameter.
 *   Corrected CWMH function calls to use proper lag_update parameter.
 * - v1.4 (2025-09-23): Updated test_adapt_cwmh_parameters to include
 *   min_deviation_threshold parameter for optimized adaptive MCMC testing.
 *   Added test_adapt_cwmh_parameters_legacy for backward compatibility
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
  generate_normal_vector(result_ptr, y, a, b, n, 0, add_a);

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
  adapt_cwmh_parameters(theta_updated, accept_prop_out, log_sigma_out,
                        lag_update, n, iter, max_step_size,
                        base_adaptation_rate, decay_exponent, target_acceptance,
                        min_deviation_threshold);

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
 * @brief Test wrapper for the legacy adapt_cwmh_parameters function for comparison testing
 *
 * @details Exposes the legacy C function 'adapt_cwmh_parameters_legacy' to R for
 *          backward compatibility testing and performance comparison. This wrapper
 *          provides the same interface as the original function, using a practical
 *          default threshold of 1.0/lag_update internally. Essential for validating
 *          that optimizations maintain equivalent behavior and for benchmarking
 *          performance improvements.
 *
 *          **Testing applications:**
 *          - Comparison of results between optimized and legacy versions
 *          - Performance benchmarking to quantify optimization benefits
 *          - Regression testing to ensure backward compatibility
 *          - Validation of threshold behavior with practical defaults
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
 * @return A named list (VECSXP) with two elements: 'accept_prop' and 'log_sigma'.
 *
 * @note Computational complexity: O(n * lag_update) with standard implementation
 * @note Memory access: Standard allocation patterns without optimization
 * @note Algorithm: Uses practical default threshold of 1.0/lag_update
 * @note Testing focus: Provides baseline for performance and correctness comparison
 *
 * @warning This is a legacy function intended primarily for testing
 * @warning Performance may be significantly slower than optimized version
 * @warning Future versions may deprecate this function
 *
 * @see adapt_cwmh_parameters_legacy
 * @see test_adapt_cwmh_parameters
 * @since version 1.4
 */
SEXP test_adapt_cwmh_parameters_legacy(SEXP theta_updated_, SEXP log_sigma_,
                                       SEXP lag_update_, SEXP n_, SEXP iter_,
                                       SEXP max_step_size_, SEXP base_adaptation_rate_,
                                       SEXP decay_exponent_, SEXP target_acceptance_) {

  // Protect and coerce inputs (9 parameters)
  PROTECT(theta_updated_ = coerceVector(theta_updated_, REALSXP));
  PROTECT(log_sigma_ = coerceVector(log_sigma_, REALSXP));
  PROTECT(lag_update_ = coerceVector(lag_update_, INTSXP));
  PROTECT(n_ = coerceVector(n_, INTSXP));
  PROTECT(iter_ = coerceVector(iter_, INTSXP));
  PROTECT(max_step_size_ = coerceVector(max_step_size_, REALSXP));
  PROTECT(base_adaptation_rate_ = coerceVector(base_adaptation_rate_, REALSXP));
  PROTECT(decay_exponent_ = coerceVector(decay_exponent_, REALSXP));
  PROTECT(target_acceptance_ = coerceVector(target_acceptance_, REALSXP));

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

  // Call the legacy C function (uses practical default threshold internally)
  adapt_cwmh_parameters_legacy(theta_updated, accept_prop_out, log_sigma_out,
                               lag_update, n, iter, max_step_size,
                               base_adaptation_rate, decay_exponent, target_acceptance);

  // Set names for the list elements
  SEXP names = PROTECT(allocVector(STRSXP, 2));
  SET_STRING_ELT(names, 0, mkChar("accept_prop"));
  SET_STRING_ELT(names, 1, mkChar("log_sigma"));
  setAttrib(result_list, R_NamesSymbol, names);

  // Populate the list with the results
  SET_VECTOR_ELT(result_list, 0, accept_prop_sexp);
  SET_VECTOR_ELT(result_list, 1, log_sigma_sexp);

  // Unprotect all SEXPs (9 inputs + 1 list + 2 vectors + 1 names = 13)
  UNPROTECT(13);

  return result_list;
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
  double *y = REAL(coerceVector(y_, REALSXP));
  int n = LENGTH(y_);

  // Simulate the MCMC history array required by the C function
  double *theta_1_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_1_post + n, REAL(coerceVector(theta_1_, REALSXP)), n * sizeof(double));

  double nu_y = REAL(coerceVector(nu_y_, REALSXP))[0];
  double eta_y = REAL(coerceVector(eta_y_, REALSXP))[0];

  double *prec_y_post = (double *) R_alloc(2, sizeof(double));

  GetRNGstate();
  // Call the function for iter = 1 (the second position)
  generate_precision_data(y, theta_1_post, prec_y_post, nu_y, eta_y, n, 1);
  PutRNGstate();

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
  int n = LENGTH(coerceVector(theta_k_, REALSXP));

  // Simulate MCMC history arrays
  double *theta_0k_post = (double *) R_alloc(2, sizeof(double));
  theta_0k_post[0] = REAL(coerceVector(theta_0k_, REALSXP))[0];

  double *theta_0kp1_post = (double *) R_alloc(2, sizeof(double));
  theta_0kp1_post[1] = REAL(coerceVector(theta_0kp1_, REALSXP))[0];

  double *theta_k_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_k_post + n, REAL(coerceVector(theta_k_, REALSXP)), n * sizeof(double));

  double *theta_kp1_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_kp1_post + n, REAL(coerceVector(theta_kp1_, REALSXP)), n * sizeof(double));

  double nu_0k = REAL(coerceVector(nu_0k_, REALSXP))[0];
  double eta_0k = REAL(coerceVector(eta_0k_, REALSXP))[0];

  double *prec_theta_k_post = (double *) R_alloc(2, sizeof(double));

  GetRNGstate();
  // Call for iter = 1
  generate_precision_theta_k(theta_0k_post, theta_0kp1_post, theta_k_post, theta_kp1_post,
                             prec_theta_k_post, nu_0k, eta_0k, n, 1);
  PutRNGstate();

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
  int n = LENGTH(coerceVector(theta_p_, REALSXP));

  // Simulate MCMC history arrays
  double *theta_0p_post = (double *) R_alloc(2, sizeof(double));
  theta_0p_post[0] = REAL(coerceVector(theta_0p_, REALSXP))[0];

  double *theta_p_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_p_post + n, REAL(coerceVector(theta_p_, REALSXP)), n * sizeof(double));

  double nu_0p = REAL(coerceVector(nu_0p_, REALSXP))[0];
  double eta_0p = REAL(coerceVector(eta_0p_, REALSXP))[0];

  double *prec_theta_p_post = (double *) R_alloc(2, sizeof(double));

  GetRNGstate();
  // Call for iter = 1
  generate_precision_theta_p(theta_0p_post, theta_p_post, prec_theta_p_post, nu_0p, eta_0p, n, 1);
  PutRNGstate();

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
  double *data = REAL(coerceVector(data_, REALSXP));
  int n = LENGTH(data_);
  double prec_data = REAL(coerceVector(prec_data_, REALSXP))[0];
  double prec_theta_1 = REAL(coerceVector(prec_theta_1_, REALSXP))[0];
  double theta_01 = REAL(coerceVector(theta_01_, REALSXP))[0];

  double *theta_1_post = (double *) R_alloc(2 * n, sizeof(double));
  double *prec_data_post = (double *) R_alloc(2, sizeof(double));
  double *prec_theta_1_post = (double *) R_alloc(2, sizeof(double));
  double *theta_01_post = (double *) R_alloc(2, sizeof(double));

  prec_data_post[0] = prec_data;
  prec_theta_1_post[0] = prec_theta_1;
  theta_01_post[0] = theta_01;

  GetRNGstate();
  generate_theta_1_locallevel(data, theta_1_post, prec_data_post, prec_theta_1_post, theta_01_post, n, 1);
  PutRNGstate();

  SEXP result_sexp = PROTECT(allocVector(REALSXP, n));
  memcpy(REAL(result_sexp), theta_1_post + n, n * sizeof(double));

  UNPROTECT(1);
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
  double *data = REAL(coerceVector(data_, REALSXP));
  int n = LENGTH(data_);

  double *theta_2 = REAL(coerceVector(theta_2_, REALSXP));
  double prec_data = REAL(coerceVector(prec_data_, REALSXP))[0];
  double prec_theta_1 = REAL(coerceVector(prec_theta_1_, REALSXP))[0];
  double theta_01 = REAL(coerceVector(theta_01_, REALSXP))[0];
  double theta_02 = REAL(coerceVector(theta_02_, REALSXP))[0];

  double *theta_1_post = (double *) R_alloc(2 * n, sizeof(double));
  double *theta_2_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_2_post + n, theta_2, n * sizeof(double));

  double *prec_data_post = (double *) R_alloc(2, sizeof(double));
  prec_data_post[0] = prec_data;

  double *prec_theta_1_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_1_post[0] = prec_theta_1;

  double *theta_01_post = (double *) R_alloc(2, sizeof(double));
  theta_01_post[0] = theta_01;

  double *theta_02_post = (double *) R_alloc(2, sizeof(double));
  theta_02_post[1] = theta_02;

  GetRNGstate();
  generate_theta_1(data, theta_1_post, theta_2_post, prec_data_post, prec_theta_1_post, theta_01_post, theta_02_post, n, 1);
  PutRNGstate();

  SEXP result_sexp = PROTECT(allocVector(REALSXP, n));
  memcpy(REAL(result_sexp), theta_1_post + n, n * sizeof(double));

  UNPROTECT(1);
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
  int n = LENGTH(coerceVector(theta_km1_, REALSXP));

  double *theta_km1 = REAL(coerceVector(theta_km1_, REALSXP));
  double *theta_kp1 = REAL(coerceVector(theta_kp1_, REALSXP));
  double prec_km1 = REAL(coerceVector(prec_km1_, REALSXP))[0];
  double prec_k = REAL(coerceVector(prec_k_, REALSXP))[0];
  double theta_0k = REAL(coerceVector(theta_0k_, REALSXP))[0];
  double theta_0kp1 = REAL(coerceVector(theta_0kp1_, REALSXP))[0];

  double *theta_km1_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_km1_post, theta_km1, n * sizeof(double));

  double *theta_k_post = (double *) R_alloc(2 * n, sizeof(double));

  double *theta_kp1_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_kp1_post + n, theta_kp1, n * sizeof(double));

  double *prec_theta_km1_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_km1_post[0] = prec_km1;

  double *prec_theta_k_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_k_post[0] = prec_k;

  double *theta_0k_post = (double *) R_alloc(2, sizeof(double));
  theta_0k_post[0] = theta_0k;

  double *theta_0kp1_post = (double *) R_alloc(2, sizeof(double));
  theta_0kp1_post[1] = theta_0kp1;

  GetRNGstate();
  generate_theta_k(theta_km1_post, theta_k_post, theta_kp1_post, prec_theta_km1_post, prec_theta_k_post, theta_0k_post, theta_0kp1_post, n, 1);
  PutRNGstate();

  SEXP result_sexp = PROTECT(allocVector(REALSXP, n));
  memcpy(REAL(result_sexp), theta_k_post + n, n * sizeof(double));

  UNPROTECT(1);
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
  int n = LENGTH(coerceVector(theta_pm1_, REALSXP));

  double *theta_pm1 = REAL(coerceVector(theta_pm1_, REALSXP));
  double prec_pm1 = REAL(coerceVector(prec_pm1_, REALSXP))[0];
  double prec_p = REAL(coerceVector(prec_p_, REALSXP))[0];
  double theta_0p = REAL(coerceVector(theta_0p_, REALSXP))[0];

  double *theta_pm1_post = (double *) R_alloc(2 * n, sizeof(double));
  memcpy(theta_pm1_post, theta_pm1, n * sizeof(double));

  double *theta_p_post = (double *) R_alloc(2 * n, sizeof(double));

  double *prec_theta_pm1_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_pm1_post[0] = prec_pm1;

  double *prec_theta_p_post = (double *) R_alloc(2, sizeof(double));
  prec_theta_p_post[0] = prec_p;

  double *theta_0p_post = (double *) R_alloc(2, sizeof(double));
  theta_0p_post[0] = theta_0p;

  GetRNGstate();
  generate_theta_p(theta_pm1_post, theta_p_post, prec_theta_pm1_post, prec_theta_p_post, theta_0p_post, n, 1);
  PutRNGstate();

  SEXP result_sexp = PROTECT(allocVector(REALSXP, n));
  memcpy(REAL(result_sexp), theta_p_post + n, n * sizeof(double));

  UNPROTECT(1);
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
  int n = LENGTH(coerceVector(theta_1_, REALSXP));
  double *theta_1 = REAL(coerceVector(theta_1_, REALSXP));
  double prec_theta_1 = REAL(coerceVector(prec_theta_1_, REALSXP))[0];
  double mean_theta_01 = REAL(coerceVector(mean_theta_01_, REALSXP))[0];
  double prec_theta_01 = REAL(coerceVector(prec_theta_01_, REALSXP))[0];

  double *theta_1_post = (double*) R_alloc(2 * n, sizeof(double));
  memcpy(theta_1_post + n, theta_1, n * sizeof(double));

  double *prec_theta_1_post = (double*) R_alloc(2, sizeof(double));
  prec_theta_1_post[1] = prec_theta_1;

  double *theta_01_post = (double*) R_alloc(2, sizeof(double));

  GetRNGstate();
  generate_theta_01_locallevel(theta_01_post, theta_1_post, prec_theta_1_post, mean_theta_01, prec_theta_01, n, 1);
  PutRNGstate();

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
  int n = LENGTH(coerceVector(theta_1_, REALSXP));
  double *theta_1 = REAL(coerceVector(theta_1_, REALSXP));
  double theta_02 = REAL(coerceVector(theta_02_, REALSXP))[0];
  double prec_theta_1 = REAL(coerceVector(prec_theta_1_, REALSXP))[0];
  double mean_theta_01 = REAL(coerceVector(mean_theta_01_, REALSXP))[0];
  double prec_theta_01 = REAL(coerceVector(prec_theta_01_, REALSXP))[0];

  double *theta_1_post = (double*) R_alloc(2 * n, sizeof(double));
  memcpy(theta_1_post + n, theta_1, n * sizeof(double));

  double *theta_02_post = (double*) R_alloc(2, sizeof(double));
  theta_02_post[1] = theta_02;

  double *prec_theta_1_post = (double*) R_alloc(2, sizeof(double));
  prec_theta_1_post[1] = prec_theta_1;

  double *theta_01_post = (double*) R_alloc(2, sizeof(double));

  GetRNGstate();
  generate_theta_01(theta_01_post, theta_02_post, theta_1_post, prec_theta_1_post, mean_theta_01, prec_theta_01, n, 1);
  PutRNGstate();

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
  int n = LENGTH(coerceVector(theta_k_, REALSXP));

  double *theta_km1_post = (double*) R_alloc(2*n, sizeof(double));
  memcpy(theta_km1_post, REAL(coerceVector(theta_km1_, REALSXP)), n * sizeof(double));

  double *theta_k_post = (double*) R_alloc(2*n, sizeof(double));
  memcpy(theta_k_post + n, REAL(coerceVector(theta_k_, REALSXP)), n * sizeof(double));

  double *theta_0km1_post = (double*) R_alloc(2, sizeof(double));
  theta_0km1_post[0] = REAL(coerceVector(theta_0km1_, REALSXP))[0];

  double *theta_0kp1_post = (double*) R_alloc(2, sizeof(double));
  theta_0kp1_post[1] = REAL(coerceVector(theta_0kp1_, REALSXP))[0];

  double *prec_theta_km1_post = (double*) R_alloc(2, sizeof(double));
  prec_theta_km1_post[0] = REAL(coerceVector(prec_km1_, REALSXP))[0];

  double *prec_theta_k_post = (double*) R_alloc(2, sizeof(double));
  prec_theta_k_post[1] = REAL(coerceVector(prec_k_, REALSXP))[0];

  double mean_0k = REAL(coerceVector(mean_0k_, REALSXP))[0];
  double prec_0k = REAL(coerceVector(prec_0k_, REALSXP))[0];

  double *theta_0k_post = (double*) R_alloc(2, sizeof(double));

  GetRNGstate();
  generate_theta_0k(theta_0km1_post, theta_0k_post, theta_0kp1_post, theta_km1_post, theta_k_post,
                    prec_theta_km1_post, prec_theta_k_post, mean_0k, prec_0k, n, 1);
  PutRNGstate();

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
  int n = LENGTH(coerceVector(theta_p_, REALSXP));

  double *theta_pm1_post = (double*) R_alloc(2 * n, sizeof(double));
  memcpy(theta_pm1_post, REAL(coerceVector(theta_pm1_, REALSXP)), n * sizeof(double));

  double *theta_p_post = (double*) R_alloc(2 * n, sizeof(double));
  memcpy(theta_p_post + n, REAL(coerceVector(theta_p_, REALSXP)), n * sizeof(double));

  double *theta_0pm1_post = (double*) R_alloc(2, sizeof(double));
  theta_0pm1_post[0] = REAL(coerceVector(theta_0pm1_, REALSXP))[0];

  double *prec_theta_pm1_post = (double*) R_alloc(2, sizeof(double));
  prec_theta_pm1_post[0] = REAL(coerceVector(prec_pm1_, REALSXP))[0];

  double *prec_theta_p_post = (double*) R_alloc(2, sizeof(double));
  prec_theta_p_post[1] = REAL(coerceVector(prec_p_, REALSXP))[0];

  double mean_0p = REAL(coerceVector(mean_0p_, REALSXP))[0];
  double prec_0p = REAL(coerceVector(prec_0p_, REALSXP))[0];

  double *theta_0p_post = (double*) R_alloc(2, sizeof(double));

  GetRNGstate();
  generate_theta_0p(theta_0pm1_post, theta_0p_post, theta_pm1_post, theta_p_post,
                    prec_theta_pm1_post, prec_theta_p_post, mean_0p, prec_0p, n, 1);
  PutRNGstate();

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
 * @param theta_1_in_ SEXP: Input first-order state vector
 * @param theta_01_in_ SEXP: Input initial state value
 * @param prec_1_in_ SEXP: Input state precision parameter
 * @param y_ SEXP: Binomial count observations
 * @param n_trials_ SEXP: Number of trials for binomial model
 * @param log_sigma_in_ SEXP: Input log proposal standard deviations
 * @return Named list with theta_1, alpha, and updated components
 *
 * @note Computational complexity: O(n) for component-wise MH updates
 * @note Memory access: Manages MCMC state arrays and working memory
 * @note Algorithm: Adaptive CWMH with logit transform for binomial likelihood
 *
 * @warning Uses external log_sigma input for adaptation loop closure
 * @warning Requires proper initialization of proposal variances
 *
 * @see cwmh_alpha_logit_binomial_locallevel
 * @since version 1.3
 */
SEXP test_cwmh_alpha_logit_binomial_locallevel(SEXP theta_1_in_, SEXP theta_01_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_, SEXP log_sigma_in_) {
  int n = LENGTH(coerceVector(y_, REALSXP));
  double n_trials = REAL(coerceVector(n_trials_, REALSXP))[0];

  // MCMC history arrays
  double *theta_1_post = (double*) R_alloc(2 * n, sizeof(double));
  memcpy(theta_1_post, REAL(coerceVector(theta_1_in_, REALSXP)), n * sizeof(double));

  double *theta_01_post = (double*) R_alloc(2, sizeof(double));
  theta_01_post[0] = REAL(coerceVector(theta_01_in_, REALSXP))[0];

  double *prec_1_post = (double*) R_alloc(2, sizeof(double));
  prec_1_post[0] = REAL(coerceVector(prec_1_in_, REALSXP))[0];

  double *y = REAL(coerceVector(y_, REALSXP));

  double *theta_1_updated = (double*) R_alloc(2 * n, sizeof(double));
  double *alpha_post = (double*) R_alloc(2 * n, sizeof(double));

  // CWMH working arrays
  double *log_sigma = REAL(coerceVector(log_sigma_in_, REALSXP)); // Use log_sigma from R
  double *hat_theta_1 = (double*) R_alloc(n, sizeof(double));
  double *theta_1_new = (double*) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double*) R_alloc(n, sizeof(double));

  GetRNGstate();
  cwmh_alpha_logit_binomial_locallevel(
    theta_1_post, theta_01_post, theta_1_updated, alpha_post, prec_1_post, y,
    log_sigma, hat_theta_1, theta_1_new, log_accept_prob,
    2, n_trials, n, 1  // lag_update = 2
  );
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 3));
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));
  SEXP updated_out = PROTECT(allocVector(LGLSXP, n));

  memcpy(REAL(theta_1_out), theta_1_post + n, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_post + n, n * sizeof(double));
  for(int i = 0; i < n; i++) LOGICAL(updated_out)[i] = updated[i];

  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);
  SET_VECTOR_ELT(res, 2, updated_out);

  SEXP nms = PROTECT(allocVector(STRSXP, 3));
  SET_STRING_ELT(nms, 0, mkChar("theta_1"));
  SET_STRING_ELT(nms, 1, mkChar("alpha"));
  SET_STRING_ELT(nms, 2, mkChar("updated"));
  setAttrib(res, R_NamesSymbol, nms);

  UNPROTECT(5);
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
 * @param theta_1_in_ SEXP: Input first-order state vector
 * @param theta_2_in_ SEXP: Input second-order state vector
 * @param theta_01_in_ SEXP: Input initial first-order state value
 * @param theta_02_in_ SEXP: Input initial second-order state value
 * @param prec_1_in_ SEXP: Input state precision parameter
 * @param y_ SEXP: Binomial count observations
 * @param n_trials_ SEXP: Number of trials for binomial model
 * @return Named list with theta_1, alpha, and updated components
 *
 * @note Computational complexity: O(n) for component-wise MH in trend model
 * @note Memory access: Coordinates multiple state vectors and parameters
 * @note Algorithm: CWMH for polynomial state space with binomial observations
 *
 * @warning Initializes log_sigma internally with default values
 * @warning Requires consistent state vector dimensions
 *
 * @see cwmh_alpha_logit_binomial
 * @since version 1.0
 */
SEXP test_cwmh_alpha_logit_binomial(SEXP theta_1_in_, SEXP theta_2_in_, SEXP theta_01_in_, SEXP theta_02_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_) {
  int n = LENGTH(coerceVector(y_, REALSXP));
  double n_trials = REAL(coerceVector(n_trials_, REALSXP))[0];

  // MCMC history arrays
  double *theta_1_post = (double*) R_alloc(2 * n, sizeof(double));
  memcpy(theta_1_post, REAL(coerceVector(theta_1_in_, REALSXP)), n * sizeof(double));

  double *theta_2_post = (double*) R_alloc(2 * n, sizeof(double));
  memcpy(theta_2_post + n, REAL(coerceVector(theta_2_in_, REALSXP)), n * sizeof(double));

  double *theta_01_post = (double*) R_alloc(2, sizeof(double));
  theta_01_post[0] = REAL(coerceVector(theta_01_in_, REALSXP))[0];

  double *theta_02_post = (double*) R_alloc(2, sizeof(double));
  theta_02_post[0] = REAL(coerceVector(theta_02_in_, REALSXP))[0];

  double *prec_1_post = (double*) R_alloc(2, sizeof(double));
  prec_1_post[0] = REAL(coerceVector(prec_1_in_, REALSXP))[0];

  double *y = REAL(coerceVector(y_, REALSXP));

  double *theta_1_updated = (double*) R_alloc(2 * n, sizeof(double));
  double *alpha_post = (double*) R_alloc(2 * n, sizeof(double));

  // CWMH working arrays
  double *log_sigma = (double*) R_alloc(n, sizeof(double));
  for(int i = 0; i < n; i++) log_sigma[i] = log(0.1); // Initialize
  double *hat_theta_1 = (double*) R_alloc(n, sizeof(double));
  double *theta_1_new = (double*) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double*) R_alloc(n, sizeof(double));
  int *updated = (int*) R_alloc(n, sizeof(int));

  GetRNGstate();
  cwmh_alpha_logit_binomial(
    theta_1_post, theta_2_post, theta_01_post, theta_02_post,
    theta_1_updated, alpha_post, prec_1_post, y,
    log_sigma, hat_theta_1, theta_1_new, log_accept_prob, updated,
    n_trials, n, 1
  );
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 3));
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));
  SEXP updated_out = PROTECT(allocVector(LGLSXP, n));

  memcpy(REAL(theta_1_out), theta_1_post + n, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_post + n, n * sizeof(double));
  for(int i = 0; i < n; i++) LOGICAL(updated_out)[i] = updated[i];

  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);
  SET_VECTOR_ELT(res, 2, updated_out);

  SEXP nms = PROTECT(allocVector(STRSXP, 3));
  SET_STRING_ELT(nms, 0, mkChar("theta_1"));
  SET_STRING_ELT(nms, 1, mkChar("alpha"));
  SET_STRING_ELT(nms, 2, mkChar("updated"));
  setAttrib(res, R_NamesSymbol, nms);

  UNPROTECT(5);
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
 * @param theta_1_in_ SEXP: Input first-order state vector
 * @param theta_01_in_ SEXP: Input initial state value
 * @param prec_1_in_ SEXP: Input state precision parameter
 * @param y_ SEXP: Binomial count observations
 * @param n_trials_ SEXP: Number of trials for binomial model
 * @return Named list with theta_1, alpha, and updated components
 *
 * @note Computational complexity: O(n) for adaptive MCMC with tuning
 * @note Memory access: Manages adaptive arrays and working memory
 * @note Algorithm: Adaptive CWMH with automatic proposal variance tuning
 *
 * @warning Uses fixed adaptation parameters for testing
 * @warning Initializes proposal variances with default values
 *
 * @see generate_alpha_logit_binomial_locallevel
 * @since version 1.0
 */
SEXP test_generate_alpha_logit_binomial_locallevel(SEXP theta_1_in_, SEXP theta_01_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_) {
  int n = LENGTH(coerceVector(y_, REALSXP));
  double n_trials = REAL(coerceVector(n_trials_, REALSXP))[0];

  // MCMC history arrays
  double *theta_1_post = (double*) R_alloc(2 * n, sizeof(double));
  memcpy(theta_1_post, REAL(coerceVector(theta_1_in_, REALSXP)), n * sizeof(double));

  double *theta_01_post = (double*) R_alloc(2, sizeof(double));
  theta_01_post[0] = REAL(coerceVector(theta_01_in_, REALSXP))[0];

  double *prec_1_post = (double*) R_alloc(2, sizeof(double));
  prec_1_post[0] = REAL(coerceVector(prec_1_in_, REALSXP))[0];

  double *y = REAL(coerceVector(y_, REALSXP));

  double *theta_1_updated = (double*) R_alloc(2 * n, sizeof(double));
  double *alpha_post = (double*) R_alloc(2 * n, sizeof(double));

  // CWMH working arrays
  double *accept_prop = (double*) R_alloc(n, sizeof(double));
  double *log_sigma = (double*) R_alloc(n, sizeof(double));
  for(int i = 0; i < n; i++) log_sigma[i] = log(0.1); // Initialize
  double *hat_theta_1 = (double*) R_alloc(n, sizeof(double));
  double *theta_1_new = (double*) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double*) R_alloc(n, sizeof(double));
  int *updated = (int*) R_alloc(n, sizeof(int));

  GetRNGstate();
  generate_alpha_logit_binomial_locallevel(
    theta_1_post, theta_01_post, theta_1_updated, alpha_post, prec_1_post, y,
    accept_prop, log_sigma, hat_theta_1, theta_1_new, log_accept_prob,
    0, n_trials, n, 1, 0.1, 1.0, 0.5, 0.44, 0.02  // min_deviation_threshold
  );
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 3));
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));
  SEXP updated_out = PROTECT(allocVector(LGLSXP, n));

  memcpy(REAL(theta_1_out), theta_1_post + n, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_post + n, n * sizeof(double));
  for(int i = 0; i < n; i++) LOGICAL(updated_out)[i] = updated[i];

  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);
  SET_VECTOR_ELT(res, 2, updated_out);

  SEXP nms = PROTECT(allocVector(STRSXP, 3));
  SET_STRING_ELT(nms, 0, mkChar("theta_1"));
  SET_STRING_ELT(nms, 1, mkChar("alpha"));
  SET_STRING_ELT(nms, 2, mkChar("updated"));
  setAttrib(res, R_NamesSymbol, nms);

  UNPROTECT(5);
  return res;
}

/**
 * @brief Test wrapper for generate_alpha_logit_binomial from generate_alpha_binomial.c
 *
 * @details Enables testing of adaptive alpha generation in binomial local trend models.
 *          This function validates the adaptive CWMH workflow for a trend model,
 *          including proposal tuning and component-wise updates.
 *
 * @param theta_1_in_ SEXP: Input first-order state vector
 * @param theta_2_in_ SEXP: Input second-order state vector
 * @param theta_01_in_ SEXP: Input initial first-order state value
 * @param theta_02_in_ SEXP: Input initial second-order state value
 * @param prec_1_in_ SEXP: Input state precision parameter
 * @param y_ SEXP: Binomial count observations
 * @param n_trials_ SEXP: Number of trials for binomial model
 * @return Named list with theta_1, alpha, log_sigma, accept_prop, and updated components
 *
 * @note Computational complexity: O(n) for adaptive MCMC with tuning
 * @note Memory access: Manages adaptive arrays and working memory
 * @note Algorithm: Adaptive CWMH with automatic proposal variance tuning (trend)
 *
 * @warning Uses fixed adaptation parameters for testing
 * @warning Initializes proposal variances with default values
 *
 * @see generate_alpha_logit_binomial
 * @since version 1.0
 */
SEXP test_generate_alpha_logit_binomial(SEXP theta_1_in_, SEXP theta_2_in_, SEXP theta_01_in_, SEXP theta_02_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_) {
  int n = LENGTH(coerceVector(y_, REALSXP));
  double n_trials = REAL(coerceVector(n_trials_, REALSXP))[0];

  // MCMC history arrays
  double *theta_1_post = (double*) R_alloc(2 * n, sizeof(double));
  memcpy(theta_1_post, REAL(coerceVector(theta_1_in_, REALSXP)), n * sizeof(double));

  double *theta_2_post = (double*) R_alloc(2 * n, sizeof(double));
  memcpy(theta_2_post + n, REAL(coerceVector(theta_2_in_, REALSXP)), n * sizeof(double));

  double *theta_01_post = (double*) R_alloc(2, sizeof(double));
  theta_01_post[0] = REAL(coerceVector(theta_01_in_, REALSXP))[0];

  double *theta_02_post = (double*) R_alloc(2, sizeof(double));
  theta_02_post[0] = REAL(coerceVector(theta_02_in_, REALSXP))[0];

  double *prec_1_post = (double*) R_alloc(2, sizeof(double));
  prec_1_post[0] = REAL(coerceVector(prec_1_in_, REALSXP))[0];

  double *y = REAL(coerceVector(y_, REALSXP));

  double *theta_1_updated = (double*) R_alloc(2 * n, sizeof(double));
  double *alpha_post = (double*) R_alloc(2 * n, sizeof(double));

  // CWMH working arrays
  double *accept_prop = (double*) R_alloc(n, sizeof(double));
  double *log_sigma = (double*) R_alloc(n, sizeof(double));
  for(int i = 0; i < n; i++) log_sigma[i] = log(0.1); // Initialize
  double *hat_theta_1 = (double*) R_alloc(n, sizeof(double));
  double *theta_1_new = (double*) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double*) R_alloc(n, sizeof(double));
  int *updated = (int*) R_alloc(n, sizeof(int));

  GetRNGstate();
  generate_alpha_logit_binomial(
    theta_1_post, theta_2_post, theta_01_post, theta_02_post,
    theta_1_updated, alpha_post, prec_1_post, y,
    accept_prop, log_sigma, hat_theta_1, theta_1_new, log_accept_prob,
    0, n_trials, n, 1, 0.1, 1.0, 0.5, 0.44, 0.02  // min_deviation_threshold
  );
  PutRNGstate();

  SEXP res = PROTECT(allocVector(VECSXP, 3));
  SEXP theta_1_out = PROTECT(allocVector(REALSXP, n));
  SEXP alpha_out = PROTECT(allocVector(REALSXP, n));
  SEXP updated_out = PROTECT(allocVector(LGLSXP, n));

  memcpy(REAL(theta_1_out), theta_1_post + n, n * sizeof(double));
  memcpy(REAL(alpha_out), alpha_post + n, n * sizeof(double));
  for(int i = 0; i < n; i++) LOGICAL(updated_out)[i] = updated[i];

  SET_VECTOR_ELT(res, 0, theta_1_out);
  SET_VECTOR_ELT(res, 1, alpha_out);
  SET_VECTOR_ELT(res, 2, updated_out);

  SEXP nms = PROTECT(allocVector(STRSXP, 3));
  SET_STRING_ELT(nms, 0, mkChar("theta_1"));
  SET_STRING_ELT(nms, 1, mkChar("alpha"));
  SET_STRING_ELT(nms, 2, mkChar("updated"));
  setAttrib(res, R_NamesSymbol, nms);

  UNPROTECT(5);
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

  /* Parse data vector and validate */
  double *y = REAL(coerceVector(y_, REALSXP));
  int n = LENGTH(y_);
  if (n < 3) {
    error("Sample size 'n' must be at least 3, got %d", n);
  }

  double n_trials = REAL(coerceVector(n_trials_, REALSXP))[0];

  /* Validate binomial constraints */
  for (int i = 0; i < n; i++) {
    if (y[i] < 0 || y[i] > n_trials) {
      error("y[%d] = %f violates 0 <= y <= n_trials = %f", i, y[i], n_trials);
    }
  }

  /* Parse MCMC settings */
  int burnin = INTEGER(coerceVector(burnin_, INTSXP))[0];
  int thinning = INTEGER(coerceVector(thinning_, INTSXP))[0];
  int n_chain = INTEGER(coerceVector(n_chain_, INTSXP))[0];
  int n_iter = burnin + (n_chain - 1) * thinning + 1;

  /* Validate MCMC parameters */
  if (burnin < 0) error("Burnin must be non-negative");
  if (thinning <= 0) error("Thinning must be positive");
  if (n_chain <= 0) error("n_chain must be positive");

  /* Parse fixed parameter flags and values */
  int fix_theta_1 = !isNull(theta_1_true_);
  int fix_theta_01 = !isNull(theta_01_true_);
  int fix_prec_1 = !isNull(prec_1_true_);

  double *theta_1_true = fix_theta_1 ? REAL(coerceVector(theta_1_true_, REALSXP)) : NULL;
  double theta_01_true = fix_theta_01 ? REAL(coerceVector(theta_01_true_, REALSXP))[0] : 0.0;
  double prec_1_true = fix_prec_1 ? REAL(coerceVector(prec_1_true_, REALSXP))[0] : 0.0;

  /* Parse priors and validate */
  double prior_theta01_mean = REAL(coerceVector(prior_theta01_mean_, REALSXP))[0];
  double prior_theta01_prec = REAL(coerceVector(prior_theta01_prec_, REALSXP))[0];
  double prior_prec1_shape = REAL(coerceVector(prior_prec1_shape_, REALSXP))[0];
  double prior_prec1_rate = REAL(coerceVector(prior_prec1_rate_, REALSXP))[0];

  if (prior_theta01_prec <= 0) error("Prior precision must be positive");
  if (prior_prec1_shape <= 0 || prior_prec1_rate <= 0) {
    error("Prior shape and rate must be positive");
  }

  /* Parse adaptation parameters */
  int lag_update = INTEGER(coerceVector(lag_update_, INTSXP))[0];
  double max_step_size = REAL(coerceVector(max_step_size_, REALSXP))[0];
  double base_adaptation_rate = REAL(coerceVector(base_adaptation_rate_, REALSXP))[0];
  double decay_exponent = REAL(coerceVector(decay_exponent_, REALSXP))[0];
  double target_acceptance = REAL(coerceVector(target_acceptance_, REALSXP))[0];

  /* Parse diagnostic output options */
  int return_log_sigma = LOGICAL(coerceVector(return_log_sigma_, LGLSXP))[0];
  int return_accept_prop = LOGICAL(coerceVector(return_accept_prop_, LGLSXP))[0];

  /* Allocate storage for posterior samples */
  SEXP theta_1_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_1_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP alpha_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));

  /* Conditional allocation for diagnostics */
  SEXP log_sigma_samples = R_NilValue;
  SEXP accept_prop_samples = R_NilValue;
  int n_outputs = 4;  // Base outputs: theta_1, theta_01, prec_1, alpha
  int n_protect = 4;  // Base protection count

  if (return_log_sigma) {
    log_sigma_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
    n_outputs++;
    n_protect++;
  }
  if (return_accept_prop) {
    accept_prop_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
    n_outputs++;
    n_protect++;
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
  int *updated = (int*) R_Calloc(n, int);

  /* Initialize log_sigma with reasonable starting values */
  for (int j = 0; j < n; j++) {
    log_sigma[j] = log(0.1);  /* Initial proposal sd = 0.1 */
  }

  GetRNGstate();

  /* Initialization (iter = 0), respecting fixed values */
  if (fix_theta_01) {
    theta_01_post[0] = theta_01_true;
  } else {
    theta_01_post[0] = rnorm(prior_theta01_mean, 1.0/sqrt(prior_theta01_prec));
    /* Truncate to avoid extreme values in logit scale */
    theta_01_post[0] = fmax(-10.0, fmin(10.0, theta_01_post[0]));
  }

  if (fix_prec_1) {
    prec_1_post[0] = prec_1_true;
  } else {
    prec_1_post[0] = rgamma(prior_prec1_shape, 1.0/prior_prec1_rate);
    /* Ensure minimum precision for numerical stability */
    prec_1_post[0] = fmax(prec_1_post[0], 1e-6);
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
        theta_1_post, theta_01_post, theta_1_updated, alpha_post, prec_1_post, y,
        accept_prop, log_sigma, hat_theta_1, theta_1_new, log_accept_prob,
        lag_update, n_trials, n, ii, max_step_size, base_adaptation_rate,
        decay_exponent, target_acceptance, 1.0/(double)lag_update);
    }

    /* Step 2: Sample prec_1 */
    if (fix_prec_1) {
      prec_1_post[ii] = prec_1_true;
    } else {
      generate_precision_theta_p(theta_01_post, theta_1_post, prec_1_post,
                                 prior_prec1_shape, prior_prec1_rate, n, ii);

      /* Sanity check for precision */
      if (prec_1_post[ii] <= 0 || !isfinite(prec_1_post[ii])) {
        warning("Invalid precision value at iteration %d, using previous value", ii);
        prec_1_post[ii] = prec_1_post[ii-1];
      }
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
  R_Free(log_accept_prob); R_Free(updated);

  /* Package results into a named list */
  SEXP result_list = PROTECT(allocVector(VECSXP, n_outputs));
  SEXP names = PROTECT(allocVector(STRSXP, n_outputs));

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

  /* Adjust UNPROTECT count: n_protect (samples) + 2 (result_list and names) */
  UNPROTECT(n_protect + 2);
  return result_list;
}
