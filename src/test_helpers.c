/**
 * @file test_helpers.c
 * @brief C wrappers for testing internal C functions from R.
 * @author Michel H. Montoril
 * @date 2025-08-30
 * @version 1.2
 *
 * @details This file contains wrapper functions that expose internal C
 * functions to R's .Call interface, specifically for the purpose of
 * unit testing with packages like 'testthat'.
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h> // Include for rnorm and rgamma
#include <string.h> // For memcpy

// Include headers for the C functions to be tested
#include "utils.h"
#include "cwmh_adaptive.h"
#include "conditional_precision.h"
#include "conditional_state.h"
#include "conditional_theta0.h"
#include "cwmh_binomial.h"
#include "generate_alpha_binomial.h"
#include "mcmc_binomial_locallevel.h" // For the fixed params test


/**
 * @brief R interface wrapper for the internal C ilogit function.
 *
 * @details This function serves as a bridge to allow the internal C `ilogit`
 * function to be called directly from R for unit testing. It takes a
 * numeric SEXP, applies the transformation to the first element, and
 * returns the result as a scalar SEXP.
 *
 * @param x_ A numeric SEXP from R. Only the first element is used.
 * @return A scalar real SEXP containing the result of the ilogit transformation.
 */
SEXP test_ilogit(SEXP x_) {
  if (!isReal(x_) || length(x_) == 0) {
    error("Input must be a non-empty numeric vector");
  }

  double val = ilogit(REAL(x_)[0]);
  return ScalarReal(val);
}

/**
 * @brief Test wrapper for the generate_normal_vector function.
 *
 * @details Allows calling the C function 'generate_normal_vector' with parameters
 * defined in R to verify its output. This is crucial for testing the
 * correctness of the multivariate normal sampling with a tridiagonal
 * precision matrix.
 *
 * @param y_ SEXP: A numeric vector for 'y' (the right-hand side of the system).
 * @param a_ SEXP: A numeric scalar for 'a'.
 * @param b_ SEXP: A numeric scalar for 'b'.
 * @param add_a_ SEXP: An integer scalar for the 'add_a' flag.
 * @return A SEXP containing the generated random vector.
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

/**
 * @brief Test wrapper for the adapt_cwmh_parameters function.
 *
 * @details Exposes the C function 'adapt_cwmh_parameters' to R for testing.
 * This wrapper takes all necessary parameters from R, calls the
 * adaptation function, and returns the updated 'accrate' and 'log_sigma'
 * vectors in a named list.
 *
 * @param theta_updated_ SEXP: A numeric vector representing the acceptance history.
 * @param log_sigma_ SEXP: A numeric vector of the current log proposal standard deviations.
 * @param lag_update_ SEXP: An integer scalar for the window size.
 * @param n_ SEXP: An integer scalar for the number of components.
 * @param iter_ SEXP: An integer scalar for the current MCMC iteration.
 * @param max_step_size_ SEXP: A numeric scalar for the maximum step size.
 * @param base_adaptation_rate_ SEXP: A numeric scalar for the base adaptation rate.
 * @param decay_exponent_ SEXP: A numeric scalar for the decay exponent.
 * @param target_acceptance_ SEXP: A numeric scalar for the target acceptance rate.
 * @return A named list (VECSXP) with two elements: 'accrate' and 'log_sigma'.
 */
SEXP test_adapt_cwmh_parameters(SEXP theta_updated_, SEXP log_sigma_,
                                SEXP lag_update_, SEXP n_, SEXP iter_,
                                SEXP max_step_size_, SEXP base_adaptation_rate_,
                                SEXP decay_exponent_, SEXP target_acceptance_) {

  // Protect and coerce inputs
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
  SEXP accrate_sexp = PROTECT(allocVector(REALSXP, n));
  SEXP log_sigma_sexp = PROTECT(allocVector(REALSXP, n));

  double *accrate_out = REAL(accrate_sexp);
  double *log_sigma_out = REAL(log_sigma_sexp);

  // Copy input log_sigma to output log_sigma, as it is modified in place
  for(int i = 0; i < n; i++) {
    log_sigma_out[i] = log_sigma_in[i];
  }

  // Call the C function
  adapt_cwmh_parameters(theta_updated, accrate_out, log_sigma_out,
                        lag_update, n, iter, max_step_size,
                        base_adaptation_rate, decay_exponent, target_acceptance);

  // Set names for the list elements
  SEXP names = PROTECT(allocVector(STRSXP, 2));
  SET_STRING_ELT(names, 0, mkChar("accrate"));
  SET_STRING_ELT(names, 1, mkChar("log_sigma"));
  setAttrib(result_list, R_NamesSymbol, names);

  // Populate the list with the results
  SET_VECTOR_ELT(result_list, 0, accrate_sexp);
  SET_VECTOR_ELT(result_list, 1, log_sigma_sexp);

  // Unprotect all SEXPs (9 inputs + 1 list + 2 vectors + 1 names = 13)
  UNPROTECT(13);

  return result_list;
}


// --- Test wrapper for generate_precision_data from conditional_precision.c ---
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

// --- Test wrapper for generate_precision_theta_k from conditional_precision.c ---
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

// --- Test wrapper for generate_precision_theta_p from conditional_precision.c ---
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

// --- Test wrapper for generate_theta_1_locallevel from conditional_state.c ---
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

// --- Test wrapper for generate_theta_1 from conditional_state.c ---
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

// --- Test wrapper for generate_theta_k from conditional_state.c ---
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

// --- Test wrapper for generate_theta_p from conditional_state.c ---
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

/* --- Wrappers for functions in conditional_theta0.c --- */
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


/* --- Wrappers for functions in cwmh_binomial.c --- */
SEXP test_CWMH_alpha_logit_binomial_locallevel(SEXP theta_1_in_, SEXP theta_01_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_) {
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
  double *log_sigma = (double*) R_alloc(n, sizeof(double));
  for(int i = 0; i < n; i++) log_sigma[i] = log(0.1); // Initialize
  double *hat_theta_1 = (double*) R_alloc(n, sizeof(double));
  double *theta_1_new = (double*) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double*) R_alloc(n, sizeof(double));
  int *updated = (int*) R_alloc(n, sizeof(int));

  GetRNGstate();
  CWMH_alpha_logit_binomial_locallevel(
    theta_1_post, theta_01_post, theta_1_updated, alpha_post, prec_1_post, y,
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

SEXP test_CWMH_alpha_logit_binomial(SEXP theta_1_in_, SEXP theta_2_in_, SEXP theta_01_in_, SEXP theta_02_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_) {
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
  CWMH_alpha_logit_binomial(
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

/* --- Wrappers for functions in generate_alpha_binomial.c --- */
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
  double *accrate = (double*) R_alloc(n, sizeof(double));
  double *log_sigma = (double*) R_alloc(n, sizeof(double));
  for(int i = 0; i < n; i++) log_sigma[i] = log(0.1); // Initialize
  double *hat_theta_1 = (double*) R_alloc(n, sizeof(double));
  double *theta_1_new = (double*) R_alloc(n, sizeof(double));
  double *log_accept_prob = (double*) R_alloc(n, sizeof(double));
  int *updated = (int*) R_alloc(n, sizeof(int));

  GetRNGstate();
  generate_alpha_logit_binomial_locallevel(
    theta_1_post, theta_01_post, theta_1_updated, alpha_post, prec_1_post, y,
    accrate, log_sigma, hat_theta_1, theta_1_new, log_accept_prob, updated,
    0, n_trials, n, 1, 0.1, 1.0, 0.5, 0.44
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
  double *accrate = (double*) R_alloc(n, sizeof(double));
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
    accrate, log_sigma, hat_theta_1, theta_1_new, log_accept_prob, updated,
    0, n_trials, n, 1, 0.1, 1.0, 0.5, 0.44
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
 * @param return_accrate_ SEXP: Logical scalar, whether to return accrate diagnostics
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
                                                SEXP return_log_sigma_, SEXP return_accrate_) {

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
  int return_accrate = LOGICAL(coerceVector(return_accrate_, LGLSXP))[0];

  /* Allocate storage for posterior samples */
  SEXP theta_1_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
  SEXP theta_01_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP prec_1_samples = PROTECT(allocVector(REALSXP, n_chain));
  SEXP alpha_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));

  /* Conditional allocation for diagnostics */
  SEXP log_sigma_samples = R_NilValue;
  SEXP accrate_samples = R_NilValue;
  int n_outputs = 4;  // Base outputs: theta_1, theta_01, prec_1, alpha
  int n_protect = 4;  // Base protection count

  if (return_log_sigma) {
    log_sigma_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
    n_outputs++;
    n_protect++;
  }
  if (return_accrate) {
    accrate_samples = PROTECT(allocMatrix(REALSXP, n_chain, n));
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
  double *accrate = (double*) R_Calloc(n, double);
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
        accrate, log_sigma, hat_theta_1, theta_1_new, log_accept_prob, updated,
        lag_update, n_trials, n, ii, max_step_size, base_adaptation_rate,
        decay_exponent, target_acceptance);
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
        if (return_accrate) {
          REAL(accrate_samples)[chain_idx + j * n_chain] = accrate[j];
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
  R_Free(alpha_post); R_Free(theta_1_updated); R_Free(accrate);
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
  if (return_accrate) {
    SET_VECTOR_ELT(result_list, output_idx, accrate_samples);
    SET_STRING_ELT(names, output_idx++, mkChar("accrate"));
  }

  setAttrib(result_list, R_NamesSymbol, names);

  /* Adjust UNPROTECT count: n_protect (samples) + 2 (result_list and names) */
  UNPROTECT(n_protect + 2);
  return result_list;
}
