#include <R.h>
#include <Rmath.h>
#include "conditional_state.h"

/**
 * Generates a sample for the state vector theta_1 (local trend component)
 * assuming a simple local trend model (theta_1[j] = theta_1[j-1] + error).
 * This is a step within a Gibbs sampler iteration.
 *
 * **Important Assumption:** This function assumes the dimension `n` is strictly greater than 2 (n > 2).
 * Behavior for n <= 2 is undefined as boundary case checks have been removed.
 *
 * The function calculates the mean vector and uses `generate_normal_vector`
 * to sample from the multivariate normal conditional posterior distribution of theta_1.
 * The precision matrix (Phi) of this distribution is tridiagonal, determined by
 * the data precision (prec_y) and the state innovation precision (prec_1).
 *
 * @param data                Array (size n) of observed data points, indexed 0 to n-1.
 * @param theta_1_post        Output array (vectorized B x n matrix) storing posterior samples of theta_1.
 *                            The vector for the current iteration `iter` will be updated.
 * @param prec_data_post      Array storing posterior samples of data precision (1/V). Uses value from `iter - 1`.
 * @param prec_theta_1_post   Array storing posterior samples of theta_1 innovation precision (1/W_1).
 *                            Uses value from `iter - 1`.
 * @param theta_01_post       Array storing posterior samples of the initial level theta_01. Uses value from `iter`.
 * @param n                   Sample size (number of data points/state elements, 0 to n-1). Must be > 2.
 * @param iter                Current MCMC iteration index (0-based). Assumes iter > 0.
 */
void generate_theta_1_localtrend(double *data,
                                 double *theta_1_post,
                                 double *prec_data_post,
                                 double *prec_theta_1_post,
                                 double *theta_01_post,
                                 int n,
                                 int iter) {

  // Allocate space for the mean vector of the conditional posterior
  double *mean_theta_1 = (double *)R_alloc(n, sizeof(double));

  // Get precisions from the previous iteration (iter - 1)
  double prec_y = prec_data_post[iter - 1];    // Data precision 1/V
  double prec_1 = prec_theta_1_post[iter - 1]; // State theta_1 innovation precision 1/W_1

  // Compute the mean vector (y-vector for generate_normal_vector)
  // Based on E[theta_1 | ...] derived from model equations:
  // y[j] = theta_1[j] + obs_error_j => term data[j] * prec_y
  // theta_1[j] = theta_1[j-1] + state_error_j => terms involving prec_1
  // theta_1[0] = theta_01 + state_error_0 => term involving theta_01_post[iter] * prec_1

  // All elements have the data component
  for (int j = 0; j < n; j++) {
    mean_theta_1[j] = data[j] * prec_y;
  }
  // The first element also includes the initial condition component
  mean_theta_1[0] += theta_01_post[iter] * prec_1; // Uses theta_01 from current iter

  // Generate theta_1 for current iteration 'iter' using the auxiliary function
  // The precision matrix Phi for theta_1 | ... is tridiagonal:
  // a = prec_y (from data likelihood)
  // b = prec_1 (from state evolution theta_1[j] vs theta_1[j-1])
  // Diag: prec_y + 2*prec_1 (except last)
  // Off-diag: -prec_1
  // Last diag: prec_y + prec_1
  // add_a = 1 reflects this last diagonal element structure.
  generate_normal_vector(theta_1_post, mean_theta_1, prec_y, prec_1, n, iter, 1);
}


//----------------------------------------------------------------------

/**
 * Generates a sample for the state vector theta_1 (local level component)
 * assuming a local linear trend model (theta_1 depends on theta_2).
 * This is a step within a Gibbs sampler iteration.
 *
 * **Important Assumption:** This function assumes the dimension `n` is strictly greater than 2 (n > 2).
 * Behavior for n <= 2 is undefined as boundary case checks have been removed.
 *
 * The function calculates the mean vector and uses `generate_normal_vector`
 * to sample from the multivariate normal conditional posterior distribution of theta_1.
 * The precision matrix (Phi) of this distribution is tridiagonal, determined by
 * the data precision (prec_y) and the state innovation precision (prec_1). The mean
 * vector calculation incorporates terms related to theta_2.
 *
 * @param data                Array (size n) of observed data points, indexed 0 to n-1.
 * @param theta_1_post        Output array (vectorized B x n matrix) storing posterior samples of theta_1.
 *                            The vector for the current iteration `iter` will be updated.
 * @param theta_2_post        Array storing posterior samples of theta_2 (vectorized B x n matrix).
 *                            Uses the vector corresponding to the *current* iteration `iter`.
 * @param prec_data_post      Array storing posterior samples of data precision (1/V). Uses value from `iter - 1`.
 * @param prec_theta_1_post   Array storing posterior samples of theta_1 innovation precision (1/W_1).
 *                            Uses value from `iter - 1`.
 * @param theta_01_post       Array storing posterior samples of the initial level theta_01. Uses value from `iter`.
 * @param theta_02_post       Array storing posterior samples of the initial trend theta_02. Uses value from `iter`.
 * @param n                   Sample size (number of data points/state elements, 0 to n-1). Must be > 2.
 * @param iter                Current MCMC iteration index (0-based). Assumes iter > 0.
 */
void generate_theta_1(double *data,
                      double *theta_1_post,
                      double *theta_2_post,
                      double *prec_data_post,
                      double *prec_theta_1_post,
                      double *theta_01_post,
                      double *theta_02_post,
                      int n,
                      int iter) {

  int iter_n = iter * n; // Offset for current iteration's data

  // Allocate space for the mean vector of the conditional posterior
  double *mean_theta_1 = (double *)R_alloc(n, sizeof(double));

  // Get precisions from the previous iteration (iter - 1)
  double prec_y = prec_data_post[iter - 1];    // Data precision 1/V
  double prec_1 = prec_theta_1_post[iter - 1]; // State theta_1 innovation precision 1/W_1

  // Compute the mean vector (y-vector for generate_normal_vector)
  // Based on E[theta_1 | ...] derived from model equations:
  // y[j] = theta_1[j] + obs_error_j => term data[j] * prec_y
  // theta_1[j] = theta_1[j-1] + theta_2[j-1] + state_error_j => terms involving prec_1 and theta_2
  // theta_1[0] = theta_01 + theta_02 + state_error_0 => term involving theta_01, theta_02 * prec_1

  // Calculate first element (j=0)
  mean_theta_1[0] = data[0] * prec_y
  + (theta_01_post[iter] + theta_02_post[iter] - theta_2_post[iter_n]) * prec_1;
  // Uses theta_01, theta_02 from current iter; theta_2 from current iter (iter_n offset)

  // Calculate middle elements (j=1 to n-2)
  for (int j = 1; j < n - 1; j++) {
    mean_theta_1[j] = data[j] * prec_y
    + (theta_2_post[iter_n + j - 1] - theta_2_post[iter_n + j]) * prec_1;
    // Uses theta_2 from current iter (iter_n offset)
  }

  // Calculate last element (j = n-1)
  mean_theta_1[n - 1] = data[n - 1] * prec_y
  + theta_2_post[iter_n + n - 2] * prec_1;
  // Uses theta_2 from current iter (iter_n offset)

  // Generate theta_1 for current iteration 'iter' using the auxiliary function
  // The precision matrix Phi for theta_1 | ... is tridiagonal:
  // a = prec_y (from data likelihood)
  // b = prec_1 (from state evolution theta_1[j] vs theta_1[j-1])
  // Diag: prec_y + 2*prec_1 (except last)
  // Off-diag: -prec_1
  // Last diag: prec_y + prec_1
  // add_a = 1 reflects this last diagonal element structure.
  generate_normal_vector(theta_1_post, mean_theta_1, prec_y, prec_1, n, iter, 1);
}


//----------------------------------------------------------------------

/**
 * Generates a sample for the state vector theta_k (k from 2 to p-1)
 * within a Gibbs sampler iteration.
 *
 * **Important Assumption:** This function assumes the dimension `n` is strictly greater than 2 (n > 2).
 * Behavior for n <= 2 is undefined as boundary case checks have been removed.
 *
 * This function calculates the mean vector based on the model structure (likely involving
 * theta_{k-1} and theta_{k+1}) and uses `generate_normal_vector` to sample from the
 * multivariate normal conditional posterior distribution of theta_k. The precision matrix (Phi)
 * is tridiagonal, determined by the innovation precisions prec_km1 (1/W_{k-1}) and
 * prec_k (1/W_k).
 *
 * @param theta_km1_post      Array storing posterior samples of theta_(k-1) (vectorized B x n matrix).
 *                            Uses the vector corresponding to the *previous* iteration `iter - 1`.
 * @param theta_k_post        Output array (vectorized B x n matrix) storing posterior samples of theta_k.
 *                            The vector for the current iteration `iter` will be updated.
 * @param theta_kp1_post      Array storing posterior samples of theta_(k+1) (vectorized B x n matrix).
 *                            Uses the vector corresponding to the *current* iteration `iter`.
 * @param prec_theta_km1_post Array storing posterior samples of precision 1/W_{k-1}. Uses value from `iter - 1`.
 * @param prec_theta_k_post   Array storing posterior samples of precision 1/W_k. Uses value from `iter - 1`.
 *                            (Note: Check if this should be iter or iter-1 based on Gibbs order). Assuming iter-1 based on original code.
 * @param theta_0k_post       Array storing posterior samples of the baseline theta_0k. Uses value from `iter`.
 * @param theta_0kp1_post     Array storing posterior samples of the baseline theta_0(k+1). Uses value from `iter`.
 * @param n                   Sample size (number of elements in state vectors, 0 to n-1). Must be > 2.
 * @param iter                Current MCMC iteration index (0-based). Assumes iter > 0.
 */
void generate_theta_k(double *theta_km1_post,
                      double *theta_k_post,
                      double *theta_kp1_post,
                      double *prec_theta_km1_post,
                      double *prec_theta_k_post,
                      double *theta_0k_post,
                      double *theta_0kp1_post,
                      int n,
                      int iter) {

  int iter_n = iter * n;     // Offset for current iteration's data
  int iterm1_n = iter_n - n; // Offset for previous iteration's data

  // Allocate space for the mean vector of the conditional posterior
  double *mean_theta_k = (double *)R_alloc(n, sizeof(double));

  // Get precisions from the previous iteration (iter - 1)
  double prec_km1 = prec_theta_km1_post[iter - 1]; // Precision 1/W_{k-1}
  double prec_k = prec_theta_k_post[iter - 1];     // Precision 1/W_k

  // Compute the mean vector (y-vector for generate_normal_vector)
  // Based on E[theta_k | ...] derived from model equations, likely:
  // theta_{k-1}[j] = theta_{k-1}[j-1] + theta_k[j-1] + error => terms involving prec_km1 * diff(theta_{k-1})
  // theta_k[j] = theta_k[j-1] + theta_{k+1}[j-1] + error => terms involving prec_k and theta_{k+1}
  // theta_k[0] = theta_0k + theta_0(k+1) + error => term involving theta_0k, theta_0(k+1) * prec_k

  // Calculate first element (j = 0)
  mean_theta_k[0] = (theta_km1_post[iterm1_n + 1] - theta_km1_post[iterm1_n]) * prec_km1               // Uses theta_{k-1} from iter-1
  + (theta_0k_post[iter] + theta_0kp1_post[iter] - theta_kp1_post[iter_n]) * prec_k; // Uses thetas_0 from iter; theta_{k+1} from iter

  // Calculate middle elements (j=1 to n-2)
  for (int j = 1; j < n - 1; j++) {
    mean_theta_k[j] = (theta_km1_post[iterm1_n + j + 1] - theta_km1_post[iterm1_n + j]) * prec_km1 // Uses theta_{k-1} from iter-1
    + (theta_kp1_post[iter_n + j - 1] - theta_kp1_post[iter_n + j]) * prec_k;    // Uses theta_{k+1} from iter
  }

  // Calculate last element (j=n-1)
  mean_theta_k[n - 1] = theta_kp1_post[iter_n + n - 2] * prec_k;

  // Generate theta_k for current iteration 'iter' using the auxiliary function
  // The precision matrix Phi for theta_k | ... is tridiagonal:
  // a = prec_km1
  // b = prec_k
  // Diag: prec_km1 + 2*prec_k (except last)
  // Off-diag: -prec_k
  // Last diag: prec_k
  // add_a = 0 reflects this last diagonal element structure.
  generate_normal_vector(theta_k_post, mean_theta_k, prec_km1, prec_k, n, iter, 0);
}


//----------------------------------------------------------------------

/**
 * Generates a sample for the state vector theta_p (last level, k=p)
 * within a Gibbs sampler iteration.
 *
 * **Important Assumption:** This function assumes the dimension `n` is strictly greater than 2 (n > 2).
 * Behavior for n <= 2 is undefined as boundary case checks have been removed.
 *
 * This function calculates the mean vector based on the model structure (likely involving
 * theta_{p-1}) and uses `generate_normal_vector` to sample from the multivariate
 * normal conditional posterior distribution of theta_p. The precision matrix (Phi) is
 * tridiagonal, determined by the innovation precisions prec_pm1 (1/W_{p-1}) and prec_p (1/W_p).
 *
 * @param theta_pm1_post      Array storing posterior samples of theta_(p-1) (vectorized B x n matrix).
 *                            Uses the vector corresponding to the *previous* iteration `iter - 1`.
 * @param theta_p_post        Output array (vectorized B x n matrix) storing posterior samples of theta_p.
 *                            The vector for the current iteration `iter` will be updated.
 * @param prec_theta_pm1_post Array storing posterior samples of precision 1/W_{p-1}. Uses value from `iter - 1`.
 * @param prec_theta_p_post   Array storing posterior samples of precision 1/W_p. Uses value from `iter - 1`.
 *                           (Note: Check if this should be iter or iter-1 based on Gibbs order). Assuming iter-1 based on original code.
 * @param theta_0p_post       Array storing posterior samples of the baseline theta_0p. Uses value from `iter`.
 * @param n                   Sample size (number of elements in state vectors, 0 to n-1). Must be > 2.
 * @param iter                Current MCMC iteration index (0-based). Assumes iter > 0.
 */
void generate_theta_p(double *theta_pm1_post,
                      double *theta_p_post,
                      double *prec_theta_pm1_post,
                      double *prec_theta_p_post,
                      double *theta_0p_post,
                      int n,
                      int iter) {

  int iter_n = iter * n;     // Offset for current iteration's data
  int iterm1_n = iter_n - n; // Offset for previous iteration's data

  // Allocate space for the mean vector of the conditional posterior
  double *mean_theta_p = (double *)R_alloc(n, sizeof(double));

  // Get precisions from the previous iteration (iter - 1)
  double prec_pm1 = prec_theta_pm1_post[iter - 1]; // Precision 1/W_{p-1}
  double prec_p = prec_theta_p_post[iter - 1];     // Precision 1/W_p

  // Compute the mean vector (y-vector for generate_normal_vector)
  // Based on E[theta_p | ...] derived from model equations, likely:
  // theta_{p-1}[j] = theta_{p-1}[j-1] + theta_p[j-1] + error => terms involving prec_pm1 * diff(theta_{p-1})
  // theta_p[j] = theta_p[j-1] + error' => terms involving prec_p
  // theta_p[0] = theta_0p + error'_0 => term involving theta_0p_post[iter] * prec_p

  // Calculate elements j=0 to n-2
  for (int j = 0; j < n - 1; j++) {
    mean_theta_p[j] = (theta_pm1_post[iterm1_n + j + 1] - theta_pm1_post[iterm1_n + j]) * prec_pm1; // Uses theta_{p-1} from iter-1
  }

  // Set last element (j = n-1)
  mean_theta_p[n - 1] = 0;

  // Add the initial condition component for the first element (j=0)
  mean_theta_p[0] += theta_0p_post[iter] * prec_p; // Uses theta_0p from current iter

  // Generate theta_p for current iteration 'iter' using the auxiliary function
  // The precision matrix Phi for theta_p | ... is tridiagonal:
  // a = prec_pm1
  // b = prec_p
  // Diag: prec_pm1 + 2*prec_p (except last)
  // Off-diag: -prec_p
  // Last diag: prec_p
  // add_a = 0 reflects this last diagonal element structure.
  generate_normal_vector(theta_p_post, mean_theta_p, prec_pm1, prec_p, n, iter, 0);
}
