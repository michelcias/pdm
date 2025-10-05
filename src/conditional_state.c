/**
 * @file conditional_state.c
 * @brief Implementation of conditional posterior sampling for state parameters
 * @details Provides efficient Bayesian updating algorithms for state vector parameters
 *          in polynomial dynamic models, utilizing multivariate Normal conjugacy for
 *          fast sampling within Gibbs MCMC iterations. Implements tridiagonal precision
 *          matrix structures for computational efficiency.
 * @author Michel H. Montoril
 * @date 2025-08-12
 * @version 1.0
 */

#include <R.h>
#include <Rmath.h>
#include "utils.h"
#include "conditional_state.h"

/**
 * @brief Generates sample from conditional posterior of state vector theta_1
 *        (local level model)
 *
 * @details Implements Bayesian updating for the level state vector in local-level
 *          dynamic models. The conditional posterior is multivariate Normal with
 *          tridiagonal precision matrix, derived from combining Normal likelihood
 *          with state evolution equations.
 *
 *          Model structure:
 *          y_t = theta_{t,1} + e_t,                    e_t ~ N(0, V)
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *          theta_{1,1} = theta_{0,1} + u_{1,1}
 *
 *          Conditional posterior:
 *          theta_1 | y, precisions, theta_01 ~ MVN(mu_post, Phi_post^{-1})
 *          where Phi_post is tridiagonal with diagonal elements (prec_y + 2*prec_1)
 *          and off-diagonal elements (-prec_1)
 *
 * @param data                Double array of observed data [n]. Input observations
 * @param theta_1_post        Double array storing posterior samples of theta_1
 *                            (vectorized B*n matrix). Updated at current iteration `iter`
 * @param prec_data_post      Double array storing posterior samples of data precision 1/V.
 *                            Uses previous iteration (`iter - 1`)
 * @param prec_theta_1_post   Double array storing posterior samples of state precision 1/W_1.
 *                            Uses previous iteration (`iter - 1`)
 * @param theta_01_post       Double array storing posterior samples of initial level theta_01.
 *                            Uses previous iteration (`iter - 1`)
 * @param n                   Integer scalar, sample size (number of observations). Must be > 2
 * @param iter                Integer scalar, current MCMC iteration index (0-based)
 *
 * @note Computational complexity: O(n) for tridiagonal system solving
 * @note Numerical stability: Uses specialized tridiagonal solver in generate_normal_vector
 * @note Memory access: Sequential access with vectorized matrix indexing
 * @note Algorithm: Multivariate Normal sampling with tridiagonal precision matrix
 * @note Boundary condition: add_a = 1 for last diagonal element structure
 *
 * @warning Assumes n > 2 for proper tridiagonal structure
 * @warning Assumes iter > 0 for accessing previous iteration
 * @warning No validation of data or precision parameter positivity
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function
 *
 * @see generate_normal_vector
 * @see generate_theta_1
 * @see generate_precision_data
 */
void generate_theta_1_locallevel(double *data,
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

  // Compute the posterior mean vector: mu_post
  // Based on E[theta_1 | y, precisions, theta_01] from model equations:
  // Data contribution: y[j] * prec_y for all elements
  // Initial condition: theta_01 * prec_1 for first element only
  for (int j = 0; j < n; j++) {
    mean_theta_1[j] = data[j] * prec_y;
  }
  // The first element also includes the initial condition component
  mean_theta_1[0] += theta_01_post[iter - 1] * prec_1; // Uses theta_01 from previous iter

  // Sample theta_1 from conditional posterior MVN(mu_post, Phi_post^{-1})
  // Tridiagonal precision matrix structure:
  // a = prec_y (data likelihood contribution)
  // b = prec_1 (state evolution contribution)
  // Diag: prec_y + 2*prec_1 (except last)
  // Off-diag: -prec_1
  // Last diag: prec_y + prec_1
  // add_a = 1 (adjusts last diagonal element to prec_y + prec_1)
  generate_normal_vector(
    theta_1_post, /* theta_post: destination buffer for theta_1 samples */
    mean_theta_1, /* mean_theta: conditional posterior mean vector */
    prec_y,       /* prec_y: observation precision contribution */
    prec_1,       /* prec_1: innovation precision contribution */
    n,            /* n: number of time points */
    iter,         /* iter: current Gibbs iteration */
    1             /* add_a: adjust last diagonal element */
  );
}


//----------------------------------------------------------------------

/**
 * @brief Generates sample from conditional posterior of state vector theta_1
 *        (local trend model)
 *
 * @details Implements Bayesian updating for the level state vector in local-trend
 *          dynamic models. The conditional posterior is multivariate Normal with
 *          tridiagonal precision matrix, incorporating information from trend component
 *          theta_2 in the state evolution equations.
 *
 *          Model structure:
 *          y_t = theta_{t,1} + e_t,                                    e_t ~ N(0, V)
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                  u_{t,2} ~ N(0, W_2)
 *          theta_{1,1} = theta_{0,1} + theta_{0,2} + u_{1,1}
 *          theta_{1,2} = theta_{0,2} + u_{1,2}
 *
 *          Conditional posterior:
 *          theta_1 | y, theta_2, precisions, theta_01, theta_02 ~ MVN(mu_post, Phi_post^{-1})
 *          where mu_post incorporates trend adjustments from theta_2
 *
 * @param data                Double array of observed data [n]. Input observations
 * @param theta_1_post        Double array storing posterior samples of theta_1
 *                            (vectorized B*n matrix). Updated at current iteration `iter`
 * @param theta_2_post        Double array storing posterior samples of theta_2
 *                            (vectorized B*n matrix). Uses current iteration `iter`
 * @param prec_data_post      Double array storing posterior samples of data precision 1/V.
 *                            Uses previous iteration (`iter - 1`)
 * @param prec_theta_1_post   Double array storing posterior samples of state precision 1/W_1.
 *                            Uses previous iteration (`iter - 1`)
 * @param theta_01_post       Double array storing posterior samples of initial level theta_01.
 *                            Uses previous iteration (`iter - 1`)
 * @param theta_02_post       Double array storing posterior samples of initial trend theta_02.
 *                            Uses current iteration `iter`
 * @param n                   Integer scalar, sample size (number of observations). Must be > 2
 * @param iter                Integer scalar, current MCMC iteration index (0-based)
 *
 * @note Computational complexity: O(n) for mean vector computation and tridiagonal solving
 * @note Numerical stability: Uses specialized tridiagonal solver in generate_normal_vector
 * @note Memory access: Sequential access with proper iteration offsets for theta_2
 * @note Algorithm: Multivariate Normal sampling with trend-adjusted mean vector
 * @note Boundary condition: add_a = 1 for last diagonal element structure
 *
 * @warning Assumes n > 2 for proper tridiagonal structure
 * @warning Assumes iter > 0 for accessing previous iteration
 * @warning No validation of data or precision parameter positivity
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function
 *
 * @see generate_normal_vector
 * @see generate_theta_1_locallevel
 * @see generate_theta_2
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

  // Compute the posterior mean vector: mu_post with trend adjustments
  // First element: incorporates initial conditions and trend correction
  mean_theta_1[0] = data[0] * prec_y +
                    (theta_01_post[iter - 1] + theta_02_post[iter] -
                    theta_2_post[iter_n]) * prec_1;

  // Middle elements: incorporate trend differences between consecutive time points
  for (int j = 1; j < n - 1; j++) {
    mean_theta_1[j] = data[j] * prec_y +
                      (theta_2_post[iter_n + j - 1] - theta_2_post[iter_n + j]) * prec_1;
  }

  // Last element: incorporates final trend component
  mean_theta_1[n - 1] = data[n - 1] * prec_y +
                        theta_2_post[iter_n + n - 2] * prec_1;

  // Sample theta_1 from conditional posterior MVN(mu_post, Phi_post^{-1})
  // Tridiagonal precision matrix structure:
  // a = prec_y (data likelihood contribution)
  // b = prec_1 (state evolution contribution)
  // Diag: prec_y + 2*prec_1 (except last)
  // Off-diag: -prec_1
  // Last diag: prec_y + prec_1
  // add_a = 1 (adjusts last diagonal element to prec_y + prec_1)
  generate_normal_vector(
    theta_1_post, /* theta_post: destination buffer for theta_1 samples */
    mean_theta_1, /* mean_theta: conditional posterior mean vector */
    prec_y,       /* prec_y: observation precision contribution */
    prec_1,       /* prec_1: innovation precision contribution */
    n,            /* n: number of time points */
    iter,         /* iter: current Gibbs iteration */
    1             /* add_a: adjust last diagonal element */
  );
}


//----------------------------------------------------------------------

/**
 * @brief Generates sample from conditional posterior of intermediate state vector theta_k
 *        (k=2,...,p-1)
 *
 * @details Implements Bayesian updating for intermediate state vectors in polynomial
 *          dynamic models. The conditional posterior is multivariate Normal with
 *          tridiagonal precision matrix, incorporating information from adjacent
 *          polynomial components theta_{k-1} and theta_{k+1}.
 *
 *          Model structure:
 *          theta_{t,k-1} = theta_{t-1,k-1} + theta_{t-1,k} + u_{t,k-1}
 *          theta_{t,k} = theta_{t-1,k} + theta_{t-1,k+1} + u_{t,k}
 *          theta_{1,k} = theta_{0,k} + theta_{0,k+1} + u_{1,k}
 *
 *          Conditional posterior:
 *          theta_k | theta_{k-1}, theta_{k+1}, precisions, theta_{0k}, theta_{0,k+1} ~
 *          MVN(mu_post, Phi_post^{-1})
 *
 * @param theta_km1_post      Double array storing posterior samples of theta_{k-1}
 *                            (vectorized B*n matrix). Uses previous iteration (`iter - 1`)
 * @param theta_k_post        Double array storing posterior samples of theta_k
 *                            (vectorized B*n matrix). Updated at current iteration `iter`
 * @param theta_kp1_post      Double array storing posterior samples of theta_{k+1}
 *                            (vectorized B*n matrix). Uses current iteration `iter`
 * @param prec_theta_km1_post Double array storing posterior samples of precision 1/W_{k-1}.
 *                            Uses previous iteration (`iter - 1`)
 * @param prec_theta_k_post   Double array storing posterior samples of precision 1/W_k.
 *                            Uses previous iteration (`iter - 1`)
 * @param theta_0k_post       Double array storing posterior samples of initial state theta_{0k}.
 *                            Uses previous iteration (`iter - 1`)
 * @param theta_0kp1_post     Double array storing posterior samples of initial state theta_{0,k+1}.
 *                            Uses current iteration `iter`
 * @param n                   Integer scalar, sample size (number of observations). Must be > 2
 * @param iter                Integer scalar, current MCMC iteration index (0-based)
 *
 * @note Computational complexity: O(n) for mean vector computation and tridiagonal solving
 * @note Numerical stability: Uses specialized tridiagonal solver in generate_normal_vector
 * @note Memory access: Mixed iteration offsets for proper Gibbs sampling dependencies
 * @note Algorithm: Multivariate Normal sampling with polynomial component interactions
 * @note Boundary condition: add_a = 0 for last diagonal element structure
 * @note Notation: km1 = k-1, kp1 = k+1 for parameter naming
 *
 * @warning Assumes n > 2 for proper tridiagonal structure
 * @warning Assumes iter > 0 for accessing previous iteration
 * @warning No validation of precision parameter positivity
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function
 *
 * @see generate_normal_vector
 * @see generate_theta_1
 * @see generate_theta_p
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

  // Compute the posterior mean vector: mu_post with polynomial interactions
  // First element: incorporates initial conditions and adjacent component effects
  mean_theta_k[0] = (theta_km1_post[iterm1_n + 1] - theta_km1_post[iterm1_n]) * prec_km1 +
                    (theta_0k_post[iter - 1] + theta_0kp1_post[iter] -
                    theta_kp1_post[iter_n]) * prec_k;

  // Middle elements: incorporate differences from adjacent polynomial components
  for (int j = 1; j < n - 1; j++) {
    mean_theta_k[j] = (theta_km1_post[iterm1_n + j + 1] - theta_km1_post[iterm1_n + j]) * prec_km1 +
                      (theta_kp1_post[iter_n + j - 1] - theta_kp1_post[iter_n + j]) * prec_k;
  }

  // Last element: only higher-order component contribution
  mean_theta_k[n - 1] = theta_kp1_post[iter_n + n - 2] * prec_k;

  // Sample theta_k from conditional posterior MVN(mu_post, Phi_post^{-1})
  // Tridiagonal precision matrix structure:
  // a = prec_km1 (lower-order component contribution)
  // b = prec_k (higher-order component contribution)
  // Diag: prec_km1 + 2*prec_k (except last)
  // Off-diag: -prec_k
  // Last diag: prec_k
  // add_a = 0 (last diagonal element is prec_k only)
  generate_normal_vector(
    theta_k_post, /* theta_post: destination buffer for theta_k samples */
    mean_theta_k, /* mean_theta: conditional posterior mean vector */
    prec_km1,     /* prec_lower: precision from component k-1 */
    prec_k,       /* prec_upper: precision from component k+1 */
    n,            /* n: number of time points */
    iter,         /* iter: current Gibbs iteration */
    0             /* add_a: boundary adjustment flag */
  );
}


//----------------------------------------------------------------------

/**
 * @brief Generates sample from conditional posterior of final state vector theta_p
 *        (k=p)
 *
 * @details Implements Bayesian updating for the highest-order state vector in polynomial
 *          dynamic models. This is a boundary case handling the final polynomial component
 *          following a random walk structure, influenced only by the previous component
 *          theta_{p-1}.
 *          Notation: pm1 = p-1
 *
 *          Model structure:
 *          theta_{t,p-1} = theta_{t-1,p-1} + theta_{t-1,p} + u_{t,p-1}
 *          theta_{t,p} = theta_{t-1,p} + u_{t,p},     u_{t,p} ~ N(0, W_p)
 *          theta_{1,p} = theta_{0,p} + u_{1,p}
 *
 *          Conditional posterior:
 *          theta_p | theta_{p-1}, precisions, theta_{0p} ~ MVN(mu_post, Phi_post^{-1})
 *
 * @param theta_pm1_post      Double array storing posterior samples of theta_{p-1}
 *                            (vectorized B*n matrix). Uses previous iteration (`iter - 1`)
 * @param theta_p_post        Double array storing posterior samples of theta_p
 *                            (vectorized B*n matrix). Updated at current iteration `iter`
 * @param prec_theta_pm1_post Double array storing posterior samples of precision 1/W_{p-1}.
 *                            Uses previous iteration (`iter - 1`)
 * @param prec_theta_p_post   Double array storing posterior samples of precision 1/W_p.
 *                            Uses previous iteration (`iter - 1`)
 * @param theta_0p_post       Double array storing posterior samples of initial state theta_{0p}.
 *                            Uses previous iteration (`iter - 1`)
 * @param n                   Integer scalar, sample size (number of observations). Must be > 2
 * @param iter                Integer scalar, current MCMC iteration index (0-based)
 *
 * @note Computational complexity: O(n) for mean vector computation and tridiagonal solving
 * @note Numerical stability: Uses specialized tridiagonal solver in generate_normal_vector
 * @note Memory access: Sequential access with previous iteration offsets
 * @note Algorithm: Multivariate Normal sampling for random walk boundary case
 * @note Boundary condition: add_a = 0 for last diagonal element structure
 * @note Notation: pm1 = p-1 for parameter naming
 *
 * @warning Assumes n > 2 for proper tridiagonal structure
 * @warning Assumes iter > 0 for accessing previous iteration
 * @warning No validation of precision parameter positivity
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function
 *
 * @see generate_normal_vector
 * @see generate_theta_k
 * @see generate_precision_theta_p
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

  // Compute the posterior mean vector: mu_post for boundary case
  // Elements j=0 to n-2: incorporate differences from previous component theta_{p-1}
  for (int j = 0; j < n - 1; j++) {
    mean_theta_p[j] = (theta_pm1_post[iterm1_n + j + 1] - theta_pm1_post[iterm1_n + j]) * prec_pm1;
  }

  // Last element: no contribution from differences (boundary case)
  mean_theta_p[n - 1] = 0;

  // Add initial condition component for the first element
  mean_theta_p[0] += theta_0p_post[iter - 1] * prec_p; // Uses theta_0p from iter-1

  // Sample theta_p from conditional posterior MVN(mu_post, Phi_post^{-1})
  // Tridiagonal precision matrix structure:
  // a = prec_pm1 (previous component contribution)
  // b = prec_p (random walk innovation contribution)
  // Diag: prec_pm1 + 2*prec_p (except last)
  // Off-diag: -prec_p
  // Last diag: prec_p
  // add_a = 0 (last diagonal element is prec_p only)
  generate_normal_vector(
    theta_p_post, /* theta_post: destination buffer for theta_p samples */
    mean_theta_p, /* mean_theta: conditional posterior mean vector */
    prec_pm1,     /* prec_lower: precision from component p-1 */
    prec_p,       /* prec_upper: innovation precision for theta_p */
    n,            /* n: number of time points */
    iter,         /* iter: current Gibbs iteration */
    0             /* add_a: boundary adjustment flag */
  );
}
