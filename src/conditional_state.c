/**
 * @file conditional_state.c
 * @brief Conditional posterior sampling for state parameters in Gaussian models
 * @author Michel H. Montoril
 * @date 2025-10-11
 * @version 1.2
 *
 * @details This file contains optimized functions for Bayesian state parameter sampling
 *          in polynomial dynamic models with Gaussian observation equations:
 *          - Level state sampling (theta_1) for local level models
 *          - Level state sampling (theta_1) for local trend models
 *          - Intermediate state sampling (theta_k) for k=2,...,p-1
 *          - Final state sampling (theta_p) for highest-order component
 *          - Memory-efficient implementations using direct vector operations
 *          - Full Normal conjugacy exploitation via tridiagonal precision matrices
 *
 * @changelog
 * - v1.2 (2026-07-06): Bracketed the per-call R_alloc scratch with
 *   vmaxget()/vmaxset() in all four sampling functions. These helpers run once
 *   per Gibbs iteration, and the R_alloc stack unwinds only when the enclosing
 *   .Call returns, so the scratch previously accumulated across iterations
 *   (peak memory proportional to iterations x n). Now reclaimed every call.
 * - v1.1 (2025-10-11): Refactored all functions to use scalar parameters and direct
 *   vector operations instead of array indexing. Removed iteration index parameters.
 *   Improved memory efficiency by eliminating O(n_iter) storage requirements. Added
 *   const qualifiers for input arrays. Updated to use generate_normal_vector with
 *   simplified signature.
 * - v1.0 (2025-08-12): Initial implementation with array-based interface.
 */

#include <R.h>
#include <Rmath.h>
#include "utils.h"
#include "conditional_state.h"

/**
 * @brief Sample level state vector theta_1 from conditional posterior
 *        (local level model)
 *
 * @details Generates sample from the conditional posterior of the level state vector
 *          in local level dynamic models. The posterior is multivariate Normal with
 *          tridiagonal precision matrix, derived from combining Normal likelihood
 *          with state evolution equations.
 *
 *          **Model structure:**
 *          y_t = theta_{t,1} + e_t,                    e_t ~ N(0, V)
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *          theta_{1,1} = theta_{0,1} + u_{1,1}
 *
 *          **Conditional posterior:**
 *          theta_1 | y, precisions, theta_{0,1} ~ MVN(mu_post, Sigma_post)
 *          where Sigma_post^{-1} has tridiagonal structure:
 *          - Diagonal: prec_y + 2*prec_theta1 for t=1,...,n-1
 *          - Last diagonal: prec_y + prec_theta1
 *          - Off-diagonal: -prec_theta1
 *
 *          **Algorithm:**
 *          1. Construct mean vector from observations and initial state
 *          2. Sample from MVN with tridiagonal precision using Cholesky decomposition
 *          3. Exploit sparsity for O(n) complexity
 *
 * @param data              Observed data vector [n] (const, read-only).
 *                          Contains time series observations y_1, ..., y_n.
 * @param theta_1_current   Output buffer [n] for sampled theta_1 vector.
 *                          Will be filled with theta_{1,1}, ..., theta_{n,1}.
 * @param prec_data         Scalar observation precision 1/V (> 0).
 *                          Controls influence of observations on posterior.
 * @param prec_theta_1      Scalar innovation precision 1/W_1 (> 0).
 *                          Controls smoothness of state trajectory.
 * @param theta_01          Scalar initial state theta_{0,1}.
 *                          Starting point for random walk.
 * @param n                 Sample size (number of observations, must be > 2).
 *
 * @note Computational complexity: O(n) via tridiagonal Cholesky decomposition.
 * @note Numerical stability: Uses specialized tridiagonal solver in generate_normal_vector.
 * @note Memory access: Sequential access with O(n) temporary allocation for mean vector.
 * @note Algorithm: Multivariate Normal sampling with structured precision matrix.
 * @note Boundary condition: add_a = 1 adjusts last diagonal element to prec_y + prec_theta1.
 *
 * @warning Assumes n > 2 for proper tridiagonal structure (enforced by generate_normal_vector).
 * @warning No validation of data or precision parameter positivity.
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function.
 *
 * @see generate_normal_vector
 * @see generate_theta_1
 */
void generate_theta_1_locallevel(const double *data,
                                 double       *theta_1_current,
                                 double        prec_data,
                                 double        prec_theta_1,
                                 double        theta_01,
                                 int           n) {

  /* ========== Allocate Mean Vector ========== */
  /* Temporary storage for conditional posterior mean.
   * This vector combines information from observations and state dynamics.
   * The R_alloc stack unwinds only when the enclosing .Call returns, and this
   * function runs once per Gibbs iteration; the vmaxget()/vmaxset() bracket
   * reclaims the scratch every call so peak memory stays O(n) instead of
   * growing with the number of iterations. */
  const void *vmax = vmaxget();
  double *mean_theta_1 = (double *)R_alloc(n, sizeof(double));

  /* ========== Construct Posterior Mean Vector ========== */
  /* Data contribution: each observation contributes y_t * prec_y
   * This represents the likelihood information at each time point. */
  for (int j = 0; j < n; j++) {
    mean_theta_1[j] = data[j] * prec_data;
  }

  /* Initial condition contribution: adds theta_{0,1} * prec_theta1 to first element
   * This implements the boundary condition theta_{1,1} = theta_{0,1} + u_{1,1}
   * by incorporating prior information about the starting state. */
  mean_theta_1[0] += theta_01 * prec_theta_1;

  /* ========== Sample from Multivariate Normal Posterior ========== */
  /* Draw theta_1 ~ MVN(mu_post, Sigma_post)
   * where Sigma_post^{-1} has tridiagonal structure:
   * - a = prec_data (observation precision contribution)
   * - b = prec_theta_1 (state evolution precision contribution)
   * - Diagonal: prec_data + 2*prec_theta_1 (except last element)
   * - Off-diagonal: -prec_theta_1
   * - Last diagonal: prec_data + prec_theta_1 (add_a = 1)
   *
   * The tridiagonal structure arises from the Markovian property of the
   * random walk state evolution, allowing O(n) sampling complexity. */
  generate_normal_vector(
    theta_1_current, /* r: output buffer for theta_1 samples */
    mean_theta_1,    /* y: conditional posterior mean vector */
    prec_data,       /* a: observation precision contribution */
    prec_theta_1,    /* b: innovation precision contribution */
    n,               /* n: number of time points */
    1                /* add_a: adjust last diagonal element */
  );

  /* Release the R_alloc scratch (mean vector); results are in theta_1_current. */
  vmaxset(vmax);
}

//----------------------------------------------------------------------

/**
 * @brief Sample level state vector theta_1 from conditional posterior
 *        (local trend model)
 *
 * @details Generates sample from the conditional posterior of the level state vector
 *          in local trend dynamic models. The posterior is multivariate Normal with
 *          tridiagonal precision matrix, incorporating information from trend component
 *          theta_2 in the state evolution equations.
 *
 *          **Model structure:**
 *          y_t = theta_{t,1} + e_t,                                    e_t ~ N(0, V)
 *          theta_{t,1} = theta_{t-1,1} + theta_{t-1,2} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *          theta_{t,2} = theta_{t-1,2} + u_{t,2},                  u_{t,2} ~ N(0, W_2)
 *          theta_{1,1} = theta_{0,1} + theta_{0,2} + u_{1,1}
 *          theta_{1,2} = theta_{0,2} + u_{1,2}
 *
 *          **Conditional posterior:**
 *          theta_1 | y, theta_2, precisions, theta_{0,1}, theta_{0,2} ~ MVN(mu_post, Sigma_post)
 *          where mu_post incorporates trend adjustments from theta_2
 *
 *          **Algorithm:**
 *          1. Construct mean vector from observations and trend-adjusted dynamics
 *          2. Account for trend contributions in first, middle, and last elements
 *          3. Sample from MVN with tridiagonal precision
 *
 * @param data              Observed data vector [n] (const, read-only).
 *                          Contains time series observations y_1, ..., y_n.
 * @param theta_1_current   Output buffer [n] for sampled theta_1 vector.
 *                          Will be filled with theta_{1,1}, ..., theta_{n,1}.
 * @param theta_2_current   Current trend state vector [n] (const, read-only).
 *                          Contains theta_{1,2}, ..., theta_{n,2} from current iteration.
 * @param prec_data         Scalar observation precision 1/V (> 0).
 *                          Controls influence of observations on posterior.
 * @param prec_theta_1      Scalar innovation precision 1/W_1 (> 0).
 *                          Controls smoothness of level state trajectory.
 * @param theta_01          Scalar initial level state theta_{0,1}.
 *                          Starting point for level component.
 * @param theta_02          Scalar initial trend state theta_{0,2}.
 *                          Starting point for trend component.
 * @param n                 Sample size (number of observations, must be > 2).
 *
 * @note Computational complexity: O(n) for mean vector construction and tridiagonal solving.
 * @note Numerical stability: Uses specialized tridiagonal solver in generate_normal_vector.
 * @note Memory access: Sequential access with proper iteration offsets for theta_2.
 * @note Algorithm: Multivariate Normal sampling with trend-adjusted mean vector.
 * @note Boundary condition: add_a = 1 for last diagonal element structure.
 *
 * @warning Assumes n > 2 for proper tridiagonal structure.
 * @warning No validation of data or precision parameter positivity.
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function.
 *
 * @see generate_normal_vector
 * @see generate_theta_1_locallevel
 * @see generate_theta_2
 */
void generate_theta_1(const double *data,
                      double       *theta_1_current,
                      const double *theta_2_current,
                      double        prec_data,
                      double        prec_theta_1,
                      double        theta_01,
                      double        theta_02,
                      int           n) {

  /* ========== Allocate Mean Vector ========== */
  /* Temporary storage for conditional posterior mean.
   * This vector combines observation, trend, and initial state information.
   * vmaxget()/vmaxset() bracket: reclaim the per-call R_alloc scratch, which
   * would otherwise accumulate across Gibbs iterations until .Call returns. */
  const void *vmax = vmaxget();
  double *mean_theta_1 = (double *)R_alloc(n, sizeof(double));

  /* ========== Construct Posterior Mean Vector ========== */
  /* First element (t=1): incorporates initial conditions and trend correction
   * Structure: y_1 * prec_y + (theta_{0,1} + theta_{0,2} - theta_{1,2}) * prec_theta1
   * The term (theta_{0,1} + theta_{0,2}) represents the expected starting level,
   * and we subtract theta_{1,2} to isolate the level component contribution. */
  mean_theta_1[0] = data[0] * prec_data +
                    (theta_01 + theta_02 - theta_2_current[0]) * prec_theta_1;

  /* Middle elements (t=2 to n-1): incorporate trend differences between consecutive time points
   * Structure: y_t * prec_y + (theta_{t-1,2} - theta_{t,2}) * prec_theta1
   * The trend difference (theta_{t-1,2} - theta_{t,2}) captures how the trend
   * component influences the level evolution. */
  for (int j = 1; j < n - 1; j++) {
    mean_theta_1[j] = data[j] * prec_data +
                      (theta_2_current[j - 1] - theta_2_current[j]) * prec_theta_1;
  }

  /* Last element (t=n): incorporates final trend component
   * Structure: y_n * prec_y + theta_{n-1,2} * prec_theta1
   * At the final time point, only the previous trend contributes
   * (no subsequent observation to condition on). */
  mean_theta_1[n - 1] = data[n - 1] * prec_data +
                        theta_2_current[n - 2] * prec_theta_1;

  /* ========== Sample from Multivariate Normal Posterior ========== */
  /* Draw theta_1 ~ MVN(mu_post, Sigma_post)
   * Precision matrix structure identical to local level case,
   * but mean vector incorporates trend adjustments.
   * - a = prec_data (observation precision)
   * - b = prec_theta_1 (innovation precision)
   * - add_a = 1 (last diagonal: prec_data + prec_theta_1) */
  generate_normal_vector(
    theta_1_current, /* r: output buffer for theta_1 samples */
    mean_theta_1,    /* y: conditional posterior mean vector */
    prec_data,       /* a: observation precision contribution */
    prec_theta_1,    /* b: innovation precision contribution */
    n,               /* n: number of time points */
    1                /* add_a: adjust last diagonal element */
  );

  /* Release the R_alloc scratch (mean vector); results are in theta_1_current. */
  vmaxset(vmax);
}

//----------------------------------------------------------------------

/**
 * @brief Sample intermediate state vector theta_k from conditional posterior
 *        (k=2,...,p-1)
 *
 * @details Generates sample from the conditional posterior of intermediate state vectors
 *          in polynomial dynamic models. The posterior is multivariate Normal with
 *          tridiagonal precision matrix, incorporating information from adjacent
 *          polynomial components theta_{k-1} and theta_{k+1}.
 *
 *          **Model structure:**
 *          theta_{t,k-1} = theta_{t-1,k-1} + theta_{t-1,k} + u_{t,k-1}
 *          theta_{t,k} = theta_{t-1,k} + theta_{t-1,k+1} + u_{t,k}
 *          theta_{1,k} = theta_{0,k} + theta_{0,k+1} + u_{1,k}
 *
 *          **Conditional posterior:**
 *          theta_k | theta_{k-1}, theta_{k+1}, precisions, theta_{0k}, theta_{0,k+1}
 *              ~ MVN(mu_post, Sigma_post)
 *
 *          **Algorithm:**
 *          1. Extract information from lower-order component (k-1) differences
 *          2. Extract information from higher-order component (k+1) contributions
 *          3. Combine with initial state conditions
 *          4. Sample from MVN with tridiagonal precision
 *
 * @param theta_km1_previous  Previous iteration (k-1)-th component state vector [n] (const).
 *                            Contains theta_{1,k-1}, ..., theta_{n,k-1} from iteration t-1.
 *                            Used to compute state differences.
 * @param theta_k_current     Output buffer [n] for sampled theta_k vector.
 *                            Will be filled with theta_{1,k}, ..., theta_{n,k}.
 * @param theta_kp1_current   Current (k+1)-th component state vector [n] (const).
 *                            Contains theta_{1,k+1}, ..., theta_{n,k+1} from current iteration.
 * @param prec_theta_km1      Scalar innovation precision 1/W_{k-1} (> 0).
 *                            Controls influence of lower-order component.
 * @param prec_theta_k        Scalar innovation precision 1/W_k (> 0).
 *                            Controls smoothness of k-th component trajectory.
 * @param theta_0k            Scalar initial state theta_{0,k}.
 *                            Starting point for k-th component.
 * @param theta_0kp1          Scalar initial state theta_{0,k+1}.
 *                            Starting point for (k+1)-th component.
 * @param n                   Sample size (number of observations, must be > 2).
 *
 * @note Computational complexity: O(n) for mean vector computation and tridiagonal solving.
 * @note Numerical stability: Uses specialized tridiagonal solver in generate_normal_vector.
 * @note Memory access: Mixed iteration offsets for proper Gibbs sampling dependencies.
 * @note Algorithm: Multivariate Normal sampling with polynomial component interactions.
 * @note Boundary condition: add_a = 0 (last diagonal: prec_km1 only).
 * @note Notation: km1 = k-1, kp1 = k+1 for parameter naming.
 *
 * @warning Assumes n > 2 for proper tridiagonal structure.
 * @warning No validation of precision parameter positivity.
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function.
 * @warning theta_km1_previous must contain valid values from previous iteration.
 *
 * @see generate_normal_vector
 * @see generate_theta_1
 * @see generate_theta_p
 */
void generate_theta_k(const double *theta_km1_previous,
                      double       *theta_k_current,
                      const double *theta_kp1_current,
                      double        prec_theta_km1,
                      double        prec_theta_k,
                      double        theta_0k,
                      double        theta_0kp1,
                      int           n) {

  /* ========== Allocate Mean Vector ========== */
  /* Temporary storage for conditional posterior mean.
   * This vector combines information from adjacent polynomial components.
   * vmaxget()/vmaxset() bracket: reclaim the per-call R_alloc scratch, which
   * would otherwise accumulate across Gibbs iterations until .Call returns. */
  const void *vmax = vmaxget();
  double *mean_theta_k = (double *)R_alloc(n, sizeof(double));

  /* ========== Construct Posterior Mean Vector ========== */
  /* First element (t=1): incorporates initial conditions and adjacent component effects
   * Structure: (theta_{2,k-1} - theta_{1,k-1}) * prec_{k-1}
   *          + (theta_{0,k} + theta_{0,k+1} - theta_{1,k+1}) * prec_k
   * The first term captures information from differences in the lower component,
   * while the second term isolates theta_k contribution from the boundary condition. */
  mean_theta_k[0] = (theta_km1_previous[1] - theta_km1_previous[0]) * prec_theta_km1 +
  (theta_0k + theta_0kp1 - theta_kp1_current[0]) * prec_theta_k;

  /* Middle elements (t=2 to n-1): incorporate differences from adjacent polynomial components
   * Structure: (theta_{t+1,k-1} - theta_{t,k-1}) * prec_{k-1}
   *          + (theta_{t-1,k+1} - theta_{t,k+1}) * prec_k
   * Both terms represent how adjacent components constrain the k-th component evolution. */
  for (int j = 1; j < n - 1; j++) {
    mean_theta_k[j] = (theta_km1_previous[j + 1] - theta_km1_previous[j]) * prec_theta_km1 +
      (theta_kp1_current[j - 1] - theta_kp1_current[j]) * prec_theta_k;
  }

  /* Last element (t=n): only higher-order component contribution
   * Structure: theta_{n-1,k+1} * prec_k
   * At the boundary, only the previous time point from higher component contributes. */
  mean_theta_k[n - 1] = theta_kp1_current[n - 2] * prec_theta_k;

  /* ========== Sample from Multivariate Normal Posterior ========== */
  /* Draw theta_k ~ MVN(mu_post, Sigma_post)
   * Precision matrix structure:
   * - a = prec_theta_km1 (lower-order component contribution)
   * - b = prec_theta_k (higher-order component contribution)
   * - Diagonal: prec_km1 + 2*prec_k (except last)
   * - Off-diagonal: -prec_k
   * - Last diagonal: prec_km1 (add_a = 0) */
  generate_normal_vector(
    theta_k_current, /* r: output buffer for theta_k samples */
    mean_theta_k,    /* y: conditional posterior mean vector */
    prec_theta_km1,  /* a: precision from component k-1 */
    prec_theta_k,    /* b: precision from component k+1 */
    n,               /* n: number of time points */
    0                /* add_a: boundary adjustment flag */
  );

  /* Release the R_alloc scratch (mean vector); results are in theta_k_current. */
  vmaxset(vmax);
}

//----------------------------------------------------------------------

/**
 * @brief Sample final state vector theta_p from conditional posterior
 *        (k=p)
 *
 * @details Generates sample from the conditional posterior of the highest-order state vector
 *          in polynomial dynamic models. This is a boundary case handling the final polynomial
 *          component following a random walk structure, influenced only by the previous
 *          component theta_{p-1}.
 *
 *          **Model structure:**
 *          theta_{t,p-1} = theta_{t-1,p-1} + theta_{t-1,p} + u_{t,p-1}
 *          theta_{t,p} = theta_{t-1,p} + u_{t,p},     u_{t,p} ~ N(0, W_p)
 *          theta_{1,p} = theta_{0,p} + u_{1,p}
 *
 *          **Conditional posterior:**
 *          theta_p | theta_{p-1}, precisions, theta_{0p} ~ MVN(mu_post, Sigma_post)
 *
 *          **Algorithm:**
 *          1. Extract information from penultimate component (p-1) differences
 *          2. Account for initial state condition
 *          3. Sample from MVN with random walk structure
 *
 *          **Notation:**
 *          p represents the polynomial order (final/highest-order component).
 *          pm1 = p-1 (penultimate component).
 *
 * @param theta_pm1_previous  Previous iteration (p-1)-th component state vector [n] (const).
 *                            Contains theta_{1,p-1}, ..., theta_{n,p-1} from iteration t-1.
 *                            Used to compute state differences.
 * @param theta_p_current     Output buffer [n] for sampled theta_p vector.
 *                            Will be filled with theta_{1,p}, ..., theta_{n,p}.
 * @param prec_theta_pm1      Scalar innovation precision 1/W_{p-1} (> 0).
 *                            Controls influence of penultimate component.
 * @param prec_theta_p        Scalar innovation precision 1/W_p (> 0).
 *                            Controls smoothness of final component random walk.
 * @param theta_0p            Scalar initial state theta_{0,p}.
 *                            Starting point for highest-order component.
 * @param n                   Sample size (number of observations, must be > 2).
 *
 * @note Computational complexity: O(n) for mean vector computation and tridiagonal solving.
 * @note Numerical stability: Uses specialized tridiagonal solver in generate_normal_vector.
 * @note Memory access: Sequential access with previous iteration offsets.
 * @note Algorithm: Multivariate Normal sampling for random walk boundary case.
 * @note Boundary condition: add_a = 0 (last diagonal: prec_pm1 only).
 * @note Notation: pm1 = p-1 for parameter naming, p = polynomial order.
 *
 * @warning Assumes n > 2 for proper tridiagonal structure.
 * @warning No validation of precision parameter positivity.
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function.
 * @warning theta_pm1_previous must contain valid values from previous iteration.
 *
 * @see generate_normal_vector
 * @see generate_theta_k
 * @see generate_precision_theta_p
 */
void generate_theta_p(const double *theta_pm1_previous,
                      double       *theta_p_current,
                      double        prec_theta_pm1,
                      double        prec_theta_p,
                      double        theta_0p,
                      int           n) {

  /* ========== Allocate Mean Vector ========== */
  /* Temporary storage for conditional posterior mean.
   * This vector combines information from penultimate component.
   * vmaxget()/vmaxset() bracket: reclaim the per-call R_alloc scratch, which
   * would otherwise accumulate across Gibbs iterations until .Call returns. */
  const void *vmax = vmaxget();
  double *mean_theta_p = (double *)R_alloc(n, sizeof(double));

  /* ========== Construct Posterior Mean Vector ========== */
  /* Elements t=1 to n-1: incorporate differences from penultimate component theta_{p-1}
   * Structure: (theta_{t+1,p-1} - theta_{t,p-1}) * prec_{p-1}
   * These differences represent how the penultimate component evolution
   * informs the final component through the polynomial structure. */
  for (int j = 0; j < n - 1; j++) {
    mean_theta_p[j] = (theta_pm1_previous[j + 1] - theta_pm1_previous[j]) * prec_theta_pm1;
  }

  /* Last element (t=n): no contribution from differences (boundary case)
   * At the final time point, there's no subsequent observation to
   * provide information through differences. */
  mean_theta_p[n - 1] = 0.0;

  /* Add initial condition component for the first element
   * Structure: theta_{0,p} * prec_p
   * This implements the boundary condition theta_{1,p} = theta_{0,p} + u_{1,p}. */
  mean_theta_p[0] += theta_0p * prec_theta_p;

  /* ========== Sample from Multivariate Normal Posterior ========== */
  /* Draw theta_p ~ MVN(mu_post, Sigma_post)
   * Precision matrix structure for random walk component:
   * - a = prec_theta_pm1 (penultimate component contribution)
   * - b = prec_theta_p (random walk innovation contribution)
   * - Diagonal: prec_pm1 + 2*prec_p (except last)
   * - Off-diagonal: -prec_p
   * - Last diagonal: prec_pm1 (add_a = 0) */
  generate_normal_vector(
    theta_p_current, /* r: output buffer for theta_p samples */
    mean_theta_p,    /* y: conditional posterior mean vector */
    prec_theta_pm1,  /* a: precision from component p-1 */
    prec_theta_p,    /* b: innovation precision for theta_p */
    n,               /* n: number of time points */
    0                /* add_a: boundary adjustment flag */
  );

  /* Release the R_alloc scratch (mean vector); results are in theta_p_current. */
  vmaxset(vmax);
}
