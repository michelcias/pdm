/**
 * @file conditional_precision.c
 * @brief Conditional posterior sampling for precision parameters in Gaussian models
 * @author Michel H. Montoril
 * @date 2025-10-11
 * @version 1.1
 *
 * @details This file contains optimized functions for Bayesian precision parameter sampling:
 *          - Observation precision sampling (1/V) for Gaussian likelihood
 *          - Intermediate state precision sampling (1/W_k) for polynomial components
 *          - Final state precision sampling (1/W_p) for random walk boundary case
 *          - Memory-efficient implementations using scalar returns
 *          - Full Gamma-Normal conjugacy exploitation for computational efficiency
 *
 * @changelog
 * - v1.1 (2025-10-11): Refactored all functions to use scalar parameters and return
 *   values instead of array indexing. Removed iteration index parameters. Improved
 *   memory efficiency by eliminating O(n_iter) storage requirements. Added const
 *   qualifiers for input arrays. Enhanced numerical stability documentation.
 * - v1.0 (2025-08-12): Initial implementation with array-based interface.
 */

#include <R.h>
#include <Rmath.h>
#include "conditional_precision.h"

/**
 * @brief Sample observation precision 1/V from Gamma posterior
 *
 * @details Generates sample from the conditional posterior of the observation precision
 *          in Gaussian polynomial dynamic models. The posterior is Gamma distributed,
 *          derived from combining a Gamma prior with the Normal likelihood of observations.
 *
 *          **Model structure:**
 *          y_t = theta_{t,1} + e_t,  e_t ~ N(0, V)
 *          1/V ~ Gamma(nu_y, eta_y)
 *
 *          **Conditional posterior:**
 *          1/V | y, theta_1 ~ Gamma(nu_post, eta_post)
 *          where nu_post = nu_y + n/2
 *                eta_post = eta_y + (1/2) * sum_{t=1}^n (y_t - theta_{t,1})^2
 *
 *          **Algorithm:**
 *          1. Accumulate sum of squared errors: SSE = sum (y_t - theta_{t,1})^2
 *          2. Update Gamma parameters using conjugate formulas
 *          3. Sample precision from Gamma(nu_post, eta_post)
 *
 * @param y                Observed data vector [n] (const, read-only).
 *                         Contains time series observations y_1, ..., y_n.
 * @param theta_1_current  Current level state vector [n] (const, read-only).
 *                         Contains sampled level states theta_{1,1}, ..., theta_{n,1}
 *                         from current MCMC iteration.
 * @param nu_y             Prior shape parameter (nu_y > 0).
 *                         Controls prior precision about observation variance.
 * @param eta_y            Prior rate parameter (eta_y > 0).
 *                         Together with nu_y defines prior mean = nu_y / eta_y.
 * @param n                Sample size (number of observations).
 *
 * @return Sampled precision value 1/V from Gamma posterior.
 *
 * @note Computational complexity: O(n) for sum of squared errors calculation.
 * @note Numerical stability: Uses double precision accumulation for sum of squares.
 * @note Memory access: Sequential reads from input arrays (cache-friendly).
 * @note Algorithm: Gamma-Normal conjugate updating (closed-form posterior).
 * @note Typical values: nu_y = 0.001, eta_y = 0.001 (vague prior).
 *
 * @warning No validation of prior parameter positivity (nu_y > 0, eta_y > 0).
 *          Caller must ensure valid inputs to avoid NaN/Inf propagation.
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function.
 * @warning For n = 0, behavior is undefined (should never occur in practice).
 *
 * @see generate_precision_theta_p
 * @see generate_precision_theta_k
 * @see rgamma
 */
double generate_precision_data(const double *y,
                               const double *theta_1_current,
                               double        nu_y,
                               double        eta_y,
                               int           n) {

  /* ========== Compute Sum of Squared Errors ========== */
  /* Accumulate SSE = sum_{t=1}^n (y_t - theta_{t,1})^2
   * This represents the total unexplained variation in observations
   * after accounting for the dynamic level component. */
  double ss_y = 0.0;
  for (int j = 0; j < n; j++) {
    double residual = y[j] - theta_1_current[j];
    ss_y += residual * residual;
  }

  /* ========== Update Gamma Posterior Parameters ========== */
  /* Apply conjugate updating formulas:
   * - Shape increases by number of observations / 2
   * - Rate increases by half the sum of squared errors
   * This balances prior information with likelihood evidence. */
  double nu_y_post = nu_y + n / 2.0;          /* Posterior shape parameter */
  double eta_y_post = eta_y + ss_y / 2.0;     /* Posterior rate parameter */

  /* ========== Sample Precision from Gamma Posterior ========== */
  /* Draw 1/V ~ Gamma(nu_post, eta_post)
   * R's rgamma uses scale parameterization (inverse of rate).
   * Higher SSE leads to lower sampled precision (higher variance). */
  return rgamma(
    nu_y_post,       /* shape: posterior Gamma shape parameter */
  1.0 / eta_y_post /* scale: inverse of posterior rate parameter */
  );
}

//----------------------------------------------------------------------

/**
 * @brief Sample intermediate innovation precision 1/W_k from Gamma posterior
 *        (k=2,...,p-1)
 *
 * @details Generates sample from the conditional posterior of innovation precision
 *          for intermediate polynomial components. The posterior is Gamma distributed,
 *          derived from combining a Gamma prior with state evolution equations.
 *
 *          **Model structure:**
 *          theta_{t,k-1} = theta_{t-1,k-1} + theta_{t-1,k} + u_{t,k-1}
 *          theta_{t,k} = theta_{t-1,k} + theta_{t-1,k+1} + u_{t,k},  u_{t,k} ~ N(0, W_k)
 *          theta_{1,k} = theta_{0,k} + theta_{0,k+1} + u_{1,k}
 *          1/W_k ~ Gamma(nu_k, eta_k)
 *
 *          **Innovation calculation:**
 *          For t=1: u_{1,k} = theta_{1,k} - theta_{0,k} - theta_{0,k+1}
 *          For t>1: u_{t,k} = theta_{t,k} - theta_{t-1,k} - theta_{t-1,k+1}
 *
 *          **Conditional posterior:**
 *          1/W_k | theta_k, theta_{k+1}, theta_{0k}, theta_{0,k+1} ~ Gamma(nu_post, eta_post)
 *          where nu_post = nu_k + n/2
 *                eta_post = eta_k + (1/2) * sum_{t=1}^n u_{t,k}^2
 *
 *          **Algorithm:**
 *          1. Compute first innovation using initial states (boundary condition)
 *          2. Accumulate squared innovations for remaining time points
 *          3. Update Gamma parameters and sample precision
 *
 * @param theta_0k          Scalar initial state theta_{0,k}.
 *                          Starting value for k-th polynomial component.
 * @param theta_0kp1        Scalar initial state theta_{0,k+1}.
 *                          Starting value for (k+1)-th polynomial component.
 * @param theta_k_current   Current k-th component state vector [n] (const).
 *                          Contains theta_{1,k}, ..., theta_{n,k} from current iteration.
 * @param theta_kp1_current Current (k+1)-th component state vector [n] (const).
 *                          Contains theta_{1,k+1}, ..., theta_{n,k+1} from current iteration.
 * @param nu_0k             Prior shape parameter (nu_k > 0).
 *                          Controls prior precision about innovation variance.
 * @param eta_0k            Prior rate parameter (eta_k > 0).
 *                          Together with nu_k defines prior mean = nu_k / eta_k.
 * @param n                 Sample size (number of observations).
 *
 * @return Sampled precision value 1/W_k from Gamma posterior.
 *
 * @note Computational complexity: O(n) for sum of squared innovations calculation.
 * @note Numerical stability: Uses double precision accumulation for sum of squares.
 * @note Memory access: Sequential reads from state vectors (cache-friendly).
 * @note Algorithm: Gamma-Normal conjugate updating with state difference calculations.
 * @note Notation: kp1 = k+1 for parameter naming consistency.
 * @note Boundary condition: First innovation explicitly accounts for initial states.
 * @note Typical values: nu_k = 0.001, eta_k = 0.001 (vague prior on innovation variance).
 *
 * @warning No validation of prior parameter positivity (nu_0k > 0, eta_0k > 0).
 *          Caller must ensure valid inputs to avoid NaN/Inf propagation.
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function.
 * @warning For n = 0, behavior is undefined (should never occur in practice).
 *
 * @see generate_precision_data
 * @see generate_precision_theta_p
 * @see rgamma
 */
double generate_precision_theta_k(double        theta_0k,
                                  double        theta_0kp1,
                                  const double *theta_k_current,
                                  const double *theta_kp1_current,
                                  double        nu_0k,
                                  double        eta_0k,
                                  int           n) {

  /* ========== Compute Sum of Squared Innovations ========== */
  /* First innovation (j=0): incorporates initial state boundary condition
   * u_{1,k} = theta_{1,k} - theta_{0,k} - theta_{0,k+1}
   * This captures the initial deviation from the polynomial trend structure. */
  double theta_centered = theta_k_current[0] - theta_0k - theta_0kp1;
  double ss_theta = theta_centered * theta_centered;

  /* Remaining innovations (j=1 to n-1): standard state evolution
   * u_{t,k} = theta_{t,k} - theta_{t-1,k} - theta_{t-1,k+1}
   * These measure how well the polynomial trend propagates forward. */
  for (int j = 1; j < n; j++) {
    theta_centered = theta_k_current[j] - theta_k_current[j - 1] - theta_kp1_current[j - 1];
    ss_theta += theta_centered * theta_centered;
  }

  /* ========== Update Gamma Posterior Parameters ========== */
  /* Apply conjugate updating formulas:
   * - Shape increases by number of time points / 2
   * - Rate increases by half the sum of squared innovations
   * Higher innovation variance → lower sampled precision. */
  double nu_0k_post = nu_0k + n / 2.0;          /* Posterior shape parameter */
  double eta_0k_post = eta_0k + ss_theta / 2.0; /* Posterior rate parameter */

  /* ========== Sample Precision from Gamma Posterior ========== */
  /* Draw 1/W_k ~ Gamma(nu_post, eta_post)
   * R's rgamma uses scale parameterization (inverse of rate). */
  return rgamma(
    nu_0k_post,      /* shape: posterior Gamma shape parameter */
  1.0 / eta_0k_post /* scale: inverse of posterior rate parameter */
  );
}

//----------------------------------------------------------------------

/**
 * @brief Sample final innovation precision 1/W_p from Gamma posterior
 *        (k=p)
 *
 * @details Generates sample from the conditional posterior of innovation precision
 *          for the highest-order polynomial component. This is a boundary case handling
 *          the final component following a random walk structure.
 *
 *          **Model structure:**
 *          theta_{t,p} = theta_{t-1,p} + u_{t,p},  u_{t,p} ~ N(0, W_p)
 *          theta_{1,p} = theta_{0,p} + u_{1,p}
 *          1/W_p ~ Gamma(nu_p, eta_p)
 *
 *          **Innovation calculation (random walk):**
 *          For t=1: u_{1,p} = theta_{1,p} - theta_{0,p}
 *          For t>1: u_{t,p} = theta_{t,p} - theta_{t-1,p}
 *
 *          **Conditional posterior:**
 *          1/W_p | theta_p, theta_{0,p} ~ Gamma(nu_post, eta_post)
 *          where nu_post = nu_p + n/2
 *                eta_post = eta_p + (1/2) * sum_{t=1}^n u_{t,p}^2
 *
 *          **Algorithm:**
 *          1. Compute first innovation from initial state (boundary condition)
 *          2. Accumulate squared increments for random walk component
 *          3. Update Gamma parameters and sample precision
 *
 *          **Notation:**
 *          p represents the polynomial order (final/highest-order component).
 *
 * @param theta_0p        Scalar initial state theta_{0,p}.
 *                        Starting value for highest-order polynomial component.
 * @param theta_p_current Current p-th component state vector [n] (const).
 *                        Contains theta_{1,p}, ..., theta_{n,p} from current iteration.
 * @param nu_0p           Prior shape parameter (nu_p > 0).
 *                        Controls prior precision about innovation variance.
 * @param eta_0p          Prior rate parameter (eta_p > 0).
 *                        Together with nu_p defines prior mean = nu_p / eta_p.
 * @param n               Sample size (number of observations).
 *
 * @return Sampled precision value 1/W_p from Gamma posterior.
 *
 * @note Computational complexity: O(n) for sum of squared innovations calculation.
 * @note Numerical stability: Uses double precision accumulation for sum of squares.
 * @note Memory access: Sequential reads from state vector (cache-friendly).
 * @note Algorithm: Gamma-Normal conjugate updating for random walk boundary case.
 * @note Notation: p represents polynomial order (final component).
 * @note Boundary condition: First innovation explicitly accounts for initial state.
 * @note Typical values: nu_p = 0.001, eta_p = 0.001 (vague prior on innovation variance).
 * @note Special case: For p=1 (local level model), this samples the only innovation precision.
 *
 * @warning No validation of prior parameter positivity (nu_0p > 0, eta_0p > 0).
 *          Caller must ensure valid inputs to avoid NaN/Inf propagation.
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function.
 * @warning For n = 0, behavior is undefined (should never occur in practice).
 *
 * @see generate_precision_data
 * @see generate_precision_theta_k
 * @see rgamma
 */
double generate_precision_theta_p(double        theta_0p,
                                  const double *theta_p_current,
                                  double        nu_0p,
                                  double        eta_0p,
                                  int           n) {

  /* ========== Compute Sum of Squared Innovations ========== */
  /* First innovation (j=0): incorporates initial state boundary condition
   * u_{1,p} = theta_{1,p} - theta_{0,p}
   * This captures the initial random walk increment. */
  double theta_centered = theta_p_current[0] - theta_0p;
  double ss_theta = theta_centered * theta_centered;

  /* Remaining innovations (j=1 to n-1): random walk increments
   * u_{t,p} = theta_{t,p} - theta_{t-1,p}
   * These measure the step sizes of the random walk process. */
  for (int j = 1; j < n; j++) {
    theta_centered = theta_p_current[j] - theta_p_current[j - 1];
    ss_theta += theta_centered * theta_centered;
  }

  /* ========== Update Gamma Posterior Parameters ========== */
  /* Apply conjugate updating formulas:
   * - Shape increases by number of time points / 2
   * - Rate increases by half the sum of squared increments
   * Higher random walk variability → lower sampled precision. */
  double nu_0p_post = nu_0p + n / 2.0;          /* Posterior shape parameter */
  double eta_0p_post = eta_0p + ss_theta / 2.0; /* Posterior rate parameter */

  /* ========== Sample Precision from Gamma Posterior ========== */
  /* Draw 1/W_p ~ Gamma(nu_post, eta_post)
   * R's rgamma uses scale parameterization (inverse of rate). */
  return rgamma(
    nu_0p_post,      /* shape: posterior Gamma shape parameter */
  1.0 / eta_0p_post /* scale: inverse of posterior rate parameter */
  );
}
