/**
 * @file conditional_theta0.c
 * @brief Conditional posterior sampling for initial state parameters in Gaussian models
 * @author Michel H. Montoril
 * @date 2025-10-11
 * @version 1.1
 *
 * @details This file contains optimized functions for Bayesian initial state parameter sampling
 *          in polynomial dynamic models with Gaussian observation equations:
 *          - Initial level state sampling (theta_{0,1}) for local level models
 *          - Initial level state sampling (theta_{0,1}) for local trend models
 *          - Intermediate initial state sampling (theta_{0,k}) for k=2,...,p-1
 *          - Final initial state sampling (theta_{0,p}) for highest-order component
 *          - Memory-efficient implementations using scalar returns
 *          - Full Normal-Normal conjugacy exploitation for computational efficiency
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
#include "conditional_theta0.h"

/**
 * @brief Sample initial level state theta_{0,1} from Normal posterior
 *        (local level model)
 *
 * @details Generates sample from the conditional posterior of the initial level state
 *          in local level dynamic models. The posterior is Normal distributed, derived
 *          from combining a Normal prior with the state evolution equation.
 *
 *          **Model structure:**
 *          theta_{1,1} = theta_{0,1} + u_{1,1},  u_{1,1} ~ N(0, W_1)
 *          theta_{0,1} ~ N(mu_0, sigma_0^2)
 *
 *          **Conditional posterior:**
 *          theta_{0,1} | theta_1, prec_1 ~ N(mu_post, sigma_post^2)
 *          where sigma_post^2 = 1 / (tau_0 + tau_1)
 *                mu_post = (mu_0 * tau_0 + theta_{1,1} * tau_1) / (tau_0 + tau_1)
 *          with tau_0 = 1/sigma_0^2 (prior precision)
 *               tau_1 = 1/W_1 (innovation precision)
 *
 *          **Algorithm:**
 *          1. Compute posterior precision as sum of prior and innovation precisions
 *          2. Compute posterior mean as precision-weighted average
 *          3. Sample from Normal(mu_post, sigma_post^2)
 *
 * @param theta_1_current   Current level state vector [n] (const, read-only).
 *                          Only first element theta_{1,1} is used.
 * @param prec_1            Scalar innovation precision 1/W_1.
 *                          Controls strength of information from first observation.
 * @param mean_theta_01     Prior mean mu_0.
 *                          Center of prior distribution for initial state.
 * @param prec_theta_01     Prior precision tau_0 = 1/sigma_0^2.
 *                          Controls strength of prior information.
 * @param n                 Sample size (used for interface consistency, not computation).
 *
 * @return Sampled initial state theta_{0,1} from Normal posterior.
 *
 * @note Computational complexity: O(1) constant time operation.
 * @note Numerical stability: Uses precision parameterization to avoid division.
 * @note Algorithm: Normal-Normal conjugate updating (closed-form posterior).
 * @note Only uses first element of theta_1_current (theta_{1,1}).
 * @note Typical values: mean_theta_01 = 0, prec_theta_01 = 0.001 (vague prior).
 *
 * @warning No validation of prior parameter positivity (prec_theta_01 > 0, prec_1 > 0).
 *          Caller must ensure valid inputs to avoid division by zero.
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function.
 * @warning Parameter n is not used in computation but maintained for interface consistency.
 *
 * @see generate_theta_01
 * @see generate_theta_0p
 * @see rnorm
 */
double generate_theta_01_locallevel(const double *theta_1_current,
                                          double  prec_1,
                                          double  mean_theta_01,
                                          double  prec_theta_01,
                                          int     n) {

  /* ========== Compute Posterior Precision ========== */
  /* Combine prior precision with innovation precision.
   * Higher precision → more concentrated posterior distribution.
   * tau_post = tau_0 + tau_1 */
  double prec_theta_01_post = prec_theta_01 + prec_1;

  /* ========== Compute Posterior Mean ========== */
  /* Precision-weighted average of prior mean and observed first state.
   * As tau_1 increases (more precise innovations), posterior pulls toward theta_{1,1}.
   * As tau_0 increases (stronger prior), posterior pulls toward mu_0.
   * mu_post = (mu_0 * tau_0 + theta_{1,1} * tau_1) / (tau_0 + tau_1) */
  double mean_theta_01_post = (mean_theta_01 * prec_theta_01 +
  theta_1_current[0] * prec_1) / prec_theta_01_post;

  /* ========== Sample from Normal Posterior ========== */
  /* Draw theta_{0,1} ~ N(mu_post, 1/tau_post)
   * R's rnorm uses standard deviation parameterization (not precision). */
  return rnorm(
    mean_theta_01_post,           /* mean: posterior expectation */
    sqrt(1.0 / prec_theta_01_post)  /* sd: posterior standard deviation */
  );
}

//----------------------------------------------------------------------

/**
 * @brief Sample initial level state theta_{0,1} from Normal posterior
 *        (local trend model)
 *
 * @details Generates sample from the conditional posterior of the initial level state
 *          in local trend dynamic models. The posterior is Normal distributed, derived
 *          from combining a Normal prior with the state evolution equation that includes
 *          the initial trend component.
 *
 *          **Model structure:**
 *          theta_{1,1} = theta_{0,1} + theta_{0,2} + u_{1,1},  u_{1,1} ~ N(0, W_1)
 *          theta_{0,1} ~ N(mu_0, sigma_0^2)
 *
 *          **Conditional posterior:**
 *          theta_{0,1} | theta_1, theta_{0,2}, prec_1 ~ N(mu_post, sigma_post^2)
 *          where sigma_post^2 = 1 / (tau_0 + tau_1)
 *                mu_post = (mu_0 * tau_0 + (theta_{1,1} - theta_{0,2}) * tau_1) / (tau_0 + tau_1)
 *
 *          **Algorithm:**
 *          1. Compute posterior precision as sum of prior and innovation precisions
 *          2. Adjust first state for trend contribution: theta_{1,1} - theta_{0,2}
 *          3. Compute posterior mean as precision-weighted average
 *          4. Sample from Normal(mu_post, sigma_post^2)
 *
 * @param theta_1_current   Current level state vector [n] (const, read-only).
 *                          Only first element theta_{1,1} is used.
 * @param theta_02          Scalar initial trend state theta_{0,2}.
 *                          Trend contribution to be removed from first observation.
 * @param prec_1            Scalar innovation precision 1/W_1.
 *                          Controls strength of information from first observation.
 * @param mean_theta_01     Prior mean mu_0.
 *                          Center of prior distribution for initial level.
 * @param prec_theta_01     Prior precision tau_0 = 1/sigma_0^2.
 *                          Controls strength of prior information.
 * @param n                 Sample size (used for interface consistency, not computation).
 *
 * @return Sampled initial state theta_{0,1} from Normal posterior.
 *
 * @note Computational complexity: O(1) constant time operation.
 * @note Numerical stability: Uses precision parameterization to avoid division.
 * @note Algorithm: Normal-Normal conjugate updating with trend adjustment.
 * @note Only uses first element of theta_1_current (theta_{1,1}).
 * @note Typical values: mean_theta_01 = 0, prec_theta_01 = 0.001 (vague prior).
 *
 * @warning No validation of prior parameter positivity (prec_theta_01 > 0, prec_1 > 0).
 *          Caller must ensure valid inputs to avoid division by zero.
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function.
 * @warning Parameter n is not used in computation but maintained for interface consistency.
 *
 * @see generate_theta_01_locallevel
 * @see generate_theta_0k
 * @see rnorm
 */
double generate_theta_01(const double *theta_1_current,
                               double  theta_02,
                               double  prec_1,
                               double  mean_theta_01,
                               double  prec_theta_01,
                               int     n) {

  /* ========== Compute Posterior Precision ========== */
  /* Combine prior precision with innovation precision.
   * tau_post = tau_0 + tau_1 */
  double prec_theta_01_post = prec_theta_01 + prec_1;

  /* ========== Compute Posterior Mean ========== */
  /* Precision-weighted average of prior mean and trend-adjusted first state.
   * The term (theta_{1,1} - theta_{0,2}) isolates the level component contribution.
   * mu_post = (mu_0 * tau_0 + (theta_{1,1} - theta_{0,2}) * tau_1) / (tau_0 + tau_1) */
  double mean_theta_01_post = (mean_theta_01 * prec_theta_01 +
  (theta_1_current[0] - theta_02) * prec_1) / prec_theta_01_post;

  /* ========== Sample from Normal Posterior ========== */
  /* Draw theta_{0,1} ~ N(mu_post, 1/tau_post)
   * R's rnorm uses standard deviation parameterization (not precision). */
  return rnorm(
    mean_theta_01_post,           /* mean: posterior expectation */
    sqrt(1.0 / prec_theta_01_post) /* sd: posterior standard deviation */
  );
}

//----------------------------------------------------------------------

/**
 * @brief Sample intermediate initial state theta_{0,k} from Normal posterior
 *        (k=2,...,p-1)
 *
 * @details Generates sample from the conditional posterior of intermediate initial states
 *          in polynomial dynamic models. The posterior is Normal distributed, derived from
 *          combining a Normal prior with information from adjacent polynomial components.
 *
 *          **Model structure:**
 *          theta_{1,k-1} = theta_{0,k-1} + theta_{0,k} + u_{1,k-1},  u_{1,k-1} ~ N(0, W_{k-1})
 *          theta_{1,k} = theta_{0,k} + theta_{0,k+1} + u_{1,k},     u_{1,k} ~ N(0, W_k)
 *          theta_{0,k} ~ N(mu_{0k}, sigma_{0k}^2)
 *
 *          **Conditional posterior:**
 *          theta_{0,k} | theta_{k-1}, theta_k, theta_{0,k-1}, theta_{0,k+1}, prec_{k-1}, prec_k
 *              ~ N(mu_post, sigma_post^2)
 *          where sigma_post^2 = 1 / (tau_{0k} + tau_{k-1} + tau_k)
 *                mu_post = [mu_{0k} * tau_{0k} + (theta_{1,k-1} - theta_{0,k-1}) * tau_{k-1}
 *                          + (theta_{1,k} - theta_{0,k+1}) * tau_k] / (tau_{0k} + tau_{k-1} + tau_k)
 *
 *          **Algorithm:**
 *          1. Compute posterior precision from three sources (prior, two adjacent components)
 *          2. Extract information from lower-order component (k-1)
 *          3. Extract information from current component (k)
 *          4. Compute precision-weighted mean and sample
 *
 * @param theta_km1_current Current (k-1)-th component state vector [n] (const).
 *                          Only first element theta_{1,k-1} is used.
 * @param theta_k_current   Current k-th component state vector [n] (const).
 *                          Only first element theta_{1,k} is used.
 * @param theta_0km1        Scalar initial state theta_{0,k-1} for lower component.
 * @param theta_0kp1        Scalar initial state theta_{0,k+1} for higher component.
 * @param prec_km1          Scalar innovation precision 1/W_{k-1} for lower component.
 * @param prec_k            Scalar innovation precision 1/W_k for current component.
 * @param mean_theta_0k     Prior mean mu_{0k} for theta_{0,k}.
 * @param prec_theta_0k     Prior precision tau_{0k} = 1/sigma_{0k}^2.
 * @param n                 Sample size (used for interface consistency).
 *
 * @return Sampled initial state theta_{0,k} from Normal posterior.
 *
 * @note Computational complexity: O(1) constant time operation.
 * @note Numerical stability: Uses precision parameterization throughout.
 * @note Algorithm: Normal-Normal conjugate updating with multiple information sources.
 * @note Notation: km1 = k-1, kp1 = k+1 for parameter naming.
 * @note Only uses first elements of state vectors (theta_{1,k-1} and theta_{1,k}).
 * @note Typical values: mean_theta_0k = 0, prec_theta_0k = 0.001 (vague prior).
 *
 * @warning No validation of prior parameter positivity (all precisions must be > 0).
 *          Caller must ensure valid inputs to avoid division by zero.
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function.
 * @warning Parameter n is not used in computation but maintained for interface consistency.
 *
 * @see generate_theta_01
 * @see generate_theta_0p
 * @see rnorm
 */
double generate_theta_0k(const double *theta_km1_current,
                         const double *theta_k_current,
                               double  theta_0km1,
                               double  theta_0kp1,
                               double  prec_km1,
                               double  prec_k,
                               double  mean_theta_0k,
                               double  prec_theta_0k,
                               int     n) {

  /* ========== Compute Posterior Precision ========== */
  /* Combine three precision sources: prior, lower component, current component.
   * More information sources → higher posterior precision.
   * tau_post = tau_{0k} + tau_{k-1} + tau_k */
  double prec_theta_0k_post = prec_theta_0k + prec_km1 + prec_k;

  /* ========== Extract Information from Adjacent Components ========== */
  /* Information from lower-order component (k-1):
   * Isolates theta_{0,k} contribution by subtracting theta_{0,k-1} from first state.
   * term_{k-1} = (theta_{1,k-1} - theta_{0,k-1}) * tau_{k-1} */
  double info_from_km1 = (theta_km1_current[0] - theta_0km1) * prec_km1;

  /* Information from current component (k):
   * Isolates theta_{0,k} contribution by subtracting theta_{0,k+1} from first state.
   * term_k = (theta_{1,k} - theta_{0,k+1}) * tau_k */
  double info_from_k = (theta_k_current[0] - theta_0kp1) * prec_k;

  /* ========== Compute Posterior Mean ========== */
  /* Precision-weighted average of prior mean and information from two components.
   * mu_post = (mu_{0k} * tau_{0k} + term_{k-1} + term_k) / tau_post */
  double mean_theta_0k_post = (mean_theta_0k * prec_theta_0k +
  info_from_km1 + info_from_k) / prec_theta_0k_post;

  /* ========== Sample from Normal Posterior ========== */
  /* Draw theta_{0,k} ~ N(mu_post, 1/tau_post)
   * R's rnorm uses standard deviation parameterization (not precision). */
  return rnorm(
    mean_theta_0k_post,          /* mean: posterior expectation */
    sqrt(1.0 / prec_theta_0k_post) /* sd: posterior standard deviation */
  );
}

//----------------------------------------------------------------------

/**
 * @brief Sample final initial state theta_{0,p} from Normal posterior
 *        (k=p)
 *
 * @details Generates sample from the conditional posterior of the final initial state
 *          in polynomial dynamic models. This is a boundary case handling the highest-order
 *          polynomial component. The posterior is Normal distributed, derived from combining
 *          a Normal prior with information from the penultimate component and the final
 *          component's evolution.
 *
 *          **Model structure:**
 *          theta_{1,p-1} = theta_{0,p-1} + theta_{0,p} + u_{1,p-1},  u_{1,p-1} ~ N(0, W_{p-1})
 *          theta_{1,p} = theta_{0,p} + u_{1,p},                      u_{1,p} ~ N(0, W_p)
 *          theta_{0,p} ~ N(mu_{0p}, sigma_{0p}^2)
 *
 *          **Conditional posterior:**
 *          theta_{0,p} | theta_{p-1}, theta_p, theta_{0,p-1}, prec_{p-1}, prec_p ~ N(mu_post, sigma_post^2)
 *          where sigma_post^2 = 1 / (tau_{0p} + tau_{p-1} + tau_p)
 *                mu_post = [mu_{0p} * tau_{0p} + (theta_{1,p-1} - theta_{0,p-1}) * tau_{p-1}
 *                          + theta_{1,p} * tau_p] / (tau_{0p} + tau_{p-1} + tau_p)
 *
 *          **Algorithm:**
 *          1. Compute posterior precision from three sources
 *          2. Extract information from penultimate component (p-1)
 *          3. Extract information from final component (p) random walk
 *          4. Compute precision-weighted mean and sample
 *
 *          **Notation:**
 *          p represents the polynomial order (final/highest-order component).
 *          pm1 = p-1 (penultimate component).
 *
 * @param theta_pm1_current Current (p-1)-th component state vector [n] (const).
 *                          Only first element theta_{1,p-1} is used.
 * @param theta_p_current   Current p-th component state vector [n] (const).
 *                          Only first element theta_{1,p} is used.
 * @param theta_0pm1        Scalar initial state theta_{0,p-1} for penultimate component.
 * @param prec_pm1          Scalar innovation precision 1/W_{p-1} for penultimate component.
 * @param prec_p            Scalar innovation precision 1/W_p for final component.
 * @param mean_theta_0p     Prior mean mu_{0p} for theta_{0,p}.
 * @param prec_theta_0p     Prior precision tau_{0p} = 1/sigma_{0p}^2.
 * @param n                 Sample size (used for interface consistency).
 *
 * @return Sampled initial state theta_{0,p} from Normal posterior.
 *
 * @note Computational complexity: O(1) constant time operation.
 * @note Numerical stability: Uses precision parameterization throughout.
 * @note Algorithm: Normal-Normal conjugate updating for boundary case.
 * @note Notation: p = polynomial order (final component), pm1 = p-1.
 * @note Only uses first elements of state vectors.
 * @note Typical values: mean_theta_0p = 0, prec_theta_0p = 0.001 (vague prior).
 * @note Special case: For p=1 (local level), use generate_theta_01_locallevel instead.
 *
 * @warning No validation of prior parameter positivity (all precisions must be > 0).
 *          Caller must ensure valid inputs to avoid division by zero.
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function.
 * @warning Parameter n is not used in computation but maintained for interface consistency.
 *
 * @see generate_theta_01
 * @see generate_theta_0k
 * @see rnorm
 */
double generate_theta_0p(const double *theta_pm1_current,
                         const double *theta_p_current,
                               double  theta_0pm1,
                               double  prec_pm1,
                               double  prec_p,
                               double  mean_theta_0p,
                               double  prec_theta_0p,
                               int     n);{

  /* ========== Compute Posterior Precision ========== */
  /* Combine three precision sources: prior, penultimate component, final component.
   * tau_post = tau_{0p} + tau_{p-1} + tau_p */
  double prec_theta_0p_post = prec_theta_0p + prec_pm1 + prec_p;

  /* ========== Extract Information from Adjacent Components ========== */
  /* Information from penultimate component (p-1):
   * Isolates theta_{0,p} contribution by subtracting theta_{0,p-1} from first state.
   * term_{p-1} = (theta_{1,p-1} - theta_{0,p-1}) * tau_{p-1} */
  double info_from_pm1 = (theta_pm1_current[0] - theta_0pm1) * prec_pm1;

  /* Information from final component (p):
   * Direct observation of random walk starting point (no adjustment needed).
   * term_p = theta_{1,p} * tau_p */
  double info_from_p = theta_p_current[0] * prec_p;

  /* ========== Compute Posterior Mean ========== */
  /* Precision-weighted average of prior mean and information from two components.
   * mu_post = (mu_{0p} * tau_{0p} + term_{p-1} + term_p) / tau_post */
  double mean_theta_0p_post = (mean_theta_0p * prec_theta_0p +
  info_from_pm1 + info_from_p) / prec_theta_0p_post;

  /* ========== Sample from Normal Posterior ========== */
  /* Draw theta_{0,p} ~ N(mu_post, 1/tau_post)
   * R's rnorm uses standard deviation parameterization (not precision). */
  return rnorm(
    mean_theta_0p_post,          /* mean: posterior expectation */
    sqrt(1.0 / prec_theta_0p_post) /* sd: posterior standard deviation */
  );
}
