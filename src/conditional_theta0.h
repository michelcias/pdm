/**
 * @file conditional_theta0.h
 * @brief Conditional posterior sampling for initial state parameters
 * @details This module provides functions for sampling initial state parameters (theta_0k)
 *          from their conditional posterior distributions within Gibbs MCMC iterations,
 *          implementing Bayesian inference for polynomial dynamic models.
 * @author Michel H. Montoril
 * @date 2025-01-11
 * @version 1.0
 */

#ifndef CONDITIONAL_THETA0_H
#define CONDITIONAL_THETA0_H


/**
 * @brief Generates sample from conditional posterior of initial level state theta_01
 *        (local-level model)
 *
 * @details Implements Bayesian updating for the initial level state in local-level
 *          dynamic models. The conditional posterior is Normal, derived from combining
 *          a Normal prior with likelihood information from the first state element
 *          theta_1[0].
 *
 *          Model structure:
 *          theta_1[0] = theta_01 + u_{1,0},  u_{1,0} ~ N(0, W_1)
 *          theta_01 ~ N(mu_01, tau_01^{-1})
 *
 *          Conditional posterior derivation:
 *          tau_post = tau_01 + 1/W_1 (posterior precision)
 *          mu_post = (mu_01 * tau_01 + theta_1[0] / W_1) / tau_post (posterior mean)
 *          theta_01 | theta_1, W_1 ~ N(mu_post, tau_post^{-1})
 *
 * @param theta_01_post       Double array storing posterior samples of theta_01.
 *                            Updated at index `iter`
 * @param theta_1_post        Double array storing posterior samples of theta_1
 *                            (vectorized B*n matrix). Uses current iteration `iter`
 * @param prec_theta_1_post   Double array storing posterior samples of precision 1/W_1.
 *                            Uses current iteration `iter`
 * @param mean_theta_01       Double scalar, prior mean mu_01 for theta_01
 * @param prec_theta_01       Double scalar, prior precision tau_01 = 1/sigma_01^2
 *                            for theta_01
 * @param n                   Integer scalar, sample size (number of observations)
 * @param iter                Integer scalar, current MCMC iteration index (0-based)
 *
 * @note Computational complexity: O(1) constant time operation
 * @note Numerical stability: Uses sqrt() for standard deviation conversion from precision
 * @note Memory access: Accesses theta_1_post[iter*n] for first state element
 * @note Algorithm: Direct Normal-Normal conjugate updating with precision weighting
 *
 * @warning Assumes iter >= 0 and valid array bounds
 * @warning No validation of prior parameter positivity
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function
 *
 * @see generate_theta_01
 */
void generate_theta_01_locallevel(double *theta_01_post,
                                  double *theta_1_post,
                                  double *prec_theta_1_post,
                                  double mean_theta_01,
                                  double prec_theta_01,
                                  int n,
                                  int iter);


/**
 * @brief Generates sample from conditional posterior of initial level state theta_01
 *        (local-trend model)
 *
 * @details Implements Bayesian updating for the initial level state in local-trend
 *          dynamic models. The conditional posterior is Normal, derived from combining
 *          a Normal prior with likelihood information from the first state element
 *          theta_1[0] and trend theta_02.
 *
 *          Model structure:
 *          theta_1[0] = theta_01 + theta_02 + u_{1,0},  u_{1,0} ~ N(0, W_1)
 *          theta_01 ~ N(mu_01, tau_01^{-1})
 *
 *          Conditional posterior derivation:
 *          tau_post = tau_01 + 1/W_1 (posterior precision)
 *          mu_post = (mu_01 * tau_01 + (theta_1[0] - theta_02) / W_1) / tau_post
 *          theta_01 | theta_1, theta_02, W_1 ~ N(mu_post, tau_post^{-1})
 *
 * @param theta_01_post       Double array storing posterior samples of theta_01.
 *                            Updated at index `iter`
 * @param theta_02_post       Double array storing posterior samples of theta_02.
 *                            Uses current iteration `iter`
 * @param theta_1_post        Double array storing posterior samples of theta_1
 *                            (vectorized B*n matrix). Uses current iteration `iter`
 * @param prec_theta_1_post   Double array storing posterior samples of precision 1/W_1.
 *                            Uses current iteration `iter`
 * @param mean_theta_01       Double scalar, prior mean mu_01 for theta_01
 * @param prec_theta_01       Double scalar, prior precision tau_01 = 1/sigma_01^2
 *                            for theta_01
 * @param n                   Integer scalar, sample size (number of observations)
 * @param iter                Integer scalar, current MCMC iteration index (0-based)
 *
 * @note Computational complexity: O(1) constant time operation
 * @note Numerical stability: Uses sqrt() for standard deviation conversion from precision
 * @note Memory access: Accesses theta_1_post[iter*n] for first state element
 * @note Algorithm: Normal-Normal conjugate updating with trend adjustment
 *
 * @warning Assumes iter >= 0 and valid array bounds
 * @warning No validation of prior parameter positivity
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function
 *
 * @see generate_theta_01_locallevel
 * @see generate_theta_0k
 * @see generate_theta_0p
 */
void generate_theta_01(double *theta_01_post,
                       double *theta_02_post,
                       double *theta_1_post,
                       double *prec_theta_1_post,
                       double mean_theta_01,
                       double prec_theta_01,
                       int n,
                       int iter);


/**
 * @brief Generates sample from conditional posterior of intermediate initial state
 *        theta_0k (k=2,...,p-1)
 *
 * @details Implements Bayesian updating for intermediate initial states in polynomial
 *          dynamic models. The conditional posterior is Normal, derived from combining
 *          a Normal prior with likelihood information from adjacent state equations.
 *          Notation: km1 = k-1, kp1 = k+1
 *
 *          Model structure:
 *          theta_{k-1}[0] = theta_{0,k-1} + theta_0k + u_{k-1,0},  u_{k-1,0} ~ N(0, W_{k-1})
 *          theta_k[0] = theta_0k + theta_{0,k+1} + u_{k,0},       u_{k,0} ~ N(0, W_k)
 *          theta_0k ~ N(mu_0k, tau_0k^{-1})
 *
 *          Conditional posterior derivation:
 *          tau_post = tau_0k + 1/W_{k-1} + 1/W_k (posterior precision)
 *          mu_post = [mu_0k * tau_0k + (theta_{k-1}[0] - theta_{0,k-1}) / W_{k-1} +
 *                     (theta_k[0] - theta_{0,k+1}) / W_k] / tau_post
 *          theta_0k | ... ~ N(mu_post, tau_post^{-1})
 *
 * @param theta_0km1_post     Double array storing posterior samples of theta_{0,k-1}.
 *                            Uses previous iteration (`iter - 1`)
 * @param theta_0k_post       Double array storing posterior samples of theta_0k.
 *                            Updated at index `iter`
 * @param theta_0kp1_post     Double array storing posterior samples of theta_{0,k+1}.
 *                            Uses current iteration `iter`
 * @param theta_km1_post      Double array storing posterior samples of theta_{k-1}
 *                            (vectorized B*n matrix). Uses previous iteration (`iter - 1`)
 * @param theta_k_post        Double array storing posterior samples of theta_k
 *                            (vectorized B*n matrix). Uses current iteration `iter`
 * @param prec_theta_km1_post Double array storing posterior samples of precision 1/W_{k-1}.
 *                            Uses previous iteration (`iter - 1`)
 * @param prec_theta_k_post   Double array storing posterior samples of precision 1/W_k.
 *                            Uses current iteration `iter`
 * @param mean_theta_0k       Double scalar, prior mean mu_0k for theta_0k
 * @param prec_theta_0k       Double scalar, prior precision tau_0k = 1/sigma_0k^2
 *                            for theta_0k
 * @param n                   Integer scalar, sample size (number of observations)
 * @param iter                Integer scalar, current MCMC iteration index (0-based)
 *
 * @note Computational complexity: O(1) constant time operation
 * @note Numerical stability: Uses sqrt() for standard deviation conversion from precision
 * @note Memory access: Accesses first elements of state vectors from different iterations
 * @note Algorithm: Normal-Normal conjugate updating with multiple information sources
 * @note Notation: km1 = k-1, kp1 = k+1 for parameter naming
 *
 * @warning Assumes iter >= 1 for accessing previous iteration
 * @warning No validation of prior parameter positivity
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function
 *
 * @see generate_theta_01() for initial level state
 * @see generate_theta_0p() for final initial state
 * @see generate_precision_theta_k() for precision sampling
 */
void generate_theta_0k(double *theta_0km1_post,
                       double *theta_0k_post,
                       double *theta_0kp1_post,
                       double *theta_km1_post,
                       double *theta_k_post,
                       double *prec_theta_km1_post,
                       double *prec_theta_k_post,
                       double mean_theta_0k,
                       double prec_theta_0k,
                       int n,
                       int iter);


/**
 * @brief Generates sample from conditional posterior of final initial state theta_0p
 *        (k=p)
 *
 * @details Implements Bayesian updating for the final initial state in polynomial
 *          dynamic models. The conditional posterior is Normal, derived from combining
 *          a Normal prior with likelihood information from the state equations. This is
 *          a boundary case handling the highest-order polynomial component.
 *          Notation: pm1 = p-1
 *
 *          Model structure:
 *          theta_{p-1}[0] = theta_{0,p-1} + theta_0p + u_{p-1,0},  u_{p-1,0} ~ N(0, W_{p-1})
 *          theta_p[0] = theta_0p + u_{p,0},                        u_{p,0} ~ N(0, W_p)
 *          theta_0p ~ N(mu_0p, tau_0p^{-1})
 *
 *          Conditional posterior derivation:
 *          tau_post = tau_0p + 1/W_{p-1} + 1/W_p (posterior precision)
 *          mu_post = [mu_0p * tau_0p + (theta_{p-1}[0] - theta_{0,p-1}) / W_{p-1} +
 *                     theta_p[0] / W_p] / tau_post
 *          Note: theta_p equation doesn't involve higher-order terms (boundary case)
 *          theta_0p | ... ~ N(mu_post, tau_post^{-1})
 *
 * @param theta_0pm1_post     Double array storing posterior samples of theta_{0,p-1}.
 *                            Uses previous iteration (`iter - 1`)
 * @param theta_0p_post       Double array storing posterior samples of theta_0p.
 *                            Updated at index `iter`
 * @param theta_pm1_post      Double array storing posterior samples of theta_{p-1}
 *                            (vectorized B*n matrix). Uses previous iteration (`iter - 1`)
 * @param theta_p_post        Double array storing posterior samples of theta_p
 *                            (vectorized B*n matrix). Uses current iteration `iter`
 * @param prec_theta_pm1_post Double array storing posterior samples of precision 1/W_{p-1}.
 *                            Uses previous iteration (`iter - 1`)
 * @param prec_theta_p_post   Double array storing posterior samples of precision 1/W_p.
 *                            Uses current iteration `iter`
 * @param mean_theta_0p       Double scalar, prior mean mu_0p for theta_0p
 * @param prec_theta_0p       Double scalar, prior precision tau_0p = 1/sigma_0p^2
 *                            for theta_0p
 * @param n                   Integer scalar, sample size (number of observations)
 * @param iter                Integer scalar, current MCMC iteration index (0-based)
 *
 * @note Computational complexity: O(1) constant time operation
 * @note Numerical stability: Uses sqrt() for standard deviation conversion from precision
 * @note Memory access: Accesses first elements of state vectors from different iterations
 * @note Algorithm: Normal-Normal conjugate updating with boundary case handling
 * @note Notation: pm1 = p-1 for parameter naming
 *
 * @warning Assumes iter >= 1 for accessing previous iteration
 * @warning No validation of prior parameter positivity
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function
 *
 * @see generate_theta_01
 * @see generate_theta_0k
 * @see generate_precision_theta_p
 */
void generate_theta_0p(double *theta_0pm1_post,
                       double *theta_0p_post,
                       double *theta_pm1_post,
                       double *theta_p_post,
                       double *prec_theta_pm1_post,
                       double *prec_theta_p_post,
                       double mean_theta_0p,
                       double prec_theta_0p,
                       int n,
                       int iter);

#endif /* CONDITIONAL_THETA0_H */
