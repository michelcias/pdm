/**
 * @file conditional_precision.h
 * @brief Header for conditional posterior sampling of precision parameters
 * @details Function declarations for efficient conjugate Bayesian updating algorithms
 *          for precision parameters in polynomial dynamic models, utilizing
 *          Gamma-Normal conjugacy for fast sampling within Gibbs MCMC iterations.
 * @author Michel H. Montoril
 * @date 2025-08-12
 * @version 1.0
 */

#ifndef CONDITIONAL_PRECISION_H
#define CONDITIONAL_PRECISION_H

/**
 * @brief Generates sample from conditional posterior of observation precision 1/V
 *
 * @details Implements conjugate Bayesian updating for observation noise precision in
 *          dynamic models. Uses Gamma-Normal conjugacy where a Gamma prior on precision
 *          combines with Normal likelihood to yield a Gamma posterior.
 *
 *          Model structure:
 *          y_t = theta_{t,1} + e_t,  e_t ~ N(0, V)
 *          1/V ~ Gamma(nu_y, eta_y)
 *
 *          Conjugate updating formulas:
 *          nu_post = nu_y + n/2 (posterior shape)
 *          eta_post = eta_y + (1/2) * sum((y_t - theta_{t,1})^2) (posterior rate)
 *          1/V | y, theta_1 ~ Gamma(nu_post, eta_post)
 *
 * @param y              Double array of observed data [n]. Input observations
 * @param theta_1_post   Double array storing posterior samples of theta_1
 *                       (vectorized B*n matrix). Uses current iteration
 * @param prec_y_post    Double array storing posterior samples of precision 1/V.
 *                       Updated at index `iter`
 * @param nu_y           Double scalar, prior shape parameter nu_y for Gamma
 *                       distribution of 1/V
 * @param eta_y          Double scalar, prior rate parameter eta_y for Gamma
 *                       distribution of 1/V
 * @param n              Integer scalar, sample size (number of observations)
 * @param iter           Integer scalar, current MCMC iteration index (0-based)
 *
 * @note Computational complexity: O(n) for sum of squared residuals calculation
 * @note Numerical stability: Uses squared residuals accumulation with double precision
 * @note Memory access: Sequential access to y[0:n-1] and theta_1_post[iter*n:(iter+1)*n-1]
 * @note Algorithm: Direct Gamma-Normal conjugate updating with residual sum calculation
 *
 * @warning Assumes iter >= 0 and valid array bounds
 * @warning No validation of prior parameter positivity (nu_y > 0, eta_y > 0)
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function
 *
 * @see rgamma
 * @see generate_precision_theta_k
 * @see generate_precision_theta_p
 */
void generate_precision_data(double *y,
                             double *theta_1_post,
                             double *prec_y_post,
                             double nu_y,
                             double eta_y,
                             int n,
                             int iter);
/**
 * @brief Generates sample from conditional posterior of innovation precision 1/W_k
 *        (k=1,...,p-1)
 *
 * @details Implements conjugate Bayesian updating for state innovation precision in
 *          polynomial dynamic models. Uses Gamma-Normal conjugacy for intermediate
 *          state components that depend on both previous and next state levels.
 *          Notation: kp1 = k+1
 *
 *          Model structure:
 *          theta_{t,k} = theta_{t-1,k} + theta_{t-1,k+1} + u_{t,k},  u_{t,k} ~ N(0, W_k)
 *          1/W_k ~ Gamma(nu_k, eta_k)
 *
 *          Innovation calculation:
 *          For t=1: u_{1,k} = theta_{1,k} - theta_{0,k} - theta_{0,k+1}
 *          For t>1: u_{t,k} = theta_{t,k} - theta_{t-1,k} - theta_{t-1,k+1}
 *
 *          Conjugate updating formulas:
 *          nu_post = nu_k + n/2 (posterior shape)
 *          eta_post = eta_k + (1/2) * sum(u_{t,k}^2) (posterior rate)
 *          1/W_k | theta_k, theta_{k+1}, theta_{0k}, theta_{0,k+1} ~ Gamma(nu_post, eta_post)
 *
 * @param theta_0k_post       Double array storing posterior samples of theta_0k.
 *                            Uses previous iteration (`iter - 1`)
 * @param theta_0kp1_post     Double array storing posterior samples of theta_{0,k+1}.
 *                            Uses current iteration
 * @param theta_k_post        Double array storing posterior samples of theta_k
 *                            (vectorized B*n matrix). Uses current iteration
 * @param theta_kp1_post      Double array storing posterior samples of theta_{k+1}
 *                            (vectorized B*n matrix). Uses current iteration
 * @param prec_theta_k_post   Double array storing posterior samples of precision 1/W_k.
 *                            Updated at index `iter`
 * @param nu_0k               Double scalar, prior shape parameter nu_k for Gamma
 *                            distribution of 1/W_k
 * @param eta_0k              Double scalar, prior rate parameter eta_k for Gamma
 *                            distribution of 1/W_k
 * @param n                   Integer scalar, sample size (number of observations)
 * @param iter                Integer scalar, current MCMC iteration index (0-based)
 *
 * @note Computational complexity: O(n) for sum of squared innovations calculation
 * @note Numerical stability: Uses squared residuals accumulation with double precision
 * @note Memory access: Sequential access to state vectors with proper iteration offsets
 * @note Algorithm: Gamma-Normal conjugate updating with state difference calculations
 * @note Notation: kp1 = k+1 for parameter naming
 *
 * @warning Assumes iter >= 1 for accessing previous iteration elements
 * @warning No validation of prior parameter positivity (nu_0k > 0, eta_0k > 0)
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function
 *
 * @see generate_precision_data
 * @see generate_precision_theta_p
 * @see rgamma
 */
void generate_precision_theta_k(double *theta_0k_post,
                                double *theta_0kp1_post,
                                double *theta_k_post,
                                double *theta_kp1_post,
                                double *prec_theta_k_post,
                                double nu_0k,
                                double eta_0k,
                                int n,
                                int iter);

/**
 * @brief Generates sample from conditional posterior of final innovation precision 1/W_p
 *        (k=p)
 *
 * @details Implements conjugate Bayesian updating for the highest-order state innovation
 *          precision in polynomial dynamic models. This is a boundary case for the final
 *          component following a random walk structure.
 *
 *          Model structure:
 *          theta_{t,p} = theta_{t-1,p} + u_{t,p},  u_{t,p} ~ N(0, W_p)
 *          1/W_p ~ Gamma(nu_p, eta_p)
 *
 *          Innovation calculation (random walk):
 *          For t=1: u_{1,p} = theta_{1,p} - theta_{0,p}
 *          For t>1: u_{t,p} = theta_{t,p} - theta_{t-1,p}
 *
 *          Conjugate updating formulas:
 *          nu_post = nu_p + n/2 (posterior shape)
 *          eta_post = eta_p + (1/2) * sum(u_{t,p}^2) (posterior rate)
 *          1/W_p | theta_p, theta_{0p} ~ Gamma(nu_post, eta_post)
 *
 * @param theta_0p_post       Double array storing posterior samples of theta_0p.
 *                            Uses previous iteration (`iter - 1`)
 * @param theta_p_post        Double array storing posterior samples of theta_p
 *                            (vectorized B*n matrix). Uses current iteration
 * @param prec_theta_p_post   Double array storing posterior samples of precision 1/W_p.
 *                            Updated at index `iter`
 * @param nu_0p               Double scalar, prior shape parameter nu_p for Gamma
 *                            distribution of 1/W_p
 * @param eta_0p              Double scalar, prior rate parameter eta_p for Gamma
 *                            distribution of 1/W_p
 * @param n                   Integer scalar, sample size (number of observations)
 * @param iter                Integer scalar, current MCMC iteration index (0-based)
 *
 * @note Computational complexity: O(n) for sum of squared innovations calculation
 * @note Numerical stability: Uses squared residuals accumulation with double precision
 * @note Memory access: Sequential access to theta_p vector with iteration offset
 * @note Algorithm: Gamma-Normal conjugate updating for random walk innovations
 *
 * @warning Assumes iter >= 1 for accessing previous iteration elements
 * @warning No validation of prior parameter positivity (nu_0p > 0, eta_0p > 0)
 * @warning Requires GetRNGstate()/PutRNGstate() bracket in calling function
 *
 * @see generate_precision_data
 * @see generate_precision_theta_k
 * @see rgamma
 */
void generate_precision_theta_p(double *theta_0p_post,
                                double *theta_p_post,
                                double *prec_theta_p_post,
                                double nu_0p,
                                double eta_0p,
                                int n,
                                int iter);

#endif /* CONDITIONAL_PRECISION_H */
