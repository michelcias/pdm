/**
 * @file conditional_state.h
 * @brief Header for conditional posterior sampling of state parameters in Gaussian models
 * @author Michel H. Montoril
 * @date 2025-10-11
 * @version 1.1
 *
 * @details This header declares optimized functions for Bayesian state parameter sampling
 *          in polynomial dynamic models with Gaussian observation equations. All functions
 *          utilize multivariate Normal distributions with tridiagonal precision matrices
 *          for efficient O(n) sampling complexity.
 *
 *          **Key features:**
 *          - Level state sampling (theta_1) for local level models
 *          - Level state sampling (theta_1) for local trend models
 *          - Intermediate state sampling (theta_k, k=2,...,p-1) for polynomial components
 *          - Final state sampling (theta_p) for highest-order component (random walk)
 *          - Memory-efficient direct vector operations (no array indexing)
 *          - Const-qualified input arrays for safety and optimization
 *          - O(n) computational complexity via tridiagonal Cholesky decomposition
 *
 *          **Mathematical framework:**
 *          All state vectors follow conditional multivariate Normal posteriors with
 *          tridiagonal precision matrices arising from the Markovian structure of
 *          polynomial dynamic models. The general form is:
 *          theta | data, params ~ MVN(mu_post, Sigma_post)
 *          where Sigma_post^{-1} has tridiagonal structure exploited for efficient sampling.
 *
 *          **Tridiagonal precision structure:**
 *          The precision matrices have the form:
 *          - Diagonal elements: combination of observation and innovation precisions
 *          - Off-diagonal elements: -innovation_precision
 *          - Boundary adjustments: first and last diagonal elements may differ
 *
 *          This structure enables O(n) sampling via specialized Cholesky decomposition
 *          instead of O(n³) for general covariance matrices.
 *
 * @changelog
 * - v1.1 (2025-10-11): Refactored function signatures to use scalar parameters and
 *   direct vector operations. Removed iteration index parameters. Added const qualifiers
 *   for input arrays. Updated to use simplified generate_normal_vector signature.
 *   Enhanced documentation for memory-efficient interface.
 * - v1.0 (2025-08-12): Initial header declaration with array-based interface.
 */

#ifndef CONDITIONAL_STATE_H
#define CONDITIONAL_STATE_H

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
                                 int           n);

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
                      int           n);

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
                      int           n);

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
                      int           n);

#endif /* CONDITIONAL_STATE_H */
