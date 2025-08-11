/**
 * @file mcmc_locallevel.c
 * @brief Implementation of MCMC sampling for local-level dynamic models
 * @details Provides a complete Gibbs sampler for Bayesian estimation of polynomial
 *          dynamic models with local-level structure, utilizing conditional posterior
 *          distributions for efficient sampling of model parameters and states.
 * @author Michel H. Montoril
 * @date 2025-08-11
 * @version 1.0
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "conditional_state.h"
#include "conditional_precision.h"
#include "conditional_theta0.h"
#include "mcmc_locallevel.h"

/**
 * @brief Gibbs sampler for local-level dynamic model with polynomial structure
 *
 * @details Implements a complete Gibbs MCMC algorithm for the local-level dynamic model:
 *
 *          Observation equation:
 *          y_t = theta_{t,1} + e_t,     e_t ~ N(0, V)
 *
 *          State equation:
 *          theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *
 *          The algorithm employs the Markovian structure of polynomial dynamic models
 *          to enable efficient block sampling of the state vector through forward-backward
 *          recursions. Innovation and observation precisions are sampled from their
 *          conjugate Gamma posteriors.
 *
 *          Sampling sequence per iteration:
 *          1. theta_1 | y, theta_0, W_1, V -> Forward-Backward algorithm
 *          2. 1/W_1 | theta_1, theta_0 -> Gamma posterior
 *          3. theta_{0,1} | theta_1, W_1 -> Normal posterior
 *          4. 1/V | y, theta_1 -> Gamma posterior
 *
 *          Total iterations computed as: burnin + (n_chain - 1) x thinning + 1
 *
 * @param y                    SEXP Numeric vector of observed time series data [length n]
 * @param burnin               SEXP Integer scalar, number of burn-in iterations to discard for chain convergence
 * @param thinning             SEXP Integer scalar, thinning interval to reduce autocorrelation in samples
 * @param n_chain              SEXP Integer scalar, target number of retained posterior samples
 * @param prior_theta01_mean   SEXP Double scalar, prior mean mu_0 for initial state theta_{0,1}
 * @param prior_theta01_prec   SEXP Double scalar, prior precision tau_0 = 1/sigma_0^2 for initial state
 * @param prior_prec1_shape    SEXP Double scalar, shape parameter nu_1 for Gamma(nu_1, eta_1) prior on 1/W_1
 * @param prior_prec1_rate     SEXP Double scalar, rate parameter eta_1 for Gamma(nu_1, eta_1) prior on 1/W_1
 * @param prior_prec_y_shape   SEXP Double scalar, shape parameter nu_y for Gamma(nu_y, eta_y) prior on 1/V
 * @param prior_prec_y_rate    SEXP Double scalar, rate parameter eta_y for Gamma(nu_y, eta_y) prior on 1/V
 *
 * @return SEXP R list containing posterior samples with named components:
 *         - theta_1: Numeric matrix [n_chain x n] of complete state trajectory samples
 *         - theta_01: Numeric vector [n_chain] of initial state theta_{0,1} samples
 *         - prec_1: Numeric vector [n_chain] of innovation precision 1/W_1 samples
 *         - prec_y: Numeric vector [n_chain] of observation precision 1/V samples
 *
 * @note Computational complexity: O(n_iter x n) for n_iter total iterations
 * @note Memory requirements: O(n_iter x n) for temporary trajectory storage
 * @note RNG management: Proper GetRNGstate()/PutRNGstate() bracket for R integration
 * @note Initialization: Uses prior-based random initialization for all parameters
 *
 * @warning Minimum sample size n >= 3 enforced for numerical stability of recursions
 * @warning Integer overflow protection: n <= INT_MAX due to R's integer limitations
 * @warning Memory allocation failures will terminate R session via R_Calloc errors
 * @warning No input validation for prior hyperparameters; negative values may cause crashes
 *
 * @see generate_theta_1_locallevel
 * @see generate_precision_theta_p
 * @see generate_theta_01_locallevel
 * @see generate_precision_data
 */SEXP C_MCMC_locallevel(SEXP y_, SEXP burnin_, SEXP thinning_, SEXP n_chain_,
                       SEXP prior_theta01_mean_, SEXP prior_theta01_prec_,
                       SEXP prior_prec1_shape_, SEXP prior_prec1_rate_,
                       SEXP prior_prec_y_shape_, SEXP prior_prec_y_rate_) {
    /* Parse data vector and check its length */
    double   *y    = REAL(y_);
    R_xlen_t  len  = LENGTH(y_);
    if (len < 3)
        Rf_error("C_MCMC_locallevel: sample size 'n' must be at least 3, got %lld",
                 (long long) len);
    if (len > INT_MAX)
        Rf_error("C_MCMC_locallevel: sample size too large (%lld > %d)",
                 (long long) len, INT_MAX);
    int n = (int) len;

    /* Parse MCMC settings */
    int burnin   = INTEGER(burnin_)[0];
    int thinning = INTEGER(thinning_)[0];
    int n_chain  = INTEGER(n_chain_)[0];
    /* Use alternative formula to run just enough iters */
    int n_iter   = burnin + (n_chain - 1) * thinning + 1;

    /* Parse priors */
    double mean_theta01 = REAL(prior_theta01_mean_)[0];
    double prec_theta01 = REAL(prior_theta01_prec_)[0];
    double nu_01        = REAL(prior_prec1_shape_)[0];
    double eta_01       = REAL(prior_prec1_rate_)[0];
    double nu_y         = REAL(prior_prec_y_shape_)[0];
    double eta_y        = REAL(prior_prec_y_rate_)[0];

    /* Allocate storage for posterior samples */
    SEXP theta_1_samples  = PROTECT(allocMatrix(REALSXP, n_chain, n));
    SEXP theta_01_samples = PROTECT(allocVector(REALSXP, n_chain));
    SEXP prec_1_samples   = PROTECT(allocVector(REALSXP, n_chain));
    SEXP prec_y_samples   = PROTECT(allocVector(REALSXP, n_chain));

    /* Buffers for full MCMC trajectory (including burn‐in) */
    double *theta_1_post  = (double *) R_Calloc(n_iter * n, double);
    double *theta_01_post = (double *) R_Calloc(n_iter,     double);
    double *prec_1_post   = (double *) R_Calloc(n_iter,     double);
    double *prec_y_post   = (double *) R_Calloc(n_iter,     double);

    /* Initialize RNG state */
    GetRNGstate();

    /*--- INITIALIZATION (iter = 0) ---*/
    theta_01_post[0] = rnorm(mean_theta01,     sqrt(1.0 / prec_theta01));
    prec_1_post[0]   = rgamma(nu_01,           1.0 / eta_01);
    prec_y_post[0]   = rgamma(nu_y,            1.0 / eta_y);
    double init_sd   = sqrt(1.0 / prec_1_post[0]);
    theta_1_post[0]  = rnorm(theta_01_post[0], init_sd);
    for (int j = 1; j < n; j++) {
        theta_1_post[j] = rnorm(theta_1_post[j - 1], init_sd);
    }

    /*--- Main Gibbs sampling loop ---*/
    int chain = 0;
    for (int ii = 1; ii < n_iter; ii++) {
        /* 1) Sample state vector theta_1 */
        generate_theta_1_locallevel(
            y,
            theta_1_post,
            prec_y_post,
            prec_1_post,
            theta_01_post,
            n,
            ii
        );

        /* 2) Sample innovation precision 1/W_1 */
        generate_precision_theta_p(
            theta_01_post,    /* theta_0p */
            theta_1_post,     /* theta_p */
            prec_1_post,      /* output precision 1/W_p */
            nu_01, eta_01,
            n,
            ii
        );

        /* 3) Sample initial state theta_01 */
        generate_theta_01_locallevel(
            theta_01_post,
            theta_1_post,
            prec_1_post,
            mean_theta01,
            prec_theta01,
            n,
            ii
        );

        /* 4) Sample data precision 1/V */
        generate_precision_data(
            y,
            theta_1_post,
            prec_y_post,
            nu_y,
            eta_y,
            n,
            ii
        );

        /* Store post‐burn‐in draws, applying thinning */
        if (ii >= burnin && ((ii - burnin) % thinning) == 0) {
            int idx = chain++;
            for (int j = 0; j < n; j++) {
                REAL(theta_1_samples)[idx + j * n_chain] =
                    theta_1_post[ii * n + j];
            }
            REAL(theta_01_samples)[idx] = theta_01_post[ii];
            REAL(prec_1_samples)[idx]   = prec_1_post[ii];
            REAL(prec_y_samples)[idx]   = prec_y_post[ii];
        }
    }

    /* Return RNG state */
    PutRNGstate();

    /* Free temporary buffers */
    R_Free(theta_1_post);
    R_Free(theta_01_post);
    R_Free(prec_1_post);
    R_Free(prec_y_post);

    /* Package results into a named list */
    SEXP out = PROTECT(allocVector(VECSXP, 4));
    SET_VECTOR_ELT(out, 0, theta_1_samples);
    SET_VECTOR_ELT(out, 1, theta_01_samples);
    SET_VECTOR_ELT(out, 2, prec_1_samples);
    SET_VECTOR_ELT(out, 3, prec_y_samples);

    SEXP nms = PROTECT(allocVector(STRSXP, 4));
    SET_STRING_ELT(nms, 0, mkChar("theta_1"));
    SET_STRING_ELT(nms, 1, mkChar("theta_01"));
    SET_STRING_ELT(nms, 2, mkChar("prec_1"));
    SET_STRING_ELT(nms, 3, mkChar("prec_y"));
    setAttrib(out, R_NamesSymbol, nms);

    UNPROTECT(6);
    return out;
}
