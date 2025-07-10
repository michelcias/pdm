#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "conditional_state.h"
#include "conditional_precision.h"
#include "conditional_theta0.h"
#include "mcmc_locallevel.h"

/**
 * C_MCMC_locallevel: Gibbs sampler for a local-level dynamic model (p = 1).
 *
 * This function runs a Gibbs MCMC for the polynomial dynamic model with a simple
 * local level structure, sampling parameters in the following order:
 *   1) state vector      — generate_theta_1_locallevel
 *   2) innovation prec.  — generate_precision_theta_p  (1/W_1)
 *   3) initial state     — generate_theta_01_locallevel
 *   4) data precision    — generate_precision_data     (1/V)
 *
 * The model is:
 *   y_t        = theta_{t1} + e_t,           e_t    ~ N(0, V)
 *   theta_{t1} = theta_{(t-1)1} + u_{t1},    u_{t1} ~ N(0, W_1)
 *
 * Burn‐in and thinning are applied so that exactly n_chain posterior draws are returned.
 *
 * @param y                    Numeric vector of observations (length = n).
 * @param burnin               Integer, number of burn‐in iterations.
 * @param thinning             Integer, thinning interval.
 * @param n_chain              Integer, number of retained posterior samples.
 * @param prior_theta01_mean   Double, prior mean for theta_01.
 * @param prior_theta01_prec   Double, prior precision (1/variance) for theta_01.
 * @param prior_prec1_shape    Double, shape parameter of Gamma prior for 1/W_1.
 * @param prior_prec1_rate     Double, rate  parameter of Gamma prior for 1/W_1.
 * @param prior_prec_y_shape   Double, shape parameter of Gamma prior for 1/V.
 * @param prior_prec_y_rate    Double, rate  parameter of Gamma prior for 1/V.
 *
 * @return An R list with components:
 *   $theta_1  — numeric matrix [n_chain × n] of state samples
 *   $theta_01 — numeric vector [length = n_chain] of initial state samples
 *   $prec_1   — numeric vector [length = n_chain] of innovation precisions
 *   $prec_y   — numeric vector [length = n_chain] of data precisions
 */
SEXP C_MCMC_locallevel(SEXP y_, SEXP burnin_, SEXP thinning_, SEXP n_chain_,
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
            y, theta_1_post, prec_y_post, prec_1_post,
            theta_01_post, n, ii
        );

        /* 2) Sample innovation precision 1/W_1 */
        generate_precision_theta_p(
            theta_01_post,    /* theta_0p */
            theta_1_post,     /* theta_p */
            prec_1_post,      /* output precision 1/W_p */
            nu_01, eta_01,
            n, ii
        );

        /* 3) Sample initial state theta_01 */
        generate_theta_01_locallevel(
            theta_01_post, theta_1_post, prec_1_post,
            mean_theta01, prec_theta01,
            n, ii
        );

        /* 4) Sample data precision 1/V */
        generate_precision_data(
            y, theta_1_post, prec_y_post,
            nu_y, eta_y,
            n, ii
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
