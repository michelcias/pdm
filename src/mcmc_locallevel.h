#ifndef MCMC_LOCALLEVEL_H
#define MCMC_LOCALLEVEL_H

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>

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
 *   y_t         = theta_{t,1} + e_t,        e_t ~ N(0, V)
 *   theta_{t,1} = theta_{t-1,1} + u_{t,1},  u_{t,1} ~ N(0, W_1)
 *
 * Burn‐in and thinning are applied so that exactly n_chain posterior draws are returned.
 *
 * @param y                    Numeric vector of observations (length = n).
 * @param burnin               Integer, number of burn‐in iterations.
 * @param thinning             Integer, thinning interval.
 * @param n_chain              Integer, number of retained posterior samples.
 * @param prior_theta01_mean   Double, prior mean for theta_01.
 * @param prior_theta01_prec   Double, prior precision (1/variance) for theta_{0,1}.
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
SEXP C_MCMC_locallevel(
    SEXP y_,
    SEXP burnin_,
    SEXP thinning_,
    SEXP n_chain_,
    SEXP prior_theta01_mean_,
    SEXP prior_theta01_prec_,
    SEXP prior_prec1_shape_,
    SEXP prior_prec1_rate_,
    SEXP prior_prec_y_shape_,
    SEXP prior_prec_y_rate_
);

#endif /* MCMC_LOCALLEVEL_H */
