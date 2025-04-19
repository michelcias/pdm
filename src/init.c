#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

/**
 * Registration of native routines for the 'pdm' package.
 *
 * The CallEntries array defines the mapping between R-callable routine names and
 * their corresponding C functions and argument counts.
 */

/* Declaration of the existing wrapper in utils.c */
SEXP C_ILogit(SEXP x);

/* Declaration of the new MCMC local-trend sampler */
SEXP C_MCMC_localtrend(
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

static const R_CallMethodDef CallEntries[] = {
  /* existing ilogit wrapper */
  {"_pdm_ilogit",           (DL_FUNC) &C_ILogit,               1},
  /* new Gibbs sampler for local-trend */
  {"_pdm_mcmc_localtrend",  (DL_FUNC) &C_MCMC_localtrend,      10},
  {NULL, NULL, 0}
};

/**
 * Initializes the native routine registration for the pdm package.
 *
 * Should be called automatically by R when the package is loaded.
 *
 * Args:
 *   dll (DllInfo*): Pointer to DllInfo provided by R.
 */
void R_init_pdm(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
