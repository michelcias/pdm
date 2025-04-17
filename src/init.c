#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

/**
 * Registration of native routines for the 'pdm' package.
 *
 * The CallEntries array defines the mapping between R-callable routine names and
 * their corresponding C functions and argument counts.
 */
SEXP C_ILogit(SEXP x); // Declaration of the wrapper implemented in utils.c

static const R_CallMethodDef CallEntries[] = {
  {"_pdm_ilogit", (DL_FUNC) &C_ILogit, 1},
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
