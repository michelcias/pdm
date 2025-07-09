#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

/* Include headers for all registered functions */
#include "mcmc_locallevel.h"
#include "mcmc_localtrend.h"
#include "utils.h"

/* .Call definitions:
 *   - First argument: name of the function in R, passed to .Call()
 *   - Second argument: a pointer to the C function
 *   - Third argument: the number of arguments of the C function
 */
static const R_CallMethodDef CallEntries[] = {
  {"_pdm_C_ILogit",           (DL_FUNC) &C_ILogit,           1},
  {"_pdm_C_MCMC_locallevel",  (DL_FUNC) &C_MCMC_locallevel,  10},
  {"_pdm_C_MCMC_localtrend",  (DL_FUNC) &C_MCMC_localtrend,  14},
  {NULL, NULL, 0}
};

void R_init_pdm(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
