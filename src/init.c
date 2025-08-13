/**
 * @file init.c
 * @brief R package initialization and function registration
 * @details Handles dynamic loading and registration of C functions for the pdm package.
 *          Registers .Call entry points for MCMC algorithms and utility functions,
 *          ensuring proper interface between R and C code. Implements security
 *          measures by disabling dynamic symbol lookup.
 * @author Michel H. Montoril
 * @date 2025-08-12
 * @version 1.0
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

/* Include headers for all registered functions */
#include "mcmc_locallevel.h"
#include "mcmc_localtrend.h"
#include "mcmc_localacceleration.h"
#include "mcmc_binomial_locallevel.h"
#include "utils.h"

/**
 * @brief Static table defining .Call method entries for R-C interface
 *
 * @details Maps R function names to their corresponding C implementations with
 *          argument counts. This table is used by R's dynamic loading system
 *          to properly route .Call() invocations to the correct C functions.
 *
 *          Registration structure:
 *          - First element: R function name (string used in .Call())
 *          - Second element: C function pointer (DL_FUNC cast)
 *          - Third element: Number of expected arguments
 *          - Final element: NULL terminator for array
 *
 *          Registered functions:
 *          - C_ILogit: Inverse logit transformation utility (1 argument)
 *          - C_MCMC_locallevel: Local level model MCMC sampler (10 arguments)
 *          - C_MCMC_localtrend: Local trend model MCMC sampler (14 arguments)
 *          - C_MCMC_localacceleration: Local acceleration model MCMC sampler
 *            (18 arguments)
 *          - C_MCMC_logit_binomial_locallevel: Binomial local level model MCMC
 *            sampler (16 arguments)
 *
 * @note Function pointers must be cast to DL_FUNC for R compatibility
 * @note Argument counts are enforced by R's .Call() mechanism
 * @note NULL terminator is required for proper array traversal
 * @note Names must match exactly those used in R code .Call() invocations
 *
 * @warning Modifying this table requires corresponding changes in R wrapper functions
 * @warning Incorrect argument counts will cause runtime errors in R
 *
 * @see R_registerRoutines
 * @see DL_FUNC
 * @see R_CallMethodDef
 */
static const R_CallMethodDef CallEntries[] = {
  {"_pdm_C_ILogit",                        (DL_FUNC) &C_ILogit,                1},
  {"_pdm_C_MCMC_locallevel",               (DL_FUNC) &C_MCMC_locallevel,      10},
  {"_pdm_C_MCMC_localtrend",               (DL_FUNC) &C_MCMC_localtrend,      14},
  {"_pdm_C_MCMC_localacceleration",        (DL_FUNC) &C_MCMC_localacceleration, 18},
  {"_pdm_C_MCMC_logit_binomial_locallevel",(DL_FUNC) &C_MCMC_logit_binomial_locallevel, 16},
   {NULL, NULL, 0}
};

/**
 * @brief Package initialization function called when pdm package is loaded
 *
 * @details Mandatory initialization function for R packages with compiled code.
 *          Registers all C functions with R's dynamic loading system and configures
 *          security settings for symbol resolution. This function is automatically
 *          called by R when the package is loaded via library() or require().
 *
 *          Initialization sequence:
 *          1. Register .Call entry points from CallEntries table
 *          2. Disable dynamic symbol lookup for security
 *          3. Return control to R loading system
 *
 *          Security implications:
 *          R_useDynamicSymbols(dll, FALSE) prevents external packages from
 *          accessing internal symbols, improving package encapsulation and
 *          reducing potential conflicts or security vulnerabilities.
 *
 * @param dll Pointer to DllInfo structure containing package loading information.
 *            Provided automatically by R's loading system.
 *
 * @note This function name must be exactly "R_init_<packagename>" for R to find it
 * @note Called automatically during package loading - not intended for manual invocation
 * @note Disabling dynamic symbols is a recommended security practice
 * @note All .Call functions must be registered here to be accessible from R
 *
 * @warning Do not call this function manually
 * @warning Modifying registration without updating R code will break package functionality
 * @warning Function must be exported in package's NAMESPACE
 *
 * @see R_registerRoutines
 * @see R_useDynamicSymbols
 * @see DllInfo
 */
void R_init_pdm(DllInfo *dll) {
  // Register .Call entry points for C functions accessible from R
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);

  // Disable dynamic symbol lookup for improved security and encapsulation
  R_useDynamicSymbols(dll, FALSE);
}
