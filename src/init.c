/**
 * @file init.c
 * @brief R package initialization and function registration
 * @details Handles dynamic loading and registration of C functions for the pdm package.
 *          Registers .Call entry points for MCMC algorithms and utility functions,
 *          ensuring proper interface between R and C code. Implements security
 *          measures by disabling dynamic symbol lookup.
 * @author Michel H. Montoril
 * @date 2025-10-23
 * @version 1.5
 *
 * @changelog
 * - v1.5 (2025-10-23): Added registration for Gaussian mixture model with dynamic
 *     weights and local trend structure.
 *     C_MCMC_normal_mixture_localtrend: 30 args
 * - v1.4 (2025-10-15): Updated argument counts for binomial MCMC functions to
 *     include new verbose and bar_width parameters for progress bar support.
 *     C_MCMC_logit_binomial_localtrend: 21 -> 23 args
 *     C_MCMC_probit_bernoulli_localtrend: 12 -> 14 args
 * - v1.3 (2025-10-13): Added missing includes for probit-Bernoulli functions,
 *     corrected documentation to reflect all registered test helpers, and
 *     updated version metadata to match current development state.
 * - v1.2 (2025-10-05): Updated the table of registered methods to include
 *     the test helper for the probit Bernoulli sampler with fixed parameters,
 *     and revised the associated documentation.
 * - v1.1 (2025-09-23): Added registration for optimized adaptive MCMC functions
 *     including adapt_cwmh_parameters with threshold parameter and legacy wrapper
 *     for backward compatibility.
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

/* Include headers for all registered production functions */
#include "mcmc_normal_locallevel.h"
#include "mcmc_normal_localtrend.h"
#include "mcmc_normal_localacceleration.h"
#include "mcmc_normal_mixture_localtrend.h"
#include "mcmc_binomial_locallevel.h"
#include "mcmc_binomial_localtrend.h"
#include "mcmc_binomial_localacceleration.h"
#include "test_helpers.h"
#include "utils.h"

//==============================================================================
// METHOD REGISTRATION TABLE
//==============================================================================

/**
 * @brief Static table defining .Call method entries for R-C interface
 *
 * @details Maps R function names to their corresponding C implementations with
 *          argument counts. This table is used by R's dynamic loading system
 *          to properly route .Call() invocations to the correct C functions.
 *
 *          **Registered Functions:**
 *
 *          **Main MCMC Functions (Production):**
 *          - C_MCMC_normal_locallevel: Gaussian local level model (12 args)
 *          - C_MCMC_normal_localtrend: Gaussian local trend model (16 args)
 *          - C_MCMC_normal_localacceleration: Gaussian local acceleration model (20 args)
 *          - C_MCMC_normal_mixture_localtrend: Gaussian mixture with dynamic weights (30 args)
 *          - C_MCMC_logit_binomial_locallevel: Binomial local level with logit link (19 args)
 *          - C_MCMC_logit_binomial_localtrend: Binomial local trend with logit link (23 args)
 *          - C_MCMC_logit_binomial_localacceleration: Binomial local acceleration with logit (27 args)
 *          - C_MCMC_probit_bernoulli_locallevel: Bernoulli local level with probit link (10 args)
 *          - C_MCMC_probit_bernoulli_localtrend: Bernoulli local trend with probit link (14 args)
 *          - C_MCMC_probit_bernoulli_localacceleration: Bernoulli local acceleration with probit (18 args)
 *
 *          **Test Helper Functions:**
 *
 *          *Utility and Basic Functions (4 functions):*
 *          - test_ilogit: Inverse logit transformation (1 arg)
 *          - test_generate_normal_vector: Multivariate normal sampling (4 args)
 *          - test_adapt_cwmh_parameters: CWMH adaptation testing (10 args)
 *          - reset_adaptation_cache: Cache management (0 args)
 *
 *          *Precision Parameter Sampling (3 functions):*
 *          - test_generate_precision_data: Data precision (4 args)
 *          - test_generate_precision_theta_k: Intermediate precision (6 args)
 *          - test_generate_precision_theta_p: Final precision (4 args)
 *
 *          *State Parameter Sampling (4 functions):*
 *          - test_generate_theta_1_locallevel: Local level first state (4 args)
 *          - test_generate_theta_1: Local trend first state (6 args)
 *          - test_generate_theta_k: Intermediate state (6 args)
 *          - test_generate_theta_p: Final state (4 args)
 *
 *          *Initial State Parameter Sampling (4 functions):*
 *          - test_generate_theta_01_locallevel: Local level initial mean (4 args)
 *          - test_generate_theta_01: Local trend initial mean (5 args)
 *          - test_generate_theta_0k: Intermediate initial mean (8 args)
 *          - test_generate_theta_0p: Final initial mean (7 args)
 *
 *          *Binomial Component Testing - Logit Link (4 functions):*
 *          - test_generate_alpha_logit_binomial_locallevel: Local level alpha (5 args)
 *          - test_cwmh_alpha_logit_binomial_locallevel: Local level CWMH (6 args)
 *          - test_generate_alpha_logit_binomial: Local trend alpha (7 args)
 *          - test_cwmh_alpha_logit_binomial: Local trend CWMH (7 args)
 *
 *          *Bernoulli Component Testing - Probit Link (2 functions):*
 *          - test_generate_alpha_probit_bernoulli_locallevel: Local level probit (4 args)
 *          - test_generate_alpha_probit_bernoulli: Local trend probit (6 args)
 *
 *          *Complete MCMC Testing with Parameter Fixing (2 functions):*
 *          - test_mcmc_binomial_locallevel_fixed_params: Binomial full MCMC (19 args)
 *          - test_mcmc_probit_bernoulli_locallevel_fixed_params: Bernoulli full MCMC (11 args)
 *
 *          **Version History:**
 *          - v1.0 (Initial): Basic MCMC samplers and utility functions
 *          - v1.1 (2025-09-23): Enhanced adaptive MCMC with threshold parameters
 *          - v1.2 (2025-10-05): Added probit-Bernoulli samplers, updated binomial arg counts
 *          - v1.3 (2025-10-13): Corrected documentation and added missing includes
 *          - v1.4 (2025-10-15): Added progress bar support (verbose, bar_width parameters)
 *          - v1.5 (2025-10-23): Added Gaussian mixture model with dynamic weights
 *
 * @note Function pointers must be cast to DL_FUNC for R compatibility
 * @note Argument counts are enforced by R's .Call() mechanism at runtime
 * @note NULL terminator is required for proper array traversal by R
 * @note Names must match exactly those used in R wrapper functions
 * @note Test functions enable comprehensive unit testing of internal algorithms
 *
 * @warning Modifying this table requires corresponding changes in R wrapper functions
 * @warning Incorrect argument counts will cause runtime errors in R
 * @warning Test functions should only be used in testing/development environments
 * @warning Do not expose test functions in production NAMESPACE exports
 *
 * @see R_registerRoutines
 * @see DL_FUNC
 * @see R_CallMethodDef
 * @since version 1.0
 */
static const R_CallMethodDef CallEntries[] = {
  //============================================================================
  // PRODUCTION MCMC ALGORITHMS
  //============================================================================

  // --- Gaussian Dynamic Models ---
  {"_pdm_C_MCMC_normal_locallevel",               (DL_FUNC) &C_MCMC_normal_locallevel,        12},
  {"_pdm_C_MCMC_normal_localtrend",               (DL_FUNC) &C_MCMC_normal_localtrend,        16},
  {"_pdm_C_MCMC_normal_localacceleration",        (DL_FUNC) &C_MCMC_normal_localacceleration, 20},

  // --- Gaussian Mixture Models with Dynamic Weights ---
  {"_pdm_C_MCMC_normal_mixture_localtrend",       (DL_FUNC) &C_MCMC_normal_mixture_localtrend, 31},

  // --- Binomial Dynamic Models (Logit Link) ---
  {"_pdm_C_MCMC_logit_binomial_locallevel",              (DL_FUNC) &C_MCMC_logit_binomial_locallevel,        19},
  {"_pdm_C_MCMC_logit_binomial_localtrend",              (DL_FUNC) &C_MCMC_logit_binomial_localtrend,        23},
  {"_pdm_C_MCMC_logit_binomial_localacceleration",       (DL_FUNC) &C_MCMC_logit_binomial_localacceleration, 27},

  // --- Bernoulli Dynamic Models (Probit Link) ---
  {"_pdm_C_MCMC_probit_bernoulli_locallevel",            (DL_FUNC) &C_MCMC_probit_bernoulli_locallevel,        10},
  {"_pdm_C_MCMC_probit_bernoulli_localtrend",            (DL_FUNC) &C_MCMC_probit_bernoulli_localtrend,        14},
  {"_pdm_C_MCMC_probit_bernoulli_localacceleration",     (DL_FUNC) &C_MCMC_probit_bernoulli_localacceleration, 18},

  //============================================================================
  // TEST HELPER FUNCTIONS
  //============================================================================

  // --- Utility and Basic Functions ---
  {"_pdm_test_ilogit",                       (DL_FUNC) &test_ilogit,                       1},
  {"_pdm_test_generate_normal_vector",       (DL_FUNC) &test_generate_normal_vector,       4},
  {"_pdm_test_adapt_cwmh_parameters",        (DL_FUNC) &test_adapt_cwmh_parameters,       10},
  {"_pdm_reset_adaptation_cache",            (DL_FUNC) &reset_adaptation_cache_wrapper,    0},

  // --- Precision Parameter Sampling Tests ---
  {"_pdm_test_generate_precision_data",    (DL_FUNC) &test_generate_precision_data,    4},
  {"_pdm_test_generate_precision_theta_k", (DL_FUNC) &test_generate_precision_theta_k, 6},
  {"_pdm_test_generate_precision_theta_p", (DL_FUNC) &test_generate_precision_theta_p, 4},

  // --- State Parameter Sampling Tests ---
  {"_pdm_test_generate_theta_1_locallevel", (DL_FUNC) &test_generate_theta_1_locallevel, 4},
  {"_pdm_test_generate_theta_1",            (DL_FUNC) &test_generate_theta_1,            6},
  {"_pdm_test_generate_theta_k",            (DL_FUNC) &test_generate_theta_k,            6},
  {"_pdm_test_generate_theta_p",            (DL_FUNC) &test_generate_theta_p,            4},

  // --- Initial State Parameter Sampling Tests ---
  {"_pdm_test_generate_theta_01_locallevel", (DL_FUNC) &test_generate_theta_01_locallevel, 4},
  {"_pdm_test_generate_theta_01",            (DL_FUNC) &test_generate_theta_01,            5},
  {"_pdm_test_generate_theta_0k",            (DL_FUNC) &test_generate_theta_0k,            8},
  {"_pdm_test_generate_theta_0p",            (DL_FUNC) &test_generate_theta_0p,            7},

  // --- Binomial Model Component Tests (Logit Link) ---
  {"_pdm_test_generate_alpha_logit_binomial_locallevel", (DL_FUNC) &test_generate_alpha_logit_binomial_locallevel, 5},
  {"_pdm_test_cwmh_alpha_logit_binomial_locallevel",     (DL_FUNC) &test_cwmh_alpha_logit_binomial_locallevel,     6},
  {"_pdm_test_generate_alpha_logit_binomial",            (DL_FUNC) &test_generate_alpha_logit_binomial,            7},
  {"_pdm_test_cwmh_alpha_logit_binomial",                (DL_FUNC) &test_cwmh_alpha_logit_binomial,                7},

  // --- Bernoulli Model Component Tests (Probit Link) ---
  {"_pdm_test_generate_alpha_probit_bernoulli_locallevel", (DL_FUNC) &test_generate_alpha_probit_bernoulli_locallevel, 4},
  {"_pdm_test_generate_alpha_probit_bernoulli",            (DL_FUNC) &test_generate_alpha_probit_bernoulli,            6},

  // --- Complete MCMC Simulation Tests with Parameter Fixing ---
  {"_pdm_test_mcmc_binomial_locallevel_fixed_params",         (DL_FUNC) &test_mcmc_binomial_locallevel_fixed_params,         19},
  {"_pdm_test_mcmc_probit_bernoulli_locallevel_fixed_params", (DL_FUNC) &test_mcmc_probit_bernoulli_locallevel_fixed_params, 11},

  //============================================================================
  // END OF TABLE MARKER
  //============================================================================
  {NULL, NULL, 0}
};

//==============================================================================
// PACKAGE INITIALIZATION FUNCTION
//==============================================================================

/**
 * @brief Register compiled routines for the pdm package.
 *
 * @details The initialization routine registers every compiled entry point with
 *          R's dynamic loader and disables runtime symbol lookup to enforce
 *          encapsulation. The registration covers production MCMC samplers and
 *          comprehensive test helpers so that R's .Call interface can safely
 *          dispatch to the corresponding C implementations.
 *
 * @param dll Pointer to the DllInfo structure supplied automatically by R during library loading.
 *
 * @return Nothing. Registration occurs for its side effects on R's loader state.
 *
 * @note The function name must follow the "R_init_<package>" convention so that
 *       R invokes it during library() calls.
 * @note Disabling dynamic symbol lookup prevents unregistered entry points from
 *       being resolved at runtime.
 * @note All production and testing routines must appear in CallEntries to remain
 *       accessible from R wrappers.
 *
 * @warning Calling this function manually from user code is unsupported.
 * @warning Updating CallEntries without synchronising the R wrappers will break
 *          the package interface.
 *
 * @see R_registerRoutines
 * @see R_useDynamicSymbols
 * @see Writing R Extensions manual, Section 5.4
 * @since version 1.0
 */
void R_init_pdm(DllInfo *dll)
{
  /* Register .Call entry points for C functions accessible from R */
  R_registerRoutines(
    dll,         /* dll: package DLL information */
    NULL,        /* cMethods: no .C registrations */
    CallEntries, /* callMethods: .Call registration table */
    NULL,        /* fMethods: no .Fortran registrations */
    NULL         /* rMethods: no .External registrations */
  );

  /* Disable dynamic symbol lookup for improved security and encapsulation */
  R_useDynamicSymbols(
    dll,   /* dll: package DLL information */
    FALSE  /* value: disable dynamic symbol lookup */
  );
}
