/**
 * @file init.c
 * @brief R package initialization and function registration
 * @details Handles dynamic loading and registration of C functions for the pdm package.
 *          Registers .Call entry points for MCMC algorithms and utility functions,
 *          ensuring proper interface between R and C code. Implements security
 *          measures by disabling dynamic symbol lookup.
 * @author Michel H. Montoril
 * @date 2025-10-13
 * @version 1.3
 *
 * @changelog
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
#include "mcmc_locallevel.h"
#include "mcmc_localtrend.h"
#include "mcmc_localacceleration.h"
#include "mcmc_binomial_locallevel.h"
#include "mcmc_binomial_localtrend.h"
#include "mcmc_binomial_localacceleration.h"
#include "mcmc_bernoulli_locallevel.h"
#include "mcmc_bernoulli_localtrend.h"
#include "mcmc_bernoulli_localacceleration.h"
#include "utils.h"

//==============================================================================
// FORWARD DECLARATIONS FOR TEST HELPER FUNCTIONS
//==============================================================================

/**
 * @brief Forward declarations for test helper functions defined in test_helpers.c
 * @details These functions expose internal C routines through R's .Call interface
 *          to enable comprehensive unit testing and validation of MCMC components.
 */

// --- Utility and basic function tests ---
SEXP test_ilogit(SEXP x_);
SEXP test_generate_normal_vector(SEXP y_, SEXP a_, SEXP b_, SEXP add_a_);

// --- Adaptive MCMC parameter tests ---
SEXP test_adapt_cwmh_parameters(SEXP theta_updated_, SEXP log_sigma_,
                                SEXP lag_update_, SEXP n_, SEXP iter_,
                                SEXP max_step_size_, SEXP base_adaptation_rate_,
                                SEXP decay_exponent_, SEXP target_acceptance_,
                                SEXP min_deviation_threshold_);
SEXP reset_adaptation_cache_wrapper(void);

// --- Precision parameter sampling tests ---
SEXP test_generate_precision_data(SEXP y_, SEXP theta_1_, SEXP nu_y_, SEXP eta_y_);
SEXP test_generate_precision_theta_k(SEXP theta_0k_, SEXP theta_0kp1_, SEXP theta_k_,
                                     SEXP theta_kp1_, SEXP nu_0k_, SEXP eta_0k_);
SEXP test_generate_precision_theta_p(SEXP theta_0p_, SEXP theta_p_, SEXP nu_0p_, SEXP eta_0p_);

// --- State parameter sampling tests (Gaussian models) ---
SEXP test_generate_theta_1_locallevel(SEXP data_, SEXP prec_data_, SEXP prec_theta_1_, SEXP theta_01_);
SEXP test_generate_theta_1(SEXP data_, SEXP theta_2_, SEXP prec_data_, SEXP prec_theta_1_,
                           SEXP theta_01_, SEXP theta_02_);
SEXP test_generate_theta_k(SEXP theta_km1_, SEXP theta_kp1_, SEXP prec_km1_, SEXP prec_k_,
                           SEXP theta_0k_, SEXP theta_0kp1_);
SEXP test_generate_theta_p(SEXP theta_pm1_, SEXP prec_pm1_, SEXP prec_p_, SEXP theta_0p_);

// --- Initial state parameter sampling tests ---
SEXP test_generate_theta_01_locallevel(SEXP theta_1_, SEXP prec_theta_1_, SEXP mean_theta_01_,
                                       SEXP prec_theta_01_);
SEXP test_generate_theta_01(SEXP theta_1_, SEXP theta_02_, SEXP prec_theta_1_, SEXP mean_theta_01_,
                            SEXP prec_theta_01_);
SEXP test_generate_theta_0k(SEXP theta_km1_, SEXP theta_k_, SEXP theta_0km1_, SEXP theta_0kp1_,
                            SEXP prec_km1_, SEXP prec_k_, SEXP mean_0k_, SEXP prec_0k_);
SEXP test_generate_theta_0p(SEXP theta_pm1_, SEXP theta_p_, SEXP theta_0pm1_, SEXP prec_pm1_,
                            SEXP prec_p_, SEXP mean_0p_, SEXP prec_0p_);

// --- Binomial model component tests (logit link) ---
SEXP test_generate_alpha_logit_binomial_locallevel(SEXP theta_1_in_, SEXP theta_01_in_,
                                                   SEXP prec_1_in_, SEXP y_, SEXP n_trials_);
SEXP test_cwmh_alpha_logit_binomial_locallevel(SEXP theta_1_in_, SEXP theta_01_in_, SEXP prec_1_in_,
                                               SEXP y_, SEXP n_trials_, SEXP log_sigma_in_);
SEXP test_generate_alpha_logit_binomial(SEXP theta_1_in_, SEXP theta_2_in_, SEXP theta_01_in_,
                                        SEXP theta_02_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_);
SEXP test_cwmh_alpha_logit_binomial(SEXP theta_1_in_, SEXP theta_2_in_, SEXP theta_01_in_,
                                    SEXP theta_02_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_);

// --- Bernoulli model component tests (probit link) ---
SEXP test_generate_alpha_probit_bernoulli_locallevel(SEXP theta_1_in_, SEXP theta_01_in_,
                                                     SEXP prec_1_in_, SEXP y_);
SEXP test_generate_alpha_probit_bernoulli(SEXP theta_1_in_, SEXP theta_2_in_,
                                          SEXP theta_01_in_, SEXP theta_02_in_,
                                          SEXP prec_1_in_, SEXP y_);

// --- Complete MCMC simulation tests with parameter fixing ---
SEXP test_mcmc_binomial_locallevel_fixed_params(SEXP y_,
                                                SEXP n_trials_,
                                                SEXP burnin_,
                                                SEXP thinning_,
                                                SEXP n_chain_,
                                                SEXP theta_1_true_,
                                                SEXP theta_01_true_,
                                                SEXP prec_1_true_,
                                                SEXP prior_theta01_mean_,
                                                SEXP prior_theta01_prec_,
                                                SEXP prior_prec1_shape_,
                                                SEXP prior_prec1_rate_, SEXP lag_update_,
                                                SEXP max_step_size_, SEXP base_adaptation_rate_,
                                                SEXP decay_exponent_, SEXP target_acceptance_,
                                                SEXP return_log_sigma_, SEXP return_accept_prop_);
SEXP test_mcmc_probit_bernoulli_locallevel_fixed_params(SEXP y_, SEXP burnin_, SEXP thinning_, SEXP n_chain_,
                                                        SEXP theta_1_true_, SEXP theta_01_true_, SEXP prec_1_true_,
                                                        SEXP prior_theta01_mean_, SEXP prior_theta01_prec_,
                                                        SEXP prior_prec1_shape_, SEXP prior_prec1_rate_);

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
 *          - C_MCMC_locallevel: Gaussian local level model (10 args)
 *          - C_MCMC_localtrend: Gaussian local trend model (14 args)
 *          - C_MCMC_localacceleration: Gaussian local acceleration model (18 args)
 *          - C_MCMC_logit_binomial_locallevel: Binomial local level with logit link (17 args)
 *          - C_MCMC_logit_binomial_localtrend: Binomial local trend with logit link (21 args)
 *          - C_MCMC_logit_binomial_localacceleration: Binomial local acceleration with logit (25 args)
 *          - C_MCMC_probit_bernoulli_locallevel: Bernoulli local level with probit link (8 args)
 *          - C_MCMC_probit_bernoulli_localtrend: Bernoulli local trend with probit link (12 args)
 *          - C_MCMC_probit_bernoulli_localacceleration: Bernoulli local acceleration with probit (16 args)
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
  {"_pdm_C_MCMC_locallevel",               (DL_FUNC) &C_MCMC_locallevel,                              10},
  {"_pdm_C_MCMC_localtrend",               (DL_FUNC) &C_MCMC_localtrend,                              14},
  {"_pdm_C_MCMC_localacceleration",        (DL_FUNC) &C_MCMC_localacceleration,                       18},

  // --- Binomial Dynamic Models (Logit Link) ---
  {"_pdm_C_MCMC_logit_binomial_locallevel",(DL_FUNC) &C_MCMC_logit_binomial_locallevel,               17},
  {"_pdm_C_MCMC_logit_binomial_localtrend",(DL_FUNC) &C_MCMC_logit_binomial_localtrend,               21},
  {"_pdm_C_MCMC_logit_binomial_localacceleration",(DL_FUNC) &C_MCMC_logit_binomial_localacceleration, 25},

  // --- Bernoulli Dynamic Models (Probit Link) ---
  {"_pdm_C_MCMC_probit_bernoulli_locallevel",       (DL_FUNC) &C_MCMC_probit_bernoulli_locallevel,         8},
  {"_pdm_C_MCMC_probit_bernoulli_localtrend",       (DL_FUNC) &C_MCMC_probit_bernoulli_localtrend,        12},
  {"_pdm_C_MCMC_probit_bernoulli_localacceleration",(DL_FUNC) &C_MCMC_probit_bernoulli_localacceleration, 16},

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
  {"_pdm_test_generate_alpha_probit_bernoulli_locallevel",(DL_FUNC) &test_generate_alpha_probit_bernoulli_locallevel, 4},
  {"_pdm_test_generate_alpha_probit_bernoulli",           (DL_FUNC) &test_generate_alpha_probit_bernoulli,            6},

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
 * @brief Package initialization function called when pdm package is loaded
 *
 * @details Mandatory initialization function for R packages with compiled code.
 *          Registers all C functions with R's dynamic loading system and configures
 *          security settings for symbol resolution. This function is automatically
 *          called by R when the package is loaded via library() or require().
 *
 *          **Registration Process:**
 *          The function registers three categories of C functions:
 *
 *          1. **Production MCMC Algorithms (9 functions)**:
 *             - Gaussian models: local level, trend, acceleration
 *             - Binomial models with logit link: local level, trend, acceleration
 *             - Bernoulli models with probit link: local level, trend, acceleration
 *
 *          2. **Test Helper Functions (23 functions)**:
 *             - Utility functions (4): transformations, sampling, adaptation
 *             - Precision samplers (3): data, intermediate, final
 *             - State samplers (4): first, intermediate, final states
 *             - Initial state samplers (4): level, trend, intermediate, final
 *             - Binomial components (4): alpha generation and CWMH testing
 *             - Bernoulli components (2): probit link testing
 *             - Complete MCMC (2): full sampler validation with parameter fixing
 *
 *          **Initialization Sequence:**
 *          1. Register .Call entry points from CallEntries table
 *          2. Disable dynamic symbol lookup for security
 *          3. Return control to R loading system
 *
 *          **Security Implications:**
 *          R_useDynamicSymbols(dll, FALSE) prevents external packages from
 *          accessing internal symbols, improving package encapsulation and
 *          reducing potential conflicts or security vulnerabilities. This is
 *          a recommended security practice for all R packages with compiled code.
 *
 * @param dll Pointer to DllInfo structure containing package loading information.
 *            Provided automatically by R's loading system.
 *
 * @note This function name must be exactly "R_init_<packagename>" for R to find it
 * @note Called automatically during package loading - not intended for manual invocation
 * @note Disabling dynamic symbols is a recommended security practice per R-exts manual
 * @note All .Call functions must be registered here to be accessible from R
 * @note Test functions enable comprehensive validation of MCMC algorithm components
 * @note The registration enforces type safety and argument count validation at runtime
 *
 * @warning Do not call this function manually from user code
 * @warning Modifying registration without updating R code will break package functionality
 * @warning Function must be exported in package's NAMESPACE file
 * @warning Test functions should only be exposed in development/testing builds
 * @warning Ensure all forward declarations match actual function signatures
 *
 * @see R_registerRoutines
 * @see R_useDynamicSymbols
 * @see DllInfo
 * @see Writing R Extensions manual, section 5.4
 * @since version 1.0
 */
void R_init_pdm(DllInfo *dll) {
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
