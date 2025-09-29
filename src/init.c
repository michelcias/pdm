/**
 * @file init.c
 * @brief R package initialization and function registration
 * @details Handles dynamic loading and registration of C functions for the pdm package.
 *          Registers .Call entry points for MCMC algorithms and utility functions,
 *          ensuring proper interface between R and C code. Implements security
 *          measures by disabling dynamic symbol lookup.
 * @author Michel H. Montoril
 * @date 2025-09-23
 * @version 1.1
 *
 * @changelog
 * - v1.1 (2025-09-23): Added registration for optimized adaptive MCMC functions
 *   including adapt_cwmh_parameters with threshold parameter and legacy wrapper
 *   for backward compatibility.
 */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

/* Include headers for all registered functions */
#include "mcmc_locallevel.h"
#include "mcmc_localtrend.h"
#include "mcmc_localacceleration.h"
#include "mcmc_binomial_locallevel.h"
#include "mcmc_binomial_localtrend.h"
#include "mcmc_binomial_localacceleration.h"
#include "utils.h"

// Forward declaration for test helper functions defined in test_helpers.c
SEXP test_ilogit(SEXP x_);
SEXP test_generate_normal_vector(SEXP y_, SEXP a_, SEXP b_, SEXP add_a_);
SEXP test_adapt_cwmh_parameters(SEXP theta_updated_, SEXP log_sigma_,
                                SEXP lag_update_, SEXP n_, SEXP iter_,
                                SEXP max_step_size_, SEXP base_adaptation_rate_,
                                SEXP decay_exponent_, SEXP target_acceptance_,
                                SEXP min_deviation_threshold_);
SEXP test_adapt_cwmh_parameters_legacy(SEXP theta_updated_, SEXP log_sigma_,
                                       SEXP lag_update_, SEXP n_, SEXP iter_,
                                       SEXP max_step_size_, SEXP base_adaptation_rate_,
                                       SEXP decay_exponent_, SEXP target_acceptance_);
SEXP test_generate_precision_data(SEXP y_, SEXP theta_1_, SEXP nu_y_, SEXP eta_y_);
SEXP test_generate_precision_theta_k(SEXP theta_0k_, SEXP theta_0kp1_, SEXP theta_k_,
                                     SEXP theta_kp1_, SEXP nu_0k_, SEXP eta_0k_);
SEXP test_generate_precision_theta_p(SEXP theta_0p_, SEXP theta_p_, SEXP nu_0p_, SEXP eta_0p_);
SEXP test_generate_theta_1_locallevel(SEXP data_, SEXP prec_data_, SEXP prec_theta_1_, SEXP theta_01_);
SEXP test_generate_theta_1(SEXP data_, SEXP theta_2_, SEXP prec_data_, SEXP prec_theta_1_,
                           SEXP theta_01_, SEXP theta_02_);
SEXP test_generate_theta_k(SEXP theta_km1_, SEXP theta_kp1_, SEXP prec_km1_, SEXP prec_k_,
                           SEXP theta_0k_, SEXP theta_0kp1_);
SEXP test_generate_theta_p(SEXP theta_pm1_, SEXP prec_pm1_, SEXP prec_p_, SEXP theta_0p_);
SEXP test_generate_theta_01_locallevel(SEXP theta_1_, SEXP prec_theta_1_, SEXP mean_theta_01_,
                                       SEXP prec_theta_01_);
SEXP test_generate_theta_01(SEXP theta_1_, SEXP theta_02_, SEXP prec_theta_1_, SEXP mean_theta_01_,
                            SEXP prec_theta_01_);
SEXP test_generate_theta_0k(SEXP theta_km1_, SEXP theta_k_, SEXP theta_0km1_, SEXP theta_0kp1_,
                            SEXP prec_km1_, SEXP prec_k_, SEXP mean_0k_, SEXP prec_0k_);
SEXP test_generate_theta_0p(SEXP theta_pm1_, SEXP theta_p_, SEXP theta_0pm1_, SEXP prec_pm1_,
                            SEXP prec_p_, SEXP mean_0p_, SEXP prec_0p_);
SEXP test_generate_alpha_logit_binomial_locallevel(SEXP theta_1_in_, SEXP theta_01_in_,
                                                   SEXP prec_1_in_, SEXP y_, SEXP n_trials_);
SEXP test_cwmh_alpha_logit_binomial_locallevel(SEXP theta_1_in_, SEXP theta_01_in_, SEXP prec_1_in_,
                                               SEXP y_, SEXP n_trials_, SEXP log_sigma_in_);
SEXP test_generate_alpha_logit_binomial(SEXP theta_1_in_, SEXP theta_2_in_, SEXP theta_01_in_,
                                        SEXP theta_02_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_);
SEXP test_cwmh_alpha_logit_binomial(SEXP theta_1_in_, SEXP theta_2_in_, SEXP theta_01_in_,
                                    SEXP theta_02_in_, SEXP prec_1_in_, SEXP y_, SEXP n_trials_);
SEXP test_mcmc_binomial_locallevel_fixed_params(SEXP y_, SEXP n_trials_, SEXP burnin_, SEXP thinning_,
                                                SEXP n_chain_, SEXP theta_1_true_, SEXP theta_01_true_,
                                                SEXP prec_1_true_, SEXP prior_theta01_mean_,
                                                SEXP prior_theta01_prec_, SEXP prior_prec1_shape_,
                                                SEXP prior_prec1_rate_, SEXP lag_update_,
                                                SEXP max_step_size_, SEXP base_adaptation_rate_,
                                                SEXP decay_exponent_, SEXP target_acceptance_,
                                                SEXP return_log_sigma_, SEXP return_accept_prop_);

/**
 * @brief Static table defining .Call method entries for R-C interface
 *
 * @details Maps R function names to their corresponding C implementations with
 *          argument counts. This table is used by R's dynamic loading system
 *          to properly route .Call() invocations to the correct C functions.
 *
 *          Registered functions include:
 *          **Main MCMC Functions:**
 *          - C_MCMC_locallevel: Local level model MCMC sampler (10 arguments)
 *          - C_MCMC_localtrend: Local trend model MCMC sampler (14 arguments)
 *          - C_MCMC_localacceleration: Local acceleration model MCMC sampler (18 arguments)
 *          - C_MCMC_logit_binomial_locallevel: Binomial local level model MCMC (17 arguments)
 *          - C_MCMC_logit_binomial_localtrend: Binomial local trend model MCMC (21 arguments)
 *          - C_MCMC_logit_binomial_localacceleration: Binomial local acceleration MCMC (25 arguments)
 *
 *          **Test Helper Functions (Utility and Basic):**
 *          - test_ilogit: Inverse logit transformation testing (1 argument)
 *          - test_generate_normal_vector: Multivariate normal sampling testing (4 arguments)
 *          - test_adapt_cwmh_parameters: Optimized CWMH adaptation testing (10 arguments)
 *          - test_adapt_cwmh_parameters_legacy: Legacy CWMH adaptation testing (9 arguments)
 *
 *          **Test Helper Functions (Precision Parameters):**
 *          - test_generate_precision_data: Data precision sampling testing (4 arguments)
 *          - test_generate_precision_theta_k: Intermediate precision sampling testing (6 arguments)
 *          - test_generate_precision_theta_p: Final precision sampling testing (4 arguments)
 *
 *          **Test Helper Functions (State Parameters):**
 *          - test_generate_theta_1_locallevel: Local level state sampling testing (4 arguments)
 *          - test_generate_theta_1: Local trend state sampling testing (6 arguments)
 *          - test_generate_theta_k: Intermediate state sampling testing (6 arguments)
 *          - test_generate_theta_p: Final state sampling testing (4 arguments)
 *
 *          **Test Helper Functions (Initial States):**
 *          - test_generate_theta_01_locallevel: Initial level state sampling testing (4 arguments)
 *          - test_generate_theta_01: Initial trend state sampling testing (5 arguments)
 *          - test_generate_theta_0k: Initial intermediate state sampling testing (8 arguments)
 *          - test_generate_theta_0p: Initial final state sampling testing (7 arguments)
 *
 *          **Test Helper Functions (Binomial Components):**
 *          - test_generate_alpha_logit_binomial_locallevel: Local level alpha generation testing (5 arguments)
 *          - test_cwmh_alpha_logit_binomial_locallevel: Local level CWMH testing (6 arguments)
 *          - test_generate_alpha_logit_binomial: Local trend alpha generation testing (7 arguments)
 *          - test_cwmh_alpha_logit_binomial: Local trend CWMH testing (7 arguments)
 *
 *          **Test Helper Functions (Complete MCMC):**
 *          - test_mcmc_binomial_locallevel_fixed_params: Full MCMC testing with diagnostics (19 arguments)
 *
 *          **Version 1.1 Updates:**
 *          Enhanced adaptive MCMC testing capabilities with both optimized and legacy versions
 *          for comprehensive comparison testing. The optimized version includes configurable
 *          deviation threshold parameter, while legacy version maintains backward compatibility.
 *
 *          **Version 1.2 Updates:**
 *          Updated C_MCMC_logit_binomial_locallevel argument count from 15 to 16
 *          to accommodate the new min_deviation_threshold parameter.
 *
 * note Function pointers must be cast to DL_FUNC for R compatibility
 * @note Argument counts are enforced by R's .Call() mechanism
 * @note NULL terminator is required for proper array traversal
 * @note Names must match exactly those used in R wrapper functions
 * @note Test functions enable comprehensive unit testing of internal C algorithms
 * @note Legacy functions support backward compatibility during transition periods
 *
 * @warning Modifying this table requires corresponding changes in R wrapper functions
 * @warning Incorrect argument counts will cause runtime errors in R
 * @warning Test functions should only be used in testing environments
 * @warning Legacy functions are deprecated and should be phased out in future versions
 *
 * @see R_registerRoutines
 * @see DL_FUNC
 * @see R_CallMethodDef
 * @since version 1.0
 */
static const R_CallMethodDef CallEntries[] = {
  // --- Main MCMC algorithm functions ---
  {"_pdm_C_MCMC_locallevel",               (DL_FUNC) &C_MCMC_locallevel,      10},
  {"_pdm_C_MCMC_localtrend",               (DL_FUNC) &C_MCMC_localtrend,      14},
  {"_pdm_C_MCMC_localacceleration",        (DL_FUNC) &C_MCMC_localacceleration, 18},
  {"_pdm_C_MCMC_logit_binomial_locallevel",(DL_FUNC) &C_MCMC_logit_binomial_locallevel, 17},
  {"_pdm_C_MCMC_logit_binomial_localtrend",(DL_FUNC) &C_MCMC_logit_binomial_localtrend, 21},
  {"_pdm_C_MCMC_logit_binomial_localacceleration",(DL_FUNC) &C_MCMC_logit_binomial_localacceleration, 25},

  // --- Utility and basic function tests ---
  {"_pdm_test_ilogit",                     (DL_FUNC)   &test_ilogit,                  1},
  {"_pdm_test_generate_normal_vector",     (DL_FUNC)   &test_generate_normal_vector,  4},
  {"_pdm_test_adapt_cwmh_parameters",      (DL_FUNC)   &test_adapt_cwmh_parameters,   10},  // Updated argument count
  {"_pdm_test_adapt_cwmh_parameters_legacy", (DL_FUNC) &test_adapt_cwmh_parameters_legacy, 9},  // New legacy function

  // --- Precision parameter sampling tests ---
  {"_pdm_test_generate_precision_data",    (DL_FUNC) &test_generate_precision_data,    4},
  {"_pdm_test_generate_precision_theta_k", (DL_FUNC) &test_generate_precision_theta_k, 6},
  {"_pdm_test_generate_precision_theta_p", (DL_FUNC) &test_generate_precision_theta_p, 4},

  // --- State parameter sampling tests ---
  {"_pdm_test_generate_theta_1_locallevel", (DL_FUNC) &test_generate_theta_1_locallevel, 4},
  {"_pdm_test_generate_theta_1",           (DL_FUNC) &test_generate_theta_1,           6},
  {"_pdm_test_generate_theta_k",           (DL_FUNC) &test_generate_theta_k,           6},
  {"_pdm_test_generate_theta_p",           (DL_FUNC) &test_generate_theta_p,           4},

  // --- Initial state parameter sampling tests ---
  {"_pdm_test_generate_theta_01_locallevel", (DL_FUNC) &test_generate_theta_01_locallevel, 4},
  {"_pdm_test_generate_theta_01",            (DL_FUNC) &test_generate_theta_01,            5},
  {"_pdm_test_generate_theta_0k",            (DL_FUNC) &test_generate_theta_0k,            8},
  {"_pdm_test_generate_theta_0p",            (DL_FUNC) &test_generate_theta_0p,            7},

  // --- Binomial model component tests ---
  {"_pdm_test_generate_alpha_logit_binomial_locallevel", (DL_FUNC) &test_generate_alpha_logit_binomial_locallevel, 5},
  {"_pdm_test_cwmh_alpha_logit_binomial_locallevel", (DL_FUNC) &test_cwmh_alpha_logit_binomial_locallevel, 6},
  {"_pdm_test_generate_alpha_logit_binomial", (DL_FUNC) &test_generate_alpha_logit_binomial, 7},
  {"_pdm_test_cwmh_alpha_logit_binomial", (DL_FUNC) &test_cwmh_alpha_logit_binomial, 7},

  // --- Complete MCMC simulation tests ---
  {"_pdm_test_mcmc_binomial_locallevel_fixed_params", (DL_FUNC) &test_mcmc_binomial_locallevel_fixed_params, 19},

  // --- End of table marker ---
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
 *          The registration process includes:
 *          - Main MCMC algorithms for Gaussian and binomial dynamic models
 *          - Comprehensive test helper functions for unit testing internal components
 *          - Utility functions for mathematical operations and diagnostics
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
 * @note Test functions enable comprehensive validation of MCMC algorithm components
 *
 * @warning Do not call this function manually
 * @warning Modifying registration without updating R code will break package functionality
 * @warning Function must be exported in package's NAMESPACE
 * @warning Test functions should only be exposed in development/testing builds
 *
 * @see R_registerRoutines
 * @see R_useDynamicSymbols
 * @see DllInfo
 * @since version 1.0
 */
void R_init_pdm(DllInfo *dll) {
  // Register .Call entry points for C functions accessible from R
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);

  // Disable dynamic symbol lookup for improved security and encapsulation
  R_useDynamicSymbols(dll, FALSE);
}
