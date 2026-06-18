# Prototype source (not yet integrated)

This directory holds package source code that is still under construction and
is intentionally **not** placed in `R/`, so that `R CMD check` does not analyze
it and it is not loaded with the package.

## `plot_utils_ggplot.R`

A prototype **ggplot2 backend** for the `plot.*` methods. The idea is to let
users choose the plotting backend via an `engine = c("base", "ggplot2")`
argument. The base-graphics backend is the one currently wired into the
`plot.*` methods; this file is the ggplot2 counterpart, still being developed.

It is kept out of `R/` for two reasons:

1. The `engine` selector is not yet wired into any `plot.*` method, so these
   functions are unreachable and would only add dead, unexported code to the
   package namespace.
2. The functions use the `.data` pronoun inside `ggplot2::aes()`. Without
   declaring `rlang`/`.data` as a dependency, `R CMD check` (codetools) would
   raise a `no visible binding for global variable '.data'` NOTE for code that
   does nothing yet.

### Resuming this work

When the ggplot2 engine is ready to be integrated:

1. `git mv inst/prototype/plot_utils_ggplot.R R/`
2. Add the `.data` pronoun to the package (e.g. `@importFrom rlang .data` with
   `rlang` in `Imports`, or `utils::globalVariables(".data")`).
3. Add the `engine = c("base", "ggplot2")` argument (with `match.arg()`) to the
   `plot.*` methods and dispatch to the `*_ggplot()` helpers.
4. Re-enable the skipped test in `tests/testthat/test-plot-methods.R`
   (`"plot() validates engine argument"`).
