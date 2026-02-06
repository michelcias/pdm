# bdm: Bayesian dynamic models

<!-- badges: start -->
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->

## Overview

`bdm` is an R package that provides tools for estimating Bayesian dynamic
models, which are formulated as Bayesian regressions with a specific prior.
The package leverages the Markovian structure of these models, enabling
efficient treatment of higher-orderstates as hyperparameters.

## Development Status

- ✅ Initial package structure
- ⬜ Core C function implementation (in development)
- ⬜ R interface (in development)
- ⬜ Complete documentation (in development)
- ⬜ Unit tests (planned)

## Installation

This package is in active development and not yet recommended for production
use. For developers interested in contributing or testing, you can install the
development version directly from GitHub:

```r
# Install devtools package if necessary
# install.packages("devtools")
devtools::install_github("michelcias/bdm")
```
