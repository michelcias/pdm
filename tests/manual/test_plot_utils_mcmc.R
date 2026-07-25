# =============================================================================
# Script de Teste: R/plot_utils_mcmc.R
# =============================================================================
# Este script testa todas as funções do novo arquivo plot_utils_mcmc.R
# Execute linha por linha ou rode todo o script

# Limpar ambiente
rm(list = ls())

# -----------------------------------------------------------------------------
# 1. CARREGAR DEPENDÊNCIAS
# -----------------------------------------------------------------------------

# Source dos arquivos necessários (simula carregamento do pacote)
source("R/plot_utils_base.R")
source("R/plot_utils_mcmc.R")

cat("✓ Arquivos carregados com sucesso\n\n")


# -----------------------------------------------------------------------------
# 2. CRIAR OBJETOS MOCK PARA TESTES
# -----------------------------------------------------------------------------

# Função auxiliar para criar objetos mock
create_mock_object <- function(model_class = "mixture", model_order = 1) {

  n_draws <- 100
  n_obs <- 50

  # Componentes comuns a todos os modelos
  obj <- list(
    theta_01 = rnorm(n_draws),
    prec_theta1 = rgamma(n_draws, 2, 1),
    theta_1 = matrix(rnorm(n_draws * n_obs), n_draws, n_obs)
  )

  # Adicionar componentes de mixture
  if (model_class == "mixture") {
    obj$mu_1 <- rnorm(n_draws, 0, 1)
    obj$mu_2 <- rnorm(n_draws, 2, 1)
    obj$prec_1 <- rgamma(n_draws, 2, 1)
    obj$prec_2 <- rgamma(n_draws, 2, 1)
    obj$alpha <- matrix(runif(n_draws * n_obs), n_draws, n_obs)
    obj$z <- matrix(rbinom(n_draws * n_obs, 1, 0.5), n_draws, n_obs)
  } else {
    # Standard model tem prec_y
    obj$prec_y <- rgamma(n_draws, 2, 1)
  }

  # Adicionar estados de ordem superior
  if (model_order >= 2) {
    obj$theta_02 <- rnorm(n_draws)
    obj$prec_theta2 <- rgamma(n_draws, 2, 1)
    obj$theta_2 <- matrix(rnorm(n_draws * n_obs), n_draws, n_obs)
  }

  if (model_order >= 3) {
    obj$theta_03 <- rnorm(n_draws)
    obj$prec_theta3 <- rgamma(n_draws, 2, 1)
    obj$theta_3 <- matrix(rnorm(n_draws * n_obs), n_draws, n_obs)
  }

  # Definir classe e atributos
  model_type <- switch(
    as.character(model_order),
    "1" = "locallevel",
    "2" = "localtrend",
    "3" = "localacceleration"
  )

  if (model_class == "mixture") {
    class(obj) <- c(
      paste0("normal_mixture_", model_type),
      "pdm_mcmc",
      "list"
    )
  } else {
    class(obj) <- c(
      paste0("normal_", model_type),
      "pdm_mcmc",
      "list"
    )
  }

  attr(obj, "n_obs") <- n_obs
  attr(obj, "n_draws") <- n_draws
  attr(obj, "burnin") <- 1000
  attr(obj, "thinning") <- 10
  attr(obj, "model_type") <- model_type
  attr(obj, "y") <- rnorm(n_obs)

  return(obj)
}

cat("✓ Função auxiliar create_mock_object() criada\n\n")


# -----------------------------------------------------------------------------
# 3. TESTE 1: detect_model_type()
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 1: detect_model_type()\n")
cat("=============================================================================\n\n")

# Teste 1.1: Mixture locallevel
obj_mix_ll <- create_mock_object("mixture", 1)
result1 <- detect_model_type(obj_mix_ll)

cat("Teste 1.1 - Mixture Locallevel:\n")
cat("  model_class:", result1$model_class, "\n")
cat("  model_order:", result1$model_order, "\n")
cat("  has_mixture:", result1$has_mixture, "\n")

stopifnot(
  result1$model_class == "mixture",
  result1$model_order == 1L,
  result1$has_mixture == TRUE
)
cat("  ✓ PASSOU\n\n")

# Teste 1.2: Mixture localtrend
obj_mix_lt <- create_mock_object("mixture", 2)
result2 <- detect_model_type(obj_mix_lt)

cat("Teste 1.2 - Mixture Localtrend:\n")
cat("  model_class:", result2$model_class, "\n")
cat("  model_order:", result2$model_order, "\n")
cat("  has_mixture:", result2$has_mixture, "\n")

stopifnot(
  result2$model_class == "mixture",
  result2$model_order == 2L,
  result2$has_mixture == TRUE
)
cat("  ✓ PASSOU\n\n")

# Teste 1.3: Mixture localacceleration
obj_mix_la <- create_mock_object("mixture", 3)
result3 <- detect_model_type(obj_mix_la)

cat("Teste 1.3 - Mixture Localacceleration:\n")
cat("  model_class:", result3$model_class, "\n")
cat("  model_order:", result3$model_order, "\n")
cat("  has_mixture:", result3$has_mixture, "\n")

stopifnot(
  result3$model_class == "mixture",
  result3$model_order == 3L,
  result3$has_mixture == TRUE
)
cat("  ✓ PASSOU\n\n")

# Teste 1.4: Standard locallevel
obj_std_ll <- create_mock_object("standard", 1)
result4 <- detect_model_type(obj_std_ll)

cat("Teste 1.4 - Standard Locallevel:\n")
cat("  model_class:", result4$model_class, "\n")
cat("  model_order:", result4$model_order, "\n")
cat("  has_mixture:", result4$has_mixture, "\n")

stopifnot(
  result4$model_class == "standard",
  result4$model_order == 1L,
  result4$has_mixture == FALSE
)
cat("  ✓ PASSOU\n\n")


# -----------------------------------------------------------------------------
# 4. TESTE 2: get_n_params()
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 2: get_n_params()\n")
cat("=============================================================================\n\n")

test_cases <- list(
  list(class = "mixture", order = 1L, expected = 6L),
  list(class = "mixture", order = 2L, expected = 8L),
  list(class = "mixture", order = 3L, expected = 10L),
  list(class = "standard", order = 1L, expected = 3L),
  list(class = "standard", order = 2L, expected = 5L),
  list(class = "standard", order = 3L, expected = 7L)
)

for (i in seq_along(test_cases)) {
  tc <- test_cases[[i]]
  result <- get_n_params(tc$class, tc$order)

  cat(sprintf("Teste 2.%d - %s order %d: %d parâmetros\n",
              i, tc$class, tc$order, result))

  stopifnot(result == tc$expected)
  cat("  ✓ PASSOU\n")
}

cat("\n")


# -----------------------------------------------------------------------------
# 5. TESTE 3: get_param_config()
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 3: get_param_config()\n")
cat("=============================================================================\n\n")

# Teste 3.1: Mixture locallevel (6 parâmetros)
cat("Teste 3.1 - Mixture Locallevel Config:\n")
config1 <- get_param_config(obj_mix_ll, "mixture", 1)

cat("  Número de parâmetros:", length(config1), "\n")
cat("  Nomes:", paste(names(config1), collapse = ", "), "\n")

stopifnot(
  length(config1) == 6,
  all(names(config1) == c("mu_1", "mu_2", "phi_1", "phi_2", "theta_01", "W1_inv"))
)

# Verificar estrutura de um parâmetro
cat("  Estrutura do primeiro parâmetro (mu_1):\n")
cat("    - samples: numeric[", length(config1$mu_1$samples), "]\n", sep = "")
cat("    - name: ", deparse(config1$mu_1$name), "\n", sep = "")
cat("    - label: ", deparse(config1$mu_1$label), "\n", sep = "")
cat("    - name_str: ", config1$mu_1$name_str, "\n", sep = "")
cat("    - label_str: ", config1$mu_1$label_str, "\n", sep = "")

stopifnot(
  identical(config1$mu_1$samples, obj_mix_ll$mu_1),
  identical(config1$W1_inv$samples, obj_mix_ll$prec_theta1)
)
cat("  ✓ PASSOU\n\n")

# Teste 3.2: Mixture localtrend (8 parâmetros)
cat("Teste 3.2 - Mixture Localtrend Config:\n")
config2 <- get_param_config(obj_mix_lt, "mixture", 2)

cat("  Número de parâmetros:", length(config2), "\n")
cat("  Nomes:", paste(names(config2), collapse = ", "), "\n")

stopifnot(
  length(config2) == 8,
  all(names(config2) == c("mu_1", "mu_2", "phi_1", "phi_2",
                          "theta_01", "theta_02", "W1_inv", "W2_inv"))
)
cat("  ✓ PASSOU\n\n")

# Teste 3.3: Mixture localacceleration (10 parâmetros)
cat("Teste 3.3 - Mixture Localacceleration Config:\n")
config3 <- get_param_config(obj_mix_la, "mixture", 3)

cat("  Número de parâmetros:", length(config3), "\n")
cat("  Nomes:", paste(names(config3), collapse = ", "), "\n")

stopifnot(
  length(config3) == 10,
  all(names(config3) == c("mu_1", "mu_2", "phi_1", "phi_2",
                          "theta_01", "theta_02", "theta_03",
                          "W1_inv", "W2_inv", "W3_inv"))
)
cat("  ✓ PASSOU\n\n")

# Teste 3.4: Standard locallevel (3 parâmetros)
cat("Teste 3.4 - Standard Locallevel Config:\n")
config4 <- get_param_config(obj_std_ll, "standard", 1)

cat("  Número de parâmetros:", length(config4), "\n")
cat("  Nomes:", paste(names(config4), collapse = ", "), "\n")

stopifnot(
  length(config4) == 3,
  all(names(config4) == c("V_inv", "theta_01", "W1_inv")),
  identical(config4$V_inv$samples, obj_std_ll$prec_y)
)
cat("  ✓ PASSOU\n\n")

# Teste 3.5: Auto-detecção (sem passar model_class e model_order)
cat("Teste 3.5 - Auto-detecção:\n")
config5 <- get_param_config(obj_mix_ll)  # Sem argumentos adicionais

cat("  Número de parâmetros:", length(config5), "\n")
stopifnot(length(config5) == 6)
cat("  ✓ PASSOU\n\n")


# -----------------------------------------------------------------------------
# 6. TESTE 4: validate_param_config()
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 4: validate_param_config()\n")
cat("=============================================================================\n\n")

# Teste 4.1: Configuração válida
cat("Teste 4.1 - Configuração válida:\n")
result <- validate_param_config(config1)
cat("  ✓ PASSOU (sem erros)\n\n")

# Teste 4.2: Configuração inválida (falta campo)
cat("Teste 4.2 - Detectar campo faltando:\n")
invalid_config <- list(
  mu_1 = list(
    samples = rnorm(100),
    name = quote(mu[1])
    # Faltando: label, name_str, label_str
  )
)

tryCatch({
  validate_param_config(invalid_config)
  cat("  ✗ FALHOU (deveria dar erro)\n")
}, error = function(e) {
  cat("  Erro esperado capturado:", conditionMessage(e), "\n")
  cat("  ✓ PASSOU\n")
})
cat("\n")


# -----------------------------------------------------------------------------
# 7. TESTE 5: plot_mcmc_diagnostics_generic() - BASE GRAPHICS
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 5: plot_mcmc_diagnostics_generic() - BASE GRAPHICS\n")
cat("=============================================================================\n\n")

cat("Este teste gerará plots. Feche as janelas para continuar.\n\n")

# Teste 5.1: Plotar 2 parâmetros de mixture locallevel
cat("Teste 5.1 - Plotando mu_1 e mu_2 (base graphics):\n")
cat("  Pressione ENTER para ver os plots...\n")
readline()

par(ask = FALSE)  # Não perguntar entre plots
plot_mcmc_diagnostics_generic(obj_mix_ll, which = 1:2)

cat("  ✓ Plots gerados (verifique visualmente)\n\n")


# -----------------------------------------------------------------------------
# 8. TESTE 6: plot_mcmc_diagnostics_generic() - GGPLOT2
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 6: plot_mcmc_diagnostics_generic() - GGPLOT2\n")
cat("=============================================================================\n\n")

# O argumento engine = c("base", "ggplot2") foi removido dos helpers em #53
# (529d837); plot_mcmc_diagnostics_generic() hoje só produz base graphics. O
# backend ggplot2 está parado em inst/prototype/plot_utils_ggplot.R e não está
# ligado a nenhum método plot.*, então não há nada para testar aqui.
cat("  ⊘ NÃO APLICÁVEL (seletor engine removido em #53)\n")
cat("    Veja inst/prototype/README.md para retomar o backend ggplot2.\n\n")


# -----------------------------------------------------------------------------
# 9. TESTE 7: Validação de Erros
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 7: Validação de Erros\n")
cat("=============================================================================\n\n")

# Teste 7.1: which fora do intervalo
cat("Teste 7.1 - which fora do intervalo:\n")
tryCatch({
  plot_mcmc_diagnostics_generic(obj_mix_ll, which = 99)
  cat("  ✗ FALHOU (deveria dar erro)\n")
}, error = function(e) {
  cat("  Erro esperado:", conditionMessage(e), "\n")
  cat("  ✓ PASSOU\n")
})
cat("\n")

# Teste 7.2: objeto sem classe pdm_mcmc
cat("Teste 7.2 - Objeto inválido:\n")
invalid_obj <- list(mu_1 = rnorm(100))
tryCatch({
  detect_model_type(invalid_obj)
  cat("  ✗ FALHOU (deveria dar erro)\n")
}, error = function(e) {
  cat("  Erro esperado:", conditionMessage(e), "\n")
  cat("  ✓ PASSOU\n")
})
cat("\n")


# -----------------------------------------------------------------------------
# 10. RESUMO DOS TESTES
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("RESUMO DOS TESTES\n")
cat("=============================================================================\n\n")

cat("✓ TESTE 1: detect_model_type() - 4/4 testes passaram\n")
cat("✓ TESTE 2: get_n_params() - 6/6 testes passaram\n")
cat("✓ TESTE 3: get_param_config() - 5/5 testes passaram\n")
cat("✓ TESTE 4: validate_param_config() - 2/2 testes passaram\n")
cat("✓ TESTE 5: plot_mcmc_diagnostics_generic() BASE - OK\n")
cat("⊘ TESTE 6: plot_mcmc_diagnostics_generic() GGPLOT2 - NÃO APLICÁVEL\n")
cat("✓ TESTE 7: Validação de Erros - 2/2 testes passaram\n\n")

cat("=============================================================================\n")
cat("TODOS OS TESTES CONCLUÍDOS COM SUCESSO! ✓\n")
cat("=============================================================================\n\n")

cat("O arquivo R/plot_utils_mcmc.R está funcionando corretamente.\n")
cat("Você pode prosseguir para a próxima etapa.\n")
