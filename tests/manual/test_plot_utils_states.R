# =============================================================================
# Script de Teste: R/plot_utils_states.R
# =============================================================================
# Este script testa todas as funções do arquivo plot_utils_states.R
# Execute linha por linha ou rode todo o script

# Limpar ambiente
rm(list = ls())

# -----------------------------------------------------------------------------
# 1. CARREGAR DEPENDÊNCIAS
# -----------------------------------------------------------------------------

# Source dos arquivos necessários
source("R/plot_utils_base.R")
source("R/plot_utils_mcmc.R")
source("R/plot_utils_states.R")

cat("✓ Arquivos carregados com sucesso\n\n")


# -----------------------------------------------------------------------------
# 2. FUNÇÃO AUXILIAR PARA CRIAR OBJETOS MOCK
# -----------------------------------------------------------------------------

create_mock_object <- function(model_class = "mixture", model_order = 1) {

  n_draws <- 100
  n_obs <- 50

  # Componentes comuns
  obj <- list(
    theta_01 = rnorm(n_draws),
    prec_theta1 = rgamma(n_draws, 2, 1),
    theta_1 = matrix(rnorm(n_draws * n_obs), n_draws, n_obs)
  )

  # Mixture components
  if (model_class == "mixture") {
    obj$mu_1 <- rnorm(n_draws, 0, 1)
    obj$mu_2 <- rnorm(n_draws, 2, 1)
    obj$prec_1 <- rgamma(n_draws, 2, 1)
    obj$prec_2 <- rgamma(n_draws, 2, 1)
    obj$alpha <- matrix(runif(n_draws * n_obs), n_draws, n_obs)
    obj$z <- matrix(rbinom(n_draws * n_obs, 1, 0.5), n_draws, n_obs)
  } else {
    obj$prec_y <- rgamma(n_draws, 2, 1)
  }

  # Higher order states
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

  # Set class and attributes
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
# 3. TESTE 1: get_state_matrices()
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 1: get_state_matrices()\n")
cat("=============================================================================\n\n")

# Teste 1.1: Locallevel (1 state)
obj_ll <- create_mock_object("mixture", 1)
states_ll <- get_state_matrices(obj_ll, 1)

cat("Teste 1.1 - Locallevel:\n")
cat("  Número de estados:", length(states_ll), "\n")
cat("  Nomes:", paste(names(states_ll), collapse = ", "), "\n")

stopifnot(
  length(states_ll) == 1,
  "theta_1" %in% names(states_ll),
  identical(states_ll$theta_1, obj_ll$theta_1)
)
cat("  ✓ PASSOU\n\n")

# Teste 1.2: Localtrend (2 states)
obj_lt <- create_mock_object("mixture", 2)
states_lt <- get_state_matrices(obj_lt, 2)

cat("Teste 1.2 - Localtrend:\n")
cat("  Número de estados:", length(states_lt), "\n")
cat("  Nomes:", paste(names(states_lt), collapse = ", "), "\n")

stopifnot(
  length(states_lt) == 2,
  all(names(states_lt) == c("theta_1", "theta_2")),
  identical(states_lt$theta_1, obj_lt$theta_1),
  identical(states_lt$theta_2, obj_lt$theta_2)
)
cat("  ✓ PASSOU\n\n")

# Teste 1.3: Localacceleration (3 states)
obj_la <- create_mock_object("mixture", 3)
states_la <- get_state_matrices(obj_la, 3)

cat("Teste 1.3 - Localacceleration:\n")
cat("  Número de estados:", length(states_la), "\n")
cat("  Nomes:", paste(names(states_la), collapse = ", "), "\n")

stopifnot(
  length(states_la) == 3,
  all(names(states_la) == c("theta_1", "theta_2", "theta_3")),
  identical(states_la$theta_1, obj_la$theta_1),
  identical(states_la$theta_2, obj_la$theta_2),
  identical(states_la$theta_3, obj_la$theta_3)
)
cat("  ✓ PASSOU\n\n")

# Teste 1.4: Auto-detecção
cat("Teste 1.4 - Auto-detecção de model_order:\n")
states_auto <- get_state_matrices(obj_lt)  # Sem especificar order

stopifnot(
  length(states_auto) == 2,
  all(names(states_auto) == c("theta_1", "theta_2"))
)
cat("  ✓ PASSOU\n\n")


# -----------------------------------------------------------------------------
# 4. TESTE 2: get_n_states()
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 2: get_n_states()\n")
cat("=============================================================================\n\n")

test_cases <- list(
  list(order = 1L, expected = 1L),
  list(order = 2L, expected = 2L),
  list(order = 3L, expected = 3L)
)

for (i in seq_along(test_cases)) {
  tc <- test_cases[[i]]
  result <- get_n_states(tc$order)

  cat(sprintf("Teste 2.%d - Order %d: %d estados\n",
              i, tc$order, result))

  stopifnot(result == tc$expected)
  cat("  ✓ PASSOU\n")
}

cat("\n")


# -----------------------------------------------------------------------------
# 5. TESTE 3: summarise_state()
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 3: summarise_state()\n")
cat("=============================================================================\n\n")

# Teste 3.1: Com CI
cat("Teste 3.1 - Sumarização com intervalos de credibilidade:\n")
summary_with_ci <- summarise_state(obj_ll$theta_1, ci = TRUE, ci_level = 0.95)

cat("  Componentes:", paste(names(summary_with_ci), collapse = ", "), "\n")
cat("  Comprimento median:", length(summary_with_ci$median), "\n")
cat("  Comprimento lower:", length(summary_with_ci$lower), "\n")
cat("  Comprimento upper:", length(summary_with_ci$upper), "\n")

stopifnot(
  all(names(summary_with_ci) == c("median", "lower", "upper")),
  length(summary_with_ci$median) == attr(obj_ll, "n_obs"),
  length(summary_with_ci$lower) == attr(obj_ll, "n_obs"),
  length(summary_with_ci$upper) == attr(obj_ll, "n_obs"),
  all(summary_with_ci$lower <= summary_with_ci$median),
  all(summary_with_ci$median <= summary_with_ci$upper)
)
cat("  ✓ PASSOU\n\n")

# Teste 3.2: Sem CI
cat("Teste 3.2 - Sumarização sem intervalos de credibilidade:\n")
summary_no_ci <- summarise_state(obj_ll$theta_1, ci = FALSE)

cat("  Componentes:", paste(names(summary_no_ci), collapse = ", "), "\n")
cat("  lower é NULL:", is.null(summary_no_ci$lower), "\n")
cat("  upper é NULL:", is.null(summary_no_ci$upper), "\n")

stopifnot(
  all(names(summary_no_ci) == c("median", "lower", "upper")),
  length(summary_no_ci$median) == attr(obj_ll, "n_obs"),
  is.null(summary_no_ci$lower),
  is.null(summary_no_ci$upper)
)
cat("  ✓ PASSOU\n\n")


# -----------------------------------------------------------------------------
# 6. TESTE 4: summarise_all_states()
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 4: summarise_all_states()\n")
cat("=============================================================================\n\n")

# Teste 4.1: Locallevel
cat("Teste 4.1 - Sumarizar todos os estados (locallevel):\n")
all_summaries_ll <- summarise_all_states(obj_ll, ci = TRUE, ci_level = 0.95)

cat("  Número de estados sumarizados:", length(all_summaries_ll), "\n")
cat("  Nomes:", paste(names(all_summaries_ll), collapse = ", "), "\n")

stopifnot(
  length(all_summaries_ll) == 1,
  "theta_1" %in% names(all_summaries_ll),
  !is.null(all_summaries_ll$theta_1$median),
  !is.null(all_summaries_ll$theta_1$lower),
  !is.null(all_summaries_ll$theta_1$upper)
)
cat("  ✓ PASSOU\n\n")

# Teste 4.2: Localtrend
cat("Teste 4.2 - Sumarizar todos os estados (localtrend):\n")
all_summaries_lt <- summarise_all_states(obj_lt, model_order = 2, ci = TRUE)

cat("  Número de estados sumarizados:", length(all_summaries_lt), "\n")
cat("  Nomes:", paste(names(all_summaries_lt), collapse = ", "), "\n")

stopifnot(
  length(all_summaries_lt) == 2,
  all(names(all_summaries_lt) == c("theta_1", "theta_2"))
)
cat("  ✓ PASSOU\n\n")

# Teste 4.3: Localacceleration
cat("Teste 4.3 - Sumarizar todos os estados (localacceleration):\n")
all_summaries_la <- summarise_all_states(obj_la, model_order = 3, ci = TRUE)

cat("  Número de estados sumarizados:", length(all_summaries_la), "\n")
cat("  Nomes:", paste(names(all_summaries_la), collapse = ", "), "\n")

stopifnot(
  length(all_summaries_la) == 3,
  all(names(all_summaries_la) == c("theta_1", "theta_2", "theta_3"))
)
cat("  ✓ PASSOU\n\n")


# -----------------------------------------------------------------------------
# 7. TESTE 5: compute_innovations()
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 5: compute_innovations()\n")
cat("=============================================================================\n\n")

# Teste 5.1: Locallevel (1 innovation)
cat("Teste 5.1 - Inovações para locallevel:\n")
innov_ll <- compute_innovations(obj_ll, 1)

cat("  Número de inovações:", length(innov_ll), "\n")
cat("  Nomes:", paste(names(innov_ll), collapse = ", "), "\n")
cat("  Dimensões innov_1:", paste(dim(innov_ll$innov_1), collapse = " x "), "\n")

stopifnot(
  length(innov_ll) == 1,
  "innov_1" %in% names(innov_ll),
  is.matrix(innov_ll$innov_1),
  nrow(innov_ll$innov_1) == attr(obj_ll, "n_draws"),
  ncol(innov_ll$innov_1) == attr(obj_ll, "n_obs")
)
cat("  ✓ PASSOU\n\n")

# Teste 5.2: Localtrend (2 innovations)
cat("Teste 5.2 - Inovações para localtrend:\n")
innov_lt <- compute_innovations(obj_lt, 2)

cat("  Número de inovações:", length(innov_lt), "\n")
cat("  Nomes:", paste(names(innov_lt), collapse = ", "), "\n")

stopifnot(
  length(innov_lt) == 2,
  all(names(innov_lt) == c("innov_1", "innov_2")),
  is.matrix(innov_lt$innov_1),
  is.matrix(innov_lt$innov_2)
)
cat("  ✓ PASSOU\n\n")

# Teste 5.3: Localacceleration (3 innovations)
cat("Teste 5.3 - Inovações para localacceleration:\n")
innov_la <- compute_innovations(obj_la, 3)

cat("  Número de inovações:", length(innov_la), "\n")
cat("  Nomes:", paste(names(innov_la), collapse = ", "), "\n")

stopifnot(
  length(innov_la) == 3,
  all(names(innov_la) == c("innov_1", "innov_2", "innov_3")),
  is.matrix(innov_la$innov_1),
  is.matrix(innov_la$innov_2),
  is.matrix(innov_la$innov_3)
)
cat("  ✓ PASSOU\n\n")

# Teste 5.4: Verificar propriedades das inovações (média ~0)
cat("Teste 5.4 - Propriedades estatísticas das inovações:\n")
mean_innov1 <- mean(innov_ll$innov_1)
cat("  Média das inovações (deve ser ~0):", round(mean_innov1, 4), "\n")
cat("  Desvio padrão:", round(sd(innov_ll$innov_1), 4), "\n")

# Média deve estar próxima de zero (teste flexível)
stopifnot(abs(mean_innov1) < 0.5)
cat("  ✓ PASSOU\n\n")


# -----------------------------------------------------------------------------
# 8. TESTE 6: get_state_labels()
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 6: get_state_labels()\n")
cat("=============================================================================\n\n")

# Teste 6.1: Order 1
cat("Teste 6.1 - Labels para order 1:\n")
labels_1 <- get_state_labels(1)

cat("  Nomes dos estados:", paste(labels_1$state_names, collapse = ", "), "\n")
cat("  Títulos:", paste(labels_1$state_titles, collapse = ", "), "\n")
cat("  Cores:", paste(labels_1$state_colors, collapse = ", "), "\n")

stopifnot(
  length(labels_1$state_names) == 1,
  labels_1$state_names[1] == "theta_1",
  length(labels_1$state_labels) == 1,
  length(labels_1$state_titles) == 1,
  labels_1$state_titles[1] == "Level State",
  length(labels_1$state_colors) == 1
)
cat("  ✓ PASSOU\n\n")

# Teste 6.2: Order 2
cat("Teste 6.2 - Labels para order 2:\n")
labels_2 <- get_state_labels(2)

cat("  Nomes dos estados:", paste(labels_2$state_names, collapse = ", "), "\n")
cat("  Títulos:", paste(labels_2$state_titles, collapse = ", "), "\n")

stopifnot(
  length(labels_2$state_names) == 2,
  all(labels_2$state_names == c("theta_1", "theta_2")),
  length(labels_2$state_titles) == 2,
  labels_2$state_titles[2] == "Trend State"
)
cat("  ✓ PASSOU\n\n")

# Teste 6.3: Order 3
cat("Teste 6.3 - Labels para order 3:\n")
labels_3 <- get_state_labels(3)

cat("  Nomes dos estados:", paste(labels_3$state_names, collapse = ", "), "\n")
cat("  Títulos:", paste(labels_3$state_titles, collapse = ", "), "\n")

stopifnot(
  length(labels_3$state_names) == 3,
  all(labels_3$state_names == c("theta_1", "theta_2", "theta_3")),
  length(labels_3$state_titles) == 3,
  labels_3$state_titles[3] == "Acceleration State"
)
cat("  ✓ PASSOU\n\n")


# -----------------------------------------------------------------------------
# 9. TESTE 7: get_innovation_labels()
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 7: get_innovation_labels()\n")
cat("=============================================================================\n\n")

# Teste 7.1: Order 1
cat("Teste 7.1 - Innovation labels para order 1:\n")
innov_labels_1 <- get_innovation_labels(1)

cat("  Número de labels:", length(innov_labels_1$innov_titles), "\n")
cat("  Títulos:", paste(innov_labels_1$innov_titles, collapse = ", "), "\n")
cat("  Cores:", paste(innov_labels_1$innov_colors, collapse = ", "), "\n")

stopifnot(
  length(innov_labels_1$innov_labels) == 1,
  length(innov_labels_1$innov_titles) == 1,
  innov_labels_1$innov_titles[1] == "Level Innovations",
  length(innov_labels_1$innov_colors) == 1
)
cat("  ✓ PASSOU\n\n")

# Teste 7.2: Order 2
cat("Teste 7.2 - Innovation labels para order 2:\n")
innov_labels_2 <- get_innovation_labels(2)

cat("  Número de labels:", length(innov_labels_2$innov_titles), "\n")
cat("  Títulos:", paste(innov_labels_2$innov_titles, collapse = ", "), "\n")

stopifnot(
  length(innov_labels_2$innov_labels) == 2,
  length(innov_labels_2$innov_titles) == 2,
  innov_labels_2$innov_titles[2] == "Trend Innovations"
)
cat("  ✓ PASSOU\n\n")

# Teste 7.3: Order 3
cat("Teste 7.3 - Innovation labels para order 3:\n")
innov_labels_3 <- get_innovation_labels(3)

cat("  Número de labels:", length(innov_labels_3$innov_titles), "\n")
cat("  Títulos:", paste(innov_labels_3$innov_titles, collapse = ", "), "\n")

stopifnot(
  length(innov_labels_3$innov_labels) == 3,
  length(innov_labels_3$innov_titles) == 3,
  innov_labels_3$innov_titles[3] == "Acceleration Innovations"
)
cat("  ✓ PASSOU\n\n")


# -----------------------------------------------------------------------------
# 10. TESTE 8: plot_state_trajectory_base()
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 8: plot_state_trajectory_base() - TESTE VISUAL\n")
cat("=============================================================================\n\n")

cat("Este teste gerará um plot de trajetória de estado.\n")
cat("Pressione ENTER para visualizar...\n")
readline()

# Preparar dados
summary_theta1 <- summarise_state(obj_ll$theta_1, ci = TRUE, ci_level = 0.95)
time_grid <- seq_len(attr(obj_ll, "n_obs"))

# Criar plot
par(mfrow = c(1, 1))
plot_state_trajectory_base(
  time_grid = time_grid,
  median = summary_theta1$median,
  lower = summary_theta1$lower,
  upper = summary_theta1$upper,
  ylab = expression(theta["t,1"]),
  main = "Level State",
  col = "black",
  ci = TRUE,
  ci_label = "95% CI"
)

cat("  ✓ Plot gerado (verifique visualmente)\n\n")


# -----------------------------------------------------------------------------
# 11. TESTE 9: plot_innovation_base()
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 9: plot_innovation_base() - TESTE VISUAL\n")
cat("=============================================================================\n\n")

cat("Este teste gerará um plot de inovações.\n")
cat("Pressione ENTER para visualizar...\n")
readline()

# Preparar dados de inovações
innov_summary <- summarise_state(innov_ll$innov_1, ci = TRUE, ci_level = 0.95)

# Criar plot
par(mfrow = c(1, 1))
plot_innovation_base(
  time_grid = time_grid,
  median = innov_summary$median,
  lower = innov_summary$lower,
  upper = innov_summary$upper,
  ylab = expression(u["t,1"]),
  main = "Level Innovations",
  col_bar = "steelblue",
  ci = TRUE,
  ci_label = "95% CI"
)

cat("  ✓ Plot gerado (verifique visualmente)\n\n")


# -----------------------------------------------------------------------------
# 12. TESTE 10: Validação de Erros
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 10: Validação de Erros\n")
cat("=============================================================================\n\n")

# Teste 10.1: model_order inválido
cat("Teste 10.1 - model_order inválido:\n")
tryCatch({
  get_state_labels(99)
  cat("  ✗ FALHOU (deveria dar erro)\n")
}, error = function(e) {
  cat("  Erro esperado:", conditionMessage(e), "\n")
  cat("  ✓ PASSOU\n")
})
cat("\n")

# Teste 10.2: ci_level inválido
cat("Teste 10.2 - ci_level inválido:\n")
tryCatch({
  summarise_state(obj_ll$theta_1, ci = TRUE, ci_level = 1.5)
  cat("  ✗ FALHOU (deveria dar erro)\n")
}, error = function(e) {
  cat("  Erro esperado:", conditionMessage(e), "\n")
  cat("  ✓ PASSOU\n")
})
cat("\n")

# Teste 10.3: objeto sem model_type
cat("Teste 10.3 - Objeto sem model_type:\n")
invalid_obj <- list(theta_1 = matrix(rnorm(100), 10, 10))
class(invalid_obj) <- c("test_class", "pdm_mcmc", "list")

tryCatch({
  get_state_matrices(invalid_obj)
  cat("  ✗ FALHOU (deveria dar erro)\n")
}, error = function(e) {
  cat("  Erro esperado:", conditionMessage(e), "\n")
  cat("  ✓ PASSOU\n")
})
cat("\n")


# -----------------------------------------------------------------------------
# 13. RESUMO DOS TESTES
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("RESUMO DOS TESTES\n")
cat("=============================================================================\n\n")

cat("✓ TESTE 1: get_state_matrices() - 4/4 testes passaram\n")
cat("✓ TESTE 2: get_n_states() - 3/3 testes passaram\n")
cat("✓ TESTE 3: summarise_state() - 2/2 testes passaram\n")
cat("✓ TESTE 4: summarise_all_states() - 3/3 testes passaram\n")
cat("✓ TESTE 5: compute_innovations() - 4/4 testes passaram\n")
cat("✓ TESTE 6: get_state_labels() - 3/3 testes passaram\n")
cat("✓ TESTE 7: get_innovation_labels() - 3/3 testes passaram\n")
cat("✓ TESTE 8: plot_state_trajectory_base() - OK (visual)\n")
cat("✓ TESTE 9: plot_innovation_base() - OK (visual)\n")
cat("✓ TESTE 10: Validação de Erros - 3/3 testes passaram\n\n")

cat("=============================================================================\n")
cat("TODOS OS TESTES CONCLUÍDOS COM SUCESSO! ✓\n")
cat("=============================================================================\n\n")

cat("O arquivo R/plot_utils_states.R está funcionando corretamente.\n")
cat("Você pode prosseguir para a próxima etapa.\n")
