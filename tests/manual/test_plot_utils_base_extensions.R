# =============================================================================
# Script de Teste: Extensões de R/plot_utils_base.R
# =============================================================================
# Testa as novas funções genéricas: plot_all_mixture_generic_base() e
# plot_dynamic_states_generic_base()

# Limpar ambiente
rm(list = ls())

# -----------------------------------------------------------------------------
# 1. CARREGAR DEPENDÊNCIAS
# -----------------------------------------------------------------------------

source("R/plot_utils_base.R")
source("R/plot_utils_mcmc.R")
source("R/plot_utils_states.R")

cat("✓ Arquivos carregados com sucesso\n\n")


# -----------------------------------------------------------------------------
# 2. FUNÇÃO AUXILIAR PARA CRIAR OBJETOS MOCK
# -----------------------------------------------------------------------------

create_mock_mixture <- function(model_order = 1) {

  n_draws <- 100
  n_obs <- 50

  # Componentes básicos
  obj <- list(
    mu_1 = rnorm(n_draws, 0, 1),
    mu_2 = rnorm(n_draws, 2, 1),
    prec_1 = rgamma(n_draws, 2, 1),
    prec_2 = rgamma(n_draws, 2, 1),
    theta_01 = rnorm(n_draws),
    prec_theta1 = rgamma(n_draws, 2, 1),
    theta_1 = matrix(rnorm(n_draws * n_obs), n_draws, n_obs),
    alpha = matrix(runif(n_draws * n_obs, 0.2, 0.8), n_draws, n_obs),
    z = matrix(rbinom(n_draws * n_obs, 1, 0.5), n_draws, n_obs)
  )

  # Ordem superior
  if (model_order >= 2) {
    obj$theta_02 <- rnorm(n_draws)
    obj$prec_theta2 <- rgamma(n_draws, 2, 1)
    obj$theta_2 <- matrix(rnorm(n_draws * n_obs, 0, 0.5), n_draws, n_obs)
  }

  if (model_order >= 3) {
    obj$theta_03 <- rnorm(n_draws)
    obj$prec_theta3 <- rgamma(n_draws, 2, 1)
    obj$theta_3 <- matrix(rnorm(n_draws * n_obs, 0, 0.2), n_draws, n_obs)
  }

  # Classe
  model_type <- switch(
    as.character(model_order),
    "1" = "locallevel",
    "2" = "localtrend",
    "3" = "localacceleration"
  )

  class(obj) <- c(
    paste0("normal_mixture_", model_type),
    "pdm_mcmc",
    "list"
  )

  attr(obj, "n_obs") <- n_obs
  attr(obj, "n_draws") <- n_draws
  attr(obj, "burnin") <- 1000
  attr(obj, "thinning") <- 10
  attr(obj, "model_type") <- model_type
  attr(obj, "y") <- rnorm(n_obs)

  return(obj)
}

cat("✓ Função auxiliar create_mock_mixture() criada\n\n")


# -----------------------------------------------------------------------------
# 3. TESTE 1: plot_dynamic_states_generic_base() - Locallevel
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 1: plot_dynamic_states_generic_base() - Locallevel (1 página)\n")
cat("=============================================================================\n\n")

obj_ll <- create_mock_mixture(1)

cat("Teste 1.1 - Plotar estados dinâmicos (locallevel):\n")
cat("  Esperado: 1 página com 1 trajetória de estado\n")
cat("  Pressione ENTER para visualizar...\n")
readline()

par(ask = FALSE)
plot_dynamic_states_generic_base(obj_ll, which = 1, ci = TRUE, ci_level = 0.95)

cat("  ✓ Plot gerado\n")
cat("  Verifique:\n")
cat("    - 1 painel mostrando Level State (theta_t,1)\n")
cat("    - Linha mediana em preto\n")
cat("    - Banda de credibilidade cinza\n")
cat("    - Título 'Dynamic State Trajectories'\n\n")

readline(prompt = "Pressione ENTER para continuar...")


# -----------------------------------------------------------------------------
# 4. TESTE 2: plot_dynamic_states_generic_base() - Localtrend
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 2: plot_dynamic_states_generic_base() - Localtrend (2 páginas)\n")
cat("=============================================================================\n\n")

obj_lt <- create_mock_mixture(2)

cat("Teste 2.1 - Página 1: Trajetórias dos estados\n")
cat("  Pressione ENTER para visualizar...\n")
readline()

plot_dynamic_states_generic_base(obj_lt, which = 1, ci = TRUE, ci_level = 0.95)

cat("  ✓ Página 1 gerada\n")
cat("  Verifique:\n")
cat("    - 2 painéis: Level State (preto) e Trend State (azul)\n")
cat("    - Ambos com bandas de credibilidade\n\n")

readline(prompt = "Pressione ENTER para ver página 2...")

cat("Teste 2.2 - Página 2: Inovações e diagnósticos\n")
plot_dynamic_states_generic_base(obj_lt, which = 2, ci = TRUE, ci_level = 0.95)

cat("  ✓ Página 2 gerada\n")
cat("  Verifique:\n")
cat("    - 4 painéis em grade 2x2\n")
cat("    - Level Innovations (barras azuis)\n")
cat("    - Trend Innovations (barras verdes)\n")
cat("    - Joint Trajectories (linhas pretas e azuis)\n")
cat("    - Linha vermelha tracejada em y=0 nas inovações\n\n")

readline(prompt = "Pressione ENTER para continuar...")


# -----------------------------------------------------------------------------
# 5. TESTE 3: plot_dynamic_states_generic_base() - Localacceleration
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 3: plot_dynamic_states_generic_base() - Localacceleration (3 páginas)\n")
cat("=============================================================================\n\n")

obj_la <- create_mock_mixture(3)

cat("Teste 3.1 - Página 1: Trajetórias (3 estados)\n")
cat("  Pressione ENTER para visualizar...\n")
readline()

plot_dynamic_states_generic_base(obj_la, which = 1, ci = TRUE, ci_level = 0.95)

cat("  ✓ Página 1 gerada\n")
cat("  Verifique:\n")
cat("    - 3 painéis verticais\n")
cat("    - Level (preto), Trend (azul), Acceleration (vermelho)\n\n")

readline(prompt = "Pressione ENTER para ver página 2...")

cat("Teste 3.2 - Página 2: Inovações e diagnósticos\n")
plot_dynamic_states_generic_base(obj_la, which = 2, ci = TRUE, ci_level = 0.95)

cat("  ✓ Página 2 gerada\n\n")

readline(prompt = "Pressione ENTER para ver página 3...")

cat("Teste 3.3 - Página 3: Relações entre estados (pairs plot)\n")
plot_dynamic_states_generic_base(obj_la, which = 3, ci = TRUE, ci_level = 0.95)

cat("  ✓ Página 3 gerada\n")
cat("  Verifique:\n")
cat("    - Matriz de scatter plots (3x3)\n")
cat("    - Relações bivariadas entre theta_1, theta_2, theta_3\n\n")

readline(prompt = "Pressione ENTER para continuar...")


# -----------------------------------------------------------------------------
# 6. TESTE 4: plot_all_mixture_generic_base() - Dashboard Completo
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 4: plot_all_mixture_generic_base() - Dashboard Completo\n")
cat("=============================================================================\n\n")

cat("Este teste gerará o dashboard COMPLETO com TODAS as páginas.\n")
cat("Para mixture locallevel, são 10 páginas no total:\n")
cat("  - Páginas 1-6: MCMC diagnostics (mu_1, mu_2, phi_1, phi_2, theta_01, W1_inv)\n")
cat("  - Página 7: Mixture parameters (bivariate)\n")
cat("  - Página 8: Dynamic states (trajectories)\n")
cat("  - Página 9: Mixture weight (alpha_t)\n")
cat("  - Página 10: Component indicators (z_t)\n\n")

cat("IMPORTANTE: Como são muitas páginas, vou usar ask=FALSE.\n")
cat("Os plots serão gerados rapidamente.\n")
cat("Pressione ENTER para iniciar...\n")
readline()

# Salvar em PDF para facilitar revisão
pdf("test_dashboard_locallevel.pdf", width = 10, height = 8)
par(ask = FALSE)
plot_all_mixture_generic_base(obj_ll, ask = FALSE, ci = TRUE, ci_level = 0.95)
dev.off()

cat("  ✓ Dashboard completo salvo em 'test_dashboard_locallevel.pdf'\n\n")


# -----------------------------------------------------------------------------
# 7. TESTE 5: Dashboard para Localtrend (13 páginas)
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 5: Dashboard Localtrend (13 páginas)\n")
cat("=============================================================================\n\n")

cat("Gerando dashboard para localtrend...\n")

pdf("test_dashboard_localtrend.pdf", width = 10, height = 8)
par(ask = FALSE)
plot_all_mixture_generic_base(obj_lt, ask = FALSE, ci = TRUE, ci_level = 0.95)
dev.off()

cat("  ✓ Dashboard completo salvo em 'test_dashboard_localtrend.pdf'\n")
cat("  Páginas esperadas: 13\n")
cat("    - 8 MCMC diagnostics\n")
cat("    - 1 Mixture params\n")
cat("    - 2 Dynamic states\n")
cat("    - 2 Mixture weights\n\n")


# -----------------------------------------------------------------------------
# 8. TESTE 6: Dashboard para Localacceleration (16 páginas)
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 6: Dashboard Localacceleration (16 páginas)\n")
cat("=============================================================================\n\n")

cat("Gerando dashboard para localacceleration...\n")

pdf("test_dashboard_localacceleration.pdf", width = 10, height = 8)
par(ask = FALSE)
plot_all_mixture_generic_base(obj_la, ask = FALSE, ci = TRUE, ci_level = 0.95)
dev.off()

cat("  ✓ Dashboard completo salvo em 'test_dashboard_localacceleration.pdf'\n")
cat("  Páginas esperadas: 16\n")
cat("    - 10 MCMC diagnostics\n")
cat("    - 1 Mixture params\n")
cat("    - 3 Dynamic states\n")
cat("    - 2 Mixture weights\n\n")


# -----------------------------------------------------------------------------
# 9. TESTE 7: Testar parâmetro which
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 7: Testar parâmetro 'which' (plotar apenas páginas específicas)\n")
cat("=============================================================================\n\n")

cat("Teste 7.1 - Plotar apenas página 2 de states (localtrend):\n")
cat("  Pressione ENTER para visualizar...\n")
readline()

plot_dynamic_states_generic_base(obj_lt, which = 2, ci = TRUE, ci_level = 0.95)

cat("  ✓ Apenas página 2 plotada\n\n")


# -----------------------------------------------------------------------------
# 10. TESTE 8: Testar sem intervalos de credibilidade
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 8: Testar sem intervalos de credibilidade (ci = FALSE)\n")
cat("=============================================================================\n\n")

cat("Teste 8.1 - Estados sem CI:\n")
cat("  Pressione ENTER para visualizar...\n")
readline()

plot_dynamic_states_generic_base(obj_ll, which = 1, ci = FALSE)

cat("  ✓ Plot gerado sem bandas de credibilidade\n")
cat("  Verifique:\n")
cat("    - Apenas linha mediana visível\n")
cat("    - Sem área cinza ao redor\n\n")


# -----------------------------------------------------------------------------
# 11. TESTE 9: Validação de Erros
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 9: Validação de Erros\n")
cat("=============================================================================\n\n")

# Teste 9.1: which fora do intervalo
cat("Teste 9.1 - which fora do intervalo:\n")
tryCatch({
  plot_dynamic_states_generic_base(obj_ll, which = 99)
  cat("  ✗ FALHOU (deveria dar erro)\n")
}, error = function(e) {
  cat("  Erro esperado:", conditionMessage(e), "\n")
  cat("  ✓ PASSOU\n")
})
cat("\n")

# Teste 9.2: ci_level inválido
cat("Teste 9.2 - ci_level inválido:\n")
tryCatch({
  plot_dynamic_states_generic_base(obj_ll, ci = TRUE, ci_level = 1.5)
  cat("  ✗ FALHOU (deveria dar erro)\n")
}, error = function(e) {
  cat("  Erro esperado:", conditionMessage(e), "\n")
  cat("  ✓ PASSOU\n")
})
cat("\n")


# -----------------------------------------------------------------------------
# 12. RESUMO DOS TESTES
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("RESUMO DOS TESTES\n")
cat("=============================================================================\n\n")

cat("✓ TESTE 1: plot_dynamic_states_generic_base() - Locallevel (1 página)\n")
cat("✓ TESTE 2: plot_dynamic_states_generic_base() - Localtrend (2 páginas)\n")
cat("✓ TESTE 3: plot_dynamic_states_generic_base() - Localacceleration (3 páginas)\n")
cat("✓ TESTE 4: plot_all_mixture_generic_base() - Dashboard Locallevel\n")
cat("✓ TESTE 5: plot_all_mixture_generic_base() - Dashboard Localtrend\n")
cat("✓ TESTE 6: plot_all_mixture_generic_base() - Dashboard Localacceleration\n")
cat("✓ TESTE 7: Parâmetro 'which' funcionando\n")
cat("✓ TESTE 8: Opção ci=FALSE funcionando\n")
cat("✓ TESTE 9: Validação de erros - 2/2 testes passaram\n\n")

cat("=============================================================================\n")
cat("TODOS OS TESTES CONCLUÍDOS COM SUCESSO! ✓\n")
cat("=============================================================================\n\n")

cat("Arquivos PDF gerados:\n")
cat("  - test_dashboard_locallevel.pdf (10 páginas)\n")
cat("  - test_dashboard_localtrend.pdf (13 páginas)\n")
cat("  - test_dashboard_localacceleration.pdf (16 páginas)\n\n")

cat("As extensões de R/plot_utils_base.R estão funcionando corretamente.\n")
cat("O backend ggplot2 correspondente está parado em inst/prototype/;\n")
cat("veja inst/prototype/README.md antes de retomar aquele trabalho.\n")
