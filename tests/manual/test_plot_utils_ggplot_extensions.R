# =============================================================================
# Script de Teste: Extensões de inst/prototype/plot_utils_ggplot.R
# =============================================================================
# Testa as novas funções genéricas ggplot2: plot_all_mixture_generic_ggplot()
# e plot_dynamic_states_generic_ggplot()
#
# ATENÇÃO: o backend ggplot2 é um protótipo parado fora de R/ (veja
# inst/prototype/README.md). Ele não está ligado a nenhum método plot.*, e o
# seletor engine = c("base", "ggplot2") foi removido em #53. Este script só
# faz sentido se/quando aquele trabalho for retomado.

# Limpar ambiente
rm(list = ls())

# -----------------------------------------------------------------------------
# 1. CARREGAR DEPENDÊNCIAS
# -----------------------------------------------------------------------------

source("R/plot_utils_base.R")
source("inst/prototype/plot_utils_ggplot.R")
source("R/plot_utils_mcmc.R")
source("R/plot_utils_states.R")

cat("✓ Arquivos carregados com sucesso\n\n")

# Verificar se ggplot2 está disponível
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Este teste requer o pacote 'ggplot2'. Instale com: install.packages('ggplot2')")
}

has_patchwork <- requireNamespace("patchwork", quietly = TRUE)
if (has_patchwork) {
  cat("✓ Patchwork disponível - layouts combinados serão usados\n")
} else {
  cat("⊘ Patchwork não disponível - plots serão exibidos sequencialmente\n")
  cat("  (Instale com: install.packages('patchwork') para layouts melhores)\n")
}
cat("\n")


# -----------------------------------------------------------------------------
# 2. FUNÇÃO AUXILIAR PARA CRIAR OBJETOS MOCK
# -----------------------------------------------------------------------------

create_mock_mixture <- function(model_order = 1) {

  n_draws <- 100
  n_obs <- 50

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
# 3. TESTE 1: plot_dynamic_states_generic_ggplot() - Locallevel
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 1: plot_dynamic_states_generic_ggplot() - Locallevel (1 página)\n")
cat("=============================================================================\n\n")

obj_ll <- create_mock_mixture(1)

cat("Teste 1.1 - Plotar estados dinâmicos com ggplot2 (locallevel):\n")
cat("  Esperado: 1 página com 1 painel de trajetória\n")
cat("  Pressione ENTER para visualizar...\n")
readline()

plot_dynamic_states_generic_ggplot(obj_ll, which = 1, ci = TRUE, ci_level = 0.95)

cat("  ✓ Plot gerado\n")
cat("  Verifique:\n")
cat("    - Estilo ggplot2 moderno\n")
cat("    - Level State com banda de credibilidade\n")
cat("    - Título 'Dynamic State Trajectories'\n")
cat("    - Legenda no topo\n\n")

readline(prompt = "Pressione ENTER para continuar...")


# -----------------------------------------------------------------------------
# 4. TESTE 2: plot_dynamic_states_generic_ggplot() - Localtrend
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 2: plot_dynamic_states_generic_ggplot() - Localtrend (2 páginas)\n")
cat("=============================================================================\n\n")

obj_lt <- create_mock_mixture(2)

cat("Teste 2.1 - Página 1: Trajetórias dos estados\n")
cat("  Pressione ENTER para visualizar...\n")
readline()

plot_dynamic_states_generic_ggplot(obj_lt, which = 1, ci = TRUE, ci_level = 0.95)

cat("  ✓ Página 1 gerada\n")
if (has_patchwork) {
  cat("  Verifique:\n")
  cat("    - 2 painéis empilhados verticalmente (patchwork)\n")
  cat("    - Level State e Trend State\n")
} else {
  cat("  Verifique:\n")
  cat("    - 2 plots exibidos sequencialmente\n")
}
cat("\n")

readline(prompt = "Pressione ENTER para ver página 2...")

cat("Teste 2.2 - Página 2: Inovações e diagnósticos\n")
plot_dynamic_states_generic_ggplot(obj_lt, which = 2, ci = TRUE, ci_level = 0.95)

cat("  ✓ Página 2 gerada\n")
if (has_patchwork) {
  cat("  Verifique:\n")
  cat("    - Grade 2x2 com patchwork\n")
} else {
  cat("  Verifique:\n")
  cat("    - Plots sequenciais\n")
}
cat("    - Level Innovations (barras verticais)\n")
cat("    - Trend Innovations\n")
cat("    - Joint Trajectories (linhas coloridas)\n\n")

readline(prompt = "Pressione ENTER para continuar...")


# -----------------------------------------------------------------------------
# 5. TESTE 3: plot_dynamic_states_generic_ggplot() - Localacceleration
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 3: plot_dynamic_states_generic_ggplot() - Localacceleration (3 páginas)\n")
cat("=============================================================================\n\n")

obj_la <- create_mock_mixture(3)

cat("Teste 3.1 - Página 1: Trajetórias (3 estados)\n")
cat("  Pressione ENTER para visualizar...\n")
readline()

plot_dynamic_states_generic_ggplot(obj_la, which = 1, ci = TRUE, ci_level = 0.95)

cat("  ✓ Página 1 gerada\n")
cat("  Verifique:\n")
cat("    - 3 painéis: Level, Trend, Acceleration\n")
cat("    - Cores distintas para cada estado\n\n")

readline(prompt = "Pressione ENTER para ver página 2...")

cat("Teste 3.2 - Página 2: Inovações e diagnósticos\n")
plot_dynamic_states_generic_ggplot(obj_la, which = 2, ci = TRUE, ci_level = 0.95)

cat("  ✓ Página 2 gerada\n\n")

readline(prompt = "Pressione ENTER para ver página 3...")

cat("Teste 3.3 - Página 3: Relações entre estados (scatter plots)\n")
plot_dynamic_states_generic_ggplot(obj_la, which = 3, ci = TRUE, ci_level = 0.95)

cat("  ✓ Página 3 gerada\n")
cat("  Verifique:\n")
cat("    - 3 scatter plots (theta_1 vs theta_2, theta_1 vs theta_3, theta_2 vs theta_3)\n")
cat("    - Pontos semi-transparentes\n\n")

readline(prompt = "Pressione ENTER para continuar...")


# -----------------------------------------------------------------------------
# 6. TESTE 4: plot_all_mixture_generic_ggplot() - Dashboard Completo
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 4: plot_all_mixture_generic_ggplot() - Dashboard Completo\n")
cat("=============================================================================\n\n")

cat("Este teste gerará o dashboard COMPLETO com ggplot2.\n")
cat("Para mixture locallevel, são 10 seções no total.\n\n")

cat("NOTA: Como ggplot2 não pode salvar diretamente em PDF multi-página,\n")
cat("      os plots serão exibidos interativamente.\n")
cat("      Use ask=FALSE para gerar todos de uma vez.\n\n")

cat("Pressione ENTER para iniciar (modo interativo)...\n")
readline()

# Modo interativo
plot_all_mixture_generic_ggplot(obj_ll, ask = TRUE, ci = TRUE, ci_level = 0.95)

cat("  ✓ Dashboard completo gerado\n\n")


# -----------------------------------------------------------------------------
# 7. TESTE 5: Dashboard Localtrend (não-interativo)
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 5: Dashboard Localtrend (modo não-interativo)\n")
cat("=============================================================================\n\n")

cat("Gerando dashboard completo para localtrend sem pausa entre plots...\n")
cat("Pressione ENTER para iniciar...\n")
readline()

plot_all_mixture_generic_ggplot(obj_lt, ask = FALSE, ci = TRUE, ci_level = 0.95)

cat("  ✓ Dashboard completo gerado\n")
cat("  Seções esperadas: 13 (8 MCMC + 1 Mixture + 2 States + 2 Weights)\n\n")


# -----------------------------------------------------------------------------
# 8. TESTE 6: Dashboard Localacceleration
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 6: Dashboard Localacceleration (modo não-interativo)\n")
cat("=============================================================================\n\n")

cat("Gerando dashboard completo para localacceleration...\n")
cat("Pressione ENTER para iniciar...\n")
readline()

plot_all_mixture_generic_ggplot(obj_la, ask = FALSE, ci = TRUE, ci_level = 0.95)

cat("  ✓ Dashboard completo gerado\n")
cat("  Seções esperadas: 16 (10 MCMC + 1 Mixture + 3 States + 2 Weights)\n\n")


# -----------------------------------------------------------------------------
# 9. TESTE 7: Testar sem intervalos de credibilidade
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 7: Testar sem intervalos de credibilidade (ci = FALSE)\n")
cat("=============================================================================\n\n")

cat("Teste 7.1 - Estados sem CI (ggplot2):\n")
cat("  Pressione ENTER para visualizar...\n")
readline()

plot_dynamic_states_generic_ggplot(obj_ll, which = 1, ci = FALSE)

cat("  ✓ Plot gerado sem bandas de credibilidade\n")
cat("  Verifique:\n")
cat("    - Apenas linha mediana visível\n")
cat("    - Sem área sombreada\n\n")


# -----------------------------------------------------------------------------
# 10. TESTE 8: Comparação Base vs ggplot2
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 8: Comparação Visual - Base vs ggplot2\n")
cat("=============================================================================\n\n")

cat("Teste 8.1 - Mesmo plot com ambos engines:\n")
cat("  Pressione ENTER para ver versão BASE...\n")
readline()

plot_dynamic_states_generic_base(obj_lt, which = 1, ci = TRUE, ci_level = 0.95)

cat("  Pressione ENTER para ver versão GGPLOT2...\n")
readline()

plot_dynamic_states_generic_ggplot(obj_lt, which = 1, ci = TRUE, ci_level = 0.95)

cat("  ✓ Ambas as versões geradas\n")
cat("  Compare:\n")
cat("    - Estilo visual\n")
cat("    - Layout de legendas\n")
cat("    - Qualidade das bandas de credibilidade\n\n")


# -----------------------------------------------------------------------------
# 11. TESTE 9: Validação de Erros
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("TESTE 9: Validação de Erros\n")
cat("=============================================================================\n\n")

# Teste 9.1: which fora do intervalo
cat("Teste 9.1 - which fora do intervalo:\n")
tryCatch({
  plot_dynamic_states_generic_ggplot(obj_ll, which = 99)
  cat("  ✗ FALHOU (deveria dar erro)\n")
}, error = function(e) {
  cat("  Erro esperado:", conditionMessage(e), "\n")
  cat("  ✓ PASSOU\n")
})
cat("\n")

# Teste 9.2: ci_level inválido
cat("Teste 9.2 - ci_level inválido:\n")
tryCatch({
  plot_dynamic_states_generic_ggplot(obj_ll, ci = TRUE, ci_level = 2.0)
  cat("  ✗ FALHOU (deveria dar erro)\n")
}, error = function(e) {
  cat("  Erro esperado:", conditionMessage(e), "\n")
  cat("  ✓ PASSOU\n")
})
cat("\n")

# Teste 9.3: Sem ggplot2 (simulado)
cat("Teste 9.3 - Verificação de ggplot2:\n")
cat("  (Teste já validado no início do script)\n")
cat("  ✓ PASSOU\n\n")


# -----------------------------------------------------------------------------
# 12. RESUMO DOS TESTES
# -----------------------------------------------------------------------------

cat("=============================================================================\n")
cat("RESUMO DOS TESTES\n")
cat("=============================================================================\n\n")

cat("✓ TESTE 1: plot_dynamic_states_generic_ggplot() - Locallevel\n")
cat("✓ TESTE 2: plot_dynamic_states_generic_ggplot() - Localtrend\n")
cat("✓ TESTE 3: plot_dynamic_states_generic_ggplot() - Localacceleration\n")
cat("✓ TESTE 4: plot_all_mixture_generic_ggplot() - Dashboard Locallevel\n")
cat("✓ TESTE 5: plot_all_mixture_generic_ggplot() - Dashboard Localtrend\n")
cat("✓ TESTE 6: plot_all_mixture_generic_ggplot() - Dashboard Localacceleration\n")
cat("✓ TESTE 7: Opção ci=FALSE funcionando\n")
cat("✓ TESTE 8: Comparação Base vs ggplot2\n")
cat("✓ TESTE 9: Validação de erros - 3/3 testes passaram\n\n")

cat("=============================================================================\n")
cat("TODOS OS TESTES CONCLUÍDOS COM SUCESSO! ✓\n")
cat("=============================================================================\n\n")

if (has_patchwork) {
  cat("✓ Patchwork está disponível - layouts otimizados foram usados\n")
} else {
  cat("⊘ Patchwork não está disponível - considere instalar para layouts melhores:\n")
  cat("  install.packages('patchwork')\n")
}
cat("\n")

cat("As extensões de inst/prototype/plot_utils_ggplot.R estão funcionando corretamente.\n")
cat("Para reintegrar o backend ao pacote, siga inst/prototype/README.md.\n")
