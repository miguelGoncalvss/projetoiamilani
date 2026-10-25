# Validação desta entrega (v0.2)

Executada em Python 3.13 (Windows), cobrindo o pipeline original e as 4 novas frentes.

## Testes automatizados

27 testes passaram (`python -m pytest -v`):

- 21 testes originais de validação temporal, ausência de vazamento, modelos, intervalos e conservação de estoque;
- 6 novos testes integrados cobrindo:
  1. Geração e adaptação de logs transacionais brutos de ERP (limpeza de SKUs, separação de devoluções e ajustes, grade diária completa);
  2. Reconstrução estritamente causal de demanda censurada (`stockout=1`) e invariância a alterações futuras;
  3. Adaptação automática e explícita de protocolo para históricos menores (60 meses) preservando o bloco de teste anual;
  4. Política de reposição $(s, Q)$ com lead time, estoque de segurança e arredondamento em múltiplos de lote/pallet;
  5. Restrições físicas de posições-pallet, peso, violação de altura de rack e incompatibilidade química/física de região;
  6. Alinhamento do lag sazonal por horizonte (`seasonal_target_lag`) e extrapolação de tendência por normalização de escala (`y / local_scale`) no XGBoost.

## Execução na nova base realista de ERP (`data/erp_raw` -> `data/erp_adapted`)

- **Escopo da base:** 15 SKUs distribuídos em 3 regiões (`R_SECO`, `R_REFRIG`, `R_QUIMICO`), 96 meses completos (`2018-01-01` a `2025-12-31`, 43.830 linhas no painel diário adaptado a partir de ~55,7 mil transações brutas de ERP).
- **Higienização e Censura:** 930 registros com espaços corrigidos, 834 devoluções e 495 ajustes de inventário isolados, e 300 dias de ruptura (`stockout=1`) reconstruídos causalmente.
- **Execução Mensal (`examples/erp_monthly_run`):**
  - `XGBoost` selecionado em 1º lugar após normalização por escala e lag sazonal alinhado ao horizonte (MAE médio de seleção caiu de 70,90 para **34,83**, superando `prophet` ≈ 35,17, `seasonal` ≈ 42,33 e `naive` ≈ 391,24);
  - WAPE acumulado no teste final de 2025: **1,32%** (3 meses), **1,26%** (6 meses) e **0,99%** (12 meses).
- **Execução Diária (`examples/erp_daily_run`):**
  - `XGBoost` selecionado em 1º lugar (MAE médio de seleção ≈ **4,35** vs `prophet` ≈ 4,36, `seasonal` ≈ 6,11 e `naive` ≈ 8,48);
  - WAPE acumulado no teste final: **5,99%** (7 dias) e **4,44%** (30 dias).
- **Simulações (`examples/erp_monthly_sim_sQ`, `examples/erp_monthly_sim_fixed`, `examples/erp_daily_sim_sQ`):**
  - Geração de pedidos pela política $(s, Q)$ (`replenishment_orders.csv`);
  - Identificação de gargalos físicos por região (`OVER_PALLET_POSITIONS` em `R_QUIMICO` devido a empilhamento único, e `OVER_VOLUME` em `R_REFRIG` e `R_SECO`);
  - Cálculo de risco conjunto com 300 trajetórias de Monte Carlo (`joint_sku_risk.csv` e `joint_region_risk.csv`).
