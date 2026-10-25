# Previsão de Demanda, Política de Reposição $(s, Q)$, Restrições Físicas de Pallets e Risco Conjunto para Centro de Distribuição (CD)

Implementação executável completa (**v0.2.0**) do motor preditivo e de simulação logística descrito no pré-projeto IEEE.

- **Autores:** Miguel Pereira Gonçalves, João Guilherme da Silva Santana, Renato Martins de Oliveira e Rafael Silva Cruz.
- **Instituição:** Fundação Hermínio Ometto (FHO) — Orientação do Prof. Thiago Milani.

---

## 1. Visão Geral do Projeto

Este módulo é o núcleo de Inteligência Artificial e Simulação Operacional de um **Digital Twin (Gêmeo Digital) de Armazém**. Ele recebe registros históricos de movimentação de um Centro de Distribuição (CD), treina e seleciona modelos de aprendizado de máquina e séries temporais para prever saídas futuras, planeja automaticamente os pedidos de reposição de estoque e projeta a ocupação física de **pallets, volume ($m^3$) e peso ($kg$)** de cada região do galpão, alimentando o **Simulador 3D de Pallets**.

O sistema opera em duas resoluções temporais complementares e independentes:
1. **Horizonte Tático/Estratégico Mensal ($h = 1 \dots 12$ meses):** voltado ao planejamento anual de capacidade de armazenagem, contratos e picos sazonais.
2. **Horizonte Operacional Diário ($h = 1 \dots 30$ dias, com recorte imediato de 7 dias):** voltado à programação diária de docas, expedição e alertas de curto prazo.

> **Execução 100% local e auditável:** O módulo não depende de chaves de API externas nem de LLMs para cálculos numéricos. Todos os modelos treinados são persistidos em formato JSON (`xgboost.json`, `prophet_*.json`) sem uso de `pickle`.

---

## 2. Arquitetura Funcional (As 6 Camadas do Sistema)

### Camada 1 — Simulador de ERP Bruto e Adaptador de Dados (`src/estoque_ia/adapter.py`)
Em ambientes reais, os dados brutos de um ERP não vêm prontos como séries temporais limpas. Por isso, o projeto inclui:
- **Gerador de Movimentações de ERP (`generate-erp`):** cria uma base transacional realista contendo **15 SKUs** distribuídos em **3 regiões do galpão** (`R_SECO`, `R_REFRIG`, `R_QUIMICO`) ao longo de **8 anos completos (96 meses, 2018 a 2025)**. Simula ~55,7 mil transações brutas (`SAIDA_EXPEDICAO`, `ENTRADA_FORNECEDOR`, `DEVOLUCAO_CLIENTE`, `AJUSTE_INVENTARIO` e `RUPTURA_REGISTRADA`), incluindo múltiplas expedições no mesmo dia, códigos de SKU com espaços extras, choques correlacionados por família de produto, sazonalidade anual/semanal, tendência de crescimento e ruído de Poisson.
- **Adaptador para o Contrato do Pipeline (`adapt-erp`):**
  - Higieniza identificadores de SKU (`strip`).
  - Isola devoluções de clientes (`DEVOLUCAO_CLIENTE`) e ajustes de inventário (`AJUSTE_INVENTARIO`) em um relatório de auditoria (`adaptation_report.json`), evitando que devoluções ou baixas contábeis gerem saídas negativas artificiais.
  - Agrega as saídas e entradas por `SKU/dia` e reconstrói a grade diária explícita (preenchendo dias sem movimento com zero confirmado pelo calendário operacional).
- **Ajuste Assistido de Protocolo (`--adapt-protocol`):** o protocolo padrão de avaliação mensal utiliza 5 blocos de 12 meses (exigindo 84 meses no total). Caso uma base tenha histórico menor (ex.: 48 ou 60 meses), a flag `--adapt-protocol` ajusta o número de folds de seleção e calibração (mínimo de 1 cada) **preservando obrigatoriamente o bloco de teste final com o horizonte completo de 12 meses**, registrando a adaptação em `manifest.json`.

### Camada 2 — Tratamento de Demanda Censurada / Stockout (`--reconstruct-stockouts`)
Quando falta produto no estoque (`stockout = 1`), a **saída observada** é menor do que a **demanda real** (censura à direita). Treinar modelos apenas sobre a saída observada faz o algoritmo subestimar a necessidade futura.
- O módulo `reconstruct_censored_demand` reconstrói a demanda reprimida nos dias de `stockout = 1` de forma **estritamente causal** (sem olhar dados futuros, evitando *data leakage*).
- Para cada dia com ruptura, estima a demanda latente combinando a mediana e a média das últimas 8 ocorrências do mesmo dia da semana sem ruptura e a média das últimas 14 observações livres de ruptura daquele SKU, garantindo $\hat{d}_t \ge y_t$.
- Registra todas as unidades recuperadas em `censoring_reconstruction.json` e `manifest.json`.

### Camada 3 — Motor de Previsão, Seleção Temporal e Calibração (`models.py`, `features.py`, `evaluation.py`)
Quatro modelos competem em igualdade de condições:
1. **`naive` (Persistência):** repete a última observação conhecida na origem.
2. **`seasonal` (Sazonal Ingênuo):** repete o último ciclo completo observado (12 meses atrás no mensal; 7 dias atrás no diário).
3. **`xgboost` (Regressão Global Direta Normalizada por Escala e Tendência):** treina uma floresta de *Gradient Boosting* global entre todos os SKUs (`max_depth=3`, `learning_rate=0.05`, `n_estimators=100`) sobre o alvo normalizado pelo nível local na origem (`y / local_scale`). Utiliza **estratégia direta** com lag sazonal alinhado ao horizonte (`seasonal_target_lag`, capturando exatamente o mesmo mês do ano anterior ou mesmo dia da semana para cada passo futuro $h$), inclinação de tendência linear local (`slope_norm`), âncora sazonal-tendência (`seasonal_trend_anchor_norm`), lags ($1, 3, 6, 12$ meses ou $1, 7, 14, 28$ dias), médias/desvios móveis normalizados e codificação trigonométrica ($\sin/\cos$) de calendário.
4. **`prophet` (Modelo Aditivo por SKU):** ajusta um modelo por SKU. Na frequência mensal, desativa a série de Fourier contínua anual e utiliza regressores binários explícitos para os meses 2 a 12 (seguindo a recomendação técnica do Prophet para séries mensais); na frequência diária, utiliza sazonalidade semanal e anual automática.

**Protocolo de Avaliação sem Vazamento Temporal (`make_splits`):**
- O histórico é dividido em **5 blocos futuros não sobrepostos** do tamanho exato do horizonte (12 meses ou 30 dias):
  - **2 blocos de Seleção:** todos os 4 modelos são treinados com janela expansiva e avaliados. O modelo com menor **MAE médio entre SKUs** vence e é congelado (`selected_model.json`).
  - **2 blocos de Calibração:** apenas o modelo vencedor gera previsões fora da amostra para calcular os quantis empíricos de erro absoluto escalado (`q80` e `q95`) separados por horizonte $h$ (`calibration.csv`).
  - **1 bloco de Teste Final (Holdout):** avalia o modelo vencedor e a cobertura real das bandas de 80% e 95% em dados inéditos (`test_metrics_by_sku.csv`, `test_metrics_by_horizon.csv`, `test_cumulative_metrics.csv`, `test_interval_metrics.csv`).
- Após o teste, o modelo vencedor passa por um **refit final** com 100% do histórico para gerar `forecast.csv` e salvar os artefatos em `model/`.

### Camada 4 — Política de Reposição $(s, Q)$ vs. Recebimentos Fixos (`src/estoque_ia/replenishment.py`)
A simulação de estoque suporta dois modos de abastecimento (`--policy`):
1. **`fixed_receipts` (Recebimentos Fixos):** utiliza um cronograma estático de entradas (`receipts_monthly.csv` ou `receipts_daily.csv`) para avaliar cenários determinísticos ("o que acontece se mantivermos o plano atual de compras?").
2. **`sQ` (Política de Ponto de Pedido com Lead Time, Estoque de Segurança e Lotes):**
   - Calcula a **Demanda no Lead Time ($\text{LTD}_t$)** somando as previsões nos próximos $L$ períodos (`lead_time_periods` no mensal; `lead_time_days` no diário).
   - Calcula o **Estoque de Segurança ($SS_t$)** a partir da incerteza calibrada de 95% ($\text{upper\_95}_t - \hat{y}_t$) escalada por $(z / 1{,}65)\sqrt{L}$.
   - Define o **Ponto de Pedido ($s_t = \text{LTD}_t + SS_t$)**.
   - Monitora a **Posição de Estoque** ($\text{Saldo Projetado}_t + \text{Pedidos em Trânsito}$). Sempre que a posição cai em ou abaixo de $s_t$, emite automaticamente um pedido $Q_t$ respeitando o lote mínimo (`min_order_qty`) e arredondando para cima no múltiplo de pallet/lote (`order_multiple`), agendando a chegada para $t + L$ (`replenishment_orders.csv`).

### Camada 5 — Restrições Físicas de Armazenagem e Pallets (`src/estoque_ia/physical.py`)
Um galpão real não é limitado apenas por metros cúbicos ($m^3$). O módulo físico calcula para cada SKU e região (tanto no **pico logo após o recebimento** quanto no **fechamento do período**):
- **Quantidade de Pallets (`pallets_after_receipts`, `pallets_end`):** $\lceil \text{unidades} / \text{units\_per\_pallet} \rceil$.
- **Posições-Pallet Ocupadas (`pallet_positions_after_receipts`, `pallet_positions_end`):** divide o total de pallets pelo número efetivo de níveis de empilhamento permitidos pelo produto (`stackable_levels`).
- **Restrição de Altura de Rack (`height_violation`):** verifica se `pallet_height_m > max_rack_height_m` da região.
- **Capacidade de Peso no Piso/Estrutura (`weight_after_receipts_kg`, `max_floor_weight_kg`):** soma o peso total armazenado e calcula o percentual de ocupação de carga e o excesso em kg.
- **Compatibilidade Física/Química (`compatible_region`):** valida se a classe do SKU (`SECO`, `REFRIGERADO`, `QUIMICO`) é permitida na região (`allowed_classes`).
- **Identificação do Gargalo Dominante (`dominant_constraint_after_receipts`):** aponta automaticamente qual restrição estourou ou está mais próxima do limite em cada região (`OVER_PALLET_POSITIONS`, `OVER_VOLUME`, `OVER_WEIGHT`, `PHYSICAL_COMPATIBILITY_OR_HEIGHT` ou `OK_LEADING_*`).

### Camada 6 — Risco Conjunto via Monte Carlo / Bootstrap Multivariado (`src/estoque_ia/joint_risk.py`)
As bandas `lower_95` e `upper_95` de `forecast.csv` são marginais por SKU. Somar limites individuais de vários produtos ignora a correlação entre SKUs e o acúmulo de erros no saldo de estoque ao longo do tempo.
- O módulo `joint_risk.py` extrai os vetores de resíduos escalados da calibração alinhados por `(fold, h)` entre todos os SKUs simultaneamente, preservando a **correlação transversal entre produtos da mesma família/região**.
- Gera **$B = 300$ trajetórias conjuntas de demanda** via Bootstrap Multivariado e propaga o saldo de estoque, os pallets, o volume e o peso de todos os SKUs em paralelo.
- Produz probabilidades reais de operação:
  - `stockout_probability_pct`: probabilidade de faltar produto para cada SKU em cada período futuro (`joint_sku_risk.csv`).
  - `overcapacity_any_constraint_prob_pct`: probabilidade conjunta de uma região (ou o `TOTAL_CD`) estourar volume, posições-pallet ou peso (`joint_region_risk.csv`).

---

## 3. Instalação e Preparação do Ambiente

Requer **Python 3.11 ou superior** (validado em Python 3.12 no Linux e Python 3.13 no Windows).

### No Windows (PowerShell):
```powershell
# Opcional: criar ambiente virtual isolado
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Ou instalar diretamente no Python do sistema:
python -m pip install -e ".[dev]"
```

### No Linux / macOS:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

---

## 4. Guia Passo a Passo de Execução (Comandos CLI)

> **Regra de segurança de arquivos:** O sistema nunca sobrescreve uma pasta de saída já existente (para preservar a reprodutibilidade e auditoria dos experimentos). Sempre informe um caminho novo em `--out` ou remova a pasta anterior antes de reexecutar.

### Passo 1 — Gerar a Base Bruta Simulada de ERP
Gera os logs transacionais brutos (`erp_movements.csv`), catálogo de SKUs, regiões físicas e saldo inicial:
```powershell
python -m estoque_ia.cli generate-erp --out data/minha_base_erp_raw --skus 15 --seed 42
```

### Passo 2 — Executar o Adaptador de ERP
Limpa códigos de SKU, separa devoluções e ajustes em `adaptation_report.json` e gera os arquivos validados (`history.csv`, `stock.csv`, `regions.csv`, `receipts_monthly.csv`, `receipts_daily.csv`):
```powershell
python -m estoque_ia.cli adapt-erp --raw-dir data/minha_base_erp_raw --out data/minha_base_adaptada
```

### Passo 3 — Executar o Pipeline de Previsão (Mensal e Diário)
Treina os 4 modelos, seleciona o vencedor, calibra os intervalos empíricos, avalia no bloco de teste final e gera as previsões futuras com reconstrução de `stockout`:

**Execução Mensal (12 meses futuros):**
```powershell
python -m estoque_ia.cli run `
  --history data/minha_base_adaptada/history.csv `
  --config configs/monthly.json `
  --out outputs/execucao_mensal `
  --synthetic `
  --reconstruct-stockouts
```

**Execução Diária (30 dias futuros — filtre `h <= 7` para a visão de 7 dias):**
```powershell
python -m estoque_ia.cli run `
  --history data/minha_base_adaptada/history.csv `
  --config configs/daily.json `
  --out outputs/execucao_diaria `
  --synthetic `
  --reconstruct-stockouts
```
*(Dica: se estiver usando uma base com menos de 84 meses, adicione a flag `--adapt-protocol`).*

### Passo 4 — Simular Estoque, Posições-Pallet e Risco Conjunto
**Cenário A — Política Automática de Ponto de Pedido $(s, Q)$ (`--policy sQ`):**
```powershell
python -m estoque_ia.cli simulate `
  --forecast outputs/execucao_mensal/forecast.csv `
  --stock data/minha_base_adaptada/stock.csv `
  --receipts auto `
  --regions data/minha_base_adaptada/regions.csv `
  --out outputs/simulacao_mensal_sQ `
  --scenario politica_sQ_mensal `
  --policy sQ `
  --n-sims 300
```

**Cenário B — Plano de Recebimentos Fixos (`--policy fixed_receipts`):**
```powershell
python -m estoque_ia.cli simulate `
  --forecast outputs/execucao_mensal/forecast.csv `
  --stock data/minha_base_adaptada/stock.csv `
  --receipts data/minha_base_adaptada/receipts_monthly.csv `
  --regions data/minha_base_adaptada/regions.csv `
  --out outputs/simulacao_mensal_fixa `
  --scenario recebimentos_fixos_mensal `
  --policy fixed_receipts `
  --n-sims 300
```

### Passo 5 — Reproduzir Previsões a partir do Modelo Salvo em JSON
```powershell
python -m estoque_ia.cli predict-saved `
  --model-dir outputs/execucao_mensal/model `
  --out outputs/previsao_reproduzida.csv
```

---

## 5. Resultados Validados na Base Realista de ERP (`examples/`)

Os resultados abaixo já estão gerados e versionados nas pastas `data/erp_adapted/`, `examples/erp_monthly_run/`, `examples/erp_daily_run/`, `examples/erp_monthly_sim_sQ/` e `examples/erp_daily_sim_sQ/`.

### 5.1. Higienização e Recuperação de Demanda Censurada
- **Transações brutas processadas:** 55.711 linhas (`data/erp_raw/erp_movements.csv`).
- **Limpeza do Adaptador:** 930 registros com espaços corrigidos, 834 devoluções de clientes isoladas e 495 ajustes de inventário separados (`data/erp_adapted/adaptation_report.json`).
- **Demanda Censurada (`stockout=1`):** 300 dias de ruptura reconstruídos causalmente antes do treinamento (`examples/erp_monthly_run/censoring_reconstruction.json`).

### 5.2. Desempenho dos Modelos (Seleção e Teste Final)

| Resolução | Ranking de Seleção (MAE médio por SKU) | Vencedor | Erro Acumulado no Teste Final (WAPE %) |
|---|---|---|---|
| **Mensal (12 meses)** | **1º `xgboost` (34,83)** <br> 2º `prophet` (35,17) <br> 3º `seasonal` (42,33) <br> 4º `naive` (391,24) | **`xgboost`** | • Horizonte 3 meses: **1,32%** <br> • Horizonte 6 meses: **1,26%** <br> • Horizonte 12 meses: **0,99%** |
| **Diária (30 dias)** | **1º `xgboost` (4,35)** <br> 2º `prophet` (4,36) <br> 3º `seasonal` (6,11) <br> 4º `naive` (8,48) | **`xgboost`** | • Primeiros 7 dias ($h \le 7$): **5,99%** <br> • Horizonte 30 dias ($h \le 30$): **4,44%** |

---

## 6. Guia de Integração para o Simulador 3D de Pallets

Esta seção documenta exatamente quais arquivos e colunas o integrante responsável pelo **Simulador 3D do Galpão (Three.js / Web / Engine 3D)** deve consumir nas pastas de simulação (`examples/erp_monthly_sim_sQ/` para visão mensal de 12 meses e `examples/erp_daily_sim_sQ/` para visão diária de 30 dias).

### 6.1. Cadastro das 3 Regiões do Galpão (`data/erp_adapted/regions.csv`)
Define a geometria e os limites físicos de cada zona do armazém:

| Região | Classe Permitida | Capacidade Volumétrica (`capacity_m3`) | Limite de Posições-Pallet (`max_pallet_positions`) | Limite de Peso (`max_floor_weight_kg`) | Altura Máx. Rack (`max_rack_height_m`) |
|---|---|---|---|---|---|
| **`R_SECO`** | `SECO\|EMBALAGEM` | $420,0\text{ m}^3$ | **320 posições** | $195.000\text{ kg}$ | $2,20\text{ m}$ |
| **`R_REFRIG`** | `REFRIGERADO` | $260,0\text{ m}^3$ | **190 posições** | $115.000\text{ kg}$ | $1,90\text{ m}$ |
| **`R_QUIMICO`** | `QUIMICO` | $190,0\text{ m}^3$ | **140 posições** | $98.000\text{ kg}$ | $1,80\text{ m}$ |

### 6.2. Pallets por Produto e Período (`inventory.csv`)
Contém uma linha por `(sku, date)` para instanciar e desenhar os pallets de cada produto no galpão 3D:
- `sku`, `date`, `region`, `compatibility_class`: identificação do produto, período da timeline e região onde está alocado.
- `pallets_after_receipts` e `pallets_end`: **quantidade total de pallets físicos** daquele SKU no pico (logo após a chegada dos caminhões/recebimentos) e no fechamento do período.
- `pallet_positions_after_receipts` e `pallet_positions_end`: **quantidade de posições-pallet ocupadas** no rack/piso (já descontando o empilhamento permitido por `effective_stack_levels`, que é `2` para `R_SECO` e `1` para `R_REFRIG` e `R_QUIMICO`).
- `pallet_height_m` e `unit_weight_kg`: dimensões e peso para exibir no *tooltip* ao clicar em um pallet no 3D.
- `opening_stock`, `inbound`, `forecast_outbound`, `fulfilled`, `unmet_scenario`, `closing_stock`: balanço de unidades naquele período.

### 6.3. Ocupação, Excesso (*Overflow*) e Gargalos por Região (`occupancy.csv`)
Contém uma linha por `(region, date)` para controlar as barras de capacidade, cores dos porta-pallets e área de blocagem/excesso no 3D:
- `required_pallet_positions_after_receipts` e `required_pallet_occupancy_after_receipts_pct`: posições exigidas e percentual de ocupação das posições-pallet na região.
- `excess_pallet_positions_after_receipts`: **quantas posições-pallet ultrapassaram 100% da capacidade da região** (esses pallets excedentes podem ser renderizados no corredor/área de *overflow* com alerta visual).
- `required_volume_after_receipts_m3`, `required_occupancy_after_receipts_pct`, `excess_after_receipts_m3`: ocupação e excesso em metros cúbicos.
- `required_weight_after_receipts_kg`, `required_weight_occupancy_after_receipts_pct`, `excess_weight_after_receipts_kg`: ocupação e excesso de carga em kg.
- `dominant_constraint_after_receipts`: indica qual limite físico é o gargalo principal da região naquele instante:
  - Ex.: Em `R_QUIMICO`, o gargalo dominante é `OVER_PALLET_POSITIONS` (pois não empilha), enquanto em `R_REFRIG` e `R_SECO` é `OVER_VOLUME`.

### 6.4. Alertas Probabilísticos de Risco Conjunto (`joint_region_risk.csv` e `joint_sku_risk.csv`)
Resultados das 300 simulações conjuntas de Monte Carlo para exibir indicadores de risco no HUD do Simulador 3D:
- Em **`joint_region_risk.csv`** (por `region` e também para `TOTAL_CD`):
  - `overcapacity_any_constraint_prob_pct`: probabilidade (0% a 100%) de a região sofrer superlotação física.
  - `any_sku_stockout_prob_pct`: probabilidade de ocorrer ruptura de pelo menos um SKU daquela região.
  - `peak_volume_p50_m3`, `peak_volume_p90_m3`, `peak_volume_p95_m3` e `peak_pallet_positions_p95`: quantis de estresse de espaço.
- Em **`joint_sku_risk.csv`** (por `sku` e `date`):
  - `stockout_probability_pct`: probabilidade de ruptura do SKU naquele período.
  - `closing_stock_p05`, `closing_stock_p50`, `closing_stock_p95`: faixa probabilística do saldo final.

### 6.5. Pedidos Emitidos pela Política $(s, Q)$ (`replenishment_orders.csv`)
Gerado quando `--policy sQ` é utilizado:
- `reorder_point_s`, `safety_stock`, `lead_time_demand`, `inventory_position_before_order`, `order_placed_Q`, `inbound_arriving`, `scheduled_delivery_date`.

---

## 7. Estrutura de Pastas e Módulos

```text
estoque_ia/
├── configs/
│   ├── monthly.json               # Configuração para horizonte de 12 meses
│   └── daily.json                 # Configuração para horizonte de 30 dias
├── data/
│   ├── demo/                      # Base sintética simples original (10 SKUs)
│   ├── erp_raw/                   # Logs brutos de ERP gerados (15 SKUs, 55,7 mil transações)
│   └── erp_adapted/               # Base ERP limpa e validada pelo adaptador + relatório JSON
├── docs/
│   ├── DADOS.md                   # Contrato detalhado de todos os CSVs de entrada e saída
│   ├── METODOLOGIA.md             # Formulação matemática dos modelos, (s, Q), pallets e Monte Carlo
│   ├── VALIDACAO.md               # Relatório dos 26 testes automatizados e métricas da base ERP
│   └── REFERENCIAS.md             # Artigos científicos IEEE/IJF e documentação oficial
├── examples/
│   ├── erp_monthly_run/           # Execução mensal completa na base ERP (Prophet vencedor)
│   ├── erp_daily_run/             # Execução diária completa na base ERP (Prophet vencedor)
│   ├── erp_monthly_sim_sQ/        # Simulação mensal com política (s, Q), pallets e risco conjunto
│   ├── erp_monthly_sim_fixed/     # Simulação mensal com recebimentos fixos
│   └── erp_daily_sim_sQ/          # Simulação diária (30 dias) com política (s, Q) e risco conjunto
├── src/estoque_ia/
│   ├── adapter.py                 # Gerador ERP bruto, adaptador, stockout causal e ajuste de protocolo
│   ├── cli.py                     # Interface de linha de comando (generate-erp, adapt-erp, run, simulate)
│   ├── config.py                  # Dataclass de configuração e validação de parâmetros
│   ├── data.py                    # Validação estrita do contrato de dados e montagem de painel
│   ├── evaluation.py              # Splits temporais, métricas (MAE, RMSE, WAPE, Viés) e calibração
│   ├── features.py                # Engenharia de atributos diretos na origem (lags, janelas, calendário)
│   ├── inventory.py               # Simulador de estoque integrando reposição, física e risco conjunto
│   ├── joint_risk.py              # Monte Carlo / Bootstrap Multivariado para risco conjunto
│   ├── models.py                  # Modelos Naive, Seasonal, XGBoost e Prophet + persistência JSON
│   ├── pipeline.py                # Orquestração fim a fim do treinamento, seleção, calibração e teste
│   ├── physical.py                # Cálculo de pallets, posições-pallet, empilhamento, peso e altura
│   ├── replenishment.py           # Política de reposição (s, Q) com Lead Time, SS e lote múltiplo
│   └── synthetic.py               # Gerador sintético básico da v0.1
└── tests/
    ├── test_forecasting.py        # 18 testes de previsão, splits, vazamento temporal e serialização
    ├── test_inventory.py          # 3 testes de conservação de saldo e validação de entradas
    └── test_advanced_modules.py   # 5 testes integrados (ERP, stockout causal, protocolo, sQ, física e risco)
```

---

## 8. Testes Automatizados

Para executar toda a suíte de verificação (26 testes cobrindo integridade de dados, ausência de vazamento temporal, serialização dos 4 modelos, política $(s, Q)$, restrições físicas e risco conjunto):

```powershell
python -m pytest -v
```
