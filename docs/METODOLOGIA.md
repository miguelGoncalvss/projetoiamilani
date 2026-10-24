# Metodologia implementada (v0.2)

## Escopo

A implementação transforma transações de ERP ou saídas observadas em previsões,
reconstrói demanda censurada em rupturas, aplica política de reposição $(s, Q)$
ou recebimentos fixos, avalia restrições físicas multi-dimensionais (volume,
posições-pallet, peso, altura e compatibilidade) e quantifica o risco conjunto
de estoque e superlotação via simulação de Monte Carlo / Bootstrap Multivariado.

## 1. Adaptador de ERP, Protocolo Adaptativo e Demanda Censurada

- **Adaptador (`adapter.py`):** higieniza códigos de SKU, separa devoluções de
  clientes e ajustes de inventário para não distorcer a série de saídas, agrega
  múltiplas expedições diárias e reconstrói a grade diária completa.
- **Reconstrução causal de demanda censurada (`reconstruct_censored_demand`):**
  quando `stockout = 1`, a saída observada é um limite inferior da demanda real.
  Para evitar vazamento temporal, cada dia censurado $t$ é estimado usando apenas
  observações anteriores ($k < t$) sem ruptura do mesmo SKU (combinando a mediana
  e a média das últimas 8 ocorrências do mesmo dia da semana com a média das
  últimas 14 observações livres de ruptura), garantindo $\hat{d}_t \ge y_t$.
- **Ajuste explícito de protocolo (`adapt_config_to_history`):** quando acionado
  com `--adapt-protocol` em históricos menores que 84 meses (ex.: 48 ou 60 meses),
  reduz os blocos de seleção e calibração (mínimo de 1 bloco cada) preservando
  integralmente o bloco de teste final no horizonte completo (12 meses ou 30 dias).

## 2. Modelos e Previsão Direta

1. **Naive:** repete o último valor disponível.
2. **Seasonal:** repete o último ciclo observado (12 meses ou 7 dias).
3. **XGBoost:** regressão global direta entre SKUs, árvores de profundidade 3,
   100 estimadores por padrão, aprendizado 0,05, semente fixa e CPU com 2 threads.
   Usa lags, médias/desvios móveis na origem, codificação cíclica de calendário
   e o horizonte $h$ como atributo explícito.
4. **Prophet:** um modelo por SKU; no mensal, desativa a sazonalidade anual contínua
   e usa indicadores binários dos meses 2 a 12; no diário, usa sazonalidade semanal
   e anual automática.

## 3. Política de Reposição $(s, Q)$ (`replenishment.py`)

Para cada SKU e período $t$:
- **Demanda no Lead Time ($\text{LTD}_t$):** soma das previsões nos próximos $L$
  períodos (`lead_time_periods` no mensal ou `lead_time_days` no diário).
- **Estoque de Segurança ($SS_t$):** derivado da meia-largura empírica de 95%
  ($w_{95, t} = \max(0, \text{upper\_95}_t - \hat{y}_t)$) escalada por
  $(z / 1{,}65) \sqrt{L}$.
- **Ponto de Pedido ($s_t$):** $s_t = \text{LTD}_t + SS_t$.
- **Posição de Estoque ($\text{IP}_t$):** saldo projetado ao final do período
  mais pedidos já emitidos em trânsito para chegada futura.
- **Emissão de Pedido ($Q_t$):** quando $\text{IP}_t \le s_t$, emite pedido para
  recompor o nível-alvo $S_t$, respeitando o lote mínimo (`min_order_qty`) e
  arredondando para cima no múltiplo de lote/pallet (`order_multiple`), com
  entrega agendada em $t + L$.

## 4. Restrições Físicas de Armazenagem (`physical.py`)

Além da cubagem ($m^3$), cada SKU e região são avaliados em:
- **Pallets e Posições-Pallet:** $\text{pallets} = \lceil \text{unidades} / \text{units\_per\_pallet} \rceil$
  e $\text{posições} = \lceil \text{pallets} / \text{effective\_stack\_levels} \rceil$.
- **Altura e Empilhamento:** verifica se `pallet_height_m > max_rack_height_m`
  (sinalizando violação física e restringindo o empilhamento).
- **Peso Total (kg):** $\text{unidades} \times \text{unit\_weight\_kg}$ frente a
  `max_floor_weight_kg` da região.
- **Compatibilidade Química/Física:** valida se `compatibility_class` do SKU
  pertence às classes permitidas (`allowed_classes`) da região.
- **Gargalo Dominante:** classifica cada região/período conforme a restrição mais
  pressionada (`VOLUME`, `PALLET_POSITIONS`, `WEIGHT` ou `PHYSICAL_COMPATIBILITY_OR_HEIGHT`).

## 5. Risco Conjunto via Trajetórias de Monte Carlo (`joint_risk.py`)

Como os intervalos de `forecast.csv` são marginais por SKU/horizonte, somar seus
limites subestima ou superestima o risco agregado do CD. O módulo `joint_risk.py`:
1. Extrai os vetores multivariados de resíduos escalados da calibração alinhados
   por `(fold, h)` entre todos os SKUs, preservando a correlação transversal entre
   produtos da mesma família/região.
2. Simula $B$ trajetórias conjuntas completas ($B=300$ por padrão) propagando o
   estoque período a período para todos os SKUs simultaneamente.
3. Calcula bandas conjuntas de 80% e 95% para demanda, volume, posições-pallet e
   peso por região e para o `TOTAL_CD`, além de `stockout_probability_pct` e
   `overcapacity_any_constraint_prob_pct`.
