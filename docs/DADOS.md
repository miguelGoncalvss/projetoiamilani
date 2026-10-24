# Contrato de dados (v0.2)

Todos os arquivos são CSV UTF-8, separados por vírgula, com cabeçalho, ponto
decimal e datas ISO (AAAA-MM-DD). Identificadores são strings: zeros à esquerda
são preservados. As movimentações precisam estar agregadas por SKU/dia.

## Base bruta de ERP (`erp_movements.csv`) e Adaptador (`adapter.py`)

Quando os dados vêm de um ERP transacional bruto (`generate-erp` / `adapt-erp`),
o adaptador limpa identificadores (`strip`), separa devoluções (`DEVOLUCAO_CLIENTE`)
e ajustes de inventário (`AJUSTE_INVENTARIO`) para auditoria em `adaptation_report.json`,
agrega saídas (`SAIDA_EXPEDICAO`), entradas (`ENTRADA_FORNECEDOR`) e rupturas
(`RUPTURA_REGISTRADA`), e reconstrói a grade diária explícita.

## history.csv

| Campo | Tipo | Significado |
|---|---|---|
| date | Data | Dia completo observado |
| sku | Texto | Identificador estável |
| outbound | Número >= 0 | Total de saídas observadas no dia |
| inbound | Número >= 0 | Total de entradas no dia |
| stockout | 0 ou 1 | Indicação de indisponibilidade |

Regras:

- Uma linha por dia e SKU; todos os SKUs devem cobrir o mesmo intervalo.
- Zero precisa ser confirmado no dado. Lacunas são rejeitadas.
- Na frequência mensal, o primeiro dia deve iniciar um mês e o último deve
  encerrar um mês. Meses parciais não são somados como se fossem completos.
- Quantidades negativas, infinitas e ausentes são rejeitadas.
- Ajustes e devoluções devem ser tratados antes da importação (via `adapt-erp`),
  não misturados a saídas negativas.
- Quando `--reconstruct-stockouts` é utilizado no comando `run`, os dias com
  `stockout=1` têm a demanda censurada reconstruída de forma estritamente causal
  (usando apenas observações passadas sem ruptura do mesmo SKU/dia da semana),
  registrando o total recuperado em `censoring_reconstruction.json` e `manifest.json`.

## stock.csv

| Campo | Obrigatório | Significado |
|---|---|---|
| sku | Sim | Mesmo identificador do histórico |
| snapshot_date | Sim | Data do fechamento do saldo |
| initial_stock | Sim | Saldo físico não negativo |
| unit_volume_m3 | Sim | Volume por unidade, estritamente positivo |
| region | Sim | Região alocada para o SKU |
| unit_weight_kg | Opcional (padrão 10.0) | Peso unitário em kg |
| units_per_pallet | Opcional (padrão 25) | Unidades por pallet fechado |
| pallet_height_m | Opcional (padrão 1.50) | Altura de 1 pallet carregado (m) |
| stackable_levels | Opcional (padrão 1) | Níveis máximos de empilhamento por posição |
| compatibility_class | Opcional (padrão GERAL) | Classe física/química (`SECO`, `REFRIGERADO`, `QUIMICO`) |
| lead_time_periods | Opcional (padrão 1) | Lead time em meses no modo mensal |
| lead_time_days | Opcional (padrão 5) | Lead time em dias no modo diário |
| order_multiple | Opcional (padrão 1) | Múltiplo de lote de compra/reposição |
| min_order_qty | Opcional (padrão 0) | Lote mínimo de reposição |
| safety_factor_z | Opcional (padrão 1.65) | Fator de segurança para ponto de pedido |

## receipts_monthly.csv / receipts_daily.csv

| Campo | Significado |
|---|---|
| sku | Produto |
| date | Primeiro dia do mês previsto, ou dia previsto na resolução diária |
| inbound | Recebimento planejado ou explicitamente simulado |

- No modo `--policy fixed_receipts`, é obrigatório preencher todos os SKU/períodos,
  inclusive os sem recebimento (`inbound=0`).
- No modo `--policy sQ`, os recebimentos são gerados dinamicamente pela política
  de Ponto de Pedido $(s, Q)$ respeitando `lead_time_periods`, estoque de segurança
  e lotes múltiplos (`replenishment_orders.csv`).

## regions.csv

| Campo | Obrigatório | Significado |
|---|---|---|
| region | Sim | Identificador único |
| capacity_m3 | Sim | Capacidade útil em metros cúbicos (> 0) |
| max_pallet_positions | Opcional | Capacidade máxima de posições-pallet na região |
| max_floor_weight_kg | Opcional | Capacidade máxima de carga/peso (kg) na região |
| max_rack_height_m | Opcional (padrão 2.50) | Altura máxima permitida por pallet no rack (m) |
| allowed_classes | Opcional (padrão `*`) | Classes compatíveis separadas por `|` (ex: `SECO|EMBALAGEM`) |

## Saídas adicionais da simulação

- `inventory.csv`: saldo inicial, recebimentos, atendimento, faltas, saldo final,
  volume, pallets, posições-pallet, peso (kg), verificação de compatibilidade e altura.
- `occupancy.csv`: ocupação volumétrica, ocupação de posições-pallet, ocupação de peso,
  excedentes físicos e gargalo dominante por região (`dominant_constraint_after_receipts`).
- `replenishment_orders.csv`: memória de cálculo da política $(s, Q)$ (quando `--policy sQ`).
- `joint_sku_risk.csv`: probabilidades de ruptura e quantis do saldo final por SKU.
- `joint_region_risk.csv`: bandas conjuntas de demanda, volume, posições-pallet, peso e
  probabilidade conjunta de superlotação (`overcapacity_any_constraint_prob_pct`) por região e `TOTAL_CD`.
