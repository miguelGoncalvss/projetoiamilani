================================================================================
PROJETO: ESTOQUE_IA (v0.2.0)
Motor de Previsão de Demanda, Política de Reposição (s, Q), Restrições Físicas
de Pallets e Simulação de Risco Conjunto para Gêmeo Digital (Digital Twin 3D)
================================================================================

Autores:
  - Miguel Pereira Gonçalves
  - João Guilherme da Silva Santana
  - Renato Martins de Oliveira
  - Rafael Silva Cruz
Instituição: Fundação Hermínio Ometto (FHO)
Orientação: Prof. Thiago Milani

--------------------------------------------------------------------------------
1. O QUE É O PROJETO
--------------------------------------------------------------------------------
O "estoque_ia" é o motor inteligente de previsão e simulação logística de um
Centro de Distribuição (CD). Ele transforma registros brutos de movimentação de
estoque (vindos de um ERP) em:
  1) Previsões de demanda de curto prazo (30 dias, com recorte de 7 dias) e de
     longo prazo (12 meses);
  2) Pedidos automáticos de reposição de estoque respeitando tempo de entrega
     dos fornecedores (Lead Time), estoque de segurança e lotes mínimos/pallets;
  3) Projeção física exata da ocupação do galpão em quantidade de pallets,
     posições-pallet nos racks, volume (m³), peso (kg), altura e compatibilidade
     de produtos por região (Seco, Refrigerado e Químico);
  4) Cálculo probabilístico de risco de falta de produto (ruptura) e risco de
     superlotação física do armazém para visualização em um Simulador 3D.


--------------------------------------------------------------------------------
2. PRINCIPAIS VANTAGENS E DIFERENCIAIS DO PROJETO
--------------------------------------------------------------------------------

[VANTAGEM 1] Inteligência Preditiva com Dupla Engenharia (XGBoost + Prophet)
Diferente de sistemas tradicionais que usam apenas médias móveis simples, o
projeto coloca 4 algoritmos para competir automaticamente:
  - Naive (Persistência)
  - Seasonal Naive (Repetição de ciclo sazonal)
  - Prophet (Modelo aditivo de séries temporais por SKU)
  - XGBoost (Árvores de Gradient Boosting globais com normalização por escala
    local, inclinação de tendência e lag sazonal alinhado ao horizonte futuro).
Com a normalização por escala ("y / local_scale") e o alinhamento do lag sazonal
ao horizonte previsto ("seasonal_target_lag"), o XGBoost consegue aprender
simultaneamente o comportamento de produtos de alto e baixo giro e extrapolar
tendências de crescimento, atingindo erro acumulado anual (WAPE) abaixo de 1%
(0,99%) nos testes de validação.

[VANTAGEM 2] Blindagem Contra Vazamento de Dados (Zero Data Leakage)
Muitos projetos acadêmicos erram ao misturar dados futuros no treinamento ou ao
escolher o modelo olhando para o conjunto de teste. Neste projeto:
  - Todos os atributos (lags, médias móveis e inclinação de tendência) são
    calculados estritamente até a data de origem;
  - O histórico é dividido em 5 blocos futuros independentes:
      * 2 blocos exclusivos para SELEÇÃO do melhor modelo;
      * 2 blocos exclusivos para CALIBRAÇÃO das faixas de incerteza (80% e 95%);
      * 1 bloco final blindado para TESTE (Holdout) do modelo vencedor.

[VANTAGEM 3] Recuperação de Demanda Reprimida (Tratamento de Stockout)
Em qualquer armazém real, quando o estoque acaba (stockout = 1), a venda/saída
registrada cai para zero ou fica abaixo da procura real. Se a IA treinar apenas
sobre a saída bruta, ela aprenderá que "vendeu pouco" e continuará comprando
pouco no futuro (ciclo vicioso de falta).
  - Nosso algoritmo identifica os dias de ruptura e reconstrói a demanda real
    reprimida de maneira estritamente causal (usando apenas o histórico passado
    do mesmo dia da semana sem ruptura), recuperando as unidades perdidas antes
    de treinar os modelos.

[VANTAGEM 4] Adaptador Robusto para Dados Brutos de ERP
Dados reais de empresas contêm falhas de digitação, múltiplas notas fiscais no
mesmo dia, devoluções de clientes e ajustes de inventário.
  - O módulo "adapter.py" limpa espaços em códigos de SKU, separa devoluções e
    ajustes contábeis (para não criar saídas negativas falsas), soma as saídas
    diárias e reconstrói a grade completa do calendário, gerando um relatório
    auditável ("adaptation_report.json").
  - Caso a empresa possua um histórico menor que 84 meses (ex: 48 ou 60 meses),
    o sistema possui ajuste assistido de protocolo ("--adapt-protocol") que
    adapta os blocos de seleção e calibração sem jamais encurtar o teste anual.

[VANTAGEM 5] Automação de Compras com Política (s, Q) Calibrada pela IA
Em vez de depender apenas de cronogramas manuais de recebimento, o módulo
"replenishment.py" automatiza a reposição de cada SKU:
  - Calcula a Demanda no Lead Time (soma do que vai sair enquanto o pedido está
    a caminho);
  - Dimensiona o Estoque de Segurança (SS) usando diretamente a margem de erro
    empírica de 95% calibrada pelo modelo de IA;
  - Dispara ordens de compra automaticamente sempre que a posição de estoque
    atinge o Ponto de Pedido (s), arredondando as quantidades em múltiplos
    exatos de pallets fechados ("order_multiple") e respeitando o lote mínimo.

[VANTAGEM 6] Física Real de Armazém (Muito Além de Metros Cúbicos)
Calcular apenas metros cúbicos (m³) não garante que a carga cabe no galpão.
O módulo "physical.py" valida 4 dimensões físicas reais em cada região:
  1) Posições-Pallet: converte unidades em pallets físicos e divide pelo limite
     de empilhamento permitido de cada produto ("stackable_levels");
  2) Altura do Porta-Pallet: verifica se a altura do pallet carregado respeita
     o vão máximo do rack da região ("max_rack_height_m");
  3) Peso Máximo (kg): soma o peso total dos produtos e compara com a capacidade
     estrutural da região ("max_floor_weight_kg");
  4) Compatibilidade Química/Física: impede que produtos Refrigerados ou
     Químicos sejam alocados em regiões incompatíveis;
  5) Diagnóstico de Gargalo: aponta automaticamente se a região estourou por
     falta de posições-pallet, volume, peso ou altura.

[VANTAGEM 7] Risco Conjunto via Simulação de Monte Carlo (300 Cenários)
Somar intervalos de confiança individuais de vários produtos é estatisticamente
incorreto porque ignora a correlação entre itens da mesma família e o acúmulo de
estoque de um mês para o outro.
  - O módulo "joint_risk.py" realiza Bootstrap Multivariado dos erros da IA,
    preservando a correlação real entre os SKUs no mesmo instante do tempo;
  - Simula 300 trajetórias completas do galpão inteiro e entrega a probabilidade
    exata (0% a 100%) de faltar qualquer produto ("stockout_probability_pct") e
    a probabilidade de cada região do galpão superlotar ("overcapacity_prob").

[VANTAGEM 8] Execução 100% Local, Segura e Pronta para o Simulador 3D
  - Não exige chave de API paga, não envia dados empresariais para a nuvem e
    não usa arquivos binários inseguros ("pickle"): todos os modelos são salvos
    em JSON legível.
  - Gera tabelas CSV padronizadas prontas para alimentar o Simulador 3D de
    Pallets em tempo real.


--------------------------------------------------------------------------------
3. COMO O PROJETO FUNCIONA (FLUXO PASSO A PASSO)
--------------------------------------------------------------------------------

PASSO 1 -> Ingestão e Limpeza (CLI: "generate-erp" e "adapt-erp")
  O sistema lê as movimentações brutas do ERP, limpa os identificadores, separa
  devoluções/ajustes e monta o arquivo padronizado "history.csv" junto com os
  cadastros físicos de saldo ("stock.csv") e regiões do galpão ("regions.csv").

PASSO 2 -> Reconstrução de Rupturas e Engenharia de Atributos ("features.py")
  Ao iniciar o comando "run --reconstruct-stockouts", o sistema corrige os dias
  em que houve falta de estoque de forma causal e transforma a série temporal em
  uma matriz supervisionada direta: cada linha representa um SKU, uma data de
  origem e um passo futuro "h" (1..12 meses ou 1..30 dias), contendo os lags
  recentes, médias móveis, senos/cossenos de calendário, inclinação de tendência
  e o lag sazonal exato daquele horizonte futuro.

PASSO 3 -> Seleção, Calibração e Teste ("pipeline.py" e "evaluation.py")
  1) Nos 2 blocos de Seleção, os 4 modelos (Naive, Seasonal, XGBoost, Prophet)
     fazem previsões e são ranqueados pelo menor MAE médio entre SKUs.
  2) O vencedor (XGBoost na base ERP) é congelado em "selected_model.json".
  3) Nos 2 blocos de Calibração, os resíduos escalados do vencedor são medidos
     para definir as larguras empíricas dos intervalos de 80% e 95% ("q80", "q95").
  4) No bloco de Teste Final, o sistema mede a precisão real (MAE, RMSE, WAPE e
     Viés) e a cobertura dos intervalos.
  5) Por fim, o modelo vencedor é reajustado (refit) com 100% do histórico e
     salva as previsões futuras em "forecast.csv" e os pesos em "model/".

PASSO 4 -> Simulação Logística, Física e Estocástica (CLI: "simulate")
  A partir de "forecast.csv", "stock.csv" e "regions.csv":
  1) Se a política escolhida for "--policy sQ", o módulo "replenishment.py"
     calcula período a período o Ponto de Pedido e o Estoque de Segurança,
     gerando os recebimentos automáticos em lotes de pallets e salvando a
     memória de cálculo em "replenishment_orders.csv".
  2) O módulo "inventory.py" atualiza os saldos de abertura, atendimento, faltas
     e fechamento de cada produto.
  3) O módulo "physical.py" converte os saldos (no pico logo após os recebimentos
     e no fim do período) em quantidade de pallets, posições-pallet ocupadas,
     volume (m³) e peso (kg), gerando "inventory.csv" e "occupancy.csv".
  4) O módulo "joint_risk.py" roda as 300 simulações conjuntas de Monte Carlo e
     exporta "joint_sku_risk.csv" e "joint_region_risk.csv".

PASSO 5 -> Visualização no Simulador 3D do Galpão
  O Simulador 3D lê os arquivos da pasta de simulação ("examples/erp_monthly_sim_sQ"
  e "examples/erp_daily_sim_sQ") para desenhar as 3 regiões do armazém (R_SECO,
  R_REFRIG, R_QUIMICO), posicionar a quantidade exata de pallets de cada SKU ao
  longo do tempo, destacar os pallets excedentes na área de overflow quando a
  ocupação passa de 100% e exibir os alertas de risco de ruptura e superlotação.

================================================================================
