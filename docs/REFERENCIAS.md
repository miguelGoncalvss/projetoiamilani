# Referências científicas e técnicas

1. Z. U. Rizqi, S. Y. Chou e W. N. Cahyo (2024).
   A simulation-based Digital Twin for smart warehouse: Towards standardization.
   Decision Analytics Journal, 12, 100509.
   https://doi.org/10.1016/j.dajour.2024.100509

2. T. Chen e C. Guestrin (2016).
   XGBoost: A Scalable Tree Boosting System. ACM SIGKDD, 785–794.
   https://doi.org/10.1145/2939672.2939785
   Versão aberta: https://arxiv.org/abs/1603.02754

3. S. J. Taylor e B. Letham (2017).
   Forecasting at Scale. Preprint utilizado no pré-projeto.
   https://facebook.github.io/prophet/static/prophet_paper_20170113.pdf

4. S. Makridakis et al. (2022).
   The M5 uncertainty competition: Results, findings and conclusions.
   International Journal of Forecasting, 38(4), 1365–1385.
   https://doi.org/10.1016/j.ijforecast.2021.10.009

5. J. R. Trapero, E. Holgado de Frutos e D. J. Pedregal (2024).
   Demand forecasting under lost sales stock policies.
   International Journal of Forecasting, 40(3), 1055–1068.
   https://doi.org/10.1016/j.ijforecast.2023.09.004

Documentação consultada para a implementação em 16/09/2026:

- API Python do XGBoost:
  https://xgboost.readthedocs.io/en/stable/python/python_api.html
- Prophet: início rápido:
  https://facebook.github.io/prophet/docs/quick_start.html
- Prophet: dados mensais e outras frequências:
  https://facebook.github.io/prophet/docs/non-daily_data.html

O uso de indicadores de mês no Prophet segue a discussão da documentação
sobre dados mensais. A estratégia direta global, os blocos de avaliação e
a calibração empírica são decisões de implementação documentadas em
METODOLOGIA.md, não resultados reproduzidos desses artigos.
