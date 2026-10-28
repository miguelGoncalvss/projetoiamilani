from dataclasses import replace
import argparse
import logging
from pathlib import Path
from .config import Config
from .pipeline import run
from .synthetic import generate
from .adapter import generate_erp_raw, adapt_erp_to_contract
from .inventory import simulate
from .models import Forecaster


def main():
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Previsão e simulação de estoque do CD")
    subs = parser.add_subparsers(dest="command", required=True)

    demo = subs.add_parser("generate-demo", help="Gerar oito anos de dados fictícios simples")
    demo.add_argument("--out", required=True)
    demo.add_argument("--skus", type=int, default=10)
    demo.add_argument("--seed", type=int, default=42)

    erp_gen = subs.add_parser("generate-erp", help="Gerar base bruta realista de ERP (transações, devoluções, rupturas)")
    erp_gen.add_argument("--out", required=True)
    erp_gen.add_argument("--skus", type=int, default=15)
    erp_gen.add_argument("--seed", type=int, default=42)

    erp_adapt = subs.add_parser("adapt-erp", help="Adaptar base bruta de ERP para o contrato do pipeline")
    erp_adapt.add_argument("--raw-dir", required=True)
    erp_adapt.add_argument("--out", required=True)

    training = subs.add_parser("run", help="Seleção, calibração, teste e previsão futura de todos os modelos")
    training.add_argument("--history", required=True)
    training.add_argument("--config", required=True)
    training.add_argument("--out", required=True)
    training.add_argument("--synthetic", action="store_true")
    training.add_argument("--reconstruct-stockouts", action="store_true",
                          help="Reconstruir demanda censurada nos dias de stockout=1 antes de treinar")
    training.add_argument("--adapt-protocol", action="store_true",
                          help="Ajustar folds de seleção/calibração se o histórico for menor que 84 meses")
    training.add_argument("--horizon", type=int, default=None,
                          help="Horizonte futuro em períodos (ex: 12, 24, 36 meses no mensal ou 30, 60, 90 dias no diário)")
    training.add_argument("--years", type=int, default=None,
                          help="Horizonte futuro em anos (ex: 1=12 meses, 2=24 meses, 3=36 meses)")

    sim = subs.add_parser("simulate", help="Projetar saldo, restrições físicas e risco conjunto")
    sim.add_argument("--forecast", required=True)
    sim.add_argument("--stock", required=True)
    sim.add_argument("--receipts", default="auto", help="CSV de recebimentos fixos ou 'auto' na política sQ")
    sim.add_argument("--regions", required=True)
    sim.add_argument("--out", required=True)
    sim.add_argument("--scenario", required=True)
    sim.add_argument("--policy", choices=["fixed_receipts", "sQ"], default="fixed_receipts")
    sim.add_argument("--n-sims", type=int, default=300)
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--model", default=None, help="Nome do modelo a filtrar caso use forecast_all_models.csv")

    replay = subs.add_parser("predict-saved", help="Reproduzir previsões pontuais do artefato salvo")
    replay.add_argument("--model-dir", required=True)
    replay.add_argument("--out", required=True)

    args = parser.parse_args()
    try:
        if args.command == "generate-demo":
            generate(args.out, args.skus, args.seed)
            print(f"Dados fictícios gerados: {args.out}")
        elif args.command == "generate-erp":
            generate_erp_raw(args.out, args.skus, args.seed)
            print(f"Base bruta ERP gerada: {args.out}")
        elif args.command == "adapt-erp":
            _, report = adapt_erp_to_contract(args.raw_dir, args.out)
            print(f"Base ERP adaptada: {args.out} | SKUs: {report['sku_count']} | Rupturas: {report['stockout_days_flagged']}")
        elif args.command == "run":
            cfg = Config.read(args.config)
            if args.horizon is not None and args.years is not None:
                raise ValueError("Informe apenas --horizon ou --years, não ambos")
            if args.horizon is not None:
                cfg = replace(cfg, forecast_horizon=int(args.horizon))
            elif args.years is not None:
                if args.years < 1:
                    raise ValueError("--years deve ser inteiro positivo")
                mult = 12 if cfg.frequency == "monthly" else 365
                cfg = replace(cfg, forecast_horizon=int(args.years) * mult)
            run(args.history, cfg, args.out, args.synthetic,
                reconstruct_stockouts=args.reconstruct_stockouts,
                allow_protocol_adaptation=args.adapt_protocol)
        elif args.command == "simulate":
            simulate(args.forecast, args.stock, args.receipts, args.regions, args.out, args.scenario,
                     policy=args.policy, n_sims=args.n_sims, seed=args.seed, model_name=args.model)
            print(f"Simulação salva: {args.out}")
        else:
            output = Path(args.out)
            if output.exists():
                raise ValueError("Arquivo de saída já existe")
            model = Forecaster.load(args.model_dir)
            prediction = model.predict()
            prediction["frequency"] = model.cfg.frequency
            output.parent.mkdir(parents=True, exist_ok=True)
            prediction.to_csv(output, index=False)
    except (ValueError, ImportError, FileNotFoundError) as exc:
        parser.exit(2, f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
