from __future__ import annotations

import argparse
from Model.model import train_entry


def main():
    parser = argparse.ArgumentParser(description="Retrain churn model")
    parser.add_argument("--data", default="customer_churn.json", help="Path to events JSON (lines=true)")
    parser.add_argument("--window", type=int, default=30, help="Observation window in days")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()

    train_entry(args.data, args.window)


if __name__ == "__main__":
    main()


