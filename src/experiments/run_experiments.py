import argparse
import json
from datetime import datetime
from pathlib import Path

import experiments.combined_input_variation as combined_var
import experiments.energyplus_variation as eplus_var
import experiments.plot_results as plot_results
import experiments.text_variation as text_var

# Create unique results directory for each run
base_results_dir = Path(__file__).parent / "results"
base_results_dir.mkdir(exist_ok=True)

today_str = datetime.now().strftime("%m%d%y")
existing_runs = [p for p in base_results_dir.iterdir() if p.is_dir() and p.name.startswith(today_str)]
run_number = len(existing_runs) + 1
RESULTS_DIR = base_results_dir / f"{today_str}-{run_number}"
RESULTS_DIR.mkdir(exist_ok=True)

EXPERIMENTS = {
    "text": text_var.run,
    "energyplus": eplus_var.run,
    "combined": combined_var.run,
}


def save_results(results, name):
    with open(RESULTS_DIR / f"{name}_results.json", "w") as f:
        json.dump(results, f, indent=2)


def main(client):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["text", "energyplus", "combined", "all"], nargs="?")
    args = parser.parse_args()

    if args.experiment:
        if args.experiment == "all":
            for name, exp_func in EXPERIMENTS.items():
                save_results(exp_func(client), name)
        else:
            save_results(EXPERIMENTS[args.experiment](client), args.experiment)

    plot_results.main(RESULTS_DIR)
