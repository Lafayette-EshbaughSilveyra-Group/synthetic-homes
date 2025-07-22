import argparse
from pathlib import Path
from dotenv import load_dotenv
import os
from openai import OpenAI

from pipeline.scraper import scrape_all_records_on_street, delete_folders_without_jpg_or_png, init_driver
from pipeline.geometry_generation import run_generation_for_dataset, clean_gpt_geojson_for_all_entries
from pipeline.idf_generation import transform_dataset
from pipeline.energyplus_runner import simulate_all_homes
from pipeline.postprocessing import run_postprocessing_for_dataset
from pipeline.dataset_merging import merge_dataset

from experiments.run_experiments import RESULTS_DIR, main as run_experiments
import experiments.plot_results as plot_results
from experiments.occlusion import run_occlusions


def run_pipeline(client):
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "dataset"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    STREETS = ["IRONSTONE RD", "IRONSTONE CT", "STANBRIDGE CT", "HIGHBRIDGE CT", "TUDOR CT", "SUTTON PL", "REGAL RD",
               "GRAMERCY PL", "MARGATE RD", "RAMBEAU RD", "CANTERBURY RD", "GLOUCESTER DR", "NIJARO RD"]

    driver = init_driver(headless=True)
    for street in STREETS:
        scrape_all_records_on_street(driver, street, str(OUTPUT_DIR))
    driver.quit()
    delete_folders_without_jpg_or_png()

    run_generation_for_dataset(str(OUTPUT_DIR), client)
    clean_gpt_geojson_for_all_entries(str(OUTPUT_DIR))
    transform_dataset(dataset_folder=str(OUTPUT_DIR), weather_station='KABE')
    simulate_all_homes(str(OUTPUT_DIR))
    run_postprocessing_for_dataset(str(OUTPUT_DIR), client)
    merge_dataset(str(OUTPUT_DIR))


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env file")
    client = OpenAI(api_key=api_key)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pipeline", "experiments", "occlusion"], default="pipeline",
                        help="Run full pipeline, experiments, or occlusion tests.")
    args = parser.parse_args()

    if args.mode == "pipeline":
        run_pipeline(client)
    elif args.mode == "experiments":
        run_experiments(client)
        plot_results.main(RESULTS_DIR)
    elif args.mode == "occlusion":
        run_occlusions.run_occlusion_suite()
