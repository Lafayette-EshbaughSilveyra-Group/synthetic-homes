import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

from eppy.modeleditor import IDF
from openai import OpenAI

from src.pipeline.build_concept_scaler import build_concept_scaler
from src.pipeline.generate_full_factorial import generate
from src.pipeline.scraper import scrape_all_records_on_street, delete_folders_without_jpg_or_png, init_driver
from src.pipeline.geometry_generation import run_generation_for_dataset, clean_gpt_geojson_for_all_entries
from src.pipeline.idf_generation import transform_dataset
from src.pipeline.energyplus_runner import simulate_all_homes
from src.pipeline.postprocessing import run_postprocessing_for_dataset
from src.pipeline.dataset_merging import merge_dataset

from experiments.run_experiments import RESULTS_DIR, main as run_experiments
import experiments.plot_results as plot_results
from experiments.occlusion import run_occlusions

import config


def run_pipeline(client, scrape=True):
    print("[PIPELINE] Beginning pipeline...")
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / config.OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if scrape:

        driver = init_driver(headless=True)
        for street in config.STREETS:
            scrape_all_records_on_street(driver, street, str(OUTPUT_DIR))
        driver.quit()
        delete_folders_without_jpg_or_png()

    run_generation_for_dataset(str(OUTPUT_DIR), client)
    clean_gpt_geojson_for_all_entries(str(OUTPUT_DIR))

    IDF.setiddname(config.IDD_FILE_PATH)

    transform_dataset(dataset_folder=str(OUTPUT_DIR), weather_station=config.WEATHER_STATION)
    simulate_all_homes(str(OUTPUT_DIR))
    run_postprocessing_for_dataset(str(OUTPUT_DIR), client)
    merge_dataset(str(OUTPUT_DIR))


if __name__ == "__main__":
    load_dotenv()
    print("[SETUP] Initializing OpenAI")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env file")
    client = OpenAI(api_key=api_key)
    print("[SETUP] Initialized OpenAI")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pipeline", "pipeline-no-scrape", "experiments", "occlusion", "prepare-labeler"], default="pipeline",
                        help="Run full pipeline, experiments, or occlusion tests.")
    args = parser.parse_args()

    print(f"[MODE] Running Mode: {args.mode}")

    if args.mode == "pipeline":
        run_pipeline(client)
    elif args.mode == "pipeline-no-scrape":
        run_pipeline(client, scrape=False)
    elif args.mode == "experiments":
        run_experiments(client)
        plot_results.main(RESULTS_DIR)
    elif args.mode == "occlusion":
        run_occlusions.run_occlusion_suite()
    elif args.mode == "prepare-labeler":
        generate(epw="../weather/KMSP.epw")
        build_concept_scaler("hvac", "Electricity:HVAC [J](Hourly)", "mean")
        build_concept_scaler("insulation", "Heating Coil Heating Energy [J](Hourly)", "mean")
