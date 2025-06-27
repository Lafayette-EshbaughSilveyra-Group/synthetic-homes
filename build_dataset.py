"""
This file builds the dataset from Northampton County assessor data. Specifically, it:
1. Scrapes the streets in STREETS, fetching images and tabular data,
2. Passes that information into the OpenAI API to generate GeoJSON and an inspection report,
3. Converts the GeoJSON to an IDF format for use in EnergyPlus,
4. Runs EnergyPlus on the IDF,
5. Classifies the EnergyPlus results and inspection report, giving the ground truth using OpenAI's API

OUTPUTS:
`dataset/{entry_folder}/results.json`           ⌉
                                                | INPUTS (i.e., X)
`dataset/{entry_folder}/inspection_report.txt`  ⌋

`dataset/{entry_folder}/label.json`             ] GROUND TRUTH (i.e., Y)
"""

import base64
import os
import sys
import time
import glob
import shutil
import uuid
import json
from datetime import datetime, timedelta
import pandas as pd

import requests
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import glob
import json
import math
import os
from eppy.modeleditor import IDF
import urllib.request
from pathlib import Path
import subprocess
from datetime import datetime
import requests
import zipfile
from io import BytesIO, StringIO


def init_driver(headless=False):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    return webdriver.Chrome(options=options)


def click_tab(driver, tab_text):
    tabs = driver.find_elements(By.XPATH, f"//a[span[contains(text(), '{tab_text}')]]")
    for tab in tabs:
        if tab.is_displayed():
            tab.click()
            time.sleep(2)
            return
    print(f"[WARN] Tab with text '{tab_text}' not found.")


def scrape_residential_tab(driver):
    try:
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, "Residential")))
    except:
        print("[WARN] Residential table not found.")
        return {}

    data = {}
    rows = driver.find_elements(By.CSS_SELECTOR, "#Residential tr")
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        if len(cells) >= 2:
            key = cells[0].text.strip().replace(":", "")
            value = cells[1].text.strip()
            if key:
                data[key] = value
    return data


def wait_for_photoDetails(driver, timeout=10):
    for _ in range(timeout * 10):
        try:
            result = driver.execute_script("return typeof photoDetails !== 'undefined' && photoDetails.length > 0")
            if result:
                return True
        except:
            pass
        time.sleep(0.1)
    return False


def scrape_and_download_photos_from_photoDetails(driver, address_folder):
    if not wait_for_photoDetails(driver):
        print("[WARN] photoDetails not found or empty.")
        return []

    try:
        photo_details_json = driver.execute_script("return JSON.stringify(photoDetails);")
        photo_details = json.loads(photo_details_json)
    except Exception as e:
        print(f"[ERROR] Couldn't extract photoDetails: {e}")
        return []

    photo_urls = []
    for idx, photo in enumerate(photo_details):
        std_url = f"https://www.ncpub.org/_web/api/document/{photo['Id']}/standard?token=RnNBOFBQNFhzakRDS3dzVVFPYm1wVHpMMFhZR2FvVGZSWEFmRkc5SDE0az0="
        photo_urls.append(std_url)
        try:
            img_data = requests.get(std_url).content
            with open(os.path.join(address_folder, f"photo_{idx + 1}.jpg"), 'wb') as f:
                f.write(img_data)
        except Exception as e:
            print(f"[WARN] Failed to download image {idx + 1}: {e}")
    return photo_urls


def get_total_record_count(driver):
    try:
        txt = driver.find_element(By.ID, "DTLNavigator_txtFromTo").get_attribute("value")
        return int(txt.split(" of ")[-1])
    except:
        return 1


def scrape_all_records_on_street(driver, street_name, output_dir):
    base_url = "https://www.ncpub.org/_web/search/commonsearch.aspx?mode=address"
    driver.get(base_url)
    time.sleep(2)

    try:
        driver.find_element(By.ID, "btAgree").click()
        time.sleep(2)
    except:
        pass

    driver.find_element(By.ID, "inpStreet").send_keys(street_name)
    driver.find_element(By.ID, "btSearch").click()
    time.sleep(3)

    # Click the first result
    try:
        result_links = driver.find_elements(By.CSS_SELECTOR, "#searchResults tr.SearchResults")
        if not result_links:
            print(f"[INFO] No results found for street: {street_name}")
            return
        result_links[0].click()
        time.sleep(2)
    except Exception as e:
        print(f"[ERROR] Could not click initial search result: {e}")
        return

    total = get_total_record_count(driver)
    print(f"[INFO] Found {total} records for {street_name}")

    for i in range(total):
        try:
            click_tab(driver, "Residential")
            data = scrape_residential_tab(driver)

            click_tab(driver, "Photos")
            folder = os.path.join(output_dir, data.get("address", f"{street_name}_{i}").replace(" ", "_"))
            os.makedirs(folder, exist_ok=True)
            photo_urls = scrape_and_download_photos_from_photoDetails(driver, folder)

            data["photo_urls"] = photo_urls

            out_json = os.path.join(folder, "data.json")
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            print(f"[SAVED] {data.get('address', 'Unknown')}")

            if i < total - 1:
                driver.find_element(By.ID, "DTLNavigator_imageNext").click()
                time.sleep(2)
        except Exception as e:
            print(f"[WARN] Skipped record {i + 1} due to error: {e}")
            break


def delete_folders_without_jpg(dataset_dir="dataset"):
    deleted = 0
    for folder in os.listdir(dataset_dir):
        home_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(home_path):
            continue

        has_jpg = any(f.lower().endswith(".jpg") for f in os.listdir(home_path))
        if not has_jpg:
            shutil.rmtree(home_path)
            print(f"[CLEANED] Removed: {folder}")
            deleted += 1

    print(f"Cleanup complete. {deleted} folders removed.")


def encode_image(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_geojson_and_note(house_data, image_path, model="gpt-4o"):
    image_base64 = encode_image(image_path)

    # ----- Prompt Setup -----
    prompt = f"""
You are a certified **home energy inspection expert** and data specialist working on a project to generate synthetic inspection reports for single-family homes. You are helping build a training set for a home efficiency AI model.

You are given:
- Structured residential property data in JSON format
- A photo of the exterior of the home

Use these to generate two outputs:
1. A **GeoJSON Feature** with a fictional but plausible (longitude, latitude) location in Pennsylvania. Populate `"properties"` using the provided JSON fields:
   - "Year Built"
   - "Total Square Feet Living Area"
   - "Building Style"
   - "Exterior Wall Material"
   - "Heating Fuel Type"
   - "Heating System Type"
   - "Heat/Air Cond"
   - "Bedrooms"
   - "Full Baths"
   - "Half Baths"
   - "Basement"
   - "Number of Stories"
   - "Grade"

2. A short **inspection note** written as if you had just walked around the home. Focus on energy-related characteristics: insulation, HVAC age/type, visible window quality, age, materials, and any notable upgrades or issues you can infer from the attributes or image.

Here is the structured property data:

{json.dumps(house_data)}

Return a single raw JSON object, like this:

{{
  "geojson": {{ ... }},
  "inspection_note": "..."
}}

Output **only valid JSON**, no backticks or explanation.
"""
    # ----- API Call -----

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        temperature=0.7
    )

    # ----- Parse -----

    reply = response.choices[0].message.content
    return json.loads(reply)


def generate_geojson_and_note_for_all_entries():
    for home_folder in glob.glob("dataset/*"):
        try:
            print(f"[→] Generating for {home_folder}")
            result = generate_geojson_and_note(
                json.load(open(os.path.join(home_folder, "data.json"))),
                os.path.join(home_folder, "photo_1.jpg")
            )
            json.dump(result, open(os.path.join(home_folder, "preprocessed.json"), "w", encoding='utf-8'), indent=2)
            print(f"[GENERATED] {home_folder}")
        except Exception as e:
            print(f"[FAILED] {home_folder} [WILL BE DELETED]: {e}")
            shutil.rmtree(home_folder, ignore_errors=True)

def clean_gpt_geojson(gpt_output):
    def safe_int(val):
        try:
            return int(str(val).replace(",", "").strip())
        except:
            return None

    # Get today's date and 1 year prior
    end_date = datetime.now()
    begin_date = end_date - timedelta(days=365)

    # Format in ISO8601 with Zulu time (UTC)
    begin_date_str = begin_date.strftime('%Y-%m-%dT00:00:00Z')
    end_date_str = end_date.strftime('%Y-%m-%dT00:00:00Z')

    # Create final geojson object
    full_geojson = {
        "type": "FeatureCollection",
        "mappers": [],
        "project": {
            "id": str(uuid.uuid4()),
            "name": "Generated Project",
            "begin_date": begin_date_str,
            "end_date": end_date_str,
            "cec_climate_zone": None,
            "climate_zone": "4A",
            "default_template": "90.1-2013",
            "import_surrounding_buildings_as_shading": None,
            "surface_elevation": None,
            "tariff_filename": None,
            "timesteps_per_hour": 1,
            "weather_filename": "weather.epw"
        },
        "scenarios": [
            {
                "feature_mappings": [],
                "id": str(uuid.uuid4()),
                "name": "Base Scenario"
            }
        ],
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": str(uuid.uuid4()),
                    "name": "Generated Home",
                    "type": "Building",
                    "building_type": "Single family",
                    "floor_area": safe_int(gpt_output["geojson"]["properties"].get("Total Square Feet Living Area")),
                    "number_of_stories": safe_int(gpt_output["geojson"]["properties"].get("Number of Stories")),
                    "inspection_note": gpt_output["inspection_note"],
                    **gpt_output["geojson"]["properties"]
                },
                "geometry": gpt_output["geojson"]["geometry"]
            }
        ]
    }

    return full_geojson


def clean_gpt_geojson_for_all_entries():
    home_folders = glob.glob('dataset/*')
    for home_folder in home_folders:
        result = clean_gpt_geojson(json.load(open(os.path.join(home_folder, "preprocessed.json"))))
        json.dump(result, open(os.path.join(home_folder, "cleaned.geojson"), "w", encoding='utf-8'), indent=2)


def transform_dataset(dataset_folder='dataset', weather_station='KABE'):
    preprocessed_data_set = glob.glob(f'{dataset_folder}/*/preprocessed.json')

    for preprocessed_data in preprocessed_data_set:
        with open(preprocessed_data, 'r') as f:
            data = json.load(f)
        data["weather"] = weather_station
        with open(preprocessed_data, 'w') as f:
            json.dump(data, f)


def generate_idf_from_geojson(geojson: dict, idf_path: str):
    idf = IDF(StringIO("Version,25.1;"))

    # Metadata and simulation setup - VERSION already included in StringIO above
    idf.newidfobject("TIMESTEP", Number_of_Timesteps_per_Hour=4)

    # Add Site Location (required)
    idf.newidfobject("SITE:LOCATION",
                     Name="Site Location",
                     Latitude=40.0,
                     Longitude=-75.0,
                     Time_Zone=-5.0,
                     Elevation=200.0)

    idf.newidfobject("SIMULATIONCONTROL",
                     Do_Zone_Sizing_Calculation="No",
                     Do_System_Sizing_Calculation="No",
                     Do_Plant_Sizing_Calculation="No",
                     Run_Simulation_for_Weather_File_Run_Periods="Yes",
                     Run_Simulation_for_Sizing_Periods="No")

    today = datetime.today()
    idf.newidfobject("RUNPERIOD",
                     Name="RunPeriod1",
                     Begin_Month=today.month,
                     Begin_Day_of_Month=today.day,
                     End_Month=today.month,
                     End_Day_of_Month=today.day,
                     Use_Weather_File_Holidays_and_Special_Days="Yes",
                     Use_Weather_File_Daylight_Saving_Period="Yes",
                     Apply_Weekend_Holiday_Rule="Yes",
                     Use_Weather_File_Rain_Indicators="Yes",
                     Use_Weather_File_Snow_Indicators="Yes")

    # Extract geometry info
    building = next(f for f in geojson["features"] if f["properties"]["type"] == "Building")
    props = building["properties"]
    name = props.get("name", "GeneratedHome")
    floor_area = props.get("floor_area", 100.0)
    stories = props.get("number_of_stories", 1)

    length = width = floor_area ** 0.5
    height = stories * 3

    idf.newidfobject("BUILDING",
                     Name=name,
                     North_Axis=0.0,
                     Terrain="Suburbs",
                     Loads_Convergence_Tolerance_Value=0.04,
                     Temperature_Convergence_Tolerance_Value=0.4,
                     Solar_Distribution="FullExterior",
                     Maximum_Number_of_Warmup_Days=25,
                     Minimum_Number_of_Warmup_Days=6)

    idf.newidfobject("GLOBALGEOMETRYRULES",
                     Starting_Vertex_Position="UpperLeftCorner",
                     Vertex_Entry_Direction="CounterClockWise",
                     Coordinate_System="Relative")

    idf.newidfobject("ZONE", Name="MainZone")

    # Materials
    idf.newidfobject("MATERIAL", Name="Wall Material", Roughness="Rough",
                     Thickness=0.2, Conductivity=0.5, Density=1400,
                     Specific_Heat=1000, Thermal_Absorptance=0.9,
                     Solar_Absorptance=0.7, Visible_Absorptance=0.7)
    idf.newidfobject("MATERIAL", Name="Floor Material", Roughness="Rough",
                     Thickness=0.1, Conductivity=1.0, Density=2000,
                     Specific_Heat=1000, Thermal_Absorptance=0.9,
                     Solar_Absorptance=0.7, Visible_Absorptance=0.7)
    idf.newidfobject("MATERIAL", Name="Roof Material", Roughness="Rough",
                     Thickness=0.15, Conductivity=0.3, Density=800,
                     Specific_Heat=1200, Thermal_Absorptance=0.9,
                     Solar_Absorptance=0.7, Visible_Absorptance=0.7)

    idf.newidfobject("CONSTRUCTION", Name="Wall Construction", Outside_Layer="Wall Material")
    idf.newidfobject("CONSTRUCTION", Name="Floor Construction", Outside_Layer="Floor Material")
    idf.newidfobject("CONSTRUCTION", Name="Roof Construction", Outside_Layer="Roof Material")

    # Surfaces - Fixed vertex order for floor and roof
    surfaces = {
        "South Wall": [(0, 0, 0), (0, 0, height), (length, 0, height), (length, 0, 0)],
        "East Wall": [(length, 0, 0), (length, 0, height), (length, width, height), (length, width, 0)],
        "North Wall": [(length, width, 0), (length, width, height), (0, width, height), (0, width, 0)],
        "West Wall": [(0, width, 0), (0, width, height), (0, 0, height), (0, 0, 0)],
        "Floor": [(0, width, 0), (length, width, 0), (length, 0, 0), (0, 0, 0)],  # Fixed order
        "Roof": [(0, 0, height), (length, 0, height), (length, width, height), (0, width, height)]  # Fixed order
    }

    for name, verts in surfaces.items():
        surface_type = "Wall" if "Wall" in name else "Roof" if name == "Roof" else "Floor"
        construction = f"{surface_type} Construction"
        sun = "NoSun" if surface_type == "Floor" else "SunExposed"
        wind = "NoWind" if surface_type == "Floor" else "WindExposed"
        outside = "Ground" if surface_type == "Floor" else "Outdoors"

        idf.newidfobject("BUILDINGSURFACE:DETAILED",
                         Name=name,
                         Surface_Type=surface_type,
                         Construction_Name=construction,
                         Zone_Name="MainZone",
                         Outside_Boundary_Condition=outside,
                         Sun_Exposure=sun,
                         Wind_Exposure=wind,
                         Number_of_Vertices=4,
                         Vertex_1_Xcoordinate=verts[0][0], Vertex_1_Ycoordinate=verts[0][1],
                         Vertex_1_Zcoordinate=verts[0][2],
                         Vertex_2_Xcoordinate=verts[1][0], Vertex_2_Ycoordinate=verts[1][1],
                         Vertex_2_Zcoordinate=verts[1][2],
                         Vertex_3_Xcoordinate=verts[2][0], Vertex_3_Ycoordinate=verts[2][1],
                         Vertex_3_Zcoordinate=verts[2][2],
                         Vertex_4_Xcoordinate=verts[3][0], Vertex_4_Ycoordinate=verts[3][1],
                         Vertex_4_Zcoordinate=verts[3][2])

    # Infiltration - Using correct field names
    idf.newidfobject("ZONEINFILTRATION:DESIGNFLOWRATE",
                     Name="MainZone Infiltration",
                     Zone_or_ZoneList_or_Space_or_SpaceList_Name="MainZone",
                     Schedule_Name="AlwaysOn",
                     Design_Flow_Rate_Calculation_Method="Flow/Area",
                     Design_Flow_Rate=0.0,
                     Flow_Rate_per_Floor_Area=0.0002,
                     Flow_Rate_per_Exterior_Surface_Area=0.0,
                     Air_Changes_per_Hour=0.0,
                     Constant_Term_Coefficient=1.0,
                     Temperature_Term_Coefficient=0.0,
                     Velocity_Term_Coefficient=0.0,
                     Velocity_Squared_Term_Coefficient=0.0)

    # Schedules
    idf.newidfobject("SCHEDULETYPELIMITS", Name="Fraction", Lower_Limit_Value=0.0, Upper_Limit_Value=1.0,
                     Numeric_Type="Continuous", Unit_Type="Dimensionless")

    idf.newidfobject("SCHEDULE:COMPACT", Name="AlwaysOn", Schedule_Type_Limits_Name="Fraction",
                     Field_1="Through: 12/31", Field_2="For: AllDays", Field_3="Until: 24:00, 1")

    # Remove thermostat control since we don't have HVAC system
    # This prevents the fatal error about thermostatic control without equipment connections

    # Just add basic output variables for thermal analysis
    for var in [
        "Zone Air Temperature",
        "Zone Mean Air Temperature",
        "Zone Infiltration Sensible Heat Gain",
        "Zone Infiltration Air Change Rate"
    ]:
        idf.newidfobject("OUTPUT:VARIABLE",
                         Variable_Name=var,
                         Reporting_Frequency="Hourly")

    # Write to file
    idf.save(idf_path)


def run_energyplus_simulation(home_dir, weather_path):
    """
    Runs EnergyPlus simulation from within the home_dir,
    using in.idf and a weather file located outside (e.g., in 'weather/').
    Outputs go to home_dir/simulation_output/.
    """
    original_cwd = os.getcwd()
    abs_weather_path = os.path.abspath(weather_path)  # Ensure weather path works across directories

    os.makedirs(home_dir, exist_ok=True)
    os.chdir(home_dir)

    try:
        # Run EnergyPlus with absolute weather path
        cmd = [
            "energyplus",
            "-w", abs_weather_path,
            "-d", "simulation_output",
            "-r",
            "in.idf"
        ]

        print(cmd)

        subprocess.run(cmd, check=True)
    finally:
        os.chdir(original_cwd)


def simulate_home(home_folder_name: str):
    geojson_path = f"{home_folder_name}/cleaned.geojson"
    preprocessed_path = f"{home_folder_name}/preprocessed.json"
    try:
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
            generate_idf_from_geojson(geojson_data, f"{home_folder_name}/in.idf")

        with open(preprocessed_path, 'r') as f:
            preprocessed_data = json.load(f)
            weather_station = preprocessed_data["weather"]

        run_energyplus_simulation(
            f'{home_folder_name}',
            f'weather/{weather_station}.epw'
        )
    except Exception as e:
        print(f"[ERROR] Could not run EnergyPlus simulation for {home_folder_name}: {e}")
        print(f"[ERROR] This folder will be removed from the dataset.")
        shutil.rmtree(home_folder_name)
        return

    print(f"Completed simulation for {home_folder_name}.")


def simulate_all_homes():
    homes = glob.glob("dataset/*")
    for home in homes:
        simulate_home(home)


def extract_results_from_csv(home_dir: str) -> dict:
    """
    Extracts summary statistics and hourly zone temperatures from an EnergyPlus CSV output.

    Parameters:
        home_dir (str): Name of the home (i.e., RAMBEAU_RD_15)

    Returns:
        dict: Structured dictionary of zone-level features.
    """
    df = pd.read_csv(f"{home_dir}/simulation_output/eplusout.csv")

    # Normalize column names
    mean_air_col = "MAINZONE:Zone Mean Air Temperature [C](Hourly)"
    air_col = "MAINZONE:Zone Air Temperature [C](Hourly) "

    def compute_stats(series):
        return {
            "average": round(series.mean(), 3),
            "min": round(series.min(), 3),
            "max": round(series.max(), 3),
            "hourly": [round(x, 3) for x in series.tolist()]
        }

    return {
        "zone": "MAINZONE",
        "features": {
            "mean_air_temperature": compute_stats(df[mean_air_col]),
            "air_temperature": compute_stats(df[air_col])
        }
    }


def label_data(results_json, inspection_report, home_dir_name):
    """
    Uses OpenAI API to label a datapoint based on its results.json and inspection report.
    """

    def build_prompt(results_json: dict, inspection_report: str) -> str:
        return f"""
You are an expert building energy analyst.

Below is structured simulation data for a building, followed by a narrative inspection report.

Your task is to assign a **confidence score** in the range [0, 1] for the **need** for each of the following retrofits:
- Insulation upgrade
- Window upgrade
- HVAC upgrade
- Sealing

A value of 0 means "definitely not needed". A value of 1 means "definitely needed". Intermediate values (e.g. 0.33, 0.5, 0.75) indicate uncertainty or partial need. Use your judgment to assign realistic values based solely on the data and report.

### SIMULATION DATA (JSON):
{json.dumps(results_json, indent=2)}

### INSPECTION REPORT (free text):
\"\"\"
{inspection_report}
\"\"\"

### RESPONSE FORMAT:
Return a JSON object like:
{{
  "insulation": 0.5,
  "windows": 0.0,
  "hvac": 1.0,
  "sealing": 0.25
}}

Only include the JSON. No explanation or commentary.
"""

    prompt = build_prompt(results_json, inspection_report)

    def safe_chat_response():
        while True:
            try:
                return client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
            except RateLimitError as e:
                print("Rate limit hit — sleeping for 2 seconds...")
                time.sleep(2)
    response = safe_chat_response()

    content = response.choices[0].message.content

    try:
        label_path = f"{home_dir_name}/label.json"
        with open(label_path, "w") as f:
            json.dump(json.loads(content), f, indent=2)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse model response:\n{content}")


def process_results(home_dir_name):
    results_json = extract_results_from_csv(home_dir_name)
    json.dump(results_json, open(f'{home_dir_name}/results.json', 'w'))

    inspection_note = json.load(open(f'{home_dir_name}/cleaned.geojson', 'r'))["features"][0]["properties"][
        "inspection_note"]

    label_data(results_json, inspection_note, home_dir_name)


def process_all_results():
    homes = glob.glob('dataset/*')
    for home in homes:
        process_results(home)


if __name__ == "__main__":

    # Terminal formatting helpers
    RESET = "\033[0m"
    BOLD = "\033[1m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"


    def bold_yellow(text):
        return f"{BOLD}{YELLOW}{text}{RESET}"


    def bold_green(text):
        return f"{BOLD}{GREEN}{text}{RESET}"


    CHECK = "✔"

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env file")

    client = OpenAI(api_key=api_key)
    IDF.setiddname("/Applications/EnergyPlus-25-1-0/Energy+.idd")

    driver = init_driver(headless=False)
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)

    STREETS = ["IRONSTONE RD", "IRONSTONE CT", "STANBRIDGE CT", "HIGHBRIDGE CT", "TUDOR CT", "SUTTON PL", "REGAL RD",
               "GRAMERCY PL", "MARGATE RD", "RAMBEAU RD", "CANTERBURY RD", "GLOUCESTER DR", "NIJARO RD"]

    print(bold_yellow(f"( ) [1/6] Save NorCo Assessor Data"))

    if len(sys.argv) < 2 or (sys.argv[1] != "--skip-scrape" and sys.argv[1] != "--skip-generate-geojson"):
        for street in STREETS:
            scrape_all_records_on_street(driver, street, output_dir)
        driver.quit()
        delete_folders_without_jpg()
        print(bold_green(f"({CHECK}) [1/6] NorCo Assessor Data Saved"))
    else: print(bold_green(f"(—) [1/6] Skipped Scraping of NorCo Assessor Data"))

    print(bold_yellow(f"( ) [2/6] Generate GeoJSONs and Notes"))
    if len(sys.argv) < 2 or sys.argv[1] != "--skip-generate-geojson":
        generate_geojson_and_note_for_all_entries()
        print(bold_green(f"({CHECK}) [2/6] GeoJSONs and Notes Generated"))
    else: print(bold_green(f"(—) [2/6] Skipped Generating GeoJSONs and Notes"))

    print(bold_yellow(f"( ) [3/6] Clean GeoJSONs"))
    clean_gpt_geojson_for_all_entries()
    print(bold_green(f"({CHECK}) [3/6] GeoJSONs Cleaned"))

    print(bold_yellow(f"( ) [4/6] Run EnergyPlus"))
    transform_dataset(dataset_folder='dataset', weather_station='KABE')
    simulate_all_homes()
    print(bold_green(f"({CHECK}) [4/6] EnergyPlus Complete"))

    print(bold_yellow(f"( ) [5/6] Compile Results & Label Data"))
    process_all_results()
    print(bold_green(f"({CHECK}) [5/6] Compiled Results & Labeled Data"))

    print(bold_yellow(f"( ) [6/6] Merge Dataset to final_dataset.jsonl & final_dataset_summary.csv"))


    def merge_dataset():
        rows = []
        for home in glob.glob("dataset/*"):
            try:
                label = json.load(open(f"{home}/label.json"))
                results = json.load(open(f"{home}/results.json"))
                note = json.load(open(f"{home}/cleaned.geojson"))["features"][0]["properties"]["inspection_note"]

                mean_air = results["features"]["mean_air_temperature"]
                air = results["features"]["air_temperature"]

                row = {
                    "home_id": os.path.basename(home),
                    "inspection_note": note,
                    **label,

                    # Summary stats
                    "mean_air_temp_avg": mean_air["average"],
                    "mean_air_temp_min": mean_air["min"],
                    "mean_air_temp_max": mean_air["max"],
                    "air_temp_avg": air["average"],
                    "air_temp_min": air["min"],
                    "air_temp_max": air["max"],

                    # Full hourly time series
                    "mean_air_temp_hourly": mean_air["hourly"],
                    "air_temp_hourly": air["hourly"]
                }

                rows.append(row)

            except Exception as e:
                print(f"[SKIP] Could not process {home}: {e}")

        # Save as JSONL
        with open("final_dataset.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        # Optional: Save also as CSV with summary-only stats (no hourly)
        summary_rows = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in rows]
        pd.DataFrame(summary_rows).to_csv("final_dataset_summary.csv", index=False)


    merge_dataset()

    print(bold_green(f"({CHECK}) [6/6] Merged Dataset to final_dataset.jsonl & final_dataset_summary.csv"))
