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
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from datetime import timedelta
from io import StringIO

import pandas as pd
import requests
from dotenv import load_dotenv
from eppy.modeleditor import IDF
from openai import OpenAI, RateLimitError
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def generate_geojson_and_note_for_all_entries():
    for home_folder in glob.glob("dataset/*"):
        try:
            print(f"[→] Generating for {home_folder}")
            result = generate_geojson_and_note(
                json.load(open(os.path.join(home_folder, "data.json"))),
                os.path.join(home_folder, "photo_1.jpg"),
                os.path.join(home_folder, "sketch.png")
            )
            json.dump(result, open(os.path.join(home_folder, "preprocessed.json"), "w", encoding='utf-8'), indent=2)
            print(f"[GENERATED] {home_folder}")
        except Exception as e:
            print(f"[FAILED] {home_folder} [WILL BE DELETED]: {e}")
            shutil.rmtree(home_folder, ignore_errors=True)


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


def scrape_sketch_details(driver):
    """
    Scrapes the table with class 'rgMasterTable' inside div.rgDataDiv and returns a dictionary
    mapping the third <td>'s text to the integer value of the fourth <td> in each row.

    Args:
        driver: Selenium WebDriver object, assumed to be on the target page.

    Returns:
        dict: { third_td_text: int(fourth_td_text) }
    """
    details = {}

    iframe = driver.find_element(By.TAG_NAME, "iframe")
    driver.switch_to.frame(iframe)

    # Wait for the table inside div.rgDataDiv to load
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.rgDataDiv table.rgMasterTable"))
    )

    table = driver.find_element(By.CSS_SELECTOR, "div#RadGrid1_GridData table")
    rows = table.find_elements(By.TAG_NAME, "tr")

    print(f"[INFO] {len(rows)} rows found.")

    for row in rows:
        tds = row.find_elements(By.TAG_NAME, "td")
        if len(tds) >= 4:
            key = tds[2].text.strip()
            value_text = tds[3].text.strip()
            try:
                value = int(value_text.replace(',', ''))  # Remove commas if numbers are formatted
                details[key] = value
            except ValueError:
                continue  # Skip rows where the fourth td is not an integer

    return details


def scrape_sketch_image(driver, address_folder):
    """
    Takes a screenshot of the sketch image (with overlays rendered) and saves it to address_folder.

    Args:
        driver: Selenium WebDriver object, assumed to be on the target page.
        address_folder: str, path to save the screenshot.
    """
    # Wait for image to load (max 10 seconds)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "BinImage"))
    )

    time.sleep(2)

    element = driver.find_element(By.ID, "BinImage")
    element.screenshot(f"{address_folder}/sketch.png")
    driver.switch_to.default_content()


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
            scrape_and_download_photos_from_photoDetails(driver, folder)

            click_tab(driver, "Sketch")
            sketch_data = scrape_sketch_details(driver)
            scrape_sketch_image(driver, folder)

            data["sketch_data"] = sketch_data

            out_json = os.path.join(folder, "data.json")

            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            print(f"[SAVED] {data.get('address', 'Unknown')}")

            if i < total - 1:
                # Switch to iframe containing Next button
                try:
                    time.sleep(2)
                    iframes = driver.find_elements(By.TAG_NAME, "iframe")
                    print(f"[INFO] Found {len(iframes)} iframes.")

                    next_button_found = False
                    for idx, iframe in enumerate(iframes):
                        driver.switch_to.frame(iframe)
                        try:
                            next_button = driver.find_element(By.CSS_SELECTOR, "#DTLNavigator_imageNext")
                            print(f"[INFO] Found Next button in iframe {idx}.")
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
                            time.sleep(0.2)
                            next_button.click()
                            print("[INFO] Clicked Next button successfully.")
                            next_button_found = True
                            driver.switch_to.default_content()
                            break
                        except:
                            driver.switch_to.default_content()
                            continue

                    if not next_button_found:
                        print("[ERROR] Next button not found in any iframe.")

                except Exception as e:
                    print(f"[ERROR] Switching to iframe or clicking Next button failed: {e}")
                time.sleep(2)
        except Exception as e:
            print(f"[WARN] Skipped record {i} due to error: {e}")
            break


def delete_folders_without_jpg_or_png(dataset_dir="dataset"):
    deleted = 0
    for folder in os.listdir(dataset_dir):
        home_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(home_path):
            continue

        has_jpg = any(f.lower().endswith(".jpg") for f in os.listdir(home_path))
        has_png = any(f.lower().endswith(".png") for f in os.listdir(home_path))
        if not has_jpg or not has_png:
            shutil.rmtree(home_path)
            print(f"[CLEANED] Removed {folder} due to missing home images or floor plan sketches.")
            deleted += 1

    print(f"Cleanup complete. {deleted} folders removed.")


def encode_image(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_geojson_and_note(house_data, image_path, sketch_path, client):
    image_base64 = encode_image(image_path)
    sketch_base64 = encode_image(sketch_path)

    # ----- Prompt Setup -----
    prompt = f"""
    You are a certified **home energy inspection expert** and data specialist working on a project to generate synthetic inspection reports for single-family homes. You are helping build a training set for a home efficiency AI model.

    You are given:
    - Structured residential property data in JSON format, including descriptions of the home's floorplan sketch
    - A sketch of the floorplan of the home 
    - A photo of the exterior of the home

    Use these to generate two outputs:

    1. A **GeoJSON File** with:
       - A fictional but plausible (longitude, latitude) location in Bethlehem, Pennsylvania.
       - A `"type": "FeatureCollection"` containing exactly **one Feature** (representing the entire building footprint, including any additions or attached garages that are part of the conditioned or structural footprint).
       - Each Feature having:
         - `"type": "Feature"`
         - `"geometry"` as a Polygon or MultiPolygon representing the building footprint (including additions and garages if relevant) with realistic coordinates in Bethlehem, PA.
         - `"properties"` populated using the provided JSON fields:
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
         - Also include **estimated building performance parameters** within properties:
           - `"hvac_heating_cop"`: Estimated HVAC heating system COP (Coefficient of Performance); Range [0, 1]
           - `"hvac_cooling_cop"`: Estimated HVAC cooling system COP (Coefficient of Performance)
           - `"wall_r_value"`: Estimated wall insulation R-value (ft²·°F·hr/BTU)
           - `"roof_r_value"`: Estimated roof insulation R-value (ft²·°F·hr/BTU)
           - `"air_change_rate"`: Estimated air change rate; range [0, 1]
       - The footprint polygon(s) should have plausible residential dimensions consistent with “Total Square Feet Living Area”. For example, a 2000 sqft home should not have a 10x10 ft footprint.
       - Exclude any sketch zones that are unnecessary for energy analysis (e.g. patio or porch).

    2. A short **inspection note** written as if you had just walked around the home. Focus on energy-related characteristics: insulation, HVAC age/type, visible window quality, age, materials, and any notable upgrades or issues you can infer from the attributes or image.

    Here is the structured property data:

    {json.dumps(house_data)}

    Ensure:
    - The GeoJSON is a valid FeatureCollection with exactly one Feature as described.
    - The geometry coordinates are realistic and located within Bethlehem, PA.
    - The footprint dimensions are consistent with the "Total Square Feet Living Area".
    - Additions or attached garages should be included in the geometry if they are part of the structural building footprint.
    - The performance parameters are realistic and consistent with the home characteristics and inspection note.

    Return a single raw JSON object, like this:

    {{
      "geojson": {{
        "type": "FeatureCollection",
        "features": [
          {{
            "type": "Feature",
            "geometry": {{ ... }},
            "properties": {{
              ...,
              "air_change_rate": ...,
              "hvac_heating_cop": ...,
              "hvac_cooling_cop": ...,
              "wall_r_value": ...,
              "roof_r_value": ...,
            }}
          }}
        ]
      }},
      "inspection_note": "..."
    }}

    Output **only valid JSON**, no backticks or explanation.
    """
    # ----- API Call -----

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sketch_base64}"}},
                ]
            }
        ],
        temperature=0.7
    )

    # ----- Parse -----

    reply = response.choices[0].message.content
    return json.loads(reply)


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

    # Simulation metadata
    idf.newidfobject("TIMESTEP", Number_of_Timesteps_per_Hour=4)
    idf.newidfobject("SITE:LOCATION",
                     Name="Site Location",
                     Latitude=40.0,
                     Longitude=-75.0,
                     Time_Zone=-5.0,
                     Elevation=200.0)
    idf.newidfobject("SIMULATIONCONTROL",
                     Do_Zone_Sizing_Calculation="Yes",
                     Do_System_Sizing_Calculation="Yes",
                     Do_Plant_Sizing_Calculation="No",
                     Run_Simulation_for_Weather_File_Run_Periods="Yes",
                     Run_Simulation_for_Sizing_Periods="No")
    # Fixed simulation over a year
    idf.newidfobject("RUNPERIOD",
                     Name="July",
                     Begin_Month=1,
                     Begin_Day_of_Month=1,
                     End_Month=12,
                     End_Day_of_Month=31,
                     Use_Weather_File_Holidays_and_Special_Days="Yes",
                     Use_Weather_File_Daylight_Saving_Period="Yes",
                     Apply_Weekend_Holiday_Rule="Yes",
                     Use_Weather_File_Rain_Indicators="Yes",
                     Use_Weather_File_Snow_Indicators="Yes")

    # Winter design day
    idf.newidfobject("SIZINGPERIOD:DESIGNDAY",
                     Name="WinterDesignDay",
                     Month=1,
                     Day_of_Month=21,
                     Day_Type="WinterDesignDay",
                     Maximum_DryBulb_Temperature=-6.7,
                     Daily_DryBulb_Temperature_Range=0,
                     Humidity_Condition_Type="Wetbulb",
                     Wetbulb_or_DewPoint_at_Maximum_DryBulb=-8.8,
                     Barometric_Pressure=101325,
                     Wind_Speed=4.9,
                     Wind_Direction=270,
                     Rain_Indicator="No",
                     Snow_Indicator="Yes",
                     Daylight_Saving_Time_Indicator="No",
                     Solar_Model_Indicator="ASHRAEClearSky")

    # Summer design day
    idf.newidfobject("SIZINGPERIOD:DESIGNDAY",
                     Name="SummerDesignDay",
                     Month=7,
                     Day_of_Month=21,
                     Day_Type="SummerDesignDay",
                     Maximum_DryBulb_Temperature=33.0,
                     Daily_DryBulb_Temperature_Range=11.0,
                     Humidity_Condition_Type="Wetbulb",
                     Wetbulb_or_DewPoint_at_Maximum_DryBulb=23.0,
                     Barometric_Pressure=101325,
                     Wind_Speed=4.9,
                     Wind_Direction=230,
                     Rain_Indicator="No",
                     Snow_Indicator="No",
                     Daylight_Saving_Time_Indicator="Yes",
                     Solar_Model_Indicator="ASHRAEClearSky")

    idf.newidfobject("BUILDING",
                     Name="GeneratedBuilding",
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

    # Schedule type limits
    idf.newidfobject("SCHEDULETYPELIMITS",
                     Name="Fraction",
                     Lower_Limit_Value=0.0,
                     Upper_Limit_Value=1.0,
                     Numeric_Type="Continuous",
                     Unit_Type="Dimensionless")
    idf.newidfobject("SCHEDULETYPELIMITS",
                     Name="Temperature",
                     Unit_Type="Temperature")

    # AlwaysOn and temperature schedules
    idf.newidfobject("SCHEDULE:COMPACT",
                     Name="AlwaysOn",
                     Schedule_Type_Limits_Name="Fraction",
                     Field_1="Through: 12/31",
                     Field_2="For: AllDays",
                     Field_3="Until: 24:00, 1")

    idf.newidfobject("SCHEDULE:COMPACT",
                     Name="HeatingSetpoint",
                     Schedule_Type_Limits_Name="Temperature",
                     Field_1="Through: 12/31",
                     Field_2="For: AllDays",
                     Field_3="Until: 24:00, 20")

    idf.newidfobject("SCHEDULE:COMPACT",
                     Name="CoolingSetpoint",
                     Schedule_Type_Limits_Name="Temperature",
                     Field_1="Through: 12/31",
                     Field_2="For: AllDays",
                     Field_3="Until: 24:00, 24")

    # Extract parameters from GeoJSON
    feature = geojson["features"][0]
    props = feature["properties"]

    wall_r_value = props.get("wall_r_value", 13.0)
    roof_r_value = props.get("roof_r_value", 30.0)
    window_u_value = props.get("window_u_value", 2.0)
    heating_cop = props.get("hvac_heating_cop", 0.8)
    cooling_cop = props.get("hvac_cooling_cop", 3.0)

    # Convert R-values to SI and compute conductivity
    def r_to_si(r_ip):
        return r_ip * 0.1761

    wall_conductivity = 0.2 / r_to_si(wall_r_value)
    roof_conductivity = 0.15 / r_to_si(roof_r_value)

    # Define materials
    idf.newidfobject("MATERIAL", Name="Wall Material", Roughness="Rough",
                     Thickness=0.2, Conductivity=wall_conductivity, Density=1400,
                     Specific_Heat=1000, Thermal_Absorptance=0.9,
                     Solar_Absorptance=0.7, Visible_Absorptance=0.7)
    idf.newidfobject("MATERIAL", Name="Roof Material", Roughness="Rough",
                     Thickness=0.15, Conductivity=roof_conductivity, Density=800,
                     Specific_Heat=1200, Thermal_Absorptance=0.9,
                     Solar_Absorptance=0.7, Visible_Absorptance=0.7)
    idf.newidfobject("WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM",
                     Name="Window Material",
                     UFactor=window_u_value,
                     Solar_Heat_Gain_Coefficient=0.5,
                     Visible_Transmittance=0.5)

    # Constructions
    idf.newidfobject("CONSTRUCTION", Name="Wall Construction", Outside_Layer="Wall Material")
    idf.newidfobject("CONSTRUCTION", Name="Roof Construction", Outside_Layer="Roof Material")
    idf.newidfobject("CONSTRUCTION", Name="Window Construction", Outside_Layer="Window Material")

    # Process zone
    zone_name = props.get("name", "Zone1").replace(" ", "_")
    area_sqft = props.get("Total Square Feet Living Area", 1000)
    area_m2 = area_sqft * 0.092903
    length = width = math.sqrt(area_m2)
    height = props.get("height_ft", 10.0) * 0.3048

    idf.newidfobject("ZONE", Name=zone_name)

    # ➡️ Define Floor Construction if not already defined
    if "Floor Construction" not in [c.Name for c in idf.idfobjects["CONSTRUCTION"]]:
        # Define a basic concrete floor material and construction
        idf.newidfobject("MATERIAL", Name="Floor Material", Roughness="MediumSmooth",
                         Thickness=0.15, Conductivity=1.4, Density=2200,
                         Specific_Heat=1000, Thermal_Absorptance=0.9,
                         Solar_Absorptance=0.7, Visible_Absorptance=0.7)
        idf.newidfobject("CONSTRUCTION", Name="Floor Construction", Outside_Layer="Floor Material")

    # ➡️ Create surfaces for the zone
    surfaces = {
        "South Wall": [(0, 0, 0), (0, 0, height), (length, 0, height), (length, 0, 0)],
        "East Wall": [(length, 0, 0), (length, 0, height), (length, width, height), (length, width, 0)],
        "North Wall": [(length, width, 0), (length, width, height), (0, width, height), (0, width, 0)],
        "West Wall": [(0, width, 0), (0, width, height), (0, 0, height), (0, 0, 0)],
        "Floor": [(0, width, 0), (length, width, 0), (length, 0, 0), (0, 0, 0)],
        "Roof": [(0, 0, height), (length, 0, height), (length, width, height), (0, width, height)]
    }

    for surf_name, verts in surfaces.items():
        if "Wall" in surf_name:
            surface_type = "Wall"
            construction = "Wall Construction"
            outside = "Outdoors"
            sun = "SunExposed"
            wind = "WindExposed"
        elif surf_name == "Floor":
            surface_type = "Floor"
            construction = "Floor Construction"
            outside = "Ground"
            sun = "NoSun"
            wind = "NoWind"
        elif surf_name == "Roof":
            surface_type = "Roof"
            construction = "Roof Construction"
            outside = "Outdoors"
            sun = "SunExposed"
            wind = "WindExposed"

        vertex_dict = {}
        for i, v in enumerate(verts):
            vertex_dict[f"Vertex_{i + 1}_Xcoordinate"] = v[0]
            vertex_dict[f"Vertex_{i + 1}_Ycoordinate"] = v[1]
            vertex_dict[f"Vertex_{i + 1}_Zcoordinate"] = v[2]

        idf.newidfobject("BUILDINGSURFACE:DETAILED",
                         Name=f"{zone_name}_{surf_name}",
                         Surface_Type=surface_type,
                         Construction_Name=construction,
                         Zone_Name=zone_name,
                         Outside_Boundary_Condition=outside,
                         Sun_Exposure=sun,
                         Wind_Exposure=wind,
                         Number_of_Vertices=4,
                         **vertex_dict)

    # idf.newidfobject("ZONEHVAC:EQUIPMENTCONNECTIONS",
    #                  Zone_Name=zone_name,
    #                  Zone_Conditioning_Equipment_List_Name=f"{zone_name}_EquipmentList",
    #                  Zone_Air_Inlet_Node_or_NodeList_Name=f"{zone_name}_SupplyAirNode",
    #                  Zone_Air_Exhaust_Node_or_NodeList_Name="",  # optional, empty here
    #                  Zone_Air_Node_Name=f"{zone_name}_AirNode",
    #                  Zone_Return_Air_Node_or_NodeList_Name=f"{zone_name}_ReturnAirNode")
    #
    # idf.newidfobject("ZONEHVAC:EQUIPMENTLIST",
    #                  Name=f"{zone_name}_EquipmentList",
    #                  Load_Distribution_Scheme="SequentialLoad",
    #                  Zone_Equipment_1_Object_Type="AirTerminal:SingleDuct:Uncontrolled",
    #                  Zone_Equipment_1_Name=f"{zone_name}_DirectAir",
    #                  Zone_Equipment_1_Cooling_Sequence=1,
    #                  Zone_Equipment_1_Heating_or_NoLoad_Sequence=1)

    # Thermostat
    idf.newidfobject("HVACTEMPLATE:THERMOSTAT",
                     Name=f"{zone_name}_Thermostat",
                     Heating_Setpoint_Schedule_Name="HeatingSetpoint",
                     Cooling_Setpoint_Schedule_Name="CoolingSetpoint")

    # ✅ Add internal gains
    idf.newidfobject("PEOPLE",
                     Name=f"{zone_name}_People",
                     Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
                     Number_of_People_Schedule_Name="AlwaysOn",
                     Number_of_People_Calculation_Method="People",
                     Number_of_People=2,
                     Activity_Level_Schedule_Name="AlwaysOn")
    idf.newidfobject("LIGHTS",
                     Name=f"{zone_name}_Lights",
                     Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
                     Schedule_Name="AlwaysOn",
                     Design_Level_Calculation_Method="Watts/Area",
                     Watts_per_Floor_Area=10)
    idf.newidfobject("ELECTRICEQUIPMENT",
                     Name=f"{zone_name}_Equip",
                     Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
                     Schedule_Name="AlwaysOn",
                     Design_Level_Calculation_Method="Watts/Area",
                     Watts_per_Floor_Area=5)

    # Extract air change rate from props
    air_change_rate = props.get("air_change_rate", 0.35)

    idf.newidfobject("ZONEINFILTRATION:DESIGNFLOWRATE",
                     Name=f"{zone_name}_Infiltration",
                     Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
                     Schedule_Name="AlwaysOn",
                     Design_Flow_Rate_Calculation_Method="AirChanges/Hour",
                     Air_Changes_per_Hour=air_change_rate,
                     Constant_Term_Coefficient=1,
                     Temperature_Term_Coefficient=0,
                     Velocity_Term_Coefficient=0,
                     Velocity_Squared_Term_Coefficient=0)

    idf.newidfobject("HVACTEMPLATE:ZONE:PTAC",
                     Zone_Name=zone_name,
                     Template_Thermostat_Name=f"{zone_name}_Thermostat",
                     Cooling_Coil_Gross_Rated_Total_Capacity="autosize",
                     Cooling_Coil_Gross_Rated_Cooling_COP=cooling_cop,
                     Heating_Coil_Type="Gas",  # or "ElectricResistance" if appropriate
                     Heating_Coil_Capacity="autosize",
                     Gas_Heating_Coil_Efficiency=heating_cop)

    # Outputs (unchanged)
    # Replace OUTPUT:VARIABLE for meters with OUTPUT:METER
    for var in [
        "Zone Air Temperature",
        "Heating Coil Gas Energy",
        "Cooling Coil Electric Energy",
        "Fan Electric Energy",
        "HVAC Electric Energy",
        "Zone/Site Heating:Energy",
        "Heating Coil Heating Energy",
        "Zone Infiltration Sensible Heat Loss",
        "Zone Infiltration Sensible Heat Gain"
    ]:
        idf.newidfobject("OUTPUT:VARIABLE",
                         Variable_Name=var,
                         Reporting_Frequency="Hourly")

    for meter in ["Electricity:Facility", "Electricity:HVAC", "Gas:Facility"]:
        idf.newidfobject("OUTPUT:METER",
                         Key_Name=meter,
                         Reporting_Frequency="Hourly")

    idf.newidfobject("OUTPUT:VARIABLEDICTIONARY",
                     Key_Field="IDF")

    idf.save(idf_path)

    # Run expandobjects

    cwd = os.getcwd()
    os.chdir(os.path.dirname(idf_path))
    subprocess.run(['expandobjects'])
    os.chdir(cwd)


# def generate_idf_from_geojson(geojson: dict, idf_path: str):
#     idf = IDF(StringIO("Version,25.1;"))
#
#     # Simulation metadata (unchanged)
#     idf.newidfobject("TIMESTEP", Number_of_Timesteps_per_Hour=4)
#     idf.newidfobject("SITE:LOCATION",
#                      Name="Site Location",
#                      Latitude=40.0,
#                      Longitude=-75.0,
#                      Time_Zone=-5.0,
#                      Elevation=200.0)
#     idf.newidfobject("SIMULATIONCONTROL",
#                      Do_Zone_Sizing_Calculation="No",
#                      Do_System_Sizing_Calculation="No",
#                      Do_Plant_Sizing_Calculation="No",
#                      Run_Simulation_for_Weather_File_Run_Periods="Yes",
#                      Run_Simulation_for_Sizing_Periods="No")
#     today = datetime.today()
#     idf.newidfobject("RUNPERIOD",
#                      Name="RunPeriod1",
#                      Begin_Month=today.month,
#                      Begin_Day_of_Month=today.day,
#                      End_Month=today.month,
#                      End_Day_of_Month=today.day,
#                      Use_Weather_File_Holidays_and_Special_Days="Yes",
#                      Use_Weather_File_Daylight_Saving_Period="Yes",
#                      Apply_Weekend_Holiday_Rule="Yes",
#                      Use_Weather_File_Rain_Indicators="Yes",
#                      Use_Weather_File_Snow_Indicators="Yes")
#
#     idf.newidfobject("BUILDING",
#                      Name="GeneratedBuilding",
#                      North_Axis=0.0,
#                      Terrain="Suburbs",
#                      Loads_Convergence_Tolerance_Value=0.04,
#                      Temperature_Convergence_Tolerance_Value=0.4,
#                      Solar_Distribution="FullExterior",
#                      Maximum_Number_of_Warmup_Days=25,
#                      Minimum_Number_of_Warmup_Days=6)
#
#     idf.newidfobject("GLOBALGEOMETRYRULES",
#                      Starting_Vertex_Position="UpperLeftCorner",
#                      Vertex_Entry_Direction="CounterClockWise",
#                      Coordinate_System="Relative")
#
#     # Schedule type limits (unchanged)
#     idf.newidfobject("SCHEDULETYPELIMITS",
#                      Name="Fraction",
#                      Lower_Limit_Value=0.0,
#                      Upper_Limit_Value=1.0,
#                      Numeric_Type="Continuous",
#                      Unit_Type="Dimensionless")
#
#     idf.newidfobject("SCHEDULETYPELIMITS",
#                      Name="Temperature",
#                      Unit_Type="Temperature")
#
#     # AlwaysOn schedule
#     idf.newidfobject("SCHEDULE:COMPACT",
#                      Name="AlwaysOn",
#                      Schedule_Type_Limits_Name="Fraction",
#                      Field_1="Through: 12/31",
#                      Field_2="For: AllDays",
#                      Field_3="Until: 24:00, 1")
#
#     # Heating and cooling setpoint schedules
#     idf.newidfobject("SCHEDULE:COMPACT",
#                      Name="HeatingSetpoint",
#                      Schedule_Type_Limits_Name="Temperature",
#                      Field_1="Through: 12/31",
#                      Field_2="For: AllDays",
#                      Field_3="Until: 24:00, 20")  # 20°C
#
#     idf.newidfobject("SCHEDULE:COMPACT",
#                      Name="CoolingSetpoint",
#                      Schedule_Type_Limits_Name="Temperature",
#                      Field_1="Through: 12/31",
#                      Field_2="For: AllDays",
#                      Field_3="Until: 24:00, 24")  # 24°C
#
#     # Extract performance parameters from GeoJSON properties
#     feature = geojson["features"][0]
#     props = feature["properties"]
#
#     air_change_rate = props.get("air_change_rate", 0.35)
#     hvac_heating_cop = props.get("hvac_heating_cop", 3.0)
#     hvac_cooling_cop = props.get("hvac_cooling_cop", 3.0)
#     window_u_value = props.get("window_u_value", 2.0)
#     wall_r_value = props.get("wall_r_value", 13.0)
#     roof_r_value = props.get("roof_r_value", 30.0)
#     hvac_system_type = props.get("hvac_system_type", "Unknown")
#
#     # Convert R-values to SI (m²·K/W) for IDF materials
#     def r_to_si(r_ip):
#         return r_ip * 0.1761
#
#     wall_r_si = r_to_si(wall_r_value)
#     roof_r_si = r_to_si(roof_r_value)
#
#     # Calculate conductivity from R-value (thickness assumed or set constant)
#     wall_thickness = 0.2  # m
#     wall_conductivity = wall_thickness / wall_r_si if wall_r_si > 0 else 0.5
#
#     roof_thickness = 0.15  # m
#     roof_conductivity = roof_thickness / roof_r_si if roof_r_si > 0 else 0.3
#
#     # Update materials
#     idf.newidfobject("MATERIAL", Name="Wall Material", Roughness="Rough",
#                      Thickness=wall_thickness, Conductivity=wall_conductivity, Density=1400,
#                      Specific_Heat=1000, Thermal_Absorptance=0.9,
#                      Solar_Absorptance=0.7, Visible_Absorptance=0.7)
#
#     idf.newidfobject("MATERIAL", Name="Roof Material", Roughness="Rough",
#                      Thickness=roof_thickness, Conductivity=roof_conductivity, Density=800,
#                      Specific_Heat=1200, Thermal_Absorptance=0.9,
#                      Solar_Absorptance=0.7, Visible_Absorptance=0.7)
#
#     idf.newidfobject("MATERIAL", Name="Floor Material", Roughness="Rough",
#                      Thickness=0.1, Conductivity=1.0, Density=2000,
#                      Specific_Heat=1000, Thermal_Absorptance=0.9,
#                      Solar_Absorptance=0.7, Visible_Absorptance=0.7)
#
#     # Windows (simple glazing system)
#     idf.newidfobject("WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM",
#                      Name="Window Material",
#                      UFactor=window_u_value,
#                      Solar_Heat_Gain_Coefficient=0.5,
#                      Visible_Transmittance=0.5)
#
#     # Constructions
#     idf.newidfobject("CONSTRUCTION", Name="Wall Construction", Outside_Layer="Wall Material")
#     idf.newidfobject("CONSTRUCTION", Name="Roof Construction", Outside_Layer="Roof Material")
#     idf.newidfobject("CONSTRUCTION", Name="Floor Construction", Outside_Layer="Floor Material")
#     idf.newidfobject("CONSTRUCTION", Name="Window Construction", Outside_Layer="Window Material")
#
#     # Process zones
#     for feature in geojson["features"]:
#         props = feature["properties"]
#         zone_name = props.get("name", "UnnamedZone").replace(" ", "_")
#         area_sqft = props.get("Total Square Feet Living Area", 100.0)
#         area_m2 = area_sqft * 0.092903
#         length = width = math.sqrt(area_m2)
#         height = props.get("height_ft", 10.0) * 0.3048
#         conditioned = props.get("conditioned", True)
#
#         x0, y0 = feature["geometry"]["coordinates"][0][0]
#
#         idf.newidfobject("ZONE",
#                          Name=zone_name,
#                          X_Origin=x0,
#                          Y_Origin=y0,
#                          Z_Origin=0)
#
#         # Surfaces for each zone
#         surfaces = {
#             "South Wall": [(0, 0, 0), (0, 0, height), (length, 0, height), (length, 0, 0)],
#             "East Wall": [(length, 0, 0), (length, 0, height), (length, width, height), (length, width, 0)],
#             "North Wall": [(length, width, 0), (length, width, height), (0, width, height), (0, width, 0)],
#             "West Wall": [(0, width, 0), (0, width, height), (0, 0, height), (0, 0, 0)],
#             "Floor": [(0, width, 0), (length, width, 0), (length, 0, 0), (0, 0, 0)],
#             "Roof": [(0, 0, height), (length, 0, height), (length, width, height), (0, width, height)]
#         }
#
#         for surf_name, verts in surfaces.items():
#             surface_type = "Wall" if "Wall" in surf_name else "Roof" if surf_name == "Roof" else "Floor"
#             construction = f"{surface_type} Construction"
#             sun = "NoSun" if surface_type == "Floor" else "SunExposed"
#             wind = "NoWind" if surface_type == "Floor" else "WindExposed"
#             outside = "Ground" if surface_type == "Floor" else "Outdoors"
#
#             vertex_dict = {}
#             for i, v in enumerate(verts):
#                 vertex_dict[f"Vertex_{i + 1}_Xcoordinate"] = v[0] + x0
#                 vertex_dict[f"Vertex_{i + 1}_Ycoordinate"] = v[1] + y0
#                 vertex_dict[f"Vertex_{i + 1}_Zcoordinate"] = v[2]
#
#             idf.newidfobject("BUILDINGSURFACE:DETAILED",
#                              Name=f"{zone_name}_{surf_name}",
#                              Surface_Type=surface_type,
#                              Construction_Name=construction,
#                              Zone_Name=zone_name,
#                              Outside_Boundary_Condition=outside,
#                              Sun_Exposure=sun,
#                              Wind_Exposure=wind,
#                              Number_of_Vertices=4,
#                              **vertex_dict)
#
#         # Thermostat and HVAC (IdealLoads with COP adjustments possible in expanded HVAC definitions later)
#         if conditioned:
#             idf.newidfobject("HVACTEMPLATE:THERMOSTAT",
#                              Name=f"{zone_name}_Thermostat",
#                              Heating_Setpoint_Schedule_Name="HeatingSetpoint",
#                              Cooling_Setpoint_Schedule_Name="CoolingSetpoint")
#
#             idf.newidfobject("HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM",
#                              Zone_Name=zone_name,
#                              Template_Thermostat_Name=f"{zone_name}_Thermostat")
#
#     # Outputs (unchanged)
#     output_vars = [
#         "Zone Air Temperature",
#         "Zone Mean Air Temperature",
#         "Zone Infiltration Sensible Heat Gain",
#         "Zone Infiltration Air Change Rate",
#         "Zone Ideal Loads Supply Air Total Heating Energy",
#         "Zone Ideal Loads Supply Air Total Cooling Energy",
#         "Zone Ideal Loads Supply Air Total Heating Rate",
#         "Zone Ideal Loads Supply Air Total Cooling Rate",
#         "Electricity:Facility",
#         "Electricity:HVAC",
#         "Electricity:InteriorLighting",
#     ]
#
#     for var in output_vars:
#         idf.newidfobject("OUTPUT:VARIABLE",
#                          Variable_Name=var,
#                          Reporting_Frequency="Hourly")
#
#     idf.save(idf_path)


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
            "expanded.idf"
        ]

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


def label_data(results_json, inspection_report, home_dir_name, client, text_weight=.2, energyplus_weight=.8, energyplus_label_method='heuristic'):
    """
    Uses OpenAI API to label a datapoint based on its results.json and inspection report.
    """

    def build_text_prompt(inspection_report: str) -> str:
        return f"""
You are an expert building energy analyst.

Below is a **narrative inspection report** for a building.

Your task is to assign a **confidence score** in the range [0, 1] for the **need** for each of the following retrofits:
- Insulation upgrade
- HVAC upgrade

### IMPORTANT:
- A value of 0 means \"definitely not needed\".
- A value of 1 means \"definitely needed\".
- Intermediate values (e.g. 0.33, 0.5, 0.75) indicate uncertainty or partial need.

### INSPECTION REPORT (free text):
\"\"\"
{inspection_report}
\"\"\"

### RESPONSE FORMAT:
Return a JSON object like:
{{
  \"insulation\": 0.5,
  \"hvac\": 0.5,
}}

Only include the JSON. No explanation or commentary.
"""

    def build_energyplus_prompt(results_json: dict) -> str:
        return f"""
You are an expert building energy analyst.

Below are **EnergyPlus simulation results** for a building.

Your task is to assign a **confidence score** in the range [0, 1] for the **need** for each of the following retrofits:
- Insulation upgrade
- HVAC upgrade

### IMPORTANT:
- A value of 0 means \"definitely not needed\".
- A value of 1 means \"definitely needed\".
- Intermediate values (e.g. 0.33, 0.5, 0.75) indicate uncertainty or partial need.

### ENERGYPLUS SIMULATION RESULTS (json):

{json.dumps(results_json, indent=2)}

### RESPONSE FORMAT:
Return a JSON object like:
{{
  \"insulation\": 0.5,
  \"hvac\": 0.5,
}}

Only include the JSON. No explanation or commentary.
"""

    def safe_chat_response(prompt):
        while True:
            try:
                return client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
            except RateLimitError as e:
                print("Rate limit hit — sleeping for 2 seconds...")
                time.sleep(2)

    def extract_json_from_response(response_text):
        # Remove triple backticks and optional language hints
        cleaned = re.sub(r"^```(\w+)?", "", response_text.strip())
        cleaned = re.sub(r"```$", "", cleaned.strip())
        return cleaned

    def heuristic_labeler(results):

        # Constants
        HL_WORST = 6000  # kWh / yr
        HL_BEST = 25  # kWh / yr
        HVAC_WORST = 3000  # kWh / yr
        HVAC_BEST = 25  # kWh / yr

        heating_load_hourly_J = 0.0
        hvac_hourly_J = 0.0

        for var_name, var_data in results.items():
            if "Heating Coil Heating Energy" in var_name:
                heating_load_hourly_J = float(var_data["mean"])
            elif "Electricity:HVAC" in var_name:
                hvac_hourly_J = float(var_data["mean"])

        heating_load_annual_kWh = (heating_load_hourly_J * 730 * 12) / 3600000
        hvac_annual_kWh = (hvac_hourly_J * 730 * 12) / 3600000

        insulation_score = (heating_load_annual_kWh - HL_BEST) / (HL_WORST - HL_BEST)
        hvac_score = (hvac_annual_kWh - HVAC_BEST) / (HVAC_WORST - HVAC_BEST)

        insulation_score = max(min(insulation_score, 1), 0)
        hvac_score = max(min(hvac_score, 1), 0)

        return {
            "insulation": insulation_score,
            "hvac": hvac_score,
        }

    try:
        text_prompt = build_text_prompt(inspection_report)
        energyplus_prompt = build_energyplus_prompt(results_json)

        text_response = safe_chat_response(text_prompt)
        text_content = extract_json_from_response(text_response.choices[0].message.content)
        text_data = json.loads(text_content)

        if energyplus_label_method == 'gpt':
            energyplus_response = safe_chat_response(energyplus_prompt)
            energyplus_content = extract_json_from_response(energyplus_response.choices[0].message.content)
            energyplus_data = json.loads(energyplus_content)
        elif energyplus_label_method == 'heuristic':
            energyplus_data = heuristic_labeler(results_json)
        else:
            raise ValueError(f"Unsupported energyplus label method: {energyplus_label_method}")

        # elementwise sum & normalize to [0, 1]

        result = {
            "insulation": ((text_data["insulation"] * text_weight) + (energyplus_data["insulation"] * energyplus_weight)) / 2,
            "hvac": ((text_data["hvac"] * text_weight) + (energyplus_data["hvac"] * energyplus_weight)) / 2,
        }

        try:
            if home_dir_name is None:
                return result
            label_path = f"{home_dir_name}/label.json"
            with open(label_path, "w") as f:
                json.dump(result, f, indent=2)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse model response:\n{json.dumps(result)}")
    except json.JSONDecodeError as e:
        print("⚠️ Failed to parse JSON. Raw content:")
        raise e


def process_results(home_dir_name):
    results_json = extract_results_from_csv(home_dir_name)
    json.dump(results_json, open(f'{home_dir_name}/results.json', 'w'))

    inspection_note = json.load(open(f'{home_dir_name}/cleaned.geojson', 'r'))["features"][0]["properties"][
        "inspection_note"]

    label_data(results_json, inspection_note, home_dir_name, client)


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
        delete_folders_without_jpg_or_png()
        print(bold_green(f"({CHECK}) [1/6] NorCo Assessor Data Saved"))
    else:
        print(bold_green(f"(—) [1/6] Skipped Scraping of NorCo Assessor Data"))

    if sys.argv[1] == "--end-after-scrape":
        exit(0)

    print(bold_yellow(f"( ) [2/6] Generate GeoJSONs and Notes"))
    if len(sys.argv) < 2 or sys.argv[1] != "--skip-generate-geojson":
        generate_geojson_and_note_for_all_entries()
        print(bold_green(f"({CHECK}) [2/6] GeoJSONs and Notes Generated"))
    else:
        print(bold_green(f"(—) [2/6] Skipped Generating GeoJSONs and Notes"))

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
