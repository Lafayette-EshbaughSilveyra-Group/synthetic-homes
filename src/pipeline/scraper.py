"""
scraper.py

Scrapes Northampton County public assessor data for residential properties.

Responsibilities:
- Automates Selenium scraping of property data, photos, and floorplan sketches.
- Saves structured outputs for each property into dataset/{address_folder}/.

Typical Usage:
    from pipeline.scraper import init_driver, scrape_all_streets, delete_folders_without_jpg_or_png

Pipeline Context:
    This is Step 1 in the dataset generation pipeline. It produces raw data consumed by openai_inference.py.

Outputs (per property):
    dataset/{address_folder}/
        - data.json      # Property details
        - photo_1.jpg    # Exterior photo
        - sketch.png     # Floorplan sketch
"""

import os
import tempfile
import time
import json
import shutil
import requests
from typing import List

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.remote.webdriver import WebDriver


def init_driver(headless: bool = False) -> WebDriver:
    """
    Initializes a Chrome WebDriver instance.

    Args:
        headless (bool): Whether to run Chrome in headless mode.

    Returns:
        WebDriver: Configured Chrome WebDriver instance.
    """
    options = Options()
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

    # Force Chrome to use a temp profile to avoid profile lock conflicts
    user_data_dir = tempfile.mkdtemp()
    options.add_argument(f"--user-data-dir={user_data_dir}")

    return webdriver.Chrome(options=options)


def click_tab(driver: WebDriver, tab_text: str) -> None:
    """
    Clicks on a tab in the web UI with the given text.

    Args:
        driver (WebDriver): Selenium WebDriver object.
        tab_text (str): The text of the tab to click.
    """
    tabs = driver.find_elements(By.XPATH, f"//a[span[contains(text(), '{tab_text}')]]")
    for tab in tabs:
        if tab.is_displayed():
            tab.click()
            time.sleep(2)
            return
    print(f"[WARN] Tab with text '{tab_text}' not found.")


def scrape_residential_tab(driver: WebDriver) -> dict:
    """
    Scrapes the Residential tab of the web page.

    Args:
        driver (WebDriver): Selenium WebDriver object.

    Returns:
        dict: Scraped property data.
    """
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


def wait_for_photoDetails(driver: WebDriver, timeout: int = 10) -> bool:
    """
    Waits for the photoDetails JavaScript variable to be defined and non-empty.

    Args:
        driver (WebDriver): Selenium WebDriver object.
        timeout (int): Maximum time to wait in seconds.

    Returns:
        bool: True if photoDetails is available, False otherwise.
    """
    for _ in range(timeout * 10):
        try:
            result = driver.execute_script("return typeof photoDetails !== 'undefined' && photoDetails.length > 0")
            if result:
                return True
        except:
            pass
        time.sleep(0.1)
    return False


def scrape_and_download_photos_from_photoDetails(driver: WebDriver, address_folder: str) -> List[str]:
    """
    Scrapes and downloads property photos.

    Args:
        driver (WebDriver): Selenium WebDriver object.
        address_folder (str): Folder to save images.

    Returns:
        List[str]: List of downloaded photo URLs.
    """
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


def get_total_record_count(driver: WebDriver) -> int:
    """
    Retrieves the total number of records from the UI.

    Args:
        driver (WebDriver): Selenium WebDriver object.

    Returns:
        int: Total number of records found.
    """
    try:
        txt = driver.find_element(By.ID, "DTLNavigator_txtFromTo").get_attribute("value")
        return int(txt.split(" of ")[-1])
    except:
        return 1


def scrape_sketch_details(driver: WebDriver) -> dict:
    """
    Scrapes the table with class 'rgMasterTable' inside div.rgDataDiv and returns a dictionary
    mapping the third <td>'s text to the integer value of the fourth <td> in each row.

    Args:
        driver (WebDriver): Selenium WebDriver object, assumed to be on the target page.

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


def scrape_sketch_image(driver: WebDriver, address_folder: str) -> None:
    """
    Takes a screenshot of the sketch image (with overlays rendered) and saves it to address_folder.

    Args:
        driver (WebDriver): Selenium WebDriver object, assumed to be on the target page.
        address_folder (str): Path to save the screenshot.
    """
    # Wait for image to load (max 10 seconds)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "BinImage"))
    )

    time.sleep(2)

    element = driver.find_element(By.ID, "BinImage")
    element.screenshot(f"{address_folder}/sketch.png")
    driver.switch_to.default_content()


def scrape_all_records_on_street(driver: WebDriver, street_name: str, output_dir: str) -> None:
    """
    Scrapes all assessor records for a given street.

    Args:
        driver (WebDriver): A Selenium WebDriver instance.
        street_name (str): Name of the street to search.
        output_dir (str): Directory where data for each address will be saved.

    Side Effects:
        Creates folders and JSON files with property data and images.
    """
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
            continue


def delete_folders_without_jpg_or_png(dataset_dir: str = "dataset") -> None:
    """
    Deletes folders in the dataset directory that do not contain both JPG and PNG files.

    Args:
        dataset_dir (str): Path to the dataset directory.
    """
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
