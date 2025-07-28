"""
idf_generation.py

Generates EnergyPlus IDF files from cleaned GeoJSON building footprints.

Responsibilities:
- Reads cleaned.geojson files from dataset folders.
- Converts geometry and properties into EnergyPlus IDF format.
- Saves in.idf files to each dataset folder for simulation.

Pipeline Context:
Step 4. Consumes output from geometry_generation.py (cleaned.geojson).
Produces input for energyplus_runner.py (in.idf).

Outputs (per property):
    dataset/{address_folder}/in.idf
"""

import os
import glob
import json
import subprocess
import math
from io import StringIO
from typing import Dict, Any
from eppy.modeleditor import IDF

import config


def run_expandobjects(idf_dir: str) -> None:
    """
    Runs EnergyPlus expandobjects utility in the specified directory.

    Args:
        idf_dir (str): Directory containing the IDF file to expand.
    """
    cwd = os.getcwd()
    os.chdir(idf_dir)
    subprocess.run(['ExpandObjects'])
    os.chdir(cwd)


def transform_dataset(dataset_folder: str = 'dataset', weather_station: str = 'KABE') -> None:
    """
    Adds weather station data to all preprocessed.json files in the dataset.

    Args:
        dataset_folder (str): Path to the dataset directory.
        weather_station (str): Weather station code (e.g., 'KABE').
    """
    preprocessed_data_set = glob.glob(f'{dataset_folder}/*/preprocessed.json')

    for preprocessed_data in preprocessed_data_set:
        with open(preprocessed_data, 'r') as f:
            data = json.load(f)
        data["weather"] = weather_station
        with open(preprocessed_data, 'w') as f:
            json.dump(data, f)


def generate_idf_from_geojson(geojson: Dict[str, Any], idf_path: str) -> None:
    """
    Generates an EnergyPlus IDF file from cleaned GeoJSON data.

    Args:
        geojson (Dict[str, Any]): Cleaned GeoJSON building data.
        idf_path (str): Path to save the generated IDF file.
    """
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

    wall_r_value = props.get("wall_r_value", config.DEFAULT_WALL_R_VALUE)
    roof_r_value = props.get("roof_r_value", config.DEFAULT_ROOF_R_VALUE)
    window_u_value = props.get("window_u_value", config.DEFAULT_WINDOW_U_VALUE)
    heating_cop = props.get("hvac_heating_cop", config.DEFAULT_HEATING_COP)
    cooling_cop = props.get("hvac_cooling_cop", config.DEFAULT_COOLING_COP)

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
    run_expandobjects(os.path.dirname(idf_path))
