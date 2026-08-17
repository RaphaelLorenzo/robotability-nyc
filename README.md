# The Robotability Score: Enabling Harmonious Robot Navigation on Urban Streets 
Matt Franchi*, Maria Teresa Parreira*, Frank Bu*, Wendy Ju

*_equal contribution_ 

CHI '25: ACM Conference on Human Factors in Computing Systems 

https://robotability.cornell.edu/

![10adb89d74f52711234ba592f25d8ece9bc7f83b](https://github.com/user-attachments/assets/ce5dab5b-4ffc-4047-b1ae-214c5d0d66b1)


## What is The Robotability Score?

The Robotability Score (R) is a novel metric that quantifies how suitable urban environments are for autonomous robot navigation. Through expert interviews and surveys, we've developed a standardized framework for evaluating urban landscapes to reduce uncertainty in robot deployment while respecting established mobility patterns.

Streets with high Robotability are both more navigable for robots and less disruptive to pedestrians. We've constructed a proof-of-concept Robotability Score for New York City using a wealth of open datasets from NYC OpenData, and inferred pedestrian distributions from a dataset of 8 million dashcam images taken around the city in late 2023.


**Read the paper ![here](https://doi.org/10.1145/3706598.3714009)**

##  Interactive Map
Explore the spatial distribution of Robotability Scores across New York City's diverse urban landscape. 
![Open the Interactive Map](https://robotability.cornell.edu/map)

## Running the code
### Installation

Create a conda environment with Python 3.11 (the version used in the notebooks), then install dependencies from `requirements.txt`:

```bash
conda create -n robotability python=3.11 -y
conda activate robotability
pip install -r requirements.txt
python -m ipykernel install --user --name robotability --display-name "Python (robotability)"
```

## Creating feature weights

Weights come from expert pairwise comparisons in the Qualtrics survey, converted to AHP (Analytic Hierarchy Process) scores. All of this lives in `survey_processing/`.

### 1. Split the survey by respondent type

`survey_52.csv` is the raw Qualtrics export. `survey_splits.ipynb` keeps only valid answers to **Q5** (current position), maps them to short labels, and writes one CSV per group:

| Q5 option | Output |
|---|---|
| Robotics in Academia | `survey_academia.csv` |
| Robotics in Industry | `survey_industry.csv` |
| Urban Planning | `survey_urban_planning.csv` |
| Accessibility | `survey_accessibility.csv` |
| Other | `survey_other.csv` |

It also writes `survey_other+.csv`: urban planning + accessibility + other, used as the “Other” group in the paper.

Open and run the notebook from `survey_processing/` (so relative paths resolve).

### 2. Compute AHP weights

From `survey_processing/`:

```bash
bash compute_all_weights_ahpmod.sh
```

That script calls `calculate_weights_ahp.py` six times:

| Output | Input | What it is |
|---|---|---|
| `all_weights.csv` | `survey_52.csv` | All 24 features, all respondents |
| `academia_weights.csv` | `survey_academia.csv` | Academia split |
| `industry_weights.csv` | `survey_industry.csv` | Industry split |
| `other_weights.csv` | `survey_other+.csv` | Other+ split |
| `feature_weights.csv` | `survey_52.csv`, excluding features `[8,10,16,17,19]` | NYC proof-of-concept (features we can measure in the city) |
| `trashbot_weights.csv` | `survey_52.csv`, excluding `[8,10,16,17,19,21,23,14,12]` | Trashbot scoring subset |

`--exclude_features` uses 0-based indices from the list in `calculate_weights_ahp.py` (e.g. 8 = local attitudes, 10 = weather). The script maps Qualtrics columns through `correspondence.json`, builds a pairwise contingency matrix (`contingency_matrix_ahp.csv`), and takes the principal eigenvector as the feature weights.

`feature_processing/score.ipynb` reads `feature_weights.csv`; `feature_processing/score_trashbot.ipynb` reads `trashbot_weights.csv`.

To rebuild a paper table from the weight CSVs, run `generate_weights_table.ipynb` (or `generate_weights_table.py`).

### `survey_processing/` layout

```
survey_processing/
├── survey_52.csv                 # raw Qualtrics export (all respondents)
├── survey_splits.ipynb           # split survey_52.csv by Q5
├── survey_academia.csv           # Q5 splits (and survey_other+.csv = other+urban+accessibility)
├── survey_industry.csv
├── survey_urban_planning.csv
├── survey_accessibility.csv
├── survey_other.csv
├── survey_other+.csv
├── correspondence.json           # Qualtrics column id → (feature A, feature B) pair
├── calculate_weights_ahp.py      # AHP: survey → contingency matrix → weights
├── compute_all_weights_ahpmod.sh # batch: all / academia / industry / other / NYC POC / trashbot
├── contingency_matrix_ahp.csv    # last AHP pairwise matrix written by calculate_weights_ahp.py
├── contingency_table.csv         # older pairwise table (not used by the AHP script)
├── all_weights.csv               # AHP outputs (Feature, Weight)
├── academia_weights.csv
├── industry_weights.csv
├── other_weights.csv
├── feature_weights.csv           # NYC POC (used by score.ipynb)
├── trashbot_weights.csv          # trashbot subset (used by score_trashbot.ipynb)
├── generate_weights_table.ipynb  # merge weight CSVs into a comparison / LaTeX table
└── generate_weights_table.py
```

## Getting data for the features

Run `pull_data.sh` and the feature notebooks from `feature_processing/`. Paths below are relative to that directory (`data/...`).

NYC OpenData CSV exports often get a date suffix (`_YYYYMMDD`). Shapefile zips unpack to a folder whose `geo_export_*.shp` name changes on every download. If your filenames differ from the ones in the notebooks, point the matching `read_csv` / `read_file` call at the file you actually saved.

### 1. Automatic downloads

```bash
cd feature_processing
bash pull_data.sh
mkdir -p data/processed data/street_furniture data/citibike
```

| Dataset | Saved as | Source |
|---|---|---|
| Sidewalks (planimetric CSV) | `data/sidewalks_nyc.csv` | [NYC OpenData 52n9-sdep](https://data.cityofnewyork.us/City-Government/NYC-Planimetric-Database-Sidewalk/52n9-sdep) |
| 2020 NTAs | `data/ntas_nyc.csv` | [NYC OpenData 9nt8-h7nd](https://data.cityofnewyork.us/City-Government/2020-Neighborhood-Tabulation-Areas-NTAs-/9nt8-h7nd) |
| Pedestrian curb ramps | `data/pedestrian_curb_ramp_nyc.csv` | [NYC OpenData ufzp-rrqu](https://data.cityofnewyork.us/Transportation/Pedestrian-Curb-Ramp-Locations/ufzp-rrqu) |
| Raised crosswalks | `data/raised_crosswalks_nyc.csv` | [NYC OpenData uh2s-ftgh](https://data.cityofnewyork.us/Transportation/Raised-Crosswalks/uh2s-ftgh) |
| VZW enhanced crossings | `data/vzw_enhanced_crossings_nyc.csv` | [NYC OpenData k9a2-vdr8](https://data.cityofnewyork.us/Transportation/VZV-Enhanced-Crossings/k9a2-vdr8) |
| 1-ft DEM raster | `data/1ft_dem_nyc/` (`DEM_LiDAR_1ft_2010_Improved_NYC_int.tif`) | [NYC DEM 1ft Int zip](https://sa-static-customer-assets-us-east-1-fedramp-prod.s3.amazonaws.com/data.cityofnewyork.us/NYC_DEM_1ft_Int.zip) |
| POIs | `data/pois_nyc.csv` | [NYC OpenData t95h-5fsr](https://data.cityofnewyork.us/City-Government/Points-Of-Interest/t95h-5fsr) |
| DSNY litter baskets | `data/street_furniture/dsny_litter_baskets_nyc.csv` | [NYC OpenData 8znf-7b2c](https://data.cityofnewyork.us/City-Government/DSNY-Litter-Basket-Inventory/8znf-7b2c) |
| Fire hydrants | `data/street_furniture/fire_hydrants_nyc.csv` | [NYC OpenData 5bgh-vtsn](https://data.cityofnewyork.us/Environment/NYC-Fire-Hydrant-Data/5bgh-vtsn) |
| Bus stop shelters | `data/street_furniture/bus_stop_shelters_nyc.csv` | [NYC OpenData t4f2-8md7](https://data.cityofnewyork.us/Transportation/Bus-Stop-Shelters/t4f2-8md7) |
| Bicycle parking shelters | `data/street_furniture/bicycle_parking_shelters_nyc.csv` | [NYC OpenData dimy-qyej](https://data.cityofnewyork.us/Transportation/Bicycle-Parking-Shelters/dimy-qyej) |
| CityBench | `data/street_furniture/citybench_nyc.csv` | [NYC OpenData kuxa-tauh](https://data.cityofnewyork.us/Transportation/CityBench/kuxa-tauh) |
| Forestry tree points | `data/street_furniture/forestry_tree_points_nyc.csv` | [NYC OpenData uvpi-gqnh](https://data.cityofnewyork.us/Environment/Forestry-Tree-Points/uvpi-gqnh) |
| Newsstands | `data/street_furniture/newsstands_nyc.csv` | [NYC OpenData w9zq-xm8b](https://data.cityofnewyork.us/Transportation/Newsstands/w9zq-xm8b) |
| Parking meters | `data/street_furniture/parking_meters_nyc.csv` | [NYC OpenData 693u-uax6](https://data.cityofnewyork.us/Transportation/Parking-Meters-GPS-Coordinates-and-Status/693u-uax6) |

`pull_data.sh` also fetches a surveillance-camera zip from Google Cloud. `dataset.ipynb` does **not** use that file; use the Ban the Scan download below instead.

### 2. Manual downloads

These wget URLs fail, are commented out in `pull_data.sh`, or the notebook was pointed at a different file. Download from the browser (Export → CSV or Shapefile on NYC OpenData) and place them as below.

**Basemap / admin boundaries** (needed before segmentizing sidewalks)

| Dataset | Where to get it | Put it here |
|---|---|---|
| 2020 Census Blocks | [2020 Census Blocks](https://data.cityofnewyork.us/City-Government/2020-Census-Blocks/wmsu-5muw/data_preview) — Export shapefile. The DCP `nycb2020_24c.zip` wget does not work. | `data/2020 Census Blocks_YYYYMMDD/*.shp` |
| Community Districts | [Community Districts](https://data.cityofnewyork.us/City-Government/Community-Districts/yfnk-k7r4) — Export shapefile. The geospatial wget does not work. | `data/Community Districts_YYYYMMDD/*.shp` |
| Planimetric sidewalks | [NYC Planimetric Database: Sidewalk](https://data.cityofnewyork.us/City-Government/NYC-Planimetric-Database-Sidewalk/vfx9-tbb6) — CSV export. Then run `sidewalk_widths.py` (see run order). | `data/NYC_Planimetric_Database__Sidewalk_YYYYMMDD.csv` |

**Connectivity / charging**

| Dataset | Where to get it | Put it here |
|---|---|---|
| FCC GSO + NGSO satellite (CSV) and 4G LTE + 5G NR (ESRI shapefiles) | [FCC National Broadband Map data download](https://broadbandmap.fcc.gov/data-download/nationwide-data?version=dec2025&pubDataVer=jun2024). Select **New York** state. | `data/bdc_36_GSOSatellite_fixed_broadband_*.csv`, `data/bdc_36_NGSOSatellite_fixed_broadband_*.csv`, `data/4g_ny/bdc_36_4GLTE_mobile_broadband_h3_*.shp`, `data/5g_ny/bdc_36_5GNR_mobile_broadband_h3_*.shp` |
| CitiBike stations | **Not** the trip-history zip in `pull_data.sh` (that file is unused). Stations come from GBFS: `mkdir -p data/citibike && wget https://gbfs.citibikenyc.com/gbfs/en/station_information.json -O data/citibike/station_information.json` | `data/citibike/station_information.json` |
| Zoning districts | [DCP GIS zoning features](https://www.nyc.gov/content/planning/pages/resources/datasets/gis-zoning-features) (June 2026 shapefile). The OpenData zoning wget does not work. | `data/nycgiszoningfeatures_YYYYMMshp/nyzd.shp` |

**Other `dataset.ipynb` inputs**

| Dataset | Where to get it | Put it here |
|---|---|---|
| Surveillance cameras | [Amnesty Ban the Scan / Decode](https://banthescan.amnesty.org/decode/index.html#explore-the-data) — NYC intersection counts | `data/decode-surveillance-nyc-1.1.0/data/counts_per_intersections.csv` |
| Sidewalk cleanliness scorecard | [Scorecard Ratings](https://data.cityofnewyork.us/City-Government/Scorecard-Ratings/rqhp-hivt/about_data) | `data/Scorecard_Ratings_YYYYMMDD.csv` |
| 2015 Street Tree Census | [2015 Street Tree Census](https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/uvpi-gqnh/about_data) (separate from forestry tree points above) | `data/2015_Street_Tree_Census_-_Tree_Data_YYYYMMDD.csv` |
| Vision Zero speed limits | [VZV Speed Limits](https://data.cityofnewyork.us/Transportation/VZV-Speed-Limits/5mad-ntua/about_data) | `data/VZV_Speed_Limits_YYYYMMDD.csv` |
| Neighborhood slow zones | [VZV Neighborhood Slow Zones](https://data.cityofnewyork.us/Transportation/VZV-Neighborhood-Slow-Zones/bqye-aqft/about_data) | `data/VZV_Neighborhood_Slow_Zones_YYYYMMDD.csv` |
| Turn traffic calming | [VZV Turn Traffic Calming](https://data.cityofnewyork.us/Transportation/VZV-Turn-Traffic-Calming/hz4p-9f7s) | `data/VZV_Turn_Traffic_Calming_YYYYMMDD.csv` |
| SIP intersections | [VZV SIP Intersections](https://data.cityofnewyork.us/Transportation/VZV-Street-Improvement-Projects-SIP-Intersections/shr7-eqdc/about_data) | `data/VZV_Street_Improvement_Projects_(SIP)_Intersections_YYYYMMDD.csv` |
| SIP corridors | [VZV SIP Corridor](https://data.cityofnewyork.us/Transportation/VZV-Street-Improvement-Projects-SIP-Corridor/if4c-w48d/about_data) | `data/VZV_Street_Improvement_Projects_(SIP)_Corridor_YYYYMMDD.csv` |
| Barnes Dance intersections | [Exclusive Pedestrian Signal (Barnes Dance) Locations](https://data.cityofnewyork.us/Transportation/Exclusive-Pedestrian-Signal-Barnes-Dance-Locations/8kuj-2n3u/about_data) | `data/Exclusive_Pedestrian_Signal_(Barnes_Dance)_Locations_YYYYMMDD.csv` |
| Leading pedestrian interval signals | [VZV Leading Pedestrian Interval Signals](https://data.cityofnewyork.us/Transportation/VZV-Leading-Pedestrian-Interval-Signals/xc4v-ntf4/about_data) | `data/VZV_Leading_Pedestrian_Interval_Signals_YYYYMMDD.csv` |
| Bike routes | [New York City Bike Routes](https://data.cityofnewyork.us/dataset/New-York-City-Bike-Routes-Map-/9e2b-mctv) (the older `mzxg-pwib` page is often unavailable) | `data/New_York_City_Bike_Routes_YYYYMMDD.csv` |
| Motor vehicle collisions | [Motor Vehicle Collisions - Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data) | `data/Motor_Vehicle_Collisions_-_Crashes_YYYYMMDD.csv` |

**Street furniture** (used by `street_furniture.ipynb`; not wget-able)

| Dataset | Where to get it | Put it here |
|---|---|---|
| LinkNYC kiosks | [LinkNYC Kiosk Locations](https://data.cityofnewyork.us/Social-Services/LinkNYC-Kiosk-Locations/s4kf-3yrf/about_data) | `data/street_furniture/LinkNYC_Kiosk_Locations_YYYYMMDD.csv` |
| Bicycle racks | [Bicycle Parking](https://data.cityofnewyork.us/Transportation/Bicycle-Parking/yh4a-g3fj) — Export shapefile (the “Original” wget does not work) | `data/street_furniture/Bicycle Parking_YYYYMMDD/*.shp` |
| Street sign work orders | [Street Sign Work Orders](https://data.cityofnewyork.us/Transportation/Street-Sign-Work-Orders/qt6m-xctn) — CSV; the notebook keeps `record_type == Current` | `data/street_furniture/Street_Sign_Work_Orders_YYYYMMDD.csv` |
| Traffic bollards | [Traffic Bollards Tracking and Installations](https://data.cityofnewyork.us/Transportation/Traffic-Bollards-Tracking-and-Installations/3f5t-9dqu/about_data) | `data/street_furniture/Traffic_Bollards_Tracking_and_Installations_YYYYMMDD.csv` |
| In-service alarm boxes | [In-Service Alarm Box Locations](https://data.cityofnewyork.us/Public-Safety/In-Service-Alarm-Box-Locations/v57i-gtxb/about_data) | `data/street_furniture/In-Service_Alarm_Box_Locations_YYYYMMDD.csv` |
| Scaffolding / sheds | Original `dob_active_sheds.csv` is gone. Use [DOB NOW: Build - Job Application Filings](https://data.cityofnewyork.us/Housing-Development/DOB-NOW-Build-Job-Application-Filings/w9ak-ipjd/about_data) | `data/DOB_NOW__Build___Job_Application_Filings.csv` |

Bollards are loaded but not joined onto sidewalks (no lat/lon in that table). Public recycling bins are listed in `pull_data.sh` but are not used.

### 3. How to run `dataset.ipynb`

Street furniture density is produced by a second notebook, and `dataset.ipynb` reads that CSV in the middle of the pipeline. You have to stop, run the other notebook, then continue.

1. From `feature_processing/`, download inputs (`pull_data.sh` plus the manual files above).

2. Build sidewalk widths from the planimetric CSV:

   ```bash
   cd feature_processing
   python sidewalk_widths.py
   ```

   That writes `data/sidewalk_widths.geojson` (used when `USE_ALTERNATE_GEN_METHOD = True`).

3. Open `dataset.ipynb`. Set `REGEN_SEGMENTIZATION = True` in the segmentize cell and run from the top **through sidewalk segmentization**. That writes `data/sidewalks_nyc_segmentized.csv`. Stop before the **Street Furniture** cell (`data/processed/street_furniture_density.csv` does not exist yet).

4. Open `street_furniture.ipynb` and run it. It reads `data/sidewalks_nyc_segmentized.csv` and writes `data/processed/street_furniture_density.csv`.

5. Go back to `dataset.ipynb`. Set `REGEN_SEGMENTIZATION = False` (reload the segmentized CSV instead of recomputing it) and continue from the Street Furniture cell through the end. That writes `data/processed/score_dataset.csv`.

Without Nexar dashcam detections, the Traffic Density cell fills `TRAFFIC_Pedestrian`, `TRAFFIC_Bike`, and `TRAFFIC_Car` with `np.random.uniform(0, 1)` instead of `avg_traffic_by_sidewalk_august.csv`. Scores that use pedestrian / bicycle / vehicle density will not match the paper.

## Changelog

- Replaced the private `/share/ju/urban-fingerprinting` import with in-repo `src/utils` (logger + matplotlib; LaTeX falls back to mathtext).
- Feature notebooks/scripts now use `feature_processing/data/` (run them from that directory).
- `pull_data.sh` reports download/skip/fail and comments out wget URLs that no longer work.
- Input paths updated to currently downloadable files (OpenData shapefile/CSV exports, FCC 2024 NY broadband, DCP zoning, GBFS CitiBike stations, Ban the Scan cameras, DOB NOW filings for scaffolding). Census-block columns are lowercased.
- Motor-vehicle collisions cell in `dataset.ipynb` is skipped (kernel crash).
- No Nexar data: pedestrian / bike / car traffic in `dataset.ipynb` is `np.random.uniform(0, 1)` instead of dashcam detections.
- `survey_splits.ipynb` reads `survey_52.csv`.
- Added `requirements.txt` (Python 3.11).
