#!/usr/bin/env bash
# Download NYC OpenData (and related) inputs into data/.
# Tracks what was downloaded, already present, or failed, and prints a summary.

mkdir -p data data/street_furniture

downloaded=()
present=()
failed=()

# Fetch a file if dest is missing. name is the summary label.
fetch_file() {
    local name="$1"
    local dest="$2"
    local url="$3"
    if [ -f "$dest" ]; then
        present+=("$name")
        return
    fi
    if wget "$url" -O "$dest"; then
        downloaded+=("$name")
    else
        rm -f "$dest"
        failed+=("$name")
    fi
}

# Fetch a zip, unzip into unzip_dir, then drop the zip. check is the file or dir
# that means the dataset is already present.
fetch_zip() {
    local name="$1"
    local check="$2"
    local zip_path="$3"
    local unzip_dir="$4"
    local url="$5"
    if [ -e "$check" ]; then
        present+=("$name")
        return
    fi
    if wget "$url" -O "$zip_path" && unzip "$zip_path" -d "$unzip_dir"; then
        rm -f "$zip_path"
        downloaded+=("$name")
    else
        rm -f "$zip_path"
        failed+=("$name")
    fi
}

print_list() {
    local title="$1"
    shift
    echo "$title ($#):"
    if [ $# -eq 0 ]; then
        echo "  (none)"
    else
        for item in "$@"; do
            echo "  - $item"
        done
    fi
}

fetch_file "sidewalks_nyc" data/sidewalks_nyc.csv \
    'https://data.cityofnewyork.us/api/views/52n9-sdep/rows.csv?date=20240814&accessType=DOWNLOAD'

# nyc 2020 ntas
fetch_file "ntas_nyc" data/ntas_nyc.csv \
    'https://data.cityofnewyork.us/api/views/9nt8-h7nd/rows.csv?accessType=DOWNLOAD'

# nyc 2020 census blocks (zip kept as-is) # does not work go manually see notes
# fetch_file "nyc_cbs" data/nyc_cbs.zip \
#     'https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/nycb2020_24c.zip'

# # nyc community districts (zip kept as-is) # does not work go manually see notes
# fetch_file "community_districts_nyc" data/community_districts_nyc.zip \
#     'https://data.cityofnewyork.us/api/geospatial/yfnk-k7r4?method=export&format=Shapefile'

# nyc Sidewalk cleanliness scorecard # manual download see notes
# https://data.cityofnewyork.us/City-Government/Scorecard-Ratings/rqhp-hivt/about_data

# ny 4g-lte map
# https://broadbandmap.fcc.gov/data-download/nationwide-data?version=dec2023

# ny 5g-nr map
# https://broadbandmap.fcc.gov/data-download/nationwide-data?version=dec2023

# 2023/12 nyc citibike data # not used to get station_information.json got it manually see notes
# fetch_zip "citibike_202312_nyc" data/citibike_202312_nyc.csv \
#     data/citibike_202312.csv.zip citibike_202312.csv \
#     'https://s3.amazonaws.com/tripdata/JC-202312-citibike-tripdata.csv.zip'

# nyc pedestrian curb ramp
fetch_file "pedestrian_curb_ramp_nyc" data/pedestrian_curb_ramp_nyc.csv \
    'https://data.cityofnewyork.us/api/views/ufzp-rrqu/rows.csv?accessType=DOWNLOAD'

# surveillance cameras from surveilling surveillance paper
fetch_zip "surveillance_cameras" data/surveillance_cameras \
    data/surveillance_cameras.zip data/surveillance_cameras \
    'https://storage.googleapis.com/scpl-surveillance/camera-data.zip'

# nyc raised crosswalk locations
fetch_file "raised_crosswalks_nyc" data/raised_crosswalks_nyc.csv \
    'https://data.cityofnewyork.us/api/views/uh2s-ftgh/rows.csv?accessType=DOWNLOAD'

# nyc VZW enhanced crossings locations
fetch_file "vzw_enhanced_crossings_nyc" data/vzw_enhanced_crossings_nyc.csv \
    'https://data.cityofnewyork.us/api/views/k9a2-vdr8/rows.csv?accessType=DOWNLOAD'

# nyc zoning shapefile # does not work go manually see notes
# fetch_zip "zoning_nyc" data/zoning_nyc \
#     data/zoning_nyc.zip data/zoning_nyc \
#     'https://data.cityofnewyork.us/api/geospatial/kdig-pewd?method=export&format=Shapefile'

# nyc 1 foot dem integer raster
fetch_zip "1ft_dem_nyc" data/1ft_dem_nyc \
    data/1ft_dem_nyc.zip data/1ft_dem_nyc \
    'https://sa-static-customer-assets-us-east-1-fedramp-prod.s3.amazonaws.com/data.cityofnewyork.us/NYC_DEM_1ft_Int.zip'

# nyc pois
fetch_file "pois_nyc" data/pois_nyc.csv \
    'https://data.cityofnewyork.us/api/views/t95h-5fsr/rows.csv?accessType=DOWNLOAD'

# will retain clutter from claustrophobic streets analysis

# nyc bike lanes (weirdly unavailable for download right now)
# https://data.cityofnewyork.us/dataset/New-York-City-Bike-Routes/mzxg-pwib/about_data




# STREET CLUTTER

# nyc in service alarm box locations
# https://data.cityofnewyork.us/Public-Safety/In-Service-Alarm-Box-Locations/v57i-gtxb/about_data
# NOT DOWNLOADABLE THROUGH WGET

# public recycling bins
# https://data.cityofnewyork.us/Environment/Public-Recycling-Bins/sxx4-xhzg/about_data
# NOT DOWNLOADABLE THROUGH WGET

# dsny litter baskets
fetch_file "dsny_litter_baskets_nyc" data/street_furniture/dsny_litter_baskets_nyc.csv \
    'https://data.cityofnewyork.us/api/views/8znf-7b2c/rows.csv?accessType=DOWNLOAD'

# nyc fire hydrants
fetch_file "fire_hydrants_nyc" data/street_furniture/fire_hydrants_nyc.csv \
    'https://data.cityofnewyork.us/api/views/5bgh-vtsn/rows.csv?accessType=DOWNLOAD'

# nyc bus stop shelters
fetch_file "bus_stop_shelters_nyc" data/street_furniture/bus_stop_shelters_nyc.csv \
    'https://data.cityofnewyork.us/api/views/t4f2-8md7/rows.csv?accessType=DOWNLOAD'

# nyc linknyc kiosks
# https://data.cityofnewyork.us/Social-Services/LinkNYC-Kiosk-Locations/s4kf-3yrf/about_data
# NOT DOWNLOADABLE THROUGH WGET

# nyc bicycle parking shelters
fetch_file "bicycle_parking_shelters_nyc" data/street_furniture/bicycle_parking_shelters_nyc.csv \
    'https://data.cityofnewyork.us/api/views/dimy-qyej/rows.csv?accessType=DOWNLOAD'

# nyc bicycle racks # does not work go manually see notes
# fetch_zip "bicycle_racks_nyc" data/street_furniture/bicycle_racks_nyc \
#     data/street_furniture/bicycle_racks_nyc.zip data/street_furniture/bicycle_racks_nyc \
#     'https://data.cityofnewyork.us/api/geospatial/yh4a-g3fj?method=export&format=Original'

# nyc citybench
fetch_file "citybench_nyc" data/street_furniture/citybench_nyc.csv \
    'https://data.cityofnewyork.us/api/views/kuxa-tauh/rows.csv?accessType=DOWNLOAD'

# nyc forestry tree points
fetch_file "forestry_tree_points_nyc" data/street_furniture/forestry_tree_points_nyc.csv \
    'https://data.cityofnewyork.us/api/views/uvpi-gqnh/rows.csv?accessType=DOWNLOAD'

# nyc newsstands
fetch_file "newsstands_nyc" data/street_furniture/newsstands_nyc.csv \
    'https://data.cityofnewyork.us/api/views/w9zq-xm8b/rows.csv?accessType=DOWNLOAD'

# nyc parking meters
fetch_file "parking_meters_nyc" data/street_furniture/parking_meters_nyc.csv \
    'https://data.cityofnewyork.us/api/views/693u-uax6/rows.csv?date=20240816&accessType=DOWNLOAD'


# nyc current street sign work orders
# https://data.cityofnewyork.us/Transportation/Street-Sign-Work-Orders/qt6m-xctn/explore/query/SELECT%0A%20%20%60order_number%60%2C%0A%20%20%60record_type%60%2C%0A%20%20%60order_type%60%2C%0A%20%20%60borough%60%2C%0A%20%20%60on_street%60%2C%0A%20%20%60on_street_suffix%60%2C%0A%20%20%60from_street%60%2C%0A%20%20%60from_street_suffix%60%2C%0A%20%20%60to_street%60%2C%0A%20%20%60to_street_suffix%60%2C%0A%20%20%60side_of_street%60%2C%0A%20%20%60order_completed_on_date%60%2C%0A%20%20%60sign_code%60%2C%0A%20%20%60sign_description%60%2C%0A%20%20%60sign_size%60%2C%0A%20%20%60sign_design_voided_on_date%60%2C%0A%20%20%60sign_location%60%2C%0A%20%20%60distance_from_intersection%60%2C%0A%20%20%60arrow_direction%60%2C%0A%20%20%60facing_direction%60%2C%0A%20%20%60sheeting_type%60%2C%0A%20%20%60support%60%2C%0A%20%20%60sign_notes%60%2C%0A%20%20%60sign_x_coord%60%2C%0A%20%20%60sign_y_coord%60%0AWHERE%20caseless_one_of%28%60record_type%60%2C%20%22Current%22%29/page/filter
# NOT DOWNLOADABLE THROUGH WGET


# nyc bollards
# https://data.cityofnewyork.us/Transportation/Traffic-Bollards-Tracking-and-Installations/3f5t-9dqu/about_data
# NOT DOWNLOADABLE THROUGH WGET

# traffic management systems

# vision zero - street improvement project intersections
# https://data.cityofnewyork.us/Transportation/VZV_Street-Improvement-Projects-SIPs-intersections/79sh-heg3

# vision zero - turn traffic calming
# https://data.cityofnewyork.us/Transportation/VZV_Turn-Traffic-Calming/hz4p-9f7s

# vision zero - leading pedestrian interval signals
# https://data.cityofnewyork.us/Transportation/VZV_Leading-Pedestrian-Interval-Signals/mqt5-ctec

# vision zero - street improvement project corridors
# https://data.cityofnewyork.us/Transportation/VZV_Street-Improvement-Projects-SIPs-Corridor/wqhs-q6wd

# vision zero - speed humps
# https://data.cityofnewyork.us/Transportation/VZV_Speed-Humps/7f9e-jic4

# barnes dance intersections
# https://data.cityofnewyork.us/Transportation/Exclusive-Pedestrian-Signal-Barnes-Dance-Locations/8kuj-2n3u/about_data

echo
echo "========== Download summary =========="
print_list "Downloaded" "${downloaded[@]}"
print_list "Already present" "${present[@]}"
print_list "Failed" "${failed[@]}"
echo "======================================"

if [ "${#failed[@]}" -gt 0 ]; then
    exit 1
fi
