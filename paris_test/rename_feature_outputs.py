import os

import geopandas as gpd


def rewrite_feature(in_name, out_name):
    """Copy an existing feature GeoJSON to a new canonical name and rename prefixed columns."""
    root = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(root, "data", "processed")
    in_path = os.path.join(processed_dir, f"{in_name}_paris.geojson")
    out_path = os.path.join(processed_dir, f"{out_name}_paris.geojson")
    if not os.path.exists(in_path):
        print(f"Skipping missing {in_path}")
        return
    gdf = gpd.read_file(in_path)
    rename_map = {col: col.replace(in_name, out_name, 1) for col in gdf.columns if in_name in col}
    gdf = gdf.rename(columns=rename_map)
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    rewrite_feature("curb_ramps", "curb_ramp_availability")
    rewrite_feature("zoning_regulation", "zoning_laws")
    rewrite_feature("charging_stations", "charging_station_proximity")
    rewrite_feature("bike_traffic", "bicycle_traffic")
    rewrite_feature("bike_lane", "bike_lane_availability")
