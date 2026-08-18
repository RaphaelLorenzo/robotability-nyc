# Paris test

Small testbed to see how far we can reproduce the NYC sidewalk base map in Paris with open data.

## Scripts

- `download_data.py`: downloads the Paris sidewalk, PVP tile, and arrondissement layers from the Paris Data API.
- `process_sidewalks.py`: cleans sidewalk geometries, keeps one row per sidewalk polygon, and adds `pvp_tile` and `arrondissement`.
- `sidewalk_widths.py`: extracts skeleton centerlines from each sidewalk polygon, splits them into short segments, and estimates the local sidewalk width from the average centerline-to-boundary distance. Carries `pvp_tile` and `arrondissement` labels into the output segments.
- `make_maps.py`: creates two simple maps of processed sidewalks colored by `pvp_tile` and `arrondissement`.

## Run order

```bash
conda run -n robotability python paris_test/download_data.py
conda run -n robotability python paris_test/process_sidewalks.py
conda run -n robotability python paris_test/sidewalk_widths.py   # slow ~minutes
conda run -n robotability python paris_test/make_maps.py
```

## Outputs

- `paris_test/data/raw/`: downloaded GeoJSON files
- `paris_test/data/processed/sidewalks_paris.geojson`: one polygon per sidewalk, with `pvp_tile` and `arrondissement`
- `paris_test/data/processed/sidewalk_widths_paris.geojson`: one segment per centerline piece, with `width_m`, `pvp_tile`, and `arrondissement`
- `paris_test/figures/`: output maps
