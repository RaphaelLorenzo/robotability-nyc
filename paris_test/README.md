# Paris test

Small testbed to see how far we can reproduce the NYC sidewalk base map in Paris with open data.

## Scripts

- `download_data.py`: downloads the Paris sidewalk, PVP tile, arrondissement, and pedestrian-density source layers from Paris Data, plus the Île-de-France tourist-site layer.
- `process_sidewalks.py`: cleans sidewalk geometries, keeps one row per sidewalk polygon, and adds `pvp_tile` and `arrondissement`.
- `add_quartier_administratif.py`: downloads `quartier_paris` and adds QA identifiers to both processed sidewalk formats.
- `sidewalk_widths.py`: extracts skeleton centerlines from each sidewalk polygon, splits them into short segments, and estimates the local sidewalk width from the average centerline-to-boundary distance. Carries `pvp_tile` and `arrondissement` labels into the output segments.
- `segmentize_sidewalks.py`: rebuilds one centerline per sidewalk and samples points every 50 ft (15.24 m), matching the NYC sidewalk segmentize step.
- `compute_pedestrian_density.py`: buffers each sample point by 25 ft (7.62 m) for sidewalk-adjacent features and 200 ft (60.96 m) for schools, kiosks, municipal sites, and 2025 activities; counts tourist sites per QA; then computes the weighted score.
- `compute_crowd_dynamics.py`: flags sample points inside Zones Touristiques Internationales (0.5) and adds a 0.5-weighted QA tourist-site count, then inverts the tourism score so residential areas are high, matching NYC polarity.
- `make_maps.py`: creates QA maps for the processed sidewalks and, when available, raw/clamped pedestrian density and crowd dynamics maps.

## Run order

```bash
conda run -n robotability python paris_test/download_data.py
conda run -n robotability python paris_test/process_sidewalks.py
conda run -n robotability python paris_test/sidewalk_widths.py   # slow ~minutes
conda run -n robotability python paris_test/add_quartier_administratif.py
conda run -n robotability python paris_test/segmentize_sidewalks.py
conda run -n robotability python paris_test/compute_pedestrian_density.py
conda run -n robotability python paris_test/compute_crowd_dynamics.py
conda run -n robotability python paris_test/make_maps.py
```

## Outputs

- `paris_test/data/raw/`: downloaded GeoJSON files
- `paris_test/data/processed/sidewalks_paris.geojson`: one polygon per sidewalk, with `pvp_tile`, `arrondissement`, and QA identifiers
- `paris_test/data/processed/sidewalk_widths_paris.geojson`: one segment per centerline piece, with `width_m`, `pvp_tile`, `arrondissement`, and QA identifiers
- `paris_test/data/processed/sidewalks_paris_segmentized.geojson`: NYC-style sample points every 50 ft along reconstructed sidewalk centerlines
- `paris_test/data/processed/pedestrian_density_paris.geojson`: sample points enriched with buffered component counts, normalized scores, and the final weighted score
- `paris_test/data/processed/pedestrian_density_qa.geojson`: quartier administratif population density layer used for the pedestrian base density
- `paris_test/data/processed/crowd_dynamics_paris.geojson`: sample points with ZTI flags, QA tourist-site counts, tourism score, and inverted crowd-dynamics score
- `paris_test/figures/`: output maps
