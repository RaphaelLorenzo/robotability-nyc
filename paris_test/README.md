# Paris test

Small testbed to see how far we can reproduce the NYC sidewalk base map in Paris with open data.

## Scripts

- `download_data.py`: downloads the Paris sidewalk, PVP tile, arrondissement, pedestrian-density, crowd-dynamics, surface-condition, and intersection-safety source layers from Paris Data, plus the Île-de-France tourist-site layer.
- `process_sidewalks.py`: cleans sidewalk geometries, keeps one row per sidewalk polygon, and adds `pvp_tile` and `arrondissement`.
- `add_quartier_administratif.py`: downloads `quartier_paris` and adds QA identifiers to both processed sidewalk formats.
- `sidewalk_widths.py`: extracts skeleton centerlines from each sidewalk polygon, splits them into short segments, and estimates the local sidewalk width from the average centerline-to-boundary distance. Carries `pvp_tile` and `arrondissement` labels into the output segments.
- `segmentize_sidewalks.py`: rebuilds one centerline per sidewalk and samples points every 50 ft (15.24 m), matching the NYC sidewalk segmentize step.
- `compute_pedestrian_density.py`: buffers each sample point by 25 ft (7.62 m) for sidewalk-adjacent features and 200 ft (60.96 m) for schools, kiosks, municipal sites, and 2025 activities; counts tourist sites per QA; then computes the weighted score.
- `compute_crowd_dynamics.py`: flags sample points inside Zones Touristiques Internationales (0.5) and adds a 0.5-weighted QA tourist-site count, then inverts the tourism score so residential areas are high, matching NYC polarity.
- `compute_surface_condition.py`: starts from 1 (good), subtracts 0.25 if a 25 ft buffer intersects a chantier occupying a sidewalk or bike lane, and subtracts 0.75 times the clamped Dans Ma Rue surface-anomaly count.
- `compute_intersection_safety.py`: starts from 0.75, raises to 0.9 in a zone de rencontre and 1.0 in an aire piétonne, then subtracts a 0.2 arrondissement accident-rate malus and a 0.25 clamped Dans Ma Rue intersection-anomaly malus.
- `make_maps.py`: creates QA maps for the processed sidewalks and, when available, raw/clamped feature maps. Use `--features intersection_safety` to plot only that feature.

## Run order

```bash
conda run -n robotability python paris_test/download_data.py
conda run -n robotability python paris_test/process_sidewalks.py
conda run -n robotability python paris_test/sidewalk_widths.py   # slow ~minutes
conda run -n robotability python paris_test/add_quartier_administratif.py
conda run -n robotability python paris_test/segmentize_sidewalks.py
conda run -n robotability python paris_test/compute_pedestrian_density.py
conda run -n robotability python paris_test/compute_crowd_dynamics.py
conda run -n robotability python paris_test/compute_surface_condition.py
conda run -n robotability python paris_test/compute_intersection_safety.py
conda run -n robotability python paris_test/make_maps.py
conda run -n robotability python paris_test/make_maps.py --features intersection_safety
```

## Outputs

- `paris_test/data/raw/`: downloaded GeoJSON files
- `paris_test/data/processed/sidewalks_paris.geojson`: one polygon per sidewalk, with `pvp_tile`, `arrondissement`, and QA identifiers
- `paris_test/data/processed/sidewalk_widths_paris.geojson`: one segment per centerline piece, with `width_m`, `pvp_tile`, `arrondissement`, and QA identifiers
- `paris_test/data/processed/sidewalks_paris_segmentized.geojson`: NYC-style sample points every 50 ft along reconstructed sidewalk centerlines
- `paris_test/data/processed/pedestrian_density_paris.geojson`: sample points enriched with buffered component counts, normalized scores, and the final weighted score
- `paris_test/data/processed/pedestrian_density_qa.geojson`: quartier administratif population density layer used for the pedestrian base density
- `paris_test/data/processed/crowd_dynamics_paris.geojson`: sample points with ZTI flags, QA tourist-site counts, tourism score, and inverted crowd-dynamics score
- `paris_test/data/processed/surface_condition_paris.geojson`: sample points with chantier flags, Dans Ma Rue anomaly counts, and the final surface-condition score
- `paris_test/data/processed/intersection_safety_paris.geojson`: sample points with calmed-zone flags, arrondissement accident rates, Dans Ma Rue anomalies, and the final intersection-safety score
- `paris_test/figures/`: output maps
