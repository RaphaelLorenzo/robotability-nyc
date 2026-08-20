# Paris test

Small testbed to see how far we can reproduce the NYC sidewalk base map in Paris with open data.

## Scripts

- `download_data.py`: downloads all source layers from Paris Data (sidewalks, PVP tiles, arrondissements, pedestrian-density, crowd-dynamics, surface-condition, intersection-safety, street-furniture, curb-ramp, and traffic-management layers) plus the Île-de-France tourist-site layer.
- `process_sidewalks.py`: cleans sidewalk geometries, keeps one row per sidewalk polygon, and adds `pvp_tile` and `arrondissement`.
- `add_quartier_administratif.py`: downloads `quartier_paris` and adds QA identifiers to both processed sidewalk formats.
- `sidewalk_widths.py`: extracts skeleton centerlines from each sidewalk polygon, splits them into short segments, and estimates the local sidewalk width from the average centerline-to-boundary distance. Carries `pvp_tile` and `arrondissement` labels into the output segments.
- `segmentize_sidewalks.py`: rebuilds one centerline per sidewalk and samples points every 50 ft (15.24 m), matching the NYC sidewalk segmentize step.
- `compute_pedestrian_density.py`: buffers each sample point by 25 ft (7.62 m) for sidewalk-adjacent features and 200 ft (60.96 m) for schools, kiosks, municipal sites, and 2025 activities; counts tourist sites per QA; then computes the weighted score.
- `compute_crowd_dynamics.py`: flags sample points inside Zones Touristiques Internationales (0.5) and adds a 0.5-weighted QA tourist-site count, then inverts the tourism score so residential areas are high, matching NYC polarity.
- `compute_surface_condition.py`: starts from 1 (good), subtracts 0.25 if a 25 ft buffer intersects a chantier occupying a sidewalk or bike lane, and subtracts 0.75 times the clamped Dans Ma Rue surface-anomaly count.
- `compute_intersection_safety.py`: starts from 0.75, raises to 0.9 in a zone de rencontre and 1.0 in an aire piétonne, then subtracts a 0.2 arrondissement accident-rate malus and a 0.25 clamped Dans Ma Rue intersection-anomaly malus.
- `compute_street_furniture_density.py`: counts nearby PVP street furniture, Trilib, composteurs, available Paris drinking fountains, and tagged Dans Ma Rue clutter reports in 25 ft buffers; combines into a weighted `street_furniture_density_score` (0 = none, 1 = dense).
- `compute_curb_ramps.py`: scores curb-ramp availability — 0.75 everywhere, 0.0 on voies en escalier (buffered 5 m), 0.9 inside quartiers d'accessibilité augmentée.
- `compute_traffic_management.py`: starts from 0.5, adds 0.5 × clamped feux-tricolores count (200 ft radius), subtracts 0.25 × clamped Dans Ma Rue traffic-management anomaly count.
- `compute_zoning_regulation.py`: scores zoning regulation — 0.3 base, 0.7 in aires piétonnes or zones de rencontre, +0.1 in ZTL, +0.2 in Paris Respire secteurs.
- `compute_slope_gradient.py`: averages nivellement altitudes within 25 ft of each sample point, then computes the mean slope (rise/run) to neighbors within 50 ft; normalizes so 1 = steeper (NYC polarity -1 applied later).
- `compute_street_lighting.py`: counts public lamps within 25 ft, normalizes 2.5-99.5% to 0-1.
- `compute_bike_lane.py`: flags adjacency (sidewalk width × 1.5, ≤8 m) to piste (0.8) or piste cyclable (1.0); couloir mixte excluded.
- `compute_bike_traffic.py`: 0.8 if adjacent to piste/couloir mixte + 0.2 × clamped Vélib station count (200 ft).
- `compute_charging_stations.py`: counts Vélib stations within 200 ft, normalizes 2.5-99.5% to 0-1.
- `compute_shade.py`: counts trees ≥3 m within 50 ft (young trees weighted 0.5), normalizes 2.5-99.5% to 0-1.
- `compute_vehicle_traffic.py`: queries the 2026-06-01 average road occupation rate, clamps 2.5-99.5%, and assigns the nearest traffic arc within 50 ft.
- `compute_sidewalk_width.py`: turns `width_m` already present on the segmentized sidewalks into a dedicated 0-1 width feature.
- `compute_communication_infrastructure.py`, `compute_digital_map_existence.py`, `compute_gps_signal_strength.py`, `compute_sidewalk_roughness.py`, `compute_surveillance_coverage.py`: assign fixed values directly for now (`1.0` for communication / map / GPS / surveillance, `0.0` for sidewalk roughness).
- `rename_feature_outputs.py`: one-shot helper that renames already computed Paris outputs to the canonical `feature_weights.csv` names without recomputing.
- `compute_robotability_score.py`: merges all feature layers into one CSV, applies `feature_weights.csv` and NYC polarities, and writes a per-point GeoJSON score file.
- `make_robotability_maps.py`: maps the final robotability score at `point` (default), `segment`, `qa`, `pvp`, or `arrondissement` level; writes a 0–1 map and a quantile map for each.
- `make_maps.py`: creates QA maps for processed sidewalks and raw/clamped feature maps for each score. Use `--features <name>` to plot a single group.
- `sample_paris_street_view.py`: samples points from `robotability_features_paris.csv`, estimates the local sidewalk heading from neighboring points, downloads front/right/back/left Google Street View images, and writes an NYC-style `sample_paris_street/` tree (`splits/`, `full/`, `full_with_vis/`, `metadata.yaml`) with coordinate-based filenames.

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
conda run -n robotability python paris_test/compute_street_furniture_density.py
conda run -n robotability python paris_test/compute_curb_ramps.py
conda run -n robotability python paris_test/compute_traffic_management.py
conda run -n robotability python paris_test/compute_zoning_regulation.py
conda run -n robotability python paris_test/compute_slope_gradient.py
conda run -n robotability python paris_test/compute_street_lighting.py
conda run -n robotability python paris_test/compute_bike_lane.py
conda run -n robotability python paris_test/compute_bike_traffic.py
conda run -n robotability python paris_test/compute_charging_stations.py
conda run -n robotability python paris_test/compute_shade.py
conda run -n robotability python paris_test/compute_vehicle_traffic.py
conda run -n robotability python paris_test/compute_sidewalk_width.py
conda run -n robotability python paris_test/compute_communication_infrastructure.py
conda run -n robotability python paris_test/compute_digital_map_existence.py
conda run -n robotability python paris_test/compute_gps_signal_strength.py
conda run -n robotability python paris_test/compute_sidewalk_roughness.py
conda run -n robotability python paris_test/compute_surveillance_coverage.py
conda run -n robotability python paris_test/rename_feature_outputs.py   # only needed to convert old file names
conda run -n robotability python paris_test/compute_robotability_score.py
conda run -n robotability python paris_test/sample_paris_street_view.py --sample_n 10
conda run -n robotability python paris_test/sample_paris_street_view.py --resume  # refreshes full_with_vis and retries failures
conda run -n robotability python paris_test/make_robotability_maps.py --level all
conda run -n robotability python paris_test/make_maps.py
```

## Feature recap

| Feature | Paris | New York | Polarity |
| --- | --- | --- | --- |
| `pedestrian_density` | Population + amenities + schools + stops + activities | Population + local attractors | `1 = denser pedestrian context` |
| `crowd_dynamics` | ZTI + tourist-site signal, then inverted | Tourism / crowd proxy, inverted | `1 = calmer / less tourist-heavy` |
| `surface_condition` | Chantiers + DMR surface anomalies | Sidewalk quality / defects | `1 = better surface` |
| `sidewalk_width` | Normalize existing `width_m` on segmentized sidewalks | Normalize sidewalk width | `1 = wider` |
| `street_furniture_density` | PVP furniture + Trilib + fountains + clutter reports | Street-furniture density | `1 = denser furniture` |
| `intersection_safety` | Aires piétonnes / zones de rencontre + accidents + DMR | Calmed / safer intersections | `1 = safer` |
| `curb_ramp_availability` | Escalier penalty + accessibility-quarter bonus | Actual curb-ramp availability | `1 = more available` |
| `communication_infrastructure` | Fixed `1.0` | Sensor / comms availability | `1 = more infrastructure` |
| `digital_map_existence` | Fixed `1.0` | Digital-map coverage | `1 = map existence` |
| `gps_signal_strength` | Fixed `1.0` | GPS signal quality | `1 = good signal` |
| `vehicle_traffic` | 2026-06-01 mean road occupation rate from permanent counters | Traffic intensity / congestion proxy | `1 = more traffic` |
| `sidewalk_roughness` | Fixed `0.0` | Roughness / vibration proxy | `1 = rougher` |
| `slope_gradient` | Mean slope from nivellement labels and nearby sampled neighbors | Mean local slope from elevation | `1 = steeper` |
| `traffic_management` | Feux tricolores bonus minus traffic-management anomalies | Traffic-control richness / quality | `1 = better managed` |
| `zoning_laws` | Aires piétonnes / zones de rencontre + ZTL + Paris Respire | Regulated / calmed street regime | `1 = more protective regulation` |
| `bicycle_traffic` | Bike-lane adjacency + nearby Vélib stations | Bicycle activity / bike presence proxy | `1 = more bike traffic` |
| `charging_station_proximity` | Nearby Vélib stations | Nearby charging / dock access proxy | `1 = closer / denser access` |
| `surveillance_coverage` | Fixed `1.0` | Camera / monitoring coverage | `1 = full coverage` |
| `bike_lane_availability` | Adjacent piste / piste cyclable only | Adjacent protected cycling lane | `1 = more available` |

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
- `paris_test/data/processed/street_furniture_density_paris.geojson`: sample points with nearby clutter counts per source and the weighted street-furniture density score
- `paris_test/data/processed/curb_ramps_paris.geojson`: sample points with escalier and accessibility-quarter flags and the curb-ramp availability score
- `paris_test/data/processed/traffic_management_paris.geojson`: sample points with feux-tricolores counts, Dans Ma Rue anomaly counts, and the final traffic-management score
- `paris_test/data/processed/zoning_regulation_paris.geojson`: sample points with pedestrian-zone, ZTL, and Paris Respire flags and the zoning-regulation score
- `paris_test/data/processed/slope_gradient_paris.geojson`: sample points with assigned elevation, mean slope to neighbors, and the slope-gradient score (1 = steeper)
- `paris_test/data/processed/street_lighting_paris.geojson`: sample points with lamp count and the lighting score
- `paris_test/data/processed/bike_lane_paris.geojson`: sample points with bike-lane adjacency score (0 / 0.8 / 1.0)
- `paris_test/data/processed/bike_traffic_paris.geojson`: sample points with piste flag, Vélib count, and the bicycle-traffic score
- `paris_test/data/processed/charging_stations_paris.geojson`: sample points with Vélib station count and the charging score
- `paris_test/data/processed/shade_paris.geojson`: sample points with weighted tree count and the shade score
- `paris_test/data/processed/vehicle_traffic_paris.geojson`: sample points with nearest-road occupation rate and the vehicle-traffic score
- `paris_test/data/processed/sidewalk_width_paris.geojson`: sample points with raw `width_m` and the sidewalk-width score
- `paris_test/data/processed/communication_infrastructure_paris.geojson`: sample points with constant `1.0` communication-infrastructure score
- `paris_test/data/processed/digital_map_existence_paris.geojson`: sample points with constant `1.0` digital-map score
- `paris_test/data/processed/gps_signal_strength_paris.geojson`: sample points with constant `1.0` GPS score
- `paris_test/data/processed/sidewalk_roughness_paris.geojson`: sample points with constant `0.0` roughness score
- `paris_test/data/processed/surveillance_coverage_paris.geojson`: sample points with constant `1.0` surveillance score
- `paris_test/data/processed/curb_ramp_availability_paris.geojson`: canonical rename of the curb-ramp feature
- `paris_test/data/processed/zoning_laws_paris.geojson`: canonical rename of the zoning-regulation feature
- `paris_test/data/processed/charging_station_proximity_paris.geojson`: canonical rename of the charging feature
- `paris_test/data/processed/bicycle_traffic_paris.geojson`: canonical rename of the bike-traffic feature
- `paris_test/data/processed/bike_lane_availability_paris.geojson`: canonical rename of the bike-lane feature
- `paris_test/data/processed/robotability_features_paris.csv`: one row per sample point with metadata, raw feature values, normalized feature scores, per-feature contributions, and the final robotability score
- `paris_test/data/processed/robotability_score_paris.geojson`: individual-point GeoJSON FeatureCollection with the final robotability score
- `paris_test/sample_paris_street/`: NYC-style Street View sample tree with `splits/` (`{lat},{lon}_{date}_{pano}_d{heading}_z2_{side}.jpg`), `full/` / `full_with_vis/` (`{lat},{lon}_{date}_{pano}_d{heading}_z2.jpg`), plus `metadata.yaml` and `metadata_manifest.csv`
- `paris_test/figures/robotability_score_<level>_01.png` / `_quantiles.png`: robotability score maps per geographic level
- `paris_test/figures/`: output maps
