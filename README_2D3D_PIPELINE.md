# 2D->3D Pipeline

This project fuses 2D masks, OpenSfM camera poses, and ODM point clouds to produce:

- a 3D segmented target cloud
- multiple volume estimates for the same object
- QA overlay images for projection inspection

## Run

```powershell
cd <project_root>
python .\run_2d3d_volume_pipeline.py
```

## Conda Environment

Recommended setup:

```powershell
conda create -n d2point python=3.12 -y
conda activate d2point
python -m pip install numpy opencv-python-headless pillow scipy scikit-learn
```

Or create from file:

```powershell
conda env create -f .\environment.yml
conda activate d2point
```

If you also want to use Open3D-based experiments:

```powershell
python -m pip install open3d
```

The current default configuration is:

- segmentation mode: `seeded_footprint`
- volume mode: `lowest_point`

## Outputs

- `<project_root>\results\target_segmented.ply`
- `<project_root>\results\volume_report.json`
- `<project_root>\results\qa_overlay\*.png`

## Segmentation Modes

- `seeded_footprint`
  - binary multi-view voting
  - triangulated seed from 2D polygon centroids
  - XY footprint connected component extraction
- `seeded_weighted_footprint`
  - weighted multi-view voting using mask distance scores
  - triangulated seed from 2D polygon centroids
  - useful when masks are noisy or boundaries are weak
- `largest_cluster`
  - binary multi-view voting
  - keep the largest 3D cluster
- `weighted_cluster`
  - weighted multi-view voting
  - keep the largest 3D cluster

## Volume Modes

- `ground_plane`
  - fit a local reference plane around the target
  - good for piles, stockpiles, and earthwork
- `lowest_point`
  - use the target's lowest point as the base
  - often closer to building-envelope style measurements
- `dtm_idw`
  - estimate a local base surface from surrounding ring points with IDW interpolation
  - more general than a single plane when the surrounding ground is not perfectly flat
- `voxel_columns`
  - convert footprint columns into filled voxel stacks
  - useful as a robust discrete comparison baseline
- `all`
  - compute every estimator and store them in `volume_modes`

## Example

```powershell
python .\run_2d3d_volume_pipeline.py
python .\run_2d3d_volume_pipeline.py --segmentation-mode seeded_footprint --volume-mode all --output .\results_all_modes
python .\run_2d3d_volume_pipeline.py --segmentation-mode weighted_cluster --volume-mode all --output .\results_weighted_cluster
```

## Notes

- The current sample dataset is a building dataset, so the segmentation result should follow the building footprint and roof structure.
- For this dataset, the benchmark-aligned default is `seeded_footprint + lowest_point`.
- For stockpile migration later, start by keeping `seeded_weighted_footprint` and switching the main volume interpretation to `ground_plane` or `dtm_idw`.

## Manual Web Segmentation (Polygon Selection)

This project now includes a manual workflow:

- open a local web page
- switch camera view and draw a polygon on the viewport
- estimate volume for points inside the polygon selection (lowest-point method)
- export selected points to PLY

### Start

```powershell
cd <project_root>
python .\manual_web_server.py
```

Default point display budget is now `220000` points (change via `--max-points`).

Tip for smoother loading on low-end machines:

```powershell
python .\manual_web_server.py --max-points 80000 --voxel-size 0.1
```

If your GPU/CPU is strong and you want to see more points:

```powershell
python .\manual_web_server.py --max-points 2000000 --voxel-size 0.01
```

Then open:

```text
http://127.0.0.1:8765
```

Optional arguments:

```powershell
python .\manual_web_server.py --pointcloud .\results\target_segmented.ply --port 9000 --max-points 200000 --voxel-size 0.05
```

### UI Operations

- `View Mode`: rotate/pan/zoom the camera
- `Draw Mode`: click multiple vertices to draw a polygon in current view
- `Finish Polygon`: close polygon selection (double-click / Enter / right-click also works)
- `Undo Point`: remove the last polygon vertex
- `Clear Selection`: clear selected points and redraw
- `Estimate Volume`: compute lowest-point volume for selected points
- `Export Selected Point Cloud`: save selected points to a PLY file

Parameter constraints:

- `Grid Cell (m)`: valid range is `0.01` to `5.0`
- `Base Percentile`: valid range is `0` to `100`
- decimal input supports both `0.2` and `0,2`
- report JSON now contains the effective `grid_cell_m` used in computation

Suggested workflow:

1. switch to `View Mode` and adjust the camera angle
2. switch to `Draw Mode` and click vertices around the target region
3. close the polygon with `Finish Polygon` (or double-click / Enter)
4. click `Estimate Volume` (or export the selected cloud)

Saved outputs:

- JSON reports: `results\manual_web\manual_volume_*.json`
- selected clouds: `results\manual_web\manual_selected_*.ply`

### Troubleshooting: page keeps "loading point cloud"

- Hard refresh browser (`Ctrl+F5`) to clear stale JS cache.
- Keep using local service URL: `http://127.0.0.1:8765`.
- If data is heavy, reduce frontend payload:
  - `python .\manual_web_server.py --max-points 80000 --voxel-size 0.1`
