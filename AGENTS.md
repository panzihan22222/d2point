# AGENTS

## Project Scope

- This repository is a Python 3.12 workflow for 2D mask + OpenSfM pose + ODM point cloud fusion.
- For full algorithm and usage details, link to and follow [README_2D3D_PIPELINE.md](README_2D3D_PIPELINE.md) instead of duplicating long explanations.

## Primary Entry Points

- [run_2d3d_volume_pipeline.py](run_2d3d_volume_pipeline.py): offline segmentation and volume estimation pipeline.
- [manual_web_server.py](manual_web_server.py): local HTTP server for manual selection and lowest-point volume estimation.

## Environment And Commands

- Preferred environment source: [environment.yml](environment.yml) (conda, Python 3.12).
- Typical setup:

```powershell
conda env create -f .\environment.yml
conda activate d2point
```

- Fast smoke checks after code changes:

```powershell
python .\run_2d3d_volume_pipeline.py --help
python .\manual_web_server.py --help
python .\manual_web_server.py --max-points 80000 --voxel-size 0.1
```

- If `python` resolves to a different interpreter, run commands with the explicit env Python executable.

## Data And Output Layout

- [fimages/](fimages/): per-image Labelme JSON polygons.
- [masks/](masks/): generated masks.
- [odmoutput/](odmoutput/): ODM/OpenSfM outputs, including default cloud [odmoutput/odm_filterpoints/point_cloud.ply](odmoutput/odm_filterpoints/point_cloud.ply).
- [results/](results/): default output folder for reports, overlays, and exports.
- [manual_web/](manual_web/): static assets served by manual web server.

## Repo-Specific Conventions

- Keep default pipeline behavior stable unless asked: `seeded_footprint` + `lowest_point`.
- Current manual web API is box-based selection (`min`/`max` bounds) for `/api/estimate_volume` and `/api/export_selection`.
- README may describe richer polygon UI workflows; verify current code paths before changing behavior or docs.
- For heavy experiment runs, use a separate output directory (for example `results_all_modes`) to avoid overwriting baseline outputs.

## Editing Checklist For Agents

- Keep patches small and focused; avoid unrelated refactors.
- When changing server payload fields, update both backend and frontend consumers.
- After Python edits, run at least one relevant smoke command for the changed entry point.