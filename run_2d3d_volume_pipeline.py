#!/usr/bin/env python
"""Offline 2D-to-3D segmentation and volume estimation pipeline.

Implements:
1) Labelme polygon/mask ingestion
2) OpenSfM Brown projection with z-buffer visibility voting
3) 3D foreground extraction + largest-cluster purification
4) Local-ring ground plane fitting with RANSAC
5) Grid-based volume integration
6) QA overlays + JSON report + segmented point cloud export
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


@dataclass
class CameraModel:
    width: int
    height: int
    focal_x: float
    focal_y: float
    c_x: float
    c_y: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

    @property
    def scale(self) -> float:
        return float(max(self.width, self.height))

    @property
    def cx_px(self) -> float:
        return 0.5 * (self.width - 1)

    @property
    def cy_px(self) -> float:
        return 0.5 * (self.height - 1)


@dataclass
class Shot:
    name: str
    camera: str
    rotation: np.ndarray  # 3x3 world->camera
    translation: np.ndarray  # 3,


@dataclass
class ShotVoteCache:
    shot_name: str
    vis_idx: np.ndarray
    inside_mask: np.ndarray
    mask_score: Optional[np.ndarray] = None


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="2D->3D segmentation and volume pipeline")
    p.add_argument("--root", type=Path, default=root, help="Project root")
    p.add_argument("--fimages", type=Path, default=root / "fimages")
    p.add_argument("--masks", type=Path, default=root / "masks")
    p.add_argument("--images", type=Path, default=root / "odmoutput" / "images")
    p.add_argument(
        "--reconstruction",
        type=Path,
        default=root / "odmoutput" / "opensfm" / "reconstruction.topocentric.json",
    )
    p.add_argument(
        "--pointcloud",
        type=Path,
        default=root / "odmoutput" / "odm_filterpoints" / "point_cloud.ply",
    )
    p.add_argument("--output", type=Path, default=root / "results")
    p.add_argument("--voxel-size", type=float, default=0.03)
    p.add_argument("--vis-depth-tol", type=float, default=0.03)
    p.add_argument("--min-vis", type=int, default=3)
    p.add_argument("--score-thr", type=float, default=0.35)
    p.add_argument("--neg-vote", type=float, default=-0.5)
    p.add_argument("--dbscan-eps", type=float, default=0.12)
    p.add_argument("--dbscan-min", type=int, default=30)
    p.add_argument(
        "--segmentation-mode",
        type=str,
        default="seeded_footprint",
        choices=[
            "seeded_footprint",
            "largest_cluster",
            "weighted_cluster",
            "seeded_weighted_footprint",
        ],
        help="seeded_footprint aligns better with lowest-point benchmark on the current building dataset",
    )
    p.add_argument("--weighted-neg-vote", type=float, default=-0.35)
    p.add_argument("--weighted-score-thr", type=float, default=0.12)
    p.add_argument("--footprint-cell", type=float, default=0.3)
    p.add_argument("--footprint-z-percentile", type=float, default=40.0)
    p.add_argument("--footprint-min-fg-ratio", type=float, default=0.08)
    p.add_argument("--footprint-min-fg-points", type=int, default=3)
    p.add_argument("--footprint-dilate", type=int, default=1)
    p.add_argument(
        "--volume-mode",
        type=str,
        default="lowest_point",
        choices=["ground_plane", "lowest_point", "dtm_idw", "voxel_columns", "both", "all"],
        help="lowest_point is the default for current building benchmark matching",
    )
    p.add_argument("--lowest-base-percentile", type=float, default=0.0)
    p.add_argument("--ring-inner", type=float, default=0.5)
    p.add_argument("--ring-outer", type=float, default=2.0)
    p.add_argument("--plane-inlier-thr", type=float, default=0.05)
    p.add_argument("--grid-cell", type=float, default=0.05)
    p.add_argument("--dtm-k", type=int, default=8)
    p.add_argument("--dtm-power", type=float, default=2.0)
    p.add_argument("--voxel-fill-size", type=float, default=0.2)
    p.add_argument("--stability-drop-ratio", type=float, default=0.2)
    p.add_argument("--stability-trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_ply_vertices(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY: unexpected EOF in header")
            header_lines.append(line.decode("ascii", errors="ignore").strip())
            if header_lines[-1] == "end_header":
                break
        header = header_lines
        fmt = None
        vertex_count = None
        vertex_props: List[Tuple[str, str]] = []
        in_vertex = False
        for ln in header:
            parts = ln.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError("List property in vertex is not supported")
                vertex_props.append((parts[2], parts[1]))  # name, type

        if fmt is None or vertex_count is None or not vertex_props:
            raise ValueError("Invalid PLY header")

        type_map = {
            "char": "i1",
            "int8": "i1",
            "uchar": "u1",
            "uint8": "u1",
            "short": "i2",
            "int16": "i2",
            "ushort": "u2",
            "uint16": "u2",
            "int": "i4",
            "int32": "i4",
            "uint": "u4",
            "uint32": "u4",
            "float": "f4",
            "float32": "f4",
            "double": "f8",
            "float64": "f8",
        }

        if fmt == "ascii":
            data = np.loadtxt(path, skiprows=len(header), max_rows=vertex_count)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            name_to_col = {name: i for i, (name, _) in enumerate(vertex_props)}
            xyz = np.stack(
                [
                    data[:, name_to_col["x"]],
                    data[:, name_to_col["y"]],
                    data[:, name_to_col["z"]],
                ],
                axis=1,
            ).astype(np.float32)
            colors = None
            if {"red", "green", "blue"}.issubset(name_to_col):
                colors = np.stack(
                    [
                        data[:, name_to_col["red"]],
                        data[:, name_to_col["green"]],
                        data[:, name_to_col["blue"]],
                    ],
                    axis=1,
                ).astype(np.uint8)
            return xyz, colors

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {fmt}")

        dtype = np.dtype([(name, "<" + type_map[tp]) for name, tp in vertex_props])
        raw = np.fromfile(f, dtype=dtype, count=vertex_count)
        xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
        colors = None
        if {"red", "green", "blue"}.issubset(raw.dtype.names or []):
            colors = np.stack([raw["red"], raw["green"], raw["blue"]], axis=1).astype(np.uint8)
        return xyz, colors


def write_ply_vertices(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    n = points.shape[0]
    with path.open("wb") as f:
        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if colors is not None:
            header += ["property uchar red", "property uchar green", "property uchar blue"]
        header.append("end_header")
        f.write(("\n".join(header) + "\n").encode("ascii"))

        if colors is None:
            out = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            out["x"] = points[:, 0]
            out["y"] = points[:, 1]
            out["z"] = points[:, 2]
        else:
            out = np.empty(
                n,
                dtype=[
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ],
            )
            out["x"] = points[:, 0]
            out["y"] = points[:, 1]
            out["z"] = points[:, 2]
            out["red"] = colors[:, 0]
            out["green"] = colors[:, 1]
            out["blue"] = colors[:, 2]
        out.tofile(f)


def load_reconstruction(path: Path) -> Tuple[Dict[str, CameraModel], Dict[str, Shot]]:
    rec = json.loads(path.read_text(encoding="utf-8"))
    if not rec:
        raise ValueError("Empty reconstruction")
    rec0 = rec[0]
    cams: Dict[str, CameraModel] = {}
    for cname, c in rec0["cameras"].items():
        cams[cname] = CameraModel(
            width=int(c["width"]),
            height=int(c["height"]),
            focal_x=float(c["focal_x"]),
            focal_y=float(c["focal_y"]),
            c_x=float(c["c_x"]),
            c_y=float(c["c_y"]),
            k1=float(c["k1"]),
            k2=float(c["k2"]),
            p1=float(c["p1"]),
            p2=float(c["p2"]),
            k3=float(c["k3"]),
        )

    shots: Dict[str, Shot] = {}
    for sname, s in rec0["shots"].items():
        r = Rotation.from_rotvec(np.asarray(s["rotation"], dtype=np.float64)).as_matrix()
        t = np.asarray(s["translation"], dtype=np.float64)
        shots[sname] = Shot(name=sname, camera=s["camera"], rotation=r, translation=t)
    return cams, shots


def polygon_to_mask(json_path: Path, width: int, height: int) -> np.ndarray:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for shp in data.get("shapes", []):
        if shp.get("shape_type") != "polygon":
            continue
        pts = [(float(p[0]), float(p[1])) for p in shp.get("points", [])]
        if len(pts) >= 3:
            draw.polygon(pts, outline=255, fill=255)
    return np.array(mask_img, dtype=np.uint8) > 0


def load_or_build_mask(
    image_name: str,
    fimages_dir: Path,
    masks_dir: Path,
    width: int,
    height: int,
) -> np.ndarray:
    stem = Path(image_name).stem
    mask_path = masks_dir / f"{stem}_mask.png"
    if mask_path.exists():
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is not None and m.shape == (height, width):
            return m > 0
    json_path = fimages_dir / f"{stem}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing both mask and json for image: {image_name}")
    return polygon_to_mask(json_path, width=width, height=height)


def make_mask_score_image(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    inside_dt = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    outside_dt = cv2.distanceTransform((1 - mask_u8).astype(np.uint8), cv2.DIST_L2, 5)
    score = inside_dt - outside_dt
    score = score.astype(np.float32)
    max_abs = float(np.max(np.abs(score))) if score.size else 0.0
    if max_abs > 1e-6:
        score /= max_abs
    return score


def read_labelme_polygon_centroid(json_path: Path) -> np.ndarray:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    best_pts = None
    best_area = -1.0
    for shp in data.get("shapes", []):
        if shp.get("shape_type") != "polygon":
            continue
        pts = np.asarray(shp.get("points", []), dtype=np.float64)
        if pts.shape[0] < 3:
            continue
        area = abs(float(cv2.contourArea(pts.astype(np.float32))))
        if area > best_area:
            best_area = area
            best_pts = pts
    if best_pts is None:
        raise ValueError(f"No polygon found in {json_path}")
    return np.mean(best_pts, axis=0)


def pixel_to_world_ray(
    u: float,
    v: float,
    cam: CameraModel,
    shot: Shot,
) -> Tuple[np.ndarray, np.ndarray]:
    scale = cam.scale
    fx = cam.focal_x * scale
    fy = cam.focal_y * scale
    cpx = cam.c_x * scale + cam.cx_px
    cpy = cam.c_y * scale + cam.cy_px
    k = np.array([[fx, 0.0, cpx], [0.0, fy, cpy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([cam.k1, cam.k2, cam.p1, cam.p2, cam.k3], dtype=np.float64)
    uv = np.array([[[u, v]]], dtype=np.float64)
    und = cv2.undistortPoints(uv, k, dist, P=None)
    x, y = float(und[0, 0, 0]), float(und[0, 0, 1])
    dir_cam = np.array([x, y, 1.0], dtype=np.float64)
    dir_cam /= np.linalg.norm(dir_cam)

    r = shot.rotation
    t = shot.translation
    center = -(r.T @ t)
    dir_world = r.T @ dir_cam
    dir_world /= np.linalg.norm(dir_world)
    return center, dir_world


def triangulate_seed_from_masks(
    image_names: Sequence[str],
    fimages_dir: Path,
    cameras: Dict[str, CameraModel],
    shots: Dict[str, Shot],
) -> Tuple[np.ndarray, Dict[str, float]]:
    centers = []
    dirs = []
    for name in image_names:
        shot = shots.get(name)
        if shot is None:
            continue
        json_path = fimages_dir / f"{Path(name).stem}.json"
        if not json_path.exists():
            continue
        try:
            c2d = read_labelme_polygon_centroid(json_path)
            c3d, d3d = pixel_to_world_ray(float(c2d[0]), float(c2d[1]), cameras[shot.camera], shot)
        except Exception:
            continue
        centers.append(c3d)
        dirs.append(d3d)

    if len(centers) < 2:
        raise RuntimeError("Not enough valid mask centroids for seed triangulation")

    centers = np.asarray(centers, dtype=np.float64)
    dirs = np.asarray(dirs, dtype=np.float64)
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for c, d in zip(centers, dirs):
        m = eye - np.outer(d, d)
        a += m
        b += m @ c
    seed = np.linalg.solve(a, b)

    errs = []
    for c, d in zip(centers, dirs):
        v = seed - c
        perp = v - np.dot(v, d) * d
        errs.append(float(np.linalg.norm(perp)))
    stats = {
        "ray_count": float(len(errs)),
        "ray_perp_error_mean": float(np.mean(errs)),
        "ray_perp_error_max": float(np.max(errs)),
    }
    return seed.astype(np.float64), stats


def _connected_component_cells(
    uniq_cells: np.ndarray,
    active_cells: np.ndarray,
    seed_cell_idx: int,
) -> np.ndarray:
    key_to_idx = {tuple(k.tolist()): i for i, k in enumerate(uniq_cells)}
    keep = np.zeros(active_cells.shape[0], dtype=bool)
    stack = [int(seed_cell_idx)]
    keep[seed_cell_idx] = True
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while stack:
        i = stack.pop()
        x, y = uniq_cells[i]
        for dx, dy in neighbors:
            j = key_to_idx.get((int(x + dx), int(y + dy)))
            if j is None or not active_cells[j] or keep[j]:
                continue
            keep[j] = True
            stack.append(j)
    return keep


def build_seeded_footprint_mask(
    points: np.ndarray,
    fg_mask: np.ndarray,
    seed_xy: np.ndarray,
    cell: float,
    z_percentile: float,
    min_fg_ratio: float,
    min_fg_points: int,
    dilate_iters: int = 0,
) -> Tuple[np.ndarray, Dict[str, object], Dict[str, object]]:
    xy = points[:, :2]
    mins = np.min(xy, axis=0)
    ij = np.floor((xy - mins[None, :]) / cell).astype(np.int32)
    uniq, inv, cnt = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    n_cells = uniq.shape[0]

    fg_count = np.zeros(n_cells, dtype=np.int32)
    np.add.at(fg_count, inv, fg_mask.astype(np.int32))
    fg_ratio = fg_count / np.maximum(cnt, 1)

    zmax = np.full(n_cells, -np.inf, dtype=np.float32)
    np.maximum.at(zmax, inv, points[:, 2].astype(np.float32))

    active = (fg_count >= min_fg_points) & (fg_ratio >= min_fg_ratio)
    if np.any(active):
        z_thr = float(np.percentile(zmax[active], z_percentile))
        active &= zmax >= z_thr
    else:
        z_thr = float(np.percentile(zmax, z_percentile))
        active = zmax >= z_thr

    if not np.any(active):
        raise RuntimeError("No active footprint cells found")

    seed_ij = np.floor((seed_xy - mins) / cell).astype(np.int32)
    key_to_idx = {tuple(k.tolist()): i for i, k in enumerate(uniq)}
    seed_idx = key_to_idx.get((int(seed_ij[0]), int(seed_ij[1])))
    active_idx = np.nonzero(active)[0]
    if seed_idx is None or not active[seed_idx]:
        centers = (uniq[active_idx].astype(np.float64) + 0.5) * cell + mins[None, :]
        j = int(np.argmin(np.linalg.norm(centers - seed_xy[None, :], axis=1)))
        seed_idx = int(active_idx[j])

    comp = _connected_component_cells(uniq, active, seed_idx)
    if np.count_nonzero(comp) == 0:
        raise RuntimeError("Seeded connected component is empty")

    if dilate_iters > 0:
        for _ in range(dilate_iters):
            cur = np.nonzero(comp)[0]
            exp = comp.copy()
            for i in cur:
                x, y = uniq[i]
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    j = key_to_idx.get((int(x + dx), int(y + dy)))
                    if j is not None:
                        exp[j] = True
            comp = exp

    point_mask = comp[inv]
    stats: Dict[str, object] = {
        "grid_cell_m": cell,
        "grid_active_cells": int(np.count_nonzero(active)),
        "grid_component_cells": int(np.count_nonzero(comp)),
        "grid_component_area_m2": float(np.count_nonzero(comp) * cell * cell),
        "z_top_threshold": z_thr,
        "seed_xy": [float(seed_xy[0]), float(seed_xy[1])],
        "seed_cell": [int(seed_ij[0]), int(seed_ij[1])],
    }
    grid_info: Dict[str, object] = {
        "cell": float(cell),
        "mins": mins,
        "uniq": uniq,
        "inv": inv,
        "component_cells": comp,
        "zmax": zmax,
    }
    return point_mask, stats, grid_info


def validate_data_consistency(
    image_names: Sequence[str],
    fimages_dir: Path,
    masks_dir: Path,
    shots: Dict[str, Shot],
) -> Dict[str, object]:
    missing_json = []
    missing_mask = []
    missing_shot = []
    for name in image_names:
        stem = Path(name).stem
        if not (fimages_dir / f"{stem}.json").exists():
            missing_json.append(stem)
        if not (masks_dir / f"{stem}_mask.png").exists():
            missing_mask.append(stem)
        if name not in shots:
            missing_shot.append(stem)
    return {
        "n_images": len(image_names),
        "missing_json": missing_json,
        "missing_mask": missing_mask,
        "missing_shot": missing_shot,
        "passed": not (missing_json or missing_shot),
    }


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0:
        return np.arange(points.shape[0], dtype=np.int64)
    mins = points.min(axis=0)
    keys = np.floor((points - mins) / voxel).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep.sort()
    return keep


def project_brown(points_cam: np.ndarray, cam: CameraModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = points_cam[:, 0] / points_cam[:, 2]
    y = points_cam[:, 1] / points_cam[:, 2]
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + cam.k1 * r2 + cam.k2 * r4 + cam.k3 * r6
    x_tan = 2.0 * cam.p1 * x * y + cam.p2 * (r2 + 2.0 * x * x)
    y_tan = cam.p1 * (r2 + 2.0 * y * y) + 2.0 * cam.p2 * x * y
    xd = x * radial + x_tan
    yd = y * radial + y_tan
    xn = cam.focal_x * xd + cam.c_x
    yn = cam.focal_y * yd + cam.c_y
    u = xn * cam.scale + cam.cx_px
    v = yn * cam.scale + cam.cy_px
    return u, v, points_cam[:, 2]


def build_shot_vote_cache(
    points_world: np.ndarray,
    shot: Shot,
    cam: CameraModel,
    mask: np.ndarray,
    vis_depth_tol: float,
    mask_score_img: Optional[np.ndarray] = None,
) -> ShotVoteCache:
    pc = points_world @ shot.rotation.T + shot.translation[None, :]
    front = pc[:, 2] > 1e-6
    idx_front = np.nonzero(front)[0]
    if idx_front.size == 0:
        return ShotVoteCache(shot_name=shot.name, vis_idx=np.empty(0, np.int32), inside_mask=np.empty(0, bool))

    u, v, z = project_brown(pc[front], cam)
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
    finite &= (np.abs(u) < 1e7) & (np.abs(v) < 1e7)
    if not np.any(finite):
        return ShotVoteCache(shot_name=shot.name, vis_idx=np.empty(0, np.int32), inside_mask=np.empty(0, bool))
    idx_front = idx_front[finite]
    u = u[finite]
    v = v[finite]
    z = z[finite]
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < cam.width) & (vi >= 0) & (vi < cam.height)
    if not np.any(in_bounds):
        return ShotVoteCache(shot_name=shot.name, vis_idx=np.empty(0, np.int32), inside_mask=np.empty(0, bool))

    idx = idx_front[in_bounds]
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    z = z[in_bounds]

    pix = vi.astype(np.int64) * cam.width + ui.astype(np.int64)
    zbuf = np.full(cam.width * cam.height, np.inf, dtype=np.float32)
    np.minimum.at(zbuf, pix, z.astype(np.float32))
    visible = z <= (zbuf[pix] + vis_depth_tol)
    if not np.any(visible):
        return ShotVoteCache(shot_name=shot.name, vis_idx=np.empty(0, np.int32), inside_mask=np.empty(0, bool))

    idx = idx[visible]
    ui = ui[visible]
    vi = vi[visible]
    inside = mask[vi, ui]
    mask_score = None
    if mask_score_img is not None:
        mask_score = mask_score_img[vi, ui].astype(np.float32)
    return ShotVoteCache(
        shot_name=shot.name,
        vis_idx=idx.astype(np.int32),
        inside_mask=inside.astype(bool),
        mask_score=mask_score,
    )


def vote_points(
    n_points: int,
    shot_caches: Sequence[ShotVoteCache],
    selected_shots: Optional[set] = None,
    neg_vote: float = -0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    score = np.zeros(n_points, dtype=np.float32)
    n_vis = np.zeros(n_points, dtype=np.int16)
    for cache in shot_caches:
        if selected_shots is not None and cache.shot_name not in selected_shots:
            continue
        idx = cache.vis_idx
        if idx.size == 0:
            continue
        n_vis[idx] += 1
        inc = np.full(idx.shape[0], neg_vote, dtype=np.float32)
        inc[cache.inside_mask] = 1.0
        score[idx] += inc
    return score, n_vis


def vote_points_weighted(
    n_points: int,
    shot_caches: Sequence[ShotVoteCache],
    selected_shots: Optional[set] = None,
    neg_vote: float = -0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    score = np.zeros(n_points, dtype=np.float32)
    n_vis = np.zeros(n_points, dtype=np.int16)
    for cache in shot_caches:
        if selected_shots is not None and cache.shot_name not in selected_shots:
            continue
        idx = cache.vis_idx
        if idx.size == 0:
            continue
        n_vis[idx] += 1
        if cache.mask_score is None:
            inc = np.full(idx.shape[0], neg_vote, dtype=np.float32)
            inc[cache.inside_mask] = 1.0
        else:
            inc = cache.mask_score.copy()
            inc[~cache.inside_mask] = np.minimum(inc[~cache.inside_mask], neg_vote)
        score[idx] += inc
    return score, n_vis


def largest_component_by_voxel(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    mins = points.min(axis=0)
    vox = np.floor((points - mins) / voxel_size).astype(np.int32)
    uniq, inv, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    key_to_idx = {tuple(k.tolist()): i for i, k in enumerate(uniq)}

    visited = np.zeros(uniq.shape[0], dtype=bool)
    keep_cell = np.zeros(uniq.shape[0], dtype=bool)
    best_count = -1
    neighbors = np.array(
        [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=np.int32,
    )

    for start in range(uniq.shape[0]):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp_cells = []
        comp_count = 0
        while stack:
            i = stack.pop()
            comp_cells.append(i)
            comp_count += int(counts[i])
            c = uniq[i]
            for nb in neighbors:
                key = (int(c[0] + nb[0]), int(c[1] + nb[1]), int(c[2] + nb[2]))
                j = key_to_idx.get(key)
                if j is not None and not visited[j]:
                    visited[j] = True
                    stack.append(j)
        if comp_count > best_count:
            best_count = comp_count
            keep_cell[:] = False
            keep_cell[comp_cells] = True

    return keep_cell[inv]


def extract_largest_cluster(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    def _run_dbscan(arr: np.ndarray, e: float, m: int) -> np.ndarray:
        labels = DBSCAN(eps=e, min_samples=m, n_jobs=1).fit_predict(arr)
        valid = labels >= 0
        if not np.any(valid):
            return np.zeros(arr.shape[0], dtype=bool)
        uniq, cnt = np.unique(labels[valid], return_counts=True)
        keep_label = uniq[np.argmax(cnt)]
        return labels == keep_label

    keep = _run_dbscan(points, eps, min_samples)
    if np.count_nonzero(keep) >= min_samples:
        return keep

    keep = _run_dbscan(points, eps * 1.8, max(8, min_samples // 2))
    if np.count_nonzero(keep) >= min_samples:
        return keep

    keep = _run_dbscan(points[:, :2], eps * 2.0, max(8, min_samples // 2))
    if np.count_nonzero(keep) >= min_samples:
        return keep

    voxel_keep = largest_component_by_voxel(points, voxel_size=max(0.18, eps * 1.8))
    return voxel_keep


def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float]:
    ctr = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - ctr[None, :], full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    d = -float(n @ ctr)
    return n, d


def ransac_plane(
    points: np.ndarray,
    inlier_thr: float,
    iterations: int = 600,
    seed: int = 42,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points for plane fitting")

    rng = random.Random(seed)
    best_inliers = None
    best_count = -1
    best_rmse = float("inf")

    n_pts = points.shape[0]
    indices = list(range(n_pts))
    for _ in range(iterations):
        i1, i2, i3 = rng.sample(indices, 3)
        p1, p2, p3 = points[i1], points[i2], points[i3]
        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n = n / norm
        d = -float(n @ p1)
        dist = np.abs(points @ n + d)
        inliers = dist < inlier_thr
        cnt = int(np.count_nonzero(inliers))
        if cnt < 3:
            continue
        rmse = float(np.sqrt(np.mean(dist[inliers] ** 2)))
        if cnt > best_count or (cnt == best_count and rmse < best_rmse):
            best_count = cnt
            best_rmse = rmse
            best_inliers = inliers

    if best_inliers is None:
        raise RuntimeError("RANSAC failed to find a valid plane")

    n, d = fit_plane_svd(points[best_inliers])
    dist = np.abs(points @ n + d)
    inliers = dist < inlier_thr
    rmse = float(np.sqrt(np.mean(dist[inliers] ** 2)))
    if n[2] < 0:
        n = -n
        d = -d
    return n, d, inliers, rmse


def compute_volume_grid(points: np.ndarray, normal: np.ndarray, d: float, cell: float) -> Tuple[float, float]:
    h = points @ normal + d
    if np.median(h) < 0:
        normal = -normal
        d = -d
        h = points @ normal + d
    h_pos = np.maximum(h, 0.0)

    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(axis, normal)) > 0.9:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(normal, axis)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    proj = points - h[:, None] * normal[None, :]
    cu = proj @ u
    cv = proj @ v
    u0 = float(np.min(cu))
    v0 = float(np.min(cv))
    iu = np.floor((cu - u0) / cell).astype(np.int32)
    iv = np.floor((cv - v0) / cell).astype(np.int32)
    cells = np.stack([iu, iv], axis=1)
    _, inv = np.unique(cells, axis=0, return_inverse=True)
    max_h = np.zeros(int(inv.max()) + 1, dtype=np.float32)
    np.maximum.at(max_h, inv, h_pos.astype(np.float32))
    vol = float(np.sum(max_h) * (cell * cell))
    max_hv = float(np.max(h_pos)) if h_pos.size else 0.0
    return vol, max_hv


def estimate_ground_and_volume(
    all_points: np.ndarray,
    target_mask: np.ndarray,
    ring_inner: float,
    ring_outer: float,
    plane_thr: float,
    grid_cell: float,
    seed: int,
) -> Dict[str, object]:
    target = all_points[target_mask]
    if target.shape[0] < 30:
        raise RuntimeError("Too few target points for volume estimation")

    bg = all_points[~target_mask]
    if bg.shape[0] < 30:
        raise RuntimeError("Too few background points for ground fitting")

    tree = cKDTree(target[:, :2])
    dist, _ = tree.query(bg[:, :2], k=1)
    ring = (dist >= ring_inner) & (dist <= ring_outer)
    ground_candidates = bg[ring]

    # Fallback if ring area is too sparse.
    if ground_candidates.shape[0] < 200:
        ring = (dist >= max(0.2, ring_inner * 0.4)) & (dist <= ring_outer * 1.8)
        ground_candidates = bg[ring]
    if ground_candidates.shape[0] < 30:
        raise RuntimeError("Not enough ground candidates around target")

    n, d, inliers, rmse = ransac_plane(
        ground_candidates,
        inlier_thr=plane_thr,
        iterations=700,
        seed=seed,
    )
    inlier_ratio = float(np.count_nonzero(inliers) / max(1, inliers.size))
    confidence = max(0.0, min(1.0, inlier_ratio * (1.0 - rmse / 0.08)))

    volume_m3, max_height_m = compute_volume_grid(target, n, d, cell=grid_cell)
    return {
        "volume_m3": volume_m3,
        "max_height_above_ground_m": max_height_m,
        "ground_plane": {"normal": n.tolist(), "d": float(d)},
        "ground_inlier_ratio": inlier_ratio,
        "ground_rmse_m": rmse,
        "ground_confidence": confidence,
        "ground_candidate_count": int(ground_candidates.shape[0]),
    }


def estimate_lowest_point_volume(
    all_points: np.ndarray,
    footprint_mask: np.ndarray,
    grid_cell: float,
    base_percentile: float = 0.0,
) -> Dict[str, object]:
    obj = all_points[footprint_mask]
    if obj.shape[0] < 30:
        raise RuntimeError("Too few object points for lowest-point volume")

    xy = obj[:, :2]
    z = obj[:, 2]
    mins = np.min(xy, axis=0)
    ij = np.floor((xy - mins[None, :]) / grid_cell).astype(np.int32)
    uniq, inv = np.unique(ij, axis=0, return_inverse=True)
    top = np.full(uniq.shape[0], -np.inf, dtype=np.float32)
    np.maximum.at(top, inv, z.astype(np.float32))

    pct = float(np.clip(base_percentile, 0.0, 100.0))
    base_z = float(np.percentile(z, pct))
    h = np.maximum(top - base_z, 0.0)
    vol = float(np.sum(h) * (grid_cell * grid_cell))
    area = float(uniq.shape[0] * grid_cell * grid_cell)

    return {
        "volume_m3": vol,
        "base_plane": {"type": "lowest_point", "z": base_z, "percentile": pct},
        "footprint_area_m2": area,
        "cell_count": int(uniq.shape[0]),
        "mean_height_m": float(np.mean(h)) if h.size else 0.0,
        "max_height_m": float(np.max(h)) if h.size else 0.0,
    }


def estimate_lowest_point_volume_from_component_grid(
    all_points: np.ndarray,
    grid_info: Dict[str, object],
    base_percentile: float = 0.0,
) -> Dict[str, object]:
    comp = grid_info["component_cells"]
    inv = grid_info["inv"]
    zmax = grid_info["zmax"]
    cell = float(grid_info["cell"])

    footprint_point_mask = comp[inv]
    obj_z = all_points[footprint_point_mask, 2]
    if obj_z.size < 30:
        raise RuntimeError("Too few points in footprint component for lowest-point volume")

    pct = float(np.clip(base_percentile, 0.0, 100.0))
    base_z = float(np.percentile(obj_z, pct))
    top = zmax[comp].astype(np.float64)
    h = np.maximum(top - base_z, 0.0)
    vol = float(np.sum(h) * (cell * cell))
    area = float(np.count_nonzero(comp) * cell * cell)

    return {
        "volume_m3": vol,
        "base_plane": {"type": "lowest_point", "z": base_z, "percentile": pct},
        "footprint_area_m2": area,
        "cell_count": int(np.count_nonzero(comp)),
        "mean_height_m": float(np.mean(h)) if h.size else 0.0,
        "max_height_m": float(np.max(h)) if h.size else 0.0,
        "integration_grid_cell_m": cell,
    }


def estimate_dtm_idw_volume(
    all_points: np.ndarray,
    target_mask: np.ndarray,
    grid_cell: float,
    ring_inner: float,
    ring_outer: float,
    k: int = 8,
    power: float = 2.0,
) -> Dict[str, object]:
    target = all_points[target_mask]
    bg = all_points[~target_mask]
    if target.shape[0] < 30:
        raise RuntimeError("Too few target points for DTM/IDW volume")
    if bg.shape[0] < 30:
        raise RuntimeError("Too few background points for DTM/IDW volume")

    target_tree = cKDTree(target[:, :2])
    dist_bg, _ = target_tree.query(bg[:, :2], k=1)
    ring = (dist_bg >= ring_inner) & (dist_bg <= ring_outer)
    ground = bg[ring]
    if ground.shape[0] < max(50, k):
        ring = (dist_bg >= max(0.2, ring_inner * 0.5)) & (dist_bg <= ring_outer * 2.0)
        ground = bg[ring]
    if ground.shape[0] < max(20, k):
        raise RuntimeError("Not enough ring ground points for DTM/IDW volume")

    xy = target[:, :2]
    z = target[:, 2]
    mins = np.min(xy, axis=0)
    ij = np.floor((xy - mins[None, :]) / grid_cell).astype(np.int32)
    uniq, inv = np.unique(ij, axis=0, return_inverse=True)
    top = np.full(uniq.shape[0], -np.inf, dtype=np.float32)
    np.maximum.at(top, inv, z.astype(np.float32))
    centers = (uniq.astype(np.float64) + 0.5) * grid_cell + mins[None, :]

    gtree = cKDTree(ground[:, :2])
    kk = min(max(1, int(k)), ground.shape[0])
    dist, nn_idx = gtree.query(centers, k=kk)
    if kk == 1:
        dist = dist[:, None]
        nn_idx = nn_idx[:, None]
    dist = np.maximum(dist.astype(np.float64), 1e-4)
    weights = 1.0 / np.power(dist, max(power, 1e-3))
    base = np.sum(weights * ground[nn_idx, 2], axis=1) / np.sum(weights, axis=1)
    h = np.maximum(top.astype(np.float64) - base, 0.0)
    vol = float(np.sum(h) * (grid_cell * grid_cell))
    return {
        "volume_m3": vol,
        "grid_cell_m": grid_cell,
        "cell_count": int(uniq.shape[0]),
        "ground_point_count": int(ground.shape[0]),
        "ring_inner_m": float(ring_inner),
        "ring_outer_m": float(ring_outer),
        "idw_k": int(kk),
        "idw_power": float(power),
        "mean_height_m": float(np.mean(h)) if h.size else 0.0,
        "max_height_m": float(np.max(h)) if h.size else 0.0,
    }


def estimate_voxel_column_volume(
    all_points: np.ndarray,
    footprint_mask: np.ndarray,
    xy_cell: float,
    z_cell: float,
    base_percentile: float = 0.0,
) -> Dict[str, object]:
    obj = all_points[footprint_mask]
    if obj.shape[0] < 30:
        raise RuntimeError("Too few object points for voxel-column volume")
    xy = obj[:, :2]
    z = obj[:, 2]
    mins = np.min(xy, axis=0)
    ij = np.floor((xy - mins[None, :]) / xy_cell).astype(np.int32)
    uniq, inv = np.unique(ij, axis=0, return_inverse=True)
    top = np.full(uniq.shape[0], -np.inf, dtype=np.float32)
    np.maximum.at(top, inv, z.astype(np.float32))
    base_z = float(np.percentile(z, float(np.clip(base_percentile, 0.0, 100.0))))
    filled_layers = np.maximum(np.ceil((top.astype(np.float64) - base_z) / z_cell), 0.0)
    vol = float(np.sum(filled_layers) * xy_cell * xy_cell * z_cell)
    return {
        "volume_m3": vol,
        "xy_cell_m": float(xy_cell),
        "z_cell_m": float(z_cell),
        "column_count": int(uniq.shape[0]),
        "base_z": base_z,
        "base_percentile": float(np.clip(base_percentile, 0.0, 100.0)),
        "mean_filled_layers": float(np.mean(filled_layers)) if filled_layers.size else 0.0,
        "max_filled_layers": float(np.max(filled_layers)) if filled_layers.size else 0.0,
    }


def render_qa_overlays(
    out_dir: Path,
    fimages_dir: Path,
    shots: Dict[str, Shot],
    cameras: Dict[str, CameraModel],
    masks: Dict[str, np.ndarray],
    target_points: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kernel = np.ones((3, 3), np.uint8)

    for shot_name, shot in shots.items():
        img_path = fimages_dir / shot_name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cam = cameras[shot.camera]
        mask = masks[shot_name]
        edge = cv2.Canny((mask.astype(np.uint8) * 255), 80, 160)
        img[edge > 0] = (0, 0, 255)

        if target_points.size > 0:
            pc = target_points @ shot.rotation.T + shot.translation[None, :]
            front = pc[:, 2] > 1e-6
            if np.any(front):
                u, v, _ = project_brown(pc[front], cam)
                finite = np.isfinite(u) & np.isfinite(v)
                finite &= (np.abs(u) < 1e7) & (np.abs(v) < 1e7)
                u = u[finite]
                v = v[finite]
                ui = np.rint(u).astype(np.int32)
                vi = np.rint(v).astype(np.int32)
                in_bounds = (ui >= 0) & (ui < cam.width) & (vi >= 0) & (vi < cam.height)
                ui = ui[in_bounds]
                vi = vi[in_bounds]
                if ui.size > 0:
                    pts = np.zeros((cam.height, cam.width), np.uint8)
                    pts[vi, ui] = 255
                    pts = cv2.dilate(pts, kernel, iterations=1)
                    img[pts > 0] = (0, 255, 0)

        out_path = out_dir / f"{Path(shot_name).stem}_qa.png"
        cv2.imwrite(str(out_path), img)


def compute_projection_quality(
    caches: Sequence[ShotVoteCache],
    final_target_mask: np.ndarray,
) -> Dict[str, object]:
    inside = 0
    total = 0
    for c in caches:
        if c.vis_idx.size == 0:
            continue
        is_target = final_target_mask[c.vis_idx]
        if not np.any(is_target):
            continue
        total += int(np.count_nonzero(is_target))
        inside += int(np.count_nonzero(c.inside_mask[is_target]))
    ratio = float(inside / total) if total > 0 else 0.0
    return {"foreground_inside_mask_ratio": ratio, "foreground_visible_points": int(total)}


def run_subset_volume(
    points: np.ndarray,
    caches: Sequence[ShotVoteCache],
    selected_shots: set,
    min_vis: int,
    score_thr: float,
    neg_vote: float,
    dbscan_eps: float,
    dbscan_min: int,
    ring_inner: float,
    ring_outer: float,
    plane_thr: float,
    grid_cell: float,
    seed: int,
) -> Optional[float]:
    score, n_vis = vote_points(points.shape[0], caches, selected_shots=selected_shots, neg_vote=neg_vote)
    ratio = np.divide(score, np.maximum(n_vis, 1), dtype=np.float32)
    fg = (n_vis >= min_vis) & (ratio >= score_thr)
    if np.count_nonzero(fg) < dbscan_min:
        return None
    fg_pts = points[fg]
    keep_fg = extract_largest_cluster(fg_pts, eps=dbscan_eps, min_samples=dbscan_min)
    if np.count_nonzero(keep_fg) < dbscan_min:
        return None
    final_mask = np.zeros(points.shape[0], dtype=bool)
    final_mask[np.nonzero(fg)[0][keep_fg]] = True
    try:
        vol = estimate_ground_and_volume(
            all_points=points,
            target_mask=final_mask,
            ring_inner=ring_inner,
            ring_outer=ring_outer,
            plane_thr=plane_thr,
            grid_cell=grid_cell,
            seed=seed,
        )["volume_m3"]
    except Exception:
        return None
    return float(vol)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = args.output
    qa_dir = out_dir / "qa_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("Loading reconstruction...")
    cameras, shots = load_reconstruction(args.reconstruction)
    image_names = sorted([p.name for p in args.images.glob("*.JPG")])
    if not image_names:
        raise RuntimeError(f"No JPG images found in {args.images}")

    consistency = validate_data_consistency(image_names, args.fimages, args.masks, shots)
    if not consistency["passed"]:
        raise RuntimeError(
            f"Data consistency failed: missing_json={consistency['missing_json']}, "
            f"missing_shot={consistency['missing_shot']}"
        )
    log(f"Data consistency passed with {consistency['n_images']} images.")

    log("Loading point cloud...")
    points_all, colors_all = read_ply_vertices(args.pointcloud)
    log(f"Loaded point cloud: {points_all.shape[0]} points.")

    keep_idx = voxel_downsample(points_all, args.voxel_size)
    points = points_all[keep_idx]
    colors = colors_all[keep_idx] if colors_all is not None else None
    log(f"Voxel downsample complete: {points.shape[0]} points kept (voxel={args.voxel_size}m).")

    log("Preparing masks and shot caches...")
    masks: Dict[str, np.ndarray] = {}
    mask_scores: Dict[str, np.ndarray] = {}
    shot_caches: List[ShotVoteCache] = []

    for i, name in enumerate(image_names, start=1):
        shot = shots.get(name)
        if shot is None:
            continue
        cam = cameras[shot.camera]
        mask = load_or_build_mask(name, args.fimages, args.masks, cam.width, cam.height)
        mask_score_img = make_mask_score_image(mask)
        masks[name] = mask
        mask_scores[name] = mask_score_img
        cache = build_shot_vote_cache(
            points_world=points,
            shot=shot,
            cam=cam,
            mask=mask,
            vis_depth_tol=args.vis_depth_tol,
            mask_score_img=mask_score_img,
        )
        shot_caches.append(cache)
        log(f"  [{i}/{len(image_names)}] cached {name}: visible={cache.vis_idx.size}")

    if not shot_caches:
        raise RuntimeError("No valid shot cache generated")

    use_weighted_votes = args.segmentation_mode in ("weighted_cluster", "seeded_weighted_footprint")
    vote_name = "weighted" if use_weighted_votes else "binary"
    log(f"Voting 2D->3D labels ({vote_name})...")
    if use_weighted_votes:
        score, n_vis = vote_points_weighted(
            points.shape[0], shot_caches, selected_shots=None, neg_vote=args.weighted_neg_vote
        )
        ratio = np.divide(score, np.maximum(n_vis, 1), dtype=np.float32)
        fg_mask = (n_vis >= args.min_vis) & (ratio >= args.weighted_score_thr)
    else:
        score, n_vis = vote_points(points.shape[0], shot_caches, selected_shots=None, neg_vote=args.neg_vote)
        ratio = np.divide(score, np.maximum(n_vis, 1), dtype=np.float32)
        fg_mask = (n_vis >= args.min_vis) & (ratio >= args.score_thr)
    log(f"Foreground candidate points: {int(np.count_nonzero(fg_mask))}")
    if np.count_nonzero(fg_mask) < args.dbscan_min:
        raise RuntimeError("Too few foreground candidates after voting.")

    seed_info: Dict[str, object] = {}
    footprint_info: Dict[str, object] = {}
    footprint_mask = np.zeros(points.shape[0], dtype=bool)
    footprint_grid: Optional[Dict[str, object]] = None

    if args.segmentation_mode in ("largest_cluster", "weighted_cluster"):
        log(f"Segmentation mode: {args.segmentation_mode}")
        fg_points = points[fg_mask]
        keep_local = extract_largest_cluster(fg_points, eps=args.dbscan_eps, min_samples=args.dbscan_min)
        if np.count_nonzero(keep_local) < args.dbscan_min:
            raise RuntimeError("No stable foreground cluster found.")
        target_mask = np.zeros(points.shape[0], dtype=bool)
        target_mask[np.nonzero(fg_mask)[0][keep_local]] = True
        footprint_mask = target_mask.copy()
    else:
        log(f"Segmentation mode: {args.segmentation_mode}")
        seed_pt, seed_stats = triangulate_seed_from_masks(
            image_names=image_names,
            fimages_dir=args.fimages,
            cameras=cameras,
            shots=shots,
        )
        seed_info = {"seed_xyz": seed_pt.tolist(), **seed_stats}
        log(
            "  triangulated seed "
            f"xyz=({seed_pt[0]:.3f},{seed_pt[1]:.3f},{seed_pt[2]:.3f}), "
            f"ray_err_mean={seed_stats['ray_perp_error_mean']:.3f}m"
        )
        footprint_mask, footprint_info, footprint_grid = build_seeded_footprint_mask(
            points=points,
            fg_mask=fg_mask,
            seed_xy=seed_pt[:2],
            cell=args.footprint_cell,
            z_percentile=args.footprint_z_percentile,
            min_fg_ratio=args.footprint_min_fg_ratio,
            min_fg_points=args.footprint_min_fg_points,
            dilate_iters=max(0, int(args.footprint_dilate)),
        )
        target_mask = footprint_mask & fg_mask
        if np.count_nonzero(target_mask) < args.dbscan_min:
            # Fallback to full footprint if vote overlap is too sparse.
            target_mask = footprint_mask.copy()
            footprint_info["fallback_used"] = "use_full_footprint_points"

        # Optional cleanup: keep largest connected cluster inside current target.
        cur_points = points[target_mask]
        keep_local = extract_largest_cluster(cur_points, eps=args.dbscan_eps, min_samples=max(8, args.dbscan_min // 2))
        keep_n = int(np.count_nonzero(keep_local))
        if keep_n >= args.dbscan_min and keep_n >= int(0.2 * max(1, cur_points.shape[0])):
            tmp = np.zeros(points.shape[0], dtype=bool)
            tmp[np.nonzero(target_mask)[0][keep_local]] = True
            target_mask = tmp

    target_points = points[target_mask]
    target_colors = colors[target_mask] if colors is not None else None
    log(f"Target points after segmentation: {target_points.shape[0]}")
    if target_points.shape[0] < args.dbscan_min:
        raise RuntimeError("Too few points after segmentation.")

    volume_modes: Dict[str, Dict[str, object]] = {}
    if args.volume_mode in ("ground_plane", "both", "all"):
        log("Estimating ground-plane volume...")
        try:
            volume_modes["ground_plane"] = estimate_ground_and_volume(
                all_points=points,
                target_mask=target_mask,
                ring_inner=args.ring_inner,
                ring_outer=args.ring_outer,
                plane_thr=args.plane_inlier_thr,
                grid_cell=args.grid_cell,
                seed=args.seed,
            )
        except Exception as e:
            volume_modes["ground_plane"] = {"error": str(e)}

    if args.volume_mode in ("lowest_point", "both", "all"):
        log("Estimating lowest-point baseline volume...")
        lp_mask = footprint_mask if np.count_nonzero(footprint_mask) > 0 else target_mask
        try:
            if footprint_grid is not None:
                volume_modes["lowest_point"] = estimate_lowest_point_volume_from_component_grid(
                    all_points=points,
                    grid_info=footprint_grid,
                    base_percentile=args.lowest_base_percentile,
                )
            else:
                volume_modes["lowest_point"] = estimate_lowest_point_volume(
                    all_points=points,
                    footprint_mask=lp_mask,
                    grid_cell=args.grid_cell,
                    base_percentile=args.lowest_base_percentile,
                )
        except Exception as e:
            volume_modes["lowest_point"] = {"error": str(e)}

    if args.volume_mode in ("dtm_idw", "all"):
        log("Estimating DTM/IDW volume...")
        try:
            volume_modes["dtm_idw"] = estimate_dtm_idw_volume(
                all_points=points,
                target_mask=target_mask,
                grid_cell=args.grid_cell,
                ring_inner=args.ring_inner,
                ring_outer=args.ring_outer,
                k=args.dtm_k,
                power=args.dtm_power,
            )
        except Exception as e:
            volume_modes["dtm_idw"] = {"error": str(e)}

    if args.volume_mode in ("voxel_columns", "all"):
        log("Estimating voxel-column volume...")
        try:
            vf_mask = footprint_mask if np.count_nonzero(footprint_mask) > 0 else target_mask
            volume_modes["voxel_columns"] = estimate_voxel_column_volume(
                all_points=points,
                footprint_mask=vf_mask,
                xy_cell=max(args.grid_cell, args.voxel_fill_size),
                z_cell=args.voxel_fill_size,
                base_percentile=args.lowest_base_percentile,
            )
        except Exception as e:
            volume_modes["voxel_columns"] = {"error": str(e)}

    if args.volume_mode == "ground_plane":
        volume_info = volume_modes.get("ground_plane", {"error": "ground_plane not computed"})
    elif args.volume_mode == "lowest_point":
        volume_info = volume_modes.get("lowest_point", {"error": "lowest_point not computed"})
    elif args.volume_mode == "dtm_idw":
        volume_info = volume_modes.get("dtm_idw", {"error": "dtm_idw not computed"})
    elif args.volume_mode == "voxel_columns":
        volume_info = volume_modes.get("voxel_columns", {"error": "voxel_columns not computed"})
    else:
        # Prefer IDW or lowest-point for a robust default summary, then fall back.
        if "dtm_idw" in volume_modes and "error" not in volume_modes["dtm_idw"]:
            volume_info = volume_modes["dtm_idw"]
        elif "lowest_point" in volume_modes and "error" not in volume_modes["lowest_point"]:
            volume_info = volume_modes["lowest_point"]
        elif "ground_plane" in volume_modes and "error" not in volume_modes["ground_plane"]:
            volume_info = volume_modes["ground_plane"]
        elif "voxel_columns" in volume_modes and "error" not in volume_modes["voxel_columns"]:
            volume_info = volume_modes["voxel_columns"]
        else:
            volume_info = {"error": "no valid volume mode result"}

    bbox_min = target_points.min(axis=0)
    bbox_max = target_points.max(axis=0)
    center = target_points.mean(axis=0)
    peak = target_points[np.argmax(target_points[:, 2])]

    log("Running projection quality check...")
    projection_quality = compute_projection_quality(shot_caches, target_mask)

    quality_flags: List[str] = []
    if projection_quality["foreground_inside_mask_ratio"] < 0.7:
        quality_flags.append("low_projection_alignment")
    gp = volume_modes.get("ground_plane")
    if isinstance(gp, dict) and "ground_rmse_m" in gp and gp["ground_rmse_m"] > 0.08:
        quality_flags.append("low_ground_confidence")
    if isinstance(gp, dict) and "error" in gp:
        quality_flags.append("ground_mode_failed")
    lp = volume_modes.get("lowest_point")
    if isinstance(lp, dict) and "error" in lp:
        quality_flags.append("lowest_mode_failed")
    dv = volume_modes.get("dtm_idw")
    if isinstance(dv, dict) and "error" in dv:
        quality_flags.append("dtm_mode_failed")
    vv = volume_modes.get("voxel_columns")
    if isinstance(vv, dict) and "error" in vv:
        quality_flags.append("voxel_mode_failed")

    stability: Dict[str, object]
    if args.segmentation_mode == "largest_cluster" and args.volume_mode in ("ground_plane",):
        log("Running stability test (drop 20% views)...")
        shot_names = [c.shot_name for c in shot_caches]
        n_drop = max(1, int(round(len(shot_names) * args.stability_drop_ratio)))
        stability_vols = []
        for t in range(args.stability_trials):
            rng = random.Random(args.seed + 101 + t)
            drop = set(rng.sample(shot_names, n_drop))
            keep = set(shot_names) - drop
            vol = run_subset_volume(
                points=points,
                caches=shot_caches,
                selected_shots=keep,
                min_vis=args.min_vis,
                score_thr=args.score_thr,
                neg_vote=args.neg_vote,
                dbscan_eps=args.dbscan_eps,
                dbscan_min=args.dbscan_min,
                ring_inner=args.ring_inner,
                ring_outer=args.ring_outer,
                plane_thr=args.plane_inlier_thr,
                grid_cell=args.grid_cell,
                seed=args.seed + t + 1,
            )
            if vol is not None:
                stability_vols.append(vol)

        base_vol = float(volume_info.get("volume_m3", 0.0)) if isinstance(volume_info, dict) else 0.0
        stability = {
            "trials": args.stability_trials,
            "successful_trials": len(stability_vols),
            "volumes_m3": stability_vols,
        }
        if stability_vols and base_vol > 0:
            max_rel = max(abs(v - base_vol) / max(base_vol, 1e-6) for v in stability_vols)
            stability["max_rel_change"] = float(max_rel)
            if max_rel > 0.08:
                quality_flags.append("stability_variation_high")
        else:
            stability["max_rel_change"] = None
            quality_flags.append("stability_test_failed")
    else:
        stability = {
            "skipped": True,
            "reason": "stability test currently enabled for segmentation=largest_cluster and volume_mode=ground_plane",
        }

    log("Writing outputs...")
    write_ply_vertices(out_dir / "target_segmented.ply", target_points, target_colors)
    render_qa_overlays(
        out_dir=qa_dir,
        fimages_dir=args.fimages,
        shots={k: shots[k] for k in image_names if k in shots},
        cameras=cameras,
        masks=masks,
        target_points=target_points,
    )

    report = {
        "input": {
            "fimages": str(args.fimages),
            "masks": str(args.masks),
            "images": str(args.images),
            "reconstruction": str(args.reconstruction),
            "pointcloud": str(args.pointcloud),
        },
        "params": {
            "voxel_size": args.voxel_size,
            "vis_depth_tol": args.vis_depth_tol,
            "min_vis": args.min_vis,
            "score_thr": args.score_thr,
            "neg_vote": args.neg_vote,
            "weighted_neg_vote": args.weighted_neg_vote,
            "weighted_score_thr": args.weighted_score_thr,
            "dbscan_eps": args.dbscan_eps,
            "dbscan_min": args.dbscan_min,
            "segmentation_mode": args.segmentation_mode,
            "footprint_cell": args.footprint_cell,
            "footprint_z_percentile": args.footprint_z_percentile,
            "footprint_min_fg_ratio": args.footprint_min_fg_ratio,
            "footprint_min_fg_points": args.footprint_min_fg_points,
            "footprint_dilate": args.footprint_dilate,
            "volume_mode": args.volume_mode,
            "lowest_base_percentile": args.lowest_base_percentile,
            "ring_inner": args.ring_inner,
            "ring_outer": args.ring_outer,
            "plane_inlier_thr": args.plane_inlier_thr,
            "grid_cell": args.grid_cell,
            "dtm_k": args.dtm_k,
            "dtm_power": args.dtm_power,
            "voxel_fill_size": args.voxel_fill_size,
            "stability_drop_ratio": args.stability_drop_ratio,
            "stability_trials": args.stability_trials,
            "seed": args.seed,
        },
        "checks": {
            "data_consistency": consistency,
            "projection_quality": projection_quality,
            "stability": stability,
            "seed_info": seed_info,
            "footprint_info": footprint_info,
        },
        "segmentation": {
            "point_count": int(target_points.shape[0]),
            "center_xyz": center.tolist(),
            "bbox_min_xyz": bbox_min.tolist(),
            "bbox_max_xyz": bbox_max.tolist(),
            "peak_xyz": peak.tolist(),
            "footprint_point_count": int(np.count_nonzero(footprint_mask)),
            "foreground_candidate_count": int(np.count_nonzero(fg_mask)),
            "vote_mode": vote_name,
            "vote_score_mean": float(np.mean(ratio[fg_mask])) if np.any(fg_mask) else 0.0,
        },
        "volume": volume_info,
        "volume_modes": volume_modes,
        "quality_flags": quality_flags,
    }
    (out_dir / "volume_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if isinstance(volume_info, dict) and "volume_m3" in volume_info:
        log(f"Done. volume={float(volume_info['volume_m3']):.3f} m^3, flags={quality_flags}")
    else:
        log(f"Done. volume=unavailable, flags={quality_flags}, detail={volume_info}")


if __name__ == "__main__":
    main()
