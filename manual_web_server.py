#!/usr/bin/env python
"""Manual web-based point-cloud selection and volume estimation.

This tool serves a local web UI where users can move/scale a 3D box
on top of a point cloud and estimate volume for points inside the box.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import struct
from datetime import datetime
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import numpy as np


GRID_CELL_MIN = 0.01
GRID_CELL_MAX = 5.0
GRID_CELL_EFFECTIVE_MIN = 0.109


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

        fmt = None
        vertex_count = None
        vertex_props = []
        in_vertex = False
        for ln in header_lines:
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
                vertex_props.append((parts[2], parts[1]))  # (name, type)

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
            data = np.loadtxt(path, skiprows=len(header_lines), max_rows=vertex_count)
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


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0:
        return np.arange(points.shape[0], dtype=np.int64)
    mins = points.min(axis=0)
    keys = np.floor((points - mins) / voxel).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep.sort()
    return keep


def estimate_lowest_point_volume(
    points: np.ndarray,
    grid_cell: float,
    base_percentile: float,
) -> Dict[str, object]:
    if points.shape[0] < 30:
        raise ValueError("Too few selected points for volume estimation (need >= 30)")

    xy = points[:, :2]
    z = points[:, 2]
    requested_cell = float(grid_cell)
    recommended_cell = float(GRID_CELL_EFFECTIVE_MIN)
    effective_cell = max(requested_cell, recommended_cell)

    mins = np.min(xy, axis=0)
    ij = np.floor((xy - mins[None, :]) / effective_cell).astype(np.int32)
    uniq, inv = np.unique(ij, axis=0, return_inverse=True)

    top = np.full(uniq.shape[0], -np.inf, dtype=np.float32)
    np.maximum.at(top, inv, z.astype(np.float32))

    pct = float(np.clip(base_percentile, 0.0, 100.0))
    base_z = float(np.percentile(z, pct))
    h = np.maximum(top - base_z, 0.0)

    vol = float(np.sum(h) * (effective_cell * effective_cell))
    area = float(uniq.shape[0] * (effective_cell * effective_cell))
    adjusted = effective_cell > (requested_cell + 1e-9)

    out: Dict[str, object] = {
        "volume_m3": vol,
        "grid_cell_m": float(effective_cell),
        "requested_grid_cell_m": float(requested_cell),
        "recommended_min_grid_cell_m": float(recommended_cell),
        "grid_cell_adjusted": bool(adjusted),
        "base_z": base_z,
        "base_percentile": pct,
        "footprint_area_m2": area,
        "cell_count": int(uniq.shape[0]),
        "mean_height_m": float(np.mean(h)) if h.size else 0.0,
        "max_height_m": float(np.max(h)) if h.size else 0.0,
    }
    if adjusted:
        out["grid_adjust_reason"] = (
            "Requested grid is below manual-web minimum effective grid 0.109 m; "
            "effective grid was raised to 0.109 m for stable volume integration"
        )

    return out


class ManualVolumeState:
    def __init__(
        self,
        root: Path,
        pointcloud_path: Path,
        max_points: int,
        voxel_size: float,
        output_dir: Path,
    ) -> None:
        self.root = root
        self.pointcloud_path = pointcloud_path
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.output_dir = output_dir
        self.web_root = root / "manual_web"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        points_all, colors_all = read_ply_vertices(pointcloud_path)
        keep = voxel_downsample(points_all, voxel=voxel_size)
        points = points_all[keep]
        colors = colors_all[keep] if colors_all is not None else None

        if max_points > 0 and points.shape[0] > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(points.shape[0], size=max_points, replace=False)
            idx.sort()
            points = points[idx]
            colors = colors[idx] if colors is not None else None

        self.points = np.ascontiguousarray(points.astype(np.float32))
        self.colors = np.ascontiguousarray(colors.astype(np.uint8)) if colors is not None else None

        bmin = self.points.min(axis=0)
        bmax = self.points.max(axis=0)
        center = (bmin + bmax) * 0.5

        self.meta = {
            "point_count": int(self.points.shape[0]),
            "pointcloud_path": str(self.pointcloud_path),
            "bbox_min": [float(v) for v in bmin],
            "bbox_max": [float(v) for v in bmax],
            "center": [float(v) for v in center],
            "voxel_size": float(self.voxel_size),
            "max_points": int(self.max_points),
            "default_grid_cell": 0.2,
            "default_base_percentile": 0.0,
        }

        has_color = 1 if self.colors is not None else 0
        header = struct.pack("<4sII", b"PCD1", self.points.shape[0], has_color)
        blob = bytearray(header)
        blob.extend(self.points.tobytes(order="C"))
        if self.colors is not None:
            blob.extend(self.colors.tobytes(order="C"))
        self.pointcloud_blob = bytes(blob)

    def _select_points(self, bmin: np.ndarray, bmax: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        mask = np.all((self.points >= bmin[None, :]) & (self.points <= bmax[None, :]), axis=1)
        selected = self.points[mask]
        selected_colors = self.colors[mask] if self.colors is not None else None
        return selected, selected_colors, mask

    def _select_points_by_indices(self, indices: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        if indices.size == 0:
            raise ValueError("indices is empty")

        idx = indices.astype(np.int64, copy=False).reshape(-1)
        valid = (idx >= 0) & (idx < self.points.shape[0])
        if not np.any(valid):
            raise ValueError("No valid indices in selection")

        idx = np.unique(idx[valid])
        selected = self.points[idx]
        selected_colors = self.colors[idx] if self.colors is not None else None
        return selected, selected_colors, idx

    @staticmethod
    def _parse_numeric_param(
        payload: Dict[str, object],
        key: str,
        default: float,
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        try:
            value = float(payload.get(key, default))
        except Exception as exc:
            raise ValueError(f"{key} must be a numeric value: {exc}") from exc

        if not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number")

        if min_value is not None and value < min_value:
            raise ValueError(f"{key} must be >= {min_value}")
        if max_value is not None and value > max_value:
            raise ValueError(f"{key} must be <= {max_value}")
        return value

    def estimate_volume(self, payload: Dict[str, object]) -> Dict[str, object]:
        if "min" not in payload or "max" not in payload:
            raise ValueError("Payload must contain min and max")

        bmin = np.asarray(payload["min"], dtype=np.float32)
        bmax = np.asarray(payload["max"], dtype=np.float32)
        if bmin.shape != (3,) or bmax.shape != (3,):
            raise ValueError("min/max must be arrays with 3 numbers")

        lo = np.minimum(bmin, bmax)
        hi = np.maximum(bmin, bmax)

        grid_cell = self._parse_numeric_param(
            payload,
            "grid_cell",
            0.2,
            min_value=GRID_CELL_MIN,
            max_value=GRID_CELL_MAX,
        )
        base_percentile = self._parse_numeric_param(
            payload,
            "base_percentile",
            0.0,
            min_value=0.0,
            max_value=100.0,
        )

        selected, _, _ = self._select_points(lo, hi)
        stats = estimate_lowest_point_volume(selected, grid_cell=grid_cell, base_percentile=base_percentile)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "timestamp": now,
            "method": "lowest_point",
            "selection_box": {
                "min": [float(v) for v in lo],
                "max": [float(v) for v in hi],
            },
            "selected_point_count": int(selected.shape[0]),
            "volume": stats,
        }

        report_path = self.output_dir / f"manual_volume_{now}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return {
            **report,
            "saved_report": str(report_path),
        }

    def estimate_volume_by_indices(self, payload: Dict[str, object]) -> Dict[str, object]:
        if "indices" not in payload:
            raise ValueError("Payload must contain indices")

        grid_cell = self._parse_numeric_param(
            payload,
            "grid_cell",
            0.2,
            min_value=GRID_CELL_MIN,
            max_value=GRID_CELL_MAX,
        )
        base_percentile = self._parse_numeric_param(
            payload,
            "base_percentile",
            0.0,
            min_value=0.0,
            max_value=100.0,
        )

        try:
            raw_indices = np.asarray(payload["indices"], dtype=np.int64)
        except Exception as exc:
            raise ValueError(f"indices must be an integer array: {exc}") from exc

        selected, _, idx = self._select_points_by_indices(raw_indices)
        stats = estimate_lowest_point_volume(selected, grid_cell=grid_cell, base_percentile=base_percentile)

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "timestamp": now,
            "method": "lowest_point",
            "selection_type": "screen_polygon",
            "selected_index_count": int(idx.size),
            "selected_point_count": int(selected.shape[0]),
            "volume": stats,
        }

        report_path = self.output_dir / f"manual_volume_{now}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        return {
            **report,
            "saved_report": str(report_path),
        }

    def export_selection(self, payload: Dict[str, object]) -> Dict[str, object]:
        if "min" not in payload or "max" not in payload:
            raise ValueError("Payload must contain min and max")

        bmin = np.asarray(payload["min"], dtype=np.float32)
        bmax = np.asarray(payload["max"], dtype=np.float32)
        if bmin.shape != (3,) or bmax.shape != (3,):
            raise ValueError("min/max must be arrays with 3 numbers")

        lo = np.minimum(bmin, bmax)
        hi = np.maximum(bmin, bmax)

        selected, selected_colors, _ = self._select_points(lo, hi)
        if selected.shape[0] == 0:
            raise ValueError("No points found in the selected box")

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"manual_selected_{now}.ply"
        write_ply_vertices(out_path, selected, selected_colors)

        return {
            "timestamp": now,
            "selected_point_count": int(selected.shape[0]),
            "saved_ply": str(out_path),
        }

    def export_selection_by_indices(self, payload: Dict[str, object]) -> Dict[str, object]:
        if "indices" not in payload:
            raise ValueError("Payload must contain indices")

        try:
            raw_indices = np.asarray(payload["indices"], dtype=np.int64)
        except Exception as exc:
            raise ValueError(f"indices must be an integer array: {exc}") from exc

        selected, selected_colors, idx = self._select_points_by_indices(raw_indices)
        if selected.shape[0] == 0:
            raise ValueError("No points found in selection")

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"manual_selected_{now}.ply"
        write_ply_vertices(out_path, selected, selected_colors)

        return {
            "timestamp": now,
            "selection_type": "screen_polygon",
            "selected_index_count": int(idx.size),
            "selected_point_count": int(selected.shape[0]),
            "saved_ply": str(out_path),
        }


class ManualHandler(BaseHTTPRequestHandler):
    server_version = "ManualVolumeServer/1.0"

    def __init__(self, *args, state: ManualVolumeState, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def _send_json(self, code: int, data: Dict[str, object]) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_bytes(self, code: int, content_type: str, data: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_static_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        data = file_path.read_bytes()
        ctype, _ = mimetypes.guess_type(str(file_path))
        self._send_bytes(HTTPStatus.OK, ctype or "application/octet-stream", data)

    def _read_json_body(self) -> Dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Empty request body")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._serve_static_file(self.state.web_root / "index.html")
            return

        if path == "/api/meta":
            self._send_json(HTTPStatus.OK, self.state.meta)
            return

        if path == "/api/pointcloud.bin":
            self._send_bytes(HTTPStatus.OK, "application/octet-stream", self.state.pointcloud_blob)
            return

        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            safe_rel = Path(rel)
            if ".." in safe_rel.parts:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid path"})
                return
            self._serve_static_file(self.state.web_root / "static" / safe_rel)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            payload = self._read_json_body()
            if path == "/api/estimate_volume":
                if "indices" in payload:
                    result = self.state.estimate_volume_by_indices(payload)
                else:
                    result = self.state.estimate_volume(payload)
                self._send_json(HTTPStatus.OK, result)
                return
            if path == "/api/export_selection":
                if "indices" in payload:
                    result = self.state.export_selection_by_indices(payload)
                else:
                    result = self.state.export_selection(payload)
                self._send_json(HTTPStatus.OK, result)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Manual web selection and volume estimation server")
    p.add_argument("--root", type=Path, default=root)
    p.add_argument(
        "--pointcloud",
        type=Path,
        default=root / "odmoutput" / "odm_filterpoints" / "point_cloud.ply",
        help="Input point cloud for manual selection",
    )
    p.add_argument("--output", type=Path, default=root / "results" / "manual_web")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-points", type=int, default=220000)
    p.add_argument("--voxel-size", type=float, default=0.07)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    state = ManualVolumeState(
        root=args.root,
        pointcloud_path=args.pointcloud,
        max_points=max(0, int(args.max_points)),
        voxel_size=max(0.0, float(args.voxel_size)),
        output_dir=args.output,
    )

    handler = partial(ManualHandler, state=state)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Loaded points: {state.meta['point_count']}", flush=True)
    print(f"Point cloud: {args.pointcloud}", flush=True)
    print(f"Open UI: http://{args.host}:{args.port}", flush=True)
    print(f"Reports: {args.output}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
