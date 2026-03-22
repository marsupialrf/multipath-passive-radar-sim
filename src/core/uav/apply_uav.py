from __future__ import annotations
import math
from typing import List, Set, Tuple

import numpy as np

try:
    from numba import cuda as _cuda
    _HAS_CUDA = True
except ImportError:
    _HAS_CUDA = False; _cuda = None

from src.core.scene.ray         import Ray
from src.core.scene.propagation import (compute_sphere_rcs_bounce_gain,
                                         compute_scattered_doppler)
from src.core.gpu.kernels        import _HAS_CUDA
if _HAS_CUDA:
    from src.core.gpu.kernels import mini_trace_kernel
from src.core.gpu.utils import fspl_const, obs_arrays


def apply_uav(static, uav, scene) -> Tuple[List[Ray], List[Ray], List[Ray]]:
    """
    Apply a UAV to a precomputed StaticField.

    Steps
    -----
    1. Spatial hash query → candidate (ray_id, seg_id) pairs near UAV.
    2. Vectorised sphere-hit filter (NumPy) → confirmed hits + blocked ray ids.
    3. Noise-floor filter (vectorised).
    4. Small Python loop over confirmed hits (≤ ~20) to build path prefixes.
    5. GPU batch: mini_trace_kernel traces all (N_hits × n_samples) post-UAV
       rays in one launch.
    6. Assemble Ray objects; split anchors into visible / occluded.

    Returns
    -------
    anchors_vis  : visible baseline paths
    anchors_occ  : occluded baseline paths (visible=False)
    uav_bounces  : new paths via UAV (is_uav_bounce=True)
    """
    uav_pos   = np.asarray(uav.position, dtype=np.float64)
    uav_vel   = np.asarray(uav.velocity, dtype=np.float64)
    uav_rad   = float(uav.radius)
    scene_ref = static.scene_ref
    roughness = float(scene_ref.roughness)
    uav_rough = float(scene_ref.uav_roughness)
    n_samp    = int(scene_ref.n_samples_uav)
    noise_f   = float(scene_ref.noise_floor_dbm) if scene_ref.use_physics else float('-inf')
    n_post    = max(4, int(scene_ref.n_max) // 2)
    rx_pos    = np.asarray(scene_ref.receiver.position, dtype=np.float64)
    rx_rad    = float(scene_ref.receiver.radius)
    fc        = static.fc
    rcs_gain  = compute_sphere_rcs_bounce_gain(uav_rad, fc)
    fc_c      = fspl_const(fc)

    # ── 1. Spatial hash query ─────────────────────────────────────────────────
    candidates = static.spatial_hash.query(uav_pos, uav_rad)

    if not candidates:
        # No candidates — all anchors visible, no bounces
        anchors_vis = []
        for ray in static.anchors:
            r = Ray(ray.transmitter_id, ray.points, ray.arrival_dir,
                    ray.frequency, ray.power_dbm)
            r.is_uav_bounce = False; r.doppler_shift = 0.0; r.visible = True
            anchors_vis.append(r)
        return anchors_vis, [], []

    # ── 2. Vectorised sphere-hit filter ──────────────────────────────────────
    cands_arr = np.array(candidates, dtype=np.int32)   # (K, 2)
    rids_k    = cands_arr[:, 0]
    sids_k    = cands_arr[:, 1]

    # Drop segments beyond ray end
    n_valid_k = static.n_pts_cpu[rids_k]
    mask_v    = (sids_k + 1) < n_valid_k
    rids_k    = rids_k[mask_v];  sids_k = sids_k[mask_v]
    if rids_k.size == 0:
        return _all_visible(static), [], []

    # Extract segment endpoints (K, 3)
    A = static.pos_cpu[sids_k,   rids_k, :].astype(np.float64)
    B = static.pos_cpu[sids_k+1, rids_k, :].astype(np.float64)
    AB = B - A
    L  = np.linalg.norm(AB, axis=1)           # (K,)

    mask_len = L >= 1e-9
    A = A[mask_len]; B = B[mask_len]; AB = AB[mask_len]; L = L[mask_len]
    rids_k = rids_k[mask_len]; sids_k = sids_k[mask_len]
    if rids_k.size == 0:
        return _all_visible(static), [], []

    seg_dirs = AB / L[:, None]                # (K, 3)  unit vectors

    # Sphere intersection — vectorised quadratic (a=1 since seg_dirs are unit)
    oc   = A - uav_pos[None, :]               # (K, 3)
    b    = 2.0 * (oc * seg_dirs).sum(axis=1)  # (K,)
    c_   = (oc * oc).sum(axis=1) - uav_rad**2 # (K,)
    disc = b*b - 4.0*c_
    sq   = np.sqrt(np.maximum(disc, 0.0))
    t1   = (-b - sq) * 0.5
    t2   = (-b + sq) * 0.5
    t    = np.where(t1 > 1e-5, t1, t2)        # prefer nearer root
    mask_hit = (disc >= 0) & (t > 1e-5) & (t < L)

    # All rays that geometrically hit the UAV sphere block the anchor path
    blocked_ids: Set[int] = set(rids_k[mask_hit].tolist())

    # ── 3. Noise-floor filter (vectorised) ────────────────────────────────────
    rids_h = rids_k[mask_hit];  sids_h = sids_k[mask_hit]
    t_h    = t[mask_hit];       segs_h = seg_dirs[mask_hit]
    A_h    = A[mask_hit]

    pwr_seg  = static.step_powers[sids_h, rids_h].astype(np.float64)
    fspl_h   = 20.0 * np.log10(np.maximum(t_h, 1e-9)) + fc_c
    pwr_hit  = pwr_seg - fspl_h + rcs_gain
    mask_pwr = pwr_hit > noise_f

    rids_f  = rids_h[mask_pwr];  sids_f = sids_h[mask_pwr]
    t_f     = t_h[mask_pwr];     segs_f = segs_h[mask_pwr]
    A_f     = A_h[mask_pwr];     pow_f  = pwr_hit[mask_pwr]

    hit_pts = A_f + t_f[:, None] * segs_f     # (M, 3)
    n_uavs  = (hit_pts - uav_pos[None, :]) / uav_rad  # (M, 3) outward normals

    # ── 4. Build path prefixes (small loop over M confirmed hits) ─────────────
    hit_pts_list : List[np.ndarray] = []
    v_in_list    : List[np.ndarray] = []
    n_uav_list   : List[np.ndarray] = []
    powers_list  : List[np.float32] = []
    pre_pts_list : List[List]       = []
    v_in_f64_list: List[np.ndarray] = []

    for i in range(len(rids_f)):
        rid = int(rids_f[i]);  sid = int(sids_f[i])
        n_valid = int(static.n_pts_cpu[rid])
        pre_pts = [static.pos_cpu[j, rid, :].astype(np.float64)
                   for j in range(min(sid + 1, n_valid))]
        pre_pts.append(hit_pts[i].copy())

        hit_pts_list.append(hit_pts[i].astype(np.float32))
        v_in_list.append(segs_f[i].astype(np.float32))
        n_uav_list.append(n_uavs[i].astype(np.float32))
        powers_list.append(np.float32(pow_f[i]))
        pre_pts_list.append(pre_pts)
        v_in_f64_list.append(segs_f[i].copy())

    # ── 5. GPU batch mini-trace ───────────────────────────────────────────────
    uav_bounces: List[Ray] = []

    if hit_pts_list and _HAS_CUDA:
        N_hits  = len(hit_pts_list)
        N_total = N_hits * n_samp
        TPB     = 256
        BPG     = max(1, (N_total + TPB - 1) // TPB)

        hit_pts_g = _cuda.to_device(np.stack(hit_pts_list).astype(np.float32))
        v_in_g    = _cuda.to_device(np.stack(v_in_list).astype(np.float32))
        n_uav_g   = _cuda.to_device(np.stack(n_uav_list).astype(np.float32))
        powers_g  = _cuda.to_device(np.array(powers_list, dtype=np.float32))

        obs_min_np, obs_max_np = obs_arrays(scene_ref.obstacles)
        obs_min_g = _cuda.to_device(obs_min_np)
        obs_max_g = _cuda.to_device(obs_max_np)
        bmin_g    = _cuda.to_device(np.asarray(scene_ref.box.box_min, dtype=np.float32))
        bmax_g    = _cuda.to_device(np.asarray(scene_ref.box.box_max, dtype=np.float32))
        rx_pos_g  = _cuda.to_device(rx_pos.astype(np.float32))

        rch_g  = _cuda.to_device(np.zeros(N_total, dtype=np.int32))
        pwr_g  = _cuda.device_array((N_total,),             dtype=np.float32)
        adir_g = _cuda.device_array((N_total, 3),           dtype=np.float32)
        sdir_g = _cuda.device_array((N_total, 3),           dtype=np.float32)
        pos_g  = _cuda.device_array((n_post+2, N_total, 3), dtype=np.float32)
        npts_g = _cuda.to_device(np.ones(N_total, dtype=np.int32))

        mini_trace_kernel[BPG, TPB](
            rch_g, pwr_g, adir_g, sdir_g, pos_g, npts_g,
            hit_pts_g, v_in_g, n_uav_g, powers_g,
            obs_min_g, obs_max_g, rx_pos_g, bmin_g, bmax_g,
            np.float32(rx_rad), np.int32(n_post), np.float32(noise_f),
            np.float32(roughness), np.float32(uav_rough),
            np.int32(n_samp), np.float32(fc_c), np.int32(42),
        )
        _cuda.synchronize()

        rch_cpu  = rch_g.copy_to_host()
        pwr_cpu  = pwr_g.copy_to_host()
        adir_cpu = adir_g.copy_to_host()
        sdir_cpu = sdir_g.copy_to_host()
        pos_cpu  = pos_g.copy_to_host()
        npts_cpu = npts_g.copy_to_host()

        hit_done: Set[int] = set()
        for tid in range(N_total):
            if rch_cpu[tid] != 1: continue
            hit_id = tid // n_samp
            if hit_id in hit_done: continue
            hit_done.add(hit_id)

            n       = int(npts_cpu[tid])
            post    = [pos_cpu[j, tid, :].astype(np.float64) for j in range(n)]
            fin_pwr = float(pwr_cpu[tid])
            arr_dir = adir_cpu[tid].astype(np.float64)
            arr_dir = arr_dir / (np.linalg.norm(arr_dir) + 1e-30)

            d_sample = sdir_cpu[tid].astype(np.float64)
            doppler  = compute_scattered_doppler(uav_vel, v_in_f64_list[hit_id], d_sample, fc)

            all_pts = pre_pts_list[hit_id] + post[1:]
            r = Ray(transmitter_id=0, points=all_pts,
                    arrival_dir=arr_dir, frequency=float(fc), power_dbm=fin_pwr)
            r.is_uav_bounce = True; r.doppler_shift = doppler; r.visible = True
            uav_bounces.append(r)

    # ── 6. Split anchors ─────────────────────────────────────────────────────
    anchor_path_ids = sorted(static.anchor_ids)
    anchors_vis: List[Ray] = []; anchors_occ: List[Ray] = []

    for i, ray in enumerate(static.anchors):
        gid = anchor_path_ids[i] if i < len(anchor_path_ids) else -1
        r = Ray(transmitter_id=ray.transmitter_id, points=ray.points,
                arrival_dir=ray.arrival_dir, frequency=ray.frequency,
                power_dbm=ray.power_dbm)
        r.is_uav_bounce = False; r.doppler_shift = 0.0
        r.visible = gid not in blocked_ids
        (anchors_vis if r.visible else anchors_occ).append(r)

    return anchors_vis, anchors_occ, uav_bounces


def _all_visible(static) -> List[Ray]:
    """Return all anchors as visible (used when there are no candidates)."""
    result = []
    for ray in static.anchors:
        r = Ray(ray.transmitter_id, ray.points, ray.arrival_dir,
                ray.frequency, ray.power_dbm)
        r.is_uav_bounce = False; r.doppler_shift = 0.0; r.visible = True
        result.append(r)
    return result