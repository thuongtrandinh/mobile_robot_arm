#!/usr/bin/env python3
"""Generate ACADOS NMPC C code for DDMR with IPOPT-equivalent math.

Key features:
- Augmented state for smoothness recovery:
  X = [x, y, theta, v, r, acc, dr]
- Real control input:
  U_real = [j_v, j_r]
- Virtual polygon dual controls in input:
  U = [j_v, j_r, lam_1, ..., lam_M]
- Single flat runtime parameter vector p for scene data per stage:
  [walls(8), humans(3H), static_obstacles(3S), polygon_edges(3M)]
- Nonlinear constraints include wheel limits, walls, humans, static obstacles,
  dual norm relaxation, and polygon min-penetration.
- Soft constraints apply only to humans, static obstacles, polygon min-penetration.
"""

from pathlib import Path
import sys
from typing import Dict, Any

import numpy as np
from casadi import SX, vertcat, cos, sin

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except ImportError as exc:
    print("acados_template is not available. Source ACADOS env first.")
    print(f"Import error: {exc}")
    sys.exit(1)


def _default_config() -> Dict[str, Any]:
    return {
        "mpc_structure": {
            "n_horizon": 20,
            "dt": 0.1,
            "wheel_half_track": 0.3,
            "inflation_radius": 0.3,
        },
        "scene_limits": {
            "num_walls": 2,
            "num_humans": 5,
            "num_static_obstacles": 40,
            "num_polygon_edges": 20,
        },
        "cost_weights": {
            "w_x": 5.0,
            "w_y": 5.0,
            "w_theta": 1.0,
            "w_v": 1.0,
            "w_r": 1.0,
            "w_acc": 2.0,
            "w_dr": 0.5,
            "w_smooth_acc": 0.05,
            "w_smooth_dr": 0.05,
            "w_x_e": 100.0,
            "w_y_e": 100.0,
            "w_theta_e": 1.0,
            "w_v_e": 1.0,
            "w_r_e": 1.0,
            "w_slack": 99999.0,
            "w_lambda_reg": 1e-4,
        },
        "constraints": {
            "max_linear_vel": 1.0,
            "max_linear_acc": 1.0,
            "max_angular_vel": 1.0,
            "max_angular_acc": 1.0,
            "max_linear_jerk": 5.0,
            "max_angular_jerk": 5.0,
            "lambda_upper_bound": 1.0e3,
        },
    }


def _load_config(package_dir: Path) -> Dict[str, Any]:
    cfg = _default_config()
    cfg_path = package_dir / "config" / "ddmr_mpc_config.yaml"
    if not cfg_path.exists() or yaml is None:
        return cfg

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    params = raw.get("opt_planner", {}).get("ros__parameters", {})

    ms = cfg["mpc_structure"]
    ms["n_horizon"] = int(params.get("mpc_structure.n_horizon", ms["n_horizon"]))
    ms["dt"] = float(params.get("mpc_structure.dt", ms["dt"]))
    ms["wheel_half_track"] = float(params.get("mpc_structure.wheel_half_track", ms["wheel_half_track"]))

    sl = cfg["scene_limits"]
    sl["num_walls"] = int(params.get("scene_limits.num_walls", sl["num_walls"]))
    sl["num_humans"] = int(params.get("scene_limits.num_humans", sl["num_humans"]))
    sl["num_static_obstacles"] = int(
        params.get("scene_limits.num_static_obstacles", sl["num_static_obstacles"])
    )
    sl["num_polygon_edges"] = int(params.get("scene_limits.num_polygon_edges", sl["num_polygon_edges"]))

    cw = cfg["cost_weights"]
    cw["w_x"] = float(params.get("cost_weights.w_x", cw["w_x"]))
    cw["w_y"] = float(params.get("cost_weights.w_y", cw["w_y"]))
    cw["w_theta"] = float(params.get("cost_weights.w_theta", cw["w_theta"]))
    cw["w_v"] = float(params.get("cost_weights.w_v", cw["w_v"]))
    cw["w_r"] = float(params.get("cost_weights.w_r", cw["w_r"]))
    cw["w_acc"] = float(params.get("cost_weights.w_acc", cw["w_acc"]))
    cw["w_dr"] = float(params.get("cost_weights.w_dr", cw["w_dr"]))
    cw["w_x_e"] = float(params.get("cost_weights.w_x_e", cw["w_x_e"]))
    cw["w_y_e"] = float(params.get("cost_weights.w_y_e", cw["w_y_e"]))
    cw["w_theta_e"] = float(params.get("cost_weights.w_theta_e", cw["w_theta_e"]))
    cw["w_v_e"] = float(params.get("cost_weights.w_v_e", cw["w_v_e"]))
    cw["w_r_e"] = float(params.get("cost_weights.w_r_e", cw["w_r_e"]))
    cw["w_smooth_acc"] = float(params.get("cost_weights.w_smooth_acc", params.get("mpc.weights.smooth_acc", cw["w_smooth_acc"])))
    cw["w_smooth_dr"] = float(params.get("cost_weights.w_smooth_dr", params.get("mpc.weights.smooth_yaw_acc", cw["w_smooth_dr"])))
    cw["w_slack"] = float(params.get("cost_weights.w_slack", params.get("mpc.weights.slack", cw["w_slack"])))

    ct = cfg["constraints"]
    ct["max_linear_vel"] = float(params.get("constraints.max_linear_vel", ct["max_linear_vel"]))
    ct["max_linear_acc"] = float(params.get("constraints.max_linear_acc", ct["max_linear_acc"]))
    ct["max_angular_vel"] = float(params.get("constraints.max_angular_vel", ct["max_angular_vel"]))
    ct["max_angular_acc"] = float(params.get("constraints.max_angular_acc", ct["max_angular_acc"]))
    return cfg


def _make_param_layout(cfg: Dict[str, Any]) -> Dict[str, int]:
    walls = int(cfg["scene_limits"]["num_walls"])
    humans = int(cfg["scene_limits"]["num_humans"])
    statics = int(cfg["scene_limits"]["num_static_obstacles"])
    edges = int(cfg["scene_limits"]["num_polygon_edges"])
    if walls != 2:
        raise ValueError("This script assumes exactly 2 walls.")

    offset_walls = 0
    size_walls = 4 * walls

    offset_humans = offset_walls + size_walls
    size_humans = 3 * humans

    offset_statics = offset_humans + size_humans
    size_statics = 3 * statics

    offset_edges = offset_statics + size_statics
    size_edges = 3 * edges

    np_total = size_walls + size_humans + size_statics + size_edges

    return {
        "walls": walls,
        "humans": humans,
        "statics": statics,
        "edges": edges,
        "offset_walls": offset_walls,
        "offset_humans": offset_humans,
        "offset_statics": offset_statics,
        "offset_edges": offset_edges,
        "np_total": np_total,
    }


def build_ddmr_model(param_layout: Dict[str, int]) -> AcadosModel:
    model = AcadosModel()
    model.name = "ddmr"

    # X = [x, y, theta, v, r, acc, dr]
    x = SX.sym("x")
    y = SX.sym("y")
    theta = SX.sym("theta")
    v = SX.sym("v")
    r = SX.sym("r")
    acc = SX.sym("acc")
    dr = SX.sym("dr")
    x_vec = vertcat(x, y, theta, v, r, acc, dr)

    # U = [j_v, j_r, lam_1, ..., lam_M]
    j_v = SX.sym("j_v")
    j_r = SX.sym("j_r")
    lam = SX.sym("lam", param_layout["edges"], 1)
    u_vec = vertcat(j_v, j_r, lam)

    xdot = SX.sym("xdot", x_vec.rows(), 1)

    f_expl = vertcat(
        v * cos(theta),
        v * sin(theta),
        r,
        acc,
        dr,
        j_v,
        j_r,
    )
    f_impl = xdot - f_expl

    p = SX.sym("p", param_layout["np_total"], 1)

    model.x = x_vec
    model.u = u_vec
    model.xdot = xdot
    model.p = p
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    return model


def build_ocp(package_dir: Path) -> AcadosOcp:
    cfg = _load_config(package_dir)
    pl = _make_param_layout(cfg)

    n_horizon = int(cfg["mpc_structure"]["n_horizon"])
    dt = float(cfg["mpc_structure"]["dt"])
    wheel_half_track = float(cfg["mpc_structure"]["wheel_half_track"])
    inflation_radius = float(cfg["mpc_structure"]["inflation_radius"])

    if n_horizon < 2:
        raise ValueError("n_horizon must be >= 2")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    ocp = AcadosOcp()
    ocp.model = build_ddmr_model(pl)

    ocp.dims.N = n_horizon
    ocp.solver_options.tf = n_horizon * dt

    nx = int(ocp.model.x.rows())
    nu = int(ocp.model.u.rows())
    ny = nx + nu
    ny_e = 5

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :nu] = np.eye(nu)

    cw = cfg["cost_weights"]
    w_j_v = float(cw["w_smooth_acc"]) * (dt ** 2)
    w_j_r = float(cw["w_smooth_dr"]) * (dt ** 2)
    w_lambda_reg = float(cw["w_lambda_reg"])

    w_diag = [
        float(cw["w_x"]),
        float(cw["w_y"]),
        float(cw["w_theta"]),
        float(cw["w_v"]),
        float(cw["w_r"]),
        float(cw["w_acc"]),
        float(cw["w_dr"]),
        w_j_v,
        w_j_r,
    ] + [w_lambda_reg] * pl["edges"]
    ocp.cost.W = np.diag(w_diag)

    ocp.cost.Vx_e = np.zeros((ny_e, nx))
    ocp.cost.Vx_e[0, 0] = 1.0
    ocp.cost.Vx_e[1, 1] = 1.0
    ocp.cost.Vx_e[2, 2] = 1.0
    ocp.cost.Vx_e[3, 3] = 1.0
    ocp.cost.Vx_e[4, 4] = 1.0
    ocp.cost.W_e = np.diag([
        float(cw["w_x_e"]),
        float(cw["w_y_e"]),
        float(cw["w_theta_e"]),
        float(cw["w_v_e"]),
        float(cw["w_r_e"]),
    ])

    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    ct = cfg["constraints"]
    max_linear_vel = float(ct["max_linear_vel"])
    max_linear_acc = float(ct["max_linear_acc"])
    max_angular_vel = float(ct["max_angular_vel"])
    max_angular_acc = float(ct["max_angular_acc"])
    max_linear_jerk = float(ct["max_linear_jerk"])
    max_angular_jerk = float(ct["max_angular_jerk"])
    lambda_upper = float(ct["lambda_upper_bound"])

    x = ocp.model.x
    u = ocp.model.u
    p = ocp.model.p

    px = x[0]
    py = x[1]
    v = x[3]
    r = x[4]
    acc = x[5]
    dr = x[6]
    lam = u[2:2 + pl["edges"]]

    # 1) Wheel limits
    wheel_v_left = v - r * wheel_half_track
    wheel_v_right = v + r * wheel_half_track
    wheel_a_left = acc - dr * wheel_half_track
    wheel_a_right = acc + dr * wheel_half_track

    # 2) Walls: dx * (Y - sy) - dy * (X - sx) >= 0
    wall_terms = []
    for i in range(pl["walls"]):
        base = pl["offset_walls"] + 4 * i
        sx = p[base + 0]
        sy = p[base + 1]
        dx_w = p[base + 2]
        dy_w = p[base + 3]
        wall_terms.append(dx_w * (py - sy) - dy_w * (px - sx))

    # 3) Humans
    human_terms = []
    for i in range(pl["humans"]):
        base = pl["offset_humans"] + 3 * i
        hx = p[base + 0]
        hy = p[base + 1]
        h_safe2 = p[base + 2]
        human_terms.append((px - hx) ** 2 + (py - hy) ** 2 - h_safe2)

    # 4) Static obstacles
    static_terms = []
    for i in range(pl["statics"]):
        base = pl["offset_statics"] + 3 * i
        ox = p[base + 0]
        oy = p[base + 1]
        o_safe2 = p[base + 2]
        static_terms.append((px - ox) ** 2 + (py - oy) ** 2 - o_safe2)

    # 5) Polygon dual constraints
    # norm relaxation: 0.95 <= ||A^T lam||^2 <= 1.05
    axtlam = 0
    aytlam = 0
    min_pen_terms = []
    for i in range(pl["edges"]):
        base = pl["offset_edges"] + 3 * i
        ax_i = p[base + 0]
        ay_i = p[base + 1]
        b_i = p[base + 2]
        axtlam = axtlam + ax_i * lam[i]
        aytlam = aytlam + ay_i * lam[i]
        min_pen_terms.append((ax_i * px + ay_i * py - b_i) * lam[i])

    dual_norm_relaxed = axtlam * axtlam + aytlam * aytlam
    clearance_target = inflation_radius + 0.1

    h_terms = [
        wheel_v_left,
        wheel_v_right,
        wheel_a_left,
        wheel_a_right,
        *wall_terms,
        *human_terms,
        *static_terms,
        dual_norm_relaxed,
        *min_pen_terms,
    ]
    ocp.model.con_h_expr = vertcat(*h_terms)

    n_wheel = 4
    n_wall = pl["walls"]
    n_human = pl["humans"]
    n_static = pl["statics"]
    n_norm = 1
    n_min_pen = pl["edges"]

    nh = n_wheel + n_wall + n_human + n_static + n_norm + n_min_pen
    lh = np.zeros(nh)
    uh = np.zeros(nh)

    idx = 0
    # Wheel limits
    lh[idx:idx + 4] = np.array([-max_linear_vel, -max_linear_vel, -max_linear_acc, -max_linear_acc])
    uh[idx:idx + 4] = np.array([max_linear_vel, max_linear_vel, max_linear_acc, max_linear_acc])
    idx += 4

    # Walls
    lh[idx:idx + n_wall] = 0.0
    uh[idx:idx + n_wall] = 1.0e9
    idx += n_wall

    # Humans (soft)
    idx_human_start = idx
    lh[idx:idx + n_human] = 0.0
    uh[idx:idx + n_human] = 1.0e9
    idx += n_human

    # Static obstacles (soft)
    idx_static_start = idx
    lh[idx:idx + n_static] = 0.0
    uh[idx:idx + n_static] = 1.0e9
    idx += n_static

    # Dual norm relaxed (hard)
    lh[idx] = 0.95
    uh[idx] = 1.05
    idx += 1

    # Polygon min-penetration (soft)
    idx_min_pen_start = idx
    lh[idx:idx + n_min_pen] = clearance_target
    uh[idx:idx + n_min_pen] = 1.0e9

    ocp.constraints.lh = lh
    ocp.constraints.uh = uh

    # Bounds on state and input
    ocp.constraints.idxbx = np.array([4], dtype=np.int64)
    ocp.constraints.lbx = np.array([-max_angular_vel])
    ocp.constraints.ubx = np.array([max_angular_vel])

    # Input bounds: jerk for first 2 controls, lambda >=0 for virtual controls
    idxbu = [0, 1]
    lbu = [-max_linear_jerk, -max_angular_jerk]
    ubu = [max_linear_jerk, max_angular_jerk]
    for i in range(pl["edges"]):
        idxbu.append(2 + i)
        lbu.append(0.0)
        ubu.append(lambda_upper)
    ocp.constraints.idxbu = np.array(idxbu, dtype=np.int64)
    ocp.constraints.lbu = np.array(lbu)
    ocp.constraints.ubu = np.array(ubu)

    ocp.constraints.x0 = np.zeros(nx)

    # Soft constraints ONLY for human/static/min-penetration constraints.
    idxsh_list = list(range(idx_human_start, idx_human_start + n_human))
    idxsh_list += list(range(idx_static_start, idx_static_start + n_static))
    idxsh_list += list(range(idx_min_pen_start, idx_min_pen_start + n_min_pen))
    idxsh = np.array(idxsh_list, dtype=np.int64)
    nsh = idxsh.size

    ocp.constraints.idxsh = idxsh
    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)

    w_slack = float(cw["w_slack"])
    ocp.cost.zl = np.zeros(nsh)
    ocp.cost.zu = np.zeros(nsh)
    ocp.cost.Zl = w_slack * np.ones(nsh)
    ocp.cost.Zu = w_slack * np.ones(nsh)

    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.qp_solver_cond_N = n_horizon

    ocp.parameter_values = np.zeros(pl["np_total"])
    ocp.code_export_directory = str(package_dir / "c_generated_code")
    return ocp


def main() -> int:
    package_dir = Path(__file__).resolve().parents[1]
    ocp = build_ocp(package_dir)

    json_file = package_dir / "ddmr_acados_ocp.json"
    AcadosOcpSolver(ocp, json_file=str(json_file))

    print("ACADOS OCP generated successfully.")
    print(f"- JSON: {json_file}")
    print(f"- C code: {ocp.code_export_directory}")
    print("- State: [x,y,theta,v,r,acc,dr]")
    print("- Input: [j_v,j_r,lam_1..lam_M]")
    print("- Soft constraints: humans + static obstacles + polygon min-penetration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
