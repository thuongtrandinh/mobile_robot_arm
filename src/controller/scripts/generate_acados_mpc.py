#!/usr/bin/env python3
"""Generate ACADOS C code for the DDMR MPC model.

This script creates an ACADOS OCP with a fixed horizon and exports C sources
into <controller_pkg>/c_generated_code.
"""

from pathlib import Path
import sys

import numpy as np
from casadi import SX, vertcat, cos, sin

try:
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
except ImportError as exc:
    print("acados_template is not available. Source ACADOS env first.")
    print(f"Import error: {exc}")
    sys.exit(1)


def build_ddmr_model() -> AcadosModel:
    model = AcadosModel()
    model.name = "ddmr"

    # State: X = [x, y, theta, v, r]
    x = SX.sym("x")
    y = SX.sym("y")
    theta = SX.sym("theta")
    v = SX.sym("v")
    r = SX.sym("r")
    x_vec = vertcat(x, y, theta, v, r)

    # Control: U = [acc, dr]
    acc = SX.sym("acc")
    dr = SX.sym("dr")
    u_vec = vertcat(acc, dr)

    xdot = SX.sym("xdot", x_vec.rows(), 1)

    # Differential model equivalent to RobotModelDifferential in mpc.cc
    f_expl = vertcat(
        v * cos(theta),
        v * sin(theta),
        r,
        acc,
        dr,
    )
    f_impl = xdot - f_expl

    model.x = x_vec
    model.u = u_vec
    model.xdot = xdot
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl

    return model


def build_ocp(package_dir: Path) -> AcadosOcp:
    ocp = AcadosOcp()
    ocp.model = build_ddmr_model()

    # Horizon setup
    n_horizon = 20
    dt = 0.1
    tf = n_horizon * dt
    ocp.dims.N = n_horizon
    ocp.solver_options.tf = tf

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu

    # LINEAR_LS cost (basic static setup)
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:, :nu] = np.eye(nu)

    ocp.cost.W = np.diag([
        5.0,   # x
        5.0,   # y
        1.0,   # theta
        1.0,   # v
        1.0,   # r
        2.0,   # acc
        0.5,   # dr
    ])

    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.W_e = np.diag([
        100.0,  # x_N
        100.0,  # y_N
        1.0,    # theta_N
        1.0,    # v_N
        1.0,    # r_N
    ])

    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(nx)

    # Constraints aligned with the legacy wheel-space limits in mpc.cc
    max_linear_vel = 1.0
    max_linear_acc = 1.0
    max_angular_vel = 1.0
    max_angular_acc = 1.0
    wheel_half_track = 0.3

    x = ocp.model.x
    u = ocp.model.u
    wheel_v_left = x[3] - x[4] * wheel_half_track
    wheel_v_right = x[3] + x[4] * wheel_half_track
    wheel_a_left = u[0] - u[1] * wheel_half_track
    wheel_a_right = u[0] + u[1] * wheel_half_track

    ocp.model.con_h_expr = vertcat(wheel_v_left, wheel_v_right, wheel_a_left, wheel_a_right)
    ocp.constraints.lh = np.array([
        -max_linear_vel,
        -max_linear_vel,
        -max_linear_acc,
        -max_linear_acc,
    ])
    ocp.constraints.uh = np.array([
        max_linear_vel,
        max_linear_vel,
        max_linear_acc,
        max_linear_acc,
    ])
    ocp.constraints.idxsh = np.array([], dtype=np.int64)

    # Direct bounds for rotational channels from legacy config values.
    ocp.constraints.lbx = np.array([-max_angular_vel])
    ocp.constraints.ubx = np.array([max_angular_vel])
    ocp.constraints.idxbx = np.array([4], dtype=np.int64)

    ocp.constraints.lbu = np.array([-max_linear_acc, -max_angular_acc])
    ocp.constraints.ubu = np.array([max_linear_acc, max_angular_acc])
    ocp.constraints.idxbu = np.array([0, 1], dtype=np.int64)

    # Initial state placeholder (set at runtime in C/C++)
    ocp.constraints.x0 = np.zeros(nx)

    # Solver configuration
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.qp_solver_cond_N = n_horizon

    ocp.code_export_directory = str(package_dir / "c_generated_code")

    return ocp


def main() -> int:
    package_dir = Path(__file__).resolve().parents[1]
    ocp = build_ocp(package_dir)

    json_file = package_dir / "ddmr_acados_ocp.json"
    AcadosOcpSolver(ocp, json_file=str(json_file))

    print(f"ACADOS OCP generated successfully.")
    print(f"- JSON: {json_file}")
    print(f"- C code: {ocp.code_export_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
