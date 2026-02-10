// Copyright 2026 H-AMPCC Authors
// SPDX-License-Identifier: Apache-2.0
//
// Common type definitions for the Adaptive MPCC backend.
// References:
//   - H-AMPCC §2.1  (LIP dynamics, proxy parameters)
//   - H-AMPCC §2.3  (contouring / lag errors in Frenet frame)
//   - H-AMPCC §3.2  (unified MPCC cost function)

#pragma once

#include <Eigen/Dense>
#include <cstdint>

namespace amr_controller {

// ──────────────────────────────────────────────
//  Dimension constants
// ──────────────────────────────────────────────
inline constexpr int kStateDim       = 5;   // [x, y, ψ, v, ω]
inline constexpr int kAugStateDim    = 6;   // + θ  (path progress)
inline constexpr int kControlDim     = 2;   // [R, M]  (force, torque)
inline constexpr int kAugControlDim  = 3;   // + v_θ (virtual progress velocity)
inline constexpr int kParamDim       = 4;   // [α_v, β_v, α_ω, β_ω]
inline constexpr int kSubParamDim    = 2;   // per-subsystem (linear / angular)

// ──────────────────────────────────────────────
//  Fixed-size Eigen aliases
// ──────────────────────────────────────────────
using State            = Eigen::Matrix<double, kStateDim,      1>;
using AugmentedState   = Eigen::Matrix<double, kAugStateDim,   1>;
using ControlInput     = Eigen::Matrix<double, kControlDim,    1>;
using AugmentedControl = Eigen::Matrix<double, kAugControlDim, 1>;
using ParamVector      = Eigen::Matrix<double, kParamDim,      1>;

using StateMatrix      = Eigen::Matrix<double, kAugStateDim, kAugStateDim>;
using InputMatrix      = Eigen::Matrix<double, kAugStateDim, kAugControlDim>;

// ──────────────────────────────────────────────
//  Index enumerations
// ──────────────────────────────────────────────

/// Indices into the (augmented) state vector.
enum StateIdx : int {
  kX     = 0,   ///< position x   [m]
  kY     = 1,   ///< position y   [m]
  kPsi   = 2,   ///< heading ψ    [rad]
  kV     = 3,   ///< linear vel.  [m/s]
  kOmega = 4,   ///< angular vel. [rad/s]
  kTheta = 5    ///< path-progress θ (arc-length)
};

/// Indices into the (augmented) control vector.
enum ControlIdx : int {
  kForce  = 0,  ///< driving force R    [N]
  kTorque = 1,  ///< yaw torque   M    [N·m]
  kVTheta = 2   ///< virtual progress velocity [m/s]
};

/// Indices into the proxy-parameter vector θ.
enum ParamIdx : int {
  kAlphaV     = 0,  ///< α_v  = 1 − bΔt/m
  kBetaV      = 1,  ///< β_v  = Δt/m
  kAlphaOmega = 2,  ///< α_ω  = 1 − cΔt/J
  kBetaOmega  = 3   ///< β_ω  = Δt/J
};

// ──────────────────────────────────────────────
//  Lightweight POD structs
// ──────────────────────────────────────────────

/// A single point on the reference path, evaluated at arc-length θ.
struct PathPoint {
  double x     = 0.0;   ///< position [m]
  double y     = 0.0;   ///< position [m]
  double phi   = 0.0;   ///< tangent angle [rad]
  double kappa = 0.0;   ///< signed curvature [1/m]
};

/// Linearized augmented system:  z_{k+1} = A z_k + B u_k
struct LinearizedModel {
  StateMatrix A = StateMatrix::Zero();
  InputMatrix B = InputMatrix::Zero();
};

/// Decomposed path-tracking error in the Frenet frame (§2.3).
struct ContouringErrors {
  double e_c = 0.0;   ///< contouring error  (perpendicular to path)
  double e_l = 0.0;   ///< lag error         (along path)
};

/// Jacobian of [e_c, e_l] w.r.t. the augmented state vector (§2.3).
struct ContouringErrorJacobian {
  Eigen::Matrix<double, 2, kAugStateDim> C =
      Eigen::Matrix<double, 2, kAugStateDim>::Zero();
  Eigen::Vector2d e = Eigen::Vector2d::Zero();   ///< [e_c, e_l]
};

// ──────────────────────────────────────────────
//  Tuning-parameter structs
// ──────────────────────────────────────────────

/// MPCC optimizer parameters (§3.2 cost weights & limits).
struct MPCCParams {
  // Horizon
  int    N   = 20;       ///< prediction horizon length
  double dt  = 0.05;     ///< sampling period [s]

  // Cost weights
  double q_c   = 10.0;   ///< contouring-error weight  Q_c
  double q_l   =  5.0;   ///< lag-error weight          Q_l
  double q_u_R     =  0.1;   ///< v_cmd input weight
  double q_u_M     =  0.1;   ///< ω_cmd input weight
  double q_v       =  1.0;   ///< velocity-tracking weight  Q_v
  double q_theta   =  2.0;   ///< path-progress incentive (−q_θ·v_θ·dt)
  double q_u_vtheta =  0.01; ///< small regularization on v_θ input

  // Reference & actuator limits
  double v_ref       =  1.0;   ///< desired cruise speed [m/s]
  double v_max       =  2.0;   ///< max linear velocity  [m/s]
  double omega_max   =  2.0;   ///< max angular velocity [rad/s]
  double v_cmd_max   =  2.0;   ///< max linear command   [m/s]
  double omega_cmd_max = 2.0;  ///< max angular command  [rad/s]
  double v_theta_max =  2.5;   ///< max virtual progress vel [m/s]
  double v_theta_min = -0.1;   ///< min virtual progress vel [m/s]

  /// @deprecated Use v_cmd_max/omega_cmd_max for the velocity-command model.
  double R_max     = 10.0;   ///< max driving force    [N] (legacy)
  double M_max     =  5.0;   ///< max yaw torque       [N·m] (legacy)
};

/// iLQR-based MPCC solver configuration.
struct SolverParams {
  int    max_iters       = 10;     ///< max iLQR outer iterations
  int    max_line_search = 8;      ///< max backtracking line-search steps
  double cost_tol        = 1e-4;   ///< relative cost decrease tolerance
  double mu_init         = 1.0;    ///< initial Hessian regularization
  double mu_factor       = 10.0;   ///< regularization scaling factor
  double mu_max          = 1e6;    ///< max regularization before failure
  double mu_min          = 1e-8;   ///< min regularization (success)
  double alpha_min       = 0.01;   ///< min acceptable step size
};

/// RLS estimator parameters (§3.1).
struct RLSParams {
  double      lambda     = 0.98;      ///< forgetting factor  ∈ (0, 1]
  double      p_init     = 1000.0;    ///< initial covariance diagonal
  ParamVector theta_init =            ///< nominal proxy parameters
      (ParamVector() << 0.95, 0.05, 0.95, 0.05).finished();
};

}  // namespace amr_controller
