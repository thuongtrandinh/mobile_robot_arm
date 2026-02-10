// Copyright 2026 H-AMPCC Authors
// SPDX-License-Identifier: Apache-2.0

#include "amr_controller/adaptive_mpcc.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace amr_controller {

// ╔═══════════════════════════════════════════════════════════════╗
// ║  WaypointPath – piecewise-linear reference path              ║
// ╚═══════════════════════════════════════════════════════════════╝

WaypointPath::WaypointPath(const std::vector<Eigen::Vector2d>& waypoints) {
  setWaypoints(waypoints);
}

void WaypointPath::setWaypoints(
    const std::vector<Eigen::Vector2d>& waypoints) {
  segments_.clear();
  total_length_ = 0.0;

  if (waypoints.size() < 2) {
    return;  // degenerate – evaluate() will return the origin
  }

  segments_.reserve(waypoints.size() - 1);
  for (std::size_t i = 0; i + 1 < waypoints.size(); ++i) {
    Segment seg;
    seg.p0      = waypoints[i];
    seg.p1      = waypoints[i + 1];
    seg.s_start = total_length_;

    const Eigen::Vector2d d = seg.p1 - seg.p0;
    seg.length = d.norm();
    seg.phi    = std::atan2(d.y(), d.x());

    total_length_ += seg.length;
    segments_.push_back(seg);
  }
}

PathPoint WaypointPath::evaluate(double theta) const {
  PathPoint pt;

  if (segments_.empty()) {
    return pt;  // all zeros
  }

  // Clamp θ to [0, L]
  theta = std::clamp(theta, 0.0, total_length_);

  // Binary search for the segment that contains θ
  // (segments are sorted by s_start by construction)
  std::size_t lo = 0;
  std::size_t hi = segments_.size() - 1;

  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo + 1) / 2;
    if (segments_[mid].s_start <= theta) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }

  const Segment& seg = segments_[lo];
  const double   t   = (seg.length > 1e-12)
                            ? (theta - seg.s_start) / seg.length
                            : 0.0;

  const Eigen::Vector2d p = seg.p0 + t * (seg.p1 - seg.p0);
  pt.x     = p.x();
  pt.y     = p.y();
  pt.phi   = seg.phi;
  pt.kappa = 0.0;  // piecewise-linear → zero curvature within segments

  return pt;
}

double WaypointPath::getLength() const { return total_length_; }

double WaypointPath::projectOntoPath(
    double x, double y, double theta_hint) const {
  if (segments_.empty()) {
    return 0.0;
  }

  // Clamp hint to valid range
  theta_hint = std::clamp(theta_hint, 0.0, total_length_);

  double best_theta = theta_hint;
  double best_dist2 = std::numeric_limits<double>::max();

  // Search all segments (for short paths this is fast enough;
  // for long paths, narrow the search window around theta_hint)
  for (const auto& seg : segments_) {
    // Project (x,y) onto the segment line
    const Eigen::Vector2d p(x, y);
    const Eigen::Vector2d d = seg.p1 - seg.p0;
    const double seg_len2 = d.squaredNorm();

    double t = 0.0;
    if (seg_len2 > 1e-12) {
      t = std::clamp((p - seg.p0).dot(d) / seg_len2, 0.0, 1.0);
    }

    const Eigen::Vector2d proj = seg.p0 + t * d;
    const double dist2 = (p - proj).squaredNorm();

    if (dist2 < best_dist2) {
      best_dist2 = dist2;
      best_theta = seg.s_start + t * seg.length;
    }
  }

  return best_theta;
}

// ╔═══════════════════════════════════════════════════════════════╗
// ║  AdaptiveMPCC                                                ║
// ╚═══════════════════════════════════════════════════════════════╝

// ─────────────── Construction ───────────────────────────────────

AdaptiveMPCC::AdaptiveMPCC(const MPCCParams& mpcc_params,
                           const RLSParams&  rls_params)
    : params_(mpcc_params), rls_(rls_params) {}

// ─────────────── Path management ────────────────────────────────

void AdaptiveMPCC::setReferencePath(std::shared_ptr<ReferencePath> path) {
  ref_path_ = std::move(path);
}

bool AdaptiveMPCC::hasReferencePath() const {
  return ref_path_ != nullptr;
}

// ─────────────── RLS convenience ────────────────────────────────

void AdaptiveMPCC::updateModel(double v_meas, double omega_meas,
                               double v_prev, double omega_prev,
                               double R_prev, double M_prev) {
  rls_.update(v_meas, omega_meas, v_prev, omega_prev, R_prev, M_prev);
}

ParamVector AdaptiveMPCC::getCurrentTheta() const {
  return rls_.getTheta();
}

// ═══════════════════════════════════════════════════════════════════
//  Linearized Model  A(θ), B(θ)   (§2.1 + augmented progress)
// ═══════════════════════════════════════════════════════════════════
//
//  Augmented state   z = [x, y, ψ, v, ω, θ_prog]ᵀ   (6×1)
//  Augmented control u = [R, M, v_θ]ᵀ                (3×1)
//
//  Nonlinear dynamics:
//    x⁺      = x   + v cos(ψ) Δt
//    y⁺      = y   + v sin(ψ) Δt
//    ψ⁺      = ψ   + ω Δt
//    v⁺      = α_v  v  + β_v  R
//    ω⁺      = α_ω  ω  + β_ω  M
//    θ_prog⁺ = θ_prog + v_θ Δt
//
//  Jacobian  A = ∂f/∂z  evaluated at (z_ref, u_ref):
//
//    ┌ 1   0   −v sin(ψ) Δt   cos(ψ) Δt   0      0 ┐
//    │ 0   1    v cos(ψ) Δt   sin(ψ) Δt   0      0 │
//    │ 0   0    1              0            Δt     0 │
//    │ 0   0    0              α_v          0      0 │
//    │ 0   0    0              0            α_ω    0 │
//    └ 0   0    0              0            0      1 ┘
//
//  Jacobian  B = ∂f/∂u:
//
//    ┌ 0      0      0  ┐
//    │ 0      0      0  │
//    │ 0      0      0  │
//    │ β_v    0      0  │
//    │ 0      β_ω    0  │
//    └ 0      0      Δt ┘

LinearizedModel AdaptiveMPCC::getLinearizedModel(
    const AugmentedState&   z_ref,
    const AugmentedControl& u_ref) const {
  return getLinearizedModel(z_ref, u_ref, rls_.getTheta());
}

LinearizedModel AdaptiveMPCC::getLinearizedModel(
    const AugmentedState&   z_ref,
    const AugmentedControl& /*u_ref*/,
    const ParamVector&      theta) const {

  const double dt = params_.dt;

  // Extract operating-point state
  const double psi = z_ref(kPsi);
  const double v   = z_ref(kV);

  // Extract proxy parameters
  const double alpha_v     = theta(kAlphaV);
  const double beta_v      = theta(kBetaV);
  const double alpha_omega = theta(kAlphaOmega);
  const double beta_omega  = theta(kBetaOmega);

  // Precompute trigonometric terms
  const double cos_psi = std::cos(psi);
  const double sin_psi = std::sin(psi);

  // ── A matrix (6×6) ──
  LinearizedModel model;
  model.A.setIdentity();   // diagonal = 1 by default

  //  Row 0 (x):  ∂x⁺/∂ψ = −v sin(ψ) Δt,  ∂x⁺/∂v = cos(ψ) Δt
  model.A(kX, kPsi) = -v * sin_psi * dt;
  model.A(kX, kV)   =  cos_psi * dt;

  //  Row 1 (y):  ∂y⁺/∂ψ =  v cos(ψ) Δt,  ∂y⁺/∂v = sin(ψ) Δt
  model.A(kY, kPsi) =  v * cos_psi * dt;
  model.A(kY, kV)   =  sin_psi * dt;

  //  Row 2 (ψ):  ∂ψ⁺/∂ω = Δt
  model.A(kPsi, kOmega) = dt;

  //  Row 3 (v):  ∂v⁺/∂v = α_v
  model.A(kV, kV) = alpha_v;

  //  Row 4 (ω):  ∂ω⁺/∂ω = α_ω
  model.A(kOmega, kOmega) = alpha_omega;

  //  Row 5 (θ_prog): identity — already set

  // ── B matrix (6×3) ──
  model.B.setZero();

  //  Row 3 (v):  ∂v⁺/∂R = β_v
  model.B(kV, kForce) = beta_v;

  //  Row 4 (ω):  ∂ω⁺/∂M = β_ω
  model.B(kOmega, kTorque) = beta_omega;

  //  Row 5 (θ_prog):  ∂θ⁺/∂v_θ = Δt
  model.B(kTheta, kVTheta) = dt;

  return model;
}

// ═══════════════════════════════════════════════════════════════════
//  Contouring & Lag errors  (§2.3 – Frenet-frame decomposition)
// ═══════════════════════════════════════════════════════════════════
//
//  Given the robot position (x, y) and the closest reference point
//  p_ref(θ) = (x_ref, y_ref) with tangent angle φ(θ):
//
//    e_c =  sin(φ) · (x − x_ref) − cos(φ) · (y − y_ref)
//    e_l = −cos(φ) · (x − x_ref) − sin(φ) · (y − y_ref)
//
//  e_c measures orthogonal deviation (contouring);
//  e_l measures tangential lag.

ContouringErrors AdaptiveMPCC::computeContouringErrors(
    const AugmentedState& z) const {
  return computeContouringErrors(z(kX), z(kY), z(kTheta));
}

ContouringErrors AdaptiveMPCC::computeContouringErrors(
    double x, double y, double theta_progress) const {
  if (!ref_path_) {
    throw std::runtime_error(
        "AdaptiveMPCC::computeContouringErrors: no reference path set");
  }

  // Evaluate the reference path at the current progress variable θ
  const PathPoint ref = ref_path_->evaluate(theta_progress);

  const double dx = x - ref.x;
  const double dy = y - ref.y;

  const double cos_phi = std::cos(ref.phi);
  const double sin_phi = std::sin(ref.phi);

  ContouringErrors err;
  err.e_c =  sin_phi * dx - cos_phi * dy;   // perpendicular (contouring)
  err.e_l = -cos_phi * dx - sin_phi * dy;   // tangential    (lag)
  return err;
}

// ═════════════════════════════════════════════════════════════════
//  Contouring Error Jacobian  (§2.3 – Frenet-frame derivatives)
// ═════════════════════════════════════════════════════════════════
//
//  C = ∂[e_c, e_l]ᵀ / ∂z   (2×6 matrix)
//
//  Row 0 (e_c):  [ sinφ,  −cosφ,  0,  0,  0,  −κ·e_l        ]
//  Row 1 (e_l):  [−cosφ,  −sinφ,  0,  0,  0,   κ·e_c + 1    ]
//
//  For piecewise-linear paths κ = 0, so the θ-derivatives simplify:
//    ∂e_c/∂θ = 0,   ∂e_l/∂θ = 1

ContouringErrorJacobian AdaptiveMPCC::computeContouringErrorJacobian(
    const AugmentedState& z) const {
  if (!ref_path_) {
    throw std::runtime_error(
        "AdaptiveMPCC::computeContouringErrorJacobian: no reference path set");
  }

  const double x         = z(kX);
  const double y         = z(kY);
  const double theta_prg = z(kTheta);

  const PathPoint ref = ref_path_->evaluate(theta_prg);

  const double dx = x - ref.x;
  const double dy = y - ref.y;

  const double cos_phi = std::cos(ref.phi);
  const double sin_phi = std::sin(ref.phi);
  const double kappa   = ref.kappa;

  ContouringErrorJacobian jac;

  // Error values
  jac.e(0) =  sin_phi * dx - cos_phi * dy;   // e_c
  jac.e(1) = -cos_phi * dx - sin_phi * dy;   // e_l

  // Jacobian C (2×6): ∂[e_c, e_l]/∂[x, y, ψ, v, ω, θ]
  jac.C.setZero();

  // ∂e_c/∂x = sin(φ),   ∂e_c/∂y = −cos(φ)
  jac.C(0, kX) =  sin_phi;
  jac.C(0, kY) = -cos_phi;
  // ∂e_c/∂θ = −κ·e_l
  jac.C(0, kTheta) = -kappa * jac.e(1);

  // ∂e_l/∂x = −cos(φ),  ∂e_l/∂y = −sin(φ)
  jac.C(1, kX) = -cos_phi;
  jac.C(1, kY) = -sin_phi;
  // ∂e_l/∂θ = κ·e_c + 1
  jac.C(1, kTheta) = kappa * jac.e(0) + 1.0;

  return jac;
}

double AdaptiveMPCC::projectOntoPath(
    double x, double y, double theta_hint) const {
  if (!ref_path_) {
    throw std::runtime_error(
        "AdaptiveMPCC::projectOntoPath: no reference path set");
  }
  return ref_path_->projectOntoPath(x, y, theta_hint);
}

// ═══════════════════════════════════════════════════════════════════
//  Nonlinear forward prediction
// ═══════════════════════════════════════════════════════════════════

AugmentedState AdaptiveMPCC::predictState(
    const AugmentedState&   z,
    const AugmentedControl& u) const {
  return predictState(z, u, rls_.getTheta());
}

AugmentedState AdaptiveMPCC::predictState(
    const AugmentedState&   z,
    const AugmentedControl& u,
    const ParamVector&      theta) const {

  const double dt = params_.dt;

  const double x         = z(kX);
  const double y         = z(kY);
  const double psi       = z(kPsi);
  const double v         = z(kV);
  const double omega     = z(kOmega);
  const double theta_prg = z(kTheta);

  const double R   = u(kForce);
  const double M   = u(kTorque);
  const double v_t = u(kVTheta);

  const double alpha_v     = theta(kAlphaV);
  const double beta_v      = theta(kBetaV);
  const double alpha_omega = theta(kAlphaOmega);
  const double beta_omega  = theta(kBetaOmega);

  AugmentedState z_next;
  z_next(kX)     = x   + v * std::cos(psi) * dt;
  z_next(kY)     = y   + v * std::sin(psi) * dt;
  z_next(kPsi)   = psi + omega * dt;
  z_next(kV)     = alpha_v     * v     + beta_v     * R;
  z_next(kOmega) = alpha_omega * omega + beta_omega * M;
  z_next(kTheta) = theta_prg  + v_t * dt;

  // Normalise heading to [−π, π]
  z_next(kPsi) = std::atan2(std::sin(z_next(kPsi)),
                             std::cos(z_next(kPsi)));

  return z_next;
}

std::vector<AugmentedState> AdaptiveMPCC::predictHorizon(
    const AugmentedState&                z0,
    const std::vector<AugmentedControl>& u_seq) const {

  const auto N = static_cast<int>(u_seq.size());
  std::vector<AugmentedState> z_seq;
  z_seq.reserve(N + 1);
  z_seq.push_back(z0);

  for (int j = 0; j < N; ++j) {
    z_seq.push_back(predictState(z_seq.back(), u_seq[j]));
  }
  return z_seq;
}

// ═══════════════════════════════════════════════════════════════════
//  Cost evaluation  (§3.2)
// ═══════════════════════════════════════════════════════════════════
//
//  Stage cost:
//    ℓ(z, u) = q_c · e_c²  +  q_l · e_l²
//            + q_u_R · v_cmd²  +  q_u_M · ω_cmd²
//            + q_v · (v − v_ref)²
//            + q_u_vtheta · v_θ²  −  q_theta · v_θ · dt   (progress incentive)
//
//  Total cost:
//    J = Σ_{j=0}^{N−1}  ℓ(z_j, u_j)

double AdaptiveMPCC::computeStageCost(
    const AugmentedState&   z,
    const AugmentedControl& u) const {

  // Contouring / lag
  const ContouringErrors err = computeContouringErrors(z);

  // Velocity tracking
  const double v_err = z(kV) - params_.v_ref;

  // Input
  const double v_cmd     = u(kForce);
  const double omega_cmd = u(kTorque);
  const double v_theta   = u(kVTheta);

  return params_.q_c       * err.e_c * err.e_c
       + params_.q_l       * err.e_l * err.e_l
       + params_.q_u_R     * v_cmd * v_cmd
       + params_.q_u_M     * omega_cmd * omega_cmd
       + params_.q_v       * v_err * v_err
       + params_.q_u_vtheta * v_theta * v_theta
       - params_.q_theta   * v_theta * params_.dt;   // progress incentive
}

double AdaptiveMPCC::computeTotalCost(
    const std::vector<AugmentedState>&   z_seq,
    const std::vector<AugmentedControl>& u_seq) const {

  if (z_seq.size() != u_seq.size() + 1) {
    throw std::invalid_argument(
        "AdaptiveMPCC::computeTotalCost: "
        "|z_seq| must equal |u_seq| + 1");
  }

  double J = 0.0;
  for (std::size_t j = 0; j < u_seq.size(); ++j) {
    J += computeStageCost(z_seq[j], u_seq[j]);
  }
  return J;
}

}  // namespace amr_controller
