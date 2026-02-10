// Copyright 2026 H-AMPCC Authors
// SPDX-License-Identifier: Apache-2.0
//
// Recursive Least Squares (RLS) estimator for the proxy-parameter
// vector θ = [α_v, β_v, α_ω, β_ω]ᵀ of a differential-drive robot.
//
// Theory (§3.1):
//   The discrete dynamics can be decomposed into two decoupled
//   scalar models:
//
//     v_{k+1}  = α_v · v_k   + β_v · R_k       (linear subsystem)
//     ω_{k+1}  = α_ω · ω_k   + β_ω · M_k       (angular subsystem)
//
//   Each is of the form  y = φᵀ θ_sub  with dim(θ_sub) = 2,
//   so we run two independent 2×2 RLS filters and expose the
//   concatenated 4-vector through getTheta().

#pragma once

#include "amr_controller/types.hpp"

#include <Eigen/Dense>

namespace amr_controller {

class RLSEstimator {
 public:
  // ────────────────────────── construction ──────────────────────────

  /// Construct with default or custom RLS tuning.
  explicit RLSEstimator(const RLSParams& params = RLSParams{});

  // ──────────────────── core estimation API ────────────────────────

  /// Feed one measurement pair and update θ̂.
  ///
  /// @param v_measured      measured linear velocity  v_{k}  [m/s]
  /// @param omega_measured  measured angular velocity ω_{k}  [rad/s]
  /// @param v_prev          previous linear velocity  v_{k−1}  [m/s]
  /// @param omega_prev      previous angular velocity ω_{k−1} [rad/s]
  /// @param R_prev          previous control force    R_{k−1}  [N]
  /// @param M_prev          previous control torque   M_{k−1}  [N·m]
  ///
  /// Internally executes the RLS recursion (Eqs. in §3.1):
  ///   K  = P φ / (λ + φᵀ P φ)
  ///   ε  = y − φᵀ θ̂
  ///   θ̂ ← θ̂ + K ε
  ///   P ← (1/λ)(P − K φᵀ P)
  void update(double v_measured, double omega_measured,
              double v_prev, double omega_prev,
              double R_prev, double M_prev);

  // ──────────────────────── accessors ──────────────────────────────

  /// Return the concatenated parameter vector [α_v, β_v, α_ω, β_ω]ᵀ.
  [[nodiscard]] ParamVector getTheta() const;

  /// Individual proxy parameters.
  [[nodiscard]] double getAlphaV()     const { return linear_est_.theta(0); }
  [[nodiscard]] double getBetaV()      const { return linear_est_.theta(1); }
  [[nodiscard]] double getAlphaOmega() const { return angular_est_.theta(0); }
  [[nodiscard]] double getBetaOmega()  const { return angular_est_.theta(1); }

  /// Recover physical parameters given sampling time Δt.
  ///   m = Δt / β_v,   b = (1 − α_v) · m / Δt
  ///   J = Δt / β_ω,   c = (1 − α_ω) · J / Δt
  [[nodiscard]] double estimateMass(double dt)      const;
  [[nodiscard]] double estimateInertia(double dt)   const;
  [[nodiscard]] double estimateDampingB(double dt)  const;
  [[nodiscard]] double estimateDampingC(double dt)  const;

  /// Estimation error covariance (for diagnostics / confidence).
  [[nodiscard]] Eigen::Matrix2d getCovarianceLinear()  const;
  [[nodiscard]] Eigen::Matrix2d getCovarianceAngular() const;

  /// Latest innovation (prediction error) for each subsystem.
  [[nodiscard]] double getInnovationLinear()  const { return eps_v_; }
  [[nodiscard]] double getInnovationAngular() const { return eps_omega_; }

  // ────────────────────────── reset ────────────────────────────────

  /// Re-initialise with the same or new parameters.
  void reset();
  void reset(const RLSParams& params);

 private:
  // ─── internal 2×2 sub-estimator ───
  struct SubEstimator {
    Eigen::Vector2d theta;   ///< parameter estimate  [α, β]
    Eigen::Matrix2d P;       ///< estimation covariance
  };

  /// Single RLS step for a 2-parameter subsystem.
  /// Returns the scalar innovation ε.
  double updateSub(SubEstimator& est,
                   double y_measured,
                   const Eigen::Vector2d& phi);

  // ─── state ───
  double         lambda_;         ///< forgetting factor ∈ (0,1]
  SubEstimator   linear_est_;     ///< v-dynamics  {α_v, β_v}
  SubEstimator   angular_est_;    ///< ω-dynamics  {α_ω, β_ω}
  double         eps_v_     = 0.0;
  double         eps_omega_ = 0.0;
  RLSParams      init_params_;    ///< stored for reset()
};

}  // namespace amr_controller
