// Copyright 2026 H-AMPCC Authors
// SPDX-License-Identifier: Apache-2.0
//
// Adaptive Model Predictive Contouring Control (AMPCC) backend.
//
// This class combines:
//   • RLS-updated dynamics model  (§3.1)
//   • Linearized augmented system A(θ), B(θ)
//   • Contouring / lag error decomposition in the Frenet frame (§2.3)
//   • Stage & horizon cost evaluation  (§3.2)
//   • Nonlinear forward prediction
//   • Error Jacobian for iLQR-based solver

#pragma once

#include "amr_controller/rls_estimator.hpp"
#include "amr_controller/types.hpp"

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace amr_controller {

// ═══════════════════════════════════════════════════════════════════
//  Reference-path abstraction
// ═══════════════════════════════════════════════════════════════════

/// Pure interface: any curve parameterised by arc-length θ ∈ [0, L].
class ReferencePath {
 public:
  virtual ~ReferencePath() = default;

  /// Evaluate position, tangent angle, and curvature at arc-length θ.
  [[nodiscard]] virtual PathPoint evaluate(double theta) const = 0;

  /// Total arc-length of the path.
  [[nodiscard]] virtual double getLength() const = 0;

  /// Project (x,y) onto the path, returning the arc-length of the
  /// closest point.  @p theta_hint speeds up local search.
  [[nodiscard]] virtual double projectOntoPath(
      double x, double y, double theta_hint = 0.0) const = 0;
};

// ─────────────────────────────────────────────────────────────────
/// Concrete implementation: piecewise-linear path through waypoints.
/// Useful for quick prototyping & unit tests; replace with a cubic
/// spline for production contouring quality.
// ─────────────────────────────────────────────────────────────────
class WaypointPath : public ReferencePath {
 public:
  WaypointPath() = default;

  /// Build the path from an ordered list of 2-D waypoints.
  explicit WaypointPath(const std::vector<Eigen::Vector2d>& waypoints);

  [[nodiscard]] PathPoint evaluate(double theta) const override;
  [[nodiscard]] double    getLength() const override;
  [[nodiscard]] double    projectOntoPath(
      double x, double y, double theta_hint = 0.0) const override;

  /// Replace the path on the fly (e.g. when the GNN outputs new spline params).
  void setWaypoints(const std::vector<Eigen::Vector2d>& waypoints);

 private:
  struct Segment {
    double          s_start;   ///< cumulative arc-length at segment start
    double          length;    ///< segment length
    Eigen::Vector2d p0;        ///< start point
    Eigen::Vector2d p1;        ///< end point
    double          phi;       ///< tangent angle  atan2(Δy, Δx)
  };

  std::vector<Segment> segments_;
  double               total_length_ = 0.0;
};

// ═══════════════════════════════════════════════════════════════════
//  Adaptive MPCC  (§3.2 – Unified Optimisation Problem)
// ═══════════════════════════════════════════════════════════════════

class AdaptiveMPCC {
 public:
  // ────────────────── construction ──────────────────

  explicit AdaptiveMPCC(const MPCCParams& mpcc_params = MPCCParams{},
                        const RLSParams&  rls_params  = RLSParams{});

  // ────────────────── path ──────────────────────────

  /// Assign / replace the reference path.
  void setReferencePath(std::shared_ptr<ReferencePath> path);

  /// Check whether a reference path has been set.
  [[nodiscard]] bool hasReferencePath() const;

  // ────────────────── RLS model update ──────────────

  /// Convenience wrapper: feeds one sensor pair into the RLS filter.
  void updateModel(double v_meas, double omega_meas,
                   double v_prev, double omega_prev,
                   double R_prev, double M_prev);

  // ──────── Linearized model  A(θ), B(θ) ───────────

  /// Build the 6×6 Jacobian A and 6×3 input matrix B of the
  /// augmented system, linearised about (z_ref, u_ref) using
  /// the *current* RLS estimate θ̂.
  ///
  /// Augmented state  z = [x, y, ψ, v, ω, θ_progress]ᵀ
  /// Augmented input  u = [R, M, v_θ]ᵀ
  [[nodiscard]] LinearizedModel getLinearizedModel(
      const AugmentedState&   z_ref,
      const AugmentedControl& u_ref) const;

  /// Same, but with an *explicit* parameter vector (e.g. for
  /// sensitivity analysis or warm-start from a different θ̂).
  [[nodiscard]] LinearizedModel getLinearizedModel(
      const AugmentedState&   z_ref,
      const AugmentedControl& u_ref,
      const ParamVector&      theta) const;

  // ──────── Contouring / Lag errors  (§2.3) ────────

  /// Compute (e_c, e_l) for the augmented state z (reads θ_progress
  /// from z[kTheta] and evaluates the reference path there).
  [[nodiscard]] ContouringErrors computeContouringErrors(
      const AugmentedState& z) const;

  /// Lower-level overload when you already know (x, y, θ_progress).
  [[nodiscard]] ContouringErrors computeContouringErrors(
      double x, double y, double theta_progress) const;

  /// Compute the 2×6 Jacobian ∂[e_c,e_l]/∂z together with the error
  /// values.  Used by the iLQR solver for cost quadratization.
  [[nodiscard]] ContouringErrorJacobian computeContouringErrorJacobian(
      const AugmentedState& z) const;

  /// Project robot position onto the reference path.
  [[nodiscard]] double projectOntoPath(
      double x, double y, double theta_hint = 0.0) const;

  // ──────── Nonlinear forward model ────────────────

  /// One-step discrete prediction using the full nonlinear dynamics
  /// and the *current* RLS parameters θ̂.
  [[nodiscard]] AugmentedState predictState(
      const AugmentedState&   z,
      const AugmentedControl& u) const;

  /// Overload with an explicit θ.
  [[nodiscard]] AugmentedState predictState(
      const AugmentedState&   z,
      const AugmentedControl& u,
      const ParamVector&      theta) const;

  /// Roll out the nonlinear model over a control sequence.
  [[nodiscard]] std::vector<AugmentedState> predictHorizon(
      const AugmentedState&                z0,
      const std::vector<AugmentedControl>& u_seq) const;

  // ──────── Cost evaluation  (§3.2) ────────────────

  /// Stage cost at one time step (includes progress incentive).
  [[nodiscard]] double computeStageCost(
      const AugmentedState&   z,
      const AugmentedControl& u) const;

  /// Total cost J over an entire (state, control) trajectory.
  [[nodiscard]] double computeTotalCost(
      const std::vector<AugmentedState>&   z_seq,
      const std::vector<AugmentedControl>& u_seq) const;

  // ──────── Accessors ──────────────────────────────

  [[nodiscard]] ParamVector           getCurrentTheta() const;
  [[nodiscard]] const MPCCParams&     getParams()       const { return params_; }
  [[nodiscard]] const RLSEstimator&   getEstimator()    const { return rls_; }
  [[nodiscard]] RLSEstimator&         getEstimatorMut()       { return rls_; }

 private:
  MPCCParams                     params_;
  RLSEstimator                   rls_;
  std::shared_ptr<ReferencePath> ref_path_;
};

}  // namespace amr_controller
