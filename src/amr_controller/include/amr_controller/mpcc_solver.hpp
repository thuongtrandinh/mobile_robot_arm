// Copyright 2026 H-AMPCC Authors
// SPDX-License-Identifier: Apache-2.0
//
// iLQR-based solver for the Adaptive MPCC optimisation problem.
//
// Algorithm:  iterative Linear-Quadratic Regulator (iLQR) with
//   - Gauss-Newton cost Hessian approximation
//   - Regularised backward Riccati pass
//   - Backtracking Armijo line search
//   - Box-constraint clamping on control inputs

#pragma once

#include "amr_controller/adaptive_mpcc.hpp"
#include "amr_controller/types.hpp"

#include <Eigen/Dense>
#include <vector>

namespace amr_controller {

class MPCCSolver {
 public:
  // ─────────── Result ───────────
  struct Solution {
    std::vector<AugmentedState>   z_seq;    ///< state trajectory  [0..N]
    std::vector<AugmentedControl> u_seq;    ///< control sequence  [0..N-1]
    double                        cost  = 0.0;
    int                           iters = 0;
    bool                          converged = false;
  };

  // ─────── Construction ─────────
  explicit MPCCSolver(AdaptiveMPCC& mpcc,
                      const SolverParams& params = SolverParams{});

  // ──────── Main API ────────────

  /// Solve the MPCC problem starting from z0, using u_warm as the
  /// initial control-sequence guess (length N).  Returns the optimal
  /// trajectory and controls.
  Solution solve(const AugmentedState& z0,
                 const std::vector<AugmentedControl>& u_warm);

  /// Create a default warm-start control sequence (zero controls
  /// with a positive v_θ to seed progress).
  [[nodiscard]] std::vector<AugmentedControl> defaultWarmStart() const;

  /// Shift a previous solution by one step (warm-start for the
  /// next control cycle).  The last element is duplicated.
  static std::vector<AugmentedControl> shiftSequence(
      const std::vector<AugmentedControl>& u_seq);

 private:
  // ─── Per-stage feedback gains computed in the backward pass ───
  struct StageGains {
    Eigen::Matrix<double, kAugControlDim, kAugStateDim> K =
        Eigen::Matrix<double, kAugControlDim, kAugStateDim>::Zero();
    Eigen::Matrix<double, kAugControlDim, 1> d =
        Eigen::Matrix<double, kAugControlDim, 1>::Zero();
  };

  // ─── Quadratic cost approximation at one stage ───
  struct QuadCost {
    Eigen::Matrix<double, kAugStateDim, kAugStateDim>     Qzz;    // 6×6
    Eigen::Matrix<double, kAugStateDim, 1>                qz;     // 6×1
    Eigen::Matrix<double, kAugControlDim, kAugControlDim> Ruu;    // 3×3
    Eigen::Matrix<double, kAugControlDim, 1>              ru;     // 3×1
  };

  // ─── Internal routines ───

  /// Compute Gauss-Newton quadratic approximation of the stage cost.
  QuadCost quadratizeStageCost(const AugmentedState& z,
                               const AugmentedControl& u) const;

  /// Terminal cost quadratization (same structure, u is unused).
  QuadCost quadratizeTerminalCost(const AugmentedState& z_N) const;

  /// Backward Riccati pass.  Returns true if all Q_uu inversions
  /// succeed (otherwise, increase regularisation).
  bool backwardPass(const std::vector<AugmentedState>& z_seq,
                    const std::vector<AugmentedControl>& u_seq,
                    std::vector<StageGains>& gains,
                    double& expected_reduction);

  /// Forward rollout with feedback:  u = u_ref + α d + K (z − z_ref).
  /// Controls are clamped to box constraints.
  std::pair<std::vector<AugmentedState>, std::vector<AugmentedControl>>
  forwardPass(const AugmentedState& z0,
              const std::vector<AugmentedState>& z_ref,
              const std::vector<AugmentedControl>& u_ref,
              const std::vector<StageGains>& gains,
              double alpha);

  /// Clamp a single control to box constraints.
  AugmentedControl clampControl(const AugmentedControl& u) const;

  // ─── State ───
  AdaptiveMPCC& mpcc_;
  SolverParams  solver_params_;
  double        mu_;   ///< current regularisation coefficient
};

}  // namespace amr_controller
