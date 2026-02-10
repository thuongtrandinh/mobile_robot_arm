// Copyright 2026 H-AMPCC Authors
// SPDX-License-Identifier: Apache-2.0

#include "amr_controller/mpcc_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace amr_controller {

// ═══════════════════════════════════════════════════════════════════
//  Construction
// ═══════════════════════════════════════════════════════════════════

MPCCSolver::MPCCSolver(AdaptiveMPCC& mpcc, const SolverParams& params)
    : mpcc_(mpcc), solver_params_(params), mu_(params.mu_init) {}

// ═══════════════════════════════════════════════════════════════════
//  Warm-start helpers
// ═══════════════════════════════════════════════════════════════════

std::vector<AugmentedControl> MPCCSolver::defaultWarmStart() const {
  const int N = mpcc_.getParams().N;
  AugmentedControl u0;
  u0.setZero();
  u0(kVTheta) = mpcc_.getParams().v_ref;  // seed progress
  return std::vector<AugmentedControl>(N, u0);
}

std::vector<AugmentedControl> MPCCSolver::shiftSequence(
    const std::vector<AugmentedControl>& u_seq) {
  if (u_seq.empty()) return u_seq;
  std::vector<AugmentedControl> shifted(u_seq.begin() + 1, u_seq.end());
  shifted.push_back(u_seq.back());
  return shifted;
}

// ═══════════════════════════════════════════════════════════════════
//  Control clamping
// ═══════════════════════════════════════════════════════════════════

AugmentedControl MPCCSolver::clampControl(const AugmentedControl& u) const {
  const auto& p = mpcc_.getParams();
  AugmentedControl uc;
  uc(kForce)  = std::clamp(u(kForce),  -p.v_cmd_max,    p.v_cmd_max);
  uc(kTorque) = std::clamp(u(kTorque), -p.omega_cmd_max, p.omega_cmd_max);
  uc(kVTheta) = std::clamp(u(kVTheta),  p.v_theta_min,   p.v_theta_max);
  return uc;
}

// ═══════════════════════════════════════════════════════════════════
//  Cost quadratization  (Gauss-Newton approximation)
// ═══════════════════════════════════════════════════════════════════
//
//  Stage cost:
//    ℓ = q_c·e_c² + q_l·e_l² + q_v·(v − v_ref)²
//      + q_u_R·u₀² + q_u_M·u₁² + q_u_vθ·u₂²  − q_θ·u₂·dt
//
//  Gauss-Newton Hessian  Q_zz ≈ 2 Cᵀ W C + velocity term
//  Gradient              q_z  = 2 Cᵀ W e  + velocity term
//  where C = ∂[e_c,e_l]/∂z  (2×6) and W = diag(q_c, q_l)

MPCCSolver::QuadCost MPCCSolver::quadratizeStageCost(
    const AugmentedState& z, const AugmentedControl& u) const {

  const auto& p = mpcc_.getParams();
  QuadCost qc;

  // ── Contouring / Lag Jacobian ──
  const auto jac = mpcc_.computeContouringErrorJacobian(z);
  const Eigen::Matrix<double, 2, kAugStateDim>& C = jac.C;
  const Eigen::Vector2d& e = jac.e;

  Eigen::Matrix2d W;
  W << p.q_c, 0.0,
       0.0,   p.q_l;

  // ── Q_zz = 2 Cᵀ W C + velocity Hessian ──
  qc.Qzz = 2.0 * C.transpose() * W * C;
  qc.Qzz(kV, kV) += 2.0 * p.q_v;          // ∂²(q_v·(v−v_ref)²)/∂v²

  // ── q_z = 2 Cᵀ W e + velocity gradient ──
  qc.qz = 2.0 * C.transpose() * W * e;
  qc.qz(kV) += 2.0 * p.q_v * (z(kV) - p.v_ref);

  // ── R_uu = diag(2 q_u_R, 2 q_u_M, 2 q_u_vθ) ──
  qc.Ruu.setZero();
  qc.Ruu(kForce,  kForce)  = 2.0 * p.q_u_R;
  qc.Ruu(kTorque, kTorque) = 2.0 * p.q_u_M;
  qc.Ruu(kVTheta, kVTheta) = 2.0 * p.q_u_vtheta;

  // ── r_u ──
  qc.ru.setZero();
  qc.ru(kForce)  = 2.0 * p.q_u_R * u(kForce);
  qc.ru(kTorque) = 2.0 * p.q_u_M * u(kTorque);
  qc.ru(kVTheta) = 2.0 * p.q_u_vtheta * u(kVTheta)
                    - p.q_theta * p.dt;   // progress incentive gradient

  return qc;
}

MPCCSolver::QuadCost MPCCSolver::quadratizeTerminalCost(
    const AugmentedState& z_N) const {

  const auto& p = mpcc_.getParams();
  QuadCost qc;

  // Terminal cost = same contouring/lag structure with no input terms
  const auto jac = mpcc_.computeContouringErrorJacobian(z_N);
  const Eigen::Matrix<double, 2, kAugStateDim>& C = jac.C;
  const Eigen::Vector2d& e = jac.e;

  Eigen::Matrix2d W;
  W << p.q_c, 0.0,
       0.0,   p.q_l;

  qc.Qzz = 2.0 * C.transpose() * W * C;
  qc.Qzz(kV, kV) += 2.0 * p.q_v;

  qc.qz = 2.0 * C.transpose() * W * e;
  qc.qz(kV) += 2.0 * p.q_v * (z_N(kV) - p.v_ref);

  qc.Ruu.setZero();
  qc.ru.setZero();

  return qc;
}

// ═══════════════════════════════════════════════════════════════════
//  Backward pass  (Riccati recursion)
// ═══════════════════════════════════════════════════════════════════
//
//  For j = N−1 … 0:
//
//    linearise dynamics:  A_j, B_j
//    quadratise cost:     Q_j, q_j, R_j, r_j
//
//    Q_xx = Q_j  + A_jᵀ V_{j+1} A_j
//    Q_uu = R_j  + B_jᵀ V_{j+1} B_j  + μI   (regularisation)
//    Q_ux =        B_jᵀ V_{j+1} A_j
//    Q_x  = q_j  + A_jᵀ p_{j+1}
//    Q_u  = r_j  + B_jᵀ p_{j+1}
//
//    K_j  = −Q_uu⁻¹ Q_ux
//    d_j  = −Q_uu⁻¹ Q_u
//
//    V_j  = Q_xx + Q_uxᵀ K_j    (= Q_xx − Q_uxᵀ Q_uu⁻¹ Q_ux)
//    p_j  = Q_x  + Q_uxᵀ d_j

bool MPCCSolver::backwardPass(
    const std::vector<AugmentedState>& z_seq,
    const std::vector<AugmentedControl>& u_seq,
    std::vector<StageGains>& gains,
    double& expected_reduction) {

  const int N = static_cast<int>(u_seq.size());
  gains.resize(N);
  expected_reduction = 0.0;

  // Terminal value function from terminal cost
  const QuadCost term = quadratizeTerminalCost(z_seq[N]);
  Eigen::Matrix<double, kAugStateDim, kAugStateDim> V = term.Qzz;
  Eigen::Matrix<double, kAugStateDim, 1>             p = term.qz;

  for (int j = N - 1; j >= 0; --j) {
    // Linearise dynamics
    const auto model = mpcc_.getLinearizedModel(z_seq[j], u_seq[j]);
    const auto& A = model.A;
    const auto& B = model.B;

    // Quadratise stage cost
    const QuadCost sc = quadratizeStageCost(z_seq[j], u_seq[j]);

    // Q-function matrices
    using MatSS = Eigen::Matrix<double, kAugStateDim, kAugStateDim>;
    using MatCS = Eigen::Matrix<double, kAugControlDim, kAugStateDim>;
    using MatCC = Eigen::Matrix<double, kAugControlDim, kAugControlDim>;
    using VecS  = Eigen::Matrix<double, kAugStateDim, 1>;
    using VecC  = Eigen::Matrix<double, kAugControlDim, 1>;

    const MatSS Q_xx = sc.Qzz + A.transpose() * V * A;
    MatCC       Q_uu = MatCC(sc.Ruu + B.transpose() * V * B);
    const MatCS Q_ux =          B.transpose() * V * A;
    const VecS  Q_x  = sc.qz  + A.transpose() * p;
    const VecC  Q_u  = sc.ru  + B.transpose() * p;

    // Regularise Q_uu
    for (int i = 0; i < kAugControlDim; ++i) {
      Q_uu(i, i) += mu_;
    }

    // Check positive definiteness by trying Cholesky
    Eigen::LLT<Eigen::Matrix<double, kAugControlDim, kAugControlDim>> llt(Q_uu);
    if (llt.info() != Eigen::Success) {
      return false;  // caller should increase μ
    }

    // Feedback gains
    const auto Q_uu_inv = llt.solve(
        Eigen::Matrix<double, kAugControlDim, kAugControlDim>::Identity());

    gains[j].K = -Q_uu_inv * Q_ux;
    gains[j].d = -Q_uu_inv * Q_u;

    // Expected cost reduction  (ΔJ ≈ α dᵀ Q_u + ½ α² dᵀ Q_uu d)
    expected_reduction += gains[j].d.dot(Q_u);

    // Value function update
    V = Q_xx + Q_ux.transpose() * gains[j].K;
    V = 0.5 * (V + V.transpose());   // symmetrise
    p = Q_x  + Q_ux.transpose() * gains[j].d;
  }

  return true;
}

// ═══════════════════════════════════════════════════════════════════
//  Forward pass  (rollout with feedback + clamping)
// ═══════════════════════════════════════════════════════════════════

std::pair<std::vector<AugmentedState>, std::vector<AugmentedControl>>
MPCCSolver::forwardPass(
    const AugmentedState& z0,
    const std::vector<AugmentedState>& z_ref,
    const std::vector<AugmentedControl>& u_ref,
    const std::vector<StageGains>& gains,
    double alpha) {

  const int N = static_cast<int>(u_ref.size());
  std::vector<AugmentedState>   z_new;
  std::vector<AugmentedControl> u_new;
  z_new.reserve(N + 1);
  u_new.reserve(N);

  z_new.push_back(z0);

  for (int j = 0; j < N; ++j) {
    // State deviation
    AugmentedState dz = z_new[j] - z_ref[j];

    // Wrap heading difference to [−π, π]
    dz(kPsi) = std::atan2(std::sin(dz(kPsi)), std::cos(dz(kPsi)));

    // Feedback law:  u = u_ref + α·d + K·δz
    AugmentedControl u_j = u_ref[j] + alpha * gains[j].d + gains[j].K * dz;

    // Clamp to box constraints
    u_j = clampControl(u_j);
    u_new.push_back(u_j);

    // Propagate nonlinear dynamics
    z_new.push_back(mpcc_.predictState(z_new.back(), u_j));
  }

  return {z_new, u_new};
}

// ═══════════════════════════════════════════════════════════════════
//  Main solve loop
// ═══════════════════════════════════════════════════════════════════

MPCCSolver::Solution MPCCSolver::solve(
    const AugmentedState& z0,
    const std::vector<AugmentedControl>& u_warm) {

  const int N = mpcc_.getParams().N;

  Solution sol;
  sol.u_seq = u_warm;

  // Ensure correct horizon length
  if (static_cast<int>(sol.u_seq.size()) != N) {
    sol.u_seq = defaultWarmStart();
  }

  // Clamp initial controls
  for (auto& u : sol.u_seq) {
    u = clampControl(u);
  }

  // Initial forward rollout
  sol.z_seq = mpcc_.predictHorizon(z0, sol.u_seq);
  sol.cost  = mpcc_.computeTotalCost(sol.z_seq, sol.u_seq);

  mu_ = solver_params_.mu_init;

  for (int iter = 0; iter < solver_params_.max_iters; ++iter) {
    sol.iters = iter + 1;

    // ── Backward pass (with regularisation retry) ──
    std::vector<StageGains> gains;
    double expected_reduction = 0.0;
    bool bp_ok = false;

    while (mu_ <= solver_params_.mu_max) {
      bp_ok = backwardPass(sol.z_seq, sol.u_seq, gains, expected_reduction);
      if (bp_ok) break;
      mu_ *= solver_params_.mu_factor;
    }

    if (!bp_ok) {
      // Regularisation saturated — return best found
      break;
    }

    // ── Forward pass with Armijo line search ──
    bool accepted = false;
    double alpha = 1.0;

    for (int ls = 0; ls < solver_params_.max_line_search; ++ls) {
      auto [z_try, u_try] = forwardPass(z0, sol.z_seq, sol.u_seq,
                                         gains, alpha);

      const double J_try = mpcc_.computeTotalCost(z_try, u_try);
      const double armijo = 1e-4 * alpha * expected_reduction;

      if (J_try < sol.cost + armijo) {
        // Accept the step
        const double rel_improvement =
            std::abs(sol.cost - J_try) /
            (std::abs(sol.cost) + 1e-12);

        sol.z_seq = std::move(z_try);
        sol.u_seq = std::move(u_try);
        sol.cost  = J_try;
        accepted  = true;

        // Check convergence
        if (rel_improvement < solver_params_.cost_tol) {
          sol.converged = true;
          // Decrease regularisation for next call
          mu_ = std::max(mu_ / solver_params_.mu_factor,
                         solver_params_.mu_min);
          return sol;
        }

        // Decrease regularisation on success
        mu_ = std::max(mu_ / solver_params_.mu_factor,
                       solver_params_.mu_min);
        break;
      }

      alpha *= 0.5;
      if (alpha < solver_params_.alpha_min) break;
    }

    if (!accepted) {
      // Line search failed — increase regularisation and retry
      mu_ *= solver_params_.mu_factor;
      if (mu_ > solver_params_.mu_max) break;
    }
  }

  return sol;
}

}  // namespace amr_controller
