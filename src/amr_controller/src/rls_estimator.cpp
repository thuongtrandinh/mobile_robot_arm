// Copyright 2026 H-AMPCC Authors
// SPDX-License-Identifier: Apache-2.0

#include "amr_controller/rls_estimator.hpp"

#include <cmath>
#include <stdexcept>

namespace amr_controller {

// ═══════════════════════════════════════════════════════════════════
//  Construction / Reset
// ═══════════════════════════════════════════════════════════════════

RLSEstimator::RLSEstimator(const RLSParams& params)
    : lambda_(params.lambda), init_params_(params) {
  if (lambda_ <= 0.0 || lambda_ > 1.0) {
    throw std::invalid_argument(
        "RLSEstimator: forgetting factor λ must be in (0, 1]");
  }

  // ── Linear-velocity sub-estimator: θ_v = [α_v, β_v]ᵀ ──
  linear_est_.theta << params.theta_init(kAlphaV),
                       params.theta_init(kBetaV);
  linear_est_.P = Eigen::Matrix2d::Identity() * params.p_init;

  // ── Angular-velocity sub-estimator: θ_ω = [α_ω, β_ω]ᵀ ──
  angular_est_.theta << params.theta_init(kAlphaOmega),
                        params.theta_init(kBetaOmega);
  angular_est_.P = Eigen::Matrix2d::Identity() * params.p_init;
}

void RLSEstimator::reset() { reset(init_params_); }

void RLSEstimator::reset(const RLSParams& params) {
  lambda_ = params.lambda;
  if (lambda_ <= 0.0 || lambda_ > 1.0) {
    throw std::invalid_argument(
        "RLSEstimator::reset: forgetting factor λ must be in (0, 1]");
  }
  init_params_ = params;

  linear_est_.theta << params.theta_init(kAlphaV),
                       params.theta_init(kBetaV);
  linear_est_.P = Eigen::Matrix2d::Identity() * params.p_init;

  angular_est_.theta << params.theta_init(kAlphaOmega),
                        params.theta_init(kBetaOmega);
  angular_est_.P = Eigen::Matrix2d::Identity() * params.p_init;

  eps_v_     = 0.0;
  eps_omega_ = 0.0;
}

// ═══════════════════════════════════════════════════════════════════
//  Core RLS update  (§3.1)
// ═══════════════════════════════════════════════════════════════════

void RLSEstimator::update(double v_measured, double omega_measured,
                          double v_prev, double omega_prev,
                          double R_prev, double M_prev) {
  // ── Linear-velocity subsystem ──
  //   y_v = v_{k}              (current measurement)
  //   φ_v = [v_{k−1}, R_{k−1}]ᵀ
  //   model: v_k = α_v · v_{k−1} + β_v · R_{k−1}
  const Eigen::Vector2d phi_v(v_prev, R_prev);
  eps_v_ = updateSub(linear_est_, v_measured, phi_v);

  // ── Angular-velocity subsystem ──
  //   y_ω = ω_{k}
  //   φ_ω = [ω_{k−1}, M_{k−1}]ᵀ
  //   model: ω_k = α_ω · ω_{k−1} + β_ω · M_{k−1}
  const Eigen::Vector2d phi_omega(omega_prev, M_prev);
  eps_omega_ = updateSub(angular_est_, omega_measured, phi_omega);
}

double RLSEstimator::updateSub(SubEstimator& est,
                               double y_measured,
                               const Eigen::Vector2d& phi) {
  // ──── RLS recursion (Eqs. §3.1) ────
  //
  //  K  = P_{k−1} φ / (λ + φᵀ P_{k−1} φ)
  //  ε  = y_k − φᵀ θ̂_{k−1}
  //  θ̂_k = θ̂_{k−1} + K ε
  //  P_k = (1/λ)(P_{k−1} − K φᵀ P_{k−1})

  const double denom = lambda_ + phi.transpose() * est.P * phi;

  // Numerical guard: if denominator is near-zero, skip update.
  if (std::abs(denom) < 1e-12) {
    return 0.0;
  }

  const Eigen::Vector2d K = (est.P * phi) / denom;

  // Innovation (prediction error)
  const double epsilon = y_measured - phi.dot(est.theta);

  // Parameter update
  est.theta += K * epsilon;

  // Covariance update  (Joseph form is more numerically stable,
  // but the standard form suffices with the symmetrisation below)
  est.P = (1.0 / lambda_) * (est.P - K * phi.transpose() * est.P);

  // Symmetrise to suppress floating-point drift
  est.P = 0.5 * (est.P + est.P.transpose());

  return epsilon;
}

// ═══════════════════════════════════════════════════════════════════
//  Accessors
// ═══════════════════════════════════════════════════════════════════

ParamVector RLSEstimator::getTheta() const {
  ParamVector theta;
  theta << linear_est_.theta(0),   // α_v
           linear_est_.theta(1),   // β_v
           angular_est_.theta(0),  // α_ω
           angular_est_.theta(1);  // β_ω
  return theta;
}

// ── Recover physical parameters from proxy (§2.1 inverse map) ──
//
//  α_v = 1 − bΔt/m   →  m = Δt / β_v
//  β_v = Δt / m       →  b = (1 − α_v) · m / Δt
//  (analogously for J, c)

double RLSEstimator::estimateMass(double dt) const {
  const double beta_v = linear_est_.theta(1);
  if (std::abs(beta_v) < 1e-12) {
    return std::numeric_limits<double>::infinity();
  }
  return dt / beta_v;
}

double RLSEstimator::estimateInertia(double dt) const {
  const double beta_omega = angular_est_.theta(1);
  if (std::abs(beta_omega) < 1e-12) {
    return std::numeric_limits<double>::infinity();
  }
  return dt / beta_omega;
}

double RLSEstimator::estimateDampingB(double dt) const {
  const double alpha_v = linear_est_.theta(0);
  const double m       = estimateMass(dt);
  if (std::isinf(m)) {
    return 0.0;
  }
  return (1.0 - alpha_v) * m / dt;
}

double RLSEstimator::estimateDampingC(double dt) const {
  const double alpha_omega = angular_est_.theta(0);
  const double J           = estimateInertia(dt);
  if (std::isinf(J)) {
    return 0.0;
  }
  return (1.0 - alpha_omega) * J / dt;
}

Eigen::Matrix2d RLSEstimator::getCovarianceLinear() const {
  return linear_est_.P;
}

Eigen::Matrix2d RLSEstimator::getCovarianceAngular() const {
  return angular_est_.P;
}

}  // namespace amr_controller
