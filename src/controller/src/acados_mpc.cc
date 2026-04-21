#include "acados_mpc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace robot_plann {
namespace {

inline int SafePathIndex(int k, int path_size) {
  if (path_size <= 0) {
    return 0;
  }
  return std::max(0, std::min(k, path_size - 1));
}

double HeadingAt(const std::vector<robot_plann::Point> &path, int k, double fallback) {
  const int n = static_cast<int>(path.size());
  if (n < 2) {
    return fallback;
  }

  const int idx = SafePathIndex(k, n);
  if (idx < n - 1) {
    const double dx = path[idx + 1].x - path[idx].x;
    const double dy = path[idx + 1].y - path[idx].y;
    if (std::hypot(dx, dy) > 1e-9) {
      return std::atan2(dy, dx);
    }
  }

  if (idx > 0) {
    const double dx = path[idx].x - path[idx - 1].x;
    const double dy = path[idx].y - path[idx - 1].y;
    if (std::hypot(dx, dy) > 1e-9) {
      return std::atan2(dy, dx);
    }
  }

  return fallback;
}

MpcParams::Ptr BuildDefaultParams() {
  auto params = std::make_shared<MpcParams>();
  params->dt = 0.1;
  params->np = static_cast<uint16_t>(DDMR_N);
  params->max_linear_vel = 1.0;
  params->max_linear_acc = 1.0;
  params->max_angular_vel = 1.0;
  params->max_angular_acc = 1.0;
  params->wheel_half_track = 0.3;
  params->local_obst_num = 8;
  return params;
}

}  // namespace

AcadosMpc::AcadosMpc(int verbose)
    : acados_capsule_(nullptr),
      nlp_config_(nullptr),
      nlp_dims_(nullptr),
      nlp_in_(nullptr),
      nlp_out_(nullptr),
      nlp_solver_(nullptr),
      nlp_opts_(nullptr),
      params_(BuildDefaultParams()),
      verbose_(verbose),
      initialized_(false) {
  acados_capsule_ = ddmr_acados_create_capsule();
  if (!acados_capsule_) {
    std::cerr << "[AcadosMpc] Failed to create ACADOS capsule." << std::endl;
    return;
  }

  const int create_status = ddmr_acados_create(acados_capsule_);
  if (create_status != 0) {
    std::cerr << "[AcadosMpc] ddmr_acados_create failed, status=" << create_status
              << std::endl;
    ddmr_acados_free_capsule(acados_capsule_);
    acados_capsule_ = nullptr;
    return;
  }

  nlp_config_ = ddmr_acados_get_nlp_config(acados_capsule_);
  nlp_dims_ = ddmr_acados_get_nlp_dims(acados_capsule_);
  nlp_in_ = ddmr_acados_get_nlp_in(acados_capsule_);
  nlp_out_ = ddmr_acados_get_nlp_out(acados_capsule_);
  nlp_solver_ = ddmr_acados_get_nlp_solver(acados_capsule_);
  nlp_opts_ = ddmr_acados_get_nlp_opts(acados_capsule_);
  initialized_ = (nlp_config_ && nlp_dims_ && nlp_in_ && nlp_out_ && nlp_solver_ && nlp_opts_);

  if (!initialized_) {
    std::cerr << "[AcadosMpc] Failed to fetch ACADOS NLP pointers." << std::endl;
  }
}

AcadosMpc::~AcadosMpc() {
  if (acados_capsule_) {
    ddmr_acados_free(acados_capsule_);
    ddmr_acados_free_capsule(acados_capsule_);
    acados_capsule_ = nullptr;
  }
}

void AcadosMpc::SetParams(const MpcParams::Ptr &params) {
  if (params) {
    params_ = std::make_shared<MpcParams>(*params);
  }
}

bool AcadosMpc::UpdateRuntimeWeightsAndConstraints() {
  if (!initialized_ || !params_) {
    return false;
  }

  std::array<double, DDMR_NY * DDMR_NY> w{};
  w[0 + DDMR_NY * 0] = params_->weights.w_x;
  w[1 + DDMR_NY * 1] = params_->weights.w_y;
  w[2 + DDMR_NY * 2] = params_->weights.w_theta;
  w[3 + DDMR_NY * 3] = params_->weights.w_v;
  w[4 + DDMR_NY * 4] = params_->weights.w_r;
  w[5 + DDMR_NY * 5] = params_->weights.w_acc;
  w[6 + DDMR_NY * 6] = params_->weights.w_dr;

  for (int k = 0; k < DDMR_N; ++k) {
    ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, k, "W", w.data());
  }

  std::array<double, DDMR_NYN * DDMR_NYN> w_e{};
  w_e[0 + DDMR_NYN * 0] = params_->weights.w_x_e;
  w_e[1 + DDMR_NYN * 1] = params_->weights.w_y_e;
  w_e[2 + DDMR_NYN * 2] = params_->weights.w_theta_e;
  w_e[3 + DDMR_NYN * 3] = params_->weights.w_v_e;
  w_e[4 + DDMR_NYN * 4] = params_->weights.w_r_e;
  ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, DDMR_N, "W", w_e.data());

  const double max_linear_vel = std::max(0.0, params_->max_linear_vel);
  const double max_linear_acc = std::max(0.0, params_->max_linear_acc);
  const double max_angular_vel = std::max(0.0, params_->max_angular_vel);
  const double max_angular_acc = std::max(0.0, params_->max_angular_acc);

  std::array<double, DDMR_NH> lh{
      -max_linear_vel,
      -max_linear_vel,
      -max_linear_acc,
      -max_linear_acc,
  };
  std::array<double, DDMR_NH> uh{
      max_linear_vel,
      max_linear_vel,
      max_linear_acc,
      max_linear_acc,
  };

  std::array<double, DDMR_NBU> lbu{-max_linear_acc, -max_angular_acc};
  std::array<double, DDMR_NBU> ubu{max_linear_acc, max_angular_acc};

  std::array<double, DDMR_NBX> lbx{-max_angular_vel};
  std::array<double, DDMR_NBX> ubx{max_angular_vel};

  for (int k = 0; k < DDMR_N; ++k) {
    ocp_nlp_constraints_model_set(
        nlp_config_, nlp_dims_, nlp_in_, nlp_out_, k, "lbu", lbu.data());
    ocp_nlp_constraints_model_set(
        nlp_config_, nlp_dims_, nlp_in_, nlp_out_, k, "ubu", ubu.data());
  }

  for (int k = 1; k < DDMR_N; ++k) {
    ocp_nlp_constraints_model_set(
        nlp_config_, nlp_dims_, nlp_in_, nlp_out_, k, "lh", lh.data());
    ocp_nlp_constraints_model_set(
        nlp_config_, nlp_dims_, nlp_in_, nlp_out_, k, "uh", uh.data());
    ocp_nlp_constraints_model_set(
        nlp_config_, nlp_dims_, nlp_in_, nlp_out_, k, "lbx", lbx.data());
    ocp_nlp_constraints_model_set(
        nlp_config_, nlp_dims_, nlp_in_, nlp_out_, k, "ubx", ubx.data());
  }

  return true;
}

MpcReturn AcadosMpc::RunMpc(const JointState &state, std::vector<robot_plann::Point> &path) {
  MpcStages opt_traj;
  for (auto &stage : opt_traj) {
    stage.xk.setZero(0, 0);
    stage.uk.setZero();
  }

  if (!initialized_) {
    return {opt_traj, false};
  }

  if (path.empty()) {
    return {opt_traj, false};
  }

  UpdateRuntimeWeightsAndConstraints();

  std::array<double, DDMR_NX> current_x{
      state.robot.px,
      state.robot.py,
      state.robot.yaw,
      state.robot.v,
      state.robot.yaw_rate,
  };

  ocp_nlp_constraints_model_set(
      nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", current_x.data());
  ocp_nlp_constraints_model_set(
      nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", current_x.data());

  for (int k = 0; k < DDMR_N; ++k) {
    const int idx = SafePathIndex(k, static_cast<int>(path.size()));
    const auto &ref = path.at(static_cast<std::size_t>(idx));
    const double theta_ref = HeadingAt(path, idx, state.robot.yaw);

    std::array<double, DDMR_NY> yref{
        ref.x,
        ref.y,
        theta_ref,
        ref.v,
        0.0,
        0.0,
        0.0,
    };
    ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, k, "yref", yref.data());
  }

  const int idx_e = SafePathIndex(DDMR_N, static_cast<int>(path.size()));
  const auto &ref_e = path.at(static_cast<std::size_t>(idx_e));
  std::array<double, DDMR_NYN> yref_e{
      ref_e.x,
      ref_e.y,
      HeadingAt(path, idx_e, state.robot.yaw),
      ref_e.v,
      0.0,
  };
  ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, DDMR_N, "yref", yref_e.data());

  const int status = ddmr_acados_solve(acados_capsule_);

  std::array<double, DDMR_NU> u0{};
  ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u0.data());
  opt_traj[0].uk.acc = u0[0];
  opt_traj[0].uk.dr = u0[1];

  const int max_export_stages = std::min<int>(kNP, DDMR_N + 1);
  for (int k = 0; k < max_export_stages; ++k) {
    std::array<double, DDMR_NX> xk{};
    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "x", xk.data());
    opt_traj[static_cast<std::size_t>(k)].xk.X = xk[0];
    opt_traj[static_cast<std::size_t>(k)].xk.Y = xk[1];
    opt_traj[static_cast<std::size_t>(k)].xk.phi = xk[2];
    opt_traj[static_cast<std::size_t>(k)].xk.vx = xk[3];
    opt_traj[static_cast<std::size_t>(k)].xk.r = xk[4];

    if (k < std::min<int>(kNP, DDMR_N)) {
      std::array<double, DDMR_NU> uk{};
      ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "u", uk.data());
      opt_traj[static_cast<std::size_t>(k)].uk.acc = uk[0];
      opt_traj[static_cast<std::size_t>(k)].uk.dr = uk[1];
    }
  }

  if (verbose_ >= 2 && status != 0) {
    std::cout << "[AcadosMpc] Solve failed with status " << status << std::endl;
  }

  return {opt_traj, status == 0};
}

}  // namespace robot_plann
