#include "acados_mpc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

namespace robot_plann {
namespace {

constexpr int kNumWalls = 2;
constexpr int kNumHumans = 5;
constexpr int kNumStatics = 40;
constexpr int kNumPolyEdges = 20;
constexpr int kOffsetWalls = 0;
constexpr int kOffsetHumans = kOffsetWalls + 4 * kNumWalls;
constexpr int kOffsetStatics = kOffsetHumans + 3 * kNumHumans;
constexpr int kOffsetPoly = kOffsetStatics + 3 * kNumStatics;
constexpr int kExpectedNp = kOffsetPoly + 3 * kNumPolyEdges;

static_assert(kExpectedNp == DDMR_NP, "Parameter layout mismatch with generated ACADOS solver");

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

void BuildPolygonHalfspaces(
    const std::vector<Eigen::Vector2d> &vertices,
    std::vector<std::array<double, 3>> &halfspaces,
    int max_edges) {
  halfspaces.clear();
  if (vertices.size() < 2) {
    return;
  }

  for (std::size_t i = 0; i < vertices.size() && static_cast<int>(halfspaces.size()) < max_edges; ++i) {
    const Eigen::Vector2d &a = vertices[i];
    const Eigen::Vector2d &b = vertices[(i + 1) % vertices.size()];
    const Eigen::Vector2d e = b - a;
    const double n = e.norm();
    if (n < 1e-9) {
      continue;
    }

    // Clockwise polygon -> inward normal is [-dy, dx].
    const double ax = -e.y() / n;
    const double ay = e.x() / n;
    const double bb = ax * a.x() + ay * a.y();
    halfspaces.push_back({ax, ay, bb});
  }
}

void FillStageParams(const JointState &state, std::array<double, DDMR_NP> &p) {
  p.fill(0.0);

  std::vector<int> nearest_static_idx;
  nearest_static_idx.reserve(state.obst.size());
  for (int i = 0; i < static_cast<int>(state.obst.size()); ++i) {
    nearest_static_idx.push_back(i);
  }
  std::sort(
      nearest_static_idx.begin(),
      nearest_static_idx.end(),
      [&state](int a, int b) {
        const auto &oa = state.obst[static_cast<std::size_t>(a)];
        const auto &ob = state.obst[static_cast<std::size_t>(b)];
        const double da2 = std::pow(oa.px - state.robot.px, 2) + std::pow(oa.py - state.robot.py, 2);
        const double db2 = std::pow(ob.px - state.robot.px, 2) + std::pow(ob.py - state.robot.py, 2);
        return da2 < db2;
      });

  // Walls disabled in grid-circle mode. Keep parameters harmless.
  for (int i = 0; i < kNumWalls; ++i) {
    const int base = kOffsetWalls + 4 * i;
    p[base + 0] = 0.0;
    p[base + 1] = 0.0;
    p[base + 2] = 0.0;
    p[base + 3] = 0.0;
  }

  // Humans: [hx, hy, safe_d2]
  for (int i = 0; i < kNumHumans; ++i) {
    const int base = kOffsetHumans + 3 * i;
    if (i < static_cast<int>(state.hum.size())) {
      const auto &h = state.hum[static_cast<std::size_t>(i)];
      const double safe_r = h.radius + kInflationRadius + 0.1;
      p[base + 0] = h.px;
      p[base + 1] = h.py;
      p[base + 2] = safe_r * safe_r;
    }
  }

  // Static obstacles: [ox, oy, safe_d2] for nearest obstacles first.
  for (int i = 0; i < kNumStatics; ++i) {
    const int base = kOffsetStatics + 3 * i;
    if (i < static_cast<int>(nearest_static_idx.size())) {
      const int src_idx = nearest_static_idx[static_cast<std::size_t>(i)];
      const auto &o = state.obst[static_cast<std::size_t>(src_idx)];
      const double safe_r = o.radius + kInflationRadius + 0.1;
      p[base + 0] = o.px;
      p[base + 1] = o.py;
      p[base + 2] = safe_r * safe_r;
    }
  }

  // Polygon constraints disabled in grid-circle mode.
  // Set edges to inactive halfspaces so constraints remain trivially feasible.
  for (int i = 0; i < kNumPolyEdges; ++i) {
    const int base = kOffsetPoly + 3 * i;
    p[base + 0] = 0.0;
    p[base + 1] = 0.0;
    p[base + 2] = -1e6;
  }
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
  // Keep small positive regularization for all virtual controls.
  const double w_jv = std::max(1e-6, params_->weights.smooth_acc * params_->dt * params_->dt);
  const double w_jr = std::max(1e-6, params_->weights.smooth_yaw_acc * params_->dt * params_->dt);
  w[7 + DDMR_NY * 7] = w_jv;
  w[8 + DDMR_NY * 8] = w_jr;
  for (int i = 9; i < DDMR_NY; ++i) {
    w[i + DDMR_NY * i] = 1e-4;
  }

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
    std::array<double, DDMR_NH> lh{};
    std::array<double, DDMR_NH> uh{};
    lh.fill(0.0);
    uh.fill(1e9);
    // Wheel limits.
    lh[0] = -max_linear_vel;
    lh[1] = -max_linear_vel;
    lh[2] = -max_linear_acc;
    lh[3] = -max_linear_acc;
    uh[0] = max_linear_vel;
    uh[1] = max_linear_vel;
    uh[2] = max_linear_acc;
    uh[3] = max_linear_acc;
    // Dual-norm relaxed hard bound index (4 wheel + 2 walls + 5 hum + 5 stat = 16).
    // A too-tight band often causes QP min-step stalls in cluttered scenes.
    lh[16] = 0.70;
    uh[16] = 1.30;

    std::array<double, DDMR_NBU> lbu{};
    std::array<double, DDMR_NBU> ubu{};
    lbu.fill(0.0);
    ubu.fill(1e3);
    // j_v / j_r bounds
    lbu[0] = -5.0;
    ubu[0] = 5.0;
    lbu[1] = -5.0;
    ubu[1] = 5.0;

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
      0.0,
      0.0,
  };

  ocp_nlp_constraints_model_set(
      nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", current_x.data());
  ocp_nlp_constraints_model_set(
      nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", current_x.data());

  std::array<double, DDMR_NP> stage_p{};
  FillStageParams(state, stage_p);
  for (int k = 0; k < DDMR_N; ++k) {
    ddmr_acados_update_params(acados_capsule_, k, stage_p.data(), DDMR_NP);
  }

  for (int k = 0; k < DDMR_N; ++k) {
    const int idx = SafePathIndex(k, static_cast<int>(path.size()));
    const auto &ref = path.at(static_cast<std::size_t>(idx));
    const double theta_ref = HeadingAt(path, idx, state.robot.yaw);

    std::array<double, DDMR_NY> yref{};
    yref.fill(0.0);
    yref[0] = ref.x;
    yref[1] = ref.y;
    yref[2] = theta_ref;
    yref[3] = ref.v;
    yref[4] = 0.0;
    yref[5] = 0.0;
    yref[6] = 0.0;
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

  std::array<double, DDMR_NX> x0_sol{};
  ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "x", x0_sol.data());
  // Preserve old planner interface: publish acc/dr equivalent command channels.
  opt_traj[0].uk.acc = x0_sol[5];
  opt_traj[0].uk.dr = x0_sol[6];

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
      // Keep compatibility with old consumer expecting [acc, dr].
      opt_traj[static_cast<std::size_t>(k)].uk.acc = xk[5];
      opt_traj[static_cast<std::size_t>(k)].uk.dr = xk[6];
    }
  }

  if (verbose_ >= 2 && status != 0) {
    std::cout << "[AcadosMpc] Solve failed with status " << status << std::endl;
  }

  // ACADOS status 4 indicates NLP failure; do not treat it as valid control.
  const bool is_ok = (status == 0 || status == 3);
  return {opt_traj, is_ok};
}

}  // namespace robot_plann
