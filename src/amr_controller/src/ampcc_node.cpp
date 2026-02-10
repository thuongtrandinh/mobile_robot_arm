// Copyright 2026 H-AMPCC Authors
// SPDX-License-Identifier: Apache-2.0

#include "amr_controller/ampcc_node.hpp"

#include <tf2/utils.h>

#include <chrono>
#include <cmath>

using namespace std::chrono_literals;

namespace amr_controller {

// ═══════════════════════════════════════════════════════════════════
//  Construction
// ═══════════════════════════════════════════════════════════════════

AMPCCNode::AMPCCNode(const rclcpp::NodeOptions& options)
    : Node("ampcc_controller", options) {

  declareAllParameters();
  loadParameters();

  // ── Build controller objects ──
  mpcc_    = std::make_unique<AdaptiveMPCC>(mpcc_params_, rls_params_);
  solver_  = std::make_unique<MPCCSolver>(*mpcc_, solver_params_);

  // ── Subscribers ──
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "odom", rclcpp::SensorDataQoS(),
      std::bind(&AMPCCNode::odomCallback, this, std::placeholders::_1));

  path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
      "global_path", 10,
      std::bind(&AMPCCNode::pathCallback, this, std::placeholders::_1));

  goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "goal_pose", 10,
      std::bind(&AMPCCNode::goalCallback, this, std::placeholders::_1));

  // ── Publishers ──
  cmd_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
      "cmd_vel", 10);

  pred_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
      "predicted_path", 10);

  diag_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      "ampcc_diagnostics", 10);

  // ── Control loop timer ──
  const auto period = std::chrono::duration<double>(mpcc_params_.dt);
  control_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&AMPCCNode::controlLoop, this));

  RCLCPP_INFO(this->get_logger(),
      "H-AMPCC controller started  [dt=%.3fs, N=%d, v_ref=%.2f m/s]",
      mpcc_params_.dt, mpcc_params_.N, mpcc_params_.v_ref);
}

// ═══════════════════════════════════════════════════════════════════
//  Parameter declaration & loading
// ═══════════════════════════════════════════════════════════════════

void AMPCCNode::declareAllParameters() {
  // MPCC
  this->declare_parameter("mpcc.N",            20);
  this->declare_parameter("mpcc.dt",           0.05);
  this->declare_parameter("mpcc.q_c",          10.0);
  this->declare_parameter("mpcc.q_l",          5.0);
  this->declare_parameter("mpcc.q_u_R",        0.1);
  this->declare_parameter("mpcc.q_u_M",        0.1);
  this->declare_parameter("mpcc.q_v",          1.0);
  this->declare_parameter("mpcc.q_theta",      2.0);
  this->declare_parameter("mpcc.q_u_vtheta",   0.01);
  this->declare_parameter("mpcc.v_ref",        0.5);
  this->declare_parameter("mpcc.v_cmd_max",    2.0);
  this->declare_parameter("mpcc.omega_cmd_max",2.0);
  this->declare_parameter("mpcc.v_theta_max",  2.5);
  this->declare_parameter("mpcc.v_theta_min", -0.1);

  // RLS
  this->declare_parameter("rls.lambda",        0.98);
  this->declare_parameter("rls.p_init",        1000.0);
  this->declare_parameter("rls.alpha_v_init",  0.95);
  this->declare_parameter("rls.beta_v_init",   0.05);
  this->declare_parameter("rls.alpha_omega_init", 0.95);
  this->declare_parameter("rls.beta_omega_init",  0.05);

  // Solver
  this->declare_parameter("solver.max_iters",        10);
  this->declare_parameter("solver.max_line_search",   8);
  this->declare_parameter("solver.cost_tol",          1e-4);
  this->declare_parameter("solver.mu_init",           1.0);

  // Navigation
  this->declare_parameter("goal_tolerance",    0.15);
}

void AMPCCNode::loadParameters() {
  // MPCC
  mpcc_params_.N              = this->get_parameter("mpcc.N").as_int();
  mpcc_params_.dt             = this->get_parameter("mpcc.dt").as_double();
  mpcc_params_.q_c            = this->get_parameter("mpcc.q_c").as_double();
  mpcc_params_.q_l            = this->get_parameter("mpcc.q_l").as_double();
  mpcc_params_.q_u_R          = this->get_parameter("mpcc.q_u_R").as_double();
  mpcc_params_.q_u_M          = this->get_parameter("mpcc.q_u_M").as_double();
  mpcc_params_.q_v            = this->get_parameter("mpcc.q_v").as_double();
  mpcc_params_.q_theta        = this->get_parameter("mpcc.q_theta").as_double();
  mpcc_params_.q_u_vtheta     = this->get_parameter("mpcc.q_u_vtheta").as_double();
  mpcc_params_.v_ref          = this->get_parameter("mpcc.v_ref").as_double();
  mpcc_params_.v_cmd_max      = this->get_parameter("mpcc.v_cmd_max").as_double();
  mpcc_params_.omega_cmd_max  = this->get_parameter("mpcc.omega_cmd_max").as_double();
  mpcc_params_.v_theta_max    = this->get_parameter("mpcc.v_theta_max").as_double();
  mpcc_params_.v_theta_min    = this->get_parameter("mpcc.v_theta_min").as_double();

  // RLS
  rls_params_.lambda  = this->get_parameter("rls.lambda").as_double();
  rls_params_.p_init  = this->get_parameter("rls.p_init").as_double();
  rls_params_.theta_init <<
      this->get_parameter("rls.alpha_v_init").as_double(),
      this->get_parameter("rls.beta_v_init").as_double(),
      this->get_parameter("rls.alpha_omega_init").as_double(),
      this->get_parameter("rls.beta_omega_init").as_double();

  // Solver
  solver_params_.max_iters       = this->get_parameter("solver.max_iters").as_int();
  solver_params_.max_line_search = this->get_parameter("solver.max_line_search").as_int();
  solver_params_.cost_tol        = this->get_parameter("solver.cost_tol").as_double();
  solver_params_.mu_init         = this->get_parameter("solver.mu_init").as_double();

  // Nav
  goal_tolerance_ = this->get_parameter("goal_tolerance").as_double();
}

// ═══════════════════════════════════════════════════════════════════
//  Odometry callback
// ═══════════════════════════════════════════════════════════════════

void AMPCCNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(state_mtx_);

  x_     = msg->pose.pose.position.x;
  y_     = msg->pose.pose.position.y;

  // Extract yaw from quaternion
  const auto& q = msg->pose.pose.orientation;
  psi_ = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z));

  v_     = msg->twist.twist.linear.x;
  omega_ = msg->twist.twist.angular.z;

  odom_received_ = true;
}

// ═══════════════════════════════════════════════════════════════════
//  Path callback  (/global_path → reference path for MPCC)
// ═══════════════════════════════════════════════════════════════════

void AMPCCNode::pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
  if (msg->poses.size() < 2) {
    RCLCPP_WARN(this->get_logger(), "Received path with < 2 poses, ignoring");
    return;
  }

  std::vector<Eigen::Vector2d> waypoints;
  waypoints.reserve(msg->poses.size());
  for (const auto& ps : msg->poses) {
    waypoints.emplace_back(ps.pose.position.x, ps.pose.position.y);
  }

  {
    std::lock_guard<std::mutex> lock(state_mtx_);

    ref_path_ = std::make_shared<WaypointPath>(waypoints);
    mpcc_->setReferencePath(ref_path_);
    path_active_ = true;

    // Project current position onto new path to initialise progress
    theta_progress_ = ref_path_->projectOntoPath(x_, y_);

    // Invalidate warm-start since path changed
    u_warm_.clear();

    // Derive goal from path endpoint
    goal_position_ = waypoints.back();
  }

  RCLCPP_INFO(this->get_logger(),
      "New path received (%zu waypoints, length=%.2f m)",
      waypoints.size(), ref_path_->getLength());
}

// ═══════════════════════════════════════════════════════════════════
//  Goal callback  (direct goal → create straight-line path)
// ═══════════════════════════════════════════════════════════════════

void AMPCCNode::goalCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(state_mtx_);

  const double gx = msg->pose.position.x;
  const double gy = msg->pose.position.y;

  // Build a simple straight-line path from current position to goal
  std::vector<Eigen::Vector2d> waypoints;
  waypoints.emplace_back(x_, y_);
  waypoints.emplace_back(gx, gy);

  ref_path_ = std::make_shared<WaypointPath>(waypoints);
  mpcc_->setReferencePath(ref_path_);
  path_active_ = true;
  theta_progress_ = 0.0;
  u_warm_.clear();
  goal_position_ = Eigen::Vector2d(gx, gy);

  RCLCPP_INFO(this->get_logger(),
      "Direct goal received (%.2f, %.2f), straight-line path", gx, gy);
}

// ═══════════════════════════════════════════════════════════════════
//  Main control loop
// ═══════════════════════════════════════════════════════════════════

void AMPCCNode::controlLoop() {
  std::lock_guard<std::mutex> lock(state_mtx_);

  // ── Guard: need odometry ──
  if (!odom_received_) {
    return;
  }

  // ── Guard: need an active path ──
  if (!path_active_ || !mpcc_->hasReferencePath()) {
    publishCmd(0.0, 0.0);
    return;
  }

  // ── Check goal reached ──
  if (goalReached()) {
    publishCmd(0.0, 0.0);
    path_active_ = false;
    RCLCPP_INFO(this->get_logger(), "Goal reached!");
    return;
  }

  // ── 1. RLS model update ──
  if (!first_step_) {
    mpcc_->updateModel(v_, omega_,
                       v_prev_, omega_prev_,
                       v_cmd_prev_, omega_cmd_prev_);
  }

  // ── 2. Project position onto path for current θ ──
  theta_progress_ = ref_path_->projectOntoPath(x_, y_, theta_progress_);

  // ── 3. Build augmented state ──
  AugmentedState z0;
  z0 << x_, y_, psi_, v_, omega_, theta_progress_;

  // ── 4. Prepare warm-start ──
  if (u_warm_.empty()) {
    u_warm_ = solver_->defaultWarmStart();
  }

  // ── 5. Solve MPCC ──
  const auto sol = solver_->solve(z0, u_warm_);

  // ── 6. Extract first optimal control ──
  double v_cmd     = 0.0;
  double omega_cmd = 0.0;

  if (!sol.u_seq.empty()) {
    v_cmd     = sol.u_seq[0](kForce);    // v_cmd
    omega_cmd = sol.u_seq[0](kTorque);   // ω_cmd

    // Warm-start for next cycle: shift solution
    u_warm_ = MPCCSolver::shiftSequence(sol.u_seq);
  }

  // ── 7. Publish velocity command ──
  publishCmd(v_cmd, omega_cmd);

  // ── 8. Store for next RLS step ──
  v_prev_         = v_;
  omega_prev_     = omega_;
  v_cmd_prev_     = v_cmd;
  omega_cmd_prev_ = omega_cmd;
  first_step_     = false;

  // ── 9. Publish predicted path for visualisation ──
  if (pred_path_pub_->get_subscription_count() > 0 && !sol.z_seq.empty()) {
    nav_msgs::msg::Path pred_msg;
    pred_msg.header.stamp    = this->get_clock()->now();
    pred_msg.header.frame_id = "odom";
    for (const auto& z : sol.z_seq) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = pred_msg.header;
      ps.pose.position.x = z(kX);
      ps.pose.position.y = z(kY);
      ps.pose.orientation.z = std::sin(z(kPsi) / 2.0);
      ps.pose.orientation.w = std::cos(z(kPsi) / 2.0);
      pred_msg.poses.push_back(ps);
    }
    pred_path_pub_->publish(pred_msg);
  }

  // ── 10. Publish diagnostics ──
  publishDiagnostics();
}

// ═══════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════

void AMPCCNode::publishCmd(double v_cmd, double omega_cmd) {
  geometry_msgs::msg::TwistStamped msg;
  msg.header.stamp    = this->get_clock()->now();
  msg.header.frame_id = "base_link";
  msg.twist.linear.x  = v_cmd;
  msg.twist.angular.z = omega_cmd;
  cmd_pub_->publish(msg);
}

bool AMPCCNode::goalReached() const {
  if (!goal_position_.has_value()) return false;
  const double dx = x_ - goal_position_->x();
  const double dy = y_ - goal_position_->y();
  return std::sqrt(dx * dx + dy * dy) < goal_tolerance_;
}

void AMPCCNode::publishDiagnostics() {
  if (diag_pub_->get_subscription_count() == 0) return;

  const auto theta = mpcc_->getCurrentTheta();

  std_msgs::msg::Float64MultiArray msg;
  // [α_v, β_v, α_ω, β_ω, θ_progress, v, ω, innovation_v, innovation_ω]
  msg.data = {
    theta(kAlphaV), theta(kBetaV),
    theta(kAlphaOmega), theta(kBetaOmega),
    theta_progress_,
    v_, omega_,
    mpcc_->getEstimator().getInnovationLinear(),
    mpcc_->getEstimator().getInnovationAngular()
  };
  diag_pub_->publish(msg);
}

}  // namespace amr_controller
