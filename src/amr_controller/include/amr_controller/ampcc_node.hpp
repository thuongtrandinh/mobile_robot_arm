// Copyright 2026 H-AMPCC Authors
// SPDX-License-Identifier: Apache-2.0
//
// ROS 2 node that closes the loop between sensing and actuation
// using the Adaptive MPCC backend.
//
// Dataflow:
//   /odometry/filtered  ──▶  RLS update + state estimation
//   /global_path        ──▶  reference path for MPCC
//   MPCC solver         ──▶  optimal (v_cmd, ω_cmd)
//   /diff_cont/cmd_vel  ◀──  TwistStamped output

#pragma once

#include "amr_controller/adaptive_mpcc.hpp"
#include "amr_controller/mpcc_solver.hpp"
#include "amr_controller/types.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace amr_controller {

class AMPCCNode : public rclcpp::Node {
 public:
  explicit AMPCCNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

 private:
  // ──────────── Callbacks ────────────────────
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void pathCallback(const nav_msgs::msg::Path::SharedPtr msg);
  void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void controlLoop();

  // ──────────── Helpers ─────────────────────
  void publishCmd(double v_cmd, double omega_cmd);
  void publishDiagnostics();
  void declareAllParameters();
  void loadParameters();

  /// Check if we've reached the end of the current path.
  bool goalReached() const;

  // ──────────── ROS interfaces ──────────────
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr         odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr             path_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr   cmd_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr                pred_path_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr   diag_pub_;

  rclcpp::TimerBase::SharedPtr control_timer_;

  // ──────────── Controller core ─────────────
  MPCCParams   mpcc_params_;
  RLSParams    rls_params_;
  SolverParams solver_params_;

  std::unique_ptr<AdaptiveMPCC> mpcc_;
  std::unique_ptr<MPCCSolver>   solver_;
  std::shared_ptr<WaypointPath> ref_path_;

  // ──────────── State ───────────────────────
  std::mutex state_mtx_;

  // Current robot state (from odom)
  double x_      = 0.0;
  double y_      = 0.0;
  double psi_    = 0.0;
  double v_      = 0.0;
  double omega_  = 0.0;
  bool   odom_received_ = false;

  // Previous step (for RLS)
  double v_prev_     = 0.0;
  double omega_prev_ = 0.0;
  double v_cmd_prev_ = 0.0;
  double omega_cmd_prev_ = 0.0;
  bool   first_step_ = true;

  // Path progress
  double theta_progress_ = 0.0;

  // Warm-start from previous solve
  std::vector<AugmentedControl> u_warm_;
  bool path_active_ = false;

  // Goal
  std::optional<Eigen::Vector2d> goal_position_;
  double goal_tolerance_ = 0.15;  // [m]
};

}  // namespace amr_controller
