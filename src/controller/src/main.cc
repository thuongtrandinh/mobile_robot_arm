#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>

#include "backward.hpp"
#include "interfaces/msg/joint_state.hpp"
#include "interfaces/srv/ocp_local_plann.hpp"
#include "planner.h"

// using json = nlohmann::json;

// Keep backward available for optional debugging, but do not install
// process-wide signal handlers in runtime node execution.

namespace robot_plann {
//
std::unique_ptr<robot_plann::Planner> _planner;
int _verbose = 1;
rclcpp::Node::SharedPtr _node;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr astar_path_pub;
rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr local_costmap_pub;
rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_pub;
rclcpp::TimerBase::SharedPtr mpc_solve_timer;
rclcpp::CallbackGroup::SharedPtr mpc_callback_group;
JointState _latest_state;
bool _has_latest_state = false;
bool _has_reference = false;
double _mpc_solve_period = 0.05;
double _mpc_solve_hz = 20.0;
double _mpc_control_dt = 0.05;
double _cmd_max_linear_speed = 1.0;
double _cmd_max_angular_speed = 1.0;
double _goal_stop_distance = 0.1;
std::mutex _planner_mutex;
uint64_t _mpc_fail_count = 0;
uint64_t _ref_missing_count = 0;
uint64_t _goal_stop_count = 0;
std::atomic<bool> _shutting_down{false};

namespace {
double ClampValue(double value, double low, double high) {
  return std::max(low, std::min(high, value));
}

void PublishCmdVel(double linear, double angular) {
  if (!_node || !cmd_pub) {
    return;
  }

  geometry_msgs::msg::TwistStamped msg;
  msg.header.stamp = _node->now();
  msg.header.frame_id = "base_link";
  msg.twist.linear.x = linear;
  msg.twist.angular.z = angular;
  cmd_pub->publish(msg);
}
}

namespace {
robot_plann::MpcParams::Ptr BuildMpcParamsFromNode(const rclcpp::Node::SharedPtr &node) {
  auto params = std::make_shared<robot_plann::MpcParams>();
  params->np = static_cast<uint16_t>(
      node->declare_parameter<int>("mpc.horizon_steps", robot_plann::kNP));
  params->dt = node->declare_parameter<double>("mpc.dt", robot_plann::kDT);
  params->max_linear_vel =
      node->declare_parameter<double>("mpc.max_linear_vel", robot_plann::kMaxLinearVel);
  params->max_angular_vel =
      node->declare_parameter<double>("mpc.max_angular_vel", robot_plann::kMaxAngularVel);
  params->max_linear_acc =
      node->declare_parameter<double>("mpc.max_linear_acc", robot_plann::kMaxLinearAcc);
  params->max_angular_acc =
      node->declare_parameter<double>("mpc.max_angular_acc", robot_plann::kMaxAngularAcc);
  params->wheel_half_track = node->declare_parameter<double>("mpc.wheel_half_track", 0.3);
  params->local_obst_num = node->declare_parameter<int>("mpc.local_obst_num", 8);
  params->solver_max_iter = node->declare_parameter<int>("mpc.solver.max_iter", 250);
  params->solver_max_cpu_time = node->declare_parameter<double>("mpc.solver.max_cpu_time", 0.09);
  params->polygon_clearance_margin =
      node->declare_parameter<double>("mpc.polygon_clearance_margin", 0.03);
    params->wall_constraint_activation_distance =
      node->declare_parameter<double>("mpc.wall_constraint_activation_distance", 2.0);

  params->weights.slack = node->declare_parameter<double>("mpc.weights.slack", 99999.0);
  params->weights.pose_x = node->declare_parameter<double>("mpc.weights.pose_x", 5.0);
  params->weights.pose_y = node->declare_parameter<double>("mpc.weights.pose_y", 5.0);
  params->weights.yaw_rate = node->declare_parameter<double>("mpc.weights.yaw_rate", 1.0);
  params->weights.terminal_x = node->declare_parameter<double>("mpc.weights.terminal_x", 100.0);
  params->weights.terminal_y = node->declare_parameter<double>("mpc.weights.terminal_y", 100.0);
  params->weights.input_acc = node->declare_parameter<double>("mpc.weights.input_acc", 2.0);
  params->weights.input_yaw_acc =
      node->declare_parameter<double>("mpc.weights.input_yaw_acc", 0.5);
  params->weights.smooth_acc = node->declare_parameter<double>("mpc.weights.smooth_acc", 0.05);
  params->weights.smooth_yaw_acc =
      node->declare_parameter<double>("mpc.weights.smooth_yaw_acc", 0.05);

  if (params->np < 2) {
    RCLCPP_WARN(node->get_logger(), "mpc.horizon_steps must be >= 2, forcing 2");
    params->np = 2;
  }
  if (params->np > robot_plann::kNP) {
    RCLCPP_WARN(
        node->get_logger(),
        "mpc.horizon_steps=%u exceeds compiled max kNP=%d, clamping to %d",
        params->np,
        robot_plann::kNP,
        robot_plann::kNP);
    params->np = robot_plann::kNP;
  }
  if (params->dt <= 0.0) {
    RCLCPP_WARN(node->get_logger(), "mpc.dt must be > 0, fallback to %.3f", robot_plann::kDT);
    params->dt = robot_plann::kDT;
  }
  if (params->wheel_half_track <= 0.0) {
    RCLCPP_WARN(node->get_logger(), "mpc.wheel_half_track must be > 0, fallback to 0.3");
    params->wheel_half_track = 0.3;
  }
  if (params->max_linear_vel <= 0.0) {
    params->max_linear_vel = robot_plann::kMaxLinearVel;
  }
  if (params->max_linear_acc <= 0.0) {
    params->max_linear_acc = robot_plann::kMaxLinearAcc;
  }
  if (params->local_obst_num < 0) {
    params->local_obst_num = 0;
  }
  if (params->solver_max_iter < 20) {
    params->solver_max_iter = 20;
  }
  if (params->solver_max_cpu_time <= 0.0) {
    params->solver_max_cpu_time = 0.09;
  }
  if (params->polygon_clearance_margin < 0.0) {
    params->polygon_clearance_margin = 0.0;
  }
  if (params->wall_constraint_activation_distance < 0.0) {
    params->wall_constraint_activation_distance = 0.0;
  }

  return params;
}
}  // namespace

void MpcExecTimerCallback() {
  if (!rclcpp::ok() || _shutting_down.load()) {
    return;
  }

  JointState state_snapshot;
  bool has_state = false;
  bool has_ref = false;
  {
    std::lock_guard<std::mutex> lock(_planner_mutex);
    has_state = _has_latest_state;
    has_ref = _has_reference;
    if (has_state) {
      state_snapshot = _latest_state;
    }
  }

  if (!has_state) {
    RCLCPP_WARN_THROTTLE(
        _node->get_logger(),
        *(_node->get_clock()),
        2000,
        "MPC timer waiting for latest planner state");
    return;
  }

  const double dx_goal = state_snapshot.robot.gx - state_snapshot.robot.px;
  const double dy_goal = state_snapshot.robot.gy - state_snapshot.robot.py;
  const double goal_distance = std::hypot(dx_goal, dy_goal);
  if (goal_distance <= _goal_stop_distance) {
    ++_goal_stop_count;
    {
      std::lock_guard<std::mutex> lock(_planner_mutex);
      _latest_state.robot.v = 0.0;
      _latest_state.robot.yaw_rate = 0.0;
    }
    PublishCmdVel(0.0, 0.0);
    RCLCPP_INFO_THROTTLE(
        _node->get_logger(),
        *(_node->get_clock()),
        2000,
        "Goal-stop active: dist=%.3f <= %.3f (count=%lu)",
        goal_distance,
        _goal_stop_distance,
        static_cast<unsigned long>(_goal_stop_count));
    return;
  }

  if (!has_ref) {
    ++_ref_missing_count;
    PublishCmdVel(0.0, 0.0);
    RCLCPP_WARN_THROTTLE(
        _node->get_logger(),
        *(_node->get_clock()),
        2000,
        "No cached reference yet, hold zero cmd (count=%lu)",
        static_cast<unsigned long>(_ref_missing_count));
    return;
  }

  MpcReturn mpc_return;
  std::size_t cached_ref_len = 0;
  {
    std::lock_guard<std::mutex> lock(_planner_mutex);
    cached_ref_len = _planner->GetAStarPath().size();
    mpc_return = _planner->SolveMpcFromCachedReference(state_snapshot);
  }
  if (!mpc_return.success) {
    ++_mpc_fail_count;
    RCLCPP_WARN_THROTTLE(
        _node->get_logger(),
        *(_node->get_clock()),
        1000,
        "MPC solve failed: cached_ref_len=%zu robot=(%.3f, %.3f, yaw=%.3f, v=%.3f, w=%.3f) goal_dist=%.3f fail_count=%lu",
        cached_ref_len,
        state_snapshot.robot.px,
        state_snapshot.robot.py,
        state_snapshot.robot.yaw,
        state_snapshot.robot.v,
        state_snapshot.robot.yaw_rate,
        goal_distance,
        static_cast<unsigned long>(_mpc_fail_count));
      PublishCmdVel(0.0, 0.0);
    return;
  }

  double half_track = 0.3;
  {
    std::lock_guard<std::mutex> lock(_planner_mutex);
    half_track = _planner->GetWheelHalfTrack();
  }
  double v_left = state_snapshot.robot.v - state_snapshot.robot.yaw_rate * half_track;
  double v_right = state_snapshot.robot.v + state_snapshot.robot.yaw_rate * half_track;

  const double al = mpc_return.stages[0].uk.acc - mpc_return.stages[0].uk.dr * half_track;
  const double ar = mpc_return.stages[0].uk.acc + mpc_return.stages[0].uk.dr * half_track;

  v_left += al * _mpc_control_dt;
  v_right += ar * _mpc_control_dt;

  double linear = 0.5 * (v_left + v_right);
  double angular = (v_right - v_left) / (2.0 * half_track);
  linear = ClampValue(linear, -_cmd_max_linear_speed, _cmd_max_linear_speed);
  angular = ClampValue(angular, -_cmd_max_angular_speed, _cmd_max_angular_speed);

  if (std::abs(linear) < 1e-4 && std::abs(angular) < 1e-4) {
    RCLCPP_INFO_THROTTLE(
        _node->get_logger(),
        *(_node->get_clock()),
        1000,
        "MPC success but zero cmd: acc=%.4f dr=%.4f ref_len=%zu",
        mpc_return.stages[0].uk.acc,
        mpc_return.stages[0].uk.dr,
        cached_ref_len);
  }

  {
    std::lock_guard<std::mutex> lock(_planner_mutex);
    _latest_state.robot.v = linear;
    _latest_state.robot.yaw_rate = angular;
  }
  PublishCmdVel(linear, angular);
}


void PlannSrvCallback(
    const std::shared_ptr<interfaces::srv::OcpLocalPlann::Request> req,
    std::shared_ptr<interfaces::srv::OcpLocalPlann::Response> res) {
  if (_shutting_down.load()) {
    res->success = false;
    return;
  }

  robot_plann::JointState ob_state;
  const auto &robot_state = req->ob.robot_state;
  ob_state.robot.px = robot_state.pose.x;
  ob_state.robot.py = robot_state.pose.y;
  ob_state.robot.yaw = robot_state.pose.theta;
  ob_state.robot.v = 0.5 * (robot_state.vr + robot_state.vl);
  ob_state.robot.yaw_rate =
      0.5 * (robot_state.vr - robot_state.vl) / robot_state.radius;

  ob_state.robot.v_pref = robot_state.v_pref;
  ob_state.robot.radius = robot_state.radius;
  ob_state.robot.gx = robot_state.gx;
  ob_state.robot.gy = robot_state.gy;

  for (const auto &hum_iter : req->ob.human_states) {
    robot_plann::HumanState hum_state;
    hum_state.px = hum_iter.px;
    hum_state.py = hum_iter.py;
    hum_state.vx = hum_iter.vx;
    hum_state.vy = hum_iter.vy;
    hum_state.radius = hum_iter.radius;
    ob_state.hum.push_back(hum_state);
  }

  for (const auto &obst_iter : req->ob.obstacle_states) {
    robot_plann::ObstacleState obst_state;
    obst_state.px = obst_iter.px;
    obst_state.py = obst_iter.py;
    obst_state.radius = obst_iter.radius;
    ob_state.obst.push_back(obst_state);
  }

  // clock-wise
  ob_state.rect.vertices.clear();
  for (const auto &poly_iter : req->ob.poly_states) {
    for (const auto &vertex : poly_iter.vertices) {
      Eigen::Vector2d pt(vertex.x, vertex.y);
      ob_state.rect.vertices.push_back(pt);
    }
    break;
  }

  ob_state.walls.clear();
  const std::size_t input_walls = req->ob.walls.size();
  for (const auto &input_wall : req->ob.walls) {
    Wall wall;
    wall.first.x = input_wall.sx;
    wall.first.y = input_wall.sy;
    wall.second.x = input_wall.ex;
    wall.second.y = input_wall.ey;
    ob_state.walls.push_back(wall);
  }
  RCLCPP_INFO_THROTTLE(
      robot_plann::_node->get_logger(),
      *(robot_plann::_node->get_clock()),
      2000,
      "Planner request scene: obst=%zu walls_in=%zu walls_kept=%zu poly_in=%zu",
      ob_state.obst.size(),
      input_walls,
      ob_state.walls.size(),
      req->ob.poly_states.size());

  Eigen::Vector2d sub_goal = {req->sub_goal.x, req->sub_goal.y};

  {
    std::lock_guard<std::mutex> lock(_planner_mutex);
    _latest_state = ob_state;
    _has_latest_state = true;
  }

  bool ref_ok = false;
  {
    std::lock_guard<std::mutex> lock(_planner_mutex);
    ref_ok = _planner->UpdateReferenceOnly(ob_state, sub_goal);
    _has_reference = _has_reference || ref_ok;
  }

  res->success = ref_ok;
  res->al = 0.0;
  res->ar = 0.0;
  res->revised_goal.x = sub_goal.x();
  res->revised_goal.y = sub_goal.y();
  res->astar_path.clear();
  res->control_vars.clear();

  nav_msgs::msg::Path refline_line = _planner->GetAStarSmoothPath();
  refline_line.header.stamp = _node->now();
  if (astar_path_pub && !refline_line.poses.empty()) {
    astar_path_pub->publish(refline_line);
  }

  nav_msgs::msg::OccupancyGrid local_costmap = _planner->GetLocalCostMap("map");
  local_costmap.header.stamp = _node->now();
  if (local_costmap_pub && local_costmap.info.width > 0 && local_costmap.info.height > 0) {
    local_costmap_pub->publish(local_costmap);
  }
}

void TestRun() {}

}  // namespace robot_plann

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  robot_plann::_shutting_down.store(false);
  robot_plann::_node = std::make_shared<rclcpp::Node>("opt_planner");
  auto mpc_params = robot_plann::BuildMpcParamsFromNode(robot_plann::_node);

  robot_plann::_planner = std::make_unique<robot_plann::Planner>(
      mpc_params, robot_plann::_verbose);

  RCLCPP_INFO(
      robot_plann::_node->get_logger(),
      "MPC config: horizon=%u dt=%.3f (%.2f Hz) max_v=%.2f max_a=%.2f wheel_half_track=%.3f local_obst_num=%d",
      mpc_params->np,
      mpc_params->dt,
      1.0 / mpc_params->dt,
      mpc_params->max_linear_vel,
      mpc_params->max_linear_acc,
      mpc_params->wheel_half_track,
      mpc_params->local_obst_num);

      robot_plann::_mpc_solve_period = robot_plann::_node->declare_parameter<double>(
        "planner.mpc_exec_period", 0.05);
      robot_plann::_mpc_solve_hz = robot_plann::_node->declare_parameter<double>(
        "planner.mpc_exec_hz", 0.0);
      if (robot_plann::_mpc_solve_hz > 0.0) {
        robot_plann::_mpc_solve_period = 1.0 / robot_plann::_mpc_solve_hz;
      }
      if (robot_plann::_mpc_solve_period <= 0.0) {
        robot_plann::_mpc_solve_period = 0.05;
      }
      robot_plann::_mpc_control_dt = robot_plann::_node->declare_parameter<double>(
          "planner.mpc_control_dt", robot_plann::_mpc_solve_period);
      if (robot_plann::_mpc_control_dt <= 0.0) {
        robot_plann::_mpc_control_dt = robot_plann::_mpc_solve_period;
      }
      const std::string cmd_topic = robot_plann::_node->declare_parameter<std::string>(
        "planner.cmd_topic", "/diff_cont/cmd_vel");
      const std::string local_costmap_topic =
        robot_plann::_node->declare_parameter<std::string>(
          "planner.local_costmap_topic", "/a_star/local_costmap");
      robot_plann::_cmd_max_linear_speed = robot_plann::_node->declare_parameter<double>(
        "planner.cmd_max_linear_speed", mpc_params->max_linear_vel);
      robot_plann::_cmd_max_angular_speed = robot_plann::_node->declare_parameter<double>(
        "planner.cmd_max_angular_speed", mpc_params->max_angular_vel);
      robot_plann::_goal_stop_distance = robot_plann::_node->declare_parameter<double>(
        "planner.goal_stop_distance", 0.1);
      if (robot_plann::_goal_stop_distance < 0.0) {
        robot_plann::_goal_stop_distance = 0.0;
      }

  auto plann_srv = robot_plann::_node->create_service<interfaces::srv::OcpLocalPlann>(
      "/ocp_plann", robot_plann::PlannSrvCallback);

  robot_plann::astar_path_pub =
      robot_plann::_node->create_publisher<nav_msgs::msg::Path>("/a_star_path", 1);
      robot_plann::local_costmap_pub =
        robot_plann::_node->create_publisher<nav_msgs::msg::OccupancyGrid>(local_costmap_topic, 1);
      robot_plann::cmd_pub =
        robot_plann::_node->create_publisher<geometry_msgs::msg::TwistStamped>(cmd_topic, 10);

      robot_plann::mpc_callback_group = robot_plann::_node->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

      robot_plann::mpc_solve_timer = robot_plann::_node->create_wall_timer(
        std::chrono::duration<double>(robot_plann::_mpc_solve_period),
        robot_plann::MpcExecTimerCallback,
        robot_plann::mpc_callback_group);
      RCLCPP_INFO(
        robot_plann::_node->get_logger(),
        "Zero-latency pipeline: reference via service, MPC solve %.2f Hz, immediate cmd publish, cmd topic %s",
        1.0 / robot_plann::_mpc_solve_period,
        cmd_topic.c_str());
      RCLCPP_INFO(
        robot_plann::_node->get_logger(),
        "Goal stop distance enabled: %.3f m",
        robot_plann::_goal_stop_distance);

        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
        executor.add_node(robot_plann::_node);
        executor.spin();

        robot_plann::_shutting_down.store(true);

        if (robot_plann::mpc_solve_timer) {
          robot_plann::mpc_solve_timer->cancel();
          robot_plann::mpc_solve_timer.reset();
        }

        robot_plann::cmd_pub.reset();
        robot_plann::astar_path_pub.reset();
        robot_plann::local_costmap_pub.reset();

        executor.cancel();
        executor.remove_node(robot_plann::_node);
        robot_plann::mpc_callback_group.reset();
        robot_plann::_planner.reset();
        robot_plann::_node.reset();

  rclcpp::shutdown();
  return 0;
}
