#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>

#include "backward.hpp"
#include "interfaces/msg/control_var.hpp"
#include "interfaces/msg/joint_state.hpp"
#include "interfaces/srv/ocp_local_plann.hpp"
#include "planner.h"

// using json = nlohmann::json;

namespace backward {
backward::SignalHandling sh;
}

namespace robot_plann {
//
std::unique_ptr<robot_plann::Planner> _planner;
int _verbose = 1;
rclcpp::Node::SharedPtr _node;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr astar_path_pub;

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

  return params;
}
}  // namespace


void PlannSrvCallback(
    const std::shared_ptr<interfaces::srv::OcpLocalPlann::Request> req,
    std::shared_ptr<interfaces::srv::OcpLocalPlann::Response> res) {
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
  for (const auto &input_wall : req->ob.walls) {
    Wall wall = Planner::ClipWall(input_wall.sx,
                                  input_wall.sy,
                                  input_wall.ex,
                                  input_wall.ey, -5.5, 5.5);
    ob_state.walls.push_back(wall);
  }

  Eigen::Vector2d sub_goal = {req->sub_goal.x, req->sub_goal.y};

  auto start_stamp = std::chrono::high_resolution_clock::now();
  auto mpc_return = _planner->PlannExec(ob_state, sub_goal);
  auto end_stamp = std::chrono::high_resolution_clock::now();
  double time_cost =
      std::chrono::duration<double, std::milli>(end_stamp - start_stamp)
          .count();

  res->astar_path.clear();
  std::vector<robot_plann::Point> astar_path = _planner->GetAStarPath();
  for (size_t i = 0; i < astar_path.size(); ++i) {
    interfaces::msg::Point pt;
    pt.x = astar_path.at(i).x;
    pt.y = astar_path.at(i).y;
    res->astar_path.push_back(pt);
  }

  res->control_vars.clear();
  interfaces::msg::ControlVar cur_control_var{};
  res->success = false;
  const int horizon = _planner->GetMpcHorizonSteps();
  const double dt = _planner->GetMpcDt();
  const double half_track = _planner->GetWheelHalfTrack();

  if (!mpc_return.success) {
    res->al = -req->ob.robot_state.vl / dt;
    res->ar = -req->ob.robot_state.vr / dt;
    res->revised_goal.x = sub_goal.x();
    res->revised_goal.y = sub_goal.y();
    cur_control_var.al = res->al;
    cur_control_var.ar = res->ar;
    for (int i = 0; i < horizon; ++i) {
      res->control_vars.push_back(cur_control_var);
    }

  } else {
    res->al = mpc_return.stages[0].uk.acc - mpc_return.stages[0].uk.dr * half_track;
    res->ar = mpc_return.stages[0].uk.acc + mpc_return.stages[0].uk.dr * half_track;
    res->revised_goal.x = sub_goal.x();
    res->revised_goal.y = sub_goal.y();

    for (int i = 0; i < horizon; ++i) {
      cur_control_var.al =
          mpc_return.stages.at(i).uk.acc - mpc_return.stages.at(i).uk.dr * half_track;
      cur_control_var.ar =
          mpc_return.stages.at(i).uk.acc + mpc_return.stages.at(i).uk.dr * half_track;
      res->control_vars.push_back(cur_control_var);
    }
    res->success = true;
    if (_verbose >= 1) std::cout << "Time cost: " << time_cost << std::endl;
  }

  nav_msgs::msg::Path refline_line = _planner->GetAStarSmoothPath();
  refline_line.header.stamp = _node->now();
  if (!refline_line.poses.empty()) {
    astar_path_pub->publish(refline_line);
  }
}

void TestRun() {}

}  // namespace robot_plann

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
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

  auto plann_srv = robot_plann::_node->create_service<interfaces::srv::OcpLocalPlann>(
      "/ocp_plann", robot_plann::PlannSrvCallback);

  robot_plann::astar_path_pub =
      robot_plann::_node->create_publisher<nav_msgs::msg::Path>("/a_star_path", 1);

  rclcpp::spin(robot_plann::_node);
  rclcpp::shutdown();
  return 0;
}
