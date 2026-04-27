#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

#include <rclcpp/rclcpp.hpp>
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
double _goal_stop_distance = 0.1;
std::mutex _planner_mutex;
std::atomic<bool> _shutting_down{false};

namespace {
robot_plann::ReferencePlannerParams::Ptr BuildPlannerParamsFromNode(
    const rclcpp::Node::SharedPtr &node) {
  auto params = std::make_shared<robot_plann::ReferencePlannerParams>();
  params->horizon_steps = static_cast<uint16_t>(
    node->declare_parameter<int>("planner.reference_horizon_steps", 20));
  params->dt = node->declare_parameter<double>("planner.reference_dt", 0.1);
  params->max_linear_vel =
    node->declare_parameter<double>("constraints.max_linear_vel", robot_plann::kMaxLinearVel);
  params->max_angular_vel =
    node->declare_parameter<double>("constraints.max_angular_vel", robot_plann::kMaxAngularVel);
  params->max_linear_acc =
    node->declare_parameter<double>("constraints.max_linear_acc", robot_plann::kMaxLinearAcc);
  params->max_angular_acc =
    node->declare_parameter<double>("constraints.max_angular_acc", robot_plann::kMaxAngularAcc);
  params->local_obst_num = node->declare_parameter<int>("planner.local_obstacle_count", 8);
  params->polygon_clearance_margin =
      node->declare_parameter<double>("planner.polygon_clearance_margin", 0.03);
  params->wall_constraint_activation_distance =
      node->declare_parameter<double>("planner.wall_constraint_activation_distance", 2.0);

  if (params->horizon_steps < 2) {
    RCLCPP_WARN(node->get_logger(), "planner.reference_horizon_steps must be >= 2, forcing 2");
    params->horizon_steps = 2;
  }
  if (params->dt <= 0.0) {
    RCLCPP_WARN(node->get_logger(), "planner.reference_dt must be > 0, fallback to %.3f", robot_plann::kDT);
    params->dt = robot_plann::kDT;
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
  if (params->polygon_clearance_margin < 0.0) {
    params->polygon_clearance_margin = 0.0;
  }
  if (params->wall_constraint_activation_distance < 0.0) {
    params->wall_constraint_activation_distance = 0.0;
  }

  return params;
}
}  // namespace


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

  bool ref_ok = false;
  {
    std::lock_guard<std::mutex> lock(_planner_mutex);
    ref_ok = _planner->UpdateReferenceOnly(ob_state, sub_goal);
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
  
  for (const auto &pose : refline_line.poses) {
    interfaces::msg::Point pt;
    pt.x = pose.pose.position.x;
    pt.y = pose.pose.position.y;
    res->astar_path.push_back(pt);
  }

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
  auto planner_params = robot_plann::BuildPlannerParamsFromNode(robot_plann::_node);

  robot_plann::_planner = std::make_unique<robot_plann::Planner>(
      planner_params, robot_plann::_verbose);

    const std::string reference_mode = robot_plann::_node->declare_parameter<std::string>(
      "planner.reference_mode", "astar_subgoal");
    double debug_xref_v_max = robot_plann::_node->declare_parameter<double>(
      "planner.debug_xref_v_max", 0.6);
    double debug_xref_v_min = robot_plann::_node->declare_parameter<double>(
      "planner.debug_xref_v_min", 0.05);
    double debug_xref_slowdown_distance = robot_plann::_node->declare_parameter<double>(
      "planner.debug_xref_slowdown_distance", 1.5);
    double debug_xref_kp_dist = robot_plann::_node->declare_parameter<double>(
      "planner.debug_xref_kp_dist", 0.8);

    if (debug_xref_v_max < 0.0) debug_xref_v_max = 0.0;
    if (debug_xref_v_min < 0.0) debug_xref_v_min = 0.0;
    if (debug_xref_slowdown_distance <= 0.0) debug_xref_slowdown_distance = 1.5;
    if (debug_xref_kp_dist < 0.0) debug_xref_kp_dist = 0.0;

    robot_plann::_planner->SetReferenceMode(reference_mode);
    robot_plann::_planner->SetDebugDirectXrefParams(
      debug_xref_v_max,
      debug_xref_v_min,
      debug_xref_slowdown_distance,
      debug_xref_kp_dist);

  RCLCPP_INFO(
      robot_plann::_node->get_logger(),
      "Reference planner config: horizon=%u dt=%.3f max_v=%.2f max_a=%.2f local_obst_num=%d",
      planner_params->horizon_steps,
      planner_params->dt,
      planner_params->max_linear_vel,
      planner_params->max_linear_acc,
      planner_params->local_obst_num);
  RCLCPP_INFO(
      robot_plann::_node->get_logger(),
      "Planner reference mode: %s (debug_xref: v_max=%.2f v_min=%.2f slowdown=%.2f kp=%.2f)",
      reference_mode.c_str(),
      debug_xref_v_max,
      debug_xref_v_min,
      debug_xref_slowdown_distance,
      debug_xref_kp_dist);

      const std::string local_costmap_topic =
        robot_plann::_node->declare_parameter<std::string>(
          "planner.local_costmap_topic", "/a_star/local_costmap");
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
      RCLCPP_INFO(
        robot_plann::_node->get_logger(),
        "Planner node publishes A* reference paths only; Nav2 FollowPath/MPPI owns cmd_vel.");
      RCLCPP_INFO(
        robot_plann::_node->get_logger(),
        "Goal stop distance enabled: %.3f m",
        robot_plann::_goal_stop_distance);

        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
        executor.add_node(robot_plann::_node);
        executor.spin();

        robot_plann::_shutting_down.store(true);

        robot_plann::astar_path_pub.reset();
        robot_plann::local_costmap_pub.reset();

        executor.cancel();
        executor.remove_node(robot_plann::_node);
        robot_plann::_planner.reset();
        robot_plann::_node.reset();

  rclcpp::shutdown();
  return 0;
}
