// Copyright (c) 2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nav2_mppi_controller/critics/dynamic_human_critic.hpp"

#include <algorithm>
#include <cmath>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xoperation.hpp"
#include "xtensor/xreducer.hpp"
#include "xtensor/xview.hpp"

namespace mppi::critics
{

using xt::evaluation_strategy::immediate;feat

void DynamicHumanCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(weight_, "cost_weight", 1.0f);
  getParam(collision_cost_, "collision_cost", 100000.0f);
  getParam(safe_margin_, "safe_margin", 0.20f);
  getParam(data_timeout_, "data_timeout", 0.5f);
  getParam(human_topic_, "human_topic", std::string("/tracking/humans"));

  robot_radius_ = static_cast<float>(costmap_ros_->getRobotRadius());
  getParam(robot_radius_, "robot_radius", robot_radius_);

  auto node = parent_.lock();
  if (!node) {
    throw std::runtime_error("DynamicHumanCritic failed to lock parent lifecycle node");
  }

  humans_sub_ = node->create_subscription<interfaces::msg::HumanArray>(
    human_topic_, rclcpp::SystemDefaultsQoS(),
    std::bind(&DynamicHumanCritic::humansCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    logger_,
    "DynamicHumanCritic subscribed to %s with weight=%f collision_cost=%f "
    "safe_margin=%f robot_radius=%f data_timeout=%f.",
    human_topic_.c_str(), weight_, collision_cost_, safe_margin_, robot_radius_, data_timeout_);
}

void DynamicHumanCritic::humansCallback(const interfaces::msg::HumanArray::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(humans_mutex_);
  latest_humans_ = *msg;
  has_humans_ = true;
}

std::vector<DynamicHumanCritic::Human> DynamicHumanCritic::getHumansInFrame(
  const std::string & target_frame)
{
  interfaces::msg::HumanArray humans_msg;
  {
    std::lock_guard<std::mutex> lock(humans_mutex_);
    if (!has_humans_) {
      return {};
    }
    humans_msg = latest_humans_;
  }

  auto node = parent_.lock();
  if (!node) {
    return {};
  }

  if (data_timeout_ > 0.0f && humans_msg.header.stamp.sec != 0) {
    const auto age = node->now() - rclcpp::Time(humans_msg.header.stamp);
    if (age.seconds() > data_timeout_) {
      return {};
    }
  }

  const std::string source_frame =
    humans_msg.header.frame_id.empty() ? target_frame : humans_msg.header.frame_id;

  geometry_msgs::msg::TransformStamped transform;
  const bool needs_transform = source_frame != target_frame;
  if (needs_transform) {
    try {
      transform = costmap_ros_->getTfBuffer()->lookupTransform(
        target_frame, source_frame, tf2::TimePointZero,
        tf2::durationFromSec(0.02));
    } catch (tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        logger_, *node->get_clock(), 2000,
        "DynamicHumanCritic waiting for transform %s -> %s: %s",
        source_frame.c_str(), target_frame.c_str(), ex.what());
      return {};
    }
  }

  std::vector<Human> humans;
  humans.reserve(humans_msg.humans.size());

  for (const auto & src : humans_msg.humans) {
    Human human;
    human.radius = static_cast<float>(std::max(0.0, src.radius));

    if (needs_transform) {
      geometry_msgs::msg::PointStamped point_in;
      point_in.header = humans_msg.header;
      point_in.point.x = src.px;
      point_in.point.y = src.py;
      point_in.point.z = 0.0;

      geometry_msgs::msg::Vector3Stamped velocity_in;
      velocity_in.header = humans_msg.header;
      velocity_in.vector.x = src.vx;
      velocity_in.vector.y = src.vy;
      velocity_in.vector.z = 0.0;

      geometry_msgs::msg::PointStamped point_out;
      geometry_msgs::msg::Vector3Stamped velocity_out;
      tf2::doTransform(point_in, point_out, transform);
      tf2::doTransform(velocity_in, velocity_out, transform);

      human.px = static_cast<float>(point_out.point.x);
      human.py = static_cast<float>(point_out.point.y);
      human.vx = static_cast<float>(velocity_out.vector.x);
      human.vy = static_cast<float>(velocity_out.vector.y);
    } else {
      human.px = static_cast<float>(src.px);
      human.py = static_cast<float>(src.py);
      human.vx = static_cast<float>(src.vx);
      human.vy = static_cast<float>(src.vy);
    }

    humans.push_back(human);
  }

  return humans;
}

void DynamicHumanCritic::score(CriticData & data)
{
  if (!enabled_) {
    return;
  }

  const std::string target_frame = data.state.pose.header.frame_id.empty() ?
    costmap_ros_->getGlobalFrameID() : data.state.pose.header.frame_id;
  const auto humans = getHumansInFrame(target_frame);
  if (humans.empty()) {
    return;
  }

  const auto time_steps = data.trajectories.x.shape(1);
  const auto time = xt::arange<float>(0.0f, static_cast<float>(time_steps)) * data.model_dt;
  xt::xtensor<float, 1> total_human_cost = xt::zeros<float>({data.costs.shape(0)});

  for (const auto & human : humans) {
    const auto human_x = human.px + human.vx * time;
    const auto human_y = human.py + human.vy * time;

    const float collision_radius = human.radius + robot_radius_ + safe_margin_;
    const float collision_radius_sq = collision_radius * collision_radius;

    const auto dx = data.trajectories.x - human_x;
    const auto dy = data.trajectories.y - human_y;
    const auto dist_sq = dx * dx + dy * dy;

    auto collision_costs = xt::where(
      dist_sq < collision_radius_sq,
      collision_cost_,
      0.0f);

    total_human_cost += xt::sum(collision_costs, {1}, immediate);
  }

  data.costs += weight_ * total_human_cost;
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(mppi::critics::DynamicHumanCritic, mppi::critics::CriticFunction)
