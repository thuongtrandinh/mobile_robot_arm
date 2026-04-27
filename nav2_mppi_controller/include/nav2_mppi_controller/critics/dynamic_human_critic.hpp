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

#ifndef NAV2_MPPI_CONTROLLER__CRITICS__DYNAMIC_HUMAN_CRITIC_HPP_
#define NAV2_MPPI_CONTROLLER__CRITICS__DYNAMIC_HUMAN_CRITIC_HPP_

#include <mutex>
#include <string>
#include <vector>

#include "interfaces/msg/human_array.hpp"
#include "interfaces/msg/human_state.hpp"
#include "nav2_mppi_controller/critic_function.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace mppi::critics
{

class DynamicHumanCritic : public CriticFunction
{
public:
  void initialize() override;
  void score(CriticData & data) override;

protected:
  struct Human
  {
    float px{0.0f};
    float py{0.0f};
    float vx{0.0f};
    float vy{0.0f};
    float radius{0.0f};
  };

  void humansCallback(const interfaces::msg::HumanArray::SharedPtr msg);
  std::vector<Human> getHumansInFrame(const std::string & target_frame);

  rclcpp::Subscription<interfaces::msg::HumanArray>::SharedPtr humans_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_pub_;

  std::mutex humans_mutex_;
  interfaces::msg::HumanArray latest_humans_;
  bool has_humans_{false};

  float weight_{0.0f};
  float collision_cost_{0.0f};
  float safe_margin_{0.0f};
  float robot_radius_{0.0f};
  float human_min_radius_{0.0f};
  float human_max_radius_{0.0f};
  float data_timeout_{0.0f};
  std::string human_topic_;
  std::string debug_topic_;
  bool publish_debug_{false};
};

}  // namespace mppi::critics

#endif  // NAV2_MPPI_CONTROLLER__CRITICS__DYNAMIC_HUMAN_CRITIC_HPP_
