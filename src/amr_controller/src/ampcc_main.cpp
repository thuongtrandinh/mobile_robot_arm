// Copyright 2026 H-AMPCC Authors
// SPDX-License-Identifier: Apache-2.0

#include "amr_controller/ampcc_node.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<amr_controller::AMPCCNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
