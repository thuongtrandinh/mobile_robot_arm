#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp> // Chỉ cần Twist cho STM32
#include <termios.h>
#include <unistd.h>
#include <thread>
#include <atomic>

class KeyboardInput : public rclcpp::Node
{
public:
    KeyboardInput();
    ~KeyboardInput();
    void publishCmd(double linear, double angular);
    std::atomic<bool> running_;

private:
    void keyboardLoop();

    // Đã sửa từ TwistStamped thành Twist để khớp với KeyboardInput.cpp và STM32
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    
    std::thread input_thread_;
    struct termios oldt_;
};