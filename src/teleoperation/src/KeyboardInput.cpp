
#include "teleoperation/KeyboardInput.h"
#include <iostream>
#include <csignal>
#include <atomic>

// Forward declaration for signal handler
static KeyboardInput* g_keyboard_input_ptr = nullptr;

void sigintHandler(int signum)
{
    if (g_keyboard_input_ptr) {
        g_keyboard_input_ptr->publishCmd(0.0, 0.0);
        g_keyboard_input_ptr->running_ = false;
    }
    std::cout << "\n[INFO] Ctrl+C pressed. Robot stopped. Exiting...\n";
    std::exit(0);
}



KeyboardInput::KeyboardInput() : Node("keyboard_input"), running_(true)
{
    cmd_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/diff_cont/cmd_vel", 10);

    // Set terminal to raw mode
    tcgetattr(STDIN_FILENO, &oldt_);
    struct termios newt = oldt_;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    // Set global pointer for signal handler
    g_keyboard_input_ptr = this;
    std::signal(SIGINT, sigintHandler);

    input_thread_ = std::thread(&KeyboardInput::keyboardLoop, this);
    RCLCPP_INFO(this->get_logger(), "Keyboard teleop started. Use WASD to move, Q to quit. Press Ctrl+C to stop robot and exit.");
}

KeyboardInput::~KeyboardInput()
{
    running_ = false;
    if (input_thread_.joinable())
        input_thread_.join();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt_);
    // Reset global pointer
    g_keyboard_input_ptr = nullptr;
}

void KeyboardInput::keyboardLoop()
{
    double current_linear = 0.0;
    double current_angular = 0.0;
    char c;
    while (running_)
    {
        c = getchar();
        if (c == 'q' || c == 'Q')
        {
            running_ = false;
            break;
        }
        switch (c)
        {
        case 'w':
        case 'W':
            current_linear = 0.2; // Tiến, giữ nguyên góc quay hiện tại
            publishCmd(current_linear, current_angular);
            break;
        case 's':
        case 'S':
            current_linear = -0.2; // Lùi, giữ nguyên góc quay hiện tại
            publishCmd(current_linear, current_angular);
            break;
        case 'a':
        case 'A':
            current_angular = 0.2; // Quay trái, giữ nguyên tốc độ tiến/lùi
            publishCmd(current_linear, current_angular);
            break;
        case 'd':
        case 'D':
            current_angular = -0.2; // Quay phải, giữ nguyên tốc độ tiến/lùi
            publishCmd(current_linear, current_angular);
            break;
        case ' ':
            current_linear = 0.0;
            current_angular = 0.0;
            publishCmd(current_linear, current_angular); // Dừng
            break;
        default:
            break;
        }
    }
}

void KeyboardInput::publishCmd(double linear, double angular)
{
    geometry_msgs::msg::TwistStamped msg;
    msg.header.stamp = this->get_clock()->now();
    msg.header.frame_id = "base_link";

    msg.twist.linear.x = linear;
    msg.twist.angular.z = angular;

    cmd_pub_->publish(msg);
}

