#include "teleoperation/KeyboardInput.h"
#include <iostream>
#include <csignal>
#include <atomic>

static KeyboardInput* g_keyboard_input_ptr = nullptr;

void sigintHandler(int signum)
{
    if (g_keyboard_input_ptr) {
        g_keyboard_input_ptr->publishCmd(0.0, 0.0);
        g_keyboard_input_ptr->running_ = false;
    }
    std::cout << "\n[INFO] Robot stopped. Exiting...\n";
    std::exit(0);
}

KeyboardInput::KeyboardInput() : Node("keyboard_input"), running_(true)
{
    cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/diff_cont/cmd_vel_unstamped", 10);

    tcgetattr(STDIN_FILENO, &oldt_);
    struct termios newt = oldt_;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    g_keyboard_input_ptr = this;
    std::signal(SIGINT, sigintHandler);

    input_thread_ = std::thread(&KeyboardInput::keyboardLoop, this);
    RCLCPP_INFO(this->get_logger(), "Keyboard teleop started. WASD to move, Space to STOP. Topic: /diff_cont/cmd_vel_unstamped");
}

KeyboardInput::~KeyboardInput()
{
    running_ = false;
    if (input_thread_.joinable())
        input_thread_.join();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt_);
    g_keyboard_input_ptr = nullptr;
}

void KeyboardInput::keyboardLoop()
{
    char c;
    while (running_)
    {
        c = getchar();
        if (c == 'q' || c == 'Q')
        {
            publishCmd(0.0, 0.0); // Dừng trước khi thoát
            running_ = false;
            break;
        }

        double linear = 0.0;
        double angular = 0.0;

        switch (c)
        {
            case 'w': case 'W': linear = 0.2; break;
            case 's': case 'S': linear = -0.2; break;
            case 'a': case 'A': angular = 0.2; break;
            case 'd': case 'D': angular = -0.2; break;
            case ' ': linear = 0.0; angular = 0.0; break;
            default: continue; 
        }
        // Gửi lệnh ngay lập tức (không lưu biến current_ để tránh trôi)
        publishCmd(linear, angular);
    }
}

void KeyboardInput::publishCmd(double linear, double angular)
{
    geometry_msgs::msg::Twist msg;
    msg.linear.x = linear;
    msg.angular.z = angular;
    cmd_pub_->publish(msg);
}