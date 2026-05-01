#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/joy.hpp"

class JoyControlNode : public rclcpp::Node {
public:
    JoyControlNode() : Node("joy_control") {
        this->declare_parameter("linear_speed_limit", 0.3);
        this->declare_parameter("angular_speed_limit", 0.5);

        linear_speed_limit_ = this->get_parameter("linear_speed_limit").as_double();
        angular_speed_limit_ = this->get_parameter("angular_speed_limit").as_double();

        // Topic Twist cho STM32
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/diff_cont/cmd_vel_unstamped", 10);

        joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", 10, std::bind(&JoyControlNode::joy_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Joy Control Node: DỪNG TỨC THÌ (No Smoothing) - Topic: /diff_cont/cmd_vel_unstamped");
    }

private:
    void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg) {
        auto message = geometry_msgs::msg::Twist();

        // Nếu giữ L1 (nút index 7 trên Lenovo S04), lấy vận tốc từ cần gạt
        if (msg->buttons.size() > 7 && msg->buttons[7] == 1) {
            if (msg->axes.size() >= 2) {
                message.linear.x = msg->axes[1] * linear_speed_limit_;
                message.angular.z = msg->axes[0] * angular_speed_limit_;
            }
        } else {
            // Không nhấn L1 thì mặc định về 0 để dừng hẳn ngay lập tức
            message.linear.x = 0.0;
            message.angular.z = 0.0;
        }

        cmd_vel_pub_->publish(message);
    }

    double linear_speed_limit_;
    double angular_speed_limit_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<JoyControlNode>());
    rclcpp::shutdown();
    return 0;
}