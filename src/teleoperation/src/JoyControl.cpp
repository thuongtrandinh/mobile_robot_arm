#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "sensor_msgs/msg/joy.hpp"

class JoyControlNode : public rclcpp::Node {
public:
    JoyControlNode() : Node("joy_control") {
        // Khai báo giới hạn tốc độ theo yêu cầu của bạn
        this->declare_parameter("linear_speed_limit", 0.3);
        this->declare_parameter("angular_speed_limit", 0.5);
        // Tham số bộ lọc làm mượt (0.05-0.2 là tối ưu, nhỏ hơn = mượt hơn)
        this->declare_parameter("smoothing_factor", 0.1);

        linear_speed_limit_ = this->get_parameter("linear_speed_limit").as_double();
        angular_speed_limit_ = this->get_parameter("angular_speed_limit").as_double();
        smoothing_factor_ = this->get_parameter("smoothing_factor").as_double();

        // Khởi tạo vận tốc hiện tại
        current_linear_ = 0.0;
        current_angular_ = 0.0;

        // Topic TwistStamped để khớp với STM32 và Odom calculator
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/diff_cont/cmd_vel", 10);

        joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
            "/joy", 10, std::bind(&JoyControlNode::joy_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Joy Control Node: GIỮ L1 ĐỂ ĐIỀU KHIỂN (v_max=0.3m/s) - SMOOTH STOP");
    }

private:
    void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg) {
        // Tính toán vận tốc mục tiêu từ input joystick
        double target_linear = 0.0;
        double target_angular = 0.0;

        // Trên tay cầm Lenovo S04: 
        // Nút L1 thường có ID là 7 (kiểm tra nếu nút L1 đang được giữ)
        if (msg->buttons.size() > 7 && msg->buttons[7] == 1) {
            if (msg->axes.size() >= 2) {
                // Cần gạt trái: axes[1] là tiến/lùi, axes[0] là xoay
                target_linear = msg->axes[1] * linear_speed_limit_;
                target_angular = msg->axes[0] * angular_speed_limit_;
            }
        }

        // ===== Áp dụng bộ lọc làm mượt (Exponential Smoothing) =====
        // Công thức: new_value = old_value + smoothing_factor * (target - old_value)
        // Khi smoothing_factor nhỏ → chuyển động mượt hơn
        // Khi smoothing_factor lớn → chuyển động nhanh hơn
        current_linear_ = current_linear_ + smoothing_factor_ * (target_linear - current_linear_);
        current_angular_ = current_angular_ + smoothing_factor_ * (target_angular - current_angular_);

        auto message = geometry_msgs::msg::TwistStamped();
        message.header.stamp = this->get_clock()->now();
        message.header.frame_id = "base_link";

        message.twist.linear.x = current_linear_;
        message.twist.angular.z = current_angular_;

        cmd_vel_pub_->publish(message);
    }

    double current_linear_;
    double current_angular_;
    double smoothing_factor_;
    double linear_speed_limit_;
    double angular_speed_limit_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_vel_pub_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<JoyControlNode>());
    rclcpp::shutdown();
    return 0;
}
