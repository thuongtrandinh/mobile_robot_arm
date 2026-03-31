#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

class JointStateConverter : public rclcpp::Node
{
public:
    JointStateConverter() : Node("joint_state_converter")
    {
        sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "/sensor_data", 10,
            std::bind(&JointStateConverter::callback, this, std::placeholders::_1));

        pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);

        last_time_ = this->now();

        pos_left_ = 0.0;
        pos_right_ = 0.0;

        RCLCPP_INFO(this->get_logger(), "Joint State Converter Started");
    }

private:
    void callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
    {
        // ======= 1. Lấy dữ liệu =======
        float v_right  = msg->data[0];   // m/s
        float v_left = msg->data[1];   // m/s

        // ======= 2. Time =======
        auto now = this->now();
        double dt = (now - last_time_).seconds();
        last_time_ = now;

        if (dt <= 0.0) return;

        // ======= 3. Convert m/s → rad/s =======
        const double R = 0.05;   // bán kính bánh

        double wl = v_left  / R;
        double wr = v_right / R;

        // ======= 4. Integrate → position =======
        pos_left_  += wl * dt;
        pos_right_ += wr * dt;

        // ======= 5. Publish JointState =======
        sensor_msgs::msg::JointState js;

        js.header.stamp = now;

        js.name = {"left_wheel_joint", "right_wheel_joint"};

        js.position = {pos_left_, pos_right_};
        js.velocity = {wl, wr};

        pub_->publish(js);
    }

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;

    rclcpp::Time last_time_;

    double pos_left_;
    double pos_right_;
};