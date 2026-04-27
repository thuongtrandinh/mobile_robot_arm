#ifndef INC_DATA_STRUCT_H_
#define INC_DATA_STRUCT_H_

#include <stdint.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <rmw_microros/rmw_microros.h>
#include <geometry_msgs/msg/twist_stamped.h>
#include <sensor_msgs/msg/joint_state.h> // Khai báo chuẩn JointState

// ---------------------------------------------------------
// THÔNG SỐ VẬT LÝ ROBOT
// ---------------------------------------------------------
#define PPR             2400.0f  // Xung/vòng (đã nhân 4)
#define Ts              0.01f    // Chu kỳ PID (10ms)
#define wheel_base      0.48f    // Khoảng cách 2 bánh (m)
#define wheel_radius    0.05f    // Bán kính bánh xe (m)

// ---------------------------------------------------------
// BIẾN TOÀN CỤC PID & ENCODER
// ---------------------------------------------------------
extern int32_t enc_left, pre_enc_left, delta_left;
extern int32_t enc_right, pre_enc_right, delta_right;

extern float speed_left, speed_right;
extern float setpoint_left, setpoint_right;

// Hệ số PID (Giữ nguyên theo yêu cầu)
extern float Kp_L, Ki_L, Kd_L;
extern float Kp_R, Ki_R, Kd_R;

// ---------------------------------------------------------
// BIẾN MICRO-ROS
// ---------------------------------------------------------
extern rcl_publisher_t joint_pub;
extern sensor_msgs__msg__JointState joint_msg;
extern double pos_left, pos_right; // Tích phân vị trí (rad)

extern rcl_subscription_t cmd_vel_sub;
extern geometry_msgs__msg__TwistStamped cmd_vel_msg;
extern rclc_executor_t executor;

#endif /* INC_DATA_STRUCT_H_ */
