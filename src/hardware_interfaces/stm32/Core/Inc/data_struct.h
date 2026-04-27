#ifndef INC_DATA_STRUCT_H_
#define INC_DATA_STRUCT_H_

#include <stdint.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <rmw_microros/rmw_microros.h>
#include <geometry_msgs/msg/twist.h>
#include <sensor_msgs/msg/joint_state.h> // Khai báo chuẩn JointState

// ---------------------------------------------------------
// THÔNG SỐ VẬT LÝ ROBOT
// ---------------------------------------------------------
#define PPR             1000.0f  // Xung/vòng (đã nhân 4)
#define Ts              0.01f    // Chu kỳ PID (10ms)
#define wheel_base      0.48f    // Khoảng cách 2 bánh (m)
#define wheel_radius    0.05f    // Bán kính bánh xe (m)

// ---------------------------------------------------------
// BIẾN ENCODER & KINEMATIC DECOUPLING PID
// ---------------------------------------------------------
extern int32_t enc_left, pre_enc_left, delta_left;
extern int32_t enc_right, pre_enc_right, delta_right;

// Vận tốc bánh xe thực tế (từ encoder feedback)
extern float speed_left, speed_right;

// Vận tốc không gian điều khiển (từ forward kinematics)
extern float speed_v, speed_w;          // v (m/s), ω (rad/s)

// Setpoint từ ROS 2 cmd_vel (trực tiếp)
extern float setpoint_v, setpoint_w;    // v* (m/s), ω* (rad/s)

// Hệ số PID cho Vận tốc dài (Linear Velocity - v)
extern float Kp_v, Ki_v, Kd_v;
extern float Error_v, pre_Error_v, pre_pre_Error_v;
extern float u_v, pre_u_v;

// Hệ số PID cho Vận tốc góc (Angular Velocity - ω)
extern float Kp_w, Ki_w, Kd_w;
extern float Error_w, pre_Error_w, pre_pre_Error_w;
extern float u_w, pre_u_w;

// PWM Output cho motor
extern float duty_left, duty_right;

// ---------------------------------------------------------
// BIẾN MICRO-ROS
// ---------------------------------------------------------
extern rcl_publisher_t joint_pub;
extern sensor_msgs__msg__JointState joint_msg;
extern double pos_left, pos_right; // Tích phân vị trí (rad)

extern rcl_subscription_t cmd_vel_sub;
extern geometry_msgs__msg__Twist cmd_vel_msg;
extern rclc_executor_t executor;

#endif /* INC_DATA_STRUCT_H_ */
