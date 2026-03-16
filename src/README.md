# Hướng dẫn thư mục `src/`

Thư mục `src/` chứa toàn bộ các **ROS 2 package** của hệ thống robot tự hành (AMR). Mỗi package đảm nhiệm một chức năng riêng biệt trong pipeline điều hướng.

---

## Tổng quan kiến trúc

```
Cảm biến (LiDAR, IMU, Encoder)
        │
        ▼
[amr_localization]  ──  EKF fusion + AMCL
        │
        ├──── [amr_mapping]   (SLAM, nếu chưa có bản đồ)
        │
        ▼
[amr_planner]  ──  Informed-RRT* → /global_path
        │
        ▼
[amr_controller]  ──  H-AMPCC (RLS + iLQR/MPCC) → /cmd_vel
        │
        ▼
[amr_stm32]  ──  STM32F411 firmware PID → Động cơ bánh xe
        │
   (phát triển)
[amr_teleop]  ──  Điều khiển bàn phím (WASD)
[amr_descriptions]  ──  Mô hình robot (URDF/Xacro) + Gazebo
[amr_bringup]  ──  Launch file tổng (thực tế / mô phỏng)
[rplidar_ros]  ──  Driver ROS 2 cho cảm biến RPLidar
```

---

## Danh sách packages

| Package | Ngôn ngữ | Mô tả ngắn |
|---|---|---|
| `amr_bringup` | Launch only | Khởi động toàn bộ hệ thống |
| `amr_controller` | C++ | Bộ điều khiển H-AMPCC (nghiên cứu chính) |
| `amr_descriptions` | C++/Python | Mô hình robot URDF + môi trường Gazebo |
| `amr_localization` | C++/Python | Định vị (EKF + AMCL) |
| `amr_mapping` | C++ | Lập bản đồ SLAM (slam_toolbox) |
| `amr_planner` | Python | Lập lộ trình toàn cục (Informed-RRT*) |
| `amr_stm32` | C (Embedded) | Firmware STM32F411 điều khiển động cơ |
| `amr_teleop` | C++ | Điều khiển robot bằng bàn phím |
| `rplidar_ros` | C++ | Driver RPLidar A2M8 (bên thứ ba) |

---

## Chi tiết từng package

---

### `amr_bringup`

**Vai trò:** Package điều phối — chỉ chứa launch files, không có source code.

**Launch files:**

| File | Mục đích |
|---|---|
| `launch/real_robot.launch.py` | Khởi động robot thực: hardware interface, RPLidar, controller, teleop, IMU, định vị |
| `launch/simulated_robot.launch.py` | Khởi động mô phỏng Gazebo: controller, teleop, định vị, RViz |

**Config:**
- `config/rplidar_a2m8.yaml` — cổng serial và baud rate cho cảm biến RPLidar.

**Tham số launch phổ biến:**

```bash
# Khởi động mô phỏng với SLAM
ros2 launch amr_bringup simulated_robot.launch.py use_slam:=true

# Khởi động robot thực với AMCL
ros2 launch amr_bringup real_robot.launch.py use_slam:=false
```

---

### `amr_controller`

**Vai trò:** Đây là **đóng góp nghiên cứu chính** của đồ án. Cài đặt bộ điều khiển **H-AMPCC (Hierarchical Adaptive Model Predictive Contouring Control)**.

**Kiến trúc nội bộ:**

```
/odom ──► AMPCCNode ──► RLSEstimator  (ước lượng tham số động lực học online)
/global_path ──►         │
/goal_pose ──►           ▼
                     AdaptiveMPCC   (mô hình robot + đường tham chiếu)
                         │
                         ▼
                     MPCCSolver     (iLQR solver)
                         │
                         ▼
               /cmd_vel, /predicted_path, /ampcc_diagnostics
```

**Các file nguồn chính:**

| File | Lớp | Mô tả |
|---|---|---|
| `src/rls_estimator.cpp` | `RLSEstimator` | Recursive Least Squares (λ=0.98) ước lượng online `[α_v, β_v, α_ω, β_ω]` |
| `src/adaptive_mpcc.cpp` | `AdaptiveMPCC`, `WaypointPath` | Mô hình robot thích nghi, tham số hóa đường theo độ dài cung |
| `src/mpcc_solver.cpp` | `MPCCSolver` | Solver iLQR-style cho bài toán MPCC, hỗ trợ warm-start |
| `src/ampcc_node.cpp` | `AMPCCNode` | ROS 2 node tích hợp, vòng lặp điều khiển 20 Hz |

**Thông số chính** (`config/ampcc_params.yaml`):

```yaml
N: 20               # Prediction horizon
dt: 0.05            # Bước thời gian (50 ms → 20 Hz)
q_c: 10.0           # Trọng số sai số contouring
q_l: 5.0            # Trọng số sai số lag
q_v: 1.0            # Trọng số tốc độ
v_ref: 0.5          # Tốc độ tham chiếu (m/s)
v_cmd_max: 2.0      # Tốc độ tuyến tính tối đa
omega_cmd_max: 2.0  # Tốc độ góc tối đa
goal_tolerance: 0.15
rls_lambda: 0.98
ilqr_max_iter: 10
```

**Topics:**

| Topic | Loại | Vai trò |
|---|---|---|
| `/odom` | Subscriber | Trạng thái robot hiện tại |
| `/global_path` | Subscriber | Đường dẫn toàn cục từ planner |
| `/goal_pose` | Subscriber | Mục tiêu cần đến |
| `/cmd_vel` | Publisher | Lệnh vận tốc gửi xuống hardware |
| `/predicted_path` | Publisher | Đường dự đoán (để visualize) |
| `/ampcc_diagnostics` | Publisher | Thông tin debug của bộ điều khiển |

---

### `amr_descriptions`

**Vai trò:** Định nghĩa mô hình robot (URDF/Xacro) và môi trường mô phỏng Gazebo.

**Thông số robot:**

| Thuộc tính | Giá trị |
|---|---|
| Kiểu dẫn động | Differential drive |
| Kích thước thân | 0.3 × 0.4 × 0.3 m |
| Khối lượng | 4 kg |
| Bán kính bánh | 0.05 m |
| Khoảng cách bánh | 0.42 m |
| Cảm biến | RPLidar + IMU (liên kết ảo) |

**Cấu trúc URDF:**

```
model/wheeled/urdf/
├── mobile_robot.urdf.xacro   ← File entry point (truyền sim_mode)
├── mobile_robot.xacro        ← Định nghĩa thân, bánh xe, cảm biến
└── gazebo.xacro              ← Plugin Gazebo, ma sát, ros2_control
```

**Môi trường mô phỏng** (`worlds/`):

| World | Mô tả |
|---|---|
| `empty.world` | Sân trống |
| `room_20x20.world` | Phòng 20×20 m (mặc định) |
| `small_house.world` | Ngôi nhà nhỏ |
| `small_warehouse.world` | Kho hàng nhỏ |

**Khởi động Gazebo:**

```bash
ros2 launch amr_descriptions gazebo.launch.py world:=small_warehouse
```

---

### `amr_localization`

**Vai trò:** Ước lượng vị trí robot trong bản đồ đã biết.

**Pipeline 2 tầng:**

```
IMU (/imu) ──────────────┐
                          ├──► EKF (robot_localization) ──► /odometry/filtered
Encoder (/diff_cont/odom) ┘         TF: odom → base_footprint

LiDAR (/scan) + Bản đồ ──► AMCL (nav2_amcl) ──► TF: map → odom
```

**Launch files:**

| File | Khi nào dùng |
|---|---|
| `launch/global_localization.launch.py` | Robot đã có bản đồ, dùng AMCL |
| `launch/local_localization.launch.py` | Đang chạy SLAM, chỉ cần EKF |

**Config quan trọng:**

- `config/ekf.yaml` — Ma trận hiệp phương sai quá trình 15×15, kết hợp IMU (yaw rate) + odom (vx).
- `config/amcl.yaml` — 200–1000 hạt, mô hình laser likelihood field, phạm vi 20 m.

**Bản đồ mặc định:** `lab208b3.yaml` (thay đổi trong launch file nếu cần).

---

### `amr_mapping`

**Vai trò:** SLAM — Lập bản đồ đồng thời + định vị khi chưa có bản đồ.

**Approach:** `slam_toolbox` chế độ đồng bộ + EKF odometry đã lọc.

**Cấu hình solver** (`config/slam_toolbox.yaml`):

```yaml
solver_plugin: solver_plugins::CeresSolver
ceres_linear_solver: SPARSE_NORMAL_CHOLESKY
ceres_preconditioner: SCHUR_JACOBI
ceres_max_num_iterations: 50
# Cập nhật sau mỗi 0.3m hoặc 0.3 rad
minimum_travel_distance: 0.3
minimum_travel_heading: 0.3
# Loop closure bật, chain size tối thiểu 15
loop_search_maximum_distance: 4.0
```

**Bản đồ đã lưu sẵn** (`maps/`): `room_20x20/`, `small_house/`, `small_warehouse/`.

**Khởi động SLAM:**

```bash
ros2 launch amr_mapping slam.launch.py
```

**Lưu bản đồ sau khi quét:**

```bash
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap \
  "name: {data: 'my_map'}"
```

---

### `amr_planner`

**Vai trò:** Lập lộ trình toàn cục từ vị trí hiện tại đến mục tiêu.

**Thuật toán:** **Informed RRT*** với lấy mẫu ellipsoid thích nghi.

**Pipeline:**

```
/map (OccupancyGrid) ──► Inflate obstacles
/goal_pose ─────────────┐
TF: map→base_footprint ─┴──► Informed-RRT* ──► Path smoothing (B-spline) ──► /global_path
```

**Thông số chính:**

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `max_iter` | 3000–5000 | Số vòng lặp RRT* tối đa |
| `step_len` | 0.5 m | Bước mở rộng cây |
| `search_radius` | 2.0 m | Bán kính tìm kiếm lân cận |
| `robot_radius` | 0.25 m | Bán kính robot cho inflation |
| `goal_sample_rate` | 0.1 | Xác suất lấy mẫu hướng goal |

**Khởi động planner:**

```bash
ros2 launch amr_planner planner_test.launch.py
# Sau đó publish goal qua RViz2 (2D Goal Pose) hoặc:
ros2 topic pub /goal_pose geometry_msgs/PoseStamped '{...}'
```

---

### `amr_stm32`

**Vai trò:** Firmware nhúng cho vi điều khiển **STM32F411VET6** (Cortex-M4) — điều khiển PID động cơ.

> **Lưu ý:** Package này **không phải ROS package**, là một dự án STM32CubeIDE riêng biệt.

**Chức năng:**
- Nhận lệnh vận tốc từ máy tính nhúng (qua UART/serial) chạy ROS 2.
- Thực hiện vòng lặp PID để điều khiển PWM cho hai động cơ DC.
- Đọc encoder để tính vận tốc thực và cấp odometry ngược lại.

**Để mở và build:** Dùng **STM32CubeIDE**, mở project tại `amr_stm32/agv_PID/`.

---

### `amr_teleop`

**Vai trò:** Điều khiển robot bằng bàn phím — dùng trong lúc phát triển và kiểm thử.

**Phím tắt:**

| Phím | Hành động |
|---|---|
| `W` | Tiến (+0.5 m/s) |
| `S` | Lùi (-0.5 m/s) |
| `A` | Quay trái (+1.0 rad/s) |
| `D` | Quay phải (-1.0 rad/s) |
| `Space` | Dừng |
| `Q` | Thoát |

**Topic publish:** `/diff_cont/cmd_vel` (geometry_msgs/TwistStamped)

**Khởi động:**

```bash
ros2 run amr_teleop keyboard_input
```

---

### `rplidar_ros`

**Vai trò:** Driver ROS 2 cho cảm biến LiDAR **SLAMTEC RPLidar A2M8** (bên thứ ba, không chỉnh sửa).

**Topic publish:** `/scan` (sensor_msgs/LaserScan)

**Khởi động:**

```bash
ros2 launch rplidar_ros rplidar_a2m8_launch.py
```

> Cấu hình cổng serial được ghi đè bởi `amr_bringup/config/rplidar_a2m8.yaml`.

---

## Build toàn bộ workspace

```bash
# Build tất cả packages
cd ~/LVTN/amr_ws
colcon build --symlink-install

# Source environment
source install/setup.bash

# Build một package cụ thể
colcon build --symlink-install --packages-select amr_controller
```

---

## Quy trình chạy điển hình

### 1. Chạy mô phỏng Gazebo (có SLAM)

```bash
source install/setup.bash
ros2 launch amr_bringup simulated_robot.launch.py use_slam:=true
```

### 2. Lập bản đồ xong → Lưu bản đồ → Chạy với AMCL

```bash
# Lưu bản đồ
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap \
  "name: {data: 'my_map'}"

# Lần sau, dùng AMCL với bản đồ đã có
ros2 launch amr_bringup simulated_robot.launch.py use_slam:=false
```

### 3. Đặt mục tiêu và để controller dẫn đường

Dùng **RViz2 → 2D Goal Pose** để publish `/goal_pose`.
Planner sẽ tính đường → Controller sẽ bám theo.

---

## Phụ thuộc ROS 2 chính

| Package | Nguồn |
|---|---|
| `robot_localization` | EKF |
| `nav2_amcl` | Particle filter localization |
| `slam_toolbox` | SLAM |
| `nav2_map_server` | Map server |
| `ros_gz_bridge`, `gz_ros2_control` | Gazebo Ignition bridge |
| `Eigen3` | Tính toán ma trận (controller) |
| `scipy`, `numpy` | Planner Python |

Cài đặt dependencies:

```bash
rosdep install --from-paths src --ignore-src -r -y
```
