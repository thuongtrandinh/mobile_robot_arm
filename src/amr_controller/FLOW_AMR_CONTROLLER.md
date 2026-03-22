# AMR Controller Flow Guide (Chi tiết)

Tài liệu này mô tả chi tiết cách `amr_controller` hoạt động trong ROS 2, từ lúc node được launch đến lúc trả lệnh điều khiển qua service `/ocp_plann`.

---

## 1) Mục tiêu của `amr_controller`

`amr_controller` là backend local planner kiểu **A* + smoothing + velocity planning + MPC**.

Node chính (`ampcc_node`) không tự publish lệnh điều khiển theo vòng lặp định kỳ. Thay vào đó:

- Node **mở service server** `/ocp_plann`.
- Bất kỳ client nào (planner manager, behavior node, test client...) gọi service này sẽ nhận về:
  - `al`, `ar` (gia tốc bánh trái/phải tại bước đầu)
  - `control_vars[]` (chuỗi lệnh dự đoán cho toàn horizon)
  - `astar_path[]` (quỹ đạo tham chiếu đã xử lý)
  - `success` (trạng thái thành công/thất bại)

---

## 2) Các thành phần chính trong package

### 2.1 Executable

- `ampcc_node` (trong `src/main.cc`)
- Tên node runtime: `opt_planner`
- Service cung cấp: `/ocp_plann` (`amr_interfaces/srv/OcpLocalPlann`)
- Topic publish: `/a_star_path` (`nav_msgs/msg/Path`)

### 2.2 Core planner

`Planner` (trong `planner.h/.cc`) chứa các module:

- `AStar`: tìm đường trên costmap
- `SmoothCorner`: làm mượt path
- `LookAhead`: gán profile vận tốc cho path
- `Mpc`: tối ưu điều khiển có ràng buộc

### 2.3 Interface service

Request (`OcpLocalPlann.srv`):

- `amr_interfaces/JointState ob`
- `amr_interfaces/Point sub_goal`

Response:

- `bool success`
- `float64 al`
- `float64 ar`
- `amr_interfaces/Point revised_goal`
- `amr_interfaces/Point[] astar_path`
- `amr_interfaces/ControlVar[] control_vars`

---

## 3) Điều kiện để controller chạy đúng

## 3.1 Điều kiện môi trường

- ROS 2 đã cài và source đúng:
  - `source /opt/ros/$ROS_DISTRO/setup.bash`
  - `source /home/thuong/LVTN/amr_ws/install/setup.bash`
- Package đã build thành công (`amr_controller`, `amr_interfaces`)
- Runtime dependency sẵn sàng: Eigen, OpenCV, CasADi

## 3.2 Điều kiện logic dữ liệu đầu vào

Client gọi `/ocp_plann` nên đảm bảo:

- `robot_state.pose.{x,y,theta}` hợp lệ
- `vl`, `vr`, `radius` hợp lệ (radius > 0)
- `gx`, `gy` (global goal) có ý nghĩa
- `sub_goal` không quá gần robot và không ngoài biên map quá mức
- Danh sách vật cản/people/polygon/walls đúng định dạng

Nếu dữ liệu đầu vào không hợp lệ hoặc rơi vào hình học không khả thi, planner có thể trả `success=false`.

---

## 4) Launch và lifecycle runtime

Launch file: `launch/ampcc_controller.launch.py`

Arguments:

- `use_sim_time` (default: `false`)
- `respawn` (default: `false`)
- `log_level` (default: `info`)

Lệnh chạy:

```bash
cd /home/thuong/LVTN/amr_ws
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --packages-select amr_controller
source install/setup.bash
ros2 launch amr_controller ampcc_controller.launch.py
```

Node lên thành công khi:

- `ros2 service list` có `/ocp_plann`
- `ros2 node list` có `/opt_planner`

---

## 5) Flow xử lý chi tiết bên trong callback `/ocp_plann`

Toàn bộ flow nằm trong `PlannSrvCallback` (`main.cc`) và `Planner::PlannExec` (`planner.cc`).

## Bước 1: Nhận request và map sang trạng thái nội bộ

Từ request:

- Lấy trạng thái robot từ `ob.robot_state`
- Chuyển sang dạng internal `RobotState`

Quy đổi vận tốc:

- $v = 0.5(v_r + v_l)$
- $yaw\_rate = 0.5(v_r - v_l) / radius$

Đồng thời copy:

- `human_states` -> mảng dynamic obstacles
- `obstacle_states` -> mảng static/circle obstacles
- `poly_states` -> polygon vertices
- `walls` -> segment walls (được clip theo biên map)

`sub_goal` được nhận từ request để làm đích local trước khi vào MPC.

## Bước 2: Gọi planner chính

Callback gọi:

```cpp
auto mpc_return = _planner->PlannExec(ob_state, sub_goal);
```

Trong `PlannExec`, pipeline thực tế là:

1. `UpdateCostMap(state)`
2. `CheckAround(start_pt)` (thoát vùng kẹt nếu đang trong vật cản)
3. `CheckNavGoal(sub_goal, goal_pt, start_pt, polygon)`
4. A* search `SearchPath(cost_map_, start_pt, sub_goal)`
5. Smoothing `SmoothSharpCorner(...)`
6. Velocity profile `UpdateVelocity(...)`
7. MPC solve `RunMpc(revised_state, final_path)`

## Bước 3: UpdateCostMap

Costmap được clone từ map nền tạo sẵn (`CreateMap`). Sau đó:

- Vẽ walls
- Vẽ polygon contour
- Vẽ obstacle circle
- Erode map theo bán kính inflate (`kInflationRadius`) để đảm bảo khoảng an toàn

Ý nghĩa: tăng “biên an toàn”, giúp đường A* và MPC không cắt sát vật cản.

## Bước 4: Validate và hiệu chỉnh sub-goal

`CheckNavGoal` xử lý nhiều trường hợp:

- Nếu goal ngoài biên map: ray-cast để kéo goal vào trong biên
- Nếu goal nằm trong polygon: dịch goal sang vị trí lân cận hợp lệ
- Nếu goal rơi vào vùng occupied: thử các điểm xung quanh (circular candidates)

Nếu không tìm được goal hợp lệ -> planner fail sớm (`success=false`).

## Bước 5: A* + smoothing + lookahead

- A* tạo quỹ đạo rời rạc từ `start_pt` -> `sub_goal`
- Smoother làm mượt các góc gắt
- LookAhead gán vận tốc mong muốn trên mỗi điểm

Kết quả thành `final_path` là quỹ đạo tham chiếu cho MPC.

## Bước 6: Chuẩn bị trạng thái cho MPC

Planner có xử lý hướng chuyển động tiến/lùi:

- So yaw hiện tại với hướng đoạn path đầu
- Nếu lệch lớn hơn ~90 độ, hệ quy đổi trạng thái để đi lùi hợp lý

Sau đó pad `final_path` lên đủ `kNP` bước dự đoán (nếu path ngắn).

## Bước 7: MPC solve

`Mpc::RunMpc` thực hiện:

- Build obstacle prediction (human có vận tốc, obstacle tĩnh)
- Chọn local reference theo horizon
- Tạo initial guess
- Giải bài toán tối ưu với ràng buộc động học và giới hạn điều khiển

Một số ràng buộc quan trọng (theo code hiện tại):

- Giới hạn điều khiển bánh trái/phải thông qua tổ hợp $v$ và $r$
- Ràng buộc động học robot theo mô hình vi phân + tích phân rời rạc (RK2)
- Ràng buộc tường/biên nếu có walls

Nếu solver thành công -> `mpc_return.success=true`, ngược lại false.

## Bước 8: Tạo response trả cho client

Trong callback service:

- `res->astar_path` được điền từ `GetAStarPath()`
- `res->control_vars[]` điền theo toàn bộ horizon
- Lệnh bước đầu:
  - `al = acc - dr * 0.3`
  - `ar = acc + dr * 0.3`

Nếu MPC fail:

- Controller trả lệnh “brake-like fallback” dựa trên vận tốc bánh hiện tại
- `success=false`

Nếu thành công:

- Trả `success=true`
- In `Time cost` ra console

Song song, node publish `nav_msgs/Path` lên `/a_star_path` để quan sát/debug.

---

## 6) Sơ đồ luồng dữ liệu (thực thi runtime)

1. Client -> gửi request `/ocp_plann`
2. `opt_planner` callback -> convert dữ liệu request
3. `Planner::PlannExec` -> map update + goal check + A* + smooth + velocity + MPC
4. Callback -> build response (`al/ar`, `control_vars`, `astar_path`, `success`)
5. Client nhận response -> áp dụng điều khiển cho tầng thấp hơn
6. Đồng thời `/a_star_path` được publish cho RViz/logging

---

## 7) Các trường hợp fail thường gặp và cách hiểu

### 7.1 `Sub goal is unreachable`

Nguyên nhân:

- goal ngoài map quá mức
- goal trong vùng cấm/polygon mà không có candidate hợp lệ

Hướng xử lý:

- giảm step sub-goal
- kiểm tra nguồn cấp `sub_goal` từ planner cấp trên

### 7.2 `Invalid A* path with len < 2`

Nguyên nhân:

- start/goal không kết nối được trên costmap
- map bị bít do inflation quá lớn hoặc vật cản dữ

Hướng xử lý:

- kiểm tra walls/polygon input
- kiểm tra tham số map/inflation

### 7.3 `Ocp plann failed`

Nguyên nhân:

- NLP solver không hội tụ
- initial guess hoặc constraints quá chặt

Hướng xử lý:

- hạ yêu cầu cứng (constraint), chỉnh trọng số cost
- kiểm tra dữ liệu obstacle dự đoán

---

## 8) Quy trình tích hợp chuẩn trong hệ thống lớn

Một flow tích hợp khuyến nghị:

1. Global planner / behavior planner sinh `sub_goal` ở tần số thấp-vừa
2. Gọi `/ocp_plann` theo chu kỳ control (hoặc event-driven)
3. Lấy `al/ar` bước đầu để gửi xuống tầng điều khiển bánh
4. Theo dõi `success`; nếu false liên tiếp thì kích hoạt fallback behavior
5. Dùng `/a_star_path` để giám sát chất lượng plan trong RViz

Lưu ý: package này là **local planning service backend**, không phải toàn bộ stack điều khiển đóng vòng cảm biến-động cơ.

---

## 9) Checklist debug nhanh

- [ ] `ros2 service list | grep /ocp_plann` thấy service
- [ ] `ros2 interface show amr_interfaces/srv/OcpLocalPlann` đúng schema
- [ ] Request gửi lên có `radius > 0`, pose và goal hợp lệ
- [ ] `/a_star_path` có dữ liệu khi gọi service
- [ ] Console không spam lỗi A* hoặc MPC fail

---

## 10) Gợi ý mở rộng tiếp theo

- Thêm node test client để replay dữ liệu thực tế vào `/ocp_plann`
- Tách tham số cứng (`kDT`, `kNP`, trọng số cost, bound vận tốc) thành YAML + launch parameters
- Thêm diagnostics topic (latency, fail reason, solver status)

Nếu cần, có thể tạo tiếp một tài liệu thứ hai: **"Cách viết client gọi /ocp_plann + ví dụ request mẫu"** để test end-to-end ngay.
