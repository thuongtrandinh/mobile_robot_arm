# Mapping cong thuc MPPI sang code `nav2_mppi_controller`

Tai lieu nay map phan ly thuyet MPPI sang code trong:

- `nav2_mppi_controller`
- `src/controller/config/mppi_params.yaml`

Muc tieu la giup dua noi dung vao phan thiet ke bo dieu khien MPPI trong luan van, dong thoi co the truy nguoc duoc cong thuc ve code.

## 1. Plugin MPPI trong he thong Nav2

Trong file cau hinh, controller duoc chon la:

```yaml
FollowPath:
  plugin: "amr_nav2_mppi_controller::MPPIController"
```

Code tuong ung:

- Cau hinh plugin: `src/controller/config/mppi_params.yaml:53-55`
- Export plugin: `nav2_mppi_controller/src/controller.cpp:135-145`
- Moi chu ky dieu khien goi `computeVelocityCommands`: `nav2_mppi_controller/src/controller.cpp:80-114`
- Goi bo toi uu MPPI: `nav2_mppi_controller/src/controller.cpp:100-101`

Trong code, `MPPIController` dong vai tro plugin Nav2, con logic MPPI nam chu yeu trong class:

```cpp
mppi::Optimizer
```

Code tuong ung:

- Khai bao optimizer: `nav2_mppi_controller/include/nav2_mppi_controller/optimizer.hpp:55-264`
- Cai dat optimizer: `nav2_mppi_controller/src/optimizer.cpp`

## 2. Tham so MPPI trong YAML va y nghia toan hoc

### Cong thuc ly thuyet

MPPI dung:

```tex
K = \text{số lượng quỹ đạo mẫu}
```

```tex
T = \text{số bước trong chân trời dự báo}
```

```tex
\Delta t = \text{thời gian rời rạc giữa hai bước mô phỏng}
```

Tong thoi gian du bao:

```tex
T_{horizon} = T \Delta t
```

### Cau hinh hien tai

Trong `mppi_params.yaml`:

```yaml
time_steps: 80
model_dt: 0.1
batch_size: 2000
temperature: 0.3
gamma: 0.015
motion_model: "DiffDrive"
IterationCount: 1
```

Suy ra:

```tex
K = 2000
```

```tex
T = 80
```

```tex
\Delta t = 0.1s
```

```tex
T_{horizon} = 80 \times 0.1 = 8s
```

Trong code, tham so duoc doc tai:

- `model_dt`: `nav2_mppi_controller/src/optimizer.cpp:69`
- `time_steps`: `nav2_mppi_controller/src/optimizer.cpp:70`
- `batch_size`: `nav2_mppi_controller/src/optimizer.cpp:71`
- `iteration_count`: `nav2_mppi_controller/src/optimizer.cpp:72`
- `temperature`: `nav2_mppi_controller/src/optimizer.cpp:73`
- `gamma`: `nav2_mppi_controller/src/optimizer.cpp:74`
- Luu trong struct `OptimizerSettings`: `nav2_mppi_controller/include/nav2_mppi_controller/models/optimizer_settings.hpp:28-41`

Ghi chu quan trong:

- Trong YAML hien tai, ten tham so la `IterationCount`, nhung code doc `"iteration_count"` o `optimizer.cpp:72`. Neu tham so phan biet chu hoa/chu thuong theo ROS parameter, `IterationCount` co the khong duoc doc va code se dung default `1`.
- YAML co `ax_min`, `ax_max`, `ay_max`, `az_max` tai `src/controller/config/mppi_params.yaml:68-71`, nhung trong package nay khong thay code doc cac tham so acceleration do. Code chi doc gioi han van toc `vx_min`, `vx_max`, `vy_max`, `wz_max`.

## 3. Trang thai va chuoi dieu khien trong code

### Ly thuyet

He dong hoc roi rac:

```tex
x_{t+1} = F(x_t, v_t)
```

Chuoi dieu khien danh dinh:

```tex
U = \{u_0,u_1,\dots,u_{T-1}\}
```

Chuoi dieu khien mau:

```tex
V^k = \{v_0^k,v_1^k,\dots,v_{T-1}^k\}
```

### Code

Trong implementation nay, robot dung motion model `DiffDrive`, nen dieu khien chu yeu gom:

```tex
u_t = [v_{x,t}, \omega_{z,t}]
```

Neu la holonomic model thi co them `v_y`, nhung YAML dang dung:

```yaml
motion_model: "DiffDrive"
```

Code tuong ung:

- Chon motion model: `nav2_mppi_controller/src/optimizer.cpp:406-420`
- DiffDrive khong holonomic: `nav2_mppi_controller/include/nav2_mppi_controller/motion_models.hpp:134-154`

Chuoi dieu khien danh dinh `U` duoc luu trong:

```cpp
models::ControlSequence control_sequence_;
```

Code tuong ung:

- `ControlSequence` gom `vx`, `vy`, `wz`: `nav2_mppi_controller/include/nav2_mppi_controller/models/control_sequence.hpp:36-47`
- Reset chuoi dieu khien theo `time_steps`: `nav2_mppi_controller/src/optimizer.cpp:116-130`

Tap dieu khien mau cho toan bo batch duoc luu trong `state`:

```cpp
state.cvx, state.cvy, state.cwz
```

Code tuong ung:

- `State` gom van toc va control da nhieu: `nav2_mppi_controller/include/nav2_mppi_controller/models/state.hpp:30-55`

Mapping ky hieu:

| Ky hieu ly thuyet | Code |
|---|---|
| `U` | `control_sequence_` |
| `u_t = [v_x, \omega_z]` | `control_sequence_.vx(t)`, `control_sequence_.wz(t)` |
| `V^k` | hang thu `k` cua `state_.cvx`, `state_.cwz` |
| `v_t^k` | `state_.cvx(k,t)`, `state_.cwz(k,t)` |
| `x_t^k` | `generated_trajectories_.x(k,t)`, `.y(k,t)`, `.yaws(k,t)` |

## 4. Sinh nhieu Gaussian

### Ly thuyet

MPPI gia su dau vao thuc te duoc tao tu dieu khien danh dinh cong nhieu:

```tex
v_t^k = u_t + \epsilon_t^k
```

```tex
\epsilon_t^k \sim \mathcal{N}(0,\Sigma)
```

Neu viet rieng cho DiffDrive:

```tex
\epsilon_t^k =
[\epsilon_{v_x,t}^k,\epsilon_{\omega_z,t}^k]
```

```tex
\epsilon_{v_x,t}^k \sim \mathcal{N}(0,\sigma_{v_x}^2)
```

```tex
\epsilon_{\omega_z,t}^k \sim \mathcal{N}(0,\sigma_{\omega_z}^2)
```

### Code

Sinh nhieu Gaussian:

- `nav2_mppi_controller/src/noise_generator.cpp:107-122`

Code:

```cpp
noises_vx_ = xt::random::randn<float>(
  {s.batch_size, s.time_steps}, 0.0f, s.sampling_std.vx);

noises_wz_ = xt::random::randn<float>(
  {s.batch_size, s.time_steps}, 0.0f, s.sampling_std.wz);
```

Cong nhieu vao chuoi dieu khien danh dinh:

- `nav2_mppi_controller/src/noise_generator.cpp:65-74`

Code:

```cpp
state.cvx = control_sequence.vx + noises_vx_;
state.cwz = control_sequence.wz + noises_wz_;
```

Mapping:

```tex
v_{x,t}^k = u_{x,t} + \epsilon_{x,t}^k
```

```tex
\omega_{z,t}^k = u_{\omega,t} + \epsilon_{\omega,t}^k
```

### Tham so nhieu

Code doc:

- `vx_std`: `nav2_mppi_controller/src/optimizer.cpp:79`
- `vy_std`: `nav2_mppi_controller/src/optimizer.cpp:80`
- `wz_std`: `nav2_mppi_controller/src/optimizer.cpp:81`

Trong YAML hien tai chua thay khai bao `vx_std`, `vy_std`, `wz_std`, nen code se dung gia tri default:

```tex
\sigma_{v_x} = 0.2
```

```tex
\sigma_{v_y} = 0.2
```

```tex
\sigma_{\omega_z} = 0.4
```

Neu muon bao cao khop tuyet doi voi cau hinh, co the ghi: "Trong cau hinh hien tai, do khong khai bao `vx_std/wz_std`, bo dieu khien su dung gia tri mac dinh trong source code."

## 5. Mo hinh dong hoc roi rac

### Ly thuyet

Voi robot vi sai, trang thai co the viet:

```tex
x_t = [p_{x,t}, p_{y,t}, \theta_t]^T
```

Dieu khien:

```tex
u_t = [v_t, \omega_t]^T
```

Mo hinh roi rac:

```tex
\theta_{t+1} = \theta_t + \omega_t \Delta t
```

```tex
p_{x,t+1} = p_{x,t} + v_t \cos(\theta_t)\Delta t
```

```tex
p_{y,t+1} = p_{y,t} + v_t \sin(\theta_t)\Delta t
```

Trong code, yaw duoc tich phan truoc, sau do cac gia tri cos/sin duoc tao theo chuoi yaw da du bao. Ve ban chat, day la mo phong tien cua mo hinh non-holonomic DiffDrive.

### Code

Van toc mau duoc propagate:

- `nav2_mppi_controller/src/optimizer.cpp:245-267`
- `MotionModel::predict`: `nav2_mppi_controller/include/nav2_mppi_controller/motion_models.hpp:54-67`

Tich phan thanh quỹ đạo:

- Batch trajectories: `nav2_mppi_controller/src/optimizer.cpp:307-337`

Code chinh:

```cpp
trajectories.yaws =
  cumsum_2d(state.wz * settings_.model_dt, 1) + initial_yaw;

dx = state.vx * cos(yaw);
dy = state.vx * sin(yaw);

trajectories.x = x0 + cumsum_2d(dx * settings_.model_dt, 1);
trajectories.y = y0 + cumsum_2d(dy * settings_.model_dt, 1);
```

Mapping cong thuc:

```tex
\theta_t^k = \theta_0 + \sum_{i=0}^{t}\omega_i^k\Delta t
```

```tex
x_t^k = x_0 + \sum_{i=0}^{t}v_i^k\cos(\theta_i^k)\Delta t
```

```tex
y_t^k = y_0 + \sum_{i=0}^{t}v_i^k\sin(\theta_i^k)\Delta t
```

## 6. Chi phi quy dao tong hop tu critics

### Ly thuyet

Tong chi phi cua mau thu `k`:

```tex
S_k =
\phi(x_T^k)
+ \sum_{t=0}^{T-1} c(x_t^k, v_t^k)
```

Trong code, `S_k` duoc luu trong:

```cpp
costs_(k)
```

Chi phi khong nam trong mot ham duy nhat, ma duoc cong don qua cac critic plugin:

```tex
S_k =
\sum_i J_i(\tau^k, V^k)
```

Trong do moi critic `J_i` danh gia mot khia canh cua trajectory.

Code tuong ung:

- Vector chi phi `costs_`: `nav2_mppi_controller/src/optimizer.cpp:127`
- Dua data cho critics: `nav2_mppi_controller/include/nav2_mppi_controller/critic_data.hpp:38-52`
- Goi tat ca critics: `nav2_mppi_controller/src/critic_manager.cpp:67-75`
- Trong optimizer: `nav2_mppi_controller/src/optimizer.cpp:155-161`

Danh sach critics trong YAML:

- `ConstraintCritic`: `src/controller/config/mppi_params.yaml:82-84`
- `ObstaclesCritic`: `src/controller/config/mppi_params.yaml:83-84`
- `DynamicHumanCritic`: `src/controller/config/mppi_params.yaml:85`
- `GoalCritic`: `src/controller/config/mppi_params.yaml:86`
- `GoalAngleCritic`: `src/controller/config/mppi_params.yaml:87`
- `PathAlignCritic`: `src/controller/config/mppi_params.yaml:88`
- `CostCritic`: `src/controller/config/mppi_params.yaml:90`
- `PathFollowCritic`: `src/controller/config/mppi_params.yaml:91`
- `TwirlingCritic`: `src/controller/config/mppi_params.yaml:92`
- `PreferForwardCritic`: `src/controller/config/mppi_params.yaml:93`

## 7. Cac thanh phan chi phi cu the trong cau hinh hien tai

### 7.1. ConstraintCritic

Muc tieu: phat cac trajectory vuot gioi han van toc.

Cong thuc gan voi code:

```tex
J_{constraint}^k =
\left[
w_c
\sum_{t=0}^{T-1}
\left(
\max(v_t^k - v_{max},0)
+ \max(v_{min} - v_t^k,0)
\right)\Delta t
\right]^p
```

Code tuong ung:

- `nav2_mppi_controller/src/critics/constraint_critic.cpp:41-75`
- Cau hinh weight: `src/controller/config/mppi_params.yaml:96-98`

Trong YAML:

```yaml
ConstraintCritic:
  cost_weight: 1.0
```

### 7.2. ObstaclesCritic

Muc tieu: phat collision va phat cac trajectory di gan vat can tinh trong costmap.

Cong thuc gan voi code:

```tex
J_{obs}^k =
\left(
w_{critical} C_{collision}^k
+ w_{rep}
\frac{1}{T}
\sum_{t=0}^{T-1} C_{rep}(x_t^k)
\right)^p
```

Neu trajectory va cham:

```tex
C_{collision}^k = C_{collision\_cost}
```

Neu gan vat can hon nguong margin:

```tex
C_{rep}(x_t^k)
= d_{margin} - d_{obs}(x_t^k)
```

Code tuong ung:

- `nav2_mppi_controller/src/critics/obstacles_critic.cpp:127-190`
- Collision check: `nav2_mppi_controller/src/critics/obstacles_critic.cpp:198-237`
- Cau hinh: `src/controller/config/mppi_params.yaml:99-103`

Trong YAML:

```yaml
ObstaclesCritic:
  repulsion_weight: 5.0
  cost_weight: 90.0
  collision_cost: 100000.0
  consider_footprint: true
```

Ghi chu: code cua `ObstaclesCritic` doc `critical_weight`, khong phai `cost_weight`, tai `obstacles_critic.cpp:28`. Neu YAML chi khai bao `cost_weight`, phan `critical_weight_` co the dung default `20.0`. Tuy nhien `repulsion_weight` va `collision_cost` co duoc doc dung ten.

### 7.3. DynamicHumanCritic

Muc tieu: phat cac trajectory co nguy co va cham voi nguoi di bo/dynamic obstacles.

Code du bao vi tri nguoi theo mo hinh van toc hang:

```tex
p_{h,t} =
p_{h,0} + v_h t\Delta t
```

Va tinh khoang cach:

```tex
d_{k,t,h}^2 =
(x_t^k - x_{h,t})^2
+ (y_t^k - y_{h,t})^2
```

Neu:

```tex
d_{k,t,h} < r_h + r_{robot} + d_{safe}
```

thi cong collision cost:

```tex
J_{human}^k =
w_h
\sum_h
\sum_{t=0}^{T-1}
\mathbb{I}
\left(
d_{k,t,h}^2 < (r_h+r_{robot}+d_{safe})^2
\right)
C_h
```

Code tuong ung:

- Nhan humans topic: `nav2_mppi_controller/src/critics/dynamic_human_critic.cpp:75-80`
- Transform va loc radius: `nav2_mppi_controller/src/critics/dynamic_human_critic.cpp:82-198`
- Score: `nav2_mppi_controller/src/critics/dynamic_human_critic.cpp:211-252`
- Cau hinh: `src/controller/config/mppi_params.yaml:105-115`

Trong YAML:

```yaml
DynamicHumanCritic:
  enabled: true
  human_topic: /tracking/humans
  cost_weight: 100.0
  collision_cost: 100000.0
  safe_margin: 0.2
```

### 7.4. GoalCritic

Muc tieu: keo trajectory ve gan goal.

Cong thuc gan voi code:

```tex
J_{goal}^k =
\left[
w_g
\frac{1}{T}
\sum_{t=0}^{T-1}
\sqrt{(x_t^k-x_g)^2 + (y_t^k-y_g)^2}
\right]^p
```

Code tuong ung:

- `nav2_mppi_controller/src/critics/goal_critic.cpp:36-56`
- Cau hinh: `src/controller/config/mppi_params.yaml:140-142`

### 7.5. GoalAngleCritic

Muc tieu: gan cuoi duong, phat sai lech huong robot so voi huong goal.

Cong thuc gan voi code:

```tex
J_{goal\_angle}^k =
\left[
w_\theta
\frac{1}{T}
\sum_{t=0}^{T-1}
|\text{wrap}(\theta_g - \theta_t^k)|
\right]^p
```

Code tuong ung:

- `nav2_mppi_controller/src/critics/goal_angle_critic.cpp:37-64`
- Cau hinh: `src/controller/config/mppi_params.yaml:144-146`

### 7.6. PathAlignCritic

Muc tieu: phat trajectory lech khoi duong tham chieu.

Cong thuc thiet ke co the viet:

```tex
J_{align}^k =
w_a
\sum_{t \in \mathcal{T}_{eval}}
d((x_t^k,y_t^k,\theta_t^k), \mathcal{P})
```

Trong do `P` la global/transformed path, va `d` la khoang cach den diem duong tham chieu gan nhat hop le.

Code tuong ung:

- `nav2_mppi_controller/src/critics/path_align_critic.cpp:46-135`
- Cau hinh: `src/controller/config/mppi_params.yaml:117-119`

### 7.7. PathFollowCritic

Muc tieu: khuyen khich diem cuoi trajectory tien xa hon theo path.

Cong thuc gan voi code:

```tex
J_{follow}^k =
\left[
w_f
\sqrt{
(x_T^k - x_{ref})^2
+ (y_T^k - y_{ref})^2
}
\right]^p
```

Code tuong ung:

- `nav2_mppi_controller/src/critics/path_follow_critic.cpp:35-70`
- Cau hinh: `src/controller/config/mppi_params.yaml:126-128`

### 7.8. PreferForwardCritic

Muc tieu: phat chuyen dong lui.

Cong thuc gan voi code:

```tex
J_{forward}^k =
\left[
w_f
\sum_{t=0}^{T-1}
\max(-v_{x,t}^k,0)\Delta t
\right]^p
```

Code tuong ung:

- `nav2_mppi_controller/src/critics/prefer_forward_critic.cpp:33-46`
- Cau hinh: `src/controller/config/mppi_params.yaml:148-150`

### 7.9. TwirlingCritic

Muc tieu: phat quay tai cho/dao dong goc qua nhieu.

Cong thuc gan voi code:

```tex
J_{twirl}^k =
\left[
w_{tw}
\frac{1}{T}
\sum_{t=0}^{T-1}
|\omega_{z,t}^k|
\right]^p
```

Code tuong ung:

- `nav2_mppi_controller/src/critics/twirling_critic.cpp:31-41`
- Cau hinh: `src/controller/config/mppi_params.yaml:130-131`

## 8. Chi phi dieu khien / importance sampling correction trong code

### Ly thuyet

Trong cong thuc MPPI tong quat, chi phi dieu khien co dang:

```tex
\mathcal{L}(x_t,u_t)
=
c(x_t)
+ \frac{\lambda}{2}u_t^T\Sigma^{-1}u_t
```

Va khi dung importance sampling quanh chuoi dieu khien danh dinh, implementation thuong co them thanh phan hieu chinh chi phi lien quan den dieu khien danh dinh va nhieu.

### Code

Trong code, sau khi critics cong chi phi trajectory vao `costs_`, optimizer cong them thanh phan:

```cpp
costs_ += gamma / sigma_vx^2 * sum(U_vx * bounded_noise_vx)
costs_ += gamma / sigma_wz^2 * sum(U_wz * bounded_noise_wz)
```

Code tuong ung:

- `nav2_mppi_controller/src/optimizer.cpp:356-374`

Mapping cong thuc:

```tex
S_k \leftarrow S_k
+ \frac{\gamma}{\sigma_{v_x}^2}
\sum_{t=0}^{T-1}
u_{v_x,t}(v_{x,t}^k-u_{v_x,t})
```

```tex
S_k \leftarrow S_k
+ \frac{\gamma}{\sigma_{\omega_z}^2}
\sum_{t=0}^{T-1}
u_{\omega,t}(\omega_{z,t}^k-u_{\omega,t})
```

Trong code:

```tex
v_{x,t}^k-u_{v_x,t}
=
state.cvx(k,t)-control_sequence.vx(t)
```

```tex
\omega_{z,t}^k-u_{\omega,t}
=
state.cwz(k,t)-control_sequence.wz(t)
```

Tham so `gamma` trong YAML:

```yaml
gamma: 0.015
```

Code doc tai:

- `nav2_mppi_controller/src/optimizer.cpp:74`

## 9. Chuan hoa chi phi va trong so softmax

### Ly thuyet

MPPI tinh:

```tex
\rho = \min_k S_k
```

```tex
w_k =
\frac{
\exp\left(-\frac{1}{\lambda}(S_k-\rho)\right)
}{
\sum_{j=1}^{K}
\exp\left(-\frac{1}{\lambda}(S_j-\rho)\right)
}
```

Trong do `lambda` la temperature.

### Code

Code tuong ung:

- `nav2_mppi_controller/src/optimizer.cpp:376-379`

Code:

```cpp
costs_normalized = costs_ - amin(costs_)
exponents = exp(-1 / temperature * costs_normalized)
softmaxes = exponents / sum(exponents)
```

Mapping:

```tex
\rho = \min_k S_k
```

```tex
\lambda = \texttt{temperature}
```

Trong YAML:

```yaml
temperature: 0.3
```

Code doc tai:

- `nav2_mppi_controller/src/optimizer.cpp:73`

## 10. Luat cap nhat dieu khien

### Ly thuyet

Cong thuc ly thuyet:

```tex
u_t^*
=
\hat{u}_t
+ \sum_{k=1}^{K}\bar{w}_k\epsilon_t^k
```

Do:

```tex
v_t^k = \hat{u}_t + \epsilon_t^k
```

nen co the viet tuong duong:

```tex
u_t^*
=
\sum_{k=1}^{K}\bar{w}_k v_t^k
```

vi:

```tex
\sum_k \bar{w}_k v_t^k
=
\sum_k \bar{w}_k(\hat{u}_t+\epsilon_t^k)
=
\hat{u}_t + \sum_k\bar{w}_k\epsilon_t^k
```

### Code

Code dung dang thu hai: trung binh co trong so cua cac control da nhieu.

- `nav2_mppi_controller/src/optimizer.cpp:381-385`

Code:

```cpp
control_sequence_.vx = sum(state_.cvx * softmaxes, 0)
control_sequence_.wz = sum(state_.cwz * softmaxes, 0)
```

Mapping:

```tex
u_{v_x,t}^*
=
\sum_{k=1}^{K}w_k v_{x,t}^k
```

```tex
u_{\omega,t}^*
=
\sum_{k=1}^{K}w_k \omega_{z,t}^k
```

Day la cung mot luat cap nhat voi cong thuc MPPI cong nhieu, chi khac cach viet.

## 11. Rang buoc dieu khien

### Ly thuyet

Sau khi cap nhat, dieu khien duoc gioi han trong mien hop le:

```tex
u_{min} \leq u_t \leq u_{max}
```

Voi DiffDrive:

```tex
v_{x,min} \leq v_{x,t} \leq v_{x,max}
```

```tex
|\omega_{z,t}| \leq \omega_{z,max}
```

### Code

Code tuong ung:

- `nav2_mppi_controller/src/optimizer.cpp:231-243`

Code:

```cpp
control_sequence_.vx = clip(control_sequence_.vx, vx_min, vx_max);
control_sequence_.wz = clip(control_sequence_.wz, -wz, wz);
```

Cau hinh YAML:

```yaml
vx_min: -0.2
vx_max: 0.3
vy_max: 0.0
wz_max: 0.3
```

Code doc tai:

- `vx_max`: `nav2_mppi_controller/src/optimizer.cpp:75`
- `vx_min`: `nav2_mppi_controller/src/optimizer.cpp:76`
- `vy_max`: `nav2_mppi_controller/src/optimizer.cpp:77`
- `wz_max`: `nav2_mppi_controller/src/optimizer.cpp:78`

## 12. Lam muot Savitzky-Golay

### Ly thuyet

Sau khi cap nhat chuoi dieu khien, MPPI co the lam muot:

```tex
U = \mathrm{SGF}(U')
```

Trong code, bo loc Savitzky-Golay bac hai voi cua so 9 diem co he so:

```tex
\frac{1}{231}
[-21, 14, 39, 54, 59, 54, 39, 14, -21]
```

### Code

- Goi filter sau optimize: `nav2_mppi_controller/src/optimizer.cpp:145`
- Filter implementation: `nav2_mppi_controller/include/nav2_mppi_controller/tools/utils.hpp:451-614`

Code:

```cpp
filter = {-21, 14, 39, 54, 59, 54, 39, 14, -21};
filter /= 231.0;
```

Filter duoc ap dung cho:

```cpp
control_sequence.vx
control_sequence.vy
control_sequence.wz
```

## 13. Receding horizon va warm-start

### Ly thuyet

MPPI chi thuc thi lenh dau tien:

```tex
u_0^*
```

Sau do dich chuoi dieu khien:

```tex
u_0 \leftarrow u_1,\quad
u_1 \leftarrow u_2,\quad
\dots,\quad
u_{T-2}\leftarrow u_{T-1}
```

### Code

Lay command dau tien hoac command o offset 1 neu co shift:

- `nav2_mppi_controller/src/optimizer.cpp:390-404`

Code:

```cpp
unsigned int offset = settings_.shift_control_sequence ? 1 : 0;
vx = control_sequence_.vx(offset);
wz = control_sequence_.wz(offset);
```

Dich chuoi dieu khien:

- `nav2_mppi_controller/src/optimizer.cpp:200-219`

Code:

```cpp
control_sequence_.vx = roll(control_sequence_.vx, -1);
control_sequence_.wz = roll(control_sequence_.wz, -1);
last = previous last;
```

Dieu kien bat shift:

- `nav2_mppi_controller/src/optimizer.cpp:95-114`

Neu chu ky controller bang `model_dt`, code bat:

```cpp
settings_.shift_control_sequence = true;
```

Trong YAML:

```yaml
controller_frequency: 20.0
model_dt: 0.1
```

Chu ky controller la:

```tex
\Delta t_{ctrl} = \frac{1}{20} = 0.05s
```

Trong code `setOffset`, neu:

```tex
\Delta t_{ctrl} < model\_dt
```

thi chi canh bao, khong bat shift. Vi vay voi cau hinh hien tai `0.05 < 0.1`, `shift_control_sequence` khong duoc bat trong nhanh equality. Neu muon warm-start shift dung theo tung chu ky controller, can can nhac dat `model_dt = 0.05` hoac dong bo voi `controller_frequency`.

## 14. Chu trinh MPPI trong code

Mot chu ky dieu khien thuc hien:

1. Nhan pose, velocity hien tai va global/transformed path.
2. Reset `costs_`.
3. Tao cac dieu khien mau bang cach cong Gaussian noise vao chuoi control hien tai.
4. Mo phong model DiffDrive de sinh `K` quỹ đạo.
5. Cac critic cong chi phi vao `costs_`.
6. Cong thanh phan importance sampling/control correction.
7. Chuan hoa chi phi bang `rho = min S_k`.
8. Tinh softmax weight.
9. Cap nhat chuoi dieu khien bang trung binh co trong so cua sampled controls.
10. Clip dieu khien theo gioi han van toc.
11. Lam muot chuoi dieu khien bang Savitzky-Golay.
12. Xuat lenh `TwistStamped` dau tien cho robot.
13. Neu duoc bat, dich chuoi control de warm-start chu ky sau.

Code tuong ung:

- `evalControl`: `nav2_mppi_controller/src/optimizer.cpp:134-153`
- `optimize`: `nav2_mppi_controller/src/optimizer.cpp:155-162`
- `generateNoisedTrajectories`: `nav2_mppi_controller/src/optimizer.cpp:221-227`
- `updateControlSequence`: `nav2_mppi_controller/src/optimizer.cpp:356-388`

## 15. Thuat toan MPPI theo dung code hien tai

```text
Input:
  pose hien tai x0
  van toc hien tai
  global/transformed path
  chuoi control hien tai U
  K = batch_size
  T = time_steps
  temperature lambda
  gamma

For moi chu ky dieu khien:
  1. Sinh nhieu Gaussian epsilon_vx, epsilon_wz kich thuoc K x T
  2. Tao sampled controls:
       Vx_k,t = Ux_t + epsilon_vx_k,t
       Wz_k,t = Uw_t + epsilon_wz_k,t
  3. Propagate motion model DiffDrive de tao K trajectory
  4. Khoi tao S_k = 0
  5. Voi moi critic:
       S_k += J_critic(trajectory_k, control_k)
  6. Cong control correction:
       S_k += gamma/sigma_vx^2 * sum_t Ux_t(Vx_k,t - Ux_t)
       S_k += gamma/sigma_wz^2 * sum_t Uw_t(Wz_k,t - Uw_t)
  7. rho = min_k S_k
  8. w_k = exp(-(S_k-rho)/lambda) / sum_j exp(-(S_j-rho)/lambda)
  9. Cap nhat:
       Ux_t = sum_k w_k Vx_k,t
       Uw_t = sum_k w_k Wz_k,t
 10. Clip U theo vx_min/vx_max/wz_max
 11. Lam muot U bang Savitzky-Golay
 12. Xuat cmd_vel = [Ux_offset, Uw_offset]
 13. Neu shift_control_sequence bat, roll U sang trai mot buoc
```

## 16. Bang mapping nhanh cong thuc - code - YAML

| Thanh phan | Cong thuc/ky hieu | Code | YAML |
|---|---|---|---|
| So mau | `K` | `settings_.batch_size` | `batch_size: 2000` |
| Chan troi | `T` | `settings_.time_steps` | `time_steps: 80` |
| Buoc thoi gian | `Delta t` | `settings_.model_dt` | `model_dt: 0.1` |
| Temperature | `lambda` | `settings_.temperature` | `temperature: 0.3` |
| Control correction | `gamma` | `settings_.gamma` | `gamma: 0.015` |
| Control sequence | `U` | `control_sequence_` | - |
| Sampled controls | `V^k` | `state_.cvx`, `state_.cwz` | - |
| Noise | `epsilon^k` | `noises_vx_`, `noises_wz_` | defaults `vx_std=0.2`, `wz_std=0.4` |
| Trajectory | `tau^k` | `generated_trajectories_` | - |
| Chi phi | `S_k` | `costs_` | critics block |
| Chuan hoa | `rho = min S_k` | `costs_ - amin(costs_)` | - |
| Trong so | `w_k` | `softmaxes` | `temperature` |
| Cap nhat | `u_t = sum w_k v_t^k` | `updateControlSequence()` | - |
| Rang buoc | `clip(u)` | `applyControlSequenceConstraints()` | `vx_min/vx_max/wz_max` |
| Lam muot | `SGF(U)` | `savitskyGolayFilter()` | - |
| Lenh xuat | `u_0` | `getControlFromSequenceAsTwist()` | `cmd_vel_topic` |

