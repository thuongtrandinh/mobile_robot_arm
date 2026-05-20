# Bộ công thức cho chương thiết kế bộ điều khiển HALO

Tài liệu này tổng hợp các công thức toán học cần dùng cho chương thiết kế bộ điều khiển. Nội dung được viết theo hướng lý thuyết thiết kế, không trình bày chi tiết mã nguồn.

## 1. Kiến trúc điều hướng phân tầng

Hệ thống HALO được mô tả như một kiến trúc điều hướng phân tầng:

```tex
\text{Observation}
\rightarrow
\text{Graph Representation}
\rightarrow
\text{DRL Policy}
\rightarrow
\text{Local Goal}
\rightarrow
\text{Predictive Planner}
\rightarrow
\text{Low-level Control}
```

Tầng học tăng cường sinh mục tiêu cục bộ:

```tex
g_t^{local}
=
f_{\pi_\theta}(s_t)
```

Tầng lập kế hoạch dự báo sinh điều khiển mức thấp:

```tex
u_t^{low}
=
f_{planner}
\left(
x_t,
g_t^{local},
\mathcal{O}_t
\right)
```

Trong đó:

- `s_t` là trạng thái quan sát của tầng học tăng cường.
- `x_t` là trạng thái động học của robot.
- `g_t^{local}` là mục tiêu cục bộ.
- `\mathcal{O}_t` là tập vật cản, người đi bộ và ràng buộc môi trường.
- `u_t^{low}` là tín hiệu điều khiển thấp.

Về nguyên lý phân tầng:

```tex
\pi_\theta:
s_t
\mapsto
a_t
\mapsto
g_t^{local}
```

```tex
f_{planner}:
\left(
x_t,
g_t^{local},
\mathcal{O}_t
\right)
\mapsto
u_t^{low}
```

## 2. MDP cho tầng học tăng cường

Bài toán điều hướng được mô hình hóa thành MDP:

```tex
\mathcal{M}
=
\left(
\mathcal{S},
\mathcal{A},
P,
R,
\gamma
\right)
```

Trong đó:

```tex
s_t \in \mathcal{S},
\quad
a_t \in \mathcal{A},
\quad
r_t = R(s_t,a_t,s_{t+1})
```

Quá trình tương tác:

```tex
s_t
\xrightarrow{a_t}
\left(
r_t,
s_{t+1}
\right)
```

Chính sách của tác nhân:

```tex
\pi_\theta(a_t|s_t)
=
\mathbb{P}
\left(
A_t=a_t
\mid
S_t=s_t
\right)
```

Mục tiêu tối ưu:

```tex
\pi^*
=
\arg\max_{\pi_\theta}
\mathbb{E}_{\pi_\theta}
\left[
\sum_{t=0}^{\infty}
\gamma^t r_t
\right]
```

Hàm giá trị trạng thái:

```tex
V_\phi(s_t)
=
\mathbb{E}_{\pi_\theta}
\left[
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}
\mid
s_t
\right]
```

## 3. Không gian trạng thái

Trạng thái tổng quát:

```tex
s_t
=
\left\{
s_t^{robot},
s_t^{human},
s_t^{obs},
s_t^{wall},
g
\right\}
```

### 3.1. Trạng thái robot

Trạng thái robot đầy đủ:

```tex
s_t^{robot}
=
\left[
p_{x,t},
p_{y,t},
v_{x,t},
v_{y,t},
r_{robot},
g_x,
g_y,
v_{pref},
\theta_t
\right]^T
```

Khoảng cách đến mục tiêu:

```tex
d_{g,t}
=
\sqrt{
(g_x-p_{x,t})^2
+
(g_y-p_{y,t})^2
}
```

Góc từ robot đến mục tiêu:

```tex
\theta_{g,t}
=
\mathrm{atan2}
\left(
g_y-p_{y,t},
g_x-p_{x,t}
\right)
```

Sai lệch hướng:

```tex
\theta_{rel,t}
=
\mathrm{wrap}
\left(
\theta_t-\theta_{g,t}
\right)
```

Đặc trưng rút gọn của robot:

```tex
h_t^{robot}
=
\left[
v_{x,t},
v_{y,t},
d_{g,t},
v_{pref},
\theta_{rel,t}
\right]^T
```

### 3.2. Trạng thái người đi bộ

Với người đi bộ thứ `i`:

```tex
s_{t,i}^{human}
=
\left[
p_{x,t}^{i},
p_{y,t}^{i},
v_{x,t}^{i},
v_{y,t}^{i},
r_i
\right]^T
```

### 3.3. Trạng thái vật cản tròn

```tex
s_{t,i}^{obs}
=
\left[
p_{x,t}^{i},
p_{y,t}^{i},
r_i
\right]^T
```

### 3.4. Trạng thái tường hoặc vật cản dạng đoạn thẳng

```tex
s_{t,i}^{wall}
=
\left[
s_{x,t}^{i},
s_{y,t}^{i},
e_{x,t}^{i},
e_{y,t}^{i},
r_i
\right]^T
```

## 4. Biểu diễn môi trường bằng đồ thị

Đồ thị môi trường:

```tex
\mathcal{G}_t
=
\left(
\mathcal{V}_t,
\mathcal{E}_t
\right)
```

Tập node:

```tex
\mathcal{V}_t
=
\mathcal{V}^{robot}
\cup
\mathcal{V}^{human}
\cup
\mathcal{V}^{obs}
\cup
\mathcal{V}^{wall}
```

Tập cạnh được mô hình hóa theo quan hệ tương tác:

```tex
\mathcal{E}_t
=
\mathcal{E}_{h2r}
\cup
\mathcal{E}_{o2r}
\cup
\mathcal{E}_{w2r}
\cup
\mathcal{E}_{o2h}
\cup
\mathcal{E}_{w2h}
\cup
\mathcal{E}_{h2h}
```

Các nhóm cạnh:

```tex
\mathcal{E}_{h2r}
=
\left\{
\left(
v_i^{human},
v^{robot}
\right)
\mid
v_i^{human}
\in
\mathcal{V}^{human}
\right\}
```

```tex
\mathcal{E}_{o2r}
=
\left\{
\left(
v_i^{obs},
v^{robot}
\right)
\mid
v_i^{obs}
\in
\mathcal{V}^{obs}
\right\}
```

```tex
\mathcal{E}_{w2r}
=
\left\{
\left(
v_i^{wall},
v^{robot}
\right)
\mid
v_i^{wall}
\in
\mathcal{V}^{wall}
\right\}
```

```tex
\mathcal{E}_{o2h}
=
\left\{
\left(
v_i^{obs},
v_j^{human}
\right)
\mid
v_i^{obs}
\in
\mathcal{V}^{obs},
v_j^{human}
\in
\mathcal{V}^{human}
\right\}
```

```tex
\mathcal{E}_{w2h}
=
\left\{
\left(
v_i^{wall},
v_j^{human}
\right)
\mid
v_i^{wall}
\in
\mathcal{V}^{wall},
v_j^{human}
\in
\mathcal{V}^{human}
\right\}
```

```tex
\mathcal{E}_{h2h}
=
\left\{
\left(
v_i^{human},
v_j^{human}
\right)
\mid
i \ne j
\right\}
```

Mỗi node có vector đặc trưng:

```tex
h_i
=
\left[
h_i^{type},
h_i^{geo},
h_i^{vel},
h_i^{risk}
\right]^T
```

Trong đó:

```tex
h_i^{risk}
=
\left[
\mathbb{I}_{VO,i},
d_{min,i},
t_{exp,i}
\right]^T
```

## 5. Đặc trưng Velocity Obstacle

Vị trí tương đối:

```tex
p_{rel,i}
=
p_i
-
p_{robot}
```

Vận tốc tương đối:

```tex
v_{rel,i}
=
v_{robot}
-
v_i
```

Vùng Velocity Obstacle của đối tượng `i`:

```tex
VO_i
=
\left\{
v_{rel}
\mid
\exists t>0:
p_{rel,i}
+
v_{rel}t
\in
\mathcal{C}_i
\right\}
```

Trong đó `\mathcal{C}_i` là vùng va chạm mở rộng bởi bán kính robot và đối tượng:

```tex
\mathcal{C}_i
=
\left\{
p
\mid
\|p\|
\le
r_{robot}
+
r_i
\right\}
```

Chỉ báo nguy cơ VO:

```tex
\mathbb{I}_{VO,i}
=
\begin{cases}
1, & v_{rel,i} \in VO_i \\
0, & v_{rel,i} \notin VO_i
\end{cases}
```

Thời gian dự kiến đến va chạm:

```tex
t_{exp,i}
=
\min
\left\{
t>0
\mid
p_{rel,i}
+
v_{rel,i}t
\in
\mathcal{C}_i
\right\}
```

Khoảng cách nhỏ nhất trong tương lai gần:

```tex
d_{min,i}
=
\min_{t \in [0,T_p]}
\left\|
p_{rel,i}
+
v_{rel,i}t
\right\|
-
\left(
r_{robot}
+
r_i
\right)
```

## 6. Trích xuất đặc trưng bằng GAT

Mã hóa đặc trưng node:

```tex
z_i
=
f_{enc}(h_i)
```

Attention score:

```tex
e_{ij}
=
\mathrm{LeakyReLU}
\left(
a^T
\left[
Wz_i
\Vert
Wz_j
\right]
\right)
```

Trọng số attention:

```tex
\alpha_{ij}
=
\frac{
\exp(e_{ij})
}{
\sum_{k \in \mathcal{N}(i)}
\exp(e_{ik})
}
```

Cập nhật đặc trưng node:

```tex
h_i'
=
\sigma
\left(
\sum_{j \in \mathcal{N}(i)}
\alpha_{ij}
Wz_j
\right)
```

Biểu diễn toàn cục cho Actor-Critic:

```tex
H_t
=
F_{GAT}
\left(
\mathcal{G}_t
\right)
```

Nếu dùng residual connection:

```tex
H_t
=
h_t'
+
z_t
```

## 7. Không gian hành động local goal

Không gian hành động rời rạc:

```tex
\mathcal{A}
=
\left\{
0,1,\dots,N^2-1
\right\}
```

Với:

```tex
N=9,
\qquad
|\mathcal{A}|=81
```

Miền local goal:

```tex
x_g^{local}, y_g^{local}
\in
\left[
-d_a,
d_a
\right],
\qquad
d_a=2.25m
```

Lưới rời rạc:

```tex
\mathcal{L}
=
\left\{
l_0,l_1,\dots,l_{N-1}
\right\}
```

```tex
l_i
=
-d_a
+
\frac{2d_a}{N-1}i
```

Với action index `a`:

```tex
i
=
\left\lfloor
\frac{a}{N}
\right\rfloor
```

```tex
j
=
a \bmod N
```

Local goal trong hệ robot:

```tex
g_t^{local}(a)
=
\begin{bmatrix}
l_i \\
l_j
\end{bmatrix}
```

Ma trận quay:

```tex
R(\theta_t)
=
\begin{bmatrix}
\cos\theta_t & -\sin\theta_t \\
\sin\theta_t & \cos\theta_t
\end{bmatrix}
```

Chuyển local goal sang hệ map:

```tex
g_t^{map}(a)
=
R(\theta_t)
g_t^{local}(a)
+
\begin{bmatrix}
p_{x,t} \\
p_{y,t}
\end{bmatrix}
```

## 8. Action masking

Mask hành động:

```tex
m_t(a)
\in
\{0,1,2\}
```

```tex
m_t(a)
=
\begin{cases}
0, & \text{local goal không hợp lệ} \\
1, & \text{local goal hợp lệ} \\
2, & \text{local goal ưu tiên khi robot gần đích}
\end{cases}
```

Điều kiện ngoài vùng tác động:

```tex
\left\|
g_t^{local}(a)
\right\|
>
d_{max}
```

Điều kiện gần vật cản tròn:

```tex
\left\|
g_t^{map}(a)
-
p_i^{obs}
\right\|
<
r_i^{obs}
+
r_{robot}
+
d_{safe}
```

Điều kiện gần tường:

```tex
d
\left(
g_t^{map}(a),
wall_i
\right)
<
r_{robot}
+
d_{safe}
```

Điều kiện nằm trong vật cản đa giác:

```tex
g_t^{map}(a)
\in
\mathcal{P}_{obs}
```

Điều kiện nằm ngoài vùng tự do:

```tex
g_t^{map}(a)
\notin
\Omega_{free}
```

Với mask nhị phân `m_t(a) \in \{0,1\}`, policy sau masking:

```tex
\pi_\theta^m(a|s_t)
=
\frac{
m_t(a)
\pi_\theta(a|s_t)
}{
\sum_{b \in \mathcal{A}}
m_t(b)
\pi_\theta(b|s_t)
}
```

Với mask có trạng thái ưu tiên `m_t(a)=2`, nên mô tả bằng logit mask:

```tex
\ell_a^m
=
\begin{cases}
-\infty, & m_t(a)=0 \\
\ell_a, & m_t(a)=1 \\
+\infty, & m_t(a)=2
\end{cases}
```

Policy masked:

```tex
\pi_\theta^m(a|s_t)
=
\frac{
\exp(\ell_a^m)
}{
\sum_{b \in \mathcal{A}}
\exp(\ell_b^m)
}
```

Nếu chỉ có một hành động ưu tiên `a^+`:

```tex
\pi_\theta^m(a^+|s_t)
\approx
1
```

```tex
\pi_\theta^m(a|s_t)
\approx
0,
\qquad
a \ne a^+
```

Hành động được lấy mẫu:

```tex
a_t
\sim
\pi_\theta^m(a_t|s_t)
```

## 9. Reward shaping

Reward tổng:

```tex
r_t^{total}
=
100
\left(
r_{arrival,t}
+
w_g r_{goal,t}
+
w_\theta r_{\theta,t}
+
r_{time}
+
r_{collision,t}
+
w_s r_{safe,t}
+
w_{rvo}r_{rvo,t}
\right)
+
r_{traj,t}
```

Các hệ số mặc định:

```tex
w_g=0.1,
\qquad
w_\theta=0.01,
\qquad
w_s=0.5,
\qquad
w_{rvo}=0.01
```

```tex
r_{arrival}=0.25,
\qquad
r_{collision}=-0.25,
\qquad
r_{time}=-0.0125
```

### 9.1. Reward tiến gần mục tiêu

```tex
r_{goal,t}
=
\left\|
p_t-g
\right\|
-
\left\|
p_{t+1}-g
\right\|
```

### 9.2. Reward hướng robot

```tex
\theta_{g,t}
=
\mathrm{atan2}
\left(
g_y-p_{y,t},
g_x-p_{x,t}
\right)
```

```tex
r_{\theta,t}
=
\frac{
\cos(\theta_t-\theta_{g,t})-1
}{
\left\|
p_t-g
\right\|
+
5
}
```

### 9.3. Phạt thời gian

```tex
r_{time}
=
-0.0125
```

### 9.4. Thưởng đến đích

```tex
r_{arrival,t}
=
\begin{cases}
r_{arrival},
&
\left\|
p_{t+1}-g
\right\|
\le
r_{robot}
+
\epsilon_g
\\
0,
&
\text{ngược lại}
\end{cases}
```

### 9.5. Phạt va chạm

```tex
r_{collision,t}
=
\begin{cases}
r_{collision},
&
\text{nếu xảy ra va chạm}
\\
0,
&
\text{ngược lại}
\end{cases}
```

### 9.6. Phạt khoảng cách an toàn

Gọi `d_i` là khoảng cách nhỏ nhất từ robot đến đối tượng `i`:

```tex
r_{safe,t}
=
\sum_i
\min
\left(
0,
d_i-d_{disc}
\right)
```

Với:

```tex
d_{disc}
=
0.2m
```

### 9.7. Reward RVO chính xác

Thời gian va chạm hiệu chỉnh:

```tex
\bar{t}_{exp,i}
=
\max(t_{exp,i},0)
```

```tex
q_i
=
\frac{1}{\bar{t}_{exp,i}+1}
```

Nếu đối tượng không nằm trong vùng VO:

```tex
r_{rvo,i}=0,
\qquad
\mathbb{I}_{VO,i}=0
```

Nếu đối tượng nằm trong vùng VO:

```tex
r_{rvo,i}
=
\begin{cases}
3\left(\frac{1}{4}-q_i\right)-0.1,
&
\bar{t}_{exp,i}<1.0
\\
\frac{1}{4}-q_i-0.1,
&
\bar{t}_{exp,i}\ge 1.0
\end{cases}
```

Tổng RVO reward:

```tex
r_{rvo,t}
=
\sum_i
\mathbb{I}_{VO,i}
r_{rvo,i}
```

## 10. Privileged learning và trajectory reward

Quỹ đạo dự đoán:

```tex
\tau_t
=
\left\{
s_{t+1}^{pred},
s_{t+2}^{pred},
\dots,
s_{t+K}^{pred}
\right\}
```

Trajectory reward:

```tex
r_{traj,t}
=
\sum_{i=1}^{K-1}
\gamma_{PL}^{i}
r_{eval}
\left(
s_{t+i}^{pred}
\right)
```

Với:

```tex
K=4,
\qquad
\gamma_{PL}=0.9
```

Hàm đánh giá trạng thái tương lai:

```tex
r_{eval}(s)
=
w_\theta r_\theta(s)
+
w_s r_{safe}(s)
+
w_{rvo}r_{rvo}(s)
+
r_{collision}(s)
```

Reward tổng có privileged learning:

```tex
r_t^{total}
=
r_t
+
\sum_{i=1}^{K-1}
\gamma_{PL}^{i}
r_{eval}
\left(
s_{t+i}^{pred}
\right)
```

## 11. PPO và Actor-Critic

Actor:

```tex
\pi_\theta(a_t|s_t)
```

Actor sau action masking:

```tex
\pi_\theta^m(a_t|s_t)
```

Critic:

```tex
V_\phi(s_t)
=
\mathbb{E}_{\pi_\theta}
\left[
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}
\mid
s_t
\right]
```

TD error:

```tex
\delta_t
=
r_t
+
\gamma V_\phi(s_{t+1})
-
V_\phi(s_t)
```

GAE:

```tex
\hat{A}_t
=
\sum_{l=0}^{\infty}
\left(
\gamma\lambda_{GAE}
\right)^l
\delta_{t+l}
```

Các tham số chính:

```tex
\gamma=0.99,
\qquad
\lambda_{GAE}=0.95
```

PPO ratio:

```tex
\rho_t(\theta)
=
\frac{
\pi_\theta^m(a_t|s_t)
}{
\pi_{\theta_{old}}^m(a_t|s_t)
}
```

Clipped surrogate objective:

```tex
L^{CLIP}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)\hat{A}_t,
\mathrm{clip}
\left(
\rho_t(\theta),
1-\epsilon,
1+\epsilon
\right)
\hat{A}_t
\right)
\right]
```

Với:

```tex
\epsilon=0.2
```

Value loss:

```tex
L_V(\phi)
=
\mathbb{E}_t
\left[
\left(
V_\phi(s_t)
-
\hat{R}_t
\right)^2
\right]
```

Entropy regularization:

```tex
L_H(\theta)
=
-
\mathbb{E}_t
\left[
\mathcal{H}
\left(
\pi_\theta^m(\cdot|s_t)
\right)
\right]
```

Tổng loss:

```tex
L_{PPO}
=
L_{policy}
+
c_v L_V
+
c_e L_H
```

Với:

```tex
c_v=0.5,
\qquad
c_e=0.001
```

## 12. Planner tầng thấp trong huấn luyện

Tầng PPO chọn local goal:

```tex
a_t
\rightarrow
g_t^{local}
\rightarrow
g_t^{map}
```

Planner tầng thấp nhận trạng thái robot, local goal và vật cản:

```tex
u_t^{low}
=
f_{planner}
\left(
x_t,
g_t^{map},
\mathcal{O}_t
\right)
```

Với robot vi sai, điều khiển mức thấp có thể biểu diễn dưới dạng gia tốc hai bánh:

```tex
u_t^{low}
=
\begin{bmatrix}
a_{L,t} \\
a_{R,t}
\end{bmatrix}
```

Vận tốc bánh được cập nhật:

```tex
v_{L,t+1}
=
v_{L,t}
+
a_{L,t}\Delta t
```

```tex
v_{R,t+1}
=
v_{R,t}
+
a_{R,t}\Delta t
```

Vận tốc tịnh tiến và vận tốc góc:

```tex
v_t
=
\frac{
v_{R,t}
+
v_{L,t}
}{2}
```

```tex
\omega_t
=
\frac{
v_{R,t}
-
v_{L,t}
}{B}
```

Động học robot vi sai:

```tex
\theta_{t+1}
=
\theta_t
+
\omega_t \Delta t
```

```tex
p_{x,t+1}
=
p_{x,t}
+
v_t\cos(\theta_t)\Delta t
```

```tex
p_{y,t+1}
=
p_{y,t}
+
v_t\sin(\theta_t)\Delta t
```

## 13. MPPI trong tầng điều khiển cục bộ

Trạng thái robot:

```tex
x_t
=
\begin{bmatrix}
p_{x,t} \\
p_{y,t} \\
\theta_t
\end{bmatrix}
```

Điều khiển MPPI:

```tex
u_t
=
\begin{bmatrix}
v_{x,t} \\
\omega_{z,t}
\end{bmatrix}
```

Mô hình động học rời rạc:

```tex
x_{t+1}
=
F(x_t,u_t)
```

Cụ thể:

```tex
\theta_{t+1}
=
\theta_t
+
\omega_{z,t}\Delta t
```

```tex
p_{x,t+1}
=
p_{x,t}
+
v_{x,t}\cos(\theta_t)\Delta t
```

```tex
p_{y,t+1}
=
p_{y,t}
+
v_{x,t}\sin(\theta_t)\Delta t
```

Chuỗi điều khiển danh định:

```tex
U
=
\left\{
u_0,u_1,\dots,u_{T-1}
\right\}
```

Lấy mẫu điều khiển:

```tex
v_t^k
=
u_t
+
\epsilon_t^k
```

```tex
\epsilon_t^k
\sim
\mathcal{N}
\left(
0,\Sigma
\right)
```

Với robot vi sai:

```tex
\epsilon_t^k
=
\begin{bmatrix}
\epsilon_{v_x,t}^k \\
\epsilon_{\omega_z,t}^k
\end{bmatrix}
```

```tex
\epsilon_{v_x,t}^k
\sim
\mathcal{N}(0,\sigma_{v_x}^2)
```

```tex
\epsilon_{\omega_z,t}^k
\sim
\mathcal{N}(0,\sigma_{\omega_z}^2)
```

Thông số chân trời:

```tex
K=2000,
\qquad
T=80,
\qquad
\Delta t=0.1s
```

```tex
T_{horizon}
=
T\Delta t
=
8s
```

Quỹ đạo mẫu thứ `k`:

```tex
\tau^k
=
\left\{
x_0^k,
x_1^k,
\dots,
x_T^k
\right\}
```

Rollout:

```tex
x_{t+1}^k
=
F(x_t^k,v_t^k)
```

Tổng chi phí:

```tex
S_k
=
\phi(x_T^k)
+
\sum_{t=0}^{T-1}
c(x_t^k,v_t^k)
```

Dạng tổng hợp theo critic:

```tex
S_k
=
\sum_i
J_i(\tau^k,V^k)
```

Có thể viết:

```tex
S_k
=
J_{constraint}^k
+
J_{obs}^k
+
J_{human}^k
+
J_{goal}^k
+
J_{goal\_angle}^k
+
J_{align}^k
+
J_{follow}^k
+
J_{forward}^k
+
J_{twirl}^k
+
J_{control}^k
```

Hiệu chỉnh control cost trong importance sampling:

```tex
S_k
\leftarrow
S_k
+
\frac{\gamma_{mppi}}{\sigma_{v_x}^2}
\sum_{t=0}^{T-1}
u_{v_x,t}
\left(
v_{x,t}^k
-
u_{v_x,t}
\right)
```

```tex
S_k
\leftarrow
S_k
+
\frac{\gamma_{mppi}}{\sigma_{\omega_z}^2}
\sum_{t=0}^{T-1}
u_{\omega,t}
\left(
\omega_{z,t}^k
-
u_{\omega,t}
\right)
```

Với:

```tex
\gamma_{mppi}=0.015
```

Chuẩn hóa chi phí:

```tex
\rho
=
\min_k S_k
```

Trọng số MPPI:

```tex
w_k
=
\frac{
\exp
\left(
-
\frac{1}{\lambda}
\left(
S_k-\rho
\right)
\right)
}{
\sum_{j=1}^{K}
\exp
\left(
-
\frac{1}{\lambda}
\left(
S_j-\rho
\right)
\right)
}
```

Với:

```tex
\lambda=0.3
```

Cập nhật điều khiển theo nhiễu:

```tex
u_t^*
=
u_t
+
\sum_{k=1}^{K}
w_k
\epsilon_t^k
```

Dạng tương đương theo sampled controls:

```tex
u_t^*
=
\sum_{k=1}^{K}
w_k
v_t^k
```

Với từng thành phần:

```tex
u_{v_x,t}^*
=
\sum_{k=1}^{K}
w_k
v_{x,t}^k
```

```tex
u_{\omega,t}^*
=
\sum_{k=1}^{K}
w_k
\omega_{z,t}^k
```

Ràng buộc vận tốc:

```tex
v_{x,min}
\le
v_{x,t}
\le
v_{x,max}
```

```tex
|\omega_{z,t}|
\le
\omega_{z,max}
```

Với:

```tex
v_{x,min}=-0.2,
\qquad
v_{x,max}=0.3,
\qquad
\omega_{z,max}=0.3
```

Receding horizon:

```tex
u_t^{exec}
=
u_0^*
```

Warm-start:

```tex
u_0 \leftarrow u_1,
\quad
u_1 \leftarrow u_2,
\quad
\dots,
\quad
u_{T-2} \leftarrow u_{T-1}
```

## 14. Chi phí vật cản động trong MPPI

Dự đoán vị trí người đi bộ:

```tex
p_{h,t}
=
p_{h,0}
+
v_h t\Delta t
```

Khoảng cách giữa quỹ đạo robot mẫu `k` và người `h`:

```tex
d_{k,t,h}^2
=
\left(
x_t^k-x_{h,t}
\right)^2
+
\left(
y_t^k-y_{h,t}
\right)^2
```

Điều kiện mất an toàn:

```tex
d_{k,t,h}
<
r_h
+
r_{robot}
+
d_{safe}
```

Chi phí người đi bộ:

```tex
J_{human}^k
=
w_h
\sum_h
\sum_{t=0}^{T-1}
\mathbb{I}
\left[
d_{k,t,h}
<
r_h
+
r_{robot}
+
d_{safe}
\right]
C_h
```

## 15. Liên hệ vận tốc robot vi sai và vận tốc bánh

Từ vận tốc bánh sang vận tốc robot:

```tex
v
=
\frac{
v_R
+
v_L
}{2}
```

```tex
\omega
=
\frac{
v_R
-
v_L
}{B}
```

Từ vận tốc robot sang vận tốc bánh:

```tex
v_L
=
v
-
\frac{B}{2}
\omega
```

```tex
v_R
=
v
+
\frac{B}{2}
\omega
```

Nếu điều khiển là gia tốc bánh:

```tex
v_{L,t+1}
=
v_{L,t}
+
a_{L,t}
\Delta t
```

```tex
v_{R,t+1}
=
v_{R,t}
+
a_{R,t}
\Delta t
```

