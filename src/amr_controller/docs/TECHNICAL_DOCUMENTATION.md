# Tài liệu Kỹ thuật: Package `amr_controller`
## Adaptive Model Predictive Contouring Control (AMPCC) cho Robot Vi sai

---

## Mục lục

1. [Tổng quan: Robot cần gì để chạy?](#1-tổng-quan-robot-cần-gì-để-chạy)
2. [Sơ đồ kiến trúc tổng thể](#2-sơ-đồ-kiến-trúc-tổng-thể)
3. [Cấu trúc file trong package](#3-cấu-trúc-file-trong-package)
4. [Phần 1: Mô hình Robot — "Robot hiểu mình như thế nào?"](#4-phần-1-mô-hình-robot)
5. [Phần 2: RLS Estimator — "Học lại mô hình từ thực tế"](#5-phần-2-rls-estimator)
6. [Phần 3: Contouring & Lag Error — "Robot lệch đường bao nhiêu?"](#6-phần-3-contouring--lag-error)
7. [Phần 4: Bài toán tối ưu MPCC — "Tìm lệnh tốt nhất"](#7-phần-4-bài-toán-tối-ưu-mpcc)
8. [Phần 5: iLQR Solver — "Cách giải bài toán tối ưu"](#8-phần-5-ilqr-solver)
9. [Phần 6: ROS 2 Node — "Kết nối mọi thứ lại"](#9-phần-6-ros-2-node)
10. [Tra cứu nhanh: Công thức → Code](#10-tra-cứu-nhanh-công-thức--code)
11. [Cách chạy thử](#11-cách-chạy-thử)

---

## 1. Tổng quan: Robot cần gì để chạy?

Hãy tưởng tượng bạn đang lái xe trên đường:

1. **Bạn cần biết mình đang ở đâu** → Odometry (cảm biến vị trí)
2. **Bạn cần biết đường đi** → Global Path (đường dẫn từ A đến B)
3. **Bạn cần quyết định bẻ lái bao nhiêu, đạp ga bao nhiêu** → Controller (bộ điều khiển)

Package `amr_controller` chính là **bộ não điều khiển** — nó nhận vị trí robot + đường dẫn,
rồi tính ra vận tốc thẳng (`v`) và vận tốc xoay (`ω`) tối ưu nhất để robot bám đường.

**Điểm đặc biệt so với controller đơn giản:**

| Controller đơn giản (PID) | AMPCC (của chúng ta) |
|---|---|
| Nhìn 1 điểm phía trước | Nhìn xa N bước (20 bước × 0.05s = 1 giây) |
| Giả sử robot không đổi | Tự học lại mô hình robot khi tải thay đổi |
| Bám 1 điểm trên đường | Bám cả đường viền, cho phép tăng/giảm tốc linh hoạt |

---

## 2. Sơ đồ kiến trúc tổng thể

```
  /odometry/filtered                           /global_path
       │                                            │
       ▼                                            ▼
  ┌─────────────────────────────────────────────────────────┐
  │                    AMPCCNode (ROS 2)                     │
  │                                                         │
  │  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
  │  │   RLS    │───▶│ AdaptiveMPCC │───▶│  MPCCSolver   │  │
  │  │Estimator │    │  (Mô hình +  │    │   (iLQR)      │  │
  │  │          │    │   Sai số)    │    │               │  │
  │  └──────────┘    └──────────────┘    └───────┬───────┘  │
  │       ▲                                      │          │
  │       │              v_cmd, ω_cmd ◀──────────┘          │
  └───────┼──────────────────┬──────────────────────────────┘
          │                  │
     (học mô hình)           ▼
                     /diff_cont/cmd_vel
                    (Robot thực sự di chuyển)
```

**Luồng chạy mỗi 0.05 giây (20 Hz):**

```
Bước 1: Đọc vị trí robot từ odometry          → (x, y, ψ, v, ω)
Bước 2: RLS cập nhật mô hình robot             → θ̂ = [α_v, β_v, α_ω, β_ω]
Bước 3: Chiếu vị trí robot lên đường dẫn      → θ_progress (đã đi được bao xa)
Bước 4: Ghép thành trạng thái đầy đủ           → z = [x, y, ψ, v, ω, θ]
Bước 5: Solver tìm chuỗi điều khiển tối ưu    → u*₀, u*₁, ..., u*₁₉
Bước 6: Gửi lệnh đầu tiên ra robot            → v_cmd, ω_cmd
Bước 7: Lưu lại cho bước tiếp theo             → (warm-start + dữ liệu RLS)
```

---

## 3. Cấu trúc file trong package

```
amr_controller/
│
├── include/amr_controller/          ← Header files (khai báo)
│   ├── types.hpp                    ← Kiểu dữ liệu, hằng số, struct
│   ├── rls_estimator.hpp            ← Class RLSEstimator
│   ├── adaptive_mpcc.hpp            ← Class AdaptiveMPCC + WaypointPath
│   ├── mpcc_solver.hpp              ← Class MPCCSolver (iLQR)
│   └── ampcc_node.hpp               ← Class AMPCCNode (ROS 2)
│
├── src/                             ← Implementation files
│   ├── rls_estimator.cpp            ← Thuật toán RLS
│   ├── adaptive_mpcc.cpp            ← Mô hình + sai số + dự báo
│   ├── mpcc_solver.cpp              ← Giải bài toán tối ưu
│   ├── ampcc_node.cpp               ← Kết nối ROS 2
│   └── ampcc_main.cpp               ← Hàm main()
│
├── config/ampcc_params.yaml         ← Tham số cấu hình
├── launch/ampcc.launch.py           ← Launch file
├── CMakeLists.txt                   ← Build configuration
└── package.xml                      ← Package metadata
```

**Mối quan hệ giữa các class:**

```
types.hpp (mọi người đều dùng)
    ▲
    │
rls_estimator.hpp ──▶ adaptive_mpcc.hpp ──▶ mpcc_solver.hpp ──▶ ampcc_node.hpp
   (ước lượng θ)       (mô hình + sai số)     (tối ưu hóa)      (ROS 2 node)
```

---

## 4. Phần 1: Mô hình Robot

### 4.1 Ý tưởng cơ bản

Robot vi sai (differential drive) có 2 bánh chủ động. Để điều khiển nó,
ta cần biết: **"Nếu tôi ra lệnh v_cmd và ω_cmd, robot sẽ di chuyển thế nào
sau 0.05 giây?"**

Đó chính là **mô hình động lực học** — phương trình dự đoán trạng thái tương lai.

### 4.2 Trạng thái robot

Ta định nghĩa **trạng thái** robot bao gồm 5 giá trị:

```
z = [x, y, ψ, v, ω]
```

| Ký hiệu | Ý nghĩa | Đơn vị |
|----------|---------|--------|
| `x` | Vị trí ngang | mét (m) |
| `y` | Vị trí dọc | mét (m) |
| `ψ` (psi) | Góc hướng robot đang quay mặt về | radian (rad) |
| `v` | Vận tốc thẳng (tiến/lùi) | m/s |
| `ω` (omega) | Vận tốc xoay (quay trái/phải) | rad/s |

Trong code → `types.hpp`:
```cpp
enum StateIdx : int {
  kX     = 0,   // z[0] = x
  kY     = 1,   // z[1] = y
  kPsi   = 2,   // z[2] = ψ
  kV     = 3,   // z[3] = v
  kOmega = 4,   // z[4] = ω
  kTheta = 5    // z[5] = θ (giải thích ở phần sau)
};
```

### 4.3 Phương trình chuyển động (rời rạc hóa Euler)

**Lý thuyết (§2.1):** Nếu biết trạng thái hiện tại ở thời điểm `k` và lệnh
điều khiển `u`, trạng thái tiếp theo `k+1` là:

$$x_{k+1} = x_k + v_k \cos(\psi_k) \cdot \Delta t$$

$$y_{k+1} = y_k + v_k \sin(\psi_k) \cdot \Delta t$$

$$\psi_{k+1} = \psi_k + \omega_k \cdot \Delta t$$

$$v_{k+1} = \alpha_v \cdot v_k + \beta_v \cdot v_{cmd}$$

$$\omega_{k+1} = \alpha_\omega \cdot \omega_k + \beta_\omega \cdot \omega_{cmd}$$

**Giải thích từng dòng cho người mới:**

- **Dòng 1-2:** Robot tiến thẳng theo hướng `ψ` với tốc độ `v` trong thời gian
  `Δt`. Dùng cos/sin để tách thành phần x và y.
  - Ví dụ: `ψ = 0` (hướng đông) → `cos(0)=1, sin(0)=0` → chỉ đi theo x.

- **Dòng 3:** Góc hướng thay đổi theo vận tốc xoay.

- **Dòng 4-5:** Vận tốc mới = (phần cũ còn lại sau ma sát) + (phần do lệnh
  điều khiển tạo ra). Đây là **tham số proxy** — giải thích ở bên dưới.

**Trong code** → `adaptive_mpcc.cpp`, hàm `predictState()`:

```cpp
// File: src/adaptive_mpcc.cpp, dòng ~378-394

z_next(kX)     = x   + v * std::cos(psi) * dt;         // x_{k+1}
z_next(kY)     = y   + v * std::sin(psi) * dt;         // y_{k+1}
z_next(kPsi)   = psi + omega * dt;                      // ψ_{k+1}
z_next(kV)     = alpha_v     * v     + beta_v     * R;  // v_{k+1}
z_next(kOmega) = alpha_omega * omega + beta_omega * M;  // ω_{k+1}
z_next(kTheta) = theta_prg  + v_t * dt;                 // θ_{k+1} (progress)
```

### 4.4 Tham số Proxy θ — "Tại sao không dùng m, J trực tiếp?"

**Vấn đề:** Phương trình vật lý thực sự là:

$$m \dot{v} = F - bv \quad \Rightarrow \quad v_{k+1} = \left(1 - \frac{b \Delta t}{m}\right) v_k + \frac{\Delta t}{m} F_k$$

Nếu ước lượng trực tiếp `m`, `b` → phương trình phi tuyến theo tham số
→ khó dùng thuật toán RLS (chỉ hoạt động tốt với mô hình tuyến tính).

**Giải pháp:** Đặt các **tham số proxy**:

| Proxy | Công thức | Ý nghĩa trực giác |
|-------|-----------|-------------------|
| `α_v` | $1 - \frac{b \Delta t}{m}$ | "Bao nhiêu % vận tốc cũ còn lại sau ma sát" |
| `β_v` | $\frac{\Delta t}{m}$ | "Lệnh điều khiển tạo ra bao nhiêu vận tốc" |
| `α_ω` | $1 - \frac{c \Delta t}{J}$ | Tương tự cho vận tốc xoay |
| `β_ω` | $\frac{\Delta t}{J}$ | Tương tự cho vận tốc xoay |

Bây giờ phương trình thành **tuyến tính**: `v_{k+1} = α_v · v_k + β_v · u_k`

Trong code → `types.hpp`:
```cpp
enum ParamIdx : int {
  kAlphaV     = 0,  // θ[0] = α_v
  kBetaV      = 1,  // θ[1] = β_v
  kAlphaOmega = 2,  // θ[2] = α_ω
  kBetaOmega  = 3   // θ[3] = β_ω
};
```

### 4.5 Trạng thái mở rộng (Augmented State)

Ngoài 5 biến trên, MPCC cần thêm 1 biến nữa: **θ_progress** — robot đã đi
được bao xa trên đường dẫn.

```
z_augmented = [x, y, ψ, v, ω, θ_progress]ᵀ     (6 biến)
u_augmented = [v_cmd, ω_cmd, v_θ]ᵀ              (3 biến)
```

- `v_θ` là **vận tốc tiến độ ảo** — biến tối ưu cho phép solver tự quyết định
  robot nên đi nhanh hay chậm trên đường.

### 4.6 Linearization — "Xấp xỉ để giải nhanh"

Bài toán tối ưu cần tính đạo hàm. Ta lấy Jacobian (đạo hàm riêng) của mô hình:

$$A = \frac{\partial f}{\partial z}, \quad B = \frac{\partial f}{\partial u}$$

**Ma trận A (6×6):**

$$A = \begin{bmatrix}
1 & 0 & -v\sin\psi \cdot \Delta t & \cos\psi \cdot \Delta t & 0 & 0 \\
0 & 1 & v\cos\psi \cdot \Delta t & \sin\psi \cdot \Delta t & 0 & 0 \\
0 & 0 & 1 & 0 & \Delta t & 0 \\
0 & 0 & 0 & \alpha_v & 0 & 0 \\
0 & 0 & 0 & 0 & \alpha_\omega & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

**Đọc cách hiểu:** Dòng 1, cột 3 = `−v·sin(ψ)·Δt` nghĩa là: "Nếu góc ψ thay
đổi 1 rad, thì x thay đổi `−v·sin(ψ)·Δt` mét".

**Ma trận B (6×3):**

$$B = \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
\beta_v & 0 & 0 \\
0 & \beta_\omega & 0 \\
0 & 0 & \Delta t
\end{bmatrix}$$

**Trong code** → `adaptive_mpcc.cpp`, hàm `getLinearizedModel()`:

```cpp
// Ma trận A — bắt đầu từ ma trận đơn vị rồi thêm các thành phần
model.A.setIdentity();

model.A(kX, kPsi) = -v * sin_psi * dt;    // ∂x⁺/∂ψ
model.A(kX, kV)   =  cos_psi * dt;        // ∂x⁺/∂v
model.A(kY, kPsi) =  v * cos_psi * dt;    // ∂y⁺/∂ψ
model.A(kY, kV)   =  sin_psi * dt;        // ∂y⁺/∂v
model.A(kPsi, kOmega) = dt;               // ∂ψ⁺/∂ω
model.A(kV, kV) = alpha_v;                // ∂v⁺/∂v
model.A(kOmega, kOmega) = alpha_omega;    // ∂ω⁺/∂ω

// Ma trận B
model.B.setZero();
model.B(kV, kForce)      = beta_v;        // ∂v⁺/∂v_cmd
model.B(kOmega, kTorque) = beta_omega;    // ∂ω⁺/∂ω_cmd
model.B(kTheta, kVTheta) = dt;            // ∂θ⁺/∂v_θ
```

---

## 5. Phần 2: RLS Estimator

### 5.1 Vấn đề cần giải quyết

Robot của bạn nặng 4 kg khi không tải. Khi chở hàng 10 kg, nó nặng 14 kg.
Tham số `β_v = Δt/m` thay đổi từ `0.05/4 = 0.0125` xuống `0.05/14 = 0.0036`.

Nếu controller vẫn dùng giá trị cũ → tính sai → robot đi hỏng đường!

**RLS (Recursive Least Squares)** giải quyết bằng cách: quan sát robot thực sự
phản ứng thế nào với lệnh, rồi **tự động điều chỉnh** tham số θ.

### 5.2 Ý tưởng trực giác

Giả sử ta có mô hình: `v_{k+1} = α_v · v_k + β_v · v_cmd_k`

- Ta biết `v_k = 0.5`, `v_cmd_k = 1.0` (đã gửi bước trước)
- Mô hình dự đoán: `v_{k+1} = 0.95 × 0.5 + 0.05 × 1.0 = 0.525`
- Nhưng đo thực tế: `v_{k+1} = 0.49` (robot nặng hơn dự kiến!)
- Sai số: `ε = 0.49 − 0.525 = −0.035`
- RLS điều chỉnh α_v, β_v để lần sau dự đoán chính xác hơn.

### 5.3 Thuật toán RLS (§3.1)

Viết gọn lại: Mô hình có dạng **y = φᵀ · θ** (tuyến tính theo tham số)

Với hệ vận tốc thẳng:
- `y` = `v_k` (giá trị đo được)  
- `φ` = `[v_{k-1}, v_cmd_{k-1}]ᵀ` (dữ liệu cũ)  
- `θ` = `[α_v, β_v]ᵀ` (cần ước lượng)

**Bốn bước RLS mỗi chu kỳ:**

**Bước 1 — Tính Kalman Gain (K):**

$$K_k = \frac{P_{k-1} \cdot \phi_k}{\lambda + \phi_k^T \cdot P_{k-1} \cdot \phi_k}$$

- `P` = ma trận hiệp phương sai (2×2) — thể hiện "mức độ không chắc chắn"
  về θ. Ban đầu P rất lớn (= 1000·I) vì ta chưa biết gì.
- `λ` = hệ số quên (0.98). Nhỏ hơn 1 → quên dữ liệu cũ nhanh hơn → thích
  nghi nhanh hơn. `λ = 1` → nhớ hết → ổn định nhưng chậm thay đổi.

**Bước 2 — Tính sai số (innovation):**

$$\epsilon_k = y_k - \phi_k^T \cdot \hat{\theta}_{k-1}$$

- `y_k` = giá trị đo thực
- `φᵀ · θ̂` = giá trị mô hình dự đoán
- `ε` = chênh lệch. Nếu `ε ≈ 0` → mô hình đã chính xác.

**Bước 3 — Cập nhật tham số:**

$$\hat{\theta}_k = \hat{\theta}_{k-1} + K_k \cdot \epsilon_k$$

- Điều chỉnh θ theo hướng giảm sai số. K lớn → điều chỉnh mạnh.

**Bước 4 — Cập nhật hiệp phương sai:**

$$P_k = \frac{1}{\lambda} \left( P_{k-1} - K_k \cdot \phi_k^T \cdot P_{k-1} \right)$$

- P nhỏ dần → hệ thống ngày càng "tự tin" về θ. Nhưng nhờ `1/λ > 1`,
  P không bao giờ quá nhỏ → luôn sẵn sàng thích nghi.

### 5.4 Trong code minh họa

File: `src/rls_estimator.cpp`, hàm `updateSub()`:

```cpp
// Bước 1: Kalman Gain
const double denom = lambda_ + phi.transpose() * est.P * phi;
const Eigen::Vector2d K = (est.P * phi) / denom;

// Bước 2: Innovation (sai số dự đoán)
const double epsilon = y_measured - phi.dot(est.theta);

// Bước 3: Cập nhật tham số
est.theta += K * epsilon;

// Bước 4: Cập nhật hiệp phương sai
est.P = (1.0 / lambda_) * (est.P - K * phi.transpose() * est.P);
est.P = 0.5 * (est.P + est.P.transpose());  // giữ đối xứng
```

### 5.5 Tại sao 2 bộ RLS riêng biệt?

Phương trình v và ω **không liên quan nhau** (decoupled):
- `v_{k+1} = α_v · v_k + β_v · v_cmd_k` → RLS 2×2 cho `[α_v, β_v]`
- `ω_{k+1} = α_ω · ω_k + β_ω · ω_cmd_k` → RLS 2×2 cho `[α_ω, β_ω]`

Tách thành 2 bài toán nhỏ (2×2) thay vì 1 bài lớn (4×4) → **nhanh hơn** và
**ổn định hơn** về số học.

Trong code → `rls_estimator.cpp`, hàm `update()`:

```cpp
void RLSEstimator::update(...) {
  // Hệ vận tốc thẳng
  const Eigen::Vector2d phi_v(v_prev, R_prev);      // φ = [v_{k-1}, cmd_{k-1}]
  eps_v_ = updateSub(linear_est_, v_measured, phi_v);

  // Hệ vận tốc xoay
  const Eigen::Vector2d phi_omega(omega_prev, M_prev);
  eps_omega_ = updateSub(angular_est_, omega_measured, phi_omega);
}
```

### 5.6 Khôi phục tham số vật lý

Sau khi ước lượng θ, ta có thể tính ngược ra khối lượng, momen quán tính:

$$m = \frac{\Delta t}{\beta_v}, \quad b = \frac{(1 - \alpha_v) \cdot m}{\Delta t}$$

Trong code → `rls_estimator.cpp`:

```cpp
double RLSEstimator::estimateMass(double dt) const {
  return dt / linear_est_.theta(1);  // m = Δt / β_v
}
```

---

## 6. Phần 3: Contouring & Lag Error

### 6.1 Vấn đề: MPC thường vs MPCC

**MPC thường (trajectory tracking):** "Lúc t=2s robot phải ở (3, 4)!"
- Nếu robot chạy chậm do vật cản → nó cố đuổi theo điểm đã bỏ lỡ → giật.

**MPCC (contouring control):** "Robot phải ở **trên đường**, nhưng nhanh hay
chậm tùy ý!"
- Robot bám theo hình dạng đường, tự chọn tốc độ → mượt mà hơn nhiều.

### 6.2 Hai loại sai số

Hãy tưởng tượng bạn đang lái xe trên đường cong:

```
                    Robot ●
                   ╱      ╲
          e_c ←──╱         ╲ (vuông góc với đường)
              ╱             ╲
    ─────●────────────────────── Đường dẫn
     P_ref     e_l
              (song song với đường)
```

| Sai số | Ký hiệu | Ý nghĩa | Muốn gì? |
|--------|---------|---------|----------|
| Contouring Error | `e_c` | Khoảng cách từ robot đến đường (vuông góc) | Càng nhỏ càng tốt (bám sát đường) |
| Lag Error | `e_l` | Robot đi trước hay sau điểm tham chiếu (dọc đường) | Nhỏ, nhưng linh hoạt hơn |

### 6.3 Công thức (§2.3)

Gọi:
- `(x, y)` = vị trí robot
- `(x_ref, y_ref)` = điểm trên đường tại tiến độ θ
- `φ` = góc tiếp tuyến của đường tại điểm đó

$$e_c = \sin(\phi) \cdot (x - x_{ref}) - \cos(\phi) \cdot (y - y_{ref})$$

$$e_l = -\cos(\phi) \cdot (x - x_{ref}) - \sin(\phi) \cdot (y - y_{ref})$$

**Giải thích trực giác:**  
Đây là phép **xoay hệ tọa độ** từ (x, y) toàn cục sang hệ Frenet gắn trên
đường:
- Trục dọc đường = hướng `φ` → chiếu lên đó = `e_l`
- Trục vuông góc đường → chiếu lên đó = `e_c`

### 6.4 Trong code

File: `src/adaptive_mpcc.cpp`, hàm `computeContouringErrors()`:

```cpp
// Lấy điểm tham chiếu trên đường tại tiến độ θ
const PathPoint ref = ref_path_->evaluate(theta_progress);

// Chênh lệch vị trí
const double dx = x - ref.x;
const double dy = y - ref.y;

// Góc tiếp tuyến đường
const double cos_phi = std::cos(ref.phi);
const double sin_phi = std::sin(ref.phi);

// Công thức §2.3
err.e_c =  sin_phi * dx - cos_phi * dy;   // vuông góc
err.e_l = -cos_phi * dx - sin_phi * dy;   // dọc đường
```

### 6.5 Jacobian sai số (cho solver)

Solver cần biết: "Nếu trạng thái z thay đổi một chút, sai số thay đổi bao
nhiêu?" → Đạo hàm riêng C = ∂[e_c, e_l]/∂z

$$C = \begin{bmatrix}
\sin\phi & -\cos\phi & 0 & 0 & 0 & -\kappa \cdot e_l \\
-\cos\phi & -\sin\phi & 0 & 0 & 0 & \kappa \cdot e_c + 1
\end{bmatrix}$$

Với đường thẳng (`κ = 0`), hàng cuối đơn giản thành `[..., 0]` và `[..., 1]`.

Trong code → `computeContouringErrorJacobian()`:

```cpp
jac.C(0, kX) =  sin_phi;                // ∂e_c/∂x
jac.C(0, kY) = -cos_phi;                // ∂e_c/∂y
jac.C(0, kTheta) = -kappa * jac.e(1);   // ∂e_c/∂θ

jac.C(1, kX) = -cos_phi;                // ∂e_l/∂x
jac.C(1, kY) = -sin_phi;                // ∂e_l/∂y
jac.C(1, kTheta) = kappa * jac.e(0) + 1.0;  // ∂e_l/∂θ
```

---

## 7. Phần 4: Bài toán tối ưu MPCC

### 7.1 Câu hỏi cốt lõi

> "Trong 20 bước tiếp theo (1 giây), chuỗi lệnh `[v_cmd, ω_cmd]` nào là
> tốt nhất?"

"Tốt nhất" = **chi phí nhỏ nhất (minimize cost)** + **thỏa mãn giới hạn
(constraints)**.

### 7.2 Hàm chi phí (§3.2)

$$J^* = \min_{u_{0:N-1}} \sum_{j=0}^{N-1} \underbrace{\left( Q_c \cdot e_{c,j}^2 + Q_l \cdot e_{l,j}^2 + Q_u \cdot \|u_j\|^2 + Q_v(v_j - v_{ref})^2 \right)}_{\text{stage cost}} - \underbrace{q_\theta \cdot v_{\theta,j} \cdot \Delta t}_{\text{thưởng tiến độ}}$$

**Giải thích từng thành phần:**

| Thành phần | Trọng số | Ý nghĩa |
|-----------|----------|---------|
| $Q_c \cdot e_c^2$ | `q_c = 10` | Phạt nặng khi robot lệch khỏi đường |
| $Q_l \cdot e_l^2$ | `q_l = 5` | Phạt nhẹ hơn khi robot đi trước/sau điểm tham chiếu |
| $Q_{u_R} \cdot v_{cmd}^2$ | `q_u_R = 0.1` | Phạt nhẹ lệnh lái lớn (tiết kiệm năng lượng) |
| $Q_{u_M} \cdot \omega_{cmd}^2$ | `q_u_M = 0.1` | Phạt nhẹ lệnh xoay lớn (mượt mà) |
| $Q_v \cdot (v - v_{ref})^2$ | `q_v = 1` | Khuyến khích robot đi đúng tốc độ mong muốn |
| $-q_\theta \cdot v_\theta \cdot \Delta t$ | `q_theta = 2` | **Thưởng** khi robot tiến về phía trước (dấu trừ!) |

**Trực giác về trade-off:**

- `q_c` lớn → robot bám chặt đường, nhưng có thể chạy chậm.
- `q_theta` lớn → robot chạy nhanh, nhưng có thể lệch đường hơn.
- Cân bằng `q_c` và `q_theta` → robot vừa bám đường vừa tiến nhanh.

### 7.3 Ràng buộc (Constraints)

```
-v_cmd_max   ≤  v_cmd   ≤  v_cmd_max      (±2.0 m/s)
-ω_cmd_max   ≤  ω_cmd   ≤  ω_cmd_max      (±2.0 rad/s)
 v_theta_min ≤  v_θ     ≤  v_theta_max     (-0.1 ~ 2.5 m/s)
```

Trong code → `mpcc_solver.cpp`, hàm `clampControl()`:

```cpp
AugmentedControl MPCCSolver::clampControl(const AugmentedControl& u) const {
  AugmentedControl uc;
  uc(kForce)  = std::clamp(u(kForce),  -p.v_cmd_max,    p.v_cmd_max);
  uc(kTorque) = std::clamp(u(kTorque), -p.omega_cmd_max, p.omega_cmd_max);
  uc(kVTheta) = std::clamp(u(kVTheta),  p.v_theta_min,   p.v_theta_max);
  return uc;
}
```

### 7.4 Trong code — stage cost

File: `src/adaptive_mpcc.cpp`, hàm `computeStageCost()`:

```cpp
return params_.q_c       * err.e_c * err.e_c         // phạt lệch đường
     + params_.q_l       * err.e_l * err.e_l         // phạt trễ/sớm
     + params_.q_u_R     * v_cmd * v_cmd             // phạt lệnh v lớn
     + params_.q_u_M     * omega_cmd * omega_cmd     // phạt lệnh ω lớn
     + params_.q_v       * v_err * v_err             // bám tốc độ
     + params_.q_u_vtheta * v_theta * v_theta        // regularize v_θ
     - params_.q_theta   * v_theta * params_.dt;     // THƯỞNG tiến độ
```

---

## 8. Phần 5: iLQR Solver

### 8.1 Tại sao cần solver?

Ta có:
- 20 bước × 3 biến điều khiển = **60 biến** cần tìm
- Mô hình phi tuyến (cos, sin)
- Ràng buộc box

Không thể giải bằng tay → dùng thuật toán **iLQR** (iterative Linear-Quadratic
Regulator).

### 8.2 Ý tưởng iLQR (giải thích đơn giản)

iLQR giống như leo núi (tìm đỉnh) bằng cách lặp:

```
Lặp lại cho đến khi hội tụ:
  1. BACKWARD PASS: Đi ngược từ bước N về 0
     - Tại mỗi bước, linearize mô hình → A, B
     - Xấp xỉ cost thành dạng bậc 2 → Q, R
     - Tính feedback gains K, d   (giống Riccati)

  2. FORWARD PASS: Đi xuôi từ bước 0 đến N
     - Áp dụng lệnh mới: u = u_cũ + α·d + K·(z − z_cũ)
     - Mô phỏng robot đi theo lệnh mới
     - Tính cost mới

  3. LINE SEARCH: Chọn bước α tốt nhất
     - Nếu cost giảm → chấp nhận, giảm regularisation μ
     - Nếu cost tăng → giảm α, thử lại
```

### 8.3 Backward Pass — Chi tiết

Tại mỗi bước `j` (đi ngược N-1 → 0):

**Xấp xỉ bậc 2 của cost:**

Dùng Gauss-Newton: $Q_{zz} \approx 2 C^T W C$ với $W = \text{diag}(q_c, q_l)$

```cpp
// src/mpcc_solver.cpp — quadratizeStageCost()

Eigen::Matrix2d W;
W << p.q_c, 0.0,
     0.0,   p.q_l;

qc.Qzz = 2.0 * C.transpose() * W * C;    // Hessian xấp xỉ
qc.Qzz(kV, kV) += 2.0 * p.q_v;           // thêm phần vận tốc
qc.qz = 2.0 * C.transpose() * W * e;     // Gradient
```

**Tính Q-function:**

$$Q_{xx} = Q_{zz} + A^T V_{j+1} A$$
$$Q_{uu} = R_{uu} + B^T V_{j+1} B + \mu I$$
$$Q_{ux} = B^T V_{j+1} A$$

`μI` là **regularisation** — giúp Q_uu luôn dương xác định → nghịch đảo được.

**Tính feedback gains:**

$$K_j = -Q_{uu}^{-1} Q_{ux}, \quad d_j = -Q_{uu}^{-1} Q_u$$

- `K` (3×6): ma trận feedback — khi robot ở sai vị trí, `K` nói "điều chỉnh
  lệnh bao nhiêu".
- `d` (3×1): bước điều chỉnh cơ bản.

Trong code → `backwardPass()`:

```cpp
gains[j].K = -Q_uu_inv * Q_ux;   // feedback gain
gains[j].d = -Q_uu_inv * Q_u;    // feedforward step
```

### 8.4 Forward Pass — Chi tiết

Áp dụng chuỗi lệnh mới với feedback:

$$u_j^{new} = u_j^{ref} + \alpha \cdot d_j + K_j \cdot (z_j^{new} - z_j^{ref})$$

Sau đó clamp và mô phỏng:

```cpp
// src/mpcc_solver.cpp — forwardPass()

AugmentedControl u_j = u_ref[j] + alpha * gains[j].d + gains[j].K * dz;
u_j = clampControl(u_j);              // giới hạn lệnh
z_new.push_back(mpcc_.predictState(z_new.back(), u_j));  // mô phỏng
```

### 8.5 Line Search

Thử `α = 1, 0.5, 0.25, ...` cho đến khi cost giảm (điều kiện Armijo):

$$J_{new} < J_{old} + 10^{-4} \cdot \alpha \cdot \Delta J_{expected}$$

```cpp
if (J_try < sol.cost + armijo) {
  // Chấp nhận bước này!
  sol.z_seq = std::move(z_try);
  sol.u_seq = std::move(u_try);
  sol.cost  = J_try;
}
```

### 8.6 Warm-start — "Nhớ lời giải cũ"

Thay vì giải từ đầu mỗi chu kỳ, ta dịch lời giải cũ:

```
Chu kỳ k:   u₀  u₁  u₂  ...  u₁₉
                    ↓ shift
Chu kỳ k+1: u₁  u₂  u₃  ...  u₁₉  u₁₉
```

→ Solver hội tụ nhanh hơn (thường 2-3 iterations thay vì 10).

---

## 9. Phần 6: ROS 2 Node

### 9.1 Topics

File: `src/ampcc_node.cpp`

| Hướng | Topic | Kiểu | Ý nghĩa |
|-------|-------|------|---------|
| Subscribe | `/odometry/filtered` | `Odometry` | Vị trí + vận tốc robot |
| Subscribe | `/global_path` | `Path` | Đường dẫn từ global planner |
| Subscribe | `/goal_pose` | `PoseStamped` | Điểm đến (tạo đường thẳng nếu ko có path) |
| Publish | `/diff_cont/cmd_vel` | `TwistStamped` | Lệnh v, ω cho diff_drive_controller |
| Publish | `/predicted_path` | `Path` | Quỹ đạo dự đoán N bước (để hiện trên RViz) |
| Publish | `/ampcc_diagnostics` | `Float64MultiArray` | Tham số θ, sai số, ... |

### 9.2 Control Loop — Giải thích từng bước

```cpp
void AMPCCNode::controlLoop() {
  // Bước 1: Kiểm tra đã có odometry chưa
  if (!odom_received_) return;

  // Bước 2: Kiểm tra đã có đường dẫn chưa
  if (!path_active_) { publishCmd(0, 0); return; }

  // Bước 3: Đã đến đích chưa?
  if (goalReached()) { publishCmd(0, 0); return; }

  // Bước 4: CẬP NHẬT MÔ HÌNH bằng RLS
  //   - Input: v đo được, ω đo được, lệnh cũ
  //   - Output: θ̂ mới (α_v, β_v, α_ω, β_ω)
  mpcc_->updateModel(v_, omega_, v_prev_, omega_prev_,
                     v_cmd_prev_, omega_cmd_prev_);

  // Bước 5: CHIẾU vị trí lên đường → θ_progress
  theta_progress_ = ref_path_->projectOntoPath(x_, y_, theta_progress_);

  // Bước 6: XÂY DỰNG trạng thái z₀ = [x, y, ψ, v, ω, θ]
  AugmentedState z0;
  z0 << x_, y_, psi_, v_, omega_, theta_progress_;

  // Bước 7: GIẢI BÀI TOÁN TỐI ƯU
  const auto sol = solver_->solve(z0, u_warm_);

  // Bước 8: LẤY LỆNH ĐẦU TIÊN → gửi cho robot
  v_cmd     = sol.u_seq[0](kForce);    // v_cmd
  omega_cmd = sol.u_seq[0](kTorque);   // ω_cmd
  publishCmd(v_cmd, omega_cmd);

  // Bước 9: LƯU cho chu kỳ sau
  u_warm_ = MPCCSolver::shiftSequence(sol.u_seq);  // warm-start
}
```

### 9.3 Publish lệnh

Lệnh được gửi dưới dạng `TwistStamped` — giống y hệt KeyboardInput:

```cpp
void AMPCCNode::publishCmd(double v_cmd, double omega_cmd) {
  geometry_msgs::msg::TwistStamped msg;
  msg.header.stamp    = this->get_clock()->now();
  msg.header.frame_id = "base_link";
  msg.twist.linear.x  = v_cmd;      // [m/s]  tiến/lùi
  msg.twist.angular.z = omega_cmd;   // [rad/s] quay trái/phải
  cmd_pub_->publish(msg);
}
```

---

## 10. Tra cứu nhanh: Công thức → Code

| # | Công thức (Tài liệu) | File | Hàm |
|---|---|---|---|
| §2.1 | $z_{k+1} = f(z_k, u_k, \theta)$ | `adaptive_mpcc.cpp` | `predictState()` |
| §2.1 | Ma trận A(θ), B(θ) | `adaptive_mpcc.cpp` | `getLinearizedModel()` |
| §2.1 | Proxy θ ↔ (m, b, J, c) | `rls_estimator.cpp` | `estimateMass()` ... |
| §2.3 | $e_c$, $e_l$ | `adaptive_mpcc.cpp` | `computeContouringErrors()` |
| §2.3 | Jacobian C = ∂[e_c,e_l]/∂z | `adaptive_mpcc.cpp` | `computeContouringErrorJacobian()` |
| §3.1 | RLS: K, ε, θ̂, P | `rls_estimator.cpp` | `updateSub()` |
| §3.2 | Hàm cost J | `adaptive_mpcc.cpp` | `computeStageCost()` |
| §3.2 | Bài toán min J | `mpcc_solver.cpp` | `solve()` |
| — | Backward Riccati | `mpcc_solver.cpp` | `backwardPass()` |
| — | Forward rollout | `mpcc_solver.cpp` | `forwardPass()` |
| — | Quadratic approx | `mpcc_solver.cpp` | `quadratizeStageCost()` |
| — | Control clamping | `mpcc_solver.cpp` | `clampControl()` |

---

## 11. Cách chạy thử

### Bước 1: Build

```bash
cd ~/amr_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select amr_controller
source install/setup.bash
```

### Bước 2: Khởi động robot (simulation)

```bash
ros2 launch amr_bringup simulated_robot.launch.py
```

### Bước 3: Khởi động localization + planner

```bash
ros2 launch amr_localization local_localization.launch.py
ros2 launch amr_planner planner_test.launch.py
```

### Bước 4: Khởi động AMPCC controller

```bash
ros2 launch amr_controller ampcc.launch.py use_sim_time:=true
```

### Bước 5: Gửi mục tiêu

```bash
ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
  '{header: {frame_id: "map"}, pose: {position: {x: 3.0, y: 2.0}}}'
```

### Kiểm tra hoạt động

```bash
# Xem lệnh đang gửi
ros2 topic echo /diff_cont/cmd_vel

# Xem tham số RLS đang ước lượng
ros2 topic echo /ampcc_diagnostics

# Xem quỹ đạo dự đoán trên RViz
# Thêm display Path → topic /predicted_path
```

### Thay đổi tham số (không cần build lại)

Chỉnh file `config/ampcc_params.yaml`:

```yaml
mpcc:
  v_ref: 0.3       # Giảm tốc nếu robot đi quá nhanh
  q_c: 20.0        # Tăng nếu robot lệch đường nhiều
  q_theta: 1.0     # Giảm nếu robot quá tham tiến độ
```

---

**Tài liệu được tạo cho package `amr_controller` v0.1.0**
**Dựa trên framework H-AMPCC (Hierarchical Adaptive Model Predictive Contouring Control)**
