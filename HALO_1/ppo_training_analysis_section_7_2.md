# 7.2. Phân tích quá trình huấn luyện PPO

Tài liệu này dùng để viết trực tiếp mục **7.2. Phân tích quá trình huấn luyện PPO** trong báo cáo thiết kế/luận văn. Nội dung được viết bám theo code trong:

- `HALO_1/drl_moudle/train_ppo.py`
- `HALO_1/drl_moudle/algorithms/mpc_ppo.py`
- `HALO_1/drl_moudle/algorithms/graph_ppo.py`
- `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py`
- `HALO_1/drl_moudle/configs/config.py`
- log huấn luyện: `HALO_1/drl_moudle/train_data/run_01/output.log`

Mục tiêu của phần này là phân tích quá trình huấn luyện PPO theo các khía cạnh: hội tụ, ổn định, khả năng khám phá, ảnh hưởng của action masking, ảnh hưởng của privileged learning và hành vi policy sau huấn luyện.

## 7.2.1. Thiết lập môi trường huấn luyện

### Mô tả môi trường huấn luyện

Trong quá trình huấn luyện, tác nhân được huấn luyện trong môi trường mô phỏng `CrowdSim-v0`. Môi trường mô phỏng bài toán điều hướng robot di động vi sai trong không gian có người đi bộ, vật cản tĩnh và tường biên. Robot không trực tiếp học lệnh vận tốc bánh xe, mà học cách chọn một mục tiêu cục bộ rời rạc. Mục tiêu cục bộ này sau đó được đưa vào bộ lập kế hoạch cục bộ/MPC để sinh lệnh điều khiển cho robot.

Các thông số chính của môi trường huấn luyện được cấu hình trong `configs/config.py` và `train_ppo.py`.

| Thành phần | Giá trị trong code | Ý nghĩa |
|---|---:|---|
| Environment | `CrowdSim-v0` | Môi trường mô phỏng đám đông cho huấn luyện RL |
| Scenario train/test | `circle_crossing` | Người đi bộ di chuyển qua khu vực robot theo cấu hình giao cắt |
| Kích thước vùng mô phỏng | `square_width = 10 m` | Không gian chính theo phương ngang |
| Bán kính bố trí người đi bộ | `circle_radius = 4 m` | Bán kính sinh vị trí/người đi bộ trong kịch bản circle crossing |
| Số người đi bộ mặc định | `human_num = 5` | Trong `BaseEnvConfig`; curriculum có thể thay đổi theo phase |
| Số vật cản tĩnh mặc định | `obstacle_num = 3` | Vật cản tròn được sinh ngẫu nhiên |
| Số tường | `wall_num = 4` | Tường/cạnh môi trường và vật cản dạng polygon |
| Thời gian tối đa mỗi episode | `time_limit = 30 s` | Episode kết thúc nếu quá thời gian |
| Bước thời gian mô phỏng | `time_step = 0.25 s` | Mỗi bước `env.step()` tương ứng 0.25 s |
| Bán kính robot | `0.3 m` | Dùng cho kiểm tra va chạm/an toàn |
| Bán kính người đi bộ | `0.3 m` | Dùng cho kiểm tra va chạm/an toàn |
| Vận tốc mong muốn | `v_pref = 1.0 m/s` | Cho robot và người đi bộ |
| Chính sách người đi bộ | `orca` | Người đi bộ được điều khiển bởi ORCA/centralized ORCA |
| Cơ chế người đi bộ | `nonstop_human = True` | Người đi bộ được sinh goal mới sau khi đến đích |
| Sensor | `coordinates` | Trạng thái được biểu diễn bằng tọa độ, không dùng ảnh |
| Random seed huấn luyện | `3` | Trong `train_ppo.py` |

Code tương ứng:

- Cấu hình môi trường: `HALO_1/drl_moudle/configs/config.py`
- Tạo môi trường: `HALO_1/drl_moudle/train_ppo.py:66`
- Gán phase huấn luyện: `HALO_1/drl_moudle/train_ppo.py:68`
- Cấu hình số người/vật cản theo phase: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:167-183`
- Sinh người đi bộ: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:220-420`
- Sinh vật cản tĩnh: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:421-466`

Trong log huấn luyện hiện tại, môi trường được ghi nhận:

```text
human number: 5
Training simulation: circle_crossing, test simulation: circle_crossing
Square width: 10, circle width: 4
Using device: cuda:0
```

### Không gian hành động local goal

Policy PPO không chọn trực tiếp gia tốc bánh xe. Thay vào đó, Actor chọn một local goal trong lưới hành động rời rạc. Với `action_dim = 9`, mỗi trục của vùng local goal được chia thành 9 giá trị. Tổng số hành động là:

```tex
|\mathcal{A}| = N^2 = 9^2 = 81
```

Vùng local goal trong hệ tọa độ robot:

```tex
x_g, y_g \in [-2.25, 2.25]\ \text{m}
```

Mỗi action index `a` được ánh xạ sang tọa độ local goal:

```tex
i = \left\lfloor \frac{a}{N} \right\rfloor,\quad
j = a \bmod N
```

```tex
g(a) = [x_i, y_j]
```

Code tương ứng:

- Thiết lập action space: `HALO_1/drl_moudle/train_ppo.py:83-90`
- Mapping action sang local goal: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:676-685`

### PPO hyperparameters

Các siêu tham số PPO được thiết lập trong `train_ppo.py` và `MpcPPO.__init__`.

| Tham số | Giá trị | Code |
|---|---:|---|
| Tổng số bước huấn luyện | `5e6` | `model.learn(int(5e6))` |
| Policy | `GraphPolicy` | Actor-Critic dùng đặc trưng đồ thị |
| Learning rate | tuyến tính `2.5e-4 -> 1.0e-4` | `get_linear_fn(2.5e-4, 1.0e-4, 0.5)` |
| Rollout steps | `2048` | `n_steps=2048` |
| Batch size | `64` | `batch_size=64` |
| Epoch/update | `10` | default `n_epochs=10` |
| Discount factor | `gamma = 0.99` | default trong `MpcPPO` |
| GAE lambda | `0.95` | default trong `MpcPPO` |
| Clip range | `0.2` | default trong `MpcPPO` |
| Entropy coefficient | `0.001` | `ent_coef=0.001` |
| Value loss coefficient | `0.5` | default `vf_coef=0.5` |
| Max grad norm | `0.5` | default `max_grad_norm=0.5` |
| Advantage normalization | `True` | default |
| Target KL | `None` | không bật early stop theo KL |
| Eval frequency | mỗi `500` episode | `EvalCallback(eval_freq=500)` |
| Số episode đánh giá | `100` | `n_eval_episodes=100` |

Code tương ứng:

- Khởi tạo model PPO: `HALO_1/drl_moudle/train_ppo.py:98-111`
- Default PPO: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:69-83`
- PPO update: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:490-605`
- Log TensorBoard: `HALO_1/drl_moudle/algorithms/graph_ppo.py:412-421`

Lưu ý về tên tag TensorBoard: trong code hiện tại reward trung bình được log là:

```text
rollout/ep_ret_mean
```

không phải `rollout/ep_rew_mean`. Nếu muốn dùng đúng tên phổ biến của Stable-Baselines3 là `rollout/ep_rew_mean`, cần đổi tag trong `graph_ppo.py`. Nếu không đổi code, trong báo cáo nên ghi theo tag hiện có là `rollout/ep_ret_mean`.

### Tham số reward

Reward được shaping để phản ánh nhiều mục tiêu điều hướng: đi tới goal, tránh va chạm, giữ khoảng cách an toàn, giảm rủi ro velocity obstacle và ưu tiên hướng robot về phía goal.

Các trọng số được truyền từ command line trong `train_ppo.py`.

| Reward term | Ký hiệu báo cáo | Giá trị hiện tại | Code |
|---|---:|---:|---|
| Thưởng đến đích | `r_arrival` | `0.25` | `--re_arrival` |
| Phạt va chạm | `r_collision` | `-0.25` | `--re_collision` |
| Trọng số tiến về goal | `w_goal` | `0.1` | `--goal_weight` |
| Trọng số an toàn | `w_safe` | `0.5` | `--safe_weight` |
| Trọng số heading | `w_theta` | `0.01` | `--re_theta` |
| Trọng số RVO | `w_rvo` | `0.01` | `--re_rvo` |
| Time penalty | `r_time` | `-0.0125` | hard-code trong `reward_cal` |
| Discomfort distance | `d_safe` | `0.2 m` | `reward.discomfort_dist` |
| Privileged trajectory length | `K_PL` | `4` | `--PL_traj_length` |
| Privileged discount | `gamma_PL` | `0.9` | `--PL_traj_gamma` |

Reward một bước có thể viết:

```tex
r_t =
100\left(
r_{arrival}
+ w_g r_{goal}
+ w_\theta r_{\theta}
+ r_{time}
+ r_{collision}
+ w_{safe} r_{safe}
+ w_{rvo} r_{rvo}
\right)
+ r_{traj}
```

Trong đó:

```tex
r_{goal}
=
\|p_t - p_g\|_2 - \|p_{t+1} - p_g\|_2
```

`r_goal` dương khi robot tiến gần goal hơn.

```tex
r_{\theta}
=
\frac{\cos(\theta_t-\theta_g)-1}{\|p_t-p_g\|_2+5}
```

`r_theta` khuyến khích hướng robot quay về phía goal.

```tex
r_{safe}
=
\sum_i \mathbb{I}(d_i < d_{safe})(d_i-d_{safe})
```

`r_safe` là penalty khi robot đi quá gần người/vật cản/tường.

Code tương ứng:

- Reward chính: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:880-925`
- RVO reward: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1136-1165`
- Scale reward và cộng trajectory reward: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1431-1435`

### Tham số planner/MPC liên quan đến privileged learning

Trong phần huấn luyện AI, planner/MPC đóng vai trò sinh điều khiển thấp tầng và trajectory dự đoán cho robot. Trong code PPO hiện tại, số bước dự báo của planner được lưu bằng:

```text
MPC_NP = 10
```

Trong action masking, code dùng:

```tex
d_{max} = (MPC\_NP - 1)\Delta t
```

với `time_step = 0.25`, suy ra:

```tex
d_{max} = (10 - 1)\times 0.25 = 2.25\ \text{m}
```

Giá trị này khớp với `action_range = 2.25`, tức là vùng local goal được thiết kế phù hợp với tầm dự báo ngắn hạn của planner.

Code tương ứng:

- `MPC_NP`: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:107`
- `action_range`: `HALO_1/drl_moudle/train_ppo.py:200`
- Điều kiện mask theo tầm planner: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1190-1193`

### Hình nên đưa vào mục 7.2.1

Nên có một hình mô tả môi trường huấn luyện gồm robot, goal, người đi bộ, vật cản, tường và các local goal ứng viên.

Caption gợi ý:

```latex
\caption{Môi trường huấn luyện PPO trong kịch bản điều hướng đám đông. Robot chọn một mục tiêu cục bộ từ lưới hành động rời rạc, sau đó planner cục bộ sinh lệnh điều khiển để robot di chuyển trong môi trường có người đi bộ và vật cản.}
```

## 7.2.2. Phân tích hội tụ của PPO

### Các đại lượng cần vẽ

Phần này là phần quan trọng nhất khi trình bày quá trình huấn luyện. Nên xuất từ TensorBoard các đường sau:

| Đại lượng | Tag trong code hiện tại | Ý nghĩa |
|---|---|---|
| Mean episode return | `rollout/ep_ret_mean` | Reward trung bình theo episode |
| Mean episode length | `rollout/ep_len_mean` | Độ dài episode trung bình |
| Value loss | `train/value_loss` | Sai số học hàm giá trị của Critic |
| Entropy loss | `train/entropy_loss` | Mức độ ngẫu nhiên/khám phá của policy |
| Evaluation average return | `eval/ave_return` | Return trung bình trên 100 episode eval |
| Evaluation success rate | `eval/success_rate` | Tỷ lệ thành công khi đánh giá |

Nếu muốn giống chuẩn SB3, có thể đổi `rollout/ep_ret_mean` thành `rollout/ep_rew_mean`; tuy nhiên theo code hiện tại nên dùng đúng `ep_ret_mean`.

### Figure bắt buộc

Nên trình bày một hình gồm 4 subfigure:

```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}{0.48\linewidth}
        \includegraphics[width=\linewidth]{ppo_ep_ret_mean.png}
        \caption{Mean episode return}
    \end{subfigure}
    \begin{subfigure}{0.48\linewidth}
        \includegraphics[width=\linewidth]{ppo_ep_len_mean.png}
        \caption{Mean episode length}
    \end{subfigure}
    \begin{subfigure}{0.48\linewidth}
        \includegraphics[width=\linewidth]{ppo_value_loss.png}
        \caption{Value loss}
    \end{subfigure}
    \begin{subfigure}{0.48\linewidth}
        \includegraphics[width=\linewidth]{ppo_entropy_loss.png}
        \caption{Entropy loss}
    \end{subfigure}
    \caption{Các chỉ số hội tụ chính trong quá trình huấn luyện PPO.}
    \label{fig:ppo_convergence}
\end{figure}
```

### Cách phân tích reward/return

Đoạn có thể đưa vào báo cáo:

> Đường `rollout/ep_ret_mean` phản ánh chất lượng chính sách trong quá trình tương tác với môi trường. Ở giai đoạn đầu, return trung bình thường thấp do policy còn gần ngẫu nhiên, robot thường chọn các local goal không hiệu quả, dễ va chạm hoặc không đến được đích trong thời gian giới hạn. Khi số bước huấn luyện tăng lên, return trung bình có xu hướng tăng, cho thấy policy học được cách chọn các local goal có khả năng đưa robot tiến gần goal hơn, đồng thời giảm số lần đi vào vùng không an toàn. Khi đường return bắt đầu dao động quanh một mức ổn định, có thể xem policy đã tiến tới trạng thái hội tụ tương đối.

Điểm cần nhấn mạnh: reward trong RL có thể dao động mạnh vì dữ liệu được thu thập on-policy và môi trường có người đi bộ động. Do đó, không nên kết luận chỉ dựa trên một điểm loss đơn lẻ, mà cần phân tích xu hướng trung bình và liên hệ với hành vi robot.

### Cách phân tích episode length

Đoạn có thể đưa vào báo cáo:

> Đường `rollout/ep_len_mean` thể hiện số bước trung bình của mỗi episode. Ở giai đoạn đầu, episode length thường lớn do robot chưa biết cách đến goal, nhiều episode kết thúc bởi timeout. Khi policy tốt hơn, robot hoàn thành nhiệm vụ nhanh hơn nên episode length có xu hướng giảm. Tuy nhiên, nếu episode length giảm quá mạnh kèm theo collision rate tăng, điều đó không phải là dấu hiệu tốt, vì episode có thể kết thúc sớm do va chạm. Vì vậy, episode length cần được phân tích cùng reward, success rate và collision rate.

### Cách phân tích value loss

Đoạn có thể đưa vào báo cáo:

> `train/value_loss` biểu diễn sai số giữa giá trị mà Critic dự đoán và target return được tính từ GAE. Khi huấn luyện ổn định, value loss có xu hướng giảm hoặc dao động trong một dải hữu hạn, cho thấy Critic học được cấu trúc phần thưởng dài hạn của bài toán. Tuy nhiên, khác với supervised learning, loss của PPO không nhất thiết giảm đơn điệu. Do policy liên tục thay đổi, phân phối dữ liệu rollout cũng thay đổi theo, vì vậy dao động của value loss là hiện tượng bình thường trong huấn luyện RL.

### Cách phân tích entropy loss

Đoạn có thể đưa vào báo cáo:

> Entropy phản ánh mức độ ngẫu nhiên của phân phối hành động. Ở giai đoạn đầu, entropy cao cho thấy Actor còn khám phá nhiều local goal khác nhau. Khi quá trình huấn luyện tiến triển, entropy giảm dần vì policy trở nên tự tin hơn trong việc chọn các local goal có xác suất thành công cao. Trong bài toán điều hướng đám đông, entropy không nên giảm về gần 0 quá sớm, vì điều đó có thể làm policy mất khả năng thích nghi với các cấu hình người đi bộ khác nhau.

Trong code, entropy loss được tính là:

```tex
L_{entropy} = -\mathbb{E}[S(\pi_\theta(\cdot|s))]
```

và tổng loss:

```tex
L =
L_{policy}
+ c_v L_{value}
+ c_e L_{entropy}
```

Code tương ứng:

- PPO clipped objective: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:528-534`
- Value loss: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:549-550`
- Entropy loss: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:553-558`
- Tổng loss: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:562`

### Kết quả đánh giá từ log hiện tại

Trong `train_data/run_01/output.log`, callback đánh giá policy sau mỗi 500 episode với 100 episode đánh giá. Một số kết quả quan trọng:

| Thời điểm trong log | Success rate | Collision rate | Nav time |
|---|---:|---:|---:|
| Đánh giá sớm | `77%` | `7%` | `14.5682 s` |
| Giai đoạn tốt ban đầu | `97%` | `2%` | `10.2139 s` |
| Giai đoạn tốt nhất trong log | `100%` | `0%` | `9.0500 s` |
| Một lần eval cuối log | `82%` | `18%` | `13.1402 s` |

Đoạn phân tích nên viết cẩn thận:

> Kết quả đánh giá cho thấy policy có thể đạt tỷ lệ thành công rất cao trong một số giai đoạn, ví dụ đạt `100%` success rate và `0%` collision rate trong một lần đánh giá. Tuy nhiên, các lần đánh giá sau vẫn có dao động do môi trường huấn luyện có tính ngẫu nhiên, người đi bộ động và cấu hình vật cản thay đổi theo seed. Vì vậy, thay vì chỉ báo cáo một checkpoint đơn lẻ, cần phân tích xu hướng trung bình hoặc chọn checkpoint tốt nhất (`best_model`) để đánh giá cuối cùng trên cùng một tập test seed.

## 7.2.3. Ảnh hưởng của Action Masking

### Vai trò của action masking trong hệ thống

Action masking là một đóng góp quan trọng của hệ thống. Vì Actor chọn local goal từ lưới 81 hành động, nhiều local goal có thể không khả thi: nằm trong vật cản, quá gần tường, nằm ngoài vùng làm việc, hoặc vượt quá tầm dự báo của planner. Nếu để policy khám phá tự do toàn bộ 81 hành động, robot sẽ tốn nhiều mẫu vào các hành động nguy hiểm, làm tăng collision rate và làm chậm hội tụ.

Gọi `m(s,a)` là mask của action `a` tại trạng thái `s`:

```tex
m(s,a)=
\begin{cases}
1, & \text{nếu local goal hợp lệ}\\
0, & \text{nếu local goal không hợp lệ}
\end{cases}
```

Policy sau mask:

```tex
\pi_\theta^m(a|s)
=
\frac{
m(s,a)\pi_\theta(a|s)
}{
\sum_{a'}m(s,a')\pi_\theta(a'|s)
}
```

Trong code, mask được áp dụng ở mức logits:

```tex
\ell_a^m =
\begin{cases}
-10^8, & m(s,a)=0\\
10^8, & m(s,a)=2\\
\ell_a, & m(s,a)=1
\end{cases}
```

Giá trị `2` được dùng để ưu tiên action gần goal khi robot đã ở gần đích.

Code tương ứng:

- Tạo action mask: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1167-1240`
- Lấy action mask khi rollout: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:407-410`
- Lưu mask vào rollout buffer: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:470-474`
- Dùng mask khi PPO update: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:516-520`

### Các điều kiện loại bỏ action

Action bị mask nếu local goal:

- Vượt quá tầm di chuyển hợp lý của planner:

```tex
x_g^2+y_g^2 > d_{max}^2
```

- Quá gần vật cản tròn:

```tex
d(g, o_i) < r_i + r_{robot} + 0.1
```

- Quá gần tường:

```tex
d(g, wall_i) < r_{robot} + 0.1
```

- Nằm trong vật cản polygon.
- Nằm ngoài biên môi trường/corridor.
- Khi robot gần goal, action gần goal nhất được gán mask `2.0` để ưu tiên.

### Hình nên đưa vào

Nên có hai hình:

1. So sánh reward:

```text
PPO without masking
PPO + action masking
```

2. So sánh collision rate:

| Method | Collision rate |
|---|---:|
| PPO only | điền từ thí nghiệm ablation |
| PPO + Action Masking | điền từ thí nghiệm ablation |

Caption gợi ý:

```latex
\caption{Ảnh hưởng của action masking đến quá trình huấn luyện PPO. Action masking loại bỏ các local goal không khả thi trước khi Actor lấy mẫu, nhờ đó giảm va chạm trong quá trình exploration và cải thiện tốc độ hội tụ.}
```

### Đoạn phân tích đưa vào báo cáo

> Khi không sử dụng action masking, Actor phải khám phá toàn bộ không gian 81 local goals, bao gồm cả những mục tiêu nằm trong vật cản hoặc nằm ngoài vùng khả thi của planner. Điều này làm tăng xác suất robot sinh trajectory không an toàn trong giai đoạn đầu huấn luyện, khiến reward dao động mạnh và collision rate cao. Khi bật action masking, các hành động không khả thi bị loại bỏ trước khi phân phối hành động được lấy mẫu. Nhờ đó, exploration của PPO được tập trung vào tập local goal hợp lệ, giảm số lần va chạm và giúp policy hội tụ nhanh hơn.

> Action masking không thay thế policy học được, mà đóng vai trò như một ràng buộc an toàn ở mức hành động. Actor vẫn phải học cách phân biệt local goal nào tốt hơn trong tập hành động hợp lệ, nhưng không còn phải lãng phí mẫu vào những hành động chắc chắn nguy hiểm. Đây là lý do action masking đặc biệt phù hợp với bài toán điều hướng robot trong môi trường có vật cản động và tĩnh.

## 7.2.4. Ảnh hưởng của Privileged Learning

### Vai trò của privileged learning

Trong code, privileged learning được bật bằng:

```text
use_PL = True
PL_traj_length = 4
PL_traj_gamma = 0.9
```

Khác với reward một bước chỉ đánh giá trạng thái ngay sau khi robot thực hiện action, privileged learning dùng thêm thông tin trajectory dự đoán từ planner để đánh giá các trạng thái tương lai gần. Đây là thông tin “privileged” vì nó dùng trong huấn luyện để cung cấp tín hiệu học giàu hơn, nhưng không nhất thiết phải là đầu ra trực tiếp của policy.

Reward trajectory:

```tex
r_{traj}
=
\sum_{i=1}^{K_{PL}-1}
\gamma_{PL}^{i} r_{eval}(s_{t+i})
```

Với cấu hình hiện tại:

```tex
K_{PL}=4,\quad \gamma_{PL}=0.9
```

Nên:

```tex
r_{traj}
=
0.9 r_{eval}(s_{t+1})
+ 0.9^2 r_{eval}(s_{t+2})
+ 0.9^3 r_{eval}(s_{t+3})
```

Trong đó `r_eval` gồm các thành phần đánh giá heading, collision/safety và RVO risk trên trạng thái dự đoán.

Code tương ứng:

- Bật/tắt privileged learning: `HALO_1/drl_moudle/train_ppo.py:91-94`
- Tham số `PL_traj_length`, `PL_traj_gamma`: `HALO_1/drl_moudle/train_ppo.py:204-206`
- Tính trajectory reward: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1360-1367`
- Hàm đánh giá trạng thái dự đoán: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:943-1073`

### Hình nên đưa vào

Nên có hai hình:

1. Reward curve comparison:

```text
PPO without trajectory reward
PPO + trajectory reward
```

2. Trajectory visualization:

```text
Không privileged learning: trajectory zig-zag, nhiều dao động, dễ áp sát người/vật cản.
Có privileged learning: trajectory mượt hơn, tránh vùng rủi ro sớm hơn.
```

Caption gợi ý:

```latex
\caption{Ảnh hưởng của privileged learning đến chất lượng policy. Trajectory reward cung cấp thêm đánh giá ngắn hạn từ quỹ đạo dự đoán, giúp policy học được các local goal tạo ra chuyển động an toàn và ổn định hơn.}
```

### Đoạn phân tích đưa vào báo cáo

> Privileged learning giúp policy không chỉ tối ưu phần thưởng tức thời, mà còn nhận thêm tín hiệu từ chất lượng quỹ đạo tương lai gần do planner sinh ra. Điều này đặc biệt quan trọng trong điều hướng robot, vì một local goal có thể trông hợp lệ tại thời điểm hiện tại nhưng lại dẫn tới trajectory xấu sau vài bước, ví dụ đi quá gần người đi bộ hoặc tạo chuyển động zig-zag. Bằng cách cộng thêm trajectory reward, policy được khuyến khích chọn các local goal có chất lượng dự báo tốt hơn.

> Khi so sánh với trường hợp không dùng trajectory reward, kỳ vọng đường reward của mô hình có privileged learning sẽ tăng nhanh hơn và ít dao động hơn. Về hành vi, robot có xu hướng chọn local goal tạo ra quỹ đạo mượt, tránh vùng có mật độ người đi bộ cao và giảm oscillation khi đi qua các cấu hình hẹp.

## 7.2.5. Phân tích hành vi policy

### Mục tiêu phân tích

Sau khi PPO hội tụ, cần chứng minh policy không chỉ có reward cao mà còn học được hành vi điều hướng hợp lý. Vì action của Actor là local goal rời rạc, có thể trực quan hóa xác suất chọn hành động dưới dạng bản đồ xác suất trên lưới `9 x 9`.

Actor sinh phân phối:

```tex
\pi_\theta(a|s),\quad a \in \{0,1,\dots,80\}
```

Sau khi áp dụng action masking:

```tex
\pi_\theta^m(a|s)
```

Policy tốt nên có các đặc điểm:

- Xác suất cao ở các local goal nằm trong vùng thông thoáng.
- Xác suất thấp hoặc bằng 0 ở các vùng bị mask.
- Khi có người đi bộ phía trước, policy chuyển xác suất sang các local goal lệch trái/phải để tránh va chạm.
- Khi gần goal, action gần goal được ưu tiên rõ ràng.
- Không chọn liên tục các local goal gây zig-zag hoặc quay tại chỗ.

Code hỗ trợ phân tích:

- Lưu action probability khi predict: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:657-659`
- Lấy action probability: `HALO_1/drl_moudle/algorithms/mpc_ppo.py:672-673`
- Visualize action mask: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:1761-1771`
- Mapping action sang local goal: `HALO_1/drl_moudle/crowd_sim/envs/crowd_sim.py:2028-2036`

### Hình nên đưa vào

Nên có ít nhất một hình:

1. Local goal probability map:

```text
Mỗi ô là một local goal.
Màu càng đậm nghĩa là xác suất Actor chọn càng cao.
Ô đỏ: masked action.
Ô xanh: valid action.
Ô nổi bật: selected action.
```

2. Visualization trên môi trường:

```text
Robot, người đi bộ, vật cản, goal thật, local goal được chọn, vùng bị mask.
```

Caption gợi ý:

```latex
\caption{Trực quan hóa phân phối hành động của policy sau huấn luyện. Policy ưu tiên các local goal nằm trong vùng thông thoáng và giảm xác suất đối với các hướng có nguy cơ va chạm cao.}
```

### Đoạn phân tích đưa vào báo cáo

> Kết quả trực quan hóa local goal cho thấy policy đã học được cách phân bố xác suất hành động theo cấu trúc môi trường. Trong các trạng thái không có vật cản phía trước, xác suất thường tập trung vào các local goal hướng về goal toàn cục. Khi xuất hiện người đi bộ hoặc vật cản trong hướng di chuyển, các action tương ứng bị mask hoặc có xác suất thấp, trong khi xác suất được dịch chuyển sang các local goal an toàn hơn ở hai phía. Điều này chứng minh policy không chỉ tối đa hóa reward một cách trừu tượng, mà thực sự học được chiến lược chọn mục tiêu cục bộ phù hợp với ngữ cảnh điều hướng.

## 7.2.6. Tổng kết quá trình huấn luyện

Đoạn tổng kết có thể đưa vào báo cáo:

> Quá trình huấn luyện PPO cho thấy Actor-Critic có khả năng học chính sách chọn local goal cho robot trong môi trường đám đông động. Các chỉ số TensorBoard như mean episode return, episode length, value loss và entropy loss cho phép đánh giá đồng thời chất lượng hành vi, mức độ ổn định của Critic và quá trình chuyển đổi từ exploration sang exploitation. Kết quả đánh giá trong log cho thấy policy có thể đạt tỷ lệ thành công cao, trong một số checkpoint đạt tới `100%` success rate và `0%` collision rate trên tập đánh giá 100 episode.

> Action masking đóng vai trò quan trọng trong việc giới hạn không gian hành động vào các local goal khả thi, nhờ đó giảm va chạm trong quá trình exploration và cải thiện độ ổn định huấn luyện. Privileged learning bổ sung reward từ quỹ đạo dự đoán, giúp policy học được chất lượng điều hướng ngắn hạn thay vì chỉ tối ưu reward tức thời. Hai cơ chế này kết hợp với PPO giúp tạo ra policy có khả năng triển khai cho bài toán điều hướng robot trong môi trường có người đi bộ và vật cản động.

## Các hình quan trọng cần chuẩn bị cho mục 7.2

### Bắt buộc

| Hình | Nội dung | Nguồn dữ liệu |
|---|---|---|
| Figure 7.2a | `rollout/ep_ret_mean` hoặc `rollout/ep_rew_mean` nếu đổi tag | TensorBoard |
| Figure 7.2b | `rollout/ep_len_mean` | TensorBoard |
| Figure 7.2c | `train/value_loss` | TensorBoard |
| Figure 7.2d | `train/entropy_loss` | TensorBoard |
| Figure 7.3 | Reward comparison: masking vs no masking | Chạy ablation |
| Figure 7.4 | Collision rate: masking vs no masking | Chạy ablation/eval |

### Rất nên có

| Hình | Nội dung | Mục đích |
|---|---|---|
| Figure 7.5 | Reward curve: PPO vs PPO + trajectory reward | Chứng minh privileged learning |
| Figure 7.6 | Trajectory visualization | So sánh quỹ đạo zig-zag và quỹ đạo mượt |
| Figure 7.7 | Local goal probability map | Giải thích hành vi policy |
| Figure 7.8 | Action mask visualization | Chứng minh mask loại bỏ local goal nguy hiểm |

### Không nên đưa quá nhiều

Không nên đưa quá nhiều hình phụ như `fps`, `n_updates`, `clip_fraction`, `approx_kl`, `explained_variance` nếu không phân tích sâu. Các chỉ số này có thể giữ trong phụ lục hoặc dùng khi hội đồng hỏi thêm. Phần chính nên tập trung vào reward, episode length, value loss, entropy, success rate, collision rate và trực quan hóa hành vi robot.

## Bảng ablation nên chuẩn bị

Nếu có thời gian chạy thêm thí nghiệm, nên chuẩn bị bảng sau:

| Method | Action Masking | Privileged Learning | Success rate | Collision rate | Avg nav time | Avg return |
|---|---:|---:|---:|---:|---:|---:|
| PPO baseline | Không | Không | điền kết quả | điền kết quả | điền kết quả | điền kết quả |
| PPO + Mask | Có | Không | điền kết quả | điền kết quả | điền kết quả | điền kết quả |
| PPO + PL | Không | Có | điền kết quả | điền kết quả | điền kết quả | điền kết quả |
| PPO + Mask + PL | Có | Có | điền kết quả | điền kết quả | điền kết quả | điền kết quả |

Với code hiện tại, cấu hình mặc định đang là:

```text
use_AM = True
use_PL = True
```

Do đó, để có ablation đúng, cần chạy thêm các cấu hình:

```bash
python train_ppo.py --use_AM False --use_PL False --output_dir train_data/ppo_baseline
python train_ppo.py --use_AM True  --use_PL False --output_dir train_data/ppo_mask_only
python train_ppo.py --use_AM False --use_PL True  --output_dir train_data/ppo_pl_only
python train_ppo.py --use_AM True  --use_PL True  --output_dir train_data/ppo_mask_pl
```

Lưu ý kỹ thuật: trong `argparse`, `type=bool` có thể gây hiểu nhầm khi truyền `"False"` từ command line vì chuỗi khác rỗng có thể được hiểu là `True` tùy cách parse. Nếu cần chạy ablation nghiêm túc, nên đổi các flag `--use_AM`, `--use_PL` sang `action='store_true'`/`store_false` hoặc dùng parser riêng cho boolean.

## Checklist khi viết báo cáo

- Đưa bảng môi trường huấn luyện.
- Đưa bảng PPO hyperparameters.
- Đưa công thức reward thực tế đang dùng trong code.
- Đưa 4 hình hội tụ chính: return, episode length, value loss, entropy loss.
- Phân tích reward gắn với hành vi robot, không chỉ nói “loss giảm”.
- Đưa so sánh action masking vs no masking.
- Đưa so sánh privileged learning vs no privileged learning.
- Đưa visualization local goal/action mask nếu có.
- Kết luận bằng success rate, collision rate và khả năng triển khai policy.

