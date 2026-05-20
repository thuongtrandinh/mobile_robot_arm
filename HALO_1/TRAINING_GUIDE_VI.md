# Hướng Dẫn Train/Test HALO PPO + Hiển Thị Lưới Action Mask

## 1. Tổng quan workspace

Workspace của bạn chia 2 phần chính:

- **HALO/**: train/eval mô hình PPO + MPC trong mô phỏng CrowdSim.
- **src/**: deploy ROS2 Humble cho robot thực (navigation, controller, bringup, ...).

Tóm tắt:
- Muốn **train model** => làm trong `HALO/drl_moudle`.
- Muốn **deploy ROS2** => làm trong `src/`.

---

## 2. Bắt buộc trước khi chạy

## 2.1 Activate conda environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc_rl
```

> Luôn activate `mpc_rl` trước để tránh lỗi thiếu package / sai Python interpreter.

## 2.2 Thiết lập chống lỗi GDAL/LIBTIFF (bắt buộc nếu gặp ImportError)

Nếu bạn gặp lỗi kiểu:

```bash
ImportError: /lib/libgdal.so.30: undefined symbol: TIFFReadRGBATileExt, version LIBTIFF_4.0
```

thì thêm dòng này trước khi chạy train/eval:

```bash
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtiff.so.5
```

Lưu ý: đây là lỗi xung đột thư viện native giữa conda và thư viện hệ thống (GDAL/TIFF), không phải lỗi code Python.

## 2.3 Build MPC Python binding (ocp_planner_py)

Chọn **một** trong hai cách:

### A) Standalone build (khuyên dùng khi training)

```bash
cd ~/LVTN/amr_ws/HALO
./build_standalone.sh
```

### B) ROS2 Humble build

```bash
cd ~/LVTN/amr_ws/HALO
./build_cp_ros2.sh
```

Kiểm tra binding:

```bash
cd ~/LVTN/amr_ws/HALO/drl_moudle
python -c "import ocp_planner_py; print('OK')"
```

---

## 3. Train model PPO

```bash
cd ~/LVTN/amr_ws/HALO/drl_moudle
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc_rl
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtiff.so.5

conda activate mpc_rl
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtiff.so.5

python train_ppo.py \
  --config configs/mpc_rl.py \
  --output_dir train_data/run2 \
  --total_timesteps 1000000 \
  --eval_freq 500 \
  --n_eval_episodes 20 \
  --action_dim 9 \
  --action_range 2.25 \
  --use_AM True \
  --use_PL True
```

### Ý nghĩa tham số chính

- `--output_dir`: nơi lưu log + model tốt nhất (`best_model`).
- `--total_timesteps`: số bước train.
- `--eval_freq`: tần suất eval định kỳ.
- `--n_eval_episodes`: số episode mỗi lần eval.
- `--action_dim`: lưới action, ví dụ `9x9 = 81 action`.
- `--action_range`: phạm vi tọa độ local-goal.
- `--use_AM`: bật Action Mask.
- `--use_PL`: bật Privileged Learning.

---

## 4. Resume train

```bash
cd ~/LVTN/amr_ws/HALO/drl_moudle
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc_rl
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtiff.so.5

python train_ppo.py \
  --config configs/mpc_rl.py \
  --output_dir train_data/run_06 \
  --total_timesteps 5000000 \
  --eval_freq 300 \
  --n_eval_episodes 50 \
  --action_dim 9 \
  --action_range 2.25 \
  --use_AM True \
  --use_PL True \
  --n_steps 2048 \
  --batch_size 128 \
  --n_epochs 10 \
  --resume \
  --resume_from auto
```

Gợi ý quan trọng:
- `n_eval_episodes=20` thường nhiễu khá mạnh (dao động lớn giữa các lần eval).
- Để đường eval ổn định hơn nên dùng `50` (hoặc `100` khi làm báo cáo), đổi lại sẽ chậm hơn.
- `eval_freq=300` giúp giảm tải eval quá dày và ổn định train hơn so với `200`.

Điều kiện: trong `train_data/run1` có `best_model` hoặc `best_model.zip`.

Tuỳ chọn mới:

- `--resume_from auto|last|best`
  - `auto`: thử `last_model` trước, lỗi thì fallback sang `best_model`.
  - `last`: chỉ resume từ `last_model`.
  - `best`: chỉ resume từ `best_model`.
- `--last_save_freq N`: auto-save `last_model` mỗi `N` episode (mặc định `50`).

---

## 5. Các mode để xem kết quả

## Mode 1: Livestream metric bằng TensorBoard

```bash
tensorboard --logdir ~/LVTN/amr_ws/HALO/drl_moudle/train_data/run1
```

Mở: `http://localhost:6006`

Bạn sẽ thấy các nhóm chỉ số `train/*`, `eval/*`.

## Mode 2: Eval nhiều episode lấy thống kê

```bash
python eval_ppo.py \
  --config configs/mpc_rl.py \
  --model_dir train_data/run1 \
  --n_eval_episodes 500 \
  --use_action_mask
```

Output: success rate, collision rate.

## Mode 2.5: Test trực tiếp với best_model.zip

Điều kiện để chạy:
- Trong thư mục `model_dir` phải có `best_model` hoặc `best_model.zip`.
- File này chỉ xuất hiện sau khi callback eval lưu model tốt hơn trước đó.

Kiểm tra nhanh:

```bash
ls -lh train_data/run1/best_model*
```

Nếu chưa có `best_model`:
- Tiếp tục train thêm (nhất là khi mới bắt đầu hoặc eval chưa chạy đủ).
- Giảm `--eval_freq` để callback đánh giá thường xuyên hơn.

Test `best_model` lấy thống kê:

```bash
python eval_ppo.py \
  --config configs/mpc_rl.py \
  --model_dir train_data/run1 \
  --n_eval_episodes 200 \
  --use_action_mask
```

Test `best_model` có render:

```bash
python eval_ppo.py \
  --config configs/mpc_rl.py \
  --model_dir train_data/run1 \
  --n_eval_episodes 20 \
  --visualize \
  --use_action_mask
```

Test `best_model` để soi frame:

```bash
python eval_ppo.py \
  --config configs/mpc_rl.py \
  --model_dir train_data/run1 \
  --output_dir eval_data/run_best \
  --output_image_dir images/run_best_grid \
  --n_eval_episodes 1 \
  --use_action_mask
```

Ghi chú Gazebo:
- Các lệnh eval ở trên chạy trong môi trường HALO (CrowdSim), không tự động điều khiển robot ROS2/Gazebo.
- Muốn chạy `best_model` trong Gazebo cần thêm node bridge suy luận PPO -> publish lệnh điều khiển ROS2.

## Mode 3: Render live khi eval

```bash
python eval_ppo.py \
  --config configs/mpc_rl.py \
  --model_dir train_data/run1 \
  --n_eval_episodes 20 \
  --visualize \
  --use_action_mask
```

## Mode 4: Xuất frame ảnh có lưới (để soi kỹ)

```bash
python eval_ppo.py \
  --model_dir /home/thuong/LVTN/amr_ws/HALO/drl_moudle/data/model1 \
  --output_dir eval_data/run2 \
  --output_image_dir images/run1_grid \
  --n_eval_episodes 1 \
  --action_dim 9 \
  --action_range 2.25 \
  --use_action_mask
```

Kết quả frame nằm ở `images/run1_grid`.

---

## 6. Cách để chắc chắn hiển thị lưới action mask

- Dùng `--use_action_mask` (không tắt bằng `--no-use_action_mask`).
- Đặt `--n_eval_episodes 1` để debug 1 episode rõ ràng.
- Dùng `--output_image_dir` để lưu frame.
- Điều chỉnh độ mịn lưới bằng `--action_dim`.
- Nếu cần vẽ lưới mịn hơn (chỉ để render), dùng `--action_grid_dim`.

Màu hiển thị thường là:
- Đỏ: action không hợp lệ (mask = 0)
- Xanh lá: action hợp lệ
- Xanh dương: action đang chọn
 - Tím: text thống kê `mask grid: ...`

---

## 7. Workflow khuyến nghị hằng ngày

## Bước 1 - Build + env

```bash
cd ~/LVTN/amr_ws/HALO
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc_rl
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtiff.so.5
./build_standalone.sh
```

## Bước 2 - Train

```bash
cd ~/LVTN/amr_ws/HALO/drl_moudle
python train_ppo.py --output_dir train_data/run_exp01
```

## Bước 3 - Theo dõi live

```bash
tensorboard --logdir ~/LVTN/amr_ws/HALO/drl_moudle/train_data/run_exp01
```

## Bước 4 - Test có render

```bash
python eval_ppo.py --model_dir train_data/run_exp01 --n_eval_episodes 20 --visualize --use_action_mask
```

## Bước 5 - Test 1 episode để lưu ảnh lưới

```bash
python eval_ppo.py --model_dir train_data/run_exp01 --n_eval_episodes 1 --output_image_dir images/exp01 --use_action_mask
```

## Bước 6 - Test lại môi trường theo từng phase (smoke test + visualization)

Lệnh dưới đây sẽ:
- Khởi tạo môi trường cho từng phase `0,1,2,3,4`
- In số lượng human, obstacle, polygon obstacle
- Xuất ảnh snapshot mỗi phase vào `eval_data/phase_visualization_v3/phase_x/frame_0000.png`

```bash
cd ~/LVTN/amr_ws/HALO/drl_moudle
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc_rl
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtiff.so.5
export MPLBACKEND=Agg

python - <<'PY'
import importlib.util
import os
import gym
import matplotlib.pyplot as plt

from crowd_sim.envs.utils.robot import Robot
from modules.policies import ExternalPolicy

root_out = os.path.join('eval_data', 'phase_visualization_v3')
os.makedirs(root_out, exist_ok=True)

config_path = os.path.join('configs', 'mpc_rl.py')
spec = importlib.util.spec_from_file_location('config', config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
env_config = config.EnvConfig(debug=True)

for phase in [0, 1, 2, 3, 4]:
  env = gym.make('CrowdSim-v0', disable_env_checker=True)
  env.configure(env_config)

  robot = Robot(env_config, 'robot')
  robot.time_step = env.time_step
  robot.set_policy(ExternalPolicy())
  env.set_robot(robot)

  env.num_actions_per_dim = 9
  env.goal_coord_range = (-2.25, 2.25)
  env.use_AM = True
  env.use_action_mask = True
  env.unwrapped.num_actions_per_dim = 9
  env.unwrapped.goal_coord_range = (-2.25, 2.25)
  env.unwrapped.use_AM = True
  env.unwrapped.use_action_mask = True

  env.set_phase(phase)
  env.reset(phase='test', for_debug=True, seed=700 + phase)

  out_dir = os.path.join(root_out, f'phase_{phase}')
  os.makedirs(out_dir, exist_ok=True)
  env.render(mode='debug', output_file=out_dir)

  poly_shapes = [len(poly) - 1 for poly in env.poly_obstacles] if env.poly_obstacles else []
  print(
    f'phase={phase} | humans={len(env.humans)} | static_obs={len(env.obstacles)} '
    f'| poly_obs={len(env.poly_obstacles)} | poly_sides={poly_shapes}'
  )

  plt.close('all')

print('Done. Open eval_data/phase_visualization_v3 to inspect images.')
PY
```

Muốn kiểm tra nhanh action phía sau còn bị block không:

```bash
python - <<'PY'
import importlib.util
import os
import gym

from crowd_sim.envs.utils.robot import Robot
from modules.policies import ExternalPolicy

config_path = os.path.join('configs', 'mpc_rl.py')
spec = importlib.util.spec_from_file_location('config', config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
env_config = config.EnvConfig(debug=True)

env = gym.make('CrowdSim-v0', disable_env_checker=True)
env.configure(env_config)
robot = Robot(env_config, 'robot')
robot.time_step = env.time_step
robot.set_policy(ExternalPolicy())
env.set_robot(robot)

env.num_actions_per_dim = 9
env.goal_coord_range = (-2.25, 2.25)
env.use_AM = True
env.unwrapped.num_actions_per_dim = 9
env.unwrapped.goal_coord_range = (-2.25, 2.25)
env.unwrapped.use_AM = True

env.set_phase(3)
env.reset(phase='test', for_debug=True, seed=901)

mask = env.action_mask.detach().cpu().numpy()
back_total = 0
back_valid = 0
for idx, m in enumerate(mask):
  gx, _ = env.map_action_to_goal(idx)
  if gx < 0:
    back_total += 1
    if m > 0:
      back_valid += 1

print(f'back_actions_total={back_total}, back_actions_valid={back_valid}, total_valid={(mask > 0).sum()}')
PY
```

---

## 8. Lỗi thường gặp

## Lỗi: `ModuleNotFoundError: ocp_planner_py`

```bash
conda activate mpc_rl
cd ~/LVTN/amr_ws/HALO
./build_standalone.sh
cd drl_moudle
python -c "import ocp_planner_py; print('OK')"
```

## Lỗi: `ImportError ... libgdal ... TIFFReadRGBATileExt`

Đây là lỗi xung đột thư viện TIFF/GDAL trong môi trường conda.

Fix nhanh:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc_rl
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtiff.so.5
cd ~/LVTN/amr_ws/HALO/drl_moudle
python -c "import ocp_planner_py; print('OK')"
```

Nếu `OK` thì chạy lại lệnh train/eval trong cùng terminal đó.

## Tùy chọn: tạo conda env mới (sạch hơn)

```bash
cd ~/LVTN/amr_ws/HALO/drl_moudle
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml -n mpc_rl_clean
conda activate mpc_rl_clean
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtiff.so.5
python -c "import ocp_planner_py; print('OK')"
```

Sau đó thay `conda activate mpc_rl` bằng `conda activate mpc_rl_clean` trong tất cả lệnh train/eval.

## Lỗi: chưa có TensorBoard

```bash
pip install tensorboard
```

## Lỗi TensorBoard: `No dashboards are active for the current data set`

Nguyên nhân thường gặp:

- Bạn mở sai thư mục log (đúng nhất là thư mục run con: `run1/PPO_1`).
- Train chưa chạy đủ 1 vòng rollout để flush scalar đầu tiên.
- Bạn dừng sớm (Ctrl+C) trước khi có lần dump log đầu tiên.

Kiểm tra nhanh dữ liệu có chưa:

```bash
cd ~/LVTN/amr_ws/HALO/drl_moudle
tensorboard --inspect --logdir train_data/run1/PPO_1
```

Mở TensorBoard đúng cách:

```bash
cd ~/LVTN/amr_ws/HALO/drl_moudle
tensorboard --logdir train_data/run1/PPO_1 --port 6006
```

Nếu đang mở web mà vẫn chưa thấy, bấm refresh cứng (Ctrl+F5) sau 1-2 phút.

## Lỗi: không thấy lưới

- Kiểm tra có bật `--use_action_mask` chưa.
- Dùng `--n_eval_episodes 1` + `--output_image_dir` để đảm bảo có frame.

## Lỗi eval exit code 1

- Chạy lại sau khi chắc chắn đã activate đúng env:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc_rl
cd ~/LVTN/amr_ws/HALO/drl_moudle
python eval_ppo.py --help
```

- Nếu `--help` chạy được, thử lại lệnh eval.
- Nếu vẫn fail, mở `output.log` trong thư mục `--output_dir` để xem traceback đầy đủ.

---

## 9. Cấu hình train khuyến nghị (ổn định + ít nhiễu)

### 9.1 Cấu hình resume nên dùng hằng ngày

```bash
python train_ppo.py \
  --config configs/mpc_rl.py \
  --output_dir train_data/run_06 \
  --total_timesteps 5000000 \
  --eval_freq 300 \
  --n_eval_episodes 50 \
  --action_dim 9 \
  --action_range 2.25 \
  --use_AM True \
  --use_PL True \
  --n_steps 2048 \
  --batch_size 128 \
  --n_epochs 10 \
  --resume \
  --resume_from auto
```

### 9.2 Cấu hình khi cần số liệu báo cáo chắc hơn

```bash
python train_ppo.py \
  --config configs/mpc_rl.py \
  --output_dir train_data/run_06 \
  --total_timesteps 5000000 \
  --eval_freq 500 \
  --n_eval_episodes 100 \
  --action_dim 9 \
  --action_range 2.25 \
  --use_AM True \
  --use_PL True \
  --n_steps 2048 \
  --batch_size 128 \
  --n_epochs 10 \
  --resume \
  --resume_from auto



  export LD_PRELOAD=/lib/x86_64-linux-gnu/libtiff.so.5 && python train_ppo.py \
  --config configs/mpc_rl.py \
  --output_dir train_data/run_07 \
  --total_timesteps 5000000 \
  --eval_freq 500 \
  --n_eval_episodes 100 \
  --action_dim 9 \
  --action_range 2.25 \
  --use_AM True \
  --use_PL True \
  --n_steps 2048 \
  --batch_size 256 \
  --n_epochs 10 \
  --re_arrival 2.0 \
  --re_collision -2.0 \
  --safe_weight 0.1 \
  --goal_weight 0.2 \
  --re_rvo 0.05 \
  --re_theta 0.01
```

### 9.3 Nên train tiếp hay train lại từ đầu?

- Nên **train tiếp** nếu success rate eval dao động nhưng vẫn có xu hướng đi lên sau vài lần eval liên tiếp.
- Với curriculum hiện tại, phase 1 bắt đầu từ episode 4000 nên ở mốc thấp hơn có thể chưa phản ánh hết năng lực model.
- Chỉ nên **train lại từ đầu** khi bạn thay đổi lớn về reward, action space, hoặc observation/perception.
- Trước khi quyết định train lại, hãy chạy eval riêng với nhiều episode để giảm nhiễu:

```bash
python eval_ppo.py \
  --config configs/mpc_rl.py \
  --model_dir train_data/run_06 \
  --n_eval_episodes 200 \
  --use_action_mask
```

// check môi trường 
 cd /home/thuong/LVTN/HALO_LVTN/drl_moudle && export MPLBACKEND=Agg && /home/thuong/miniconda3/envs/mpc_rl/bin/python - <<'PY'

import importlib.util

import os

import gym

import matplotlib.pyplot as plt


from crowd_sim.envs.utils.robot import Robot

from modules.policies import ExternalPolicy


out_root = os.path.join('eval_data', 'eval_phase_visualization_run07')

os.makedirs(out_root, exist_ok=True)


spec = importlib.util.spec_from_file_location('config', os.path.join('configs', 'mpc_rl.py'))

config = importlib.util.module_from_spec(spec)

spec.loader.exec_module(config)

env_config = config.EnvConfig(debug=True)


for phase in [0, 1, 2, 3, 4]:

    env = gym.make('CrowdSim-v0', disable_env_checker=True)

    env.configure(env_config)


    robot = Robot(env_config, 'robot')

    robot.time_step = env.time_step

    robot.set_policy(ExternalPolicy())

    env.set_robot(robot)


    env.num_actions_per_dim = 9

    env.goal_coord_range = (-2.25, 2.25)

    env.use_AM = True

    env.unwrapped.num_actions_per_dim = 9

    env.unwrapped.goal_coord_range = (-2.25, 2.25)

    env.unwrapped.use_AM = True


    env.set_phase(phase)

    env.reset(phase='test', for_debug=True, seed=1000 + phase)


    out_dir = os.path.join(out_root, f'phase_{phase}')

    os.makedirs(out_dir, exist_ok=True)

    env.render(mode='debug', output_file=out_dir)


    poly_shapes = [len(poly) - 1 for poly in env.poly_obstacles] if env.poly_obstacles else []

    print(f'phase={phase} humans={len(env.humans)} static={len(env.obstacles)} poly={len(env.poly_obstacles)} poly_sides={poly_shapes}')

    plt.close('all')


print('saved_to', out_root)

PY





cd /home/thuong/LVTN/HALO_LVTN/drl_moudle && /home/thuong/miniconda3/envs/mpc_rl/bin/python - <<'PY'

import importlib.util

import os

import gym


from crowd_sim.envs.utils.robot import Robot

from modules.policies import ExternalPolicy


spec = importlib.util.spec_from_file_location('config', os.path.join('configs', 'mpc_rl.py'))

config = importlib.util.module_from_spec(spec)

spec.loader.exec_module(config)

env_config = config.EnvConfig(debug=True)


for phase in [0,1,2,3,4]:

    print(f'--- phase {phase} ---')

    for seed in [2000, 2001, 2002]:

        env = gym.make('CrowdSim-v0', disable_env_checker=True)

        env.configure(env_config)

        robot = Robot(env_config, 'robot')

        robot.time_step = env.time_step

        robot.set_policy(ExternalPolicy())

        env.set_robot(robot)

        env.set_phase(phase)

        env.reset(phase='test', for_debug=True, seed=seed)

        print(f'seed={seed} humans={len(env.humans)} static={len(env.obstacles)} poly={len(env.poly_obstacles)}')

PY







---

## 10. GPU thấp (1-10%) có bình thường không?

Với HALO (MPC + RL), **GPU thấp là thường gặp** vì:

- Phần nặng nhất mỗi step là solver MPC (`ocp_planner_py`) chạy CPU.
- GPU chủ yếu dùng cho forward/backward của policy PPO.

Nên dù có CUDA, `GPU-Util` vẫn có thể thấp nếu môi trường/simulator là bottleneck.

## 11. Cách tăng tốc train thực tế

### 11.1 Tăng khối lượng update trên GPU

Script `train_ppo.py` đã hỗ trợ:

- `--n_steps`
- `--batch_size`
- `--n_epochs`

Profile nhanh (tham khảo cho GTX 1650 4GB):

```bash
cd ~/LVTN/amr_ws/HALO/drl_moudle
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mpc_rl
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtiff.so.5

python train_ppo.py \
  --config configs/mpc_rl.py \
  --output_dir train_data/run1 \
  --total_timesteps 5000000 \
  --eval_freq 2000 \
  --n_eval_episodes 20 \
  --action_dim 9 \
  --action_range 2.25 \
  --use_AM True \
  --use_PL True \
  --n_steps 4096 \
  --batch_size 256 \
  --n_epochs 10
```

Ghi chú:

- Nếu bị OOM GPU, giảm `--batch_size` còn `128`.
- `--eval_freq` lớn hơn và `--n_eval_episodes` nhỏ hơn sẽ giảm thời gian eval xen giữa train.

### 11.2 Resume train (không train lại từ đầu)

```bash
python train_ppo.py \
  --config configs/mpc_rl.py \
  --output_dir train_data/run1 \
  --resume \
  --resume_from auto \
  --last_save_freq 50 \
  --total_timesteps 5000000 \
  --n_steps 4096 \
  --batch_size 256 \
  --n_epochs 10
```

## 12. Chạy best_model.zip trong Gazebo (ROS2 `src/`)

Đã thêm node bridge: `amr_planner/rl_local_goal_bridge`.

Node này:
- Load `best_model.zip` từ `--model_dir`.
- Suy luận local-goal từ PPO.
- Xuất lệnh `TwistStamped` ra `/diff_cont/cmd_vel` để robot chạy trong Gazebo.

### 12.1 Build package

```bash
cd ~/LVTN/amr_ws
colcon build --packages-select amr_planner
source install/setup.bash
```

### 12.2 Chạy Gazebo

```bash
ros2 launch amr_descriptions gazebo.launch.py world:=room_20x20.world
```

### 12.3 Chạy bridge best_model (terminal mới)

```bash
cd ~/LVTN/amr_ws
source install/setup.bash

ros2 launch amr_planner rl_local_goal_bridge.launch.py \
  model_dir:=/home/thuong/LVTN/amr_ws/HALO/drl_moudle/train_data/run_04 \
  halo_drl_dir:=/home/thuong/LVTN/amr_ws/HALO/drl_moudle \
  config:=configs/mpc_rl.py
```

### 12.4 Gửi goal để robot chạy

- Mở RViz (từ launch Gazebo), dùng công cụ **2D Goal Pose** để publish vào `/goal_pose`.
- Node bridge sẽ nhận `/odometry/filtered`, `/scan`, `/goal_pose` và phát `/diff_cont/cmd_vel`.

### 12.5 Kiểm tra nhanh topic

```bash
ros2 topic echo /diff_cont/cmd_vel
ros2 topic hz /diff_cont/cmd_vel
```

Nếu không thấy robot chạy:
- Kiểm tra có file `best_model.zip` trong `model_dir`.
- Kiểm tra có `/goal_pose` hay chưa.
- Kiểm tra `odom_topic` đúng (mặc định `/odometry/filtered`).
  --re_theta 0.01
```

Sau khi ổn định mới tăng lên 5e6 timesteps.

---

## 10. Kết luận nhanh

- Train: `HALO/drl_moudle/train_ppo.py`
- Eval/Test: `HALO/drl_moudle/eval_ppo.py`
- Livestream chỉ số: TensorBoard
- Xem lưới action mask: eval với `--use_action_mask`, tốt nhất `--n_eval_episodes 1` + lưu ảnh
- Luôn chạy trước: `conda activate mpc_rl`
