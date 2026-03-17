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
python train_ppo.py \
  --config configs/mpc_rl.py \
  --output_dir train_data/run2 \
  --total_timesteps 1000000 \
  --eval_freq 500 \
  --n_eval_episodes 20 \
  --action_dim 9 \
  --action_range 2.25 \
  --use_AM True \
  --use_PL True \
  --resume \
  --start_episode 4048
```

Điều kiện: trong `train_data/run1` có `best_model` hoặc `best_model.zip`.

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

Màu hiển thị thường là:
- Đỏ: action không hợp lệ (mask = 0)
- Xanh lá: action hợp lệ
- Xanh dương: action đang chọn

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

## 9. Profile train gợi ý cho lần đầu

```bash
python train_ppo.py \
  --output_dir train_data/baseline_obs \
  --total_timesteps 1000000 \
  --eval_freq 200 \
  --n_eval_episodes 50 \
  --action_dim 9 \
  --action_range 2.25 \
  --use_AM True \
  --use_PL True \
  --safe_weight 0.5 \
  --goal_weight 0.1 \
  --re_collision -0.25 \
  --re_arrival 0.25 \
  --re_rvo 0.01 \

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
  --total_timesteps 5000000 \
  --n_steps 4096 \
  --batch_size 256 \
  --n_epochs 10
```
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
