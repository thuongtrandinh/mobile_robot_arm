# Hướng dẫn Setup để Training với HALO

## 📋 Tóm tắt nhanh

**Để training, bạn có 3 lựa chọn:**

1. ✅ **Standalone (KHÔNG CẦN ROS)** - Đơn giản nhất, khuyến nghị
2. 🔵 **ROS2 Humble** - Nếu cần integration với ROS2
3. 🟡 **ROS1 Noetic** - Theo hướng dẫn gốc (Ubuntu 20.04)

---

## 🚀 PHƯƠNG ÁN 1: STANDALONE BUILD (KHUYẾN NGHỊ)

### Bước 1: Cài đặt dependencies cơ bản

```bash
# System dependencies (Ubuntu 22.04)
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libeigen3-dev \
    git

# Casadi (optimization library)
# Download từ: https://github.com/casadi/casadi/releases
# Hoặc cài qua pip trong conda env
```

### Bước 2: Tạo conda environment

```bash
cd ~/LVTN/HALO
conda env create -f drl_moudle/environment.yml
conda activate mpc_rl
```

### Bước 3: Clone pybind11

```bash
cd ~/LVTN/HALO/src/ocp_planner
mkdir -p extern && cd extern
git clone https://github.com/pybind/pybind11.git
```

### Bước 4: Sử dụng CMakeLists standalone

```bash
cd ~/LVTN/HALO/src/ocp_planner
cp CMakeLists_standalone.txt CMakeLists.txt
```

### Bước 5: Build

```bash
cd ~/LVTN/HALO
chmod +x build_standalone.sh
./build_standalone.sh
```

### Bước 6: Training

```bash
cd ~/LVTN/HALO/drl_moudle
conda activate mpc_rl
python train_ppo.py
```

---

## 🔵 PHƯƠNG ÁN 2: ROS2 HUMBLE

### Bước 1: Cài đặt ROS2 Humble (Ubuntu 22.04)

```bash
# Follow: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html

sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-colcon-common-extensions
```

### Bước 2: Source ROS2

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Bước 3: Dependencies khác

```bash
# Giống như phương án 1
sudo apt install -y libopencv-dev libeigen3-dev
cd ~/LVTN/HALO
conda env create -f drl_moudle/environment.yml
cd src/ocp_planner
mkdir -p extern && cd extern
git clone https://github.com/pybind/pybind11.git
```

### Bước 4: Sử dụng files ROS2

```bash
cd ~/LVTN/HALO/src/ocp_planner
cp CMakeLists_ros2.txt CMakeLists.txt
cp package_ros2.xml package.xml
```

### Bước 5: Build với colcon

```bash
cd ~/LVTN/HALO
chmod +x build_cp_ros2.sh
./build_cp_ros2.sh
```

### Bước 6: Training

```bash
cd ~/LVTN/HALO/drl_moudle
conda activate mpc_rl
python train_ppo.py
```

---

## 🟡 PHƯƠNG ÁN 3: ROS1 NOETIC (GỐC)

Yêu cầu Ubuntu 20.04. Xem [assets/Run.md](../assets/Run.md)

---

## ⚠️ LƯU Ý QUAN TRỌNG

### 1. Python paths trong CMakeLists.txt

Dòng 113-114 có hardcoded paths:
```cmake
set(PYTHON_EXECUTABLE "/home/wangchao/miniconda3/envs/mpc_rl/bin/python")
set(PYTHON_INCLUDE_DIRECTORY "/home/wangchao/miniconda3/envs/mpc_rl/include/python3.8")
```

**Phải đổi thành đường dẫn của bạn** hoặc dùng biến môi trường:
```cmake
set(PYTHON_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
set(PYTHON_INCLUDE_DIRECTORY "$ENV{CONDA_PREFIX}/include/python3.8")
```

### 2. Casadi library

Casadi cần được cài đặt và CMake có thể tìm thấy. Kiểm tra:
```bash
find /usr -name "libcasadi*" 2>/dev/null
```

### 3. MA57 solver

MPC solver yêu cầu MA57 từ HSL (academic license):
- http://www.hsl.rl.ac.uk/ipopt/
- Cần verify academic email để download

### 4. GPU Support (Optional nhưng nên có)

Environment.yml có CUDA 12.1 dependencies:
- Cần NVIDIA driver tương thích
- CUDA toolkit 12.1
- Kiểm tra: `nvidia-smi`

---

## 🎯 KHUYẾN NGHỊ

**Cho mục đích TRAINING RL**:
- ✅ Sử dụng **Phương án 1 (Standalone)**
- Đơn giản, không cần ROS
- Dễ debug
- Có thể chuyển sang máy khác (copy file .so)

**Cho mục đích DEPLOYMENT với Robot thật**:
- Dùng **ROS2 Humble** (Phương án 2)
- Cần wrapper code để integrate

---

## 📝 Checklist

- [ ] Cài đặt system dependencies (OpenCV, Eigen3, Casadi)
- [ ] Tạo conda environment từ environment.yml
- [ ] Clone pybind11 vào extern/
- [ ] Sửa Python paths trong CMakeLists.txt
- [ ] Chọn và copy file CMakeLists phù hợp
- [ ] Build thành công (có file .so trong drl_moudle/)
- [ ] Test training: `python train_ppo.py`

---

## 🆘 Troubleshooting

**Lỗi: "casadi not found"**
```bash
conda activate mpc_rl
pip install casadi
```

**Lỗi: "ocp_planner_py module not found"**
```bash
# Kiểm tra file .so có tồn tại không
ls -lh ~/LVTN/HALO/drl_moudle/ocp_planner_py*.so

# Kiểm tra Python có import được không
cd ~/LVTN/HALO/drl_moudle
python -c "import ocp_planner_py; print('OK')"
```

**Lỗi build: "Python.h not found"**
```bash
# Đảm bảo conda env active và sửa PYTHON_INCLUDE_DIRECTORY
conda activate mpc_rl
echo $CONDA_PREFIX/include/python3.8
```
