import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REQUIRED_TIMESERIES_COLUMNS = [
    'time', 'x', 'y',
    'v_linear_cmd', 'v_linear_real',
    'v_angular_cmd', 'v_angular_real',
    'clearance',
]


def plot_rl_mppi_details(data_dir='./planner_comparison_data'):
    files = glob.glob(os.path.join(data_dir, "rl_mppi_timeseries_*.csv"))
    if not files:
        print("Không tìm thấy file timeseries nào của RL-MPPI!")
        return
    
    latest_file = max(files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    if df.empty:
        print("File timeseries RL-MPPI không có dữ liệu!")
        return

    missing = [col for col in REQUIRED_TIMESERIES_COLUMNS if col not in df.columns]
    if missing:
        print(f"File timeseries RL-MPPI thiếu cột: {missing}")
        return

    print(f"Đang vẽ biểu đồ từ: {latest_file}")

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    # --- ÉP KIỂU SANG NUMPY ARRAY ĐỂ FIX LỖI PANDAS ---
    t = df['time'].to_numpy()
    t = t - t[0]
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    v_lin_cmd = df['v_linear_cmd'].to_numpy()
    v_lin_real = df['v_linear_real'].to_numpy()
    v_ang_cmd = df['v_angular_cmd'].to_numpy()
    v_ang_real = df['v_angular_real'].to_numpy()
    clearance = df['clearance'].to_numpy()

    # -------------------------------------------------------------
    # ẢNH 1: QUỸ ĐẠO DI CHUYỂN
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='Quỹ đạo RL-MPPI', color='#1f77b4', linewidth=2.5)
    plt.scatter(x[0], y[0], color='green', s=100, label='Xuất phát', zorder=5)
    plt.scatter(x[-1], y[-1], color='red', s=100, label='Đích đến', zorder=5)
    plt.title('Quỹ Đạo Di Chuyển Thực Tế (RL-MPPI)', fontsize=14, fontweight='bold')
    plt.xlabel('Tọa độ X (m)')
    plt.ylabel('Tọa độ Y (m)')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('RL_MPPI_1_Trajectory.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # ẢNH 2: ĐÁP ỨNG VẬN TỐC (LỆNH vs THỰC TẾ)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Đồ thị Vận tốc thẳng (Linear Velocity)
    ax1.plot(t, v_lin_cmd, label='Lệnh vận tốc (Command)', color='red', linestyle='--', linewidth=2)
    ax1.plot(t, v_lin_real, label='Encoder raw', color='blue', linewidth=2, alpha=0.85)
    ax1.set_title('Đáp Ứng Vận Tốc Thẳng (Linear Velocity)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('v (m/s)')
    ax1.legend()

    # Đồ thị Vận tốc góc (Angular Velocity)
    ax2.plot(t, v_ang_cmd, label='Lệnh góc xoay (Command)', color='red', linestyle='--', linewidth=2)
    ax2.plot(t, v_ang_real, label='Encoder raw', color='green', linewidth=2, alpha=0.85)
    ax2.set_title('Đáp Ứng Vận Tốc Góc (Angular Velocity)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Thời gian (s)')
    ax2.set_ylabel('w (rad/s)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('RL_MPPI_2_Tracking_Performance.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # ẢNH 3: KHOẢNG CÁCH AN TOÀN
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(t, clearance, label='Khoảng cách tới vật cản gần nhất', color='purple', linewidth=2)
    plt.axhline(y=0.25, color='red', linestyle='-.', label='Ranh giới va chạm (0.25m)')
    plt.text(0.02, 0.02, "[TF pose | raw encoder velocity | no plot smoothing]",
             transform=plt.gca().transAxes, fontsize=8, color='gray', style='italic')
    plt.title('Khoảng Cách An Toàn Của AGV (RL-MPPI)', fontsize=14, fontweight='bold')
    plt.xlabel('Thời gian (s)')
    plt.ylabel('Khoảng cách (m)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('RL_MPPI_3_Clearance.png', dpi=300)
    plt.close()

    print("Đã tạo xong 3 ảnh HD! Encoder dùng raw CSV, không smooth/filter.")

if __name__ == '__main__':
    plot_rl_mppi_details()
