import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_rl_mppi_details(data_dir='./planner_comparison_data'):
    files = glob.glob(os.path.join(data_dir, "rl_mppi_timeseries_*.csv"))
    if not files:
        print("Không tìm thấy file timeseries nào của RL-MPPI!")
        return
    
    latest_file = max(files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    print(f"Đang vẽ biểu đồ từ: {latest_file}")

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    # Extract data to numpy arrays
    t = df['time'].to_numpy()
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    v_lin_cmd = df['v_linear_cmd'].to_numpy()
    
    v_lin_real = df['v_linear_real'].to_numpy()
    
    v_ang_cmd = df['v_angular_cmd'].to_numpy()
    
    v_ang_real = df['v_angular_real'].to_numpy()
    
    clearance = df['clearance'].to_numpy()

    # IMAGE 1: TRAJECTORY
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='Quy trac RL-MPPI', color='#1f77b4', linewidth=2.5)
    plt.scatter(x[0], y[0], color='green', s=100, label='Xuat phat', zorder=5)
    plt.scatter(x[-1], y[-1], color='red', s=100, label='Dich den', zorder=5)
    plt.title('Quy Trac Di Chuyen Thuc Te (RL-MPPI)', fontsize=14, fontweight='bold')
    plt.xlabel('Toa do X (m)')
    plt.ylabel('Toa do Y (m)')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('RL_MPPI_1_Trajectory.png', dpi=300)
    plt.close()

    # IMAGE 2: VELOCITY TRACKING
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(t, v_lin_cmd, label='Lenh van toc (Command)', color='red', linestyle='--', linewidth=2)
    ax1.plot(t, v_lin_real, label='Encoder', color='blue', linewidth=2, alpha=0.9)
    ax1.set_title('Dap Ung Van Toc Thang (Linear Velocity)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('v (m/s)')
    ax1.legend()

    ax2.plot(t, v_ang_cmd, label='Lenh goc xoay (Command)', color='red', linestyle='--', linewidth=2)
    ax2.plot(t, v_ang_real, label='Encoder', color='green', linewidth=2, alpha=0.9)
    ax2.set_title('Dap Ung Van Toc Goc (Angular Velocity)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Thoi gian (s)')
    ax2.set_ylabel('w (rad/s)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('RL_MPPI_2_Tracking_Performance.png', dpi=300)
    plt.close()

    # IMAGE 3: CLEARANCE
    plt.figure(figsize=(10, 5))
    plt.plot(t, clearance, label='Khoang cach', color='purple', linewidth=2.5)
    plt.axhline(y=0.25, color='red', linestyle='-.', label='Ranh gioi va cham (0.25m)', linewidth=2)
    
    info_text = "[TF2 trajectory | encoder velocity raw CSV values]"
    plt.text(0.02, 0.02, info_text, transform=plt.gca().transAxes, fontsize=9,
            verticalalignment='bottom', style='italic', color='gray')
    
    plt.title('Khoang Cach An Toan Cua AGV (RL-MPPI)', fontsize=14, fontweight='bold')
    plt.xlabel('Thoi gian (s)')
    plt.ylabel('Khoang cach (m)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('RL_MPPI_3_Clearance.png', dpi=300)
    plt.close()

    print("OK! Da tao xong 3 anh HD!")
    print("  - Trajectory: TF2 (map frame)")
    print("  - Encoder velocity: raw CSV values")


if __name__ == '__main__':
    plot_rl_mppi_details()
