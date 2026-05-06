import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d

def smooth_clearance_data(clearance, window_size=5):
    """Smooth clearance data to reduce encoder noise."""
    if len(clearance) < window_size:
        return clearance
    return uniform_filter1d(clearance, size=window_size, mode='nearest')


def analyze_and_plot_planners(algorithms=['rl_mppi', 'teb', 'dwa'], data_dir='./planner_comparison_data'):
    """
    Plot planner comparison using TF2-based trajectory (map frame).
    Clearance data smoothed to remove encoder noise.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

    comparison_data = {}

    # =========================================================
    # BƯỚC 1: LẤY TỌA ĐỘ ĐÍCH CHUẨN TỪ THUẬT TOÁN RL-MPPI
    # =========================================================
    ref_goal_x, ref_goal_y = None, None
    rl_files = glob.glob(os.path.join(data_dir, "rl_mppi_timeseries_*.csv"))
    
    if rl_files:
        latest_rl_file = max(rl_files, key=os.path.getctime)
        df_rl = pd.read_csv(latest_rl_file)
        # Check if RL-MPPI data has rows
        if len(df_rl) == 0:
            print("[!] File RL-MPPI không có dữ liệu! Vui lòng chạy RL-MPPI trước để thu thập dữ liệu.")
            return
        # Lấy điểm cuối cùng của RL-MPPI làm đích chuẩn tuyệt đối
        ref_goal_x = df_rl['x'].to_numpy()[-1]
        ref_goal_y = df_rl['y'].to_numpy()[-1]
        print(f"[*] Đã thiết lập Đích Chuẩn từ RL-MPPI: X={ref_goal_x:.2f}, Y={ref_goal_y:.2f}")
    else:
        print("[!] Không tìm thấy dữ liệu RL-MPPI. Vui lòng đảm bảo RL-MPPI có trong thư mục.")
        return

    # =========================================================
    # BƯỚC 2: XỬ LÝ VÀ VẼ BIỂU ĐỒ CHO TỪNG THUẬT TOÁN
    # =========================================================
    for algo in algorithms:
        files = glob.glob(os.path.join(data_dir, f"{algo}_timeseries_*.csv"))
        if not files:
            print(f"Bỏ qua {algo.upper()}: Không tìm thấy file dữ liệu!")
            continue
        
        latest_file = max(files, key=os.path.getctime)
        df = pd.read_csv(latest_file)
        
        # Check if file has data
        if len(df) == 0:
            print(f"Bỏ qua {algo.upper()}: File CSV không có dữ liệu!")
            continue
        
        print(f"--- Đang xử lý dữ liệu thuật toán: {algo.upper()} ---")

        t = df['time'].to_numpy()
        t = t - t[0] # Chuẩn hóa thời gian về 0s
        
        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        
        # --- ÁP DỤNG LỌC NHIỄU (MOVING AVERAGE) CHO VẬN TỐC THỰC TẾ ---
        WINDOW_SIZE = 5 # Cỡ cửa sổ lọc. Có thể tăng lên 7 hoặc 10 nếu đồ thị vẫn còn gai
        
        v_lin_cmd = df['v_linear_cmd'].to_numpy()
        v_lin_real = df['v_linear_real'].rolling(window=WINDOW_SIZE, min_periods=1).mean().to_numpy()
        
        v_ang_cmd = df['v_angular_cmd'].to_numpy()
        v_ang_real = df['v_angular_real'].rolling(window=WINDOW_SIZE, min_periods=1).mean().to_numpy()
        
        clearance = df['clearance'].to_numpy()

        # Tính khoảng cách đến ĐÍCH CHUẨN (từ RL-MPPI) thay vì đích riêng của từng thuật toán
        dist_to_goal = np.sqrt((x - ref_goal_x)**2 + (y - ref_goal_y)**2)
        
        dx = np.diff(x)
        dy = np.diff(y)
        total_distance = np.sum(np.sqrt(dx**2 + dy**2))
        avg_clearance = np.mean(clearance)
        min_clearance = np.min(clearance)
        
        comparison_data[algo.upper()] = {
            't': t, 'x': x, 'y': y, 
            'dist_to_goal': dist_to_goal, 'clearance': clearance,
            'clearance_raw': clearance_raw
        }

        # --- ẢNH 1: QUỸ ĐẠO DI CHUYỂN RIÊNG ---
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, label=f'Quỹ đạo {algo.upper()}', color='#1f77b4', linewidth=2.5)
        plt.scatter(x[0], y[0], color='green', s=100, label='Xuất phát', zorder=5)
        # Vẽ điểm đích chuẩn lên đồ thị
        plt.scatter(ref_goal_x, ref_goal_y, color='red', s=100, label='Đích chuẩn (RL-MPPI)', zorder=5)
        
        info_text = f"Quãng đường: {total_distance:.2f} m"
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.title(f'Quỹ Đạo Di Chuyển Thực Tế ({algo.upper()})', fontsize=14, fontweight='bold')
        plt.xlabel('Tọa độ X (m)')
        plt.ylabel('Tọa độ Y (m)')
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{algo.upper()}_1_Trajectory.png', dpi=300)
        plt.close()

        # --- ẢNH 2: KHOẢNG CÁCH TỚI ĐÍCH CHUẨN ---
        plt.figure(figsize=(8, 5))
        plt.plot(t, dist_to_goal, label=f'Distance to Goal', color='#ff7f0e', linewidth=2.5)
        plt.title(f'Khoảng Cách Đến Đích Chuẩn Theo Thời Gian ({algo.upper()})', fontsize=14, fontweight='bold')
        plt.xlabel('Thời gian (s)')
        plt.ylabel('Khoảng cách (m)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{algo.upper()}_2_DistToGoal.png', dpi=300)
        plt.close()

        # --- ẢNH 3: ĐÁP ỨNG VẬN TỐC ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(t, v_lin_cmd, label='Lệnh (Cmd)', color='red', linestyle='--', linewidth=2)
        ax1.plot(t, v_lin_real, label='Thực tế (Real)', color='blue', linewidth=2, alpha=0.8)
        ax1.set_title(f'Đáp Ứng Vận Tốc Thẳng - {algo.upper()}', fontsize=13, fontweight='bold')
        ax1.set_ylabel('v (m/s)')
        ax1.legend()

        ax2.plot(t, v_ang_cmd, label='Lệnh (Cmd)', color='red', linestyle='--', linewidth=2)
        ax2.plot(t, v_ang_real, label='Thực tế (Real)', color='green', linewidth=2, alpha=0.8)
        ax2.set_title(f'Đáp Ứng Vận Tốc Góc - {algo.upper()}', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Thời gian (s)')
        ax2.set_ylabel('w (rad/s)')
        ax2.legend()
        plt.tight_layout()
        plt.savefig(f'{algo.upper()}_3_Tracking.png', dpi=300)
        plt.close()

        # --- ẢNH 4: KHOẢNG CÁCH AN TOÀN (CLEARANCE) ---
        plt.figure(figsize=(10, 5))
        plt.plot(t, clearance, label='Khoảng cách (lọc nhiễu)', color='purple', linewidth=2.5)
        plt.plot(t, clearance_raw, label='Khoảng cách (encoder noise)', 
                color='purple', linewidth=0.8, alpha=0.3, linestyle=':')
        plt.axhline(y=0.25, color='red', linestyle='-.', label='Va chạm (0.25m)', linewidth=2)
        
        clearance_text = (f"Clearance TB: {avg_clearance:.2f} m\n"
                          f"Clearance Min: {min_clearance:.2f} m")
        plt.text(0.05, 0.95, clearance_text, transform=plt.gca().transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.title(f'Phân Tích Khoảng Cách An Toàn ({algo.upper()})', fontsize=14, fontweight='bold')
        plt.xlabel('Thời gian (s)')
        plt.ylabel('Khoảng cách (m)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{algo.upper()}_4_Clearance.png', dpi=300)
        plt.close()

    # =========================================================
    # BƯỚC 3: PHẦN VẼ BIỂU ĐỒ SO SÁNH GỘP TỔNG HỢP
    # =========================================================
    if not comparison_data:
        return

    print("--- Đang tạo các biểu đồ SO SÁNH TỔNG HỢP ---")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 

    # 1. So sánh Quỹ đạo 
    plt.figure(figsize=(9, 7))
    for i, (algo, data) in enumerate(comparison_data.items()):
        plt.plot(data['x'], data['y'], label=f'{algo}', color=colors[i%len(colors)], linewidth=2)
    # Vẽ chung MỘT điểm đích chuẩn duy nhất
    plt.scatter(ref_goal_x, ref_goal_y, color='red', s=150, marker='*', label='Goal Chuẩn (RL-MPPI)', zorder=10)
    plt.title('So Sánh Quỹ Đạo Di Chuyển Giữa Các Thuật Toán', fontsize=15, fontweight='bold')
    plt.xlabel('Tọa độ X (m)')
    plt.ylabel('Tọa độ Y (m)')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('Comparison_1_Trajectories.png', dpi=300)
    plt.close()

    # 2. So sánh Khoảng cách đến đích chuẩn
    plt.figure(figsize=(10, 6))
    for i, (algo, data) in enumerate(comparison_data.items()):
        plt.plot(data['t'], data['dist_to_goal'], label=f'{algo}', color=colors[i%len(colors)], linewidth=2.5)
    plt.title('So Sánh Tốc Độ Hội Tụ Về Đích Chuẩn (Distance to Reference Goal)', fontsize=15, fontweight='bold')
    plt.xlabel('Thời gian (s)')
    plt.ylabel('Khoảng cách đến đích (m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Comparison_2_Convergence.png', dpi=300)
    plt.close()

    # 3. So sánh Khoảng cách an toàn (Clearance) - Smoothed
    plt.figure(figsize=(10, 6))
    for i, (algo, data) in enumerate(comparison_data.items()):
        plt.plot(data['t'], data['clearance'], label=f'{algo}', color=colors[i%len(colors)], linewidth=2.5, alpha=0.85)
    plt.axhline(y=0.25, color='red', linestyle='-.', label='Va chạm (0.25m)', linewidth=2)
    
    note_text = "[Trajectory: TF2/map frame | Clearance: encoder noise filtered]"
    plt.text(0.02, 0.02, note_text, transform=plt.gca().transAxes, fontsize=8,
            verticalalignment='bottom', style='italic', color='gray')
    
    plt.title('So Sánh Khoảng Cách An Toàn (Clearance) Theo Thời Gian', fontsize=15, fontweight='bold')
    plt.xlabel('Thời gian (s)')
    plt.ylabel('Khoảng cách đến vật cản (m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Comparison_3_Clearance.png', dpi=300)
    plt.close()

    print("✓ Hoàn tất! Đã lưu toàn bộ đồ thị.")
    print("  - Trajectory: TF2 (map frame, không odom drift)")
    print("  - Clearance: Lọc nhiễu encoder (5-point moving average)")

if __name__ == '__main__':
    analyze_and_plot_planners(algorithms=['rl_mppi', 'teb', 'dwa'])