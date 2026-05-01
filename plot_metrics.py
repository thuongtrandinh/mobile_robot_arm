import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_dir='./planner_comparison_data'):
    """Đọc tất cả các file CSV và gộp thành 1 DataFrame"""
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not all_files:
        print("Không tìm thấy file CSV nào trong thư mục:", data_dir)
        return None

    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
        
    final_df = pd.concat(df_list, ignore_index=True)
    return final_df

def plot_metrics(df):
    # Cài đặt style cho đồ thị chuẩn khoa học
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Tạo 1 bảng (figure) gồm 4 đồ thị con (2 hàng, 2 cột)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ĐÁNH GIÁ VÀ SO SÁNH HIỆU SUẤT CÁC BỘ ĐIỀU KHIỂN (LOCAL PLANNERS)', fontsize=16, fontweight='bold')

    # Định dạng màu cho từng thuật toán để dễ nhận diện
    palette = {'rl_mppi': '#1f77b4', 'dwa': '#ff7f0e', 'teb': '#2ca02c'}

    # ---------------------------------------------------------
    # Đồ thị 1: Thời gian di chuyển đến đích (Time to Goal)
    # ---------------------------------------------------------
    sns.barplot(data=df, x='planner', y='time_to_goal', ax=axes[0, 0], palette=palette, capsize=.1)
    axes[0, 0].set_title('Thời gian hoàn thành quỹ đạo (Thấp hơn là Tốt hơn)')
    axes[0, 0].set_ylabel('Thời gian (giây)')
    axes[0, 0].set_xlabel('Thuật toán')

    # ---------------------------------------------------------
    # Đồ thị 2: Tổng quãng đường thực tế vs Quãng đường lý thuyết
    # (Đánh giá mức độ bám đường / đánh võng)
    # ---------------------------------------------------------
    # Chuẩn bị dữ liệu cho biểu đồ cột ghép (Grouped Bar Chart)
    df_melted = df.melt(id_vars=['planner'], value_vars=['total_distance', 'path_length'], 
                        var_name='Loại khoảng cách', value_name='Khoảng cách (m)')
    df_melted['Loại khoảng cách'] = df_melted['Loại khoảng cách'].replace({
        'total_distance': 'Thực tế di chuyển (Odom)', 
        'path_length': 'Lý thuyết (A* Path)'
    })
    
    sns.barplot(data=df_melted, x='planner', y='Khoảng cách (m)', hue='Loại khoảng cách', ax=axes[0, 1], capsize=.1)
    axes[0, 1].set_title('Quãng đường di chuyển (Thực tế vs Lý thuyết)')
    axes[0, 1].set_ylabel('Khoảng cách (mét)')
    axes[0, 1].set_xlabel('Thuật toán')

    # ---------------------------------------------------------
    # Đồ thị 3: Số lần dao động (Oscillations) - Mức độ mượt mà
    # ---------------------------------------------------------
    sns.barplot(data=df, x='planner', y='num_oscillations', ax=axes[1, 0], palette=palette, capsize=.1)
    axes[1, 0].set_title('Số lần dao động/đảo chiều vận tốc (Thấp hơn là Mượt hơn)')
    axes[1, 0].set_ylabel('Số lần')
    axes[1, 0].set_xlabel('Thuật toán')

    # ---------------------------------------------------------
    # Đồ thị 4: Thời gian tính toán trung bình (Computation Time)
    # ---------------------------------------------------------
    sns.barplot(data=df, x='planner', y='avg_computation_time', ax=axes[1, 1], palette=palette, capsize=.1)
    axes[1, 1].set_title('Thời gian xử lý trung bình mỗi nhịp (Thấp hơn là Tốt hơn)')
    axes[1, 1].set_ylabel('Thời gian tính toán (giây)')
    axes[1, 1].set_xlabel('Thuật toán')

    # Căn chỉnh layout cho đẹp và Lưu file
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_img = 'planner_comparison_chart.png'
    plt.savefig(output_img, dpi=300) # Xuất ảnh độ phân giải cao 300dpi để đưa vào Word
    print(f"Đã lưu biểu đồ thành công tại: {output_img}")
    
    # Hiển thị lên màn hình
    plt.show()

if __name__ == '__main__':
    df = load_data()
    if df is not None:
        plot_metrics(df)