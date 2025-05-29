"""
解析、分析和选择CTF参数文件
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
# 导入basement目录中的模块
import sys
import os

# 添加basement目录到系统路径
basement_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'basement')
if basement_dir not in sys.path:
    sys.path.append(basement_dir)

# 导入basement模块
from basement.ctf import compute_ctf
import torch

def load_ctf_pkl(pkl_file):
    """加载CTF参数文件"""
    with open(pkl_file, 'rb') as f:
        ctf_params = pickle.load(f)
    return ctf_params

def analyze_ctf_params(ctf_params):
    """分析CTF参数的基本信息"""
    print(f"CTF参数形状: {ctf_params.shape}")
    
    if ctf_params.shape[1] == 9:
        print("\n前10组CTF参数:")
        column_names = ["尺寸D", "像素大小(Å/px)", "DefocusU(Å)", "DefocusV(Å)", 
                        "散焦角度(°)", "电压(kV)", "球差Cs(mm)", "振幅对比度比w", "相位偏移(°)"]
        
        # 打印列名
        header = "序号".ljust(6)
        for name in column_names:
            header += name.ljust(15)
        print(header)
        
        # 打印前10组数据
        for i in range(min(10, len(ctf_params))):
            row = f"{i}".ljust(6)
            for j, val in enumerate(ctf_params[i]):
                if j == 0:  # 尺寸通常是整数
                    row += f"{int(val)}".ljust(15)
                else:
                    row += f"{val:.6f}".ljust(15)
            print(row)
            
        # 计算各参数的统计量
        print("\n参数统计信息:")
        for i, name in enumerate(column_names):
            values = ctf_params[:, i]
            print(f"{name}:")
            print(f"  范围: {np.min(values):.6f} - {np.max(values):.6f}")
            print(f"  均值: {np.mean(values):.6f}")
            print(f"  标准差: {np.std(values):.6f}")
            print()

def plot_defocus_distribution(ctf_params, output_dir="."):
    """绘制散焦值的分布"""
    if ctf_params.shape[1] < 4:
        print("参数列数不足，无法绘制散焦分布")
        return
    
    # 创建散焦分布直方图
    plt.figure(figsize=(12, 8))
    
    # 散焦U的分布
    plt.subplot(2, 2, 1)
    plt.hist(ctf_params[:, 2], bins=50)
    plt.title('DefocusU分布')
    plt.xlabel('DefocusU (Å)')
    plt.ylabel('频数')
    
    # 散焦V的分布
    plt.subplot(2, 2, 2)
    plt.hist(ctf_params[:, 3], bins=50)
    plt.title('DefocusV分布')
    plt.xlabel('DefocusV (Å)')
    plt.ylabel('频数')
    
    # 散焦U vs 散焦V
    plt.subplot(2, 2, 3)
    plt.scatter(ctf_params[:, 2], ctf_params[:, 3], alpha=0.5, s=5)
    plt.title('DefocusU vs DefocusV')
    plt.xlabel('DefocusU (Å)')
    plt.ylabel('DefocusV (Å)')
    
    # 散焦差异(U-V)分布
    plt.subplot(2, 2, 4)
    plt.hist(ctf_params[:, 2] - ctf_params[:, 3], bins=50)
    plt.title('DefocusU - DefocusV分布')
    plt.xlabel('DefocusU - DefocusV (Å)')
    plt.ylabel('频数')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'defocus_distribution.png'))
    print(f"已保存散焦分布图到 {os.path.join(output_dir, 'defocus_distribution.png')}")
    plt.close()

def plot_ctf_examples(ctf_params, output_dir="."):
    """绘制几个CTF的例子"""
    if ctf_params.shape[1] < 9:
        print("参数列数不足，无法绘制CTF示例")
        return
    
    D = int(ctf_params[0, 0])  # 图像尺寸
    Apix = ctf_params[0, 1]    # 像素大小
    
    # 选择5个CTF参数样本绘制
    sample_indices = np.linspace(0, len(ctf_params)-1, 5, dtype=int)
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(sample_indices):
        params = ctf_params[idx]
        
        # 计算频率网格
        freqs = torch.zeros((D, D, 2))
        x = torch.linspace(-0.5, 0.5, D, endpoint=False)
        y = torch.linspace(-0.5, 0.5, D, endpoint=False)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        freqs = torch.stack([xx, yy], -1) / Apix
        freqs = freqs.reshape(-1, 2)
        
        # 计算CTF
        ctf_params_torch = torch.tensor(params[2:9].astype(np.float32))
        ctf = compute_ctf(freqs, *ctf_params_torch.unbind())
        ctf_2d = ctf.reshape(D, D).detach().numpy()
        
        # 绘制CTF
        plt.subplot(2, 3, i+1)
        plt.imshow(ctf_2d, cmap='gray')
        plt.title(f'样本 #{idx}\nDefU={params[2]:.0f}Å, DefV={params[3]:.0f}Å')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ctf_examples.png'))
    print(f"已保存CTF示例图到 {os.path.join(output_dir, 'ctf_examples.png')}")
    plt.close()

def display_ctf_page(ctf_params, page_num, page_size=10):
    """显示CTF参数的一页数据"""
    start_idx = page_num * page_size
    end_idx = min(start_idx + page_size, len(ctf_params))
    
    total_pages = (len(ctf_params) + page_size - 1) // page_size
    
    print(f"\n=== 显示CTF参数 (第{page_num+1}页/共{total_pages}页) ===")
    print(f"正在显示 {start_idx} - {end_idx-1} 组 (共 {len(ctf_params)} 组)\n")
    
    column_names = ["尺寸D", "像素大小(Å/px)", "DefocusU(Å)", "DefocusV(Å)", 
                    "散焦角度(°)", "电压(kV)", "球差Cs(mm)", "振幅对比度比w", "相位偏移(°)"]
    
    # 打印列名
    header = "序号".ljust(6)
    for name in column_names:
        header += name.ljust(15)
    print(header)
    
    # 打印当前页的数据
    for i in range(start_idx, end_idx):
        row = f"{i}".ljust(6)
        for j, val in enumerate(ctf_params[i]):
            if j == 0:  # 尺寸通常是整数
                row += f"{int(val)}".ljust(15)
            else:
                row += f"{val:.6f}".ljust(15)
        print(row)
    
    print("\n操作指令: [0-9]选择参数组 | n-下一页 | p-上一页 | q-退出")
    return total_pages

def select_ctf_mode(ctf_params, output_pkl=None):
    """选择CTF参数模式"""
    page_size = 10
    current_page = 0
    total_pages = (len(ctf_params) + page_size - 1) // page_size
    
    while True:
        total_pages = display_ctf_page(ctf_params, current_page, page_size)
        choice = input("请输入您的选择: ").strip().lower()
        
        if choice == 'q':
            print("退出选择模式")
            return None
        elif choice == 'n':
            if current_page < total_pages - 1:
                current_page += 1
            else:
                print("已经是最后一页了")
        elif choice == 'p':
            if current_page > 0:
                current_page -= 1
            else:
                print("已经是第一页了")
        elif choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(ctf_params):
                print(f"\n您选择了参数组 #{idx}:")
                # 显示选中的参数
                column_names = ["尺寸D", "像素大小(Å/px)", "DefocusU(Å)", "DefocusV(Å)", 
                            "散焦角度(°)", "电压(kV)", "球差Cs(mm)", "振幅对比度比w", "相位偏移(°)"]
                for j, val in enumerate(ctf_params[idx]):
                    print(f"{column_names[j]}: {val}")
                
                # 确认选择
                confirm = input("\n确认使用这组参数? (y/n): ").strip().lower()
                if confirm == 'y':
                    if output_pkl:
                        # 保存选中的单组参数到文件
                        selected_params = ctf_params[idx:idx+1]  # 只保留选中的一组
                        with open(output_pkl, 'wb') as f:
                            pickle.dump(selected_params, f)
                        print(f"\n已将选中的参数保存到: {output_pkl}")
                        print(f"\n使用方法: 将此文件用作add_ctf.py的--ctf-pkl参数")
                        print(f"示例: python add_ctf.py your_particles.mrcs --ctf-pkl {output_pkl} -o output.mrcs")
                    return ctf_params[idx:idx+1]
            else:
                print(f"无效的选择，请输入0-{len(ctf_params)-1}之间的数字")
        else:
            print("无效的输入，请重试")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CTF参数文件分析和选择工具")
    parser.add_argument("--pkl", type=str, 
                        default=r"C:\Shiina_Chizuru\PDB_multi_view_ParticleGenerated\my_code\9076_ctfs.pkl", 
                        help="CTF参数文件路径")
    parser.add_argument("--mode", type=str, choices=["analyze", "select"], default="analyze",
                        help="运行模式: analyze-分析模式, select-选择模式")
    parser.add_argument("--output", type=str, default=None,
                        help="选择模式下，保存选中CTF参数的输出文件路径")
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    pkl_file = args.pkl
    
    # 加载CTF参数
    print(f"正在加载CTF参数文件: {pkl_file}")
    ctf_params = load_ctf_pkl(pkl_file)
    
    # 创建输出目录
    output_dir = os.path.dirname(pkl_file)
    
    # 根据模式执行不同操作
    if args.mode == "analyze":
        # 分析CTF参数
        print("\n=== CTF参数分析 ===")
        analyze_ctf_params(ctf_params)
        
        # 绘制散焦分布
        print("\n绘制散焦分布...")
        plot_defocus_distribution(ctf_params, output_dir)
        
        # 绘制CTF示例
        print("\n绘制CTF示例...")
        try:
            plot_ctf_examples(ctf_params, output_dir)
        except Exception as e:
            print(f"绘制CTF示例时出错: {e}")
        
        print("\n分析完成！")
    
    elif args.mode == "select":
        # 选择CTF参数模式
        output_pkl = args.output if args.output else os.path.join(output_dir, "selected_ctf.pkl")
        selected_params = select_ctf_mode(ctf_params, output_pkl)
        if selected_params is not None:
            print("\n选择完成！")

if __name__ == "__main__":
    main()
