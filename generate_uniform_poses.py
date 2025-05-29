"""
生成均匀采样的姿态文件，用于project3d.py
采样覆盖整个欧拉角空间 (360° × 360° × 180°)
"""

import numpy as np
import pickle
import os
import argparse
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except Exception as e:
    print(f"设置中文字体时出错: {e}，将使用英文显示")
from mpl_toolkits.mplot3d import Axes3D

def euler_to_rot(euler_angles):
    """
    将欧拉角（ZYZ约定）转换为旋转矩阵
    
    参数:
        euler_angles: 形状为 (n, 3) 的数组，包含 n 组欧拉角 (alpha, beta, gamma)
                      alpha和gamma范围为[0, 360]度，beta范围为[0, 180]度
    
    返回:
        shape (n, 3, 3) 的旋转矩阵数组
    """
    n = euler_angles.shape[0]
    R = np.zeros((n, 3, 3))
    
    for i in range(n):
        alpha, beta, gamma = euler_angles[i]
        
        # 转换为弧度
        alpha = np.radians(alpha)
        beta = np.radians(beta)
        gamma = np.radians(gamma)
        
        # ZYZ欧拉角到旋转矩阵的转换
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)
        
        R[i] = np.array([
            [ca*cb*cg - sa*sg, -ca*cb*sg - sa*cg, ca*sb],
            [sa*cb*cg + ca*sg, -sa*cb*sg + ca*cg, sa*sb],
            [-sb*cg, sb*sg, cb]
        ])
    
    return R

def generate_uniform_poses(angle_step, skip_alpha=False):
    """
    生成均匀采样的欧拉角，覆盖整个欧拉角空间
    
    参数:
        angle_step: 角度采样的步长（度）
        skip_alpha: 是否跳过alpha角度采样，若为True则alpha固定为0
    
    返回:
        euler_angles: 欧拉角数组
        rot_matrices: 对应的旋转矩阵数组
    """
    n_alpha = 1 if skip_alpha else int(np.ceil(360 / angle_step))
    n_beta = int(np.ceil(180 / angle_step))
    n_gamma = int(np.ceil(360 / angle_step))
    
    # 计算实际步长（确保完全覆盖）
    # 注意：对于alpha和gamma，我们确保360度空间是完全封闭的
    # 对于beta，我们需要覆盖0到180度的范围
    
    # 如果我们想要收尾相接，需要确保第一个点和最后一个点相差360度
    # 但在meshgrid中，如果包含了端点，就会重复计算0度和360度（它们在旋转意义上是相同的）
    # 因此对于alpha和gamma，我们采样[0, 360)范围（不包含360度）
    # 对于beta，我们采样[0, 180]范围（包含180度）
    
    alpha_step = 360 / n_alpha
    beta_step = 180 / n_beta
    gamma_step = 360 / n_gamma
    
    alphas = np.array([0.0]) if skip_alpha else np.linspace(0, 360 - alpha_step, n_alpha)  # [0, 360)范围
    if n_beta > 1:
        betas = np.linspace(0, 180 - beta_step, n_beta)  # [0, 180)范围
    else:
        betas = np.array([0])
    gammas = np.linspace(0, 360 - gamma_step, n_gamma)  # [0, 360)范围
    
    total_poses = n_alpha * n_beta * n_gamma
    print(f"采样步长: Alpha={alpha_step:.2f}°, Beta={beta_step:.2f}°, Gamma={gamma_step:.2f}°")
    print(f"Alpha: {n_alpha}个采样点 (0°-{360-alpha_step:.2f}°)")
    print(f"Beta: {n_beta}个采样点 (0°-{180-beta_step:.2f}°)")
    print(f"Gamma: {n_gamma}个采样点 (0°-{360-gamma_step:.2f}°)")
    print(f"总姿态数: {total_poses}")
    
    euler_angles = []
    
    for beta in betas:
        for alpha in alphas:
            for gamma in gammas:
                euler_angles.append([alpha, beta, gamma])
    
    euler_angles = np.array(euler_angles)
    
    print(f"前100个姿态角度示例:")
    for i in range(min(100, len(euler_angles))):
        print(f"#{i}: \u03b1={euler_angles[i,0]:.1f}\u00b0 \u03b2={euler_angles[i,1]:.1f}\u00b0 \u03b3={euler_angles[i,2]:.1f}\u00b0")
    
    rot_matrices = euler_to_rot(euler_angles)
    
    return euler_angles, rot_matrices

def visualize_euler_distribution(euler_angles, output_png):
    """可视化欧拉角分布"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    max_points = 1000
    if len(euler_angles) > max_points:
        indices = np.random.choice(len(euler_angles), max_points, replace=False)
        euler_angles_subset = euler_angles[indices]
    else:
        euler_angles_subset = euler_angles
    
    ax.scatter(
        euler_angles_subset[:, 0], 
        euler_angles_subset[:, 1], 
        euler_angles_subset[:, 2], 
        c='b', alpha=0.6, s=10
    )
    
    ax.set_xlabel('Alpha (°)')
    ax.set_ylabel('Beta (°)')
    ax.set_zlabel('Gamma (°)')
    
    total_points = len(euler_angles)
    shown_points = len(euler_angles_subset)
    ax.set_title(f'Euler Angle Distribution (Showing {shown_points} out of {total_points} points)')
    
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"分布图已保存到: {output_png}")

def save_poses_to_csv(euler_angles, rot_matrices, trans_vectors, output_csv):
    """将姿态信息保存为CSV格式
    
    参数:
        euler_angles: 欧拉角数组
        rot_matrices: 旋转矩阵数组
        trans_vectors: 平移向量数组
        output_csv: 输出文件路径
    """
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("姿态索引,Alpha(度),Beta(度),Gamma(度),R11,R12,R13,R21,R22,R23,R31,R32,R33,TransX,TransY\n")
        
        for i in range(len(euler_angles)):
            alpha, beta, gamma = euler_angles[i]
            R = rot_matrices[i]
            tx, ty = trans_vectors[i]
            
            line = f"{i},{alpha:.2f},{beta:.2f},{gamma:.2f},"
            line += f"{R[0,0]:.6f},{R[0,1]:.6f},{R[0,2]:.6f},"
            line += f"{R[1,0]:.6f},{R[1,1]:.6f},{R[1,2]:.6f},"
            line += f"{R[2,0]:.6f},{R[2,1]:.6f},{R[2,2]:.6f},"
            line += f"{tx:.6f},{ty:.6f}\n"
            
            f.write(line)
    
    print(f"姿态信息已保存到CSV文件: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description='生成均匀采样的姿态文件用于project3d.py')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出姿态文件路径(.pkl)')
    parser.add_argument('-s', '--step', type=float, default=30.0, help='角度采样步长(度)')
    parser.add_argument('--viz', type=str, default=None, help='可视化欧拉角分布的输出图片路径(.png)')
    parser.add_argument('--no-trans', action='store_true', help='不生成平移向量（默认生成零平移）')
    parser.add_argument('--csv', type=str, help='将姿态信息以CSV格式输出的文件路径(.csv)')
    parser.add_argument('--skip-alpha', action='store_true', help='跳过alpha角度采样，固定alpha为0度（二维图像旋转）')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.csv:
        csv_dir = os.path.dirname(args.csv)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
    
    euler_angles, rot_matrices = generate_uniform_poses(args.step, skip_alpha=args.skip_alpha)
    
    if args.viz:
        visualize_euler_distribution(euler_angles, args.viz)
    
    n_poses = rot_matrices.shape[0]
    if not args.no_trans:
        trans = np.zeros((n_poses, 2))
        poses = (rot_matrices, trans)
    else:
        poses = rot_matrices
        trans = np.zeros((n_poses, 2))
    
    trans = np.zeros((n_poses, 2))
    pose_data = {
        'rot_matrices': rot_matrices,
        'trans_vectors': trans,
        'angle_step': args.step,
        'euler_angles': euler_angles,
        'skip_alpha': args.skip_alpha
    }
    
    with open(args.output, 'wb') as f:
        pickle.dump(pose_data, f)
    
    print(f"已生成 {n_poses} 个姿态并保存到: {args.output}")
    print(f"角度步长信息 ({args.step:.2f}°) 已保存到PKL文件中")
    if args.skip_alpha:
        print(f"跳过alpha角度采样: 所有姿态的alpha值固定为0度")
    
    if args.csv:
        save_poses_to_csv(euler_angles, rot_matrices, trans, args.csv)

if __name__ == "__main__":
    main()
