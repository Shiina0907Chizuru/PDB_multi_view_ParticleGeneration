#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量处理PDB文件，自动执行从PDB到多视图粒子图像的完整流程
"""

import os
import sys
import glob
import argparse
import subprocess
import time
from pathlib import Path
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="批量处理PDB文件生成多视图粒子图像")
    
    # 基础目录参数
    parser.add_argument("--pdb-dir", type=str, required=True, help="PDB文件所在目录")
    parser.add_argument("--output-dir", type=str, required=True, help="主输出目录")
    
    # 采样方式
    parser.add_argument("--sampling-mode", choices=["random", "uniform"], default="uniform",
                       help="采样方式: random(随机采样)或uniform(均匀采样)")
    
    # ChimeraX路径
    parser.add_argument("--chimerax-path", default="D:\\Program Files\\ChimeraX 1.8\\bin\\ChimeraX.exe", 
                       help="ChimeraX或者Chimera可执行文件路径")
    
    # Step 1参数: PDB转MRC
    parser.add_argument("--Apix", type=float, default=1.5, help="体积的像素大小(Å/pixel)")
    parser.add_argument("--D", type=int, default=256, help="体积的盒子大小")
    parser.add_argument("--res", type=float, default=3.0, help="模拟密度的分辨率(Å)")
    parser.add_argument("--use-python", action="store_true", help="使用纯Python方法生成MRC文件，无需ChimeraX/Chimera")
    
    # Step 2参数: 生成均匀姿态（仅uniform模式使用）
    parser.add_argument("--angle-step", type=float, default=30.0, help="角度采样步长(度)")
    parser.add_argument("--generate-csv", action="store_true", help="是否生成姿态的CSV文件")
    parser.add_argument("--skip-alpha", action="store_true", 
                       help="是否跳过alpha角采样(固定为0度，适用于二维图像旋转)")
    
    # Step 3参数: 3D投影
    parser.add_argument("--N", type=int, default=1000, 
                       help="随机采样模式下的投影数量（uniform模式下由角度步长决定）")
    parser.add_argument("--t-extent", type=float, default=5.0, help="像素平移范围")
    parser.add_argument("--batch-size", type=int, default=100, help="投影生成的批量大小，较大的值可加快处理速度，但可能占用更多内存 (默认: %(default)s)")
    
    # Step 5参数: 添加CTF
    parser.add_argument("--ctf-pkl", type=str, required=True, help="CTF参数文件路径")
    parser.add_argument("--snr1", type=float, default=20.0, help="CTF前噪声的信噪比")
    parser.add_argument("--snr2", type=float, default=10.0, help="CTF后噪声的信噪比")
    parser.add_argument("--no-noise", action="store_true", help="设置此参数时不添加任何噪声，只应用CTF")
    
    return parser.parse_args()


def create_directory_structure(output_dir, args):
    """创建输出目录结构"""
    # 为每种产物创建子目录
    subdirs = {
        "mrc": os.path.join(output_dir, "mrc"),        # 密度图
        "poses": os.path.join(output_dir, "poses"),    # 姿态文件
        "projections": os.path.join(output_dir, "projections"),  # 投影图像
        "particles": os.path.join(output_dir, "particles"),      # 带CTF的粒子
    }
    
    # 如果需要生成CSV文件，创建CSV目录
    if args.generate_csv:
        subdirs["csv"] = os.path.join(output_dir, "csv")
    
    # 创建目录
    for dir_name, dir_path in subdirs.items():
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    return subdirs


def step1_pdb_to_mrc(pdb_file, output_mrc_dir, args):
    """步骤1: 将PDB转换为MRC密度图"""
    pdb_basename = os.path.splitext(os.path.basename(pdb_file))[0]
    output_mrc_subdir = os.path.join(output_mrc_dir, pdb_basename)
    os.makedirs(output_mrc_subdir, exist_ok=True)
    
    # 构建命令
    cmd = [
        "python", "pdb2mrc.py",
        pdb_file,  # PDB文件
        pdb_file,  # 轨迹文件（使用同一文件）
        "1",       # 只生成一个模型
        "--Apix", str(args.Apix),
        "-D", str(args.D),
        "--res", str(args.res),
        "-o", output_mrc_subdir
    ]
    
    # 如果使用纯Python方法，添加--use-python参数
    if args.use_python:
        cmd.append("--use-python")
    else:
        cmd.extend(["-c", args.chimerax_path])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True)
        
        # 返回生成的MRC文件路径
        mrc_file = os.path.join(output_mrc_subdir, "vol_00000.mrc")
        if os.path.exists(mrc_file):
            print(f"成功生成MRC文件: {mrc_file}")
            return mrc_file
        else:
            print(f"警告: 未找到生成的MRC文件 {mrc_file}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"错误: PDB转MRC过程中出错: {e}")
        return None


def step2_generate_poses(pdb_basename, output_poses_dir, args):
    """步骤2: 生成均匀采样的姿态信息"""
    # 仅在均匀采样模式下执行
    if args.sampling_mode != "uniform":
        return None
    
    poses_pkl = os.path.join(output_poses_dir, f"{pdb_basename}_poses.pkl")
    
    # 可选的CSV输出
    csv_output = ""
    if args.generate_csv:
        csv_dir = os.path.join(args.output_dir, "csv")
        csv_file = os.path.join(csv_dir, f"{pdb_basename}_poses.csv")
        csv_output = f"--csv {csv_file}"
    
    # 是否跳过alpha角采样
    skip_alpha = "--skip-alpha" if args.skip_alpha else ""
    
    # 构建命令
    cmd = [
        "python", "generate_uniform_poses.py",
        "-o", poses_pkl,
        "-s", str(args.angle_step)
    ]
    
    if args.generate_csv:
        cmd.extend(["--csv", os.path.join(args.output_dir, "csv", f"{pdb_basename}_poses.csv")])
    
    if args.skip_alpha:
        cmd.append("--skip-alpha")
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True)
        
        if os.path.exists(poses_pkl):
            print(f"成功生成姿态文件: {poses_pkl}")
            return poses_pkl
        else:
            print(f"警告: 未找到生成的姿态文件 {poses_pkl}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"错误: 生成姿态过程中出错: {e}")
        return None


def step3_project_3d(mrc_file, poses_file, output_proj_dir, pdb_basename, args):
    """步骤3: 应用姿态生成2D投影图像"""
    mrcs_file = os.path.join(output_proj_dir, f"{pdb_basename}_projections.mrcs")
    out_pose_file = os.path.join(output_proj_dir, f"{pdb_basename}_proj_poses.pkl")
    
    # 构建基本命令
    cmd = [
        "python", "project3d.py",
        mrc_file,
        "-o", mrcs_file,
        "--out-pose", out_pose_file,
    ]
    
    # 添加采样方式相关参数
    if args.sampling_mode == "uniform" and poses_file:
        cmd.extend(["--in-pose", poses_file])
    else:
        # 随机采样模式
        cmd.extend(["-N", str(args.N)])
    
    # 添加平移范围
    cmd.extend(["--t-extent", str(args.t_extent)])
    
    # 添加批量大小参数
    cmd.extend(["-b", str(args.batch_size)])
    
    # 可选的CSV输出
    if args.generate_csv:
        csv_file = os.path.join(args.output_dir, "csv", f"{pdb_basename}_projections.csv")
        cmd.extend(["--csv", csv_file])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True)
        
        if os.path.exists(mrcs_file):
            print(f"成功生成投影文件: {mrcs_file}")
            return mrcs_file, out_pose_file
        else:
            print(f"警告: 未找到生成的投影文件 {mrcs_file}")
            return None, None
    except subprocess.CalledProcessError as e:
        print(f"错误: 生成投影过程中出错: {e}")
        return None, None


def step5_add_ctf(mrcs_file, ctf_pkl, output_particles_dir, pdb_basename, args, pose_file=None):
    """步骤5: 将CTF影响添加到投影图像中"""
    # 设置输出文件路径
    output_file = os.path.join(output_particles_dir, f"{pdb_basename}_with_ctf.mrcs")
    output_star = os.path.join(output_particles_dir, f"{pdb_basename}_with_ctf.star")
    output_pkl = os.path.join(output_particles_dir, f"{pdb_basename}_with_ctf.pkl")
    
    # 构建命令
    cmd = [
        "python", "add_ctf.py",
        mrcs_file,
        "--ctf-pkl", ctf_pkl,
        "-o", output_file,
        "--out-star", output_star,
        "--out-pkl", output_pkl,
        "--snr1", str(args.snr1),
        "--snr2", str(args.snr2),
        "--Apix", str(args.Apix)
    ]
    
    # 如果提供了位姿文件，添加到命令中
    if pose_file and os.path.exists(pose_file):
        print(f"将使用位姿文件: {pose_file} 添加到STAR文件中")
        cmd.extend(["--pose-pkl", pose_file])
    
    # 如果设置了不添加噪声选项
    if args.no_noise:
        cmd.append("--no-noise")
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True)
        
        if os.path.exists(output_file):
            print(f"成功生成带CTF的粒子文件: {output_file}")
            return output_file, output_star, output_pkl
        else:
            print(f"警告: 未找到生成的粒子文件 {output_file}")
            return None, None, None
    except subprocess.CalledProcessError as e:
        print(f"错误: 添加CTF过程中出错: {e}")
        return None, None, None


def process_single_pdb(pdb_file, dirs, args):
    """处理单个PDB文件的完整流程"""
    pdb_basename = os.path.splitext(os.path.basename(pdb_file))[0]
    print(f"\n========== 开始处理 {pdb_basename} ==========")
    
    # Step 1: PDB转MRC
    print(f"\n--- 步骤1: 将PDB转换为MRC密度图 ---")
    mrc_file = step1_pdb_to_mrc(pdb_file, dirs["mrc"], args)
    if not mrc_file:
        print(f"错误: 无法为 {pdb_basename} 生成MRC文件，跳过后续处理")
        return False
    
    # Step 2: 生成均匀姿态（仅uniform模式）
    poses_file = None
    if args.sampling_mode == "uniform":
        print(f"\n--- 步骤2: 生成均匀采样的姿态信息 ---")
        poses_file = step2_generate_poses(pdb_basename, dirs["poses"], args)
        if not poses_file and args.sampling_mode == "uniform":
            print(f"错误: 无法为 {pdb_basename} 生成姿态文件，跳过后续处理")
            return False
    else:
        print(f"\n--- 步骤2: 采用随机采样模式，跳过生成均匀姿态 ---")
    
    # Step 3: 生成投影图像
    print(f"\n--- 步骤3: 应用姿态生成2D投影图像 ---")
    mrcs_file, out_pose_file = step3_project_3d(mrc_file, poses_file, dirs["projections"], pdb_basename, args)
    if not mrcs_file:
        print(f"错误: 无法为 {pdb_basename} 生成投影文件，跳过后续处理")
        return False
    
    # Step 4: 使用用户提供的CTF参数文件
    print(f"\n--- 步骤4: 使用提供的CTF参数文件 ---")
    if not os.path.exists(args.ctf_pkl):
        print(f"错误: 指定的CTF参数文件不存在: {args.ctf_pkl}")
        return False
    print(f"将使用CTF参数文件: {args.ctf_pkl}")
    
    # Step 5: 添加CTF影响
    print(f"\n--- 步骤5: 将CTF影响添加到投影图像 ---")
    final_file, star_file, pkl_file = step5_add_ctf(mrcs_file, args.ctf_pkl, dirs["particles"], pdb_basename, args, out_pose_file)
    if not final_file:
        print(f"错误: 无法为 {pdb_basename} 添加CTF影响")
        return False
    
    print(f"生成的文件：")
    print(f"  - 粒子图像：{final_file}")
    print(f"  - CTF参数（STAR）：{star_file}")
    print(f"  - CTF参数（PKL）：{pkl_file}")
    
    print(f"\n========== 成功完成 {pdb_basename} 的处理 ==========")
    return True


def main():
    """主函数，批量处理PDB文件"""
    args = parse_args()
    
    # 验证输入目录
    if not os.path.isdir(args.pdb_dir):
        print(f"错误: PDB目录不存在: {args.pdb_dir}")
        return 1
    
    # 验证CTF参数文件
    if not os.path.exists(args.ctf_pkl):
        print(f"错误: CTF参数文件不存在: {args.ctf_pkl}")
        return 1
    
    # 创建输出目录结构
    print(f"创建输出目录结构: {args.output_dir}")
    dirs = create_directory_structure(args.output_dir, args)
    
    # 扫描PDB文件(.pdb和.ent格式)
    pdb_files = glob.glob(os.path.join(args.pdb_dir, "*.pdb"))
    ent_files = glob.glob(os.path.join(args.pdb_dir, "*.ent"))
    structure_files = pdb_files + ent_files
    
    if not structure_files:
        print(f"错误: 在 {args.pdb_dir} 中未找到PDB或ENT文件")
        return 1
    
    print(f"找到 {len(structure_files)} 个结构文件需要处理（PDB：{len(pdb_files)}，ENT：{len(ent_files)}）")
    
    # 记录成功和失败的文件
    success_files = []
    failed_files = []
    
    # 处理每个结构文件
    for i, pdb_file in enumerate(structure_files):
        print(f"\n处理第 {i+1}/{len(structure_files)} 个文件: {os.path.basename(pdb_file)}")
        
        success = process_single_pdb(pdb_file, dirs, args)
        if success:
            success_files.append(pdb_file)
        else:
            failed_files.append(pdb_file)
    
    # 输出处理结果统计
    print("\n========== 处理完成 ==========")
    print(f"总共处理: {len(structure_files)} 个文件（PDB：{len(pdb_files)}，ENT：{len(ent_files)}）")
    print(f"成功: {len(success_files)} 个文件")
    print(f"失败: {len(failed_files)} 个文件")
    
    if failed_files:
        print("\n失败的文件:")
        for f in failed_files:
            print(f"  - {os.path.basename(f)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())