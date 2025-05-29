"""
在Windows环境下将PDB文件转换为MRC体积数据
基于CryoBench/cryosim的pdb2mrc.py脚本修改
"""

import os
import argparse
import subprocess
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime

import sys

basement_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'basement')
if basement_dir not in sys.path:
    sys.path.append(basement_dir)

from basement import mrcfile

CHUNK = 10000  # 定期重启ChimeraX会话以避免内存溢出

def parse_args():
    parser = argparse.ArgumentParser(description='从原子模型生成MRC体积数据 - Windows版本')
    parser.add_argument('pdb', help='PDB文件路径')
    parser.add_argument('traj', help='轨迹文件路径（可选，如果只有单一结构，可与PDB文件使用相同路径）')
    parser.add_argument('num_models', type=int, help='要生成体积的结构数量')
    parser.add_argument('--Apix', type=float, default=1.5, help='体积的像素大小（默认：1.5）')
    parser.add_argument('-D', type=int, default=256, help='体积的盒子大小（默认：256）')
    parser.add_argument('--res', type=float, default=3.0, help='模拟密度的分辨率（默认：3.0）')
    parser.add_argument('-c', default='D:\\Program Files\\ChimeraX 1.8\\bin\\ChimeraX.exe', help='ChimeraX可执行文件路径（默认：D:\\Program Files\\ChimeraX 1.8\\bin\\ChimeraX.exe）')
    parser.add_argument('-o', required=True, help='存储体积的输出目录')
    parser.add_argument('--create-montage', action='store_true', help='将所有MRC文件拼接成一个完整的MRC用于可视化')
    parser.add_argument('--montage-layout', type=str, default='grid', choices=['grid', 'line'], help='拼接布局方式：grid(网格) 或 line(线性)')
    parser.add_argument('--montage-spacing', type=int, default=10, help='拼接时各体积之间的间距（像素）')
    parser.add_argument('--debug', action='store_true', help='显示调试信息')
    return parser.parse_args()


def is_chimerax(executable_path):
    """判断可执行文件是否为ChimeraX"""
    base_name = os.path.basename(executable_path).lower()
    return 'chimerax' in base_name

class CommandFile:
    def __init__(self, is_chimerax_mode=True, debug=False):
        self.commands = []
        self.is_chimerax = is_chimerax_mode
        self.debug = debug

    def add(self, command_chimerax, command_chimera=None):
        """添加命令，根据模式选择适当的命令格式
        
        参数:
            command_chimerax: ChimeraX格式的命令
            command_chimera: Chimera格式的命令，如果为None则使用与ChimeraX相同的命令
        """
        if command_chimera is None:
            command_chimera = command_chimerax
            
        if self.is_chimerax:
            self.commands.append(command_chimerax)
        else:
            self.commands.append(command_chimera)

    def save(self, file_path):
        """保存命令到文件
        
        Parameters
        ----------
        file_path : str
            要保存的文件路径
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as file:
            if self.is_chimerax:
                # ChimeraX使用纯文本命令格式
                for cmd in self.commands:
                    file.write(f"{cmd}\n")
            else:
                # 传统Chimera需要Python脚本格式
                file.write("from chimera import runCommand as rc\n\n")
                for cmd in self.commands:
                    file.write(f'rc("{cmd}")\n')
        
        if self.debug:
            with open(file_path, "r") as file:
                content = file.read()
                print("\n===== 命令文件内容 =====")
                print(f"文件: {file_path}")
                print(content)
                print("========================\n")

    def execute(self, executable_path, script_path):
        """执行命令并将完整输出保存到日志文件"""
        self.save(script_path)
        executable_path = os.path.normpath(executable_path)
        script_path = os.path.normpath(script_path)
        log_dir = os.path.dirname(script_path)
        log_basename = os.path.basename(script_path).split('.')[0]
        log_file = os.path.join(log_dir, f"{log_basename}_log.txt")
        
        def log_output(message, to_console=True):
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
            if to_console and self.debug:
                print(message)
        
        log_output(f"\n\n===== 命令执行日志 =====")
        log_output(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_output(f"脚本文件: {script_path}")
        
        try:
            if self.is_chimerax:
                command = [executable_path, "--nogui", script_path]
            else:
                command = [executable_path, "--nogui", "--script", script_path]
            
            command_str = ' '.join(command)
            log_output(f"执行命令: {command_str}")
            log_output("\n----- 脚本内容 -----")
            with open(script_path, 'r') as f:
                script_content = f.read()
            log_output(script_content)
            log_output("----- 脚本内容结束 -----\n")
            
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            
            log_output(f"\n返回码: {result.returncode}")
            
            if result.returncode == 0:
                if self.debug:
                    print("命令执行成功")
            else:
                print(f"\n命令执行失败，返回码: {result.returncode}")
            
            log_output("\n----- 标准输出(STDOUT) -----")
            if result.stdout:
                log_output(result.stdout, to_console=False)
                if self.debug:
                    print(f"输出: {result.stdout[:200]}...")
            else:
                log_output("[无标准输出]")
            log_output("----- 标准输出结束 -----\n")
            
            log_output("\n----- 错误输出(STDERR) -----")
            if result.stderr:
                log_output(result.stderr, to_console=False)
                if self.debug:
                    print(f"错误输出: {result.stderr[:200]}...")
                else:
                    print("检测到错误输出，请使用--debug参数查看详细信息")
            else:
                log_output("[无错误输出]")
            log_output("----- 错误输出结束 -----\n")
            
            if result.returncode != 0:
                error_msg = f"错误: {'ChimeraX' if self.is_chimerax else 'Chimera'}返回了非零状态码 {result.returncode}"
                log_output(error_msg)
                print(error_msg)
            else:
                success_msg = "命令执行成功"
                log_output(success_msg)
                if self.debug:
                    print(success_msg)
            
            if self.debug:
                print(f"完整日志已保存到: {log_file}")
            
            return result.returncode == 0
            
        except subprocess.CalledProcessError as e:
            error_msg = f"错误: {e}"
            log_output(error_msg)
            print(error_msg)
            return False
        except Exception as e:
            error_msg = f"意外错误: {e}"
            log_output(error_msg)
            print(error_msg)
            import traceback
            tb = traceback.format_exc()
            log_output("\n----- 异常调用堆栈 -----")
            log_output(tb)
            log_output("----- 异常调用堆栈结束 -----\n")
            return False
class CXCFile(CommandFile):
    def __init__(self):
        super().__init__(is_chimerax_mode=True)
        
    def add(self, command: str):
        """添加ChimeraX命令"""
        super().add(command)
        
    def execute(self, chimerax_path, cxc_path):
        """执行ChimeraX命令"""
        return super().execute(chimerax_path, cxc_path)


def pad_vol(path, Apix, D, debug=False):
    """填充体积到指定大小"""
    if debug:
        print(f"\n[DEBUG][pad_vol] 开始为{path}文件填充体积")
    
    if not os.path.exists(path):
        if debug:
            print(f"[DEBUG][pad_vol] 错误: 文件{path}不存在")
        return
    
    try:
        data, header = mrcfile.parse_mrc(path)
        if debug:
            print(f"[DEBUG][pad_vol] 原始体积大小: {data.shape}")
        
        x, y, z = data.shape    
        new_data = np.zeros((D, D, D), dtype=np.float32)    
        i, j, k = (D-x)//2, (D-y)//2, (D-z)//2
        if debug:
            print(f"[DEBUG][pad_vol] 填充偏移量: i={i}, j={j}, k={k}")
        
        new_data[i:(i+x), j:(j+y), k:(k+z)] = data
        orig_x, orig_y, orig_z = header.origin
        if debug:
            print(f"[DEBUG][pad_vol] 原始原点: ({orig_x}, {orig_y}, {orig_z})")
        
        new_header = mrcfile.get_mrc_header(
            new_data, True, 
            Apix=Apix, 
            xorg=(orig_x-k*Apix), 
            yorg=(orig_y-j*Apix), 
            zorg=(orig_z-i*Apix)
        )
        
        if debug:
            print(f"[DEBUG][pad_vol] 填充后体积大小: {new_data.shape}")
        mrcfile.write_mrc(path, new_data, new_header)
        if debug:
            print(f"[DEBUG][pad_vol] 已保存填充后的体积到: {path}")
    except Exception as e:
        if debug:
            print(f"[DEBUG][pad_vol] 错误: {e}")
        else:
            print(f"处理文件 {path} 时出错：{e}")


def center_all_vols(num_models, outdir):
    """居中所有体积的原点"""
    for i in range(num_models):
        path = os.path.join(outdir, f'vol_{i:05d}.mrc')
        data, header = mrcfile.parse_mrc(path)
        header.origin = (0., 0., 0.)
        mrcfile.write_mrc(path, data, header)


def generate_ref_vol(pdb_path, outdir, executable_path, res, Apix, D, debug=False):
    """生成参考体积"""
    is_cx = is_chimerax(executable_path)
    cmd = CommandFile(is_chimerax_mode=is_cx, debug=debug)
    pdb_abs_path = os.path.normpath(os.path.abspath(pdb_path))
    out_abs_path = os.path.normpath(os.path.abspath(os.path.join(outdir, 'ref.mrc')))
    if is_cx:
        # ChimeraX命令
        cmd.add(f"open \"{pdb_abs_path}\"")
        cmd.add(f"molmap #1 {res} gridSpacing {Apix}")
        cmd.add(f"save \"{out_abs_path}\" #2")
        cmd.add("exit")
    else:
        # 传统Chimera命令
        cmd.add(f"open {pdb_abs_path}")
        cmd.add(f"molmap #0 {res} gridSpacing {Apix}")
        cmd.add(f"volume #0.1 save {out_abs_path}")
        cmd.add("stop")
    
    script_ext = ".cxc" if is_cx else ".com"
    cmd_path = os.path.normpath(os.path.abspath(os.path.join(outdir, f'commands{script_ext}')))
    
    if debug:
        print(f"使用{'ChimeraX' if is_cx else 'Chimera'}生成参考体积...")
    cmd.execute(executable_path, cmd_path)
    
    if os.path.exists(out_abs_path):
        pad_vol(out_abs_path, Apix, D)
    else:
        print(f"\n\n错误: 没有找到输出文件 {out_abs_path}\n")


def generate_all_vols(pdb_path, traj_path, num_models, outdir, executable_path, res, Apix, D, debug=False):
    """生成所有体积数据"""
    is_cx = is_chimerax(executable_path)
    
    for start in range(0, num_models, CHUNK):
        cmd = CommandFile(is_chimerax_mode=is_cx, debug=debug)
        
        pdb_abs_path = os.path.normpath(os.path.abspath(pdb_path))
        traj_abs_path = os.path.normpath(os.path.abspath(traj_path))
        ref_abs_path = os.path.normpath(os.path.abspath(os.path.join(outdir, 'ref.mrc')))
        
        if is_cx:
            cmd.add(f"open \"{pdb_abs_path}\"")
            cmd.add(f"open \"{traj_abs_path}\"")
            cmd.add(f"open \"{ref_abs_path}\"")
            
            for i in range(start, min(start+CHUNK, num_models)):
                cmd.add(f"coordset #1 {i+1}")
                cmd.add(f"molmap #1 {res} gridSpacing {Apix}")
                cmd.add("vol resample #3 onGrid #2")
                vol_path = os.path.normpath(os.path.abspath(os.path.join(outdir, f'vol_{i:05d}.mrc')))
                cmd.add(f"save \"{vol_path}\" #4")
                cmd.add("close #3-4")
            
            cmd.add("exit")
        else:
            # 传统Chimera命令 - 移除引号并修正命令语法
            # 注意: Chimera的模型编号从0开始，而ChimeraX从1开始
            # 对于每个模型，我们需要单独打开、处理和关闭
            for i in range(start, min(start+CHUNK, num_models)):
                # 打开PDB文件
                cmd.add(f"open {pdb_abs_path}")
                # 设置坐标集
                cmd.add(f"coordset #0 {i+1}")
                # 生成体积
                cmd.add(f"molmap #0 {res} gridSpacing {Apix}")
                # 保存体积数据
                vol_path = os.path.normpath(os.path.abspath(os.path.join(outdir, f'vol_{i:05d}.mrc')))
                cmd.add(f"volume #0.1 save {vol_path}")
                # 关闭所有模型以便处理下一个
                cmd.add("close all")
            
            cmd.add("stop")
        
        script_ext = ".cxc" if is_cx else ".com"
        cmd_path = os.path.normpath(os.path.abspath(os.path.join(outdir, f'commands{script_ext}')))
        end = min(start + CHUNK, num_models)
        if debug:
            print(f"使用{'ChimeraX' if is_cx else 'Chimera'}处理模型 {start+1}-{end}...")
        cmd.execute(executable_path, cmd_path)
    
    print(f"对所有生成的体积进行填充处理，目标尺寸: {D}×{D}×{D}...")
    for i in range(num_models):
        vol_path = os.path.join(outdir, f'vol_{i:05d}.mrc')
        pad_vol(vol_path, Apix, D)
    
    center_all_vols(num_models, outdir)
    try:
        os.remove(os.path.join(outdir, 'ref.mrc'))
    except Exception as e:
        print(f"无法删除参考体积文件: {e}")



def create_mrc_montage(outdir, num_models, layout='grid', spacing=10, D=256, debug=False):
    """将多个MRC文件拼接成一个可视化用的MRC文件"""
    
    # 确定布局
    if layout == 'grid':
        grid_size = math.ceil(math.sqrt(num_models))
        rows, cols = grid_size, grid_size
    else:  # layout == 'line'
        rows, cols = 1, num_models
    
    montage_width = cols * D + (cols - 1) * spacing
    montage_height = rows * D + (rows - 1) * spacing
    
    montage = np.zeros((montage_height, montage_width, D), dtype=np.float32)
    
    for i in range(num_models):
        if layout == 'grid':
            row = i // cols
            col = i % cols
        else:  # layout == 'line'
            row = 0
            col = i
        
        y_start = row * (D + spacing)
        x_start = col * (D + spacing)
        
        file_path = os.path.join(outdir, f'vol_{i:05d}.mrc')
        if debug:
            print(f"\n[DEBUG][create_mrc_montage] 处理文件: {file_path}")
        
        if os.path.exists(file_path):
            data, _ = mrcfile.parse_mrc(file_path)
            if debug:
                print(f"[DEBUG][create_mrc_montage] 读取到体积大小: {data.shape}")
            
            projection = np.max(data, axis=0)
            if debug:
                print(f"[DEBUG][create_mrc_montage] 投影大小: {projection.shape}")
                print(f"[DEBUG][create_mrc_montage] 拼接区域大小: {D}x{D} 位于 ({y_start}:{y_start+D}, {x_start}:{x_start+D})")
            
            if projection.shape != (D, D):
                if debug:
                    print(f"[DEBUG][create_mrc_montage] 警告: 尺寸不匹配! 将调整投影尺寸")
                new_proj = np.zeros((D, D), dtype=np.float32)
                ph, pw = projection.shape
                h_offset = (D - ph) // 2
                w_offset = (D - pw) // 2
                new_proj[h_offset:h_offset+ph, w_offset:w_offset+pw] = projection
                projection = new_proj
                if debug:
                    print(f"[DEBUG][create_mrc_montage] 调整后投影大小: {projection.shape}")
            
            montage[y_start:y_start+D, x_start:x_start+D, 0] = projection
    
    montage_path = os.path.join(outdir, 'montage.mrc')
    header = mrcfile.get_mrc_header(montage, False)
    
    mrcfile.write_mrc(montage_path, montage, header)
    print(f"已创建拼接图: {montage_path}")
    
    plt.figure(figsize=(20, 20))
    plt.imshow(montage[:, :, 0], cmap='gray')
    plt.axis('off')
    png_path = os.path.join(outdir, 'montage.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已创建拼接图PNG预览: {png_path}")
    
    return montage_path, png_path


if __name__=="__main__":
    args = parse_args()
    
    os.makedirs(args.o, exist_ok=True)
    
    is_cx = is_chimerax(args.c)
    software_name = "ChimeraX" if is_cx else "UCSF Chimera"
    
    print(f"检测到软件: {software_name}")
    if args.debug:
        print(f"软件路径: {args.c}")
    
    if args.debug:
        print(f"开始使用{software_name}从PDB生成参考体积...")
    generate_ref_vol(args.pdb, args.o, args.c, args.res, args.Apix, args.D, debug=args.debug)
    
    if args.debug:
        print(f"开始使用{software_name}从轨迹生成所有体积，总数: {args.num_models}...")
    generate_all_vols(args.pdb, args.traj, args.num_models, args.o, args.c, args.res, args.Apix, args.D, debug=args.debug)
    
    if args.debug:
        print(f"对所有生成的体积进行填充处理，目标尺寸: {args.D}×{args.D}×{args.D}...")
    for i in range(args.num_models):
        vol_path = os.path.join(args.o, f'vol_{i:05d}.mrc')
        pad_vol(vol_path, args.Apix, args.D, debug=args.debug)
    
    if args.create_montage:
        if args.debug:
            print("开始创建MRC拼接图...")
        montage_path, png_path = create_mrc_montage(
            args.o, 
            args.num_models, 
            layout=args.montage_layout,
            spacing=args.montage_spacing,
            D=args.D,
            debug=args.debug
        )
        print(f"拼接图已保存至: {montage_path} 和 {png_path}")
    
    print(f"完成! 所有{args.num_models}个体积已保存到 {args.o}")

