"""
Corrupt particle images with structural noise, CTF, digital/shot noise
"""

import argparse
import numpy as np
import sys, os
import pickle
from datetime import datetime as dt
import matplotlib.pyplot as plt

import sys
import os

basement_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'basement')
if basement_dir not in sys.path:
    sys.path.append(basement_dir)

from basement.ctf import compute_ctf
from basement import mrcfile
import torch

log = print


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # 基本参数
    parser.add_argument("particles", help="输入的MRC堆栈文件，包含粒子图像数据")
    
    # 噪声相关参数
    parser.add_argument(
        "--snr1",
        default=20,
        type=float,
        help="应用CTF前添加的第一阶段噪声的信噪比，值越大噪声越小 (默认: %(default)s)",
    )
    parser.add_argument(
        "--snr2",
        default=10,
        type=float,
        help="应用CTF后添加的第二阶段噪声的信噪比，模拟数字/散粒噪声 (默认: %(default)s)",
    )
    parser.add_argument(
        "--s1", type=float, help="直接指定CTF前噪声的高斯标准差，覆盖--snr1参数"
    )
    parser.add_argument(
        "--s2", type=float, help="直接指定CTF后噪声的高斯标准差，覆盖--snr2参数"
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="设置此参数时，不添加任何噪声，只应用CTF",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机数种子，用于生成散焦值和噪声，确保结果可重复 (默认: %(default)s)",
    )
    
    # 输出文件参数
    parser.add_argument("-o", required=True, type=os.path.abspath, help="输出的MRC堆栈文件路径(.mrcs)")
    parser.add_argument(
        "--out-star",
        type=os.path.abspath,
        help="输出的STAR文件路径，用于存储CTF参数 (默认: [输出mrcs文件名].star)",
    )
    parser.add_argument(
        "--out-pkl",
        type=os.path.abspath,
        help="输出的Pickle文件路径，用于存储CTF参数 (默认: [输出mrcs文件名].pkl)",
    )
    parser.add_argument("--out-png", help="输出的PNG图像路径，用于保存处理后粒子的示例图")
    parser.add_argument(
        "--pose-pkl",
        type=os.path.abspath,
        help="位姿信息PKL文件路径，包含投影图像的欧拉角信息，将被写入STAR文件",
    )

    # CTF参数组
    group = parser.add_argument_group("CTF parameters")
    group.add_argument("--Apix", type=float, help="像素大小，单位为埃/像素(Å/pixel)，影响CTF计算的频率范围")
    group.add_argument(
        "--ctf-pkl", metavar="pkl", help="从指定的pkl文件加载CTF参数，文件格式与cryodrgn兼容，可通过analyze_ctf_pkl.py的选择模式生成"
    )
    group.add_argument(
        "--df-file",
        metavar="pkl",
        help="从指定的pkl文件加载散焦参数，格式为Nx2的numpy数组，包含每个图像的散焦U和散焦V值",
    )
    group.add_argument(
        "--kv",
        default=300,
        type=float,
        help="显微镜电压，单位为千伏特(kV)，影响CTF的形状 (默认: %(default)s)",
    )
    group.add_argument(
        "--dfu", default=15000, type=float, help="散焦U值，单位为埃(Å)，较大的值会产生更快的CTF振荡 (默认: %(default)s)"
    )
    group.add_argument(
        "--dfv", default=15000, type=float, help="散焦V值，单位为埃(Å)，与散焦U值不同时会产生像散 (默认: %(default)s)"
    )
    group.add_argument(
        "--ang",
        default=0,
        type=float,
        help="像散角度，单位为度，定义像散的方向 (默认: %(default)s)",
    )
    group.add_argument(
        "--cs",
        default=2,
        type=float,
        help="球差系数，单位为毫米(mm)，影响高频CTF行为 (默认: %(default)s)",
    )
    group.add_argument(
        "--wgh",
        default=0.1,
        type=float,
        help="振幅对比度比，控制CTF中振幅对比度与相位对比度的比例 (默认: %(default)s)",
    )
    group.add_argument(
        "--ps", default=0, type=float, help="相位偏移，单位为度，由相位板引入 (默认: %(default)s)"
    )
    group.add_argument(
        "-b",
        default=100,
        type=float,
        help="B因子，用于高斯包络函数，单位为埃的平方(Å²)，控制高频信息的衰减 (默认: %(default)s)",
    )
    group.add_argument(
        "--sample-df",
        type=float,
        help="散焦抖动标准差，为每个图像随机生成不同的散焦值，单位为埃(Å)，模拟实际数据采集中的散焦变化 (默认: None)",
    )
    group.add_argument(
        "--no-astigmatism",
        action="store_true",
        help="禁用像散，确保每个粒子的散焦U和散焦V值相同，即不产生像散效应",
    )
    return parser


# todo - switch to cryodrgn starfile api
def write_starfile(out, mrc, Nimg, df, ang, kv, wgh, cs, ps, metadata=None):
    header = [
        "data_images",
        "loop_",
        "_rlnImageName",
        "_rlnDefocusU",
        "_rlnDefocusV",
        "_rlnDefocusAngle",
        "_rlnVoltage",
        "_rlnAmplitudeContrast",
        "_rlnSphericalAberration",
        "_rlnPhaseShift",
    ]

    if metadata is not None:
        header.extend(["_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi", "_rlnOriginXAngst", "_rlnOriginYAngst\n"])
        metadata_data = pickle.load(open(metadata, "rb"))
        log(f"Loaded metadata from {metadata} with {len(metadata_data)} entries")
        
        # 检查元数据格式，适配不同的数据结构
        if isinstance(metadata_data, dict) and 'euler_angles' in metadata_data:
            # project3d.py生成的标准格式
            euler_angles = metadata_data['euler_angles']
            assert len(euler_angles) == Nimg, f"Mismatch in number of images: {Nimg} vs {len(euler_angles)}"
            metadata = euler_angles
        elif isinstance(metadata_data, list) and len(metadata_data) == Nimg:
            # 假设是直接的角度列表
            metadata = metadata_data
        else:
            # 尝试其他可能的格式
            log(f"Warning: Unknown metadata format, attempting to extract Euler angles")
            try:
                if isinstance(metadata_data, dict):
                    for key in metadata_data:
                        if isinstance(metadata_data[key], np.ndarray) and metadata_data[key].shape[0] == Nimg:
                            metadata = metadata_data[key]
                            break
                elif isinstance(metadata_data, np.ndarray) and metadata_data.shape[0] == Nimg:
                    metadata = metadata_data
                else:
                    log(f"Error: Could not extract Euler angles from metadata")
                    metadata = None
                    header[-1] += "\n"  # 添加新行
            except Exception as e:
                log(f"Error extracting Euler angles: {e}")
                metadata = None
                header[-1] += "\n"  # 添加新行
    else:
        header[-1] += "\n"
    lines = []
    filename = os.path.basename(mrc)
    for i in range(Nimg):
        line = [
            "{:06d}@{}".format(i + 1, filename),
            "{:1f}".format(df[i][0]),
            "{:1f}".format(df[i][1]),
            ang[i] if type(ang) in (list, np.ndarray) else ang,
            kv,
            wgh,
            cs,
            ps,
        ]
        if metadata is not None:
            # 添加欧拉角
            if len(metadata[i]) >= 3:
                line.extend(metadata[i][:3])  # 只取前三个值作为欧拉角
            else:
                line.extend(metadata[i])
            
            # 添加X,Y位移（默认为0）
            line.extend([0.0, 0.0])
        lines.append(" ".join([str(x) for x in line]))
    f = open(out, "w")
    f.write("# Created {}\n".format(dt.now()))
    f.write("\n".join(header))
    f.write("\n".join(lines))
    f.write("\n")


def add_noise(particles, D, sigma):
    particles += np.random.normal(0, sigma, particles.shape)
    return particles


def compute_full_ctf(D, Nimg, args):
    freqs = np.arange(-D / 2, D / 2) / (args.Apix * D)
    x0, x1 = np.meshgrid(freqs, freqs)
    
    # 检测是否有可用的CUDA设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"使用设备: {device}")
    
    freqs = torch.tensor(np.stack([x0.ravel(), x1.ravel()], axis=1)).to(device)
    
    if args.ctf_pkl:  # todo: refator
        params = torch.tensor(pickle.load(open(args.ctf_pkl, "rb"))).to(device)
        
        # 检查参数数量，如果只有一组参数但有多个图像，将参数复制到相同长度
        if len(params) == 1 and Nimg > 1:
            log(f"发现单组CTF参数，正在复制应用到所有{Nimg}个图像...")
            single_param = params[0]
            params = single_param.repeat(Nimg, 1)
        else:
            assert len(params) == Nimg, f"CTF参数数量({len(params)})与图像数量({Nimg})不匹配"
        
        params = params[:, 2:]
        df = params[:, :2]
        ctf = torch.stack([compute_ctf(freqs, *x, args.b) for x in params])
        print("ctf:", ctf.shape)
        ctf = ctf.reshape((Nimg, D, D))
    elif args.df_file:
        df = pickle.load(open(args.df_file, "rb"))
        assert len(df) == Nimg
        ctf = np.array(
            [
                compute_ctf(
                    freqs, i, j, args.ang, args.kv, args.cs, args.wgh, args.ps, args.b
                )
                for i, j in df
            ]
        )
        ctf = ctf.reshape((Nimg, D, D))
        # df = np.stack([df,df], axis=1)
    elif args.sample_df:
        df1 = np.random.normal(args.dfu, args.sample_df, Nimg)
        if args.no_astigmatism:
            assert args.dfv == args.dfu, "--dfu and --dfv must be the same"
            df2 = df1
        else:
            df2 = np.random.normal(args.dfv, args.sample_df, Nimg)
        ctf = np.array(
            [
                compute_ctf(
                    freqs, i, j, args.ang, args.kv, args.cs, args.wgh, args.ps, args.b
                )
                for i, j in zip(df1, df2)
            ]
        )
        ctf = ctf.reshape((Nimg, D, D))
        df = np.stack([df1, df2], axis=1)
    else:
        ctf = compute_ctf(
            freqs,
            args.dfu,
            args.dfv,
            args.ang,
            args.kv,
            args.cs,
            args.wgh,
            args.ps,
            args.b,
        )
        ctf = ctf.reshape((D, D))
        df = np.stack([np.ones(Nimg) * args.dfu, np.ones(Nimg) * args.dfv], axis=1)
    return ctf, df


def add_ctf(particles, ctf):
    particles = np.array(
        [np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x))) for x in particles]
    )
    # 如果ctf是PyTorch tensor且在GPU上，将其移到CPU
    if isinstance(ctf, torch.Tensor):
        ctf_np = ctf.cpu().numpy() if ctf.is_cuda else ctf.numpy()
    else:
        ctf_np = np.array(ctf)
    
    particles *= ctf_np
    del ctf, ctf_np
    particles = np.array(
        [
            np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(x))).astype(np.float32)
            for x in particles
        ]
    )
    return particles


def normalize(particles):
    mu, std = np.mean(particles), np.std(particles)
    particles -= mu
    particles /= std
    log("Shifting input images by {}".format(mu))
    log("Scaling input images by {}".format(std))
    return particles


def plot_projections(out_png, imgs):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    axes = axes.ravel()
    for i in range(min(len(imgs), 9)):
        axes[i].imshow(imgs[i])
    plt.savefig(out_png)


def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))


def warnexists(out):
    if os.path.exists(out):
        log("Warning: {} already exists. Overwriting.".format(out))


def main(args):
    np.random.seed(args.seed)
    log("RUN CMD:\n" + " ".join(sys.argv))
    log("Arguments:\n" + str(args))
    particles = mrcfile.parse_mrc(args.particles)[0]
    Nimg = len(particles)
    D, D2 = particles[0].shape
    assert D == D2, "Images must be square"

    log("Loaded {} images".format(Nimg))

    mkbasedir(args.o)
    warnexists(args.o)

    # if not args.rad: args.rad = D/2
    # x0, x1 = np.meshgrid(np.arange(-D/2,D/2),np.arange(-D/2,D/2))
    # mask = np.where((x0**2 + x1**2)**.5 < args.rad)

    # 检查是否设置了不添加噪声的选项
    if args.no_noise:
        log("已设置--no-noise参数，跳过所有噪声添加步骤，只应用CTF")
        s1 = 0
        s2 = 0
    elif args.s1 is not None:
        assert args.s2 is not None, "Need to provide both --s1 and --s2"
        s1 = args.s1
    else:
        Nstd = min(1000, Nimg)
        mask = np.where(particles[:Nstd] > 0)
        std = np.std(particles[mask])
        s1 = std / np.sqrt(args.snr1)
        
    if s1 > 0:
        log("添加CTF前噪声，标准差为 {}".format(s1))
        particles = add_noise(particles, D, s1)

    log("Applying the CTF")
    ctf, defocus_list = compute_full_ctf(D, Nimg, args)
    particles = add_ctf(particles, ctf)

    if args.no_noise:
        s2 = 0
    elif args.s2 is not None:
        s2 = args.s2
    elif not args.no_noise:
        std = np.std(particles[mask])
        try:
            # cascading of noise processes according to Frank and Al-Ali (1975) & Baxter (2009)
            snr2 = (1 + 1 / args.snr1) / (1 / args.snr2 - 1 / args.snr1)
            log("SNR2目标值 {} 最终总体信噪比 {}".format(snr2, args.snr2))
            s2 = std / np.sqrt(snr2)
        except ZeroDivisionError:
            log("警告: snr1和snr2值太接近，使用默认值")
            s2 = std * 0.1  # 使用默认噪声水平
    
    if s2 > 0:
        log("添加CTF后噪声，标准差为 {}".format(s2))
        particles = add_noise(particles, D, s2)

    log("Writing image stack to {}".format(args.o))
    mrcfile.write_mrc(args.o, particles.astype(np.float32))

    log("Writing png sample to {}".format(args.out_png))
    if args.out_png:
        plot_projections(args.out_png, particles[:9])

    if args.out_star is None:
        args.out_star = f"{args.o}.star"
    log(f"Writing associated .star file to {args.out_star}")
    
    # 检查是否提供了位姿文件
    if args.pose_pkl and os.path.exists(args.pose_pkl):
        log(f"Including pose information from {args.pose_pkl} in STAR file")
        write_starfile(
            args.out_star,
            args.o,
            Nimg,
            defocus_list,
            args.ang,
            args.kv,
            args.wgh,
            args.cs,
            args.ps,
            metadata=args.pose_pkl
        )
    else:
        write_starfile(
            args.out_star,
            args.o,
            Nimg,
            defocus_list,
            args.ang,
            args.kv,
            args.wgh,
            args.cs,
            args.ps,
        )

    if not args.ctf_pkl:
        if args.out_pkl is None:
            args.out_pkl = f"{args.o}.pkl"
        log(f"Writing CTF params pickle to {args.out_pkl}")
        params = np.ones((Nimg, 9), dtype=np.float32)
        params[:, 0] = D
        params[:, 1] = args.Apix
        params[:, 2:4] = defocus_list
        params[:, 4] = args.ang
        params[:, 5] = args.kv
        params[:, 6] = args.cs
        params[:, 7] = args.wgh
        params[:, 8] = args.ps
        log(params[0])
        with open(args.out_pkl, "wb") as f:
            pickle.dump(params, f)

    log("Done")


if __name__ == "__main__":
    main(parse_args().parse_args())
