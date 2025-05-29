"""
Generate projections of a 3D volume
"""

import argparse
import numpy as np
import sys, os
import time
import pickle
from scipy.ndimage import fourier_shift

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import sys
import os

basement_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'basement')
if basement_dir not in sys.path:
    sys.path.append(basement_dir)

from basement import utils, mrcfile, lie_tools, so3_grid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = print


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mrc", help="Input volume")
    parser.add_argument(
        "-o",
        type=os.path.abspath,
        required=True,
        help="Output projection stack (.mrcs)",
    )
    parser.add_argument(
        "--out-pose", type=os.path.abspath, required=True, help="Output poses (.pkl)"
    )
    parser.add_argument(
        "--out-png", type=os.path.abspath, help="Montage of first 9 projections"
    )
    parser.add_argument(
        "--in-pose",
        type=os.path.abspath,
        help="Optionally provide input poses instead of random poses (.pkl)",
    )
    parser.add_argument("-N", type=int, help="Number of random projections")
    parser.add_argument(
        "-b", type=int, default=100, help="Minibatch size (default: %(default)s)"
    )
    parser.add_argument(
        "--t-extent",
        type=float,
        default=5,
        help="Extent of image translation in pixels (default: +/-%(default)s)",
    )
    parser.add_argument(
        "--grid",
        type=int,
        help="Generate projections on a uniform deterministic grid on SO3. Specify resolution level",
    )
    parser.add_argument("--tilt", type=float, help="Right-handed x-axis tilt offset in degrees")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--csv", type=str, help="将每张投影图像对应的位姿信息保存为CSV格式的文件路径(.csv)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increaes verbosity"
    )
    return parser


class Projector:
    def __init__(self, vol, tilt=None):
        nz, ny, nx = vol.shape
        assert nz == ny == nx, "Volume must be cubic"
        x2, x1, x0 = np.meshgrid(
            np.linspace(-1, 1, nz, endpoint=True),
            np.linspace(-1, 1, ny, endpoint=True),
            np.linspace(-1, 1, nx, endpoint=True),
            indexing="ij",
        )

        lattice = np.stack([x0.ravel(), x1.ravel(), x2.ravel()], 1).astype(np.float32)
        self.lattice = torch.from_numpy(lattice)

        self.vol = torch.from_numpy(vol.astype(np.float32))
        self.vol = self.vol.unsqueeze(0)
        self.vol = self.vol.unsqueeze(0)

        self.nz = nz
        self.ny = ny
        self.nx = nx

        # FT is not symmetric around origin
        D = nz
        c = 2 / (D - 1) * (D / 2) - 1
        self.center = torch.tensor([c, c, c])  # pixel coordinate for vol[D/2,D/2,D/2]

        if tilt is not None:
            assert tilt.shape == (3, 3)
            tilt = torch.tensor(tilt)
        self.tilt = tilt

    def rotate(self, rot):
        B = rot.size(0)
        if self.tilt is not None:
            rot = self.tilt @ rot
        grid = self.lattice @ rot  # B x D^3 x 3
        grid = grid.view(-1, self.nz, self.ny, self.nx, 3)
        offset = (
            self.center - grid[:, int(self.nz / 2), int(self.ny / 2), int(self.nx / 2)]
        )
        grid += offset[:, None, None, None, :]
        grid = grid.view(1, -1, self.ny, self.nx, 3)
        vol = F.grid_sample(self.vol, grid)
        vol = vol.view(B, self.nz, self.ny, self.nx)
        return vol

    def project(self, rot):
        return self.rotate(rot).sum(dim=1)


class Poses(data.Dataset):
    def __init__(self, pose_pkl):
        poses = utils.load_pkl(pose_pkl)
        
        if isinstance(poses, dict):
            self.rots = torch.tensor(poses['rot_matrices']).float()
            self.trans = poses['trans_vectors']
            self.angle_step = poses.get('angle_step', 30.0)
            self.euler_angles = poses.get('euler_angles', None)
        else:
            if isinstance(poses, tuple) and len(poses) == 2:
                self.rots = torch.tensor(poses[0]).float()
                self.trans = poses[1]
            else:
                self.rots = torch.tensor(poses).float()
                self.trans = np.zeros((len(poses), 2))
            self.angle_step = 30.0
            self.euler_angles = None
        
        self.N = len(self.rots)
        assert self.rots.shape == (self.N, 3, 3)
        assert self.trans.shape == (self.N, 2)
        assert self.trans.max() < 1

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.rots[index]


class RandomRot(data.Dataset):
    def __init__(self, N):
        self.N = N
        self.rots = lie_tools.random_SO3(N)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.rots[index]


class GridRot(data.Dataset):
    def __init__(self, resol):
        quats = so3_grid.grid_SO3(resol)
        self.rots = lie_tools.quaternions_to_SO3(torch.tensor(quats))
        self.N = len(self.rots)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.rots[index]


def plot_projections(out_png, imgs):
    n = min(len(imgs), 9)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 10), dpi=100)
    fig.patch.set_facecolor('black')
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(imgs[i], cmap='gray')
            ax.set_facecolor('black')
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, facecolor='black', bbox_inches='tight', pad_inches=0)


def mkbasedir(out):
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))


def warnexists(out):
    if os.path.exists(out):
        log("Warning: {} already exists. Overwriting.".format(out))


def translate_img(img, t):
    """
    img: BxYxX real space image
    t: Bx2 shift in pixels
    """
    ff = np.fft.fft2(np.fft.fftshift(img))
    ff = fourier_shift(ff, t)
    return np.fft.fftshift(np.fft.ifft2(ff)).real


def rot_to_euler(R, angle_step=30.0):
    """
    从旋转矩阵提取欧拉角（ZYZ约定），优化处理万向节锁情况
    
    参数:
        R: 形状为 (3, 3) 的旋转矩阵
        angle_step: 角度步长（度），用于万向节锁情况下的处理
    
    返回:
        欧拉角 (alpha, beta, gamma)，单位为度
    """
    beta = np.arccos(max(-1, min(1, R[2, 2])))  
    beta_deg = np.degrees(beta)
    
    if np.isclose(abs(R[2, 2]), 1.0):
        # 当beta接近0或180度时，alpha和gamma存在万向节锁（Gimbal Lock）
        # 改进万向节锁处理：按照generate_uniform_poses.py的生成模式
        # 从旋转矩阵中提取α+γ的总角度
        angle_sum = np.arctan2(R[1, 0], R[0, 0])
        angle_sum_deg = (np.degrees(angle_sum) + 360) % 360
        
        gamma_max = 360 - angle_step
        
        gamma_steps = int(angle_sum_deg / angle_step)
        gamma_deg = (gamma_steps % int(360 / angle_step)) * angle_step
        
        alpha_steps = int(gamma_steps / (360 / angle_step))
        alpha_deg = alpha_steps * angle_step
        
        alpha_deg = alpha_deg % 360
        gamma_deg = gamma_deg % 360
    else:
        alpha = np.arctan2(R[1, 2], R[0, 2])
        gamma = np.arctan2(R[2, 1], -R[2, 0])
        
        alpha_deg = (np.degrees(alpha) + 360) % 360
        gamma_deg = (np.degrees(gamma) + 360) % 360
    
    return np.array([alpha_deg, beta_deg, gamma_deg])


def save_poses_to_csv(rot_matrices, trans_vectors, output_csv, angle_step=30.0, euler_angles_input=None):
    """将每张投影图像对应的位姿信息保存为CSV格式
    
    参数:
        rot_matrices: 旋转矩阵数组
        trans_vectors: 平移向量数组
        output_csv: 输出文件路径
        angle_step: 角度步长（度），用于万向节锁情况下的处理
        euler_angles_input: 可选的输入欧拉角，如果提供则直接使用
    """
    if euler_angles_input is not None:
        euler_angles = euler_angles_input
    else:
        euler_angles = []
        for R in rot_matrices:
            alpha, beta, gamma = rot_to_euler(R, angle_step)
            euler_angles.append((alpha, beta, gamma))
    
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("图像索引,Alpha(度),Beta(度),Gamma(度),R11,R12,R13,R21,R22,R23,R31,R32,R33,TransX,TransY\n")
        
        for i in range(len(rot_matrices)):
            alpha, beta, gamma = euler_angles[i]
            R = rot_matrices[i]
            if trans_vectors is not None:
                tx, ty = trans_vectors[i]
            else:
                tx, ty = 0.0, 0.0
            line = f"{i},{alpha:.2f},{beta:.2f},{gamma:.2f},"
            line += f"{R[0,0]:.6f},{R[0,1]:.6f},{R[0,2]:.6f},"
            line += f"{R[1,0]:.6f},{R[1,1]:.6f},{R[1,2]:.6f},"
            line += f"{R[2,0]:.6f},{R[2,1]:.6f},{R[2,2]:.6f},"
            line += f"{tx:.6f},{ty:.6f}\n"
            
            f.write(line)
    
    log(f"投影图像位姿信息已保存到CSV文件: {output_csv}")


def main(args):
    for out in (args.o, args.out_png, args.out_pose):
        if not out:
            continue
        mkbasedir(out)
        warnexists(out)

    if args.in_pose is None and args.t_extent == 0.0:
        log("Not shifting images")
    elif args.in_pose is None:
        assert args.t_extent > 0

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    log("Use cuda {}".format(use_cuda))
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    t1 = time.time()
    vol, _ = mrcfile.parse_mrc(args.mrc)
    log("Loaded {} volume".format(vol.shape))

    if args.tilt:
        theta = args.tilt * np.pi / 180
        args.tilt = np.array(
            [
                [1.0, 0.0, 0.0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        ).astype(np.float32)

    projector = Projector(vol, args.tilt)
    if use_cuda:
        projector.lattice = projector.lattice.cuda()
        projector.vol = projector.vol.cuda()

    if args.grid is not None:
        rots = GridRot(args.grid)
        log(
            "Generating {} rotations at resolution level {}".format(
                len(rots), args.grid
            )
        )
    elif args.in_pose is not None:
        rots = Poses(args.in_pose)
        log("Generating {} rotations from {}".format(len(rots), args.grid))
    else:
        log("Generating {} random rotations".format(args.N))
        rots = RandomRot(args.N)

    log("Projecting...")
    imgs = []
    iterator = data.DataLoader(rots, batch_size=args.b)
    for i, rot in enumerate(iterator):
        log("Projecting {}/{}".format((i + 1) * len(rot), args.N))
        projections = projector.project(rot)
        projections = projections.cpu().numpy()
        imgs.append(projections)

    td = time.time() - t1
    log("Projected {} images in {}s ({}s per image)".format(rots.N, td, td / rots.N))
    imgs = np.vstack(imgs)
    
    min_vals = imgs.min(axis=(1, 2), keepdims=True)
    imgs = imgs - min_vals

    if args.in_pose is None and args.t_extent:
        log("Shifting images between +/- {} pixels".format(args.t_extent))
        trans = np.random.rand(args.N, 2) * 2 * args.t_extent - args.t_extent
    elif args.in_pose is not None:
        log("Shifting images by input poses")
        D = imgs.shape[-1]
        trans = rots.trans * D  # convert to pixels
        trans = -trans[:, ::-1]  # convention for scipy
    else:
        trans = None

    if trans is not None:
        imgs = np.asarray([translate_img(img, t) for img, t in zip(imgs, trans)])
        # convention: we want the first column to be x shift and second column to be y shift
        # reverse columns since current implementation of translate_img uses scipy's
        # fourier_shift, which is flipped the other way
        # convention: save the translation that centers the image
        trans = -trans[:, ::-1]
        # convert translation from pixel to fraction
        D = imgs.shape[-1]
        assert D % 2 == 0
        trans /= D

    log("Saving {}".format(args.o))
    mrcfile.write_mrc(args.o, imgs.astype(np.float32))
    log("Saving {}".format(args.out_pose))
    rots = rots.rots.cpu().numpy()
    with open(args.out_pose, "wb") as f:
        if args.t_extent:
            pickle.dump((rots, trans), f)
        else:
            pickle.dump(rots, f)
    
    if args.csv:
        log("将位姿信息保存为CSV格式: {}".format(args.csv))
        csv_dir = os.path.dirname(args.csv)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        angle_step = getattr(rots, 'angle_step', 30.0)  
        euler_angles = getattr(rots, 'euler_angles', None)  
        
        save_poses_to_csv(rots, trans, args.csv, angle_step=angle_step, euler_angles_input=euler_angles)
            
    if args.out_png:
        log("Saving {}".format(args.out_png))
        plot_projections(args.out_png, imgs[:9])


if __name__ == "__main__":
    args = parse_args().parse_args()
    utils._verbose = args.verbose
    main(args)
#  python project3d.py  "C:\Shiina_Chizuru\mutiview_out\vol_00000.mrc" -o "C:\Shiina_Chizuru\mutiview_out\1008.mrcs" --out-pose 1008_out.pkl --in-pose output_poses.pkl 