"""
MRCS图像与姿态可视化上位机

功能:
1. 加载MRCS图像堆栈和对应的姿态PKL文件
2. 按欧拉角排序显示图像列表
3. 点击查看特定图像及其姿态信息
4. 根据姿态角度范围查找图像
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QListWidget, 
                            QFileDialog, QGroupBox, QGridLayout, QDoubleSpinBox,
                            QSplitter, QListWidgetItem, QRadioButton, QButtonGroup,
                            QScrollArea, QSizePolicy, QFrame, QComboBox, QSlider,
                            QMessageBox)
from PyQt5.QtCore import Qt, QSize
import scipy.ndimage

# 导入basement目录中的模块
basement_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'basement')
if basement_dir not in sys.path:
    sys.path.append(basement_dir)

# 导入basement模块
from basement import mrcfile

# 检查是否可以将欧拉角转换为旋转矩阵
try:
    from generate_uniform_poses import euler_to_rot
except ImportError:
    # 如果导入失败，提供自己的实现
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

# 反向函数：从旋转矩阵提取欧拉角
def rot_to_euler(R, angle_step=30.0):
    """
    从旋转矩阵提取欧拉角（ZYZ约定），优化处理万向节锁情况
    
    参数:
        R: 形状为 (3, 3) 的旋转矩阵
        angle_step: 角度步长（度），用于万向节锁情况下的处理
        
    返回:
        欧拉角 (alpha, beta, gamma)，单位为度
    """
    beta = np.arccos(max(-1, min(1, R[2, 2])))  # 防止数值误差导致arccos的输入超出[-1,1]范围
    beta_deg = np.degrees(beta)
    
    if np.isclose(abs(R[2, 2]), 1.0):
        # 当beta接近0或180度时，alpha和gamma存在万向节锁（Gimbal Lock）
        # 改进万向节锁处理：按照generate_uniform_poses.py的生成模式
        # 从旋转矩阵中提取α+γ的总角度
        angle_sum = np.arctan2(R[1, 0], R[0, 0])
        angle_sum_deg = (np.degrees(angle_sum) + 360) % 360
        
        # 计算γ的最大值（通常为360°-步长）
        gamma_max = 360 - angle_step
        
        # 根据总角度计算α和γ，优先填充γ
        gamma_steps = int(angle_sum_deg / angle_step)
        gamma_deg = (gamma_steps % int(360 / angle_step)) * angle_step
        
        # 如果总角度超过了一轮γ，递增α
        alpha_steps = int(gamma_steps / (360 / angle_step))
        alpha_deg = alpha_steps * angle_step
        
        # 确保α和γ在正确范围内
        alpha_deg = alpha_deg % 360
        gamma_deg = gamma_deg % 360
    else:
        # 正常情况下的欧拉角提取
        alpha = np.arctan2(R[1, 2], R[0, 2])
        gamma = np.arctan2(R[2, 1], -R[2, 0])
        
        # 转换为度数并确保在正确范围内
        alpha_deg = (np.degrees(alpha) + 360) % 360
        gamma_deg = (np.degrees(gamma) + 360) % 360
    
    return np.array([alpha_deg, beta_deg, gamma_deg])

def parse_mrc(mrc_file):
    """加载MRC/MRCS文件"""
    # 使用basement目录中的mrcfile模块解析MRC文件
    from basement.mrcfile import parse_mrc as local_parse_mrc
    images, _ = local_parse_mrc(mrc_file)
    return images

def load_poses(pkl_file):
    """加载姿态文件"""
    with open(pkl_file, 'rb') as f:
        poses = pickle.load(f)
    
    # 检查poses的格式
    angle_step = 30.0  # 默认角度步长
    original_euler_angles = None  # 原始欧拉角
    
    if isinstance(poses, dict):
        # 新格式：包含额外信息的字典
        rot_matrices = poses['rot_matrices']
        trans_vectors = poses['trans_vectors']
        angle_step = poses.get('angle_step', 30.0)  # 角度步长
        original_euler_angles = poses.get('euler_angles', None)  # 原始欧拉角
    elif isinstance(poses, tuple) and len(poses) == 2:
        # 旧格式：包含旋转矩阵和平移向量的元组
        rot_matrices = poses[0]
        trans_vectors = poses[1]
    else:
        # 只有旋转矩阵
        rot_matrices = poses
        trans_vectors = np.zeros((len(rot_matrices), 2))
    
    # 计算欧拉角（如果没有提供原始欧拉角）
    if original_euler_angles is not None:
        # 使用原始欧拉角
        euler_angles = np.array(original_euler_angles)
    else:
        # 使用rot_to_euler函数计算欧拉角，并传入角度步长
        euler_angles = np.array([rot_to_euler(R, angle_step) for R in rot_matrices])
    
    return rot_matrices, trans_vectors, euler_angles

class MRCSViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.images = None
        self.rot_matrices = None
        self.trans_vectors = None
        self.euler_angles = None
        self.current_index = 0
        self.sorted_indices = None
        self.sort_by = 'alpha'  # 默认按alpha角排序
        self.preview_file = None  # 存储预览图片路径
        
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('MRCS图像与姿态可视化')
        self.setGeometry(100, 100, 1200, 800)
        
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 文件加载部分
        file_group = QGroupBox("文件加载")
        file_layout = QGridLayout()
        
        self.mrcs_btn = QPushButton('加载MRCS文件')
        self.pose_btn = QPushButton('加载姿态文件')
        self.preview_btn = QPushButton('生成预览图')
        self.mrcs_btn.clicked.connect(self.load_mrcs)
        self.pose_btn.clicked.connect(self.load_pose)
        self.preview_btn.clicked.connect(self.generate_preview)
        
        # 预览图格式选择
        self.preview_format_label = QLabel("预览图格式:")
        self.preview_format = QComboBox()
        self.preview_format.addItem("3 × 3 (9张)")
        self.preview_format.addItem("3 × 5 (15张)")
        self.preview_format.addItem("4 × 4 (16张)")
        self.preview_format.addItem("5 × 5 (25张)")
        self.preview_format.setCurrentIndex(1)  # 默认选择3×5
        
        self.mrcs_label = QLabel("未加载MRCS文件")
        self.pose_label = QLabel("未加载姿态文件")
        
        file_layout.addWidget(self.mrcs_btn, 0, 0)
        file_layout.addWidget(self.mrcs_label, 0, 1)
        file_layout.addWidget(self.pose_btn, 1, 0)
        file_layout.addWidget(self.pose_label, 1, 1)
        file_layout.addWidget(self.preview_format_label, 2, 0)
        file_layout.addWidget(self.preview_format, 2, 1)
        file_layout.addWidget(self.preview_btn, 3, 0)
        file_group.setLayout(file_layout)
        
        # 排序选项
        sort_group = QGroupBox("排序方式")
        sort_layout = QGridLayout()
        
        self.sort_alpha = QRadioButton("按Alpha角排序")
        self.sort_beta = QRadioButton("按Beta角排序")
        self.sort_gamma = QRadioButton("按Gamma角排序")
        self.sort_original = QRadioButton("按原始编号排序")
        self.sort_alpha.setChecked(True)
        
        sort_btn_group = QButtonGroup(self)
        sort_btn_group.addButton(self.sort_alpha)
        sort_btn_group.addButton(self.sort_beta)
        sort_btn_group.addButton(self.sort_gamma)
        sort_btn_group.addButton(self.sort_original)
        sort_btn_group.buttonClicked.connect(self.change_sort)
        
        sort_layout.addWidget(self.sort_alpha, 0, 0)
        sort_layout.addWidget(self.sort_beta, 1, 0)
        sort_layout.addWidget(self.sort_gamma, 2, 0)
        sort_layout.addWidget(self.sort_original, 3, 0)
        sort_group.setLayout(sort_layout)
        
        # 姿态查找部分
        search_group = QGroupBox("按姿态查找")
        search_layout = QGridLayout()
        
        search_layout.addWidget(QLabel("Alpha范围:"), 0, 0)
        self.alpha_min = QDoubleSpinBox()
        self.alpha_min.setRange(0, 360)
        self.alpha_max = QDoubleSpinBox()
        self.alpha_max.setRange(0, 360)
        self.alpha_max.setValue(360)
        
        search_layout.addWidget(QLabel("Beta范围:"), 1, 0)
        self.beta_min = QDoubleSpinBox()
        self.beta_min.setRange(0, 180)
        self.beta_max = QDoubleSpinBox()
        self.beta_max.setRange(0, 180)
        self.beta_max.setValue(180)
        
        search_layout.addWidget(QLabel("Gamma范围:"), 2, 0)
        self.gamma_min = QDoubleSpinBox()
        self.gamma_min.setRange(0, 360)
        self.gamma_max = QDoubleSpinBox()
        self.gamma_max.setRange(0, 360)
        self.gamma_max.setValue(360)
        
        search_layout.addWidget(self.alpha_min, 0, 1)
        search_layout.addWidget(self.alpha_max, 0, 2)
        search_layout.addWidget(self.beta_min, 1, 1)
        search_layout.addWidget(self.beta_max, 1, 2)
        search_layout.addWidget(self.gamma_min, 2, 1)
        search_layout.addWidget(self.gamma_max, 2, 2)
        
        self.search_btn = QPushButton("查找")
        self.search_btn.clicked.connect(self.search_by_pose)
        search_layout.addWidget(self.search_btn, 3, 0, 1, 3)
        
        search_group.setLayout(search_layout)
        
        # 添加到左侧面板
        left_layout.addWidget(file_group)
        left_layout.addWidget(sort_group)
        left_layout.addWidget(search_group)
        left_layout.addStretch(1)
        
        # 中间图像列表
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        
        middle_layout.addWidget(QLabel("图像列表 (按姿态角排序)"))
        
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(100, 100))
        self.image_list.itemClicked.connect(self.image_selected)
        middle_layout.addWidget(self.image_list)
        
        # 右侧图像和姿态显示
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 图像显示
        self.fig = plt.figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        image_group = QGroupBox("当前图像")
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.toolbar)
        image_layout.addWidget(self.canvas)
        image_group.setLayout(image_layout)
        
        # 姿态信息显示
        pose_info_group = QGroupBox("姿态信息")
        pose_info_layout = QGridLayout()
        
        pose_info_layout.addWidget(QLabel("欧拉角 (ZYZ约定):"), 0, 0)
        self.euler_label = QLabel("未选择图像")
        pose_info_layout.addWidget(self.euler_label, 0, 1)
        
        pose_info_layout.addWidget(QLabel("旋转矩阵:"), 1, 0)
        self.rot_matrix_label = QLabel("未选择图像")
        pose_info_layout.addWidget(self.rot_matrix_label, 1, 1)
        
        pose_info_layout.addWidget(QLabel("平移向量:"), 2, 0)
        self.trans_label = QLabel("未选择图像")
        pose_info_layout.addWidget(self.trans_label, 2, 1)
        
        pose_info_group.setLayout(pose_info_layout)
        
        # 亮度和对比度调整
        adjustment_group = QGroupBox("图像调整")
        adjustment_layout = QGridLayout()
        
        adjustment_layout.addWidget(QLabel("亮度:"), 0, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_image)
        adjustment_layout.addWidget(self.brightness_slider, 0, 1)
        
        adjustment_layout.addWidget(QLabel("对比度:"), 1, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(1, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_image)
        adjustment_layout.addWidget(self.contrast_slider, 1, 1)
        
        adjustment_group.setLayout(adjustment_layout)
        
        # 添加到右侧面板
        right_layout.addWidget(image_group, 3)
        right_layout.addWidget(pose_info_group, 1)
        right_layout.addWidget(adjustment_group, 1)
        
        # 设置各面板宽度比例
        splitter1 = QSplitter(Qt.Horizontal)
        splitter1.addWidget(left_panel)
        splitter1.addWidget(middle_panel)
        splitter1.addWidget(right_panel)
        splitter1.setSizes([250, 300, 650])
        
        main_layout.addWidget(splitter1)
        
        # 初始化状态
        self.update_ui_state()
    
    def load_mrcs(self):
        """加载MRCS文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择MRCS文件", "", "MRCS文件 (*.mrcs *.mrc)")
        if file_path:
            try:
                self.images = parse_mrc(file_path)
                self.mrcs_label.setText(os.path.basename(file_path))
                self.update_ui_state()
                QMessageBox.information(self, "加载成功", f"成功加载 {len(self.images)} 张图像")
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"无法加载MRCS文件: {str(e)}")
    
    def load_pose(self):
        """加载姿态文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择姿态文件", "", "PKL文件 (*.pkl)")
        if file_path:
            try:
                self.rot_matrices, self.trans_vectors, self.euler_angles = load_poses(file_path)
                self.pose_label.setText(os.path.basename(file_path))
                self.update_ui_state()
                QMessageBox.information(self, "加载成功", f"成功加载 {len(self.rot_matrices)} 组姿态")
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"无法加载姿态文件: {str(e)}")
    
    def update_ui_state(self):
        """更新UI状态"""
        # 检查是否同时加载了图像和姿态
        if self.images is not None and self.euler_angles is not None:
            # 检查图像数量和姿态数量是否匹配
            if len(self.images) != len(self.euler_angles):
                QMessageBox.warning(self, "数量不匹配", 
                                  f"图像数量 ({len(self.images)}) 与姿态数量 ({len(self.euler_angles)}) 不匹配!")
            
            # 更新排序和图像列表
            self.update_sorting()
        else:
            # 清空图像列表
            self.image_list.clear()
    
    def change_sort(self, button):
        """更改排序方式"""
        if button == self.sort_alpha:
            self.sort_by = 'alpha'
        elif button == self.sort_beta:
            self.sort_by = 'beta'
        elif button == self.sort_gamma:
            self.sort_by = 'gamma'
        elif button == self.sort_original:
            self.sort_by = 'original'
        
        # 更新排序
        self.update_sorting()
    
    def update_sorting(self):
        """根据选择的欧拉角或原始编号更新排序"""
        if self.euler_angles is None:
            return
        
        if self.sort_by == 'original':
            # 按原始编号排序（即索引从0到n-1）
            self.sorted_indices = np.arange(len(self.euler_angles))
        else:
            # 按欧拉角排序
            # 确定排序列
            if self.sort_by == 'alpha':
                sort_col = 0
            elif self.sort_by == 'beta':
                sort_col = 1
            elif self.sort_by == 'gamma':
                sort_col = 2
            
            # 按指定列排序
            self.sorted_indices = np.argsort(self.euler_angles[:, sort_col])
        
        # 更新图像列表
        self.update_image_list()
    
    def update_image_list(self):
        """更新图像列表"""
        self.image_list.clear()
        
        if self.images is None or self.euler_angles is None or self.sorted_indices is None:
            return
        
        # 获取用于显示的图像数量
        num_display = min(len(self.images), len(self.euler_angles), len(self.sorted_indices))
        
        for i in range(num_display):
            idx = self.sorted_indices[i]
            # 获取欧拉角
            alpha, beta, gamma = self.euler_angles[idx]
            
            # 创建列表项
            item = QListWidgetItem(f"#{idx}: α={alpha:.1f}° β={beta:.1f}° γ={gamma:.1f}°")
            
            # 添加到列表
            self.image_list.addItem(item)
    
    def image_selected(self, item):
        """图像被选中时的处理"""
        # 解析图像索引
        text = item.text()
        self.current_index = int(text.split(':')[0][1:])
        
        # 更新图像显示
        self.update_image()
        
        # 更新姿态信息
        self.update_pose_info()
    
    def update_image(self):
        """更新图像显示"""
        if self.images is None or self.current_index >= len(self.images):
            return
        
        # 获取图像
        img = self.images[self.current_index].copy()
        
        # 应用亮度和对比度调整
        brightness = self.brightness_slider.value() / 100.0
        contrast = self.contrast_slider.value() / 100.0
        
        # 应用对比度
        img = (img - np.mean(img)) * contrast + np.mean(img)
        
        # 应用亮度
        img = img + brightness * np.std(img)
        
        # 清除之前的图
        self.fig.clear()
        
        # 绘制新图
        ax = self.fig.add_subplot(111)
        im = ax.imshow(img, cmap='gray')
        ax.set_title(f"图像 #{self.current_index}")
        ax.axis('off')
        self.fig.colorbar(im)
        
        # 更新画布
        self.canvas.draw()
    
    def update_pose_info(self):
        """更新姿态信息显示"""
        if (self.rot_matrices is None or self.euler_angles is None or 
            self.current_index >= len(self.rot_matrices)):
            return
        
        # 获取欧拉角
        alpha, beta, gamma = self.euler_angles[self.current_index]
        self.euler_label.setText(f"α = {alpha:.2f}°, β = {beta:.2f}°, γ = {gamma:.2f}°")
        
        # 获取旋转矩阵
        R = self.rot_matrices[self.current_index]
        rot_text = f"[{R[0,0]:.3f} {R[0,1]:.3f} {R[0,2]:.3f}]<br/>"
        rot_text += f"[{R[1,0]:.3f} {R[1,1]:.3f} {R[1,2]:.3f}]<br/>"
        rot_text += f"[{R[2,0]:.3f} {R[2,1]:.3f} {R[2,2]:.3f}]"
        self.rot_matrix_label.setText(rot_text)
        
        # 获取平移向量
        if self.trans_vectors is not None and self.current_index < len(self.trans_vectors):
            t = self.trans_vectors[self.current_index]
            self.trans_label.setText(f"[{t[0]:.3f}, {t[1]:.3f}]")
    
    def search_by_pose(self):
        """根据姿态范围查找图像"""
        if self.euler_angles is None:
            QMessageBox.warning(self, "无法查找", "请先加载姿态数据")
            return
        
        # 获取范围
        alpha_min = self.alpha_min.value()
        alpha_max = self.alpha_max.value()
        beta_min = self.beta_min.value()
        beta_max = self.beta_max.value()
        gamma_min = self.gamma_min.value()
        gamma_max = self.gamma_max.value()
        
        # 查找匹配的索引
        mask_alpha = (self.euler_angles[:, 0] >= alpha_min) & (self.euler_angles[:, 0] <= alpha_max)
        mask_beta = (self.euler_angles[:, 1] >= beta_min) & (self.euler_angles[:, 1] <= beta_max)
        mask_gamma = (self.euler_angles[:, 2] >= gamma_min) & (self.euler_angles[:, 2] <= gamma_max)
        
        mask_combined = mask_alpha & mask_beta & mask_gamma
        matching_indices = np.where(mask_combined)[0]
        
        # 更新排序索引
        if len(matching_indices) > 0:
            # 按当前排序方式重新排序匹配结果
            if self.sort_by == 'alpha':
                sort_col = 0
            elif self.sort_by == 'beta':
                sort_col = 1
            else:  # gamma
                sort_col = 2
            
            # 获取匹配欧拉角
            matching_angles = self.euler_angles[matching_indices]
            
            # 对匹配结果排序
            sorted_matching_indices = matching_indices[np.argsort(matching_angles[:, sort_col])]
            
            # 更新排序索引
            self.sorted_indices = sorted_matching_indices
            
            # 更新图像列表
            self.update_image_list()
            
            QMessageBox.information(self, "查找结果", f"找到 {len(matching_indices)} 个匹配的图像")
        else:
            QMessageBox.information(self, "查找结果", "没有找到匹配的图像")

    def generate_preview(self):
        """生成预览图，支持不同的行列格式，背景为纯黑色"""
        if self.images is None or len(self.images) == 0:
            QMessageBox.warning(self, "无法生成预览", "请先加载MRCS文件")
            return
        
        # 获取选择的预览格式
        format_idx = self.preview_format.currentIndex()
        if format_idx == 0:  # 3 × 3
            nrow, ncol = 3, 3
            n_max = 9
            figsize = (10, 10)
        elif format_idx == 1:  # 3 × 5
            nrow, ncol = 3, 5
            n_max = 15
            figsize = (15, 9)
        elif format_idx == 2:  # 4 × 4
            nrow, ncol = 4, 4
            n_max = 16
            figsize = (12, 12)
        elif format_idx == 3:  # 5 × 5
            nrow, ncol = 5, 5
            n_max = 25
            figsize = (15, 15)
        else:  # 默认 3 × 5
            nrow, ncol = 3, 5
            n_max = 15
            figsize = (15, 9)
        
        # 弹出保存对话框
        file_path, _ = QFileDialog.getSaveFileName(self, "保存预览图", "", "PNG图像 (*.png)")
        if not file_path:
            return  # 用户取消了保存
        
        # 确保文件名有.png后缀
        if not file_path.lower().endswith('.png'):
            file_path += '.png'
            
        # 获取前 n_max 张图像（或全部，如果不足 n_max 张）
        n_images = min(n_max, len(self.images))
        if self.sorted_indices is not None:
            # 使用排序后的索引
            indices = self.sorted_indices[:n_images]
        else:
            # 使用原始顺序
            indices = range(n_images)
        
        # 创建图像网格
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, dpi=100)
        # 设置图表背景为纯黑
        fig.patch.set_facecolor('black')
        
        # 处理只有一行或一列的情况
        if nrow == 1 and ncol == 1:
            axes = np.array([axes])
        elif nrow == 1 or ncol == 1:
            axes = axes.reshape(-1)
        
        # 填充图像
        for i, ax in enumerate(axes.flat):
            if i < n_images:
                img_idx = indices[i]
                img = self.images[img_idx].copy()
                
                # 调整对比度（可选）
                # 找到每张图像的最小值并设为0（纯黑背景）
                min_val = img.min()
                img = img - min_val
                
                # 显示图像
                ax.imshow(img, cmap='gray')
                
                # 添加欧拉角标签（如果有）
                if self.euler_angles is not None:
                    alpha, beta, gamma = self.euler_angles[img_idx]
                    ax.set_title(f"#{img_idx}: α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°", color='white', fontsize=8)
            
            # 设置坐标轴背景为黑色
            ax.set_facecolor('black')
            ax.axis('off')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(file_path, facecolor='black', bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        self.preview_file = file_path
        QMessageBox.information(self, "预览图已生成", f"预览图 ({nrow}×{ncol}) 已保存至: {file_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MRCSViewer()
    viewer.show()
    sys.exit(app.exec_())