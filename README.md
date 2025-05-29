# PDB_multi_view_ParticleGeneration使用指南

这个工具集实现了从蛋白质结构(PDB文件)到多视图粒子图像的完整生成流程，包括添加CTF和噪声等电镜模拟效果。我们可以使用单步骤的一步步生成多视图粒子图像，也可以使用单脚本一步到位完成所有过程和对多PDB文件的批处理。

## 环境要求

- Python 3.7+
- PyTorch
- NumPy, SciPy
- Matplotlib
- ChimeraX 或者Chimera(用于从PDB生成密度图,脚本中根据可执行文件名称自动切换指令)

## 批处理（一体化流程）

使用`main.py`脚本可以自动化执行从PDB文件到多视图粒子图像的完整批处理流程，一次性处理多个PDB文件并生成相应的产物。

**参数说明：**

基础目录参数：

- `--pdb-dir`: PDB文件所在目录（必需）
- `--output-dir`: 主输出目录（必需）
- `--sampling-mode`: 采样方式，可选"random"(随机采样)或"uniform"(均匀采样)，默认为"uniform"
- `--chimerax-path`: ChimeraX可执行文件路径，默认为"D:\Program Files\ChimeraX 1.8\bin\ChimeraX.exe"

Step 1参数（PDB转MRC）：

- `--Apix`: 体素大小(Å/pixel)，默认为1.5
- `--D`: 密度图尺寸大小，默认为128
- `--res`: 模拟密度图的分辨率(Å)，默认为3.0

Step 2参数（生成均匀姿态，仅uniform模式使用））：

- `--angle-step`: 角度采样步长(度)，默认为30.0
- `--generate-csv`: 是否生成姿态的CSV文件
- `--skip-alpha`: 是否跳过alpha角采样(固定为0度)

Step 3参数（3D投影）：

- `--N`: 随机采样模式下的投影数量，默认为1000（uniform模式下由角度步长决定）
- `--t-extent`: 像素平移范围，默认为5.0
- `--batch-size`: 投影生成的批量大小，较大的值可加快处理速度，但可能占用更多内存，默认为100

Step 5参数（添加CTF）：

- `--ctf-pkl`: CTF参数文件路径（必需）
- `--snr1`: CTF前噪声的信噪比，默认为20.0
- `--snr2`: CTF后噪声的信噪比，默认为10.0
- `--no-noise`: 设置此参数时不添加任何噪声，只应用CTF

**运行示例：**

```bash
python main.py --pdb-dir "path/to/pdb_files" --output-dir "path/to/output" --sampling-mode uniform --chimerax-path "D:\\Program Files\\ChimeraX 1.8\\bin\\ChimeraX.exe" --angle-step 30 --ctf-pkl "selected_ctf.pkl" --Apix 1.5 --D 256 --generate-csv --skip-alpha
```

## 完整流程

### Step 1：为PDB生成模拟密度图

使用`pdb2mrc.py`脚本将PDB原子结构模型转换为电镜密度图。

**参数说明：**

必需参数：
- `pdb`: PDB文件路径（必需）
- `traj`: 轨迹文件路径（必需，若只有单一结构，可与PDB文件使用相同路径）
- `num_models`: 要生成的结构数量（必需，一个PDB对应一个结构）
- `-o`: 输出目录（必需）

可选参数：
- `--Apix`: 体素大小，单位为埃/像素（默认值：1.5）
- `-D`: 密度图尺寸大小（默认值：256）
- `--res`: 模拟密度的分辨率，单位为埃（默认值：3.0）
- `-c`: Chimera或ChimeraX可执行文件路径（默认将尝试查找系统路径）
- `--debug`: 显示详细的调试信息，包括命令执行过程和中间结果（默认关闭）

可视化参数（可选）：

- `--create-montage`: 将所有MRC文件拼接成一个完整的MRC用于可视化（默认关闭）
- `--montage-layout`: 拼接布局方式（默认值：grid，可选grid或line）
- `--montage-spacing`: 拼接时各密度图之间的间距（默认值：10像素）

**运行示例：**

```bash
python pdb2mrc.py "example\pdb5ye1.ent" "example\pdb5ye1.ent" 1 --Apix 1.5 -D 256 --res 3.0 -c "D:\Program Files\ChimeraX 1.8\bin\ChimeraX.exe" -o output_dir 
```

这里注意一下，指令中对于PDB的路径要输入两遍，是为了保留轨迹文件的接口，如果有轨迹的.xml文件将其填入第二个PDB路径替代即可。

### Step 2：生成均匀采样的位姿信息[如果希望随机采样可跳过这一步]

使用`generate_uniform_poses.py`脚本生成均匀采样的欧拉角和旋转矩阵，用于后续的投影生成。

**参数说明：**

- `-o, --output`: 输出姿态文件路径(.pkl)
- `-s, --step`: 角度采样步长(度)，默认为30度
- `--viz`: 可视化欧拉角分布的输出图片路径(.png)
- `--no-trans`: 不生成平移向量（默认生成零平移）
- `--csv`: 将姿态信息以CSV格式输出的文件路径(.csv) *[option]*
- `--skip-alpha`: 跳过alpha角度采样，固定alpha为0度

**其中`--skip-alpha`就是李老师您所说的情况，这个自由度的360旋转是体现在生成的二维图像的旋转，而非视角的变换。谢谢老师指点！**

**运行示例：**

```bash
python generate_uniform_poses.py -o output_poses.pkl -s 30 --viz poses_viz.png --csv poses.csv --skip-alpha
```

### Step 3：应用位姿信息生成二维投影图像

使用`project3d.py`脚本从3D密度图生成2D投影图像。

**参数说明：**

- `mrc`: 输入密度图文件路径
- `-o`: 输出投影集文件路径(.mrcs)
- `--out-pose`: 输出姿态文件路径(.pkl)
- `--out-png`: 保存前9个投影的图像的路径
- `--in-pose`: 可选的输入姿态文件路径，代替随机姿态(.pkl)
- `-N`: 随机投影的数量(采用随机姿态时生效)
- `-b`: 小批量大小（默认：100）
- `--t-extent`: 像素平移范围（默认：+/-5）

可选择的参数输入：

- `--grid`: 在SO3上生成均匀网格投影，指定分辨率大小

- `--tilt`: 右手x轴倾斜偏移（度）
- `--seed`: 随机种子
- `--csv`: 将投影图像对应的位姿信息保存为CSV格式的文件路径
- `-v, --verbose`: 增加详细输出

**运行示例：**

```bash
python project3d.py "output_dir/vol_00000.mrc" -o "output_dir/particles.mrcs" --out-pose output_poses.pkl --in-pose generated_poses.pkl --out-png projections.png
```

### Step 4：分析CTF参数文件

使用`analyze_ctf_pkl.py`脚本分析和选择CTF参数。

**参数说明：**

- `--pkl`: CTF参数文件路径(我在代码文件中给出了一个ctf的数据集pkl，9076_ctfs.pkl)
- `--mode`: 运行模式
  - `analyze`: 分析模式，分析CTF参数并生成统计图表
  - `select`: 选择模式，从CTF参数文件中选择指定的参数组
- `--output`: 选择模式下，保存选中CTF参数的输出文件路径

**运行示例：**

```bash
# 分析模式
python analyze_ctf_pkl.py --pkl 9076_ctfs.pkl --mode analyze

# 选择模式
python analyze_ctf_pkl.py --pkl 9076_ctfs.pkl --mode select --output selected_ctf.pkl
```

### Step 5：将CTF影响嵌入图像

使用`add_ctf.py`脚本将CTF效应和噪声添加到投影图像中。

**参数说明：**

- `particles`: 输入的MRC堆栈文件，包含粒子图像数据
- `--snr1`: 应用CTF前添加的第一阶段噪声的信噪比（默认：20），用于模拟样本物理噪声
- `--snr2`: 应用CTF后添加的第二阶段噪声的信噪比（默认：10），用于模拟数字/散粒噪声
- `--s1`: 直接指定CTF前噪声的高斯标准差
- `--s2`: 直接指定CTF后噪声的高斯标准差
- `--no-noise`: 设置此参数时，不添加噪声，只应用CTF
- `--seed`: 随机数种子（默认：0）
- `-o`: 输出的二维图像集文件路径(.mrcs)
- `--out-star`: 输出的STAR文件路径（默认：[输出mrcs文件名].star）
- `--out-pkl`: 输出的ctfPickle文件路径（默认：[输出mrcs文件名].pkl）
- `--out-png`: 输出的PNG图像路径
- `--Apix`: 体素大小，单位为埃/像素
- `--ctf-pkl`: 从指定的pkl文件加载CTF参数
- `--df-file`: 从指定的pkl文件加载散焦参数

下面是手动指定ctf参数时用的argument：

- `--kv`: 显微镜电压，单位为千伏特（默认：300）
- `--dfu`: 散焦U值，单位为埃（默认：15000）
- `--dfv`: 散焦V值，单位为埃（默认：15000）
- `--ang`: 像散角度，单位为度（默认：0）
- `--cs`: 球差系数，单位为毫米（默认：2）
- `--wgh`: 振幅对比度比（默认：0.1）
- `--ps`: 相位偏移，单位为度（默认：0）
- `--defocus-std`: 散焦值的标准差（默认：0）

**运行示例：**

```bash
python add_ctf.py "output_dir/particles.mrcs" --ctf-pkl selected_ctf.pkl -o "output_dir/particles_with_ctf.mrcs" --Apix 1.5 --snr1 20 --snr2 10 --out-png ctf_preview.png
```

### 上位机：查看生成的粒子图像

使用`view_online.py`脚本以图形界面方式查看生成的粒子图像及其位姿信息。

**运行示例：**

```bash
python view_online.py
```

在界面中可以：
1. 加载MRCS二维图像集文件
2. 加载对应的姿态PKL文件
3. 按欧拉角排序浏览图像
4. 查看每个图像的详细位姿信息
5. 生成预览图

## 其他工具

### 向图像添加噪声

使用`add_noise.py`脚本向粒子图像添加噪声。

**参数说明：**

- `mrcs`: 输入粒子文件(.mrcs)
- `--snr`: 信噪比
- `--sigma`: 噪声标准差
- `--invert`: 反转数据(乘以-1)
- `--mask`: 用于计算信号方差的掩码类型
- `--mask-r`: 圆形掩码的半径
- `-o`: 输出粒子堆栈
- `--out-png`: 输出的PNG图像路径

**运行示例：**

```bash
python add_noise.py "output_dir/particles.mrcs" --snr 10 -o "output_dir/particles_with_noise.mrcs" --out-png noise_preview.png
```