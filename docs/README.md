# 力控机械臂超声体模扫描与 3D 重建系统

基于 UR5e 机械臂 + Orbbec 深度相机 + USB 超声采集卡 + **vLLM 多模态大模型**，实现"**自然语言一句话 → 力控自动扫描 → 3D 重建**"的端到端闭环。

## 旗舰工作流：vLLM 自然语言指令 + 力控自动扫描

> 一条 service call 就能完成：识图 → 决定扫查方向 → 标定起止点 → 力控接触 → 沿曲面扫描 → 自动录制 → 3D 重建。**不需要贴 ArUco，不需要手动改参数。**

```bash
# ① 一句话下指令（任意中文 / 英文）
rosservice call /us3d/plan_from_instruction "instruction: '沿体模长轴扫 80mm，慢一点 2mm/s'
dry_run: false"

# ② 预览路径（RViz 红色箭头）
rosservice call /us3d/preview_scan

# ③ 力控执行（一键完成 touchdown + 扫描 + 录制）
rosservice call /us3d/start_scan

# ④ 离线 3D 重建
python3 pbm_reconstruct.py --scan_dir data/scans/scan_YYYYMMDD_HHMMSS \
                           --calib_dir data/calibration \
                           --output data/reconstructions/pbm.npy

# ⑤ napari 交互式 3D 体积浏览
python3 visualize_volume.py --volume data/reconstructions/pbm.npy --mode napari
```

### 端到端时序

```
①  自然语言指令 + 相机彩色图
        │
        ▼
   instruction_planner: 单次多模态 VLM 调用 (vLLM / Kimi-k2 / Qwen-VL)
        │     输出: phantom_type, scan_length_mm, scan_speed_mms,
        │            reverse_direction, endpoints_norm[2 个像素端点]
        │
        ▼
   vlm_anchor: depth 反投影端点 → /us3d/markers (虚拟, 替代 ArUco)
        │
        ▼
   scan_planner: 点云裁剪 + 顶层提取 + RBF 曲面拟合 + 路径平滑
        │     输出: /us3d/scan_path (per-waypoint Z + 朝向)
        │
        ▼
②  preview_scan: RViz 红色箭头 + 关节路径动画
        │
        ▼
③  ★ force_scan_node 力控自动时序:
   1) Cartesian SLERP approach   (避开 octomap, 全速)
   2) F/T 清零                    (悬空校准)
   3) ★ 力控 Touchdown            (3mm 增量步进, 接触自动停)
   4) 应用 dz_correction + Z 安全 floor (不允许深压)
   5) UR speed slider 降到 ~15%   (只在扫描段慢)
   6) ★ Force-adaptive 分段扫描   (每段后查 |Fz|<2N 就降 Z 1mm)
   7) 自动录制                     (warmup 跳帧 + stable wait 抑制瞬态)
   8) speed slider 恢复 1.0        (后续动作全速)
        │
        ▼
④  PBM 重建 (elevation 高斯 splat + Gaussian hole filling)
        │
        ▼
⑤  napari 交互式 3D 可视化 (--align_to_scan 自动对齐扫描方向)
```

### 为什么这样设计

| 模块 | 职责 | 安全边界 |
|------|------|---------|
| **VLM (instruction_planner)** | 看图 + 理解指令 → 输出**结构化扫查参数** + 像素端点 | 永远不直接产出 robot waypoint / 关节角；数值在 `vlm.yaml` 的 min/max 内 clamp |
| **vlm_anchor** | 像素端点 + depth → base_link 中两个 3D 点（虚拟 marker） | 端点超 FOV / depth 空洞时直接 `success=false`；7×7 patch 中位数抗噪 |
| **scan_planner** | marker + 体模点云 → 沿曲面的 waypoint 序列 | RBF 拟合 + SG 平滑；Z 安全 floor `max_push_below_contact_mm: 0` |
| **★ force_scan_node** | 力控 touchdown + 力自适应 Z 调整 + 自动录制 | UR 内部 PROTECTIVE_STOP；touchdown_force ~2.5N 软接触；force_adaptive 累积上限 -10mm |

VLM 出错最坏后果是**路径方向不对**或**端点超 FOV**——力控层的 touchdown / Z floor / force_adaptive **三道防线都不会被绕过**：

- 端点反投影到桌面 → touchdown 到桌面就停（不深压）
- VLM 给的方向偏 → 探头扫到空气 → force_adaptive 检测到脱接触只往**下**调（不抬头），上限 10mm
- 任何一步 VLM / anchor 失败 → service 立即 `success=false`，机械臂不动作

### 实测效果（100mm 长轴扫一次）

- VLM 单次调用 30–60s（云端 thinking 模型 streaming）
- Touchdown 接触精度 < 1mm（depth-cam 估计差 18mm 也能精确补偿）
- Post-scan probe_tip Z 误差 ≈ 0.6mm
- PBM 重建 Pass 2 fill ratio ~66%（197 帧）
- 零 PROTECTIVE_STOP / 零保护停

## 系统架构

```
   自然语言指令                                                力曲线 / RViz
       │                                                         ▲
       ▼                                                         │
┌──────────────┐  vision+text  ┌─────────────┐  endpoints  ┌──────────────┐
│   vLLM 服务  │ ◄───────────► │ instruction │ ──────────► │  vlm_anchor  │
│ (Kimi/Qwen-VL│   1 次调用    │   planner   │             │ depth→3D 点  │
│ /InternVL …) │               └─────────────┘             └──────┬───────┘
└──────────────┘                                                  │
                                                       /us3d/markers (虚拟)
                                                                  ▼
   硬件层                ROS 中间层                  应用层
   ─────────         ──────────────             ──────────────
   UR5e 机械臂   →   ur_robot_driver  →  ★ force_scan_node (touchdown +
                                            force-adaptive + auto record)
                                                                  ▲
   Orbbec 相机   →   OrbbecSDK_ROS1   →   scan_planner (RBF 曲面 + 路径平滑)
                                                                  ▲
   超声采集卡    →   us_capture_node  →   sync_recorder → PBM 重建 → napari
```

## 经典工作流（仍可用，作为 fallback）

如果 VLM 服务不可用、或者你就是想用 ArUco 走老路：参见下方 [§5. 曲面扫查](#5-曲面扫查)（贴 ArUco → `detect_markers` → `plan_scan` → `start_scan`）。下面"vLLM 工作流"小节也保留了关闭 VLM 的方式。

## 目录结构

```
us3dscan/
├── catkin_ws/src/
│   ├── us3d_msgs/              # 自定义 ROS 消息
│   ├── us3d_perception/        # 感知节点（超声采集、ArUco、表面定位）
│   ├── us3d_control/           # 控制节点（路径生成、曲面扫查规划、力控扫描、标定辅助、避障）
│   ├── us3d_vlm/               # ★ 视觉语言模型（接 vLLM）：体模识别 + 指令转扫查计划
│   ├── us3d_acquisition/       # 同步数据采集（自动录制、warmup 丢帧）
│   ├── us3d_reconstruction/    # 离线 3D 重建与可视化（voxel/PBM/TSDF）
│   ├── us3d_bringup/           # Launch 文件、配置、URDF
│   ├── OrbbecSDK_ROS1/         # Orbbec 相机驱动
│   ├── universal_robot/        # UR URDF / MoveIt / Gazebo
│   └── easy_handeye/           # 手眼标定
├── data/                       # 扫描数据与重建结果
│   ├── aruco_markers/          # ArUco 标记图片（打印用）
│   ├── calibration/            # probe_calibration.yaml + 手眼标定结果
│   ├── scans/                  # scan_YYYYMMDD_HHMMSS/ 数据集
│   └── reconstructions/        # 重建后的 .npy/.ply
├── tools/                      # 辅助工具脚本
│   ├── generate_aruco_markers.py
│   └── 99-us-capture.rules     # udev 规则: /dev/us_capture 稳定符号链接
└── requirements.txt
```

## 环境配置

### 前置条件

- Ubuntu 20.04 + ROS Noetic
- Conda（Miniconda / Anaconda）

### 安装步骤

```bash
# 1. 创建 conda 环境
conda create -n joeros python=3.8
conda activate joeros

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 安装 ROS 系统包
sudo apt install -y \
  ros-noetic-ur-robot-driver ros-noetic-ur-calibration ros-noetic-ur-msgs \
  ros-noetic-moveit ros-noetic-easy-handeye ros-noetic-aruco-ros \
  ros-noetic-cv-bridge ros-noetic-image-transport \
  ros-noetic-message-filters ros-noetic-gazebo-ros-pkgs \
  ros-noetic-gazebo-ros-control ros-noetic-controller-manager \
  libdw-dev

# 4. 安装 catkin 工具 & 修复 empy 兼容性
pip install catkin_tools
pip install empy==3.3.4

# 5. 安装 rosbag 依赖（到 conda 环境内）
pip install pycryptodomex python-gnupg \
  --target=$(python -c "import site; print(site.getsitepackages()[0])")

# 6. 编译
cd catkin_ws
catkin init
catkin build
source devel/setup.bash

# 7.（推荐）添加到 ~/.bashrc 自动初始化
echo '
source /opt/ros/noetic/setup.bash
conda activate joeros
source ~/joe/us3dscan/catkin_ws/devel/setup.bash
' >> ~/.bashrc

# 8. 安装 USB 采集卡 udev 规则（让设备名固定为 /dev/us_capture）
sudo cp tools/99-us-capture.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
ls -la /dev/us_capture     # 应看到 -> /dev/videoN

# 9. 安装鼠标控制工具（扫描时自动隐藏光标，避免污染超声画面）
sudo apt install -y xdotool
```

> **udev 规则说明**：USB 采集卡（MACROSILICON USB3 Video, idProduct=2131）每次重新枚举时
> `/dev/videoN` 的索引可能变化（如 8 → 0）。安装规则后会创建稳定符号链接 `/dev/us_capture`，
> `us_capture_node` 默认就用这个路径。如果你的采集卡序列号不是 `20210621`，需要修改
> `tools/99-us-capture.rules` 里的 `ATTRS{serial}=="..."` 字段（用 `lsusb -v` 查询）。

## 模块说明

### us3d_msgs — 自定义消息


| 消息              | 字段                                                                                                                   | 用途                  |
| --------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `USFrame`       | Header + Image + PoseStamped                                                                                         | 超声帧 + 探头位姿          |
| `ScanDataPoint` | Header + Image + PoseStamped + WrenchStamped                                                                         | 单帧同步数据（图像 + 位姿 + 力） |
| `ScanRegion`    | Header + origin + scan_dir + lateral_dir + normal + length + width                                                   | 扫描区域定义              |
| `PhantomInfo`   | Header + phantom_type + confidence + description + scan_axis_hint + scan_length_hint_mm + bbox_image_xyxy + endpoints_xyxy_norm + raw_json | VLM 识别 + 指令驱动的扫查端点 |


**Service：**

- `us3d_msgs/PlanFromInstruction` — 输入自然语言扫查指令，输出已派发的结构化扫查计划 JSON（见下方 us3d_vlm 节）

### us3d_perception — 感知模块


| 节点                          | 功能                                       | 订阅                                                                                | 发布                               |
| --------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------- |
| `us_capture_node.py`        | 从 USB 采集卡读取超声视频流（支持设备路径或整数 ID）；连续黑帧时主动告警 | —                                                                                 | `/us3d/image_raw`                |
| `aruco_detector_node.py`    | 检测 ArUco 标记，多帧融合到基座坐标系                   | `/camera/color/image_raw`, `/camera/depth/image_raw`, `/camera/color/camera_info` | `/us3d/markers` (PoseArray)      |
| `surface_localizer_node.py` | PCA 估计扫描平面，定义 phantom_frame              | `/us3d/markers`                                                                   | `/us3d/scan_region` (ScanRegion) |
| `tcp_pose_publisher.py`     | 把 base_link → tool0 转成 PoseStamped 高频发布  | `/tf`                                                                             | `/us3d/current_pose`             |


`**us_capture_node` 关键参数：**

- `~device_id` — 设备路径 `/dev/us_capture`（默认）或整数索引（fallback）
- `~blank_warn_secs` — 全黑画面持续多少秒后告警（默认 3.0s）

**ArUco 检测节点服务：**

- `/us3d/detect_markers` — 触发检测
- `/us3d/clear_markers` — 清除历史数据

### us3d_control — 控制模块


| 节点                       | 功能                                                  | 订阅                                      | 发布                                                   |
| ------------------------ | --------------------------------------------------- | --------------------------------------- | ---------------------------------------------------- |
| `scan_planner_node.py`   | 曲面扫查路径规划（点云 + ArUco），输出探头垂直表面、长轴垂直路径的位姿             | `/us3d/markers`, `/camera/depth/points` | `/us3d/scan_path`, `/us3d/surface_cloud`             |
| `path_generator_node.py` | 根据扫描区域生成平面平行扫描线路点                                   | `/us3d/scan_region`                     | `/us3d/scan_path` (PoseArray)                        |
| `force_scan_node.py`     | MoveIt Cartesian 路径执行 + 自动录制 + 稳定等待 + F/T 清零 + 隐藏鼠标 | `/us3d/scan_path`                       | 服务调用 `/us3d/start_recording`, `/us3d/stop_recording` |
| `hand_eye_calib_node.py` | 辅助 easy_handeye 手眼标定自动采样                            | —                                       | —                                                    |
| `add_scene_objects.py`   | 向 MoveIt 添加桌面等碰撞体（避障）                               | —                                       | PlanningScene                                        |


**曲面扫查规划节点服务：**

- `/us3d/plan_scan` — 结合点云和标记生成贴合曲面的扫查路径

**力控扫描节点服务：**

- `/us3d/preview_scan` — 在 RViz 中预览路径与姿态，**不执行**
- `/us3d/start_scan` — 执行扫描；自动按下面的时序运行
- `/us3d/stop_scan` — 中止扫描
- `/us3d/touch_marker` — 单次触碰最近检测到的 marker（标定/调试用）
- `/us3d/record_home` — 记录当前关节角为起始位置
- `/us3d/move_home` — 移动到已保存的起始位置
- `/us3d/restore_speed` — 紧急恢复 UR speed slider 到 1.0（扫描中断时用）

`**/us3d/start_scan` 自动时序**（每条扫描线）：

```
1. 自动选择最近的等价朝向 (避免手腕绕大圈)
   - planner 给的 scan_dir vs -scan_dir 是等价 (镜像) 解
   - 选择从当前姿态旋转角度更小的那个
2. 用 Cartesian SLERP 移到扫描起点上方 50 mm，同时旋转手腕
   - 优先 Cartesian 直线 (短路径, 100% 可行就用)
   - 失败时 fallback 到 RRTConnect (2.5s 超时, 不再 RRTstar 浪费 10s)
3. 隐藏鼠标光标 (xdotool)         ← 避免污染超声画面
4. 清零 F/T 传感器 (zero_ftsensor) ← 在悬空时校准, 消除重力 bias
5. ★ 力控 Touchdown 标定 (新):
   - 多次 3mm 增量步进同步执行 (每步结束机器人停止 → 接触瞬间无冲击)
   - 实时监测 Fz, |dFz| ≥ touchdown_force 时停止
   - 记录实际接触 Z 与计划 Z 的差 = dz_correction
   - 修正全部后续 waypoint 的 Z
6. 等待 stable_wait_before 秒     ← 默认 0.7 s, 吸收瞬态抖动
7. 自动 call /us3d/start_recording
8. ★ UR speed slider 设为 ~15% (只在此段慢, approach/retract 不影响)
9. 沿路径执行 Cartesian 扫描:
   - 每个 waypoint 用 planner 的独立姿态 (per-wp orientation)
   - 用 _tool0_for_probe_tip 几何补偿: probe_tip 精确落在 waypoint
   - Z 安全 floor: 不允许探头比 touchdown 接触位置更深
   - Cartesian 路径 avoid_collisions=False (与 octomap 中的体模点云不冲突)
   - ★ Force-adaptive 分段执行 (默认开):
     • 把路径切 N 段, 每段后查 Fz
     • |Fz|<2N (脱接触) → z_offset -= 1mm, 后续段全部下移
     • z_offset 累积上限 -10mm (安全)
10. ★ UR speed slider 恢复到 1.0 (try/finally 保证, 节点 shutdown 也强制恢复)
11. 等待 stable_wait_after 秒      ← 默认 0.3 s
12. 自动 call /us3d/stop_recording ← 在抬起前停录, 避免空气帧入数据集
13. 抬起退回起点上方
```

**关键参数（`config/scan.yaml` 中的 `scan:` 段）：**

*基础运动控制*

- `scan_speed` — 扫描速度（默认 **0.003** = 3 mm/s，配合 28 fps ≈ 0.10 mm/帧）
- `scan_accel_factor` — 加速度倍率（默认 **0.05**，避免接触下 C153 关节加速度保护停）
- `approach_height` — 接近高度（默认 0.05 m）
- `probe_length` — fallback 探头长度（默认 **0.160 m** = 5mm 安装座 + 155mm 探头本体）

*采集质量控制*

- `auto_record` — 自动开关录制（默认 true）
- `stable_wait_before/after` — 录制前后稳定等待秒数
- `zero_ft_sensor` — 每条线开始时悬空清零 F/T（默认 true）
- `hide_cursor` — 扫描时隐藏鼠标避免画面污染（默认 true）

*力控 Touchdown 标定（新）*

- `touchdown_enabled` — 是否启用（默认 true）
- `touchdown_force` — 接触判定阈值，N（默认 **2.5**，越小越温柔）
- `touchdown_speed` — 下降速度，m/s（默认 **0.002** = 2mm/s，越慢越安全）
- `touchdown_max_descent` — 最大下行距离（默认 0.10 m）
- `touchdown_step` — 每个增量步进的距离（默认 0.003 m）
- `touchdown_extra_press` — 接触后额外下压（默认 0，软组织时可设 0.002–0.005）

*Z 控制策略*

- `flat_scan_at_contact_z` — 强制全程平面扫描（默认 false，按 path 跟曲面）
- `max_push_below_contact_mm` — Z 安全 floor（默认 0，不允许探头比 touchdown 接触位置更深）
- `flat_force_vertical` — 平面表面时强制探头竖直（默认 true）
- `flat_use_constant_z` — 平面表面时强制 Z 恒定（默认 false）
- `flat_surface_threshold_mm` — Z 范围阈值（默认 1000，几乎总走 flat 模式）
- `invert_z_trend` — 反转路径 Z 趋势（默认 false，调试用）
- `reverse_scan_direction` — 反转扫描方向（marker 0/1 swap，默认 false）
- `z_smooth_window_mm` — 路径 Z SG 滤波窗口（默认 30 mm）

`**config/robot.yaml`：**

- `robot.ip` — UR5e 控制器 IP（默认 192.168.1.3）

### us3d_vlm — 视觉语言模型（VLM）模块

接 vLLM 风格的 OpenAI 兼容 HTTP 服务（多模态模型，例如 Qwen2-VL / InternVL / Llama-3.2-Vision，或云端 Kimi-k2 等 thinking model），实现两件事：

1. **体模识别**（独立服务）：把相机彩色图发给 VLM，识别体模类型（膝/腹/颈/甲状腺/血管…）并给出像素 bbox。
2. **指令驱动的扫查计划**：把一句话指令（"沿这个膝盖的**长轴**扫 80mm"、"沿短轴扫 60mm"、"对角扫"…）+ 当前彩色图一次发给 VLM，让它**直接产出两个像素端点**（按指令决定方向）+ 扫描长度 / 速度 / 反向。两个端点经 depth 反投影成 base_link 中的 3D 点，作为虚拟 marker 喂给 `scan_planner`，最终生成真实路径。

**安全边界**：VLM 永远不直接产出机器人 waypoint 或关节角；它只能输出 `scan_length_mm / scan_speed_mms / reverse_direction` 这样的标量参数 + 两个像素端点。节点会把数值按 `vlm.yaml` 中的 `min/max_*` 范围 clamp 后写入 `/scan/*` 参数服务器；端点也会按图像尺寸 clamp、避开退化情况。然后让 `scan_planner` 真正算路径，力控 / 避障 / 安全限位流程不被绕过。


| 节点                            | 功能                                                                         | 订阅                                                                           | 发布 / 服务                                                                                                 |
| ----------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `phantom_recognizer_node.py`  | 单独服务化的"看图识别"。**不参与扫查规划**，仅作为开机核对 / 调试用                                     | `/camera/color/image_raw`                                                    | `/us3d/phantom_info` (PhantomInfo)；`/us3d/recognize_phantom` (Trigger)                                  |
| `instruction_planner_node.py` | ★ 一站式入口：相机彩色图 + 自然语言指令 → 单次多模态 VLM 调用 → 同时输出端点 + 扫描参数 → 自动调 anchor + plan_scan | `/camera/color/image_raw`                                                    | `/us3d/phantom_info` (latch, 含 endpoints_xyxy_norm); `/us3d/plan_from_instruction` (PlanFromInstruction) |
| `vlm_anchor_node.py`          | 把 PhantomInfo 的 endpoints (优先) 或 bbox 长轴 (fallback) 端点用 depth 反投影成 3D，**替代 ArUco** | `/us3d/phantom_info`, `/camera/depth/image_raw`, `/camera/color/camera_info` | `/us3d/markers` (PoseArray, latch); `/us3d/vlm_anchor_debug` (Image); `/us3d/anchor_from_vlm` (Trigger) |


**vLLM 服务部署示例**（典型本地部署）：

```bash
# 在另一台带 GPU 的机器或本机 GPU 上启动 vLLM（OpenAI 兼容）
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192
# 也可换 InternVL2 / Llama-3.2-Vision 等任意多模态模型
```

`**config/vlm.yaml` 关键参数**：


| 参数                                              | 默认值                         | 说明                                  |
| ----------------------------------------------- | --------------------------- | ----------------------------------- |
| `vlm.base_url`                                  | `http://127.0.0.1:8000/v1`  | OpenAI 兼容 base URL                  |
| `vlm.api_key`                                   | `EMPTY`                     | vLLM 不校验，但 client 必须填非空字符串          |
| `vlm.model`                                     | `Qwen/Qwen2-VL-7B-Instruct` | 多模态模型名（识别 + 规划共用）                  |
| `vlm.text_model`                                | `""`                        | 已弃用（regplanner 现在也是 vision call，留空即可） |
| `vlm.timeout_s`                                 | 180                         | 单次请求 idle read timeout（开 streaming 时是 chunks 之间最大空闲时间） |
| `vlm.max_retries`                               | 0                           | thinking 模型重试只会更慢，建议 0              |
| `vlm.max_tokens`                                | 0                           | 0 = 不发 max_tokens，让服务端用模型默认上限       |
| `vlm.stream`                                    | true                        | SSE 流式响应（thinking 模型必须开）            |
| `vlm.use_response_format`                       | false                       | 火山方舟 / Moonshot 等云网关常常忽略此参数；只在自家 vLLM / OpenAI 上 true |
| `vlm.phantom_recognizer.rate_hz`                | 0.0                         | 0 = 关后台 timer，仅用 service 触发         |
| `vlm.phantom_recognizer.image_max_side_px`      | 512                         | 发送前长边下采样到此像素                        |
| `vlm.phantom_recognizer.image_jpeg_quality`     | 75                          | JPEG 压缩质量                           |
| `vlm.instruction_planner.image_max_side_px`     | 512                         | 同上，规划用 VLM 调用的图像缩放                  |
| `vlm.instruction_planner.max_scan_length_mm`    | 200                         | LLM 输出扫描长度的安全上限                     |
| `vlm.instruction_planner.min_scan_length_mm`    | 10                          | 安全下限                                |
| `vlm.instruction_planner.max_scan_speed_mms`    | 10                          | 速度安全上限                              |
| `vlm.instruction_planner.use_vlm_anchor`        | true                        | dispatch 前自动调 `/us3d/anchor_from_vlm` |
| `vlm.instruction_planner.apply_to_param_server` | true                        | true=自动改 `/scan/*` 参数；false=只算计划不下发 |
| `vlm.vlm_anchor.bbox_inset_fraction`            | 0.10                        | 端点向中点收缩比例（避开 bbox 边缘的 depth 空洞）    |
| `vlm.vlm_anchor.depth_patch_px`                 | 7                           | 在每个端点周围取 NxN 中位数深度，抗噪声              |


**启动方式**：

```bash
# A. 单独启动 VLM 模块（需要先启动 camera 和 aruco_detector）
roslaunch us3d_bringup vlm.launch

# 也支持命令行覆盖 base_url / model
roslaunch us3d_bringup vlm.launch \
  vlm_base_url:=http://192.168.1.50:8000/v1 \
  vlm_model:=Qwen/Qwen2-VL-7B-Instruct

# B. 跟 full_system 一起拉起（默认关，加 enable_vlm:=true 才启动）
roslaunch us3d_bringup full_system.launch enable_vlm:=true
```

**典型用法**（一句话搞定，不需要再先调 `recognize_phantom`）：

```bash
# 1. 一句话给 VLM：看图 + 决定端点方向 + 出参数 + 派发 plan_scan
rosservice call /us3d/plan_from_instruction "instruction: '沿体模长轴扫 80mm，慢一点 2mm/s'
dry_run: false"
# 内部时序（自动）:
#   ① VLM 单次多模态调用 (image + instruction)
#      → JSON: phantom_type, scan_length_mm, scan_speed_mms,
#              reverse_direction, endpoints_norm, bbox_norm
#   ② publish /us3d/phantom_info (latch, 含 endpoints_xyxy_norm)
#   ③ set /scan/scan_half_length, /scan/scan_speed, /scan/reverse_scan_direction
#   ④ /us3d/anchor_from_vlm: endpoints 反投影到 base_link → /us3d/markers
#   ⑤ /us3d/plan_scan: 用虚拟 markers 生成 RViz 中红色路径
# → message 例: "dispatched: type=knee axis=long len=80mm speed=2.0mm/s
#                 ep=[0.28,0.45]->[0.71,0.55] | downstream: Generated 81
#                 waypoints (..., scan_length=80mm) | anchor[endpoints_norm]:
#                 pixel (175,217)->(454,265) → 3D distance 142.3mm | ..."

# 2. 同样语法切换轴向 / 速度 / 反向
rosservice call /us3d/plan_from_instruction "instruction: '沿短轴扫 60mm'
dry_run: false"
rosservice call /us3d/plan_from_instruction "instruction: '从体模左下角到右上角对角扫 100mm'
dry_run: false"
rosservice call /us3d/plan_from_instruction "instruction: '反向扫一遍'
dry_run: false"

# 3. 像往常一样执行
rosservice call /us3d/preview_scan
rosservice call /us3d/start_scan

# 4.（可选）单独识别（不下发任何参数）
rosservice call /us3d/recognize_phantom    # 仅刷新 /us3d/phantom_info
rqt_image_view /us3d/vlm_anchor_debug      # 看 anchor 抠的端点是否落在体模上
```

**故障排查**：

- VLM 服务连不上 → `/us3d/plan_from_instruction` 返回 `success=false`，message 含 HTTP 状态码 + 错误原文
- LLM 返回非 JSON → 客户端内置三级解析（直接 JSON / 围栏码块 / 多段花括号取最末），还失败时把原文放在 `plan_json` 字段
- 端点不准（落在桌面 / marker 上）→ `rqt_image_view /us3d/vlm_anchor_debug` 看红线，重发指令更明确（如 "沿体模本体的长轴，不要标记物" ）
- depth 在端点处空洞 → anchor 自动 7×7 patch 中位数兜底，仍失败会 `success=false`，重新调一次让 VLM 出新端点
- 想看 LLM 给了什么 → `rostopic echo /us3d/phantom_info -n 1` 里的 `raw_json` 字段包含原始 plan
- 担心安全 → `apply_to_param_server: false`，或在 service 调用里加 `dry_run: true` 看 plan_json 满意再切回

### us3d_acquisition — 同步采集模块


| 节点                      | 功能                                           | 订阅                                                      | 输出                            |
| ----------------------- | -------------------------------------------- | ------------------------------------------------------- | ----------------------------- |
| `sync_recorder_node.py` | 时间同步超声图 + 位姿 + 力，保存到文件；启动后丢弃前 N 帧消除 USB 缓存延迟 | `/us3d/image_raw`, `/us3d/current_pose`, `/us3d/wrench` | PNG 帧 + metadata.csv + rosbag |


**关键参数（`config/scan.yaml` 中的 `recording:` 段）：**

- `output_dir` — 数据集根目录
- `sync_tolerance` — `ApproximateTimeSynchronizer` 时间容差（默认 10 ms）
- `save_rosbag` — 是否同时保存 rosbag 备份
- `warmup_frames` — 启动后跳过前 N 帧（默认 5），用于排空 USB 采集卡缓存的旧帧

**服务：**

- `/us3d/start_recording` — 开始记录（force_scan_node 自动调用）
- `/us3d/stop_recording` — 停止记录（force_scan_node 自动调用）

**输出目录结构：**

```
data/scans/scan_YYYYMMDD_HHMMSS/
├── metadata.csv          # 时间戳, 位姿(6), 力(6)
├── frames/               # 超声帧图像
│   ├── 000000.png
│   └── ...
└── scan.bag              # rosbag 原始数据
```

### us3d_reconstruction — 离线 3D 重建

四个独立脚本，不依赖 ROS 运行时：


| 脚本                           | 算法                                                            | 适用场景                          |
| ---------------------------- | ------------------------------------------------------------- | ----------------------------- |
| `pbm_reconstruct.py`         | **PLUS 风格 PBM**：elevation 方向高斯 splat + 归一化高斯 hole-filling（推荐） | 单线 freehand 3D US 扫描；填充率高、伪影少 |
| `voxel_reconstruct.py`       | 最近邻体素嵌入 + 加权平均                                                | 简单基线；产生稀疏体积                   |
| `tsdf_reconstruct.py`        | Open3D TSDF 融合（实验性，对 US 不严格适用）                                | 需要表面 mesh 输出时勉强能用             |
| `visualize_volume.py`        | 5 种模式：napari/MIP/iso-surface/points/slices                    | 所有 .npy/.ply 通用               |
| `diagnose_outlier_frames.py` | 离群帧检测（位姿 + 亮度异常）                                              | 数据集质量分析                       |


**PBM 重建（推荐）**：

```bash
python3 pbm_reconstruct.py \
    --scan_dir   data/scans/scan_YYYYMMDD_HHMMSS \
    --calib_dir  data/calibration \
    --output     data/reconstructions/pbm_v1.npy \
    --voxel_size 0.0007                  # 0.7 mm/体素
    --elevation_sigma_mm 2.5             # 探头切片厚度的一半
    --intensity_min 15                   # 跳过黑色背景像素
    --hole_fill_sigma_mm 1.5             # 第二遍高斯填洞
    --hole_fill_iter 2                   # 填洞迭代次数
    --force_threshold 1.0                # 仅保留 |Fz|>=1N 的接触帧
    --pose_percentile 0                  # 0=不过滤；改为 5 时砍掉 5%/95% 之外的位姿离群帧
```

**重建关键参数调节指南**：


| 现象          | 调哪个参数                                       | 怎么调                                        |
| ----------- | ------------------------------------------- | ------------------------------------------ |
| 重建出现"百叶窗"条纹 | `--elevation_sigma_mm`                      | 加大到 3.5 ~ 5.0                              |
| 整体太糊、细节丢失   | `--elevation_sigma_mm`                      | 减小到 1.5 ~ 2.0                              |
| 体素空洞多       | `--hole_fill_sigma_mm` 或 `--hole_fill_iter` | sigma 加大到 2.5，iter 加到 3                    |
| 想要更细粒度      | `--voxel_size`                              | 0.0007 → 0.0005 (内存 ×3) → 0.0003 (~130 MB) |
| 接触帧太少       | `--force_threshold`                         | 5.0 → **1.0**（容许轻接触帧）                      |


**3D 可视化（5 种模式 + 旋转/翻转/平滑选项）**：

```bash
# ★ 推荐 - napari 交互式 3D 体积查看器（医疗影像标准工具）
pip install 'napari[all]'

# 全功能：自动对齐扫描方向 + 翻 Y 轴方向 + 显示平滑
python3 visualize_volume.py \
  --volume $OUT/pbm.npy \
  --mode napari \
  --align_to_scan \           # 自动从 scan_*/metadata.csv 算 scan_dir, 旋转 volume 让其对齐 +X
  --flip_y \                  # napari Y 默认向下, base_link Y 向左, 翻一下
  --smooth_display 0.5        # 高斯模糊 0.5 voxel sigma 减少锯齿

# napari 提供:
#   - 拖底部滑块: 逐张 2D 切片浏览（沿任意轴）
#   - 按 2 / 3 切 2D / 3D
#   - rendering 下拉: attenuated_mip(超声风格) / mip / translucent / iso
#   - 3D 模式 + plane rendering: Shift+拖 旋转切面到任意角度

# MIP - 最大密度投影（无需新依赖，看强亮区域分布）
python3 visualize_volume.py --volume $OUT/pbm.npy --mode mip

# Iso-surface - 等值面 mesh（marching cubes, 需要 scikit-image）
python3 visualize_volume.py --volume $OUT/pbm.npy --mode isosurface --isolevel 80 --smooth 2

# Points - 离散点云（速度最快但视觉不连续）
python3 visualize_volume.py --volume $OUT/pbm.npy --mode points --threshold 30

# Slices - 只看 2D 切片网格（最快, 不开 3D 窗口）
python3 visualize_volume.py --volume $OUT/pbm.npy --mode slices --axis 2
```

**可视化选项详解**：


| 选项                                   | 作用                                                                           |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| `--align_to_scan`                    | 自动检测 `scan_dir` 并旋转 volume 让 X 轴 = 扫描方向。然后 napari XY 切片就是垂直于扫描方向的 B-mode 横断面 |
| `--rotate_z DEG`                     | 手动旋转角度（如已知 scan_dir）                                                         |
| `--flip_y` / `--flip_x` / `--flip_z` | 修正 napari 显示的轴向反                                                             |
| `--smooth_display SIGMA`             | Gaussian 模糊（仅显示，不修改 .npy）。0.5-1.5 voxel sigma 能消除锯齿                          |


| 体积里出现孤立亮块（位姿离群） | `--pose_percentile` | 设为 5 ~ 10 |
| 体积偏暗 | `--intensity_min` | 减小到 5 ~ 10 |

### us3d_bringup — 启动与配置

#### Launch 文件


| Launch 文件                  | 功能                 | 用法                                                      |
| -------------------------- | ------------------ | ------------------------------------------------------- |
| `ur5e_bringup.launch`      | UR5e 驱动 + 自定义 URDF | `roslaunch us3d_bringup ur5e_bringup.launch`            |
| `camera.launch`            | Orbbec 相机          | `roslaunch us3d_bringup camera.launch`                  |
| `us_capture.launch`        | 超声 USB 采集          | `roslaunch us3d_bringup us_capture.launch device_id:=0` |
| `calibration.launch`       | 手眼标定 (eye-in-hand) | `roslaunch us3d_bringup calibration.launch`             |
| `full_system.launch`       | 一键启动完整扫描系统         | `roslaunch us3d_bringup full_system.launch`             |
| `real_robot_moveit.launch` | 实机 + MoveIt + RViz | `roslaunch us3d_bringup real_robot_moveit.launch`       |
| `gazebo_sim.launch`        | Gazebo + MoveIt 仿真 | `roslaunch us3d_bringup gazebo_sim.launch`              |
| `vlm.launch`               | VLM 体模识别 + 指令规划    | `roslaunch us3d_bringup vlm.launch`                     |


#### 配置文件

- `config/robot.yaml` — 机器人 IP、端口、安全限位、力控参数
- `config/scan.yaml` — 扫描间距/速度、超声设备参数、ArUco 字典、曲面拟合参数、观测位姿、录制配置
- `config/home_position.yaml` — 起始位置关节角度（由 `/us3d/record_home` 自动生成）
- `config/ur5e_calibration.yaml` — 从机器人提取的运动学标定参数
- `config/us3d.rviz` — RViz 可视化配置（机器人模型、点云、扫查路径、力矢量等）
- `config/gazebo_moveit_controllers.yaml` — Gazebo 仿真 MoveIt 控制器映射
- `us3d_vlm/config/vlm.yaml` — VLM 端点 URL、模型名、识别频率、指令规划安全限位

#### URDF 文件

- `urdf/ur5e_with_probe.urdf.xacro` — 实机用：UR5e + 超声探头 + 相机
- `urdf/ur5e_with_probe_gazebo.urdf.xacro` — 仿真用：附加 gazebo_ros_control 插件和传输层

---

## 使用流程

### 1. 标定

```bash
# 提取 UR5e 工厂运动学标定
roslaunch ur_calibration calibration_correction.launch \
  robot_ip:=192.168.1.3 \
  target_filename:=$(rospack find us3d_bringup)/config/ur5e_calibration.yaml

# 启动手眼标定
roslaunch us3d_bringup calibration.launch

# 另一终端启动标定辅助节点（可选，自动采样）
rosrun us3d_control hand_eye_calib_node.py
rosservice call /us3d/calib_connect
rosservice call /us3d/calib_collect
```

### 2. 操作真实机械臂

#### 前置条件（示教器端）

1. 示教器 Settings > System > Network 设置 IP（如 `192.168.1.3`）
2. 安装 ExternalControl URCap（文件在 `/opt/ros/noetic/share/ur_robot_driver/resources/`）
3. 创建程序，添加 External Control 节点，Host IP 设为电脑 IP（`192.168.1.102`）

#### 方式 A：分步启动（推荐调试用）

```bash
# 终端1: 机械臂驱动
roslaunch us3d_bringup ur5e_bringup.launch

# → 示教器上运行 External Control 程序（点播放按钮）
# → ROS 终端出现 "Robot connected to reverse interface"

# 终端2: MoveIt 规划
roslaunch ur5e_moveit_config moveit_planning_execution.launch

# 终端3: RViz 可视化
roslaunch ur5e_moveit_config moveit_rviz.launch

# 终端4: 添加桌面碰撞体（避障）
rosrun us3d_control add_scene_objects.py
```

#### 方式 B：一键启动

```bash
roslaunch us3d_bringup real_robot_moveit.launch
# → 示教器上运行 External Control 程序
```

#### 在 RViz 中操作

1. 确认 MotionPlanning 面板的 Start State 为 **current**
2. 拖动橙色交互球设置目标位姿
3. **Plan** 预览路径 → **Execute** 执行（机械臂真实运动）

#### 安全注意事项

- 首次运行时示教器速度滑块调到 **10%**
- 手持急停按钮随时准备
- `robot.yaml` 安全限位：最大力 20N、最大速度 0.25 m/s
- 每次重启 ROS 驱动后需在示教器上重新点播放

### 3. 起始位置设置

首次使用时，手动将机械臂移到合适的起始位置（相机朝下正对体模），然后记录：

```bash
roslaunch us3d_bringup full_system.launch
# → 示教器运行 External Control

# 记录当前位置为起始位（保存到 config/home_position.yaml）
rosservice call /us3d/record_home

# 之后每次启动系统，节点会自动移动到起始位置
# 也可手动触发
rosservice call /us3d/move_home
```

> 注意：起始位置的记录和移动仅需 ExternalControl 模式，无需 Remote Control。

### 4. ArUco 标记准备

```bash
# 生成标记图片（输出到 data/aruco_markers/）
python3 tools/generate_aruco_markers.py
```

打印 `data/aruco_markers/print_sheet.png`（实际打印尺寸为 75mm x 75mm），剪下后贴在体模表面四角，围出扫查区域。

### 5. ★ 推荐：vLLM 自然语言扫查（无需贴 ArUco）

整个 vLLM + 力控扫描的端到端工作流。一条 service 同时完成识图、决定扫查方向、生成路径，再交给力控节点接管。

#### 5.1 启动

```bash
# 启动全系统 + VLM（默认关，加 enable_vlm:=true 才启动 instruction_planner / vlm_anchor）
roslaunch us3d_bringup full_system.launch enable_vlm:=true \
  vlm_base_url:=http://127.0.0.1:8000/v1
# 也可对接云端 OpenAI 兼容服务（Kimi-k2 / Doubao / Qwen-VL …）：
roslaunch us3d_bringup full_system.launch enable_vlm:=true \
  vlm_base_url:=https://ark.cn-beijing.volces.com/api/coding/v3 \
  vlm_model:=kimi-k2.6
# → 示教器运行 External Control，等待 "Robot connected to reverse interface"
```

#### 5.2 一句话生成扫查路径

```bash
# 沿长轴扫 80mm，慢速 2mm/s（最常见）
rosservice call /us3d/plan_from_instruction "instruction: '沿体模长轴扫 80mm，慢一点 2mm/s'
dry_run: false"

# 沿短轴扫 60mm
rosservice call /us3d/plan_from_instruction "instruction: '沿短轴扫 60mm'
dry_run: false"

# 自由方向（VLM 自己挑两个像素端点）
rosservice call /us3d/plan_from_instruction "instruction: '从体模左下角到右上角对角扫 100mm'
dry_run: false"

# 反向重扫一次（用同一对端点，order 翻过来）
rosservice call /us3d/plan_from_instruction "instruction: '反过来扫一遍'
dry_run: false"
```

`message` 字段返回类似：

```
dispatched: type=knee axis=long len=80mm speed=2.0mm/s rev=False
ep=[0.28,0.45]->[0.71,0.55]
| downstream: Generated 81 waypoints (..., scan_length=80mm)
| anchor[endpoints_norm]: pixel (175,217)->(454,265) → 3D distance 142.3mm |
  p0=(-0.527,0.177,0.002) p1=(-0.351,0.067,0.001) (depths 0.530 / 0.530 m)
```

里面同时能看到 VLM 给的端点 + 反投影后的 3D 距离 + 生成的 waypoint 数。RViz 中**红色箭头**就是规划的路径。

#### 5.3 预览 + 力控执行

```bash
# 预览（可选但强烈推荐，检查路径方向 + 探头朝向）
rosservice call /us3d/preview_scan
# → RViz 中看到完整轨迹动画（含手腕旋转）
#   日志: "Scan attitude: rotating wrist by XX.X°"
#   如果角度过大（>120°），先 record_home 调整起始姿态再重试

# ★ 力控执行（详见 us3d_control 章节"start_scan 自动时序"）
rosservice call /us3d/start_scan
# 自动 8 步: SLERP approach → F/T 清零 → 力控 touchdown →
#            speed slider 降速 → 沿曲面扫描 + force-adaptive 调 Z →
#            自动录制 → speed slider 恢复 → 抬起
# → rqt_plot 实时显示力曲线
```

扫描数据保存到 `data/scans/scan_YYYYMMDD_HHMMSS/`，可直接进 PBM 重建。

#### 5.4 故障排查

> **VLM 没出端点 / 抠到桌面**：`rqt_image_view /us3d/vlm_anchor_debug` 看红线是否落在体模上；重发更明确的指令（如"沿体模本体的长轴，避开标记物"）。
>
> **VLM 调用慢 / 不出结果**：thinking 模型 streaming 中节点会每 5s 打 `VLM streaming... Xs elapsed, reasoning=N chunks`。完全没心跳通常是网络层问题（VPN、socket 残留）→ `rosnode kill /instruction_planner` 重启即可。
>
> **dry_run 模式**：在 service call 里改 `dry_run: true`，节点只计算 plan 不派发，返回的 `plan_json` 包含结构化参数（用来核对 LLM 是否理解对了）。
>
> **回到 ArUco 流程**：在 `vlm.yaml` 把 `instruction_planner.use_vlm_anchor: false`，再按下面 §5.5 走 ArUco 路。
>
> **同时支持 VLM + ArUco**：默认 `use_vlm_anchor: true` 时，一旦 VLM 调用失败 anchor 也会返回 `success=false`，但 `/us3d/markers` 上如果有 ArUco 检测的旧值，会被 scan_planner 直接拿来用——所以贴了 ArUco 也不冲突，VLM 只是优先级更高。

### 5.5 经典 ArUco 流程（fallback）

VLM 不可用时仍可贴 ArUco 走老路：

```bash
# 启动（不要加 enable_vlm:=true）
roslaunch us3d_bringup full_system.launch
# → 示教器运行 External Control

rosservice call /us3d/detect_markers   # 检测 ArUco
rosservice call /us3d/plan_scan        # 生成路径
rosservice call /us3d/preview_scan
rosservice call /us3d/start_scan
```

`detect_markers` 之前需要先用 [§4. ArUco 标记准备](#4-aruco-标记准备) 打印 + 贴标记。

### 6. RViz 可视化说明

启动后 RViz 自动显示：


| 显示项           | 说明                   |
| ------------- | -------------------- |
| RobotModel    | UR5e + 探头 + 相机模型     |
| TF            | 坐标系树                 |
| ColorCamera   | 彩色图像                 |
| DepthCamera   | 深度图像                 |
| Ultrasound    | 超声图像                 |
| DepthCloud    | 原始深度点云               |
| SurfaceCloud  | 裁剪后的体模表面点云（绿色）       |
| ScanWaypoints | 扫查路径箭头（红色，方向 = 探头法线） |
| ForceWrench   | 力矢量箭头（扫描时实时显示）       |


力曲线图由 `rqt_plot` 单独窗口显示 Fz 随时间变化。

### 7. 离线重建

```bash
cd catkin_ws/src/us3d_reconstruction/scripts
SCAN=~/joe/us3dscan/data/scans/scan_YYYYMMDD_HHMMSS
CALIB=~/joe/us3dscan/data/calibration
OUT=~/joe/us3dscan/data/reconstructions

# 推荐: PLUS 风格 PBM 重建（高填充率、少伪影）
python3 pbm_reconstruct.py \
  --scan_dir $SCAN --calib_dir $CALIB \
  --output $OUT/pbm.npy \
  --voxel_size 0.0007 \
  --elevation_sigma_mm 2.5 \
  --intensity_min 15 \
  --hole_fill_sigma_mm 1.5 --hole_fill_iter 2 \
  --force_threshold 1.0

python3 pbm_reconstruct.py \
  --scan_dir $SCAN --calib_dir $CALIB \
  --output $OUT/pbm_full.npy \
  --voxel_size 0.0005 \
  --elevation_sigma_mm 2.0 \
  --intensity_min 15 \
  --hole_fill_sigma_mm 1.0 --hole_fill_iter 1 \
  --force_threshold 1.0 \
  --algo v2 --trilinear \
  --auto_lag \
  --smooth_window 11 \
  --tgc_eq --tgc_gamma 0.5 \
  --despeckle median --despeckle_ksize 3

# 数据质量诊断（找出位姿/亮度异常帧）
python3 diagnose_outlier_frames.py \
  --scan_dir $SCAN --calib_dir $CALIB --force_threshold 1.0

# 简单基线方法（作对比用）
python3 voxel_reconstruct.py --scan_dir $SCAN --calib_dir $CALIB \
  --output $OUT/voxel.npy --voxel_size 0.0005

# 可视化 - napari 交互式（推荐）
pip install 'napari[all]'         # 一次性安装
python3 visualize_volume.py --volume $OUT/pbm.npy --mode napari

# 或快速看 2D 切片
python3 visualize_volume.py --volume $OUT/pbm.npy --mode slices --axis 2
```

---

## Gazebo + MoveIt 仿真

无需连接实际硬件，在仿真环境中验证运动规划。

### 一键启动

```bash
roslaunch us3d_bringup gazebo_sim.launch
```

启动后同时打开 Gazebo（UR5e + 探头 + 相机）、MoveIt move_group、RViz。

### 可选参数

```bash
roslaunch us3d_bringup gazebo_sim.launch gui:=false      # 无 Gazebo GUI
roslaunch us3d_bringup gazebo_sim.launch paused:=true     # 启动时暂停
```

> 注意：Gazebo 不模拟力/力矩传感器，`force_scan_node.py` 不适用于仿真。仿真主要用于验证运动规划和坐标变换。

---

## 避障配置

### 手动添加碰撞体（桌面等已知障碍）

MoveIt 启动后，在新终端运行：

```bash
rosrun us3d_control add_scene_objects.py
```

可调参数：

```bash
rosrun us3d_control add_scene_objects.py \
  _table_height:=-0.02 \
  _table_size_x:=1.5 \
  _table_size_y:=1.5 \
  _back_wall_distance:=0.3
```

RViz 中会显示绿色碰撞体，规划路径自动绕开。

### Octomap 动态避障（深度相机）

已配置 MoveIt 的 sensor_manager 订阅 `/camera/depth/points` 自动构建 Octomap。启动相机后自动生效：

```bash
roslaunch us3d_bringup camera.launch
```

RViz 中可通过 MotionPlanning 的 Scene Objects 查看 Octomap 体素。

---

## 坐标系

```
base_link (机器人基座)
 └── tool0 (法兰盘, UR 正运动学)
      ├── camera_link (手眼标定: T_tool0_cam)
      │    ├── camera_color_optical_frame
      │    └── camera_depth_optical_frame
      └── probe_tip (URDF 固定变换: T_tool0_probe)
           └── us_image_plane (超声空间标定: T_probe_us)
```

## 关键参数

**机器人与扫描运动**


| 参数      | 默认值         | 说明                                                                                                                                                |
| ------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 机器人 IP  | 192.168.1.3 | `robot.ip`                                                                                                                                        |
| 扫描速度    | **3 mm/s**  | `scan.scan_speed`（接触状态下越慢越柔。实际通过 UR speed slider 实现。MoveIt 自带的 retime/velocity_scaling 在 Cartesian path 上不可靠，**改用 UR speed slider 全局降速**，仅在扫描段生效） |
| 扫描加速度倍率 | **0.05**    | `scan.scan_accel_factor`（防止 C153 关节加速度保护停）                                                                                                        |
| 速度参考    | 20 mm/s     | `scan.speed_slider_reference_mms`（slider=1.0 时 Cartesian 自然速度，用于反算 slider 比例 = scan_speed/此值）                                                     |
| 扫描线间距   | 3 mm        | `scan.line_spacing`                                                                                                                               |
| 路点插值步长  | 1 mm        | `scan.waypoint_step`                                                                                                                              |
| 接近高度    | 50 mm       | `scan.approach_height`                                                                                                                            |
| 探头长度    | **160 mm**  | `scan.probe_length`（5mm 安装座 + 155mm 探头本体）                                                                                                         |
| 伺服频率    | 125 Hz      | `scan.servo_frequency`                                                                                                                            |


**力控 Touchdown 标定（新增）**


| 参数      | 默认值        | 说明                                    |
| ------- | ---------- | ------------------------------------- |
| 是否启用    | true       | `scan.touchdown_enabled`              |
| 接触判定阈值  | **2.5 N**  | `scan.touchdown_force`（`              |
| 下降速度    | **2 mm/s** | `scan.touchdown_speed`（增量步进 + 每步阻塞）   |
| 最大下行距离  | 100 mm     | `scan.touchdown_max_descent`          |
| 增量步距    | 3 mm       | `scan.touchdown_step`（每步同步执行，结束机器人静止） |
| 接触后额外下压 | 0 mm       | `scan.touchdown_extra_press`          |


**Z 控制策略（新增）**


| 参数           | 默认值     | 说明                                                       |
| ------------ | ------- | -------------------------------------------------------- |
| 强制平面扫描       | false   | `scan.flat_scan_at_contact_z`（true = 全程用 touchdown 接触 Z） |
| Z 安全 floor   | 0.0 mm  | `scan.max_push_below_contact_mm`（不允许探头比接触位置更深）           |
| 平面表面强制竖直     | true    | `scan.flat_force_vertical`                               |
| 平面表面 Z 恒定    | false   | `scan.flat_use_constant_z`                               |
| Flat 模式 Z 阈值 | 1000 mm | `scan.flat_surface_threshold_mm`（默认几乎总走 flat）            |
| Z 平滑窗口       | 30 mm   | `scan.z_smooth_window_mm`（SG 滤波）                         |
| 反转 Z 趋势      | false   | `scan.invert_z_trend`（调试 workaround）                     |
| 反转扫描方向       | false   | `scan.reverse_scan_direction`（swap marker 0/1）           |


**Force-adaptive scan（poor-man's force control，新增）**

每段扫完后查 |Fz|，丢接触就降 Z 继续。是 forceMode 真力控的简化版，不需要切控制器。


| 参数   | 默认值         | 说明                                               |
| ---- | ----------- | ------------------------------------------------ |
| 启用   | true        | `scan.force_adaptive_scan`（false = 单次预规划执行）      |
| 接触阈值 | **2.0 N**   | `scan.force_adaptive_min_force`（                 |
| 单次降幅 | **1.0 mm**  | `scan.force_adaptive_step`（每次 z_offset 减少多少）     |
| 累计上限 | **10.0 mm** | `scan.force_adaptive_max_push`（z_offset 不超过 −此值） |
| 段间稳定 | 0.15 s      | `scan.force_adaptive_settle_s`（执行后等多久再读力）        |


逻辑：z_offset 单调下降（只压不抬），上限是 max_push。安全 floor 同步下移确保新位置可达。

**采集质量控制**


| 参数           | 默认值   | 说明                                       |
| ------------ | ----- | ---------------------------------------- |
| 自动录制         | true  | `scan.auto_record`                       |
| 录制前稳定等待      | 0.7 s | `scan.stable_wait_before`                |
| 录制后稳定等待      | 0.3 s | `scan.stable_wait_after`                 |
| F/T 清零       | true  | `scan.zero_ft_sensor`                    |
| 隐藏鼠标         | true  | `scan.hide_cursor`                       |
| 同步容差         | 10 ms | `recording.sync_tolerance`               |
| 启动 warmup 帧数 | 5     | `recording.warmup_frames`（丢弃 USB 缓存里的旧帧） |


**标定与超声**


| 参数         | 默认值               | 说明                                   |
| ---------- | ----------------- | ------------------------------------ |
| 超声采集设备     | `/dev/us_capture` | `ultrasound.device_id`（udev 符号链接，稳定） |
| 全黑画面告警     | 3 s               | `ultrasound.blank_warn_secs`         |
| 超声 ROI x   | 165               | `us_roi.x`                           |
| 超声 ROI y   | 62                | `us_roi.y`                           |
| 超声 ROI 宽   | 360               | `us_roi.width`                       |
| 超声 ROI 高   | 220               | `us_roi.height`                      |
| 像素物理尺寸 X   | 0.111 mm          | `pixel_size_x`                       |
| 像素物理尺寸 Y   | 0.173 mm          | `pixel_size_y`                       |
| ArUco 标记尺寸 | 75 mm             | `aruco.marker_size`                  |


**点云裁剪 + 表面提取（新增）**


| 参数           | 默认值        | 说明                                                                  |
| ------------ | ---------- | ------------------------------------------------------------------- |
| 曲面拟合体素       | 2 mm       | `surface_fitting.voxel_size`                                        |
| Crop XY 外扩   | **60 mm**  | `surface_fitting.crop_margin`（marker 周围 XY 扩展，太小则 marker 旁边的体模被切掉）  |
| Crop Z 范围    | **300 mm** | `surface_fitting.crop_z_extent_min`（Z 必须够高才能捕获体模顶面）                 |
| 启用顶层提取       | true       | `surface_fitting.extract_top_layer`（去除桌面/背景点）                       |
| 顶层 gap 阈值    | **3 mm**   | `surface_fitting.top_layer_min_gap_mm`（用于 gap-detection 策略）         |
| 顶层保留比例       | **30%**    | `surface_fitting.top_layer_keep_percent`（fallback 策略：按 Z 保留 top N%） |
| 顶层最少点数       | 8          | `surface_fitting.top_layer_min_points`                              |
| Outlier 邻居数  | 20         | `surface_fitting.outlier_nb_neighbors`                              |
| Outlier σ 倍数 | 2.0        | `surface_fitting.outlier_std_ratio`                                 |


**表面提取流程**：

```
原始 cloud → crop (XY 60mm 外扩, Z 300mm 范围) 
          → voxel downsample (2mm) 
          → top-layer 提取
              1. gap-detection: 找最大间隙 ≥ 3mm 且能保留 ≥ 8 个点 → 用作 table↔phantom 分界
              2. fallback: 按 Z 排序保留 top 30%
          → statistical outlier filter
          → RBF 拟合
```

**避障**


| 参数                | 默认值    | 说明                                        |
| ----------------- | ------ | ----------------------------------------- |
| Octomap min_range | 0.30 m | sensor_manager（过滤近距离点云，防 eye-in-hand 自碰撞） |
| Octomap 分辨率       | 25 mm  | `sensor_manager`                          |


## 分步调试

```bash
# 1. 验证 ROS 消息
rosmsg show us3d_msgs/USFrame
rosmsg show us3d_msgs/ScanDataPoint
rosmsg show us3d_msgs/ScanRegion

# 2. 单独测试相机
roslaunch us3d_bringup camera.launch
rostopic hz /camera/color/image_raw

# 3. 单独测试超声采集
ls /dev/us_capture                # 确认 udev rule 装好
roslaunch us3d_bringup us_capture.launch
rostopic hz /us3d/image_raw       # 应 ~30 Hz
rqt_image_view /us3d/image_raw    # 看到画面

# 4. 单独测试机械臂
roslaunch us3d_bringup ur5e_bringup.launch
# → 示教器运行 External Control
rostopic echo /joint_states
rostopic echo /wrench
rosservice call /ur_hardware_interface/zero_ftsensor   # 验证 F/T 清零可用
```

## 故障排查

### 超声画面全黑（mean=0）

1. **超声机器没开** → 看屏幕是否亮、有无 B-mode 画面
2. **HDMI 线松了** → 拔下来重插**两端**（超声机端 + 采集卡端）
3. **超声机视频输出关了** → 在机器菜单里启用 `Video Output`
4. **绕过 ROS 直接验证**：`ffplay -f v4l2 -input_format mjpeg -video_size 640x480 /dev/us_capture`
5. `us_capture_node` 会在画面持续黑 3s 后打印 WARN，留意日志

### `/dev/us_capture` 不存在

1. udev rule 未装：`sudo cp tools/99-us-capture.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules && sudo udevadm trigger`
2. 序列号不对：`lsusb -v 2>/dev/null | grep -i -A3 macrosilicon`，把 `iSerial` 字段填到 rules 里
3. 临时 fallback：`rosrun us3d_perception us_capture_node.py _device_id:=0`

### `Failed to reach approach position` 或扫描手腕角度过大

日志中 `Scan attitude: rotating wrist by XX.X°` 显示需要旋转的角度。

- 角度 > 120° 时 wrist3 关节可能转不过去 → 重新设置 home 位姿，让起始姿态更接近扫描方向
- `rosservice call /us3d/move_home` 调整后 `rosservice call /us3d/record_home` 保存

### 重建结果稀疏 / "百叶窗"伪影

- 用 PBM 重建（不要用 voxel_reconstruct）：`pbm_reconstruct.py`
- 调大 `--elevation_sigma_mm` 到 3.5-5.0
- 调大 `--hole_fill_sigma_mm` 到 2.5，`--hole_fill_iter` 到 3
- 检查扫描速度是否过快（应 ≤ 5 mm/s）

### 数据集里有"飞出去"的位姿离群帧

- 用 `diagnose_outlier_frames.py` 找出哪些帧
- 重建时加 `--pose_percentile 5` 砍掉
- 根本解决：用最新的 `force_scan_node`（自动 stable_wait 已经避免了头尾过渡帧入数据集）

### Touchdown 触发 UR PROTECTIVE_STOP（C153/C157）

- **C153 = 关节加速度超限**：把 `scan.scan_accel_factor` 调到 0.05 或更小
- **C157 = 关节力矩超限**：通常是探头被压进表面太深
  - 检查 `Scan geometry check` 日志里的 `target probe_tip.z (last)` 是不是远低于 first
  - 如果是：`max_push_below_contact_mm: 0` 严格不允许探头深压（默认）
  - 检查 `touchdown_force` 是否过大（建议 2.5–3 N）

### 路径 Z 趋势看着不对

- 在 RViz 里同时看：
  - 白色 raw cloud `/camera/depth/points`（应该有体模点）
  - 绿色 surface cloud `/us3d/surface_cloud`（filter 后）
  - 红色 path arrows `/us3d/scan_path`
- 如果绿色 cloud 没贴上体模 → **crop 漏了体模**：
  - 加大 `surface_fitting.crop_margin`（默认 60mm，可加到 100mm）
  - 加大 `surface_fitting.crop_z_extent_min`（默认 300mm）
- 如果绿色 cloud 在体模上但 path 反向 → 看 `RAW path Z (BEFORE smoothing)` 诊断行
  - 三个趋势诊断（DEPTH-CLOUD / RAW path / SMOOTHED）应该一致
  - 都反向 → marker 摆放方向问题 → `reverse_scan_direction: true`
- 极端情况：开 `invert_z_trend: true` 或 `flat_scan_at_contact_z: true` 应急

### 探头不接触表面 / 探头压不上 / "翘起"

- 可能是 path Z 上升超过实际表面 → 探头悬空
- 默认 `max_push_below_contact_mm: 0` + `flat_force_vertical: true` 已尽量避免
- 终极方案：用 forceMode 真力控（接触力恒定，Z 自动调），见 README 待办章节

### Octomap 阻挡 approach

- 错误：`Found a contact between '<octomap>' and 'camera_mount'`
- 自动修复：`force_scan` 在 approach 前会 `/clear_octomap`
- 长期修复：`ur5e_moveit_sensor_manager.launch.xml` 已配 `min_range: 0.30` 过滤近距离点
- 还失败：把相机移远点 / 关闭 octomap：在 launch 里去掉 sensor_plugin

### 扫描后机械臂动得很慢（move_home 超时）

- 上次扫描中断了，UR speed slider 卡在低速（force_scan 没执行 `_restore_speed_slider`）
- 立即修复：`rosservice call /us3d/restore_speed`
- 或直接用 UR 接口：`rosservice call /ur_hardware_interface/set_speed_slider "speed_slider_fraction: 1.0"`
- 长期：force_scan_node 的 `try/finally` + `on_shutdown` 会保护，但中断时机不对仍可能漏

### 中间一段帧被过滤（reconstruction 缺一段）

- 跑 `python3 -c` 脚本看 metadata.csv 中间是不是 |Fz| < threshold
- 如果是 → 探头中途脱接触：
  - 启用 `force_adaptive_scan: true`（默认开），让 z_offset 累积下压
  - 或重建时降低过滤阈值：`--force_threshold 0.3`
  - 或目测确认探头其实贴着，用 `--force_threshold 0.0`（仅靠图像 intensity 过滤）
- 视觉确认：看 frames 中间几张 PNG，有清晰超声纹理 = 还在贴

### napari 显示斜的 / 上下颠倒 / 模糊

- 斜的：加 `--align_to_scan` 自动旋转
- 颠倒：加 `--flip_y`（最常见）或 `--flip_z` / `--flip_x`
- 模糊/锯齿：
  - 短期：加 `--smooth_display 0.5` 显示平滑
  - 长期：重建用更小 voxel `--voxel_size 0.0004`

## 技术栈

- **机器人:** ROS Noetic, ur_robot_driver, ur_rtde（fallback 用）, MoveIt
- **仿真:** Gazebo, ros_control
- **视觉:** OpenCV (ArUco), Orbbec SDK, easy_handeye
- **避障:** MoveIt PlanningScene, Octomap
- **重建:** PBM + Gaussian hole-filling（PLUS toolkit 风格），NumPy/SciPy；Open3D（TSDF/可视化）
- **采集稳定性:** xdotool（隐藏鼠标）, udev rules（稳定设备名）
- **多模态 LLM:** vLLM + Qwen2-VL / InternVL（OpenAI 兼容 HTTP），requests 客户端
- **语言:** Python 3.8

