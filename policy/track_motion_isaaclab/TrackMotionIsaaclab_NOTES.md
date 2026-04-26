# TrackMotionIsaaclab 实现笔记

日期：2026-04-20

目的：记录在当前框架下新增 `TrackMotionIsaaclab` 状态（sim2sim）时的实现思路、关键改动和参考来源，便于复现与维护。

**概览**
- 本次实现在 `policy/track_motion_isaaclab/` 下新增了状态实现、配置与模型：
  - 状态代码：[policy/track_motion_isaaclab/TrackMotionIsaaclab.py](policy/track_motion_isaaclab/TrackMotionIsaaclab.py)
  - 配置文件：[policy/track_motion_isaaclab/config/TrackMotionIsaaclab.yaml](policy/track_motion_isaaclab/config/TrackMotionIsaaclab.yaml)
  - ONNX 模型：位于 [policy/track_motion_isaaclab/model/](policy/track_motion_isaaclab/model/)

**实现来源与参考**
- 主要参考并改写自已有的 `TableTennis` 实现：
  - [policy/table_tennis/TableTennis.py](policy/table_tennis/TableTennis.py)
  - 比较点：初始化结构、配置加载、关节顺序映射、obs 构建、ONNX 推理流程、enter/run/exit/checkChange 逻辑。
- 使用的公共类型与枚举：
  - `StateAndCmd`, `PolicyOutput`（来自 `common/ctrlcomp.py`）
  - `FSMStateName`, `FSMCommand`（来自 `common/utils.py`）
  - FSM 集成点：已在 [FSM/FSM.py](FSM/FSM.py) 中 import 并实例化 `TrackMotionIsaaclab`（见 `self.track_motion_isaaclab_policy = TrackMotionIsaaclab(...)`）

**关键实现要点（按模块）**
- TrackMotionIsaaclab 类（policy/track_motion_isaaclab/TrackMotionIsaaclab.py）
  - 结构上沿用了 `TableTennis` 的 FSMState 模板：初始化 config、构造 `mj_joint_names` 与 `train_joint_names`、创建 `mj_to_train` / `train_to_mj` 重排索引。
  - 配置读取：优先读取 `policy/track_motion_isaaclab/config/TrackMotionIsaaclab.yaml`，并通过 `_resolve_path` 支持相对/绝对路径与 PROJECT_ROOT 回退。
  - ONNX 加载：使用 `onnxruntime.InferenceSession` 做推理，支持 `use_external_data`（当模型带外部 `.data` 时需要为 True 并携带 `.data` 文件）。
  - 元数据补全：实现 `_fill_from_onnx_metadata_if_needed()`，会从 ONNX metadata 读取 `joint_names` / `default_joint_pos` / `joint_stiffness` / `joint_damping` / `action_scale`，在 config 未提供或为默认值时填充。
  - 观测（obs）构建：定义 `term_dims` 与 `term_order`（与 TableTennis 类似但项不同），通过 `_build_obs()` 将历史帧 roll 并拼接成最终 `obs`，并做 shape 校验（`obs_dim * history_length == num_obs`）。
  - 控制输出：从 ONNX 输出得到动作后做 `action_scale` 与 `default_angles` 组合，映射回 MuJoCo 关节顺序，输出 `policy_output.actions/kps/kds`。

- 配置文件（policy/track_motion_isaaclab/config/TrackMotionIsaaclab.yaml）
  - 核心字段：`onnx_path`, `num_actions`, `num_obs`, `obs_dim`, `history_length`, `motion_length`, `kps`, `kds`, `default_angles`, `tau_limit`, `action_scale`, `use_external_data` 等。
  - 注意 `use_external_data` 与 `onnx_path` 的对应关系（若模型导出为外部数据，需启用并提供 `.data` 文件）。

- FSM 集成（FSM/FSM.py）
  - 将 `TrackMotionIsaaclab` import 到 FSM，构造时传入 `state_cmd` 与 `policy_output`，并在 `get_next_policy` 中处理 `FSMStateName.SKILL_TRACK_MOTION_ISAACLAB` 的切换逻辑。

**我如何从其他文件/状态借鉴并修改得到 TrackMotionIsaaclab**
- 复制/对比：从 `policy/table_tennis/TableTennis.py` 复制 FSMState 模板（init / _load_policy / _build_obs / run / exit / checkChange），逐行对比并替换：
  - joint 列表与训练维度不同，建立 `mj_to_train`/`train_to_mj` 来重排 config 中的数组（kps/kds/default_angles/action_scale/tau_limit）。
  - 观察项（term_dims/term_order）根据 track_motion_isaaclab 任务调整（包含 racket 相关项），并相应修改 `_build_obs()` 的内容与默认命令读取。
  - ONNX 读写细节：TableTennis 假定 single-input single-output；TrackMotionIsaaclab 支持多个输入名（`obs`, `time_step` 等），并对 inputs/outputs 名称、shape 做额外检查与兼容处理。
  - metadata 补全：新增了 `_parse_csv_metadata` 与 `_fill_from_onnx_metadata_if_needed`，以便在模型导出时携带训练时的参数可自动填回配置缺省值。

**关键注意事项 & 常见坑**
- 关节顺序不一致是最常见的 bug：务必确认 config（YAML）中数组的原始顺序（通常为 MuJoCo actuator 顺序），并使用 `mj_to_train` 做重排，或在 YAML 中直接写训练顺序。
- ONNX 外部数据：如果 ONNX 是按外部数据导出（大模型分 `.onnx` + `.onnx.data`），必须设置 `use_external_data: true` 且把 `.data` 文件一并放置，`onnx.load(..., load_external_data=True)` 才能成功。
- obs/action 维度校验：`obs_dim * history_length` 必须等于 `num_obs`，ONNX 的输入/输出 shape 也需要与 config 一致；确保在 `_load_policy()` 做早期检查。
- 配置数组长度：`kps/kds/default_angles/action_scale/tau_limit` 的长度需为 1 或 `num_actions`，并在初始化时根据 `mj_to_train` 重排。
- 推理热身：ONNX session 需要用零张量做几次 warm-up，避免第一次推理延时影响时序逻辑。

**快速测试建议**
- 在本地 MuJoCo 仿真或 sim2sim 环境中，把 FSM 切换到 `SKILL_TRACK_MOTION_ISAACLAB`，观察：
  - 是否能成功加载 ONNX（日志中会打印初始化信息或错误原因）
  - `policy_output.actions` 是否按期望输出（非默认角），kps/kds 是否被正确映射
  - 若使用外部 data，请确认 `policy/track_motion_isaaclab/model/policy_*.onnx.data` 可访问

**相关文件索引（便于复查）**
- [policy/track_motion_isaaclab/TrackMotionIsaaclab.py](policy/track_motion_isaaclab/TrackMotionIsaaclab.py)
- [policy/track_motion_isaaclab/config/TrackMotionIsaaclab.yaml](policy/track_motion_isaaclab/config/TrackMotionIsaaclab.yaml)
- [policy/track_motion_isaaclab/model/](policy/track_motion_isaaclab/model/)
- [policy/table_tennis/TableTennis.py](policy/table_tennis/TableTennis.py)
- [FSM/FSM.py](FSM/FSM.py)
- [common/utils.py](common/utils.py)

如果需要，我可以：
- 将本笔记转为仓库根目录下的文档（README 或 docs/），或生成一份变更补丁清单（git diff 风格）。
- 根据这个笔记再生成一个快速验收脚本来跑一次 sim2sim 加载与推理测试。

—— 记录结束
