# v0.6 · 逐 GPU 训练执行基准

本轮将训练从“所有 Worker 共用阶段标签”改为按物理 GPU、micro batch 和通信依赖推进。协议为 `rank-events-v1`，新训练使用 `adam-v2` 与 `document-chunks-v2`；历史作业保留原执行协议。

## 固定工程对象与接口

Atlas 70B Dense：80 层、64 个注意力头、8 个 KV 头、hidden size 8192、BF16。默认集群为四台计算节点，每台八张 H100，另有管理和存储节点。TP≤8；TP/PP/CP/DP 乘积必须等于实际申请的 GPU 数，PP 必须整除 80 层。

逻辑 rank 按 TP、CP、DP、PP 从快到慢编号：

```text
rank = (((pp × DP + dp) × CP + cp) × TP + tp)
```

每个 rank 绑定调度器实际分配的 `node + GPU index`。监控、执行时间线与终端读取这些对象；它们不是独立生成的展示数据。

## 一个 optimizer step 如何执行

1. 数据读取使用按领域权重归一的文档采样器。分片边界不产生零监督尾片，训练与验证保持冻结版本的隔离。逻辑采样次序由 seed 和已提交数据游标确定，与快慢节点的完成顺序无关。
2. `accumulation = global_batch / (micro_batch × DP)`。每个 DP 副本执行对应数量的 micro batch；同一份模型参数在整批梯度完成前保持不变。
3. PP 使用非交错 1F1B。各 stage 先执行 `min(PP-stage-1, microbatches)` 个 warmup forward，再交替 forward/backward，最后 drain 剩余 backward。跨 stage 的激活和梯度通过独立 P2P 事件传递。
4. TP AllReduce、CP attention 通信、DP/CP ReduceScatter 和参数 AllGather 有实际参与 rank、逻辑字节量与资源端点。只有前置事件完成，后续事件才能开始。
5. 开启 activation recompute 时，反向前增加显式 `RecomputeForward` 计算事件；保留激活更少，计算工作更多。
6. 预训练与 SFT 按有效监督 Token 数汇总损失和梯度；DPO 按偏好对求均值，并单独统计 chosen/rejected 的监督 Token。全局 norm、裁剪、Adam 与参数同步全部结束后，才提交一次模型、优化器和数据游标更新。

默认 Python 前向接口还可读取 `state.rank`、`dp_rank`、`pp_rank`、`tp_rank`、`cp_rank`、`microbatch` 与 `train_state.optimizer_step`。数值 loss 在每个 DP 副本最后一个 PP stage 的 TP0/CP0 代表 rank 上执行，物理计算/通信图覆盖全部 GPU。

## 数值执行与页面交互

训练页支持暂停自动执行、恢复，以及推进一个训练事件边界；同一时间戳上的事件一起完成。暂停后仍可操作界面、查看日志和定位机柜，适合逐项检查快卡等待与通信依赖。系统安装或检查点 I/O 可通过恢复执行继续推进。

页面内联训练 Worker 的全部代码，无外部运行时下载。Backward 事件完成后才向 Worker 提交相应数值工作；不存在预先完成整轮模型更新再播放阶段的过程。浏览器只在完整梯度屏障后接收并提交更新。

工程存档保存事件依赖、已完成 micro batch、剩余工作、rank 时间和采样位置。刷新后，对尚未提交步骤重放已完成 micro batch 的数值计算；按固定逻辑顺序归并，避免不同完成顺序造成重复更新。浏览器数值计算尚未完成时，工程事件时钟在屏障处等待，页面仍可操作。

没有 Worker 的执行环境使用同一数值接口的同步路径；自动验收使用该路径。浏览器另行实测内联 Worker、导入、刷新和参数提交。

## 资源模型与公开假设

硬件文件为 `/etc/foundry/hardware.json`。保存文件不会改变运行中作业；排空训练和推理资源后执行“应用硬件配置”，后续运行绑定新快照。

| 资源 | 默认配置 / 使用方式 |
| --- | --- |
| GPU 计算 | 989 TFLOPS BF16 dense 参考峰值，MFU 取 0.42；MFU 是可配置假设 |
| HBM | 80 GiB 工程容量，3,350 GB/s 参考带宽；保留运行时与安全余量 |
| NVLink | 单方向按 450 GB/s 计，区分厂商公布的 900 GB/s 双向指标 |
| 节点网络 | 每节点 400 Gbps HCA，所有经过同一端点的活动通信共享预算 |
| Spine | 聚合 1,600 Gbps，多个作业共享；可调整以研究超售与瓶颈 |
| 主机 | 1,024 GiB RAM、128 GB/s PCIe 总预算、200 GB/s CPU 内存预算，均为本工程设计值 |
| 存储 | 使用当前生效的 NFS 配置。预取槽、检查点写入和转换任务共享带宽份额 |

Forward 包括参数相关的 `2 × P × tokens` 计算及随序列长度平方增长的 attention 项；Backward 和重计算具有各自工作量。活动集合变化后重新分配链路份额，不把固定吞吐作为任务进度。

显存区分模型/梯度/优化器状态与活动 micro batch 激活。CPU optimizer offload 增加主机 RAM、D2H、CPU Adam 与 H2D 事件。新协议检查点按 BF16 模型状态与 FP32 master/m/v 的 14 bytes/parameter 建立逻辑容量，约 980 GB，不包含梯度。

GPU 频率限制是节点的实时状态，会影响对应 rank 的计算，及同一节点上 TP 推理副本的瓶颈速率。故障注入不改写已提交的模型或历史硬件快照。

## 可复现的验收

- 将某张 GPU 算力限制到 25%，观察同 TP 组其他卡出现 `WAIT_PEER`，整步变慢；保持数据与配方相同，数值更新保持一致。
- 在跨节点训练中断开 IB，已有计算可先完成；到达依赖该链路的通信后阻塞，由 watchdog 给出 operation、ranks 和 endpoints。未完成步骤不更新 Adam。
- 同一节点内的 TP8 任务使用 NVLink，不因 IB 拔出而失败；其 CPU offload 仍需满足主机内存与 PCIe 预算。
- 降低主机 RAM，CPU offload 会给出 Host OOM；GPU 激活不足给出对应 rank 的 CUDA OOM。
- 运行两个占用不同 GPU 的作业，在较低 Spine 预算下观察通信争用。资源台账禁止重复分配。
- 在一轮 backward 已部分完成时导出/导入工程，结果应与不中断运行一致。
- 在完整 step 后提交检查点，检查分片、状态、RNG 和数据游标；停止并恢复后只执行剩余更新。

独立数值例子：两个样本的 loss 为 1、3，有效 Token 数为 1、3，梯度为 2、6，则全局 loss 为 2.5、梯度为 5；Adam 首步 `m=0.5`、`v=0.025`。验收还检查 DPO `beta=0` 与 `grad_clip=0` 不被默认值覆盖。

```bash
node verify.mjs
node preview.mjs --qa
```

`/__qa/` 可下载运行中的逐 GPU 工程，正常导入后，在 MLflow 训练页查看执行概览、每张 GPU、时间线、通信组、检查点和实验对比。终端支持 `trainctl inspect <job>` 与 `trainctl trace <job>`。

## 验收边界

训练数值核心仍为条件 Token 行为模型，监督来自冻结语料的有限分片窗口；没有在浏览器中展开 70B Transformer 张量。工程图覆盖 rank 级计算、内存与通信依赖，但不是 NCCL、CUDA、网络交换机或全部 Megatron kernel 的逐指令实现。所用 MFU、拓扑、通信和激活估算尚无完整实机标定，指标不能当作采购或生产容量承诺。MoE、交错/VPP、完整 CP 算法选择、全部调度策略仍不属于本协议。

参考：[Megatron 非交错 1F1B](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/apidocs/core/core.pipeline_parallel.schedules.html)、[NVIDIA 调度实现](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)、[H100 公开规格](https://www.nvidia.com/en-eu/data-center/h100/)、[Hopper NVLink 说明](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)。硬件设计假设与公开器件指标分开记录。
