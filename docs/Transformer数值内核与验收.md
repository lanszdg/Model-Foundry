# v0.7 · 可训练、可检查、可恢复的 Transformer

> v0.8 更新：新任务统一从 `model-spec.json` 读取 numeric；`model.json` 保留兼容入口。规格、资产与容量关系见 [统一模型验收](统一模型与操作语义验收.md)。

新训练使用 `transformer-v1`。旧存档中已开始的任务保留原协议，新任务不再建立“前两个 Token → 条件概率表”。专业工程流程保持一致：建设环境、治理语料、编写脚本、调度训练、后训练、评测、转换与服务。

## 计算对象

每个 Token 经过嵌入，随后通过重复的 decoder block：RMSNorm → Q/K/V 线性投影 → RoPE → 带因果遮罩的 GQA → 输出投影与残差 → RMSNorm → SwiGLU 前馈网络与残差。最后经过 RMSNorm 和 LM head，得到整个冻结词表的 logits。训练使用稳定的 log-sum-exp 交叉熵。

前向图有独立的反向传播实现，梯度覆盖嵌入、归一化缩放、Q/K/V/O、gate/up/down 和输出头。GQA 共享 K/V 头，反向时汇总同组 Query head 的梯度；RoPE 反向使用旋转的转置。没有用固定曲线、语料检索或预置回答代替数值结果。

## 运行结构与工程资源

机房里的 Atlas 70B 仍按 80 层、8192 hidden、64 Q heads、8 KV heads 和 32 张虚拟 H100 组织工程资源。浏览器实际执行的数值结构由 `/workspace/atlas/model.json` 明确配置，默认如下：

```json
{
  "schema": 1,
  "hidden_size": 32,
  "intermediate_size": 80,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 1,
  "max_position_embeddings": 256,
  "rope_theta": 10000,
  "rms_norm_eps": 0.00001
}
```

词表大小取决于冻结的 BPE Tokenizer。验收语料为 595 个 Token，对应 58,720 个数值参数。所有层都计算；这不是把完整 70B 权重压缩到相同语言能力。这里缩小数值分配，使普通浏览器能执行完整结构的前向、反向和调优；70B 的资源与分布式场景由工程事件系统承载。没有入门/专业两套流程。

当前用 JavaScript FP64 计算和保存数值参数，服务权重导出 F64。训练配方里的 BF16 对应虚拟资源协议，不表示浏览器使用了 BF16 CUDA 算术。配置有显式分配上限；超出预算会报告错误。数值配置变化会使既有预检失效。后训练必须沿用父检查点的结构、词表与 Tokenizer。

预训练的每个文档窗口长度取 `min(seq_length, packing_length, numeric context length)`，保留完整全局样本数量。验证按相同数值窗口分段计算。过长的 SFT/DPO 样本保留末尾窗口，包括 completion/EOS；没有把截断后的损失当作未截断长上下文结果。服务要求输入与最大输出长度之和不超过数值上下文，超出时返回错误。

## 从脚本到参数更新

Python 前向回调仍经 AST 解释器执行。`model(tokens, labels=...)` 返回可微分的逐 Token 损失，loss mask、加减乘除与标量归约通过向量雅可比积连接到 Transformer。脚本错误保留源码位置；这不是完整 PyTorch/Python 运行时。

预训练与 SFT 按有效监督 Token 数归约。SFT 只对 assistant/EOS 计损失，提示词仍能通过注意力影响后续 Token 的梯度。DPO 使用冻结 reference policy，比较 chosen/rejected 的 completion 序列 log probability；偏好对等权归约，使用稳定的 logistic loss。`beta=0` 时损失为 `log(2)`、偏好梯度为零。

一个全局批次期间参数冻结。各 micro batch 的梯度按固定逻辑顺序归约；完整通信屏障后计算全局 norm、裁剪、AdamW 一阶/二阶矩、偏差修正、学习率预热/余弦衰减和权重衰减。归一化缩放不施加 weight decay。不存在按阶段放大学习率的隐藏倍率。

`activation_recompute=true` 时初次前向不保留完整可微图，反向请求到达后重建前向图；数值结果与不重计算相同。rank 事件图仍记录虚拟 70B 的 RecomputeForward、显存与通信工作。代表 rank 的数值梯度与全部 rank 的资源事件通过同一全局提交屏障关联。

## 检查与操作

1. 在 Code 中打开 `model.json`，设定数值结构并保存。
2. 检查并提交训练配方。更改模型配置后需要重新预检。
3. 在训练工作区打开 **Transformer 内核**，查看实际参数数量、结构与 AdamW 更新次数。
4. 选择 Layer 和 Q Head，点击 **刷新注意力与梯度**。系统使用当前权重与冻结语料的前 32 个 Token，执行一次独立前向/反向检查。
5. 在注意力图中悬停查看 Query/Key 概率；因果遮罩上方为零。张量表展示实际权重 RMS 与梯度范数。检查不会应用参数更新。
6. 检查点完整提交后，可下载实际 `.safetensors` 数值权重。

终端也可执行：

```bash
modelctl inspect 24012 --layer 1 --head 2
```

检查结果写入 `/runs/<job>/numeric-inspection.json`，包含检查步骤、权重摘要、语料 ID、Token、注意力矩阵和张量统计。它是明确时刻的快照，训练继续后不会假装仍代表最新权重。

## 检查点、产物与服务

检查点 payload 保存实际参数数组、全部 AdamW moments、更新计数、模型结构、Tokenizer、数据游标、RNG 和冻结源码。刷新发生在未提交步骤内时，重建该步骤并重放已完成的 micro batch；不重复应用已提交的更新。

转换完成后额外生成：

- `numeric.safetensors.b64`：真实 safetensors 字节的 Base64 表达，适配工程虚拟文件系统。
- `numeric-config.json`：实际数值结构与 dtype。
- `numeric-manifest.json`：来源检查点、参数数量、结构和权重摘要。

权重使用 F64，矩阵按输出/输入轴转置导出，嵌入保持词表/hidden 轴。服务从此文件反序列化，并校验摘要、Tokenizer 与注册参数；损坏文件会阻止加载。虚拟 70B 的八分片描述仍用于工程容量与 I/O 场景，不能当作实际 140 GB 权重数据。自有 BPE Tokenizer 没有实现全部 Hugging Face tokenizer 格式转换，因此不宣称整个目录可直接作为完整外部模型仓库。

推理的 Prefill 实际生成每层 K/V 张量。Decode 只计算新增 Token，使用已有 K/V，greedy、temperature、top-p 和停止串均作用于模型 logits。同版本相同前缀可以复用真实数值 KV，减少实际前向 Token 工作；抢占清空数值 KV，重计算不重新消耗采样 RNG。请求和缓存都可随项目保存恢复。

## 验收证据

```bash
node verify.mjs
node verify-v7.cjs --fixture
# 可选：已安装 NumPy 时，运行独立实现的前向参考
python verify-transformer-reference.py
```

新增检查包括每类命名张量的中心差分、因果遮罩、超过两个 Token 的上下文影响、RoPE 相对位置、增量/整段前向等价、零监督、重复样本学习、DPO 数值梯度、激活重计算等价、精确 checkpoint/Adam 往返、safetensors 字节与矩阵轴、预训练/SFT/DPO、转换重载、数值 Prefill、前缀缓存复用与权重损坏拒绝。

NumPy 参考使用独立的批矩阵运算、复数旋转和 einsum 计算完整 logits 与 NLL，与 JavaScript 输出对比；不是再调用同一数值内核。

这些证据验证已实现的计算与状态契约，不表示完成所有大模型算法、所有软件界面或所有硬件行为。当前未实现 MoE、LoRA/QLoRA、FlashAttention/CUDA kernel、BF16 舍入、真实 NCCL 传输、完整操作系统与完整框架 API；数值规模和语言质量也不等同于预训练 70B。项目保持可检查的明确实现范围。

## 研究参照

结构与缓存参照 [Meta Llama 官方模型实现](https://github.com/meta-llama/llama/blob/main/llama/model.py)；位置旋转参照 [RoFormer 论文](https://arxiv.org/abs/2104.09864)；分组头参照 [GQA 论文](https://arxiv.org/abs/2305.13245)。更新规则参照 [PyTorch AdamW 文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)，字节格式参照 [safetensors 官方格式说明](https://github.com/safetensors/safetensors/tree/main/safetensors#format)。本实现是项目自行编写的 JavaScript 数值路径。
