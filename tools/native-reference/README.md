# 原生骨干独立参考

这些工具从仓库根目录运行，需完整仓库。交付 HTML 本身不需要 Python、PyTorch 或网络。

1. 先运行 `npm test`，生成基础设施与工程测试存档。
2. 按 `docs/native-backbone-sources-v17.json` 的固定 URL、SHA256 准备源码缓存；对应目录保留 URL 中模型与源码路径。DeepSeek 的 6 个推理源放在 `.test-artifacts/v12-evidence/deepseek-v4.1-flash-1017/inference/`；Transformers 和 V-JEPA 放在 `.test-artifacts/v12-evidence/sources/` 下相应仓库路径。参考脚本对这些路径有明确引用。
3. CPU 参考环境使用 Python 3.12、PyTorch 2.15.0+cpu、NumPy、SymPy 1.14.0、tokenizers 0.23.3。可将 PyTorch 及其依赖安装到 `.test-artifacts/v17-python`，tokenizers 和 Transformers/Jinja 安装到 `.test-artifacts/v16-python`；脚本优先读取这些隔离目录，也支持已安装依赖的环境。PyTorch 使用官方 CPU wheel 索引；不需要 CUDA。
4. DeepSeek 的 Engram 参考还需要固定版本 `tokenizer.json`，放入 `.test-artifacts/v16-evidence/deepseek-v4.1-flash-1017/`；来源、摘要见 `docs/原生Tokenizer来源.json`。

先生成同一随机种子、输入和权重快照，再由独立参考对所有输出及梯度逐元素核验：

```powershell
node tools/native-reference/probe-deepseek-v17.cjs
node tools/native-reference/probe-trunks-v17.cjs
node tools/native-reference/probe-vision-v17.cjs
node tools/native-reference/probe-jepa-v17.cjs
python tools/native-reference/reference-deepseek-v17.py
python tools/native-reference/reference-trunks-v17.py
python tools/native-reference/reference-vision-v17.py
python tools/native-reference/reference-jepa-v17.py
```

`verify-v17.cjs` 使用已冻结的 PyTorch 输出与梯度坐标，因此日常回归不需再次安装 PyTorch。`probe-worker-v17.cjs` 对比实际内联 Worker 与前台的损失和完整梯度；`verify-archive-v17.cjs` 检查新旧存档、数值字节、分块去重和损坏检测。

数值对照采用 float64 来核验计算与求导；默认 DeepSeek 软件低精度缓存/权重路径另有执行与缓存检查，不将 CPU 参考称作 CUDA 原生量化逐比特一致性证明。

后台生命周期可单独运行 `node verify-v17-async.cjs`：使用实际内联 Worker 计算，核验取消后的旧结果不提交、恢复后只更新一次。
