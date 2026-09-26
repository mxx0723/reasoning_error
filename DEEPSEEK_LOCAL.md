# 用 DeepSeek 开源权重运行实验（不是 API）

**本页仅对应 R1-Distill / Qwen。老师的 V4 Flash 请使用 [V4 专用说明](DEEPSEEK_V4_FLASH.md)，不要安装下面的旧 Transformers 环境。**

入口：`qwen_standard_v2/local_runner.py`。通过 Transformers 在自己的 GPU 上加载本地权重，不需要 DeepSeek API key，不调用远程推理服务。无需原 Skynet 历史目录，也不必运行旧集群的 `prepare.py`。

适配目标是 **DeepSeek-R1-Distill-Qwen** 等 Transformers 支持的 dense checkpoint，示例使用 32B。它不等同于完整版 DeepSeek-R1/V3；完整版 MoE 的分布式部署、量化专用格式和其他架构不在这次已实现的支持范围内。老师应传入实际要研究的模型路径。

## 安装和输入

在 GPU 环境中安装适合 CUDA 版本的 PyTorch，然后：

```bash
pip install -r requirements-local.txt
```

依赖基线为 Python 3.9+、PyTorch 2.6+、Transformers 4.48.1。**尚未在真实 DeepSeek GPU 推理环境完成端到端运行**；已完成离线逻辑和模拟后端测试。

准备：

- 本地模型目录，含 `config.json`、tokenizer、完整权重等标准 Transformers 文件。建议使用不可变的 Hugging Face snapshot。
- `oeq.jsonl`：每条需要唯一的字符串 `id`、`question`、`answer`，可选 `answer_key_points`；也支持 JSON 数组。
- 足够的 GPU 显存。32B 的 BF16 权重本身约需 64 GB，运行还需 KV cache 和其他空间。两张 A40 是可配置示例，不是对任意上下文长度的显存保证。

## 先检查配置

以下为 Linux/集群命令，请替换实际路径：

```bash
python qwen_standard_v2/local_runner.py \
  --model-path /models/DeepSeek-R1-Distill-Qwen-32B \
  --data /data/oeq.jsonl \
  --output runs/deepseek-r1-pilot \
  --profile deepseek-r1 \
  --ids oeq_0064,oeq_0416 --limit 2 \
  --max-memory '{"0":"42GiB","1":"42GiB"}' \
  --dry-run
```

`--dry-run` 不加载权重、不需要 GPU，不产生模型回答。正式运行删除 `--dry-run` 即可。先申请 GPU 作业再运行，不能在集群登录节点进行模型推理。程序使用 `CUDA_VISIBLE_DEVICES` 暴露的 GPU；没有写死必须两张 GPU。

默认只处理 1 个病例；`--limit N` 指定数量，`--all` 才处理全部选中数据。输入和结果在不同模型间分开存储。完整筛选阶段的门槛、证据构造、最小 K 搜索和独立验证直接复用 `pipeline.py`。

## R1 特有的适配

依据 [DeepSeek R1 官方说明](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/README.md)，R1 profile 将 system 指令原文合并进第一个 user 消息。默认 temperature=0.6、top_p=0.95；分类和构造请求也使用采样，避免为 R1 强行使用贪心解码。每次调用保存种子、实际采样配置、chat template 渲染后的完整输入。

因此它与原 Qwen 的**筛选规则相同，但模型输入适配和采样配置并非逐字逐参数相同**。不要把跨模型差异直接归因为某个机制。

默认输出预算 `--max-new-tokens 8192`，计入模型的思考 tokens；可调整。评分只读取 `</think>` 后的最终 JSON，不把思考中的候选答案当作最终答案。未闭合思考、无效 JSON、达到 token 上限一律不能算通过，并保存原始输出供复核。

Sequential 的历史有两个明确配置：

- `--history final`（默认）：只回放模型实际最终答案 JSON，不回放思考，测量 prior answer 的影响。
- `--history full`：把原始生成文本送入后续 chat template，包括思考文本。实际是否保留由模板决定，保存的 `rendered_prompt` 可检查。这是不同实验条件，应使用新输出目录。

不会编造初次答案或用 gold 替代回答。`--profile qwen` 保留 system role、默认 temperature=0.7/top_p=1，可用同一便携入口运行 Qwen。

## 复核、恢复和结果

使用 `--reviews reviews.json` 载入语义和临床审核，`--seed-plans seed_plans.json` 复用已有原题证据切分；两者都不能绕过原题答对门槛。格式见 [统一流程说明](qwen_standard_v2/README.md)。

输出包括运行清单、GPU/模型类信息、逐调用缓存、原始输出、解析答案、audit、复核队列和行为表。`summary.json` 分开报告稳定完整 pair、最终行为候选和正式审核通过，绝不把 UPDATE pair 自动算作错误病例。

同命令同目录恢复时复用模型回答。配置、数据选择、代码、checkpoint 文件元信息或 history 模式变化要求新目录；审核可更新后重算。权重记录的是文件元信息而非整套权重的内容哈希，因此应固定 snapshot。默认每病例最多 400 次调用，可用 `--max-calls-per-source` 提高。中断不冒充完成；强杀留下锁时先确认旧进程已退出再删除锁文件。

默认禁止 CPU/disk offload，避免无意变成极慢运行；确需允许时显式设置 `--allow-cpu-offload`。需要自定义模型代码的 checkpoint 必须显式设置 `--trust-remote-code`；本接口不因此保证该架构受支持。

本次提供的是**本地行为实验接口**。模型对象保留在 `HFBackend.model`，之后可扩展 hidden-state 抽取；本次没有宣称已经实现 probe 训练或 steering。

## 测试

```bash
python -m unittest discover -s qwen_standard_v2 -p 'test_*.py' -v
```

测试不用真实权重或 GPU，包含门槛、最小证据搜索、思考/最终答案分离、历史条件、缓存和完整入口恢复逻辑。
