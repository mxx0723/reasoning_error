# DeepSeek V4 Flash 本地开源模型接口

本入口在 GPU 节点内通过 `vllm.LLM.generate` 加载 **DeepSeek-V4-Flash 开源权重**，执行离线推理，无需 HTTP 推理服务或 API key。

它与 R1-Distill 是不同模型，不能使用本仓库 R1/Qwen 的旧 Transformers 4.48.1 环境直接加载。这里复用实验流程，另加 V4 专用的权重加载、消息编码和输出解析适配。

## 环境要求

需要支持**实际 checkpoint 版本和 GPU 架构**的 vLLM 环境。请优先沿用已经能运行该模型的环境；不要安装 `requirements-local.txt` 覆盖它，那份依赖仅供 R1/Qwen Transformers 后端。

官方提供 [V4 Flash 模型说明](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/README.md)、[专用编码器](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/encoding) 和 [vLLM 按硬件部署配方](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)。V4 使用自定义消息编码，不依赖 `apply_chat_template`；本接口加载 checkpoint 自带的 `encoding/encoding_dsv4.py`，不复制或猜测模板。

## 最小使用步骤

1. 在具备 V4 Flash 推理环境的 GPU 节点克隆本仓库。
2. 准备本地 V4 Flash checkpoint、同版本的 `encoding_dsv4.py` 和 OEQ 数据。
3. 将引擎配置写成 JSON 文件，参数名使用 `vllm.LLM` 的 Python 参数名，例如 `tensor_parallel_size`、`enable_expert_parallel`、`kv_cache_dtype`、`block_size` 等。参数应匹配实际 GPU、checkpoint 和已验证的部署配置。R1 示例中的两张 A40 配置不适用于此入口。
4. 先 dry-run，再运行一个源病例。

```bash
python qwen_standard_v2/local_runner.py \
  --profile deepseek-v4 \
  --model-path /path/to/DeepSeek-V4-Flash \
  --encoder-path /path/to/DeepSeek-V4-Flash/encoding/encoding_dsv4.py \
  --engine-args /path/to/working-vllm-engine.json \
  --data /path/to/oeq.jsonl \
  --output runs/v4-flash-chat-pilot \
  --ids oeq_0064 --limit 1 \
  --thinking-mode chat --history final \
  --max-model-len 32768 --max-input-tokens 16384 --max-new-tokens 8192 \
  --dry-run
```

`--dry-run` 校验文件、参数和数据，不加载模型。删除该参数即在当前 GPU 作业里加载权重并跑实验。是否需要 `--trust-remote-code` 取决于实际 checkpoint/引擎，不会默认启用。

`--engine-args` 不是 `vllm serve` 的全部 CLI 参数集合。端口、HTTP 服务、路由、API parser 等服务端选项不能传给离线 `LLM`。模型路径、种子、最大上下文等由命令行固定，不能被 JSON 静默覆盖。多机部署须先按现有集群框架配置好，再传兼容的引擎参数。

数据支持 JSONL 或 JSON 数组，记录需要唯一字符串 `id`、字符串 `question`、`answer`，可选 `answer_key_points`。不需要 Skynet 的历史文件。默认只运行一条；指定 `--limit N` 或 `--all` 扩大范围。

## 实验条件

- `--thinking-mode chat`：官方 non-thinking 模式，本项目默认。
- `--thinking-mode thinking`：官方思考模式。两者需不同输出目录，不能混在同一个重复实验内。
- `--history final`：Sequential 仅带前次实际最终答案，移除前次思考。
- `--history full`：在 thinking 模式中将前次思考放入 `reasoning_content`，并显式禁止编码器丢弃。它是另一种 commitment 条件。

本适配未开启 Think Max；不要把 thinking 模式结果标成 Max。V4 默认保留 system role；不套用 R1 的 system-to-user 改写。采样默认 temperature=1、top_p=1，分类和构造同样采样；种子、实际消息和编码后 prompt 都保存。和 Qwen 的筛选规则一致，但模型模板/采样设置有差异，跨模型比较需说明。

## 输出、恢复与限制

同一套流程仍执行：原始完整题答对门槛 → E1/K 构造 → 稳定性 → UPDATE 最小证据组搜索 / MAINTAIN 检查 → 独立最终验证 → Fresh / Sequential / 固定 prior。

评分使用官方解析器提取的最终内容。截断、格式错误、工具调用不计为通过。原始输出、解析后的思考和最终答案分开保存；不能把思考中的诊断词当作最终选择。

参数、代码、编码器、模型路径、checkpoint 元信息和数据选择会进入 manifest。相同目录重跑复用缓存；改变这些内容需新目录。通过 `--reviews` 补充语义/医学审核后可重算，不会重复模型调用。结果保留候选与正式审核通过的区别。

已完成离线流程测试及官方编码器兼容检查，**尚未完成真实 V4 GPU 推理验证**。正式实验前，应在目标部署环境运行一次小规模推理测试，验证引擎版本、硬件配置和输出格式。

该接口用于行为数据筛选；尚未实现 vLLM 内部 hidden-state 导出或 steering。之后提取 vector 需要另接 worker/model 内部激活 hook。

## 测试

```bash
python -m unittest discover -s qwen_standard_v2 -p 'test_*.py' -v
```

可额外设置 `DEEPSEEK_V4_ENCODER` 为官方 `encoding_dsv4.py` 的路径，运行与真实编码器的集成测试（不需要 GPU）。
