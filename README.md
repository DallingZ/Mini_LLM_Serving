# Mini LLM Serving

这是一个简化版 LLM 推理服务调度实验

## 项目结构

```text
Mini_LLM_Serving/
├── mini_serving/
│   ├── request.py      # 请求状态：waiting/running/finished
│   ├── kv_cache.py     # KV Cache block 分配、追加和释放
│   ├── scheduler.py    # 等待队列、运行队列和 admission 逻辑
│   └── engine.py       # prefill/decode 主循环和指标统计
├── benchmark.py        # 串行 decode 和 continuous batching 对比
├── demo.py             # 小规模运行示例
└── tests/              # 基础单元测试
```