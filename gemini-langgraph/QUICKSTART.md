# 快速开始指南

## 5 分钟快速上手

### 1. 环境准备

```bash
# 克隆或进入课程目录
cd gemini-langgraph

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API 密钥

创建 `.env` 文件：

```bash
# 必需
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key

# 可选（用于追踪）
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
```

**获取密钥**：
- Gemini: https://makersuite.google.com/app/apikey
- Tavily: https://tavily.com/
- LangSmith: https://smith.langchain.com/

### 3. 验证安装

```bash
python -c "import langgraph, langchain, fastapi; print('✅ 所有依赖安装成功！')"
```

### 4. 开始学习

```bash
# 进入第一个实验
cd lab01-langgraph-fundamentals

# 阅读 README
cat README.md

# 运行第一个示例
python examples/simple_graph.py
```

## 学习路径

### 🟢 初学者（10-12 小时）
**Labs 1-3**: 基础概念
- Lab 01: LangGraph 基础
- Lab 02: Gemini API 集成
- Lab 03: 智能体状态管理

### 🟡 中级（12-15 小时）
**Labs 4-6**: 工具与后端
- Lab 04: 工具调用和函数
- Lab 05: Web 研究智能体
- Lab 06: FastAPI 后端开发

### 🟠 高级（15-18 小时）
**Labs 7-9**: 高级模式
- Lab 07: 反思与规划
- Lab 08: 多智能体系统
- Lab 09: 持久化与内存

### 🔴 综合项目（15-20 小时）
**Lab 10**: 深度搜索智能体
- 完整的生产级系统
- 多智能体协作
- Docker 部署

## 快速参考

### 常用命令

```bash
# 激活环境
source venv/bin/activate

# 运行示例
python examples/example_name.py

# 运行测试
pytest tests/

# 启动 FastAPI 服务
uvicorn main:app --reload

# 查看 API 文档
# 访问 http://localhost:8000/docs
```

### 目录结构

```
labXX-module-name/
├── README.md           # 学习目标和说明
├── theory/             # 理论文档
│   ├── 01-concept.md
│   └── 02-patterns.md
├── examples/           # 可运行示例
│   ├── basic.py
│   └── advanced.py
├── exercises/          # 练习题
│   ├── exercise1.py
│   └── solutions/
└── project/            # 实践项目
    ├── README.md
    └── starter_code.py
```

### 学习建议

1. **按顺序学习**：每个 Lab 都基于前面的知识
2. **动手实践**：运行所有示例代码
3. **完成练习**：先尝试自己解决
4. **构建项目**：每个 Lab 的项目都很重要
5. **查看解答**：只在尝试后查看

### 获取帮助

- 📖 查看 [COURSE_STRUCTURE.md](COURSE_STRUCTURE.md) 了解详细内容
- 🔧 查看 [README.md](README.md) 了解故障排除
- 📝 查看 [CHANGES.md](CHANGES.md) 了解课程调整

## 下一步

准备好了？开始你的学习之旅：

```bash
cd lab01-langgraph-fundamentals
cat README.md
```

祝学习愉快！🚀
