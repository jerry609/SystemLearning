# Gemini LangGraph Deep Search Agent 学习课程

欢迎来到使用 LangGraph 和 Google Gemini 构建深度搜索智能体的综合学习课程！本课程将引导你从基础概念到生产级 AI 智能体系统。

## 🎯 课程概述

本课程教你如何使用 LangGraph 和 Google Gemini 模型构建复杂的深度搜索智能体。你将学习创建有状态的、能够研究、推理和综合信息的智能体系统。

**你将构建的内容：**
- 能够搜索和综合信息的智能研究代理
- 具有专业角色的多智能体系统
- 带有反思和规划能力的高级智能体
- 生产就绪的部署方案，包含完整的测试

**课程结构：**
- 10 个渐进式实验模块（深度覆盖）
- 60+ 个实践示例（从基础到高级）
- 40+ 个练习及详细解答
- 10+ 个完整项目
- 1 个生产级综合项目

**深度内容特色：**
- 每个模块包含 5-7 个理论文档
- 每个主题包含 6-8 个渐进式示例
- 涵盖高级模式：ReAct、CoT、ToT、多智能体协作
- 完整的生产部署和测试策略
- 向量数据库和语义搜索集成

## 📋 前置要求

### 必需知识
- **Python**: 中级水平（函数、类、async/await）
- **命令行**: 熟悉终端/shell 命令
- **Git**: 基本的版本控制操作
- **HTTP/REST**: 理解基本的 API 概念

### 推荐知识
- FastAPI 或类似 Web 框架
- Docker 基础
- 对 LLM 和 AI 概念的基本理解
- 数据库基础（PostgreSQL、Redis）

### 系统要求
- **操作系统**: macOS, Linux, 或 Windows with WSL2
- **Python**: 3.10 或更高版本
- **内存**: 8GB 最低，16GB 推荐
- **存储**: 3GB 可用空间

## 🚀 快速开始

### 1. 克隆或下载课程

```bash
# 如果是大型仓库的一部分
cd gemini-langgraph

# 或独立克隆
git clone <repository-url>
cd gemini-langgraph
```

### 2. 设置 Python 环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置 API 密钥

在根目录创建 `.env` 文件：

```bash
# Google Gemini API Key (必需)
GOOGLE_API_KEY=your_gemini_api_key_here

# Tavily Search API Key (研究智能体必需)
TAVILY_API_KEY=your_tavily_api_key_here

# 可选：LangSmith 用于追踪
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=gemini-langgraph-course
```

**获取 API 密钥：**
- **Gemini API**: 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Tavily Search**: 在 [Tavily](https://tavily.com/) 注册
- **LangSmith**: 在 [LangSmith](https://smith.langchain.com/) 注册

### 4. 验证设置

```bash
# 运行设置验证脚本
python utils/setup_check.py
```

## 📚 学习路径

### 基础路径 (Labs 1-3)
如果你是 LangGraph 和 AI 智能体的新手，从这里开始。

1. **Lab 01**: LangGraph 基础 - 学习核心概念
2. **Lab 02**: Gemini API 集成 - 连接 LLM
3. **Lab 03**: 智能体状态管理 - 处理复杂状态

**预计时间**: 13-16 小时

### 中级路径 (Labs 4-6)
构建完整的搜索智能体和工具集成。

4. **Lab 04**: 工具调用和函数 - 扩展智能体能力
5. **Lab 05**: Web 研究智能体 - 构建第一个完整智能体
6. **Lab 06**: FastAPI 后端开发 - 构建健壮的 API

**预计时间**: 16-19 小时

### 高级路径 (Labs 7-10)
高级智能体模式、多智能体系统和生产部署。

7. **Lab 07**: 反思与规划 - 高级智能体模式
8. **Lab 08**: 多智能体系统 - 协调多个智能体
9. **Lab 09**: 持久化与内存 - 添加长期记忆
10. **Lab 10**: 综合项目 - 构建完整的深度搜索智能体

**预计时间**: 36-42 小时

### 快速入门路径
Labs 1, 2, 4, 5 - 快速原型开发

**预计时间**: 18-22 小时

## 🗂️ 课程结构

每个实验模块遵循一致的结构：

```
labXX-module-name/
├── README.md           # 学习目标、前置要求、时间估计
├── theory/             # 概念解释和图表
├── examples/           # 可运行的代码演示
├── exercises/          # 练习题及解答
└── project/            # 实践项目应用概念
```

查看 [COURSE_STRUCTURE.md](COURSE_STRUCTURE.md) 了解详细的模块分解。

## 💡 如何使用本课程

### 自学者
1. 从 Lab 01 开始按顺序学习
2. 阅读理论文档理解概念
3. 运行和修改示例代码观察效果
4. 完成练习测试理解程度
5. 构建项目实际应用知识
6. 只在尝试后查看解答

### 讲师
- 每个模块独立完整，可在 2-4 小时内教授
- 项目可作为作业或小组作业
- 练习包含难度评级便于区分
- 解答在单独目录中提供

### 快速参考
- 使用理论文档作为参考资料
- 示例展示最佳实践
- 项目可作为实际应用的模板

## 🛠️ 故障排除

### 常见问题

**导入错误**
```bash
# 确保虚拟环境已激活
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 重新安装依赖
pip install -r requirements.txt
```

**API 密钥问题**
```bash
# 验证 .env 文件存在且包含密钥
cat .env

# 测试 API 连接
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GOOGLE_API_KEY'))"
```

**模块未找到**
```bash
# 将课程目录添加到 Python 路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

查看各个实验的 README 了解特定模块的故障排除。

## 📖 额外资源

### 官方文档
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [LangChain 文档](https://python.langchain.com/)
- [Google Gemini API](https://ai.google.dev/docs)
- [FastAPI 文档](https://fastapi.tiangolo.com/)

### 社区
- [LangChain Discord](https://discord.gg/langchain)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/langchain)

### 相关课程
- LangChain Academy（官方课程）
- DeepLearning.AI LangChain 课程
- FastAPI 教程

## 🤝 贡献

发现问题或想改进课程？欢迎贡献！

查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

## 📝 许可证

本课程仅供教育目的。各个组件可能有自己的许可证。

## 🎓 课程完成

完成所有模块和综合项目后，你将：
- 构建 10+ 个 AI 智能体项目
- 掌握 LangGraph 和 Gemini 集成
- 创建一个值得展示的综合项目
- 具备构建生产级深度搜索智能体的能力

## 🚦 获取帮助

- **问题**: 查看实验特定的故障排除部分
- **疑问**: 查看理论文档和示例
- **Bug**: 在 issues 部分报告
- **讨论**: 加入社区频道

---

**准备开始？** 前往 [Lab 01: LangGraph 基础](lab01-langgraph-fundamentals/README.md) 开始你的学习之旅！

**需要指导？** 查看 [COURSE_STRUCTURE.md](COURSE_STRUCTURE.md) 了解详细的模块信息。

**想跳过某些内容？** 查看课程结构文档中每个模块的前置要求。
