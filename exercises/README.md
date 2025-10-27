# HPFS Bug 学习练习

欢迎来到 HPFS Flow Control Bug 学习练习！

## 📚 学习目标

通过这套练习，你将：
- 完全理解 HPFS 并发 bug 的根本原因
- 掌握 Go 并发问题的调试技术
- 学会实现和评估多种修复方案
- 编写并发安全的代码
- 获得端到端的问题解决能力

## 🗂️ 目录结构

```
exercises/
├── module1/          # 模块 1: Bug 根因分析
├── module2/          # 模块 2: 触发条件研究
├── module3/          # 模块 3: 并发调试技术
├── module4/          # 模块 4: 实现修复方案
│   ├── solution_a/   # 方案 A: 禁用全局流控
│   ├── solution_b/   # 方案 B: 引用计数
│   └── solution_c/   # 方案 C: rate.Limiter
├── module5/          # 模块 5: 编写测试用例
├── module6/          # 模块 6: 并发安全模式
├── module7/          # 模块 7: 端到端实践
└── answers/          # 答案和解析
```

## 🚀 开始学习

### 前置要求

- Go 1.21 或更高版本
- Docker 和 Kubernetes (minikube) - 用于模块 7
- 基础的 Go 并发知识

### 学习路径

根据你的时间和目标选择：

**初学者路径** (8-12 小时)
1. 模块 1: Bug 根因分析
2. 模块 2: 触发条件研究
3. 模块 4: 实现方案 A
4. 模块 5: 基础测试
5. 模块 7: 端到端实践

**进阶路径** (16-20 小时)
- 完成所有 7 个模块
- 实现所有 3 种修复方案
- 编写完整的测试套件

**专家路径** (20+ 小时)
- 完成进阶路径
- 设计新的流控架构
- 编写性能基准测试
- 撰写技术博客

## 📖 使用方法

1. 按顺序完成每个模块
2. 每个模块包含练习文件和说明
3. 在 `answers/` 目录查看参考答案
4. 使用 `progress_tracker.sh` 跟踪学习进度

## 🔗 相关资源

- [需求文档](../.kiro/specs/hpfs-bug-learning/requirements.md)
- [设计文档](../.kiro/specs/hpfs-bug-learning/design.md)
- [任务清单](../.kiro/specs/hpfs-bug-learning/tasks.md)
- [Bug 分析文档](../review/HPFS_FLOW_CONTROL_BUG_ANALYSIS.md)
- [Bug 修复指南](../review/HPFS_BUG_FIX_GUIDE.md)

## 💡 提示

- 不要急于查看答案，先尝试自己解决
- 使用 Go race detector 检测并发问题
- 多运行几次测试，观察不同的行为
- 记录你的思考过程和发现

## 🆘 获取帮助

如果遇到问题：
1. 查看 `FAQ.md`
2. 检查相关的 Bug 分析文档
3. 使用 Go 调试工具（race detector, pprof）
4. 在代码中添加日志追踪执行流程

祝学习愉快！🎉
