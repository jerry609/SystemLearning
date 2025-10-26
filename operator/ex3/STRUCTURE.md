# 项目目录结构

```
operator/ex3/
│
├── 📄 README.md                    # 项目总体说明
├── 📄 GETTING_STARTED.md           # 快速开始指南 ⭐ 从这里开始
├── 📄 STRUCTURE.md                 # 本文件：目录结构说明
│
├── 📚 exercises/                   # 练习说明文档
│   ├── README.md                   # 练习总览和学习路径
│   ├── 1.md                        # 练习 1: 状态机与基础协调循环 ✅
│   ├── 2.md                        # 练习 2: 资源创建与管理 🚧
│   ├── 3.md                        # 练习 3: 更新与同步逻辑 🚧
│   ├── 4.md                        # 练习 4: 删除与 Finalizer 🚧
│   └── 5.md                        # 练习 5: 错误处理与可观测性 🚧
│
├── 🔧 framework/                   # 基础框架代码（学生工作区）
│   ├── README.md                   # 框架使用说明
│   ├── go.mod                      # Go 模块定义
│   ├── types.go                    # 数据结构定义（通常不需修改）
│   ├── client.go                   # MockK8sClient（通常不需修改）
│   ├── errors.go                   # 错误类型（通常不需修改）
│   ├── reconcile.go                # 协调循环 ⭐ 主要实现文件
│   ├── main.go                     # 主程序入口
│   └── framework_test.go           # 框架基础测试
│
└── ✅ solutions/                   # 参考答案
    ├── README.md                   # 参考答案使用说明
    │
    ├── ex1/                        # 练习 1 参考答案 ✅
    │   ├── go.mod
    │   ├── types.go
    │   ├── client.go
    │   ├── errors.go
    │   ├── reconcile.go            # 完整实现
    │   ├── main.go                 # 测试程序
    │   ├── ex1_test.go             # 单元测试
    │   └── framework_test.go
    │
    ├── ex2/                        # 练习 2 参考答案 🚧
    ├── ex3/                        # 练习 3 参考答案 🚧
    ├── ex4/                        # 练习 4 参考答案 🚧
    └── ex5/                        # 练习 5 参考答案 🚧
```

## 图例

- 📄 文档文件
- 📚 文档目录
- 🔧 代码框架
- ✅ 参考答案
- ⭐ 重要文件
- 🚧 待实现

## 工作流程

### 1. 开始学习

```bash
# 阅读快速开始指南
cat GETTING_STARTED.md

# 阅读练习 1 说明
cat exercises/1.md
```

### 2. 编写代码

```bash
cd framework/

# 编辑 reconcile.go
vim reconcile.go

# 运行测试
go run .
```

### 3. 查看答案

```bash
cd solutions/ex1/

# 查看实现
cat reconcile.go

# 运行参考实现
go run .

# 运行测试
go test -v
```

## 文件职责

### 核心实现文件

| 文件 | 位置 | 职责 | 是否需要修改 |
|------|------|------|-------------|
| `reconcile.go` | framework/ | 协调循环核心逻辑 | ✅ 是 |
| `main.go` | framework/ | 测试程序 | 可选 |
| `types.go` | framework/ | 数据结构定义 | ❌ 否 |
| `client.go` | framework/ | 模拟客户端 | ❌ 否 |
| `errors.go` | framework/ | 错误类型 | ❌ 否 |

### 文档文件

| 文件 | 职责 |
|------|------|
| `README.md` | 项目总体介绍、背景知识 |
| `GETTING_STARTED.md` | 快速开始、学习路径 |
| `STRUCTURE.md` | 目录结构说明（本文件） |
| `exercises/README.md` | 练习总览 |
| `exercises/1.md` | 练习 1 详细说明 |
| `framework/README.md` | 框架使用指南 |
| `solutions/README.md` | 参考答案说明 |

## 设计理念

### 1. 关注点分离

- **exercises/**: 纯文档，说明要做什么
- **framework/**: 待实现的代码框架
- **solutions/**: 完整的参考实现

### 2. 渐进式学习

- 从简单到复杂
- 每个练习独立
- 可以单独运行和测试

### 3. 自主学习

- 提供框架但不限制实现
- 参考答案仅供参考
- 鼓励自己的实现方式

### 4. 实践导向

- 每个练习都可以运行
- 提供测试验证
- 即时反馈

## 常见问题

### Q: 我应该在哪里写代码？

A: 在 `framework/` 目录中，主要修改 `reconcile.go`。

### Q: 什么时候看参考答案？

A: 建议先自己实现，遇到困难或完成后再查看 `solutions/ex1/`。

### Q: 可以修改 types.go 或 client.go 吗？

A: 通常不需要，但如果你有好的想法，可以修改。

### Q: 如何验证我的实现？

A: 运行 `go run .` 查看输出，或编写测试用例。

### Q: 练习之间有依赖吗？

A: 建议按顺序完成，但每个练习的代码是独立的。

## 下一步

1. 阅读 `GETTING_STARTED.md`
2. 开始 `exercises/1.md`
3. 在 `framework/` 中实现
4. 参考 `solutions/ex1/` 验证

祝你学习愉快！🚀
