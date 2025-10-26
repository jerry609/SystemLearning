# 项目目录结构

```
operator/ex2/
│
├── 📄 README.md                    # 项目总体说明
├── 📄 INDEX.md                     # 文档索引 ⭐ 从这里开始
├── 📄 GETTING_STARTED.md           # 快速开始指南
├── 📄 STRUCTURE.md                 # 本文件：目录结构说明
│
├── 📚 exercises/                   # 练习说明文档
│   ├── README.md                   # 练习总览和学习路径
│   ├── 1.md                        # 练习 1: 核心三剑客 ✅
│   ├── 2.md                        # 练习 2: 声明式编排 ✅
│   └── 3.md                        # 练习 3: 可观测性 ✅
│
├── 🔧 framework/                   # 基础框架代码（学生工作区）
│   ├── README.md                   # 框架使用说明
│   └── main.go                     # 待创建：主实现文件 ⭐
│
└── ✅ solutions/                   # 参考答案
    ├── README.md                   # 参考答案使用说明
    ├── ex1/                        # 练习 1 参考答案
    │   └── main.go                 # 完整实现
    ├── ex2/                        # 练习 2 参考答案
    │   └── main.go                 # 完整实现
    └── ex3/                        # 练习 3 参考答案
        └── main.go                 # 完整实现
```

## 图例

- 📄 文档文件
- 📚 文档目录
- 🔧 代码框架
- ✅ 参考答案
- ⭐ 重要文件

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

# 创建并编辑 main.go
vim main.go

# 运行测试
go run main.go
```

### 3. 查看答案

```bash
cd solutions/ex1/

# 查看实现
cat main.go

# 运行参考实现
go run main.go
```

## 文件职责

### 核心实现文件

| 文件 | 位置 | 职责 | 是否需要创建 |
|------|------|------|-------------|
| `main.go` | framework/ | 所有练习的实现 | ✅ 是 |

### 文档文件

| 文件 | 职责 |
|------|------|
| `README.md` | 项目总体介绍、核心概念 |
| `INDEX.md` | 文档索引和导航 |
| `GETTING_STARTED.md` | 快速开始、学习路径 |
| `STRUCTURE.md` | 目录结构说明（本文件） |
| `exercises/README.md` | 练习总览 |
| `exercises/1.md` | 练习 1 详细说明 |
| `exercises/2.md` | 练习 2 详细说明 |
| `exercises/3.md` | 练习 3 详细说明 |
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

## 与 Ex1 和 Ex3 的关系

### Ex1: 微型框架构建

```
operator/ex1/
├── README.md
├── 1.md - 5.md (练习说明)
└── main.go (实现文件)
```

**特点**:
- 从零开始构建
- 学习基础模式
- 单文件实现

### Ex2: 真实代码库理解（本项目）

```
operator/ex2/
├── exercises/ (练习说明)
├── framework/ (待实现)
└── solutions/ (参考答案)
```

**特点**:
- 阅读真实代码
- 理解设计模式
- 分离文档和代码

### Ex3: 综合实战

```
operator/ex3/
├── exercises/ (练习说明)
├── framework/ (基础框架)
└── solutions/ (参考答案)
```

**特点**:
- 完整的 Operator
- 多文件组织
- 生产级实现

## 常见问题

### Q: 我应该在哪里写代码？

A: 在 `framework/` 目录中创建 `main.go` 文件。

### Q: 可以修改目录结构吗？

A: 可以，但建议保持当前结构以便于学习。

### Q: 练习之间有依赖吗？

A: 建议按顺序完成，但每个练习的代码是独立的。

### Q: 如何验证我的实现？

A: 运行 `go run main.go` 查看输出，或对比参考答案。

## 下一步

1. 阅读 `INDEX.md` 或 `GETTING_STARTED.md`
2. 开始 `exercises/1.md`
3. 在 `framework/` 中实现
4. 参考 `solutions/ex1/` 验证

祝你学习愉快！🚀
