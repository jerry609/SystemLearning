# Lab 01: 创建型模式 - 单例与工厂

## 📚 学习目标

- 理解单例模式的应用场景和实现方式
- 掌握工厂模式的三种形式
- 学会在 Go 语言中实现这些模式
- 完成配置管理器和日志工厂实战项目

## 📖 内容概览

### 1. 单例模式 (Singleton Pattern)

**定义**: 确保一个类只有一个实例，并提供全局访问点。

**应用场景**:
- 配置管理器
- 数据库连接池
- 日志记录器
- 线程池
- 缓存管理器

**实现方式**:
1. 饿汉式（Eager Initialization）
2. 懒汉式（Lazy Initialization）
3. 双重检查锁（Double-Checked Locking）
4. Go 语言的 sync.Once 实现

### 2. 工厂模式 (Factory Pattern)

**定义**: 定义创建对象的接口，让子类决定实例化哪个类。

**三种形式**:
1. **简单工厂** (Simple Factory)
   - 一个工厂类负责创建所有产品
   - 违反开闭原则，但简单实用

2. **工厂方法** (Factory Method)
   - 定义创建对象的接口，子类决定实例化哪个类
   - 符合开闭原则

3. **抽象工厂** (Abstract Factory)
   - 创建一系列相关或相互依赖的对象
   - 适用于产品族的场景

## 🗂️ 目录结构

```
lab01-创建型模式-单例与工厂/
├── README.md                    # 本文件
├── theory/                      # 理论讲解
│   ├── 01-singleton.md         # 单例模式详解
│   ├── 02-factory.md           # 工厂模式详解
│   └── 03-design-principles.md # 设计原则
├── examples/                    # 示例代码
│   ├── singleton/
│   │   ├── eager.go            # 饿汉式
│   │   ├── lazy.go             # 懒汉式
│   │   ├── double_check.go     # 双重检查锁
│   │   └── sync_once.go        # sync.Once 实现
│   └── factory/
│       ├── simple_factory.go   # 简单工厂
│       ├── factory_method.go   # 工厂方法
│       └── abstract_factory.go # 抽象工厂
├── exercises/                   # 练习题
│   ├── exercise1.md            # 单例模式练习
│   ├── exercise2.md            # 工厂模式练习
│   └── answers/                # 练习答案
│       ├── exercise1_answer.go
│       └── exercise2_answer.go
└── project/                     # 实战项目
    ├── config-manager/         # 配置管理器
    │   ├── README.md
    │   ├── config.go
    │   └── config_test.go
    └── logger-factory/         # 日志工厂
        ├── README.md
        ├── logger.go
        ├── factory.go
        └── logger_test.go
```

## 🚀 快速开始

### 1. 学习理论

按顺序阅读 `theory/` 目录下的文档：
1. `01-singleton.md` - 单例模式
2. `02-factory.md` - 工厂模式
3. `03-design-principles.md` - 设计原则

### 2. 运行示例

```bash
# 进入示例目录
cd examples/singleton

# 运行单例模式示例
go run eager.go
go run lazy.go
go run sync_once.go

# 进入工厂模式示例
cd ../factory
go run simple_factory.go
go run factory_method.go
go run abstract_factory.go
```

### 3. 完成练习

打开 `exercises/exercise1.md` 和 `exercises/exercise2.md`，完成练习题。

### 4. 实战项目

完成两个实战项目：
1. 配置管理器 (`project/config-manager/`)
2. 日志工厂 (`project/logger-factory/`)

## 📝 学习路径

### 初学者路径 (4-6 小时)

1. **理论学习** (1-2 小时)
   - [ ] 阅读单例模式理论
   - [ ] 阅读工厂模式理论
   - [ ] 理解设计原则

2. **示例代码** (1-2 小时)
   - [ ] 运行并理解单例模式示例
   - [ ] 运行并理解工厂模式示例
   - [ ] 对比不同实现方式

3. **练习题** (1 小时)
   - [ ] 完成练习 1：实现线程安全的单例
   - [ ] 完成练习 2：实现简单工厂

4. **实战项目** (1-2 小时)
   - [ ] 完成配置管理器项目
   - [ ] 完成日志工厂项目

### 进阶路径 (6-8 小时)

在初学者路径基础上：

5. **深入研究** (2 小时)
   - [ ] 研究 Go 标准库中的单例实现
   - [ ] 分析开源项目中的工厂模式
   - [ ] 对比不同语言的实现差异

6. **扩展练习** (2 小时)
   - [ ] 实现数据库连接池（单例）
   - [ ] 实现插件系统（工厂）
   - [ ] 性能测试和优化

## 💡 关键概念

### 单例模式

**优点**:
- ✅ 控制实例数量
- ✅ 节省系统资源
- ✅ 提供全局访问点

**缺点**:
- ❌ 违反单一职责原则
- ❌ 难以测试
- ❌ 可能成为全局状态

**Go 语言最佳实践**:
```go
type Singleton struct {
    // 字段
}

var (
    instance *Singleton
    once     sync.Once
)

func GetInstance() *Singleton {
    once.Do(func() {
        instance = &Singleton{}
    })
    return instance
}
```

### 工厂模式

**简单工厂 vs 工厂方法 vs 抽象工厂**:

| 特性 | 简单工厂 | 工厂方法 | 抽象工厂 |
|------|----------|----------|----------|
| 复杂度 | 低 | 中 | 高 |
| 扩展性 | 差 | 好 | 很好 |
| 产品数量 | 单一产品 | 单一产品 | 产品族 |
| 开闭原则 | 不符合 | 符合 | 符合 |
| 适用场景 | 产品少且固定 | 产品可扩展 | 多个产品族 |

## 🎯 练习题预览

### 练习 1: 实现线程安全的缓存管理器

使用单例模式实现一个线程安全的缓存管理器，要求：
- 全局唯一实例
- 支持 Get/Set/Delete 操作
- 线程安全
- 支持过期时间

### 练习 2: 实现数据库连接工厂

使用工厂模式实现数据库连接工厂，要求：
- 支持 MySQL、PostgreSQL、MongoDB
- 根据配置创建不同的连接
- 支持连接池
- 易于扩展新的数据库类型

## 📊 学习检查清单

完成本 Lab 后，你应该能够：

- [ ] 解释单例模式的意图和适用场景
- [ ] 实现至少 3 种单例模式
- [ ] 说出单例模式的优缺点
- [ ] 区分简单工厂、工厂方法和抽象工厂
- [ ] 用代码实现三种工厂模式
- [ ] 在实际项目中应用这些模式
- [ ] 识别何时应该使用这些模式
- [ ] 评估模式的优缺点和权衡

## 🔗 相关资源

### 推荐阅读
- [Go 语言设计模式](https://github.com/tmrts/go-patterns)
- [Singleton Pattern - Refactoring.Guru](https://refactoring.guru/design-patterns/singleton)
- [Factory Pattern - Refactoring.Guru](https://refactoring.guru/design-patterns/factory-method)

### 开源项目示例
- Go 标准库的 `sync.Once`
- Kubernetes 的工厂模式应用
- Docker 的单例模式应用

## 📞 常见问题

### Q1: 单例模式在 Go 中是否必要？
Go 语言可以使用包级别的变量实现单例，但 sync.Once 提供了更好的延迟初始化和线程安全保证。

### Q2: 什么时候使用工厂模式？
当对象创建逻辑复杂，或需要根据条件创建不同类型的对象时。

### Q3: 如何选择工厂模式的类型？
- 产品少且固定 → 简单工厂
- 产品可扩展 → 工厂方法
- 多个产品族 → 抽象工厂

## 🎓 下一步

完成本 Lab 后，继续学习：
- [Lab 02: 建造者模式与原型模式](../lab02-创建型模式-建造者与原型/README.md)

---

**开始时间**: ___________  
**完成时间**: ___________  
**学习笔记**: ___________

**祝学习愉快！** 🎉
