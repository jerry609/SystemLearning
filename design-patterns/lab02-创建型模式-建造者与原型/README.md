# Lab 02: 创建型模式 - 建造者与原型

## 📚 学习目标

- 理解建造者模式的应用场景和实现方式
- 掌握 Go 语言中的 Functional Options 模式
- 理解原型模式的深拷贝和浅拷贝
- 学会在 Go 语言中实现对象克隆
- 完成 HTTP 请求构建器实战项目

## 📖 内容概览

### 1. 建造者模式 (Builder Pattern)

**定义**: 将复杂对象的构建与其表示分离，使得同样的构建过程可以创建不同的表示。

**应用场景**:
- 构建复杂对象（如 HTTP 请求、SQL 查询）
- 对象有多个可选参数
- 需要链式调用的 API
- 配置对象的构建
- 文档生成器

**实现方式**:
1. 传统建造者模式
2. 链式调用（Method Chaining）
3. Functional Options 模式（Go 语言推荐）

### 2. 原型模式 (Prototype Pattern)

**定义**: 通过复制现有对象来创建新对象，而不是通过实例化类。

**应用场景**:
- 对象创建成本高
- 需要大量相似对象
- 避免复杂的初始化过程
- 保存对象状态快照
- 实现撤销/重做功能

**实现方式**:
1. 浅拷贝（Shallow Copy）
2. 深拷贝（Deep Copy）
3. 使用 encoding/gob 序列化
4. 自定义 Clone 方法

## 🗂️ 目录结构

```
lab02-创建型模式-建造者与原型/
├── README.md                    # 本文件
├── theory/                      # 理论讲解
│   ├── 01-builder.md           # 建造者模式详解
│   └── 02-prototype.md         # 原型模式详解
├── examples/                    # 示例代码
│   ├── builder/
│   │   ├── chain_builder.go    # 链式调用实现
│   │   ├── functional_options.go # Functional Options 模式
│   │   └── http_request_builder.go # HTTP 请求构建器
│   └── prototype/
│       ├── shallow_copy.go     # 浅拷贝示例
│       └── deep_copy.go        # 深拷贝示例
├── exercises/                   # 练习题
│   ├── exercise1.md            # 建造者模式练习
│   ├── exercise2.md            # 原型模式练习
│   └── answers/                # 练习答案
│       ├── exercise1_answer.go
│       └── exercise2_answer.go
└── project/                     # 实战项目
    └── http-request-builder/   # HTTP 请求构建器
        ├── README.md
        ├── builder.go
        └── builder_test.go
```

## 🚀 快速开始

### 1. 学习理论

按顺序阅读 `theory/` 目录下的文档：
1. `01-builder.md` - 建造者模式
2. `02-prototype.md` - 原型模式

### 2. 运行示例

```bash
# 进入示例目录
cd examples/builder

# 运行建造者模式示例
go run chain_builder.go
go run functional_options.go
go run http_request_builder.go

# 进入原型模式示例
cd ../prototype
go run shallow_copy.go
go run deep_copy.go
```

### 3. 完成练习

打开 `exercises/exercise1.md` 和 `exercises/exercise2.md`，完成练习题。

### 4. 实战项目

完成 HTTP 请求构建器实战项目 (`project/http-request-builder/`)

## 📝 学习路径

### 初学者路径 (4-6 小时)

1. **理论学习** (1-2 小时)
   - [ ] 阅读建造者模式理论
   - [ ] 阅读原型模式理论
   - [ ] 理解深拷贝和浅拷贝的区别

2. **示例代码** (1-2 小时)
   - [ ] 运行并理解建造者模式示例
   - [ ] 运行并理解原型模式示例
   - [ ] 对比不同实现方式

3. **练习题** (1 小时)
   - [ ] 完成练习 1：实现 SQL 查询构建器
   - [ ] 完成练习 2：实现配置对象克隆

4. **实战项目** (1-2 小时)
   - [ ] 完成 HTTP 请求构建器项目

### 进阶路径 (6-8 小时)

在初学者路径基础上：

5. **深入研究** (2 小时)
   - [ ] 研究 Go 标准库中的 Functional Options
   - [ ] 分析开源项目中的建造者模式
   - [ ] 对比深拷贝的不同实现方式

6. **扩展练习** (2 小时)
   - [ ] 实现复杂的文档构建器
   - [ ] 实现对象池（结合原型模式）
   - [ ] 性能测试和优化

## 💡 关键概念

### 建造者模式

**优点**:
- ✅ 分离构建和表示
- ✅ 更好的控制构建过程
- ✅ 支持链式调用，代码优雅
- ✅ 易于扩展新的构建步骤

**缺点**:
- ❌ 增加代码复杂度
- ❌ 需要额外的 Builder 类

**Go 语言最佳实践 - Functional Options**:
```go
type Server struct {
    host    string
    port    int
    timeout time.Duration
}

type Option func(*Server)

func WithHost(host string) Option {
    return func(s *Server) {
        s.host = host
    }
}

func NewServer(opts ...Option) *Server {
    s := &Server{
        host: "localhost",
        port: 8080,
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}
```

### 原型模式

**浅拷贝 vs 深拷贝**:

| 特性 | 浅拷贝 | 深拷贝 |
|------|--------|--------|
| 复制内容 | 只复制值类型和引用 | 递归复制所有内容 |
| 性能 | 快 | 慢 |
| 内存占用 | 少 | 多 |
| 独立性 | 共享引用类型数据 | 完全独立 |
| 适用场景 | 不可变对象 | 需要完全独立的副本 |

**Go 语言实现方式**:
```go
// 方式 1: 自定义 Clone 方法
func (o *Object) Clone() *Object {
    return &Object{
        Field1: o.Field1,
        Field2: append([]int{}, o.Field2...),
    }
}

// 方式 2: 使用 encoding/gob
func DeepCopy(src, dst interface{}) error {
    var buf bytes.Buffer
    if err := gob.NewEncoder(&buf).Encode(src); err != nil {
        return err
    }
    return gob.NewDecoder(&buf).Decode(dst)
}
```

## 🎯 练习题预览

### 练习 1: 实现 SQL 查询构建器

使用建造者模式实现一个 SQL 查询构建器，要求：
- 支持 SELECT、WHERE、ORDER BY、LIMIT
- 支持链式调用
- 生成正确的 SQL 语句
- 防止 SQL 注入

### 练习 2: 实现游戏角色克隆

使用原型模式实现游戏角色的克隆，要求：
- 支持深拷贝和浅拷贝
- 包含装备、技能等复杂属性
- 克隆后的对象完全独立
- 性能优化

## 📊 学习检查清单

完成本 Lab 后，你应该能够：

- [ ] 解释建造者模式的意图和适用场景
- [ ] 实现链式调用的建造者
- [ ] 使用 Functional Options 模式
- [ ] 区分浅拷贝和深拷贝
- [ ] 实现对象的深拷贝
- [ ] 在实际项目中应用这些模式
- [ ] 识别何时应该使用这些模式
- [ ] 评估模式的优缺点和权衡

## 🔗 相关资源

### 推荐阅读
- [Functional Options in Go](https://dave.cheney.net/2014/10/17/functional-options-for-friendly-apis)
- [Builder Pattern - Refactoring.Guru](https://refactoring.guru/design-patterns/builder)
- [Prototype Pattern - Refactoring.Guru](https://refactoring.guru/design-patterns/prototype)

### 开源项目示例
- gRPC 的 Functional Options
- Docker SDK 的建造者模式
- Kubernetes 的对象深拷贝

## 📞 常见问题

### Q1: 什么时候使用建造者模式？
当对象有多个可选参数，或者构建过程复杂时。在 Go 中，推荐使用 Functional Options 模式。

### Q2: 深拷贝的性能开销大吗？
是的，深拷贝需要递归复制所有数据。如果性能敏感，考虑使用浅拷贝或写时复制（Copy-on-Write）。

### Q3: Go 语言如何实现深拷贝？
可以使用 encoding/gob、json 序列化，或手动实现 Clone 方法。选择取决于性能需求和对象复杂度。

## 🎓 下一步

完成本 Lab 后，继续学习：
- [Lab 03: 结构型模式 - 适配器、装饰器、代理](../lab03-结构型模式-适配器装饰器代理/README.md)

---

**开始时间**: ___________  
**完成时间**: ___________  
**学习笔记**: ___________

**祝学习愉快！** 🎉
