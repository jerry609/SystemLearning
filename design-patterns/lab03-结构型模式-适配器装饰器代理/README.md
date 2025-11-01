# Lab 03: 结构型模式 - 适配器、装饰器、代理 ⭐⭐⭐

> **重要提示**: 这是最重要的结构型模式学习单元！这三个模式在实际开发中使用频率极高，是构建可扩展系统的核心技术。

## 📚 学习目标

- 理解适配器模式的接口转换原理
- 掌握装饰器模式的动态功能扩展
- 学会使用代理模式控制对象访问
- 理解三种模式的区别和联系
- 完成 HTTP 中间件系统实战项目

## 📖 内容概览

### 1. 适配器模式 (Adapter Pattern)

**定义**: 将一个类的接口转换成客户期望的另一个接口，使原本接口不兼容的类可以一起工作。

**应用场景**:
- 集成第三方库
- 接口版本兼容
- 遗留系统改造
- 统一多个接口
- 数据格式转换

**实现方式**:
1. 类适配器（继承）
2. 对象适配器（组合）- Go 语言推荐
3. 接口适配器

### 2. 装饰器模式 (Decorator Pattern)

**定义**: 动态地给对象添加额外的职责，提供了比继承更灵活的扩展方式。

**应用场景**:
- HTTP 中间件
- 日志记录
- 性能监控
- 缓存功能
- 权限验证
- 数据压缩/加密

**实现方式**:
1. 传统装饰器（包装对象）
2. 函数式装饰器（高阶函数）
3. 中间件链模式

### 3. 代理模式 (Proxy Pattern)

**定义**: 为其他对象提供一种代理以控制对这个对象的访问。

**应用场景**:
- 远程代理（RPC）
- 虚拟代理（延迟加载）
- 保护代理（权限控制）
- 缓存代理
- 智能引用
- 日志代理

**实现方式**:
1. 静态代理
2. 动态代理
3. 远程代理

## 🗂️ 目录结构

```
lab03-结构型模式-适配器装饰器代理/
├── README.md                    # 本文件
├── theory/                      # 理论讲解
│   ├── 01-adapter.md           # 适配器模式详解
│   ├── 02-decorator.md         # 装饰器模式详解
│   └── 03-proxy.md             # 代理模式详解
├── examples/                    # 示例代码
│   ├── adapter/
│   │   ├── interface_adapter.go    # 接口适配示例
│   │   └── third_party_adapter.go  # 第三方库适配
│   ├── decorator/
│   │   ├── http_middleware.go      # HTTP 中间件
│   │   ├── logger_decorator.go     # 日志装饰器
│   │   └── cache_decorator.go      # 缓存装饰器
│   └── proxy/
│       ├── cache_proxy.go          # 缓存代理
│       └── rpc_proxy.go            # RPC 代理
├── exercises/                   # 练习题
│   ├── exercise1.md            # 适配器模式练习
│   ├── exercise2.md            # 装饰器模式练习
│   ├── exercise3.md            # 代理模式练习
│   └── answers/                # 练习答案
│       ├── exercise1_answer.go
│       ├── exercise2_answer.go
│       └── exercise3_answer.go
└── project/                     # 实战项目
    └── http-middleware-system/ # HTTP 中间件系统
        ├── README.md
        ├── middleware.go
        ├── logger.go
        ├── auth.go
        ├── ratelimit.go
        └── middleware_test.go
```

## 🚀 快速开始

### 1. 学习理论

按顺序阅读 `theory/` 目录下的文档：
1. `01-adapter.md` - 适配器模式
2. `02-decorator.md` - 装饰器模式
3. `03-proxy.md` - 代理模式

### 2. 运行示例

```bash
# 进入示例目录
cd examples/adapter

# 运行适配器模式示例
go run interface_adapter.go
go run third_party_adapter.go

# 进入装饰器模式示例
cd ../decorator
go run http_middleware.go
go run logger_decorator.go
go run cache_decorator.go

# 进入代理模式示例
cd ../proxy
go run cache_proxy.go
go run rpc_proxy.go
```

### 3. 完成练习

打开 `exercises/` 目录下的练习题，完成三个练习。

### 4. 实战项目

完成 HTTP 中间件系统实战项目 (`project/http-middleware-system/`)

## 📝 学习路径

### 初学者路径 (6-8 小时)

1. **理论学习** (2-3 小时)
   - [ ] 阅读适配器模式理论
   - [ ] 阅读装饰器模式理论
   - [ ] 阅读代理模式理论
   - [ ] 理解三种模式的区别

2. **示例代码** (2-3 小时)
   - [ ] 运行并理解适配器模式示例
   - [ ] 运行并理解装饰器模式示例
   - [ ] 运行并理解代理模式示例
   - [ ] 对比三种模式的实现

3. **练习题** (1-2 小时)
   - [ ] 完成练习 1：实现数据库适配器
   - [ ] 完成练习 2：实现日志装饰器
   - [ ] 完成练习 3：实现缓存代理

4. **实战项目** (2-3 小时)
   - [ ] 完成 HTTP 中间件系统项目

### 进阶路径 (8-12 小时)

在初学者路径基础上：

5. **深入研究** (2-3 小时)
   - [ ] 研究 Go 标准库中的这些模式
   - [ ] 分析 Web 框架的中间件实现
   - [ ] 对比不同语言的实现方式

6. **扩展练习** (3-4 小时)
   - [ ] 实现完整的 API 网关
   - [ ] 实现分布式缓存代理
   - [ ] 实现插件系统（适配器）
   - [ ] 性能测试和优化

## 💡 关键概念

### 三种模式的对比

| 特性 | 适配器模式 | 装饰器模式 | 代理模式 |
|------|-----------|-----------|---------|
| **意图** | 接口转换 | 功能增强 | 访问控制 |
| **改变接口** | 是 | 否 | 否 |
| **添加功能** | 否 | 是 | 可以 |
| **对象关系** | 包装不兼容对象 | 包装同接口对象 | 包装同接口对象 |
| **使用时机** | 接口不匹配 | 动态扩展功能 | 控制访问 |
| **典型场景** | 第三方库集成 | 中间件、日志 | 缓存、权限 |

### 适配器模式

**优点**:
- ✅ 提高类的复用性
- ✅ 解耦客户端和实现
- ✅ 符合开闭原则

**缺点**:
- ❌ 增加系统复杂度
- ❌ 可能影响性能

**Go 语言最佳实践**:
```go
// 目标接口
type Target interface {
    Request() string
}

// 被适配者
type Adaptee struct{}

func (a *Adaptee) SpecificRequest() string {
    return "Specific request"
}

// 适配器
type Adapter struct {
    adaptee *Adaptee
}

func (a *Adapter) Request() string {
    return a.adaptee.SpecificRequest()
}
```

### 装饰器模式

**优点**:
- ✅ 比继承更灵活
- ✅ 动态组合功能
- ✅ 符合单一职责原则
- ✅ 符合开闭原则

**缺点**:
- ❌ 产生很多小对象
- ❌ 调试困难

**Go 语言最佳实践 - 中间件模式**:
```go
type HandlerFunc func(http.ResponseWriter, *http.Request)

type Middleware func(HandlerFunc) HandlerFunc

func Logger() Middleware {
    return func(next HandlerFunc) HandlerFunc {
        return func(w http.ResponseWriter, r *http.Request) {
            log.Printf("Request: %s %s", r.Method, r.URL.Path)
            next(w, r)
        }
    }
}

func Chain(h HandlerFunc, middlewares ...Middleware) HandlerFunc {
    for i := len(middlewares) - 1; i >= 0; i-- {
        h = middlewares[i](h)
    }
    return h
}
```

### 代理模式

**优点**:
- ✅ 控制对象访问
- ✅ 延迟初始化
- ✅ 添加额外逻辑
- ✅ 符合开闭原则

**缺点**:
- ❌ 增加响应时间
- ❌ 增加系统复杂度

**Go 语言最佳实践**:
```go
type Subject interface {
    Request() string
}

type RealSubject struct{}

func (r *RealSubject) Request() string {
    return "Real request"
}

type Proxy struct {
    realSubject *RealSubject
    cache       map[string]string
}

func (p *Proxy) Request() string {
    if result, ok := p.cache["request"]; ok {
        return result
    }
    result := p.realSubject.Request()
    p.cache["request"] = result
    return result
}
```

## 🎯 练习题预览

### 练习 1: 实现数据库适配器

使用适配器模式统一不同数据库的接口，要求：
- 支持 MySQL、PostgreSQL、MongoDB
- 统一的 CRUD 接口
- 适配不同的查询语法
- 易于扩展新的数据库

### 练习 2: 实现日志装饰器链

使用装饰器模式实现日志系统，要求：
- 支持多个装饰器组合
- 实现时间戳、级别、格式化等装饰器
- 支持动态添加/移除装饰器
- 性能优化

### 练习 3: 实现智能缓存代理

使用代理模式实现缓存系统，要求：
- 支持多级缓存
- 实现缓存预热和失效
- 支持缓存统计
- 线程安全

## 📊 学习检查清单

完成本 Lab 后，你应该能够：

- [ ] 解释三种模式的意图和区别
- [ ] 识别何时使用哪种模式
- [ ] 实现接口适配器
- [ ] 实现 HTTP 中间件链
- [ ] 实现缓存代理
- [ ] 组合使用多种模式
- [ ] 在实际项目中应用这些模式
- [ ] 评估模式的优缺点和权衡

## 🔗 相关资源

### 推荐阅读
- [Adapter Pattern - Refactoring.Guru](https://refactoring.guru/design-patterns/adapter)
- [Decorator Pattern - Refactoring.Guru](https://refactoring.guru/design-patterns/decorator)
- [Proxy Pattern - Refactoring.Guru](https://refactoring.guru/design-patterns/proxy)
- [Go Middleware Pattern](https://drstearns.github.io/tutorials/gomiddleware/)

### 开源项目示例
- Gin 框架的中间件实现
- gRPC 的拦截器（装饰器）
- Kubernetes 的 API 代理
- Docker 的适配器模式

## 📞 常见问题

### Q1: 适配器和装饰器有什么区别？
适配器改变接口，装饰器保持接口不变但增强功能。适配器用于兼容，装饰器用于扩展。

### Q2: 装饰器和代理有什么区别？
装饰器关注功能增强，代理关注访问控制。装饰器可以多层嵌套，代理通常只有一层。

### Q3: Go 语言如何实现动态代理？
Go 没有反射代理，但可以使用接口和组合实现静态代理，或使用代码生成工具。

### Q4: 中间件是装饰器模式吗？
是的，HTTP 中间件是装饰器模式的典型应用，通过函数组合实现功能链。

## 🎓 下一步

完成本 Lab 后，继续学习：
- [Lab 04: 结构型模式 - 组合、外观、桥接](../lab04-结构型模式-组合外观桥接/README.md)

---

**开始时间**: ___________  
**完成时间**: ___________  
**学习笔记**: ___________

**祝学习愉快！** 🎉
