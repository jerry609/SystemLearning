# 设计模式组合

## 概述

在实际项目中，单个设计模式往往不足以解决复杂问题。本文档介绍常见的设计模式组合方式，帮助你构建更强大和灵活的系统。

## 为什么需要组合模式？

### 1. 复杂问题需要多种解决方案

单个模式通常只解决一个特定问题，而实际系统往往面临多个相关问题。

### 2. 模式之间可以互补

不同模式关注不同方面，组合使用可以发挥各自优势。

### 3. 提高系统灵活性

模式组合可以创建更灵活、更易扩展的架构。

## 常见模式组合

### 1. 工厂 + 策略

**场景**: 需要创建不同策略对象

**组合方式**:
- 工厂模式：创建策略对象
- 策略模式：封装算法

**示例**:
```go
// 策略接口
type PaymentStrategy interface {
    Pay(amount float64) error
}

// 具体策略
type AlipayStrategy struct{}
type WechatStrategy struct{}

// 策略工厂
type PaymentStrategyFactory struct{}

func (f *PaymentStrategyFactory) CreateStrategy(strategyType string) PaymentStrategy {
    switch strategyType {
    case "alipay":
        return &AlipayStrategy{}
    case "wechat":
        return &WechatStrategy{}
    default:
        return &AlipayStrategy{}
    }
}

// 使用
factory := &PaymentStrategyFactory{}
strategy := factory.CreateStrategy("alipay")
strategy.Pay(100.0)
```

**优点**:
- 解耦策略创建和使用
- 易于添加新策略
- 统一的创建接口

**应用场景**:
- 支付系统
- 负载均衡
- 数据压缩

### 2. 装饰器 + 责任链

**场景**: 需要灵活组合多个处理器

**组合方式**:
- 装饰器模式：动态添加功能
- 责任链模式：按顺序处理请求

**示例**:
```go
// 处理器接口
type Handler interface {
    Handle(ctx *Context) error
}

// 中间件（装饰器）
type LoggingMiddleware struct {
    next Handler
}

func (m *LoggingMiddleware) Handle(ctx *Context) error {
    log.Println("Request started")
    err := m.next.Handle(ctx)
    log.Println("Request completed")
    return err
}

// 中间件链（责任链）
type MiddlewareChain struct {
    handlers []Handler
}

func (c *MiddlewareChain) Handle(ctx *Context) error {
    for _, handler := range c.handlers {
        if err := handler.Handle(ctx); err != nil {
            return err
        }
    }
    return nil
}

// 使用
chain := &MiddlewareChain{}
chain.Add(&LoggingMiddleware{})
chain.Add(&AuthMiddleware{})
chain.Add(&BusinessHandler{})
chain.Handle(ctx)
```

**优点**:
- 灵活的功能组合
- 易于添加和删除处理器
- 清晰的处理流程

**应用场景**:
- HTTP 中间件
- 数据处理管道
- 请求过滤

### 3. 单例 + 工厂

**场景**: 需要全局唯一的工厂实例

**组合方式**:
- 单例模式：确保工厂唯一
- 工厂模式：创建对象

**示例**:
```go
// 单例工厂
type ServiceFactory struct {
    services map[string]Service
}

var (
    instance *ServiceFactory
    once     sync.Once
)

func GetServiceFactory() *ServiceFactory {
    once.Do(func() {
        instance = &ServiceFactory{
            services: make(map[string]Service),
        }
    })
    return instance
}

func (f *ServiceFactory) CreateService(serviceType string) Service {
    // 创建服务
}

// 使用
factory := GetServiceFactory()
service := factory.CreateService("user")
```

**优点**:
- 全局统一的对象创建
- 避免重复创建工厂
- 便于管理和配置

**应用场景**:
- 服务注册中心
- 对象池管理
- 资源管理器

### 4. 代理 + 单例

**场景**: 需要控制对单例对象的访问

**组合方式**:
- 单例模式：确保对象唯一
- 代理模式：控制访问

**示例**:
```go
// 单例对象
type Database struct{}

var (
    dbInstance *Database
    dbOnce     sync.Once
)

func GetDatabase() *Database {
    dbOnce.Do(func() {
        dbInstance = &Database{}
    })
    return dbInstance
}

// 代理
type DatabaseProxy struct {
    db *Database
}

func NewDatabaseProxy() *DatabaseProxy {
    return &DatabaseProxy{
        db: GetDatabase(),
    }
}

func (p *DatabaseProxy) Query(sql string) ([]Row, error) {
    // 添加权限检查、日志、缓存等
    log.Printf("Executing query: %s", sql)
    return p.db.Query(sql)
}
```

**优点**:
- 控制对单例的访问
- 添加额外功能（日志、缓存等）
- 保护单例对象

**应用场景**:
- 数据库连接
- 配置管理
- 缓存系统

### 5. 观察者 + 命令

**场景**: 需要记录和撤销事件

**组合方式**:
- 观察者模式：事件通知
- 命令模式：封装操作

**示例**:
```go
// 命令接口
type Command interface {
    Execute() error
    Undo() error
}

// 事件总线（观察者）
type EventBus struct {
    subscribers map[string][]Observer
    history     []Command
}

func (b *EventBus) Publish(event Event) error {
    // 创建命令
    cmd := NewEventCommand(event)
    
    // 执行命令
    if err := cmd.Execute(); err != nil {
        return err
    }
    
    // 记录历史
    b.history = append(b.history, cmd)
    
    // 通知观察者
    for _, observer := range b.subscribers[event.Type()] {
        observer.Update(event)
    }
    
    return nil
}

func (b *EventBus) Undo() error {
    if len(b.history) == 0 {
        return nil
    }
    
    cmd := b.history[len(b.history)-1]
    b.history = b.history[:len(b.history)-1]
    
    return cmd.Undo()
}
```

**优点**:
- 支持事件撤销
- 记录事件历史
- 解耦事件发布和处理

**应用场景**:
- 编辑器
- 工作流系统
- 事务管理

### 6. 建造者 + 工厂

**场景**: 需要创建复杂对象的不同变体

**组合方式**:
- 工厂模式：决定创建哪种建造者
- 建造者模式：构建复杂对象

**示例**:
```go
// 建造者接口
type Builder interface {
    SetEngine(engine string) Builder
    SetWheels(wheels int) Builder
    Build() *Car
}

// 具体建造者
type SportsCarBuilder struct {
    car *Car
}

type SUVBuilder struct {
    car *Car
}

// 建造者工厂
type BuilderFactory struct{}

func (f *BuilderFactory) CreateBuilder(carType string) Builder {
    switch carType {
    case "sports":
        return &SportsCarBuilder{car: &Car{}}
    case "suv":
        return &SUVBuilder{car: &Car{}}
    default:
        return &SportsCarBuilder{car: &Car{}}
    }
}

// 使用
factory := &BuilderFactory{}
builder := factory.CreateBuilder("sports")
car := builder.
    SetEngine("V8").
    SetWheels(4).
    Build()
```

**优点**:
- 灵活创建不同类型的复杂对象
- 统一的构建接口
- 易于扩展新类型

**应用场景**:
- 配置对象创建
- 文档生成
- UI 组件构建

### 7. 模板方法 + 策略

**场景**: 算法骨架固定，但某些步骤可替换

**组合方式**:
- 模板方法模式：定义算法骨架
- 策略模式：替换可变步骤

**示例**:
```go
// 策略接口
type SortStrategy interface {
    Sort(data []int) []int
}

// 模板方法
type DataProcessor struct {
    sortStrategy SortStrategy
}

func (p *DataProcessor) Process(data []int) []int {
    // 1. 预处理（固定）
    data = p.preprocess(data)
    
    // 2. 排序（可变，使用策略）
    data = p.sortStrategy.Sort(data)
    
    // 3. 后处理（固定）
    data = p.postprocess(data)
    
    return data
}

func (p *DataProcessor) preprocess(data []int) []int {
    // 固定的预处理逻辑
    return data
}

func (p *DataProcessor) postprocess(data []int) []int {
    // 固定的后处理逻辑
    return data
}
```

**优点**:
- 固定流程，灵活步骤
- 代码复用
- 易于扩展

**应用场景**:
- 数据处理流程
- 测试框架
- 工作流引擎

### 8. 适配器 + 外观

**场景**: 需要统一多个不兼容接口

**组合方式**:
- 适配器模式：转换接口
- 外观模式：提供统一接口

**示例**:
```go
// 第三方服务（接口不同）
type AlipayService struct{}
type WechatService struct{}
type BankService struct{}

// 适配器
type AlipayAdapter struct {
    service *AlipayService
}

func (a *AlipayAdapter) Pay(amount float64) error {
    // 适配 Alipay 接口
    return a.service.ProcessPayment(amount)
}

// 外观
type PaymentFacade struct {
    adapters map[string]PaymentAdapter
}

func (f *PaymentFacade) Pay(provider string, amount float64) error {
    adapter := f.adapters[provider]
    return adapter.Pay(amount)
}

// 使用
facade := &PaymentFacade{
    adapters: map[string]PaymentAdapter{
        "alipay": &AlipayAdapter{},
        "wechat": &WechatAdapter{},
        "bank":   &BankAdapter{},
    },
}
facade.Pay("alipay", 100.0)
```

**优点**:
- 统一不同接口
- 简化客户端调用
- 易于集成新服务

**应用场景**:
- 第三方服务集成
- API 网关
- 多数据源访问

## 组合模式的注意事项

### 1. 避免过度组合

**问题**: 组合太多模式导致系统过于复杂

**解决方案**:
- 只在必要时组合模式
- 保持简单性
- 定期重构

### 2. 保持清晰的职责

**问题**: 模式职责混乱

**解决方案**:
- 明确每个模式的职责
- 避免职责重叠
- 使用清晰的命名

### 3. 考虑性能影响

**问题**: 多层包装影响性能

**解决方案**:
- 测量性能影响
- 在关键路径上谨慎使用
- 必要时优化

### 4. 文档和注释

**问题**: 复杂的组合难以理解

**解决方案**:
- 编写清晰的文档
- 添加必要的注释
- 绘制架构图

## 实战案例

### 案例 1: 微服务框架

**组合模式**:
- 工厂模式：创建服务实例
- 装饰器模式：添加中间件
- 责任链模式：处理请求
- 观察者模式：事件通知
- 策略模式：负载均衡

**架构**:
```
ServiceFactory (工厂)
    ↓
Service (被装饰对象)
    ↓
MiddlewareDecorator (装饰器)
    ↓
MiddlewareChain (责任链)
    ↓
EventBus (观察者)
    ↓
LoadBalancer (策略)
```

### 案例 2: Web 框架

**组合模式**:
- 建造者模式：构建路由
- 责任链模式：处理请求
- 模板方法模式：渲染模板
- 单例模式：应用实例

**架构**:
```
Application (单例)
    ↓
RouterBuilder (建造者)
    ↓
HandlerChain (责任链)
    ↓
TemplateEngine (模板方法)
```

### 案例 3: 缓存系统

**组合模式**:
- 单例模式：缓存管理器
- 代理模式：缓存代理
- 策略模式：淘汰策略
- 观察者模式：缓存事件

**架构**:
```
CacheManager (单例)
    ↓
CacheProxy (代理)
    ↓
EvictionStrategy (策略)
    ↓
CacheEventBus (观察者)
```

## 最佳实践

### 1. 渐进式组合

- 从简单开始
- 逐步添加模式
- 根据需求调整

### 2. 保持平衡

- 功能性 vs 复杂度
- 灵活性 vs 性能
- 扩展性 vs 可维护性

### 3. 团队协作

- 统一理解
- 代码审查
- 知识分享

### 4. 持续改进

- 定期重构
- 优化性能
- 更新文档

## 总结

设计模式组合的关键是：

1. **理解每个模式的职责**: 明确各自解决什么问题
2. **识别协作关系**: 找出模式之间的配合点
3. **保持简单**: 不要过度组合
4. **注重实效**: 解决实际问题
5. **持续优化**: 根据反馈调整

记住：**好的组合是自然的，不是强制的。模式应该相互补充，而不是相互冲突。**
