# 设计模式选择指南

## 概述

选择合适的设计模式是软件设计中的关键决策。本指南帮助你根据具体场景选择最合适的设计模式，避免过度设计和模式滥用。

## 选择原则

### 1. 问题导向

**首先问自己**:
- 我要解决什么问题？
- 问题的本质是什么？
- 是否真的需要设计模式？

**记住**: 设计模式是工具，不是目的。不要为了使用模式而使用模式。

### 2. 变化维度分析

**识别变化点**:
- 哪些部分会变化？
- 哪些部分保持稳定？
- 变化的频率如何？

**原则**: 将变化的部分封装起来，使其独立于不变的部分。

### 3. 复杂度权衡

**评估成本**:
- 模式带来的好处是否大于增加的复杂度？
- 团队成员是否熟悉该模式？
- 是否有更简单的解决方案？

**原则**: 从简单开始，需要时再重构。

## 按问题分类选择

### 创建型模式：对象创建问题

#### 何时使用单例模式？

**适用场景**:
- 需要全局唯一的实例（配置管理器、日志管理器）
- 需要控制资源访问（数据库连接池）
- 需要全局访问点

**示例**:
```go
// 配置管理器
config := GetConfigManager()

// 日志管理器
logger := GetLogger()

// 缓存管理器
cache := GetCacheManager()
```

**注意事项**:
- 确保线程安全（使用 sync.Once）
- 避免滥用（不是所有全局变量都需要单例）
- 考虑依赖注入作为替代方案

#### 何时使用工厂模式？

**适用场景**:
- 需要创建多种类型的对象
- 创建逻辑复杂
- 需要解耦对象创建和使用

**选择哪种工厂**:
- **简单工厂**: 类型固定，创建逻辑简单
- **工厂方法**: 需要子类决定创建哪种对象
- **抽象工厂**: 需要创建一系列相关对象

**示例**:
```go
// 简单工厂：创建不同类型的日志器
logger := LoggerFactory.Create("file")

// 工厂方法：不同的数据库连接
conn := dbFactory.CreateConnection()

// 抽象工厂：创建一套 UI 组件
uiFactory := GetUIFactory("windows")
button := uiFactory.CreateButton()
textbox := uiFactory.CreateTextBox()
```

#### 何时使用建造者模式？

**适用场景**:
- 对象构造复杂，有多个可选参数
- 需要分步骤构建对象
- 需要创建不同表示的对象

**示例**:
```go
// HTTP 请求构建
request := NewRequestBuilder().
    URL("https://api.example.com").
    Method("POST").
    Header("Content-Type", "application/json").
    Body(data).
    Build()

// 路由构建
router := NewRouterBuilder().
    GET("/users", listUsers).
    POST("/users", createUser).
    Build()
```

**Go 语言特色**: 使用 Functional Options 模式
```go
server := NewServer(
    WithPort(8080),
    WithTimeout(30*time.Second),
    WithLogger(logger),
)
```

#### 何时使用原型模式？

**适用场景**:
- 对象创建成本高
- 需要复制现有对象
- 需要避免构造函数的复杂性

**示例**:
```go
// 复制配置对象
newConfig := originalConfig.Clone()

// 复制复杂的数据结构
newTree := tree.DeepCopy()
```

### 结构型模式：对象组织问题

#### 何时使用适配器模式？

**适用场景**:
- 需要使用现有类，但接口不匹配
- 需要集成第三方库
- 需要统一不同接口

**示例**:
```go
// 适配第三方日志库
logger := NewLoggerAdapter(thirdPartyLogger)

// 适配旧接口
newService := NewServiceAdapter(oldService)
```

#### 何时使用装饰器模式？

**适用场景**:
- 需要动态添加功能
- 需要组合多个功能
- 不想使用继承

**示例**:
```go
// HTTP 中间件
handler = LoggingDecorator(handler)
handler = AuthDecorator(handler)
handler = CacheDecorator(handler)

// 数据流处理
reader = BufferedReader(reader)
reader = CompressedReader(reader)
```

#### 何时使用代理模式？

**适用场景**:
- 需要控制对象访问
- 需要延迟加载
- 需要添加缓存层

**选择哪种代理**:
- **虚拟代理**: 延迟创建昂贵对象
- **保护代理**: 控制访问权限
- **远程代理**: 访问远程对象
- **缓存代理**: 添加缓存功能

**示例**:
```go
// 缓存代理
proxy := NewCacheProxy(dataSource)

// 延迟加载代理
proxy := NewLazyLoadProxy(heavyObject)

// 权限代理
proxy := NewAccessControlProxy(service)
```

#### 何时使用组合模式？

**适用场景**:
- 需要表示树形结构
- 需要统一处理单个对象和组合对象
- 需要递归结构

**示例**:
```go
// 文件系统
folder.Add(file1)
folder.Add(subfolder)
folder.Display()

// 组织架构
department.Add(employee)
department.Add(subDepartment)
```

#### 何时使用外观模式？

**适用场景**:
- 需要简化复杂子系统
- 需要提供统一接口
- 需要解耦客户端和子系统

**示例**:
```go
// API 网关
gateway := NewAPIGateway()
gateway.ProcessRequest(request)

// 系统启动器
starter := NewSystemStarter()
starter.Start()
```

#### 何时使用桥接模式？

**适用场景**:
- 抽象和实现需要独立变化
- 需要避免类爆炸
- 需要运行时切换实现

**示例**:
```go
// 跨平台 UI
button := NewButton(windowsRenderer)
button.SetRenderer(linuxRenderer)

// 多维度变化
message := NewMessage(emailSender)
message.SetSender(smsSender)
```

### 行为型模式：对象交互问题

#### 何时使用策略模式？

**适用场景**:
- 需要在运行时选择算法
- 有多个相关的类，仅行为不同
- 需要避免大量条件语句

**示例**:
```go
// 支付策略
payment := NewPayment(alipayStrategy)
payment.SetStrategy(wechatStrategy)

// 排序策略
sorter := NewSorter(quickSortStrategy)

// 负载均衡策略
balancer := NewLoadBalancer(roundRobinStrategy)
```

#### 何时使用观察者模式？

**适用场景**:
- 一个对象改变需要通知其他对象
- 不知道有多少对象需要通知
- 需要解耦发布者和订阅者

**示例**:
```go
// 事件系统
eventBus.Subscribe("UserCreated", emailHandler)
eventBus.Subscribe("UserCreated", logHandler)
eventBus.Publish(userCreatedEvent)

// 数据绑定
model.AddObserver(view)
model.SetValue(newValue)
```

#### 何时使用模板方法模式？

**适用场景**:
- 算法骨架固定，部分步骤可变
- 需要控制子类扩展点
- 需要代码复用

**示例**:
```go
// 数据处理流程
processor := NewDataProcessor()
processor.Process() // 调用模板方法

// 测试框架
test := NewTestCase()
test.Run() // setUp -> test -> tearDown
```

#### 何时使用状态模式？

**适用场景**:
- 对象行为取决于状态
- 有大量状态相关的条件语句
- 状态转换逻辑复杂

**示例**:
```go
// 订单状态机
order.SetState(NewPendingState())
order.Process()

// 游戏状态
game.SetState(NewPlayingState())
game.Update()
```

#### 何时使用命令模式？

**适用场景**:
- 需要参数化操作
- 需要支持撤销/重做
- 需要记录操作历史

**示例**:
```go
// 任务队列
queue.Add(NewTaskCommand(task))
queue.Execute()

// 撤销/重做
editor.Execute(NewInsertCommand(text))
editor.Undo()
```

#### 何时使用责任链模式？

**适用场景**:
- 多个对象可以处理请求
- 不知道哪个对象处理请求
- 需要动态指定处理者

**示例**:
```go
// HTTP 中间件
chain := NewChain()
chain.Use(LoggingMiddleware)
chain.Use(AuthMiddleware)
chain.Use(BusinessHandler)

// 审批流程
chain := NewApprovalChain()
chain.Add(managerApprover)
chain.Add(directorApprover)
```

## 决策树

```
开始
│
├─ 需要创建对象？
│  ├─ 全局唯一？ → 单例模式
│  ├─ 多种类型？ → 工厂模式
│  ├─ 构造复杂？ → 建造者模式
│  └─ 复制对象？ → 原型模式
│
├─ 需要组织结构？
│  ├─ 接口不匹配？ → 适配器模式
│  ├─ 动态添加功能？ → 装饰器模式
│  ├─ 控制访问？ → 代理模式
│  ├─ 树形结构？ → 组合模式
│  ├─ 简化接口？ → 外观模式
│  └─ 多维度变化？ → 桥接模式
│
└─ 需要定义交互？
   ├─ 切换算法？ → 策略模式
   ├─ 通知多个对象？ → 观察者模式
   ├─ 算法骨架？ → 模板方法模式
   ├─ 状态相关行为？ → 状态模式
   ├─ 参数化操作？ → 命令模式
   └─ 链式处理？ → 责任链模式
```

## 常见问题

### Q1: 如何避免过度设计？

**A1**: 遵循以下原则：
1. **YAGNI**: You Aren't Gonna Need It - 不要实现你不需要的功能
2. **从简单开始**: 先写简单的代码，需要时再重构
3. **实际需求**: 只在有明确需求时使用模式
4. **团队能力**: 考虑团队对模式的熟悉程度

### Q2: 多个模式都适用，如何选择？

**A2**: 考虑以下因素：
1. **简单性**: 选择最简单的方案
2. **可维护性**: 选择最易维护的方案
3. **性能**: 考虑性能影响
4. **团队熟悉度**: 选择团队熟悉的模式

### Q3: 何时应该重构为设计模式？

**A3**: 出现以下情况时考虑重构：
1. 代码重复
2. 大量条件语句
3. 类职责不清
4. 难以扩展
5. 难以测试

### Q4: 设计模式会影响性能吗？

**A4**: 
- 大多数模式对性能影响很小
- 某些模式（如代理、装饰器）可能增加间接调用
- 性能问题通常来自算法和数据结构，而非设计模式
- 先保证正确性和可维护性，再优化性能

### Q5: Go 语言中的设计模式有什么特殊之处？

**A5**: 
- **接口隐式实现**: 更灵活的多态
- **组合优于继承**: 使用嵌入实现代码复用
- **函数是一等公民**: 可以用函数实现策略模式
- **并发原语**: goroutine 和 channel 提供新的设计思路

## 实战建议

### 1. 学习顺序

**初学者**:
1. 单例模式
2. 工厂模式
3. 策略模式
4. 观察者模式

**进阶**:
1. 装饰器模式
2. 代理模式
3. 责任链模式
4. 模板方法模式

**高级**:
1. 抽象工厂模式
2. 桥接模式
3. 状态模式
4. 命令模式

### 2. 实践步骤

1. **理解问题**: 明确要解决的问题
2. **识别模式**: 找出适用的模式
3. **简单实现**: 先实现最简单的版本
4. **测试验证**: 编写测试确保正确性
5. **逐步完善**: 根据需求逐步完善

### 3. 代码审查清单

- [ ] 模式选择是否合理？
- [ ] 是否过度设计？
- [ ] 代码是否易于理解？
- [ ] 是否易于扩展？
- [ ] 是否易于测试？
- [ ] 性能是否可接受？

## 总结

选择设计模式的关键是：

1. **理解问题本质**: 不要被表面现象迷惑
2. **分析变化维度**: 找出真正需要变化的部分
3. **权衡利弊**: 考虑复杂度和收益
4. **从简单开始**: 不要过早优化
5. **持续重构**: 随着需求变化调整设计

记住：**设计模式是工具，不是目的。好的设计是简单、清晰、易于维护的。**
