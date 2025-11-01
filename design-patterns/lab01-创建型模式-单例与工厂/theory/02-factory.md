# 工厂模式 (Factory Pattern)

## 定义

工厂模式定义了一个创建对象的接口，但由子类决定要实例化的类是哪一个。工厂方法让类把实例化推迟到子类。

## 意图

- 将对象的创建与使用分离
- 提供统一的创建接口
- 隐藏对象创建的复杂性
- 提高代码的可扩展性和可维护性

## 三种形式

### 1. 简单工厂 (Simple Factory)

**定义**: 由一个工厂类根据传入的参数决定创建哪种产品类的实例。

**结构**:
```
┌─────────────────┐
│  SimpleFactory  │
├─────────────────┤
│ + Create(type)  │
└─────────────────┘
         │
         ├──────────┐
         ▼          ▼
   ┌─────────┐ ┌─────────┐
   │ ProductA│ │ ProductB│
   └─────────┘ └─────────┘
```

**优点**:
- ✅ 简单易懂
- ✅ 集中管理对象创建
- ✅ 客户端无需知道具体类名

**缺点**:
- ❌ 违反开闭原则（添加新产品需要修改工厂类）
- ❌ 工厂类职责过重
- ❌ 不易扩展

**适用场景**:
- 产品种类较少且固定
- 客户端不关心对象创建过程
- 需要集中管理对象创建

### 2. 工厂方法 (Factory Method)

**定义**: 定义一个创建对象的接口，让子类决定实例化哪个类。

**结构**:
```mermaid
classDiagram
    class Creator {
        <<interface>>
        +FactoryMethod() Product
        +Operation()
    }
    
    class ConcreteCreatorA {
        +FactoryMethod() Product
    }
    
    class ConcreteCreatorB {
        +FactoryMethod() Product
    }
    
    class Product {
        <<interface>>
        +Use()
    }
    
    class ConcreteProductA {
        +Use()
    }
    
    class ConcreteProductB {
        +Use()
    }
    
    Creator <|.. ConcreteCreatorA
    Creator <|.. ConcreteCreatorB
    Product <|.. ConcreteProductA
    Product <|.. ConcreteProductB
    ConcreteCreatorA --> ConcreteProductA
    ConcreteCreatorB --> ConcreteProductB
```

**优点**:
- ✅ 符合开闭原则
- ✅ 符合单一职责原则
- ✅ 易于扩展新产品
- ✅ 解耦对象创建和使用

**缺点**:
- ❌ 类的数量增加
- ❌ 增加系统复杂度
- ❌ 增加理解难度

**适用场景**:
- 不知道具体需要哪种产品
- 产品种类可能扩展
- 框架需要标准化创建过程

### 3. 抽象工厂 (Abstract Factory)

**定义**: 提供一个创建一系列相关或相互依赖对象的接口，而无需指定它们具体的类。

**结构**:
```mermaid
classDiagram
    class AbstractFactory {
        <<interface>>
        +CreateProductA() ProductA
        +CreateProductB() ProductB
    }
    
    class ConcreteFactory1 {
        +CreateProductA() ProductA
        +CreateProductB() ProductB
    }
    
    class ConcreteFactory2 {
        +CreateProductA() ProductA
        +CreateProductB() ProductB
    }
    
    class ProductA {
        <<interface>>
    }
    
    class ProductB {
        <<interface>>
    }
    
    class ProductA1
    class ProductA2
    class ProductB1
    class ProductB2
    
    AbstractFactory <|.. ConcreteFactory1
    AbstractFactory <|.. ConcreteFactory2
    ProductA <|.. ProductA1
    ProductA <|.. ProductA2
    ProductB <|.. ProductB1
    ProductB <|.. ProductB2
    ConcreteFactory1 --> ProductA1
    ConcreteFactory1 --> ProductB1
    ConcreteFactory2 --> ProductA2
    ConcreteFactory2 --> ProductB2
```

**优点**:
- ✅ 保证产品族的一致性
- ✅ 易于切换产品族
- ✅ 符合开闭原则
- ✅ 隔离具体类的生成

**缺点**:
- ❌ 难以支持新种类的产品
- ❌ 增加系统复杂度
- ❌ 类的数量大幅增加

**适用场景**:
- 系统需要多个产品族
- 产品族内的产品需要一起使用
- 需要约束产品的组合使用

## Go 语言实现

### 1. 简单工厂实现

```go
// Product 接口
type Logger interface {
    Log(message string)
}

// 具体产品
type ConsoleLogger struct{}

func (l *ConsoleLogger) Log(message string) {
    fmt.Println("[Console]", message)
}

type FileLogger struct {
    filename string
}

func (l *FileLogger) Log(message string) {
    // 写入文件
    fmt.Printf("[File: %s] %s\n", l.filename, message)
}

// 简单工厂
type LoggerFactory struct{}

func (f *LoggerFactory) CreateLogger(loggerType string) Logger {
    switch loggerType {
    case "console":
        return &ConsoleLogger{}
    case "file":
        return &FileLogger{filename: "app.log"}
    default:
        return &ConsoleLogger{}
    }
}

// 使用
func main() {
    factory := &LoggerFactory{}
    
    logger1 := factory.CreateLogger("console")
    logger1.Log("This is a console log")
    
    logger2 := factory.CreateLogger("file")
    logger2.Log("This is a file log")
}
```

### 2. 工厂方法实现

```go
// Product 接口
type Database interface {
    Connect() error
    Query(sql string) ([]map[string]interface{}, error)
}

// 具体产品
type MySQL struct {
    dsn string
}

func (db *MySQL) Connect() error {
    fmt.Println("Connecting to MySQL:", db.dsn)
    return nil
}

func (db *MySQL) Query(sql string) ([]map[string]interface{}, error) {
    fmt.Println("MySQL Query:", sql)
    return nil, nil
}

type PostgreSQL struct {
    dsn string
}

func (db *PostgreSQL) Connect() error {
    fmt.Println("Connecting to PostgreSQL:", db.dsn)
    return nil
}

func (db *PostgreSQL) Query(sql string) ([]map[string]interface{}, error) {
    fmt.Println("PostgreSQL Query:", sql)
    return nil, nil
}

// Creator 接口
type DatabaseFactory interface {
    CreateDatabase() Database
}

// 具体工厂
type MySQLFactory struct {
    dsn string
}

func (f *MySQLFactory) CreateDatabase() Database {
    return &MySQL{dsn: f.dsn}
}

type PostgreSQLFactory struct {
    dsn string
}

func (f *PostgreSQLFactory) CreateDatabase() Database {
    return &PostgreSQL{dsn: f.dsn}
}

// 使用
func main() {
    var factory DatabaseFactory
    
    factory = &MySQLFactory{dsn: "user:pass@tcp(localhost:3306)/db"}
    db1 := factory.CreateDatabase()
    db1.Connect()
    
    factory = &PostgreSQLFactory{dsn: "postgres://user:pass@localhost/db"}
    db2 := factory.CreateDatabase()
    db2.Connect()
}
```

### 3. 抽象工厂实现

```go
// 抽象产品 A
type Button interface {
    Render()
}

// 抽象产品 B
type Checkbox interface {
    Render()
}

// 具体产品 A1
type WindowsButton struct{}

func (b *WindowsButton) Render() {
    fmt.Println("Rendering Windows Button")
}

// 具体产品 A2
type MacButton struct{}

func (b *MacButton) Render() {
    fmt.Println("Rendering Mac Button")
}

// 具体产品 B1
type WindowsCheckbox struct{}

func (c *WindowsCheckbox) Render() {
    fmt.Println("Rendering Windows Checkbox")
}

// 具体产品 B2
type MacCheckbox struct{}

func (c *MacCheckbox) Render() {
    fmt.Println("Rendering Mac Checkbox")
}

// 抽象工厂
type GUIFactory interface {
    CreateButton() Button
    CreateCheckbox() Checkbox
}

// 具体工厂 1
type WindowsFactory struct{}

func (f *WindowsFactory) CreateButton() Button {
    return &WindowsButton{}
}

func (f *WindowsFactory) CreateCheckbox() Checkbox {
    return &WindowsCheckbox{}
}

// 具体工厂 2
type MacFactory struct{}

func (f *MacFactory) CreateButton() Button {
    return &MacButton{}
}

func (f *MacFactory) CreateCheckbox() Checkbox {
    return &MacCheckbox{}
}

// 使用
func main() {
    var factory GUIFactory
    
    // Windows 风格
    factory = &WindowsFactory{}
    button1 := factory.CreateButton()
    checkbox1 := factory.CreateCheckbox()
    button1.Render()
    checkbox1.Render()
    
    // Mac 风格
    factory = &MacFactory{}
    button2 := factory.CreateButton()
    checkbox2 := factory.CreateCheckbox()
    button2.Render()
    checkbox2.Render()
}
```

## 三种工厂模式对比

| 特性 | 简单工厂 | 工厂方法 | 抽象工厂 |
|------|----------|----------|----------|
| 复杂度 | 低 | 中 | 高 |
| 扩展性 | 差 | 好 | 很好 |
| 产品数量 | 单一产品 | 单一产品 | 产品族 |
| 开闭原则 | 不符合 | 符合 | 符合 |
| 类的数量 | 少 | 中等 | 多 |
| 适用场景 | 产品少且固定 | 产品可扩展 | 多个产品族 |

## 使用建议

### 何时使用简单工厂

✅ **应该使用**:
- 产品种类少（<5种）
- 产品种类固定，不会频繁变化
- 需要快速实现
- 代码简洁性优先

❌ **不应该使用**:
- 产品种类多且会增加
- 需要高度可扩展性
- 严格遵循开闭原则

### 何时使用工厂方法

✅ **应该使用**:
- 产品种类可能扩展
- 需要符合开闭原则
- 客户端不关心具体产品类
- 框架设计

❌ **不应该使用**:
- 产品种类固定且少
- 不需要扩展性
- 追求代码简洁

### 何时使用抽象工厂

✅ **应该使用**:
- 系统有多个产品族
- 产品族内的产品需要一起使用
- 需要约束产品组合
- 需要切换整个产品族

❌ **不应该使用**:
- 只有单一产品
- 产品族不稳定
- 系统简单

## 最佳实践

### 1. 使用接口而非具体类型

```go
// ✅ 好的做法
type Logger interface {
    Log(message string)
}

func NewLogger(loggerType string) Logger {
    // ...
}

// ❌ 不好的做法
func NewLogger(loggerType string) *ConsoleLogger {
    // ...
}
```

### 2. 提供默认实现

```go
func NewLogger(loggerType string) Logger {
    switch loggerType {
    case "console":
        return &ConsoleLogger{}
    case "file":
        return &FileLogger{}
    default:
        return &ConsoleLogger{} // 默认实现
    }
}
```

### 3. 使用函数式选项

```go
type LoggerOption func(*Logger)

func WithLevel(level string) LoggerOption {
    return func(l *Logger) {
        l.level = level
    }
}

func NewLogger(opts ...LoggerOption) Logger {
    logger := &ConsoleLogger{level: "info"}
    for _, opt := range opts {
        opt(logger)
    }
    return logger
}
```

### 4. 注册机制

```go
var loggerRegistry = make(map[string]func() Logger)

func RegisterLogger(name string, factory func() Logger) {
    loggerRegistry[name] = factory
}

func NewLogger(name string) Logger {
    if factory, ok := loggerRegistry[name]; ok {
        return factory()
    }
    return &ConsoleLogger{} // 默认
}

// 使用
func init() {
    RegisterLogger("console", func() Logger { return &ConsoleLogger{} })
    RegisterLogger("file", func() Logger { return &FileLogger{} })
}
```

## 实际应用

### 1. 数据库连接工厂

```go
type DBFactory interface {
    CreateConnection(dsn string) (Database, error)
}

type MySQLFactory struct{}

func (f *MySQLFactory) CreateConnection(dsn string) (Database, error) {
    db, err := sql.Open("mysql", dsn)
    if err != nil {
        return nil, err
    }
    return &MySQLDatabase{db: db}, nil
}
```

### 2. 日志工厂

```go
type LoggerFactory struct {
    level string
}

func (f *LoggerFactory) CreateLogger(output string) Logger {
    switch output {
    case "console":
        return &ConsoleLogger{level: f.level}
    case "file":
        return &FileLogger{level: f.level, filename: "app.log"}
    case "syslog":
        return &SyslogLogger{level: f.level}
    default:
        return &ConsoleLogger{level: f.level}
    }
}
```

### 3. HTTP 客户端工厂

```go
type HTTPClientFactory interface {
    CreateClient() *http.Client
}

type DefaultClientFactory struct{}

func (f *DefaultClientFactory) CreateClient() *http.Client {
    return &http.Client{
        Timeout: 30 * time.Second,
    }
}

type CustomClientFactory struct {
    timeout time.Duration
}

func (f *CustomClientFactory) CreateClient() *http.Client {
    return &http.Client{
        Timeout: f.timeout,
        Transport: &http.Transport{
            MaxIdleConns: 100,
        },
    }
}
```

## 常见问题

### Q1: 工厂模式和建造者模式有什么区别？

- **工厂模式**: 关注创建什么对象（对象类型）
- **建造者模式**: 关注如何创建对象（构建过程）

### Q2: 什么时候使用工厂模式？

当对象创建逻辑复杂，或需要根据条件创建不同类型的对象时。

### Q3: Go 语言中如何选择工厂模式？

- 简单场景：使用简单工厂或工厂函数
- 需要扩展：使用工厂方法
- 产品族：使用抽象工厂
- 配置复杂：结合 Functional Options

### Q4: 工厂模式会增加代码复杂度吗？

是的，但换来的是更好的可扩展性和可维护性。在简单场景下，直接创建对象即可。

## 总结

工厂模式是创建型模式中最常用的模式之一，它将对象的创建与使用分离，提高了代码的可扩展性和可维护性。

**记住**:
- ✅ 简单工厂适合产品少且固定的场景
- ✅ 工厂方法适合需要扩展的场景
- ✅ 抽象工厂适合产品族的场景
- ✅ 在 Go 中优先使用简单工厂和工厂函数
- ✅ 结合 Functional Options 提供灵活配置
- ❌ 不要过度设计，简单场景直接创建对象
