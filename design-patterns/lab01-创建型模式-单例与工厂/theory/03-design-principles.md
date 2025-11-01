# 设计原则

## 概述

设计原则是指导我们进行软件设计的基本准则。遵循这些原则可以帮助我们写出高质量、易维护、可扩展的代码。

## SOLID 原则

SOLID 是面向对象设计的五大原则的首字母缩写：

### 1. 单一职责原则 (Single Responsibility Principle, SRP)

**定义**: 一个类应该只有一个引起它变化的原因。

**核心思想**: 一个类只负责一项职责。

**示例**:

```go
// ❌ 违反 SRP - 一个类承担多个职责
type User struct {
    Name  string
    Email string
}

func (u *User) Save() error {
    // 保存到数据库
    return nil
}

func (u *User) SendEmail(message string) error {
    // 发送邮件
    return nil
}

func (u *User) GenerateReport() string {
    // 生成报告
    return ""
}

// ✅ 符合 SRP - 职责分离
type User struct {
    Name  string
    Email string
}

type UserRepository struct{}

func (r *UserRepository) Save(user *User) error {
    // 保存到数据库
    return nil
}

type EmailService struct{}

func (s *EmailService) SendEmail(to, message string) error {
    // 发送邮件
    return nil
}

type ReportGenerator struct{}

func (g *ReportGenerator) GenerateUserReport(user *User) string {
    // 生成报告
    return ""
}
```

**优点**:
- ✅ 降低类的复杂度
- ✅ 提高可读性和可维护性
- ✅ 降低变更风险
- ✅ 易于测试

**在单例和工厂模式中的应用**:
- 单例类应该只负责管理自己的实例
- 工厂类应该只负责创建对象

### 2. 开闭原则 (Open-Closed Principle, OCP)

**定义**: 软件实体应该对扩展开放，对修改关闭。

**核心思想**: 通过扩展来实现变化，而不是修改现有代码。

**示例**:

```go
// ❌ 违反 OCP - 添加新类型需要修改工厂
type LoggerFactory struct{}

func (f *LoggerFactory) CreateLogger(loggerType string) Logger {
    switch loggerType {
    case "console":
        return &ConsoleLogger{}
    case "file":
        return &FileLogger{}
    // 添加新类型需要修改这里
    default:
        return &ConsoleLogger{}
    }
}

// ✅ 符合 OCP - 使用注册机制
var loggerRegistry = make(map[string]func() Logger)

func RegisterLogger(name string, factory func() Logger) {
    loggerRegistry[name] = factory
}

func CreateLogger(name string) Logger {
    if factory, ok := loggerRegistry[name]; ok {
        return factory()
    }
    return &ConsoleLogger{}
}

// 添加新类型不需要修改现有代码
func init() {
    RegisterLogger("console", func() Logger { return &ConsoleLogger{} })
    RegisterLogger("file", func() Logger { return &FileLogger{} })
    RegisterLogger("syslog", func() Logger { return &SyslogLogger{} })
}
```

**优点**:
- ✅ 提高可扩展性
- ✅ 降低维护成本
- ✅ 提高代码稳定性

**在单例和工厂模式中的应用**:
- 简单工厂违反 OCP
- 工厂方法和抽象工厂符合 OCP

### 3. 里氏替换原则 (Liskov Substitution Principle, LSP)

**定义**: 子类对象应该能够替换其父类对象被使用。

**核心思想**: 继承必须确保超类所拥有的性质在子类中仍然成立。

**示例**:

```go
// Logger 接口
type Logger interface {
    Log(message string) error
}

// ConsoleLogger 实现
type ConsoleLogger struct{}

func (l *ConsoleLogger) Log(message string) error {
    fmt.Println(message)
    return nil
}

// FileLogger 实现
type FileLogger struct {
    filename string
}

func (l *FileLogger) Log(message string) error {
    // 写入文件
    return nil
}

// ✅ 符合 LSP - 可以互相替换
func ProcessLog(logger Logger, message string) {
    logger.Log(message) // 不关心具体实现
}

func main() {
    ProcessLog(&ConsoleLogger{}, "test")
    ProcessLog(&FileLogger{filename: "app.log"}, "test")
}
```

**优点**:
- ✅ 提高代码复用性
- ✅ 提高系统灵活性
- ✅ 降低耦合度

**在单例和工厂模式中的应用**:
- 工厂创建的对象应该可以互相替换
- 使用接口而非具体类型

### 4. 接口隔离原则 (Interface Segregation Principle, ISP)

**定义**: 客户端不应该依赖它不需要的接口。

**核心思想**: 接口应该小而专一，不要强迫客户端依赖它们不使用的方法。

**示例**:

```go
// ❌ 违反 ISP - 接口过大
type Worker interface {
    Work()
    Eat()
    Sleep()
}

type Robot struct{}

func (r *Robot) Work() {
    fmt.Println("Robot working")
}

func (r *Robot) Eat() {
    // Robot 不需要吃饭，但被迫实现
}

func (r *Robot) Sleep() {
    // Robot 不需要睡觉，但被迫实现
}

// ✅ 符合 ISP - 接口分离
type Workable interface {
    Work()
}

type Eatable interface {
    Eat()
}

type Sleepable interface {
    Sleep()
}

type Robot struct{}

func (r *Robot) Work() {
    fmt.Println("Robot working")
}

type Human struct{}

func (h *Human) Work() {
    fmt.Println("Human working")
}

func (h *Human) Eat() {
    fmt.Println("Human eating")
}

func (h *Human) Sleep() {
    fmt.Println("Human sleeping")
}
```

**优点**:
- ✅ 降低接口复杂度
- ✅ 提高系统灵活性
- ✅ 提高代码可读性

**在单例和工厂模式中的应用**:
- 定义小而专一的接口
- 避免臃肿的接口

### 5. 依赖倒置原则 (Dependency Inversion Principle, DIP)

**定义**: 高层模块不应该依赖低层模块，两者都应该依赖抽象；抽象不应该依赖细节，细节应该依赖抽象。

**核心思想**: 面向接口编程，而不是面向实现编程。

**示例**:

```go
// ❌ 违反 DIP - 依赖具体实现
type MySQLDatabase struct{}

func (db *MySQLDatabase) Query(sql string) []map[string]interface{} {
    // 查询数据库
    return nil
}

type UserService struct {
    db *MySQLDatabase // 依赖具体实现
}

func (s *UserService) GetUser(id int) *User {
    s.db.Query("SELECT * FROM users WHERE id = ?")
    return nil
}

// ✅ 符合 DIP - 依赖抽象
type Database interface {
    Query(sql string) []map[string]interface{}
}

type MySQLDatabase struct{}

func (db *MySQLDatabase) Query(sql string) []map[string]interface{} {
    return nil
}

type PostgreSQLDatabase struct{}

func (db *PostgreSQLDatabase) Query(sql string) []map[string]interface{} {
    return nil
}

type UserService struct {
    db Database // 依赖抽象
}

func (s *UserService) GetUser(id int) *User {
    s.db.Query("SELECT * FROM users WHERE id = ?")
    return nil
}

// 可以注入不同的实现
func main() {
    service1 := &UserService{db: &MySQLDatabase{}}
    service2 := &UserService{db: &PostgreSQLDatabase{}}
}
```

**优点**:
- ✅ 降低耦合度
- ✅ 提高可测试性
- ✅ 提高灵活性

**在单例和工厂模式中的应用**:
- 工厂返回接口而非具体类型
- 使用依赖注入而非直接创建

## 其他重要原则

### 1. 迪米特法则 (Law of Demeter, LoD)

**定义**: 一个对象应该对其他对象有最少的了解。

**核心思想**: 只与直接的朋友通信，不要和陌生人说话。

**示例**:

```go
// ❌ 违反迪米特法则
type Wallet struct {
    money float64
}

type Person struct {
    wallet *Wallet
}

func (p *Person) GetWallet() *Wallet {
    return p.wallet
}

// 客户端需要了解 Wallet 的内部结构
func Pay(person *Person, amount float64) {
    wallet := person.GetWallet()
    if wallet.money >= amount {
        wallet.money -= amount
    }
}

// ✅ 符合迪米特法则
type Person struct {
    wallet *Wallet
}

func (p *Person) Pay(amount float64) bool {
    if p.wallet.money >= amount {
        p.wallet.money -= amount
        return true
    }
    return false
}

// 客户端不需要了解 Wallet
func Pay(person *Person, amount float64) {
    person.Pay(amount)
}
```

### 2. 合成复用原则 (Composite Reuse Principle, CRP)

**定义**: 尽量使用对象组合，而不是继承来达到复用的目的。

**核心思想**: 优先使用组合而非继承。

**示例**:

```go
// ❌ 使用继承
type Logger struct {
    level string
}

func (l *Logger) Log(message string) {
    fmt.Printf("[%s] %s\n", l.level, message)
}

type FileLogger struct {
    Logger
    filename string
}

// ✅ 使用组合
type Logger interface {
    Log(message string)
}

type ConsoleLogger struct {
    level string
}

func (l *ConsoleLogger) Log(message string) {
    fmt.Printf("[%s] %s\n", l.level, message)
}

type FileLogger struct {
    logger   Logger // 组合
    filename string
}

func (l *FileLogger) Log(message string) {
    l.logger.Log(message)
    // 同时写入文件
}
```

## 设计原则在单例和工厂模式中的应用

### 单例模式

1. **SRP**: 单例类只负责管理自己的实例
2. **OCP**: 通过接口扩展功能
3. **DIP**: 依赖单例接口而非具体实现

```go
// 定义接口
type Config interface {
    Get(key string) string
    Set(key, value string)
}

// 单例实现
type config struct {
    data map[string]string
    mu   sync.RWMutex
}

var (
    instance Config
    once     sync.Once
)

func GetConfig() Config {
    once.Do(func() {
        instance = &config{
            data: make(map[string]string),
        }
    })
    return instance
}

// 使用接口而非具体类型
func ProcessConfig(cfg Config) {
    value := cfg.Get("key")
    // ...
}
```

### 工厂模式

1. **SRP**: 工厂只负责创建对象
2. **OCP**: 通过注册机制扩展
3. **LSP**: 产品可以互相替换
4. **ISP**: 定义小而专一的接口
5. **DIP**: 返回接口而非具体类型

```go
// 定义接口
type Logger interface {
    Log(message string)
}

// 工厂接口
type LoggerFactory interface {
    CreateLogger() Logger
}

// 具体工厂
type ConsoleLoggerFactory struct{}

func (f *ConsoleLoggerFactory) CreateLogger() Logger {
    return &ConsoleLogger{}
}

// 使用
func ProcessLog(factory LoggerFactory, message string) {
    logger := factory.CreateLogger()
    logger.Log(message)
}
```

## 实践建议

### 1. 不要过度设计

设计原则是指导而非教条，在简单场景下不必严格遵循所有原则。

```go
// 简单场景 - 直接创建
logger := &ConsoleLogger{}

// 复杂场景 - 使用工厂
logger := loggerFactory.CreateLogger("console")
```

### 2. 平衡原则之间的冲突

有时候原则之间会有冲突，需要根据实际情况权衡。

### 3. 渐进式重构

不要一次性重构所有代码，而是逐步改进。

### 4. 编写测试

遵循设计原则的代码更容易测试。

```go
// 易于测试的代码
type UserService struct {
    db Database // 接口，可以 mock
}

func TestUserService(t *testing.T) {
    mockDB := &MockDatabase{}
    service := &UserService{db: mockDB}
    // 测试
}
```

## 总结

设计原则是软件设计的基石，遵循这些原则可以帮助我们写出高质量的代码。

**记住**:
- ✅ SOLID 原则是核心
- ✅ 单一职责原则最重要
- ✅ 开闭原则是目标
- ✅ 依赖倒置原则是手段
- ✅ 接口隔离原则是细节
- ✅ 里氏替换原则是约束
- ✅ 不要过度设计
- ✅ 根据实际情况权衡

**在单例和工厂模式中**:
- 单例模式主要体现 SRP 和 DIP
- 工厂模式体现所有 SOLID 原则
- 简单工厂违反 OCP，但简单实用
- 工厂方法和抽象工厂符合 OCP
- 始终使用接口而非具体类型
