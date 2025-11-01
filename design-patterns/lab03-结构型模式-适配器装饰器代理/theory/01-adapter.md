# 适配器模式 (Adapter Pattern)

## 定义

适配器模式将一个类的接口转换成客户端期望的另一个接口，使得原本由于接口不兼容而不能一起工作的类可以协同工作。

## 意图

- 接口转换和兼容
- 复用现有类
- 集成第三方库
- 统一接口标准
- 解耦系统组件

## 别名

- 包装器（Wrapper）

## 动机

假设你正在开发一个应用，需要集成多个第三方支付系统，但它们的接口各不相同：

```go
// 支付宝的接口
type Alipay struct{}
func (a *Alipay) SendPayment(amount float64) { /* ... */ }

// 微信支付的接口
type WeChatPay struct{}
func (w *WeChatPay) Pay(money float64) { /* ... */ }

// PayPal 的接口
type PayPal struct{}
func (p *PayPal) MakePayment(value float64) { /* ... */ }
```

客户端代码需要统一的接口来处理支付，但每个支付系统的方法名和参数都不同。这时就需要适配器模式：

```go
// 统一的支付接口
type PaymentProcessor interface {
    ProcessPayment(amount float64) error
}

// 支付宝适配器
type AlipayAdapter struct {
    alipay *Alipay
}

func (a *AlipayAdapter) ProcessPayment(amount float64) error {
    a.alipay.SendPayment(amount)
    return nil
}
```


现在客户端可以使用统一的 `PaymentProcessor` 接口，而不需要关心底层使用的是哪个支付系统。

## 结构

### UML 类图

```mermaid
classDiagram
    class Target {
        <<interface>>
        +Request()
    }
    
    class Adapter {
        -adaptee: Adaptee
        +Request()
    }
    
    class Adaptee {
        +SpecificRequest()
    }
    
    class Client {
        +DoWork(target: Target)
    }
    
    Client --> Target
    Target <|.. Adapter
    Adapter --> Adaptee
```

### UML 时序图

```mermaid
sequenceDiagram
    participant Client
    participant Adapter
    participant Adaptee
    
    Client->>Adapter: Request()
    Adapter->>Adaptee: SpecificRequest()
    Adaptee-->>Adapter: result
    Adapter-->>Client: adapted result
```

## 参与者

- **Target（目标接口）**: 客户端期望的接口
- **Adapter（适配器）**: 实现 Target 接口，并持有 Adaptee 的引用
- **Adaptee（被适配者）**: 需要被适配的现有接口
- **Client（客户端）**: 使用 Target 接口的代码

## 协作

1. 客户端通过 Target 接口调用 Adapter
2. Adapter 将请求转换为 Adaptee 能理解的形式
3. Adaptee 执行实际的操作
4. Adapter 将结果转换为 Target 接口期望的格式返回给客户端


## 适用场景

1. **集成第三方库**
   - 第三方库接口与系统不兼容
   - 需要统一多个第三方库的接口
   - 避免直接依赖第三方库

2. **接口标准化**
   - 统一不同实现的接口
   - 提供一致的 API
   - 简化客户端代码

3. **遗留系统集成**
   - 新旧系统接口不兼容
   - 不能修改遗留代码
   - 需要渐进式重构

4. **数据格式转换**
   - JSON、XML、Protobuf 等格式转换
   - 数据库驱动适配
   - 消息队列适配

5. **跨平台兼容**
   - 不同操作系统的 API 适配
   - 不同数据库的 SQL 方言适配
   - 不同云服务商的 API 适配

## 实现方式

### 1. 对象适配器（组合方式，推荐）

```go
package main

import "fmt"

// Target - 目标接口
type MediaPlayer interface {
    Play(audioType string, filename string)
}

// Adaptee - 被适配的接口
type AdvancedMediaPlayer interface {
    PlayVLC(filename string)
    PlayMP4(filename string)
}

// Adaptee 的具体实现
type VLCPlayer struct{}

func (v *VLCPlayer) PlayVLC(filename string) {
    fmt.Printf("Playing VLC file: %s\n", filename)
}

func (v *VLCPlayer) PlayMP4(filename string) {
    // VLC 不支持 MP4
}

type MP4Player struct{}

func (m *MP4Player) PlayVLC(filename string) {
    // MP4 不支持 VLC
}

func (m *MP4Player) PlayMP4(filename string) {
    fmt.Printf("Playing MP4 file: %s\n", filename)
}

// Adapter - 适配器
type MediaAdapter struct {
    advancedPlayer AdvancedMediaPlayer
}

func NewMediaAdapter(audioType string) *MediaAdapter {
    var player AdvancedMediaPlayer
    
    switch audioType {
    case "vlc":
        player = &VLCPlayer{}
    case "mp4":
        player = &MP4Player{}
    }
    
    return &MediaAdapter{advancedPlayer: player}
}

func (a *MediaAdapter) Play(audioType string, filename string) {
    switch audioType {
    case "vlc":
        a.advancedPlayer.PlayVLC(filename)
    case "mp4":
        a.advancedPlayer.PlayMP4(filename)
    }
}

// ConcreteTarget - 具体目标类
type AudioPlayer struct {
    adapter *MediaAdapter
}

func (p *AudioPlayer) Play(audioType string, filename string) {
    // 内置支持 mp3
    if audioType == "mp3" {
        fmt.Printf("Playing MP3 file: %s\n", filename)
        return
    }
    
    // 使用适配器播放其他格式
    if audioType == "vlc" || audioType == "mp4" {
        p.adapter = NewMediaAdapter(audioType)
        p.adapter.Play(audioType, filename)
        return
    }
    
    fmt.Printf("Invalid media type: %s\n", audioType)
}

func main() {
    player := &AudioPlayer{}
    
    player.Play("mp3", "song.mp3")
    player.Play("vlc", "movie.vlc")
    player.Play("mp4", "video.mp4")
    player.Play("avi", "video.avi")
}
```


### 2. 接口适配器（Go 推荐）

```go
package main

import (
    "fmt"
    "io"
    "strings"
)

// 假设我们有一个旧的日志接口
type LegacyLogger interface {
    LogMessage(level int, message string)
}

type OldLogger struct{}

func (l *OldLogger) LogMessage(level int, message string) {
    levels := []string{"DEBUG", "INFO", "WARN", "ERROR"}
    fmt.Printf("[%s] %s\n", levels[level], message)
}

// 新的标准日志接口
type Logger interface {
    Debug(message string)
    Info(message string)
    Warn(message string)
    Error(message string)
}

// 适配器：将旧接口适配到新接口
type LoggerAdapter struct {
    legacy LegacyLogger
}

func NewLoggerAdapter(legacy LegacyLogger) *LoggerAdapter {
    return &LoggerAdapter{legacy: legacy}
}

func (a *LoggerAdapter) Debug(message string) {
    a.legacy.LogMessage(0, message)
}

func (a *LoggerAdapter) Info(message string) {
    a.legacy.LogMessage(1, message)
}

func (a *LoggerAdapter) Warn(message string) {
    a.legacy.LogMessage(2, message)
}

func (a *LoggerAdapter) Error(message string) {
    a.legacy.LogMessage(3, message)
}

// 客户端代码使用新接口
func DoWork(logger Logger) {
    logger.Debug("Starting work")
    logger.Info("Processing data")
    logger.Warn("Low memory")
    logger.Error("Operation failed")
}

func main() {
    // 使用适配器将旧日志系统适配到新接口
    oldLogger := &OldLogger{}
    adapter := NewLoggerAdapter(oldLogger)
    
    DoWork(adapter)
}
```

### 3. 函数适配器

```go
package main

import (
    "fmt"
    "io"
    "strings"
)

// 标准的 io.Reader 接口
// type Reader interface {
//     Read(p []byte) (n int, err error)
// }

// 假设我们有一个返回字符串的函数
type StringProvider func() string

// 适配器：将 StringProvider 适配为 io.Reader
type StringReaderAdapter struct {
    provider StringProvider
    reader   *strings.Reader
}

func NewStringReaderAdapter(provider StringProvider) *StringReaderAdapter {
    return &StringReaderAdapter{
        provider: provider,
    }
}

func (a *StringReaderAdapter) Read(p []byte) (n int, err error) {
    if a.reader == nil {
        data := a.provider()
        a.reader = strings.NewReader(data)
    }
    return a.reader.Read(p)
}

// 客户端代码期望 io.Reader
func ProcessData(reader io.Reader) {
    buf := make([]byte, 1024)
    n, _ := reader.Read(buf)
    fmt.Printf("Read %d bytes: %s\n", n, string(buf[:n]))
}

func main() {
    // 将函数适配为 io.Reader
    provider := func() string {
        return "Hello from StringProvider!"
    }
    
    adapter := NewStringReaderAdapter(provider)
    ProcessData(adapter)
}
```


### 4. 双向适配器

```go
package main

import "fmt"

// 接口 A
type PrinterA interface {
    PrintA(message string)
}

// 接口 B
type PrinterB interface {
    PrintB(text string)
}

// 实现 A
type ConcreteA struct{}

func (a *ConcreteA) PrintA(message string) {
    fmt.Printf("PrinterA: %s\n", message)
}

// 实现 B
type ConcreteB struct{}

func (b *ConcreteB) PrintB(text string) {
    fmt.Printf("PrinterB: %s\n", text)
}

// 双向适配器
type BidirectionalAdapter struct {
    a PrinterA
    b PrinterB
}

func NewBidirectionalAdapter(a PrinterA, b PrinterB) *BidirectionalAdapter {
    return &BidirectionalAdapter{a: a, b: b}
}

// 实现接口 A
func (adapter *BidirectionalAdapter) PrintA(message string) {
    adapter.a.PrintA(message)
}

// 实现接口 B
func (adapter *BidirectionalAdapter) PrintB(text string) {
    adapter.b.PrintB(text)
}

// 也可以互相转换
func (adapter *BidirectionalAdapter) AdaptAtoB(message string) {
    adapter.b.PrintB(message)
}

func (adapter *BidirectionalAdapter) AdaptBtoA(text string) {
    adapter.a.PrintA(text)
}

func main() {
    a := &ConcreteA{}
    b := &ConcreteB{}
    adapter := NewBidirectionalAdapter(a, b)
    
    // 使用 A 接口
    var printerA PrinterA = adapter
    printerA.PrintA("Hello from A")
    
    // 使用 B 接口
    var printerB PrinterB = adapter
    printerB.PrintB("Hello from B")
    
    // 互相转换
    adapter.AdaptAtoB("A to B")
    adapter.AdaptBtoA("B to A")
}
```

## 优缺点分析

### 优点

1. **解耦客户端和被适配者**
   - 客户端不直接依赖被适配者
   - 易于替换实现

2. **提高代码复用性**
   - 复用现有类
   - 不需要修改原有代码

3. **符合开闭原则**
   - 添加新适配器不影响现有代码
   - 易于扩展

4. **统一接口**
   - 提供一致的 API
   - 简化客户端代码

5. **灵活性高**
   - 可以适配多个被适配者
   - 可以添加额外功能

### 缺点

1. **增加系统复杂度**
   - 引入额外的类
   - 增加代码量

2. **可能影响性能**
   - 额外的间接调用
   - 数据转换开销

3. **过度使用会降低可读性**
   - 调用链路变长
   - 难以追踪


## Go 语言实现要点

### 1. 使用接口组合

```go
// ✅ Go 推荐：使用接口组合
type Target interface {
    Request() string
}

type Adaptee struct{}

func (a *Adaptee) SpecificRequest() string {
    return "Specific request"
}

type Adapter struct {
    *Adaptee
}

func (a *Adapter) Request() string {
    return a.SpecificRequest()
}

// ❌ 不推荐：类继承（Go 不支持）
```

### 2. 使用嵌入实现适配

```go
// 嵌入被适配者
type Adapter struct {
    adaptee Adaptee
}

// 或者嵌入指针
type Adapter struct {
    adaptee *Adaptee
}

// 实现目标接口
func (a *Adapter) Request() string {
    // 转换调用
    return a.adaptee.SpecificRequest()
}
```

### 3. 使用类型断言进行适配

```go
// 通用适配器
type UniversalAdapter struct {
    adaptee interface{}
}

func (a *UniversalAdapter) Request() string {
    // 根据类型进行适配
    switch v := a.adaptee.(type) {
    case *AdapteeA:
        return v.MethodA()
    case *AdapteeB:
        return v.MethodB()
    default:
        return "Unknown type"
    }
}
```

### 4. 使用函数适配

```go
// 函数类型适配
type HandlerFunc func(string) string

// 适配为接口
type HandlerAdapter struct {
    fn HandlerFunc
}

func (h *HandlerAdapter) Handle(input string) string {
    return h.fn(input)
}

// 使用
adapter := &HandlerAdapter{
    fn: func(s string) string {
        return "Processed: " + s
    },
}
```

## 实际应用

### 1. 数据库驱动适配器

```go
package main

import (
    "database/sql"
    "fmt"
)

// 统一的数据库接口
type Database interface {
    Connect(dsn string) error
    Query(sql string) ([]map[string]interface{}, error)
    Close() error
}

// MySQL 适配器
type MySQLAdapter struct {
    db *sql.DB
}

func NewMySQLAdapter() *MySQLAdapter {
    return &MySQLAdapter{}
}

func (m *MySQLAdapter) Connect(dsn string) error {
    db, err := sql.Open("mysql", dsn)
    if err != nil {
        return err
    }
    m.db = db
    return nil
}

func (m *MySQLAdapter) Query(query string) ([]map[string]interface{}, error) {
    rows, err := m.db.Query(query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    // 转换为统一格式
    columns, _ := rows.Columns()
    results := make([]map[string]interface{}, 0)
    
    for rows.Next() {
        values := make([]interface{}, len(columns))
        valuePtrs := make([]interface{}, len(columns))
        for i := range values {
            valuePtrs[i] = &values[i]
        }
        
        rows.Scan(valuePtrs...)
        
        row := make(map[string]interface{})
        for i, col := range columns {
            row[col] = values[i]
        }
        results = append(results, row)
    }
    
    return results, nil
}

func (m *MySQLAdapter) Close() error {
    return m.db.Close()
}

// PostgreSQL 适配器
type PostgreSQLAdapter struct {
    db *sql.DB
}

func NewPostgreSQLAdapter() *PostgreSQLAdapter {
    return &PostgreSQLAdapter{}
}

func (p *PostgreSQLAdapter) Connect(dsn string) error {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return err
    }
    p.db = db
    return nil
}

func (p *PostgreSQLAdapter) Query(query string) ([]map[string]interface{}, error) {
    // 类似 MySQL 的实现
    rows, err := p.db.Query(query)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    
    columns, _ := rows.Columns()
    results := make([]map[string]interface{}, 0)
    
    for rows.Next() {
        values := make([]interface{}, len(columns))
        valuePtrs := make([]interface{}, len(columns))
        for i := range values {
            valuePtrs[i] = &values[i]
        }
        
        rows.Scan(valuePtrs...)
        
        row := make(map[string]interface{})
        for i, col := range columns {
            row[col] = values[i]
        }
        results = append(results, row)
    }
    
    return results, nil
}

func (p *PostgreSQLAdapter) Close() error {
    return p.db.Close()
}

// 客户端代码使用统一接口
func ExecuteQuery(db Database, query string) {
    results, err := db.Query(query)
    if err != nil {
        fmt.Printf("Query error: %v\n", err)
        return
    }
    
    fmt.Printf("Results: %v\n", results)
}

func main() {
    // 使用 MySQL
    mysql := NewMySQLAdapter()
    mysql.Connect("user:pass@tcp(localhost:3306)/db")
    ExecuteQuery(mysql, "SELECT * FROM users")
    mysql.Close()
    
    // 使用 PostgreSQL
    postgres := NewPostgreSQLAdapter()
    postgres.Connect("postgres://user:pass@localhost/db")
    ExecuteQuery(postgres, "SELECT * FROM users")
    postgres.Close()
}
```


### 2. 第三方支付系统适配器

```go
package main

import (
    "fmt"
    "time"
)

// 统一的支付接口
type PaymentGateway interface {
    Pay(amount float64, currency string) (string, error)
    Refund(transactionID string, amount float64) error
    QueryStatus(transactionID string) (string, error)
}

// 支付宝 SDK（第三方库）
type AlipaySDK struct{}

func (a *AlipaySDK) SendPayment(money float64) string {
    return fmt.Sprintf("ALIPAY_%d", time.Now().Unix())
}

func (a *AlipaySDK) RefundPayment(orderID string, money float64) bool {
    return true
}

func (a *AlipaySDK) CheckOrder(orderID string) string {
    return "SUCCESS"
}

// 支付宝适配器
type AlipayAdapter struct {
    sdk *AlipaySDK
}

func NewAlipayAdapter() *AlipayAdapter {
    return &AlipayAdapter{sdk: &AlipaySDK{}}
}

func (a *AlipayAdapter) Pay(amount float64, currency string) (string, error) {
    // 货币转换（如果需要）
    if currency != "CNY" {
        return "", fmt.Errorf("alipay only supports CNY")
    }
    
    transactionID := a.sdk.SendPayment(amount)
    return transactionID, nil
}

func (a *AlipayAdapter) Refund(transactionID string, amount float64) error {
    success := a.sdk.RefundPayment(transactionID, amount)
    if !success {
        return fmt.Errorf("refund failed")
    }
    return nil
}

func (a *AlipayAdapter) QueryStatus(transactionID string) (string, error) {
    status := a.sdk.CheckOrder(transactionID)
    return status, nil
}

// 微信支付 SDK（第三方库）
type WeChatPaySDK struct{}

func (w *WeChatPaySDK) Pay(money float64) map[string]interface{} {
    return map[string]interface{}{
        "transaction_id": fmt.Sprintf("WECHAT_%d", time.Now().Unix()),
        "status":         "PAID",
    }
}

func (w *WeChatPaySDK) Refund(orderNo string, money float64) map[string]interface{} {
    return map[string]interface{}{
        "refund_id": fmt.Sprintf("REFUND_%d", time.Now().Unix()),
        "status":    "SUCCESS",
    }
}

func (w *WeChatPaySDK) Query(orderNo string) map[string]interface{} {
    return map[string]interface{}{
        "order_no": orderNo,
        "status":   "SUCCESS",
    }
}

// 微信支付适配器
type WeChatPayAdapter struct {
    sdk *WeChatPaySDK
}

func NewWeChatPayAdapter() *WeChatPayAdapter {
    return &WeChatPayAdapter{sdk: &WeChatPaySDK{}}
}

func (w *WeChatPayAdapter) Pay(amount float64, currency string) (string, error) {
    if currency != "CNY" {
        return "", fmt.Errorf("wechat pay only supports CNY")
    }
    
    result := w.sdk.Pay(amount)
    transactionID := result["transaction_id"].(string)
    return transactionID, nil
}

func (w *WeChatPayAdapter) Refund(transactionID string, amount float64) error {
    result := w.sdk.Refund(transactionID, amount)
    status := result["status"].(string)
    if status != "SUCCESS" {
        return fmt.Errorf("refund failed")
    }
    return nil
}

func (w *WeChatPayAdapter) QueryStatus(transactionID string) (string, error) {
    result := w.sdk.Query(transactionID)
    status := result["status"].(string)
    return status, nil
}

// PayPal SDK（第三方库）
type PayPalSDK struct{}

func (p *PayPalSDK) CreatePayment(value float64, curr string) string {
    return fmt.Sprintf("PAYPAL_%d", time.Now().Unix())
}

func (p *PayPalSDK) RefundTransaction(txnID string, value float64) bool {
    return true
}

func (p *PayPalSDK) GetTransactionStatus(txnID string) string {
    return "COMPLETED"
}

// PayPal 适配器
type PayPalAdapter struct {
    sdk *PayPalSDK
}

func NewPayPalAdapter() *PayPalAdapter {
    return &PayPalAdapter{sdk: &PayPalSDK{}}
}

func (p *PayPalAdapter) Pay(amount float64, currency string) (string, error) {
    transactionID := p.sdk.CreatePayment(amount, currency)
    return transactionID, nil
}

func (p *PayPalAdapter) Refund(transactionID string, amount float64) error {
    success := p.sdk.RefundTransaction(transactionID, amount)
    if !success {
        return fmt.Errorf("refund failed")
    }
    return nil
}

func (p *PayPalAdapter) QueryStatus(transactionID string) (string, error) {
    status := p.sdk.GetTransactionStatus(transactionID)
    return status, nil
}

// 支付服务（客户端代码）
type PaymentService struct {
    gateway PaymentGateway
}

func NewPaymentService(gateway PaymentGateway) *PaymentService {
    return &PaymentService{gateway: gateway}
}

func (s *PaymentService) ProcessPayment(amount float64, currency string) {
    fmt.Printf("\n=== Processing payment: %.2f %s ===\n", amount, currency)
    
    // 支付
    transactionID, err := s.gateway.Pay(amount, currency)
    if err != nil {
        fmt.Printf("Payment failed: %v\n", err)
        return
    }
    fmt.Printf("Payment successful, transaction ID: %s\n", transactionID)
    
    // 查询状态
    status, err := s.gateway.QueryStatus(transactionID)
    if err != nil {
        fmt.Printf("Query failed: %v\n", err)
        return
    }
    fmt.Printf("Payment status: %s\n", status)
    
    // 退款
    err = s.gateway.Refund(transactionID, amount)
    if err != nil {
        fmt.Printf("Refund failed: %v\n", err)
        return
    }
    fmt.Printf("Refund successful\n")
}

func main() {
    // 使用支付宝
    alipay := NewAlipayAdapter()
    service := NewPaymentService(alipay)
    service.ProcessPayment(100.00, "CNY")
    
    // 使用微信支付
    wechat := NewWeChatPayAdapter()
    service = NewPaymentService(wechat)
    service.ProcessPayment(200.00, "CNY")
    
    // 使用 PayPal
    paypal := NewPayPalAdapter()
    service = NewPaymentService(paypal)
    service.ProcessPayment(50.00, "USD")
}
```


### 3. 日志系统适配器

```go
package main

import (
    "fmt"
    "log"
    "os"
)

// 统一的日志接口
type Logger interface {
    Debug(message string)
    Info(message string)
    Warn(message string)
    Error(message string)
}

// 标准库 log 适配器
type StdLogAdapter struct {
    logger *log.Logger
}

func NewStdLogAdapter() *StdLogAdapter {
    return &StdLogAdapter{
        logger: log.New(os.Stdout, "", log.LstdFlags),
    }
}

func (s *StdLogAdapter) Debug(message string) {
    s.logger.Printf("[DEBUG] %s", message)
}

func (s *StdLogAdapter) Info(message string) {
    s.logger.Printf("[INFO] %s", message)
}

func (s *StdLogAdapter) Warn(message string) {
    s.logger.Printf("[WARN] %s", message)
}

func (s *StdLogAdapter) Error(message string) {
    s.logger.Printf("[ERROR] %s", message)
}

// 第三方日志库（如 logrus）适配器
type LogrusAdapter struct {
    // 假设这是 logrus 的实例
    logger interface {
        Debug(args ...interface{})
        Info(args ...interface{})
        Warn(args ...interface{})
        Error(args ...interface{})
    }
}

func NewLogrusAdapter(logger interface {
    Debug(args ...interface{})
    Info(args ...interface{})
    Warn(args ...interface{})
    Error(args ...interface{})
}) *LogrusAdapter {
    return &LogrusAdapter{logger: logger}
}

func (l *LogrusAdapter) Debug(message string) {
    l.logger.Debug(message)
}

func (l *LogrusAdapter) Info(message string) {
    l.logger.Info(message)
}

func (l *LogrusAdapter) Warn(message string) {
    l.logger.Warn(message)
}

func (l *LogrusAdapter) Error(message string) {
    l.logger.Error(message)
}

// 应用代码使用统一接口
type Application struct {
    logger Logger
}

func NewApplication(logger Logger) *Application {
    return &Application{logger: logger}
}

func (a *Application) Run() {
    a.logger.Debug("Application starting")
    a.logger.Info("Application initialized")
    a.logger.Warn("Low memory warning")
    a.logger.Error("Failed to connect to database")
}

func main() {
    // 使用标准库日志
    stdLogger := NewStdLogAdapter()
    app := NewApplication(stdLogger)
    app.Run()
}
```

### 4. 消息队列适配器

```go
package main

import (
    "fmt"
    "time"
)

// 统一的消息队列接口
type MessageQueue interface {
    Publish(topic string, message []byte) error
    Subscribe(topic string, handler func([]byte)) error
    Close() error
}

// RabbitMQ 适配器
type RabbitMQAdapter struct {
    // RabbitMQ 连接
    conn interface{}
}

func NewRabbitMQAdapter(url string) *RabbitMQAdapter {
    // 连接 RabbitMQ
    fmt.Printf("Connecting to RabbitMQ: %s\n", url)
    return &RabbitMQAdapter{}
}

func (r *RabbitMQAdapter) Publish(topic string, message []byte) error {
    fmt.Printf("[RabbitMQ] Publishing to %s: %s\n", topic, string(message))
    return nil
}

func (r *RabbitMQAdapter) Subscribe(topic string, handler func([]byte)) error {
    fmt.Printf("[RabbitMQ] Subscribing to %s\n", topic)
    // 模拟接收消息
    go func() {
        time.Sleep(time.Second)
        handler([]byte("Message from RabbitMQ"))
    }()
    return nil
}

func (r *RabbitMQAdapter) Close() error {
    fmt.Println("[RabbitMQ] Closing connection")
    return nil
}

// Kafka 适配器
type KafkaAdapter struct {
    // Kafka 连接
    producer interface{}
    consumer interface{}
}

func NewKafkaAdapter(brokers []string) *KafkaAdapter {
    fmt.Printf("Connecting to Kafka: %v\n", brokers)
    return &KafkaAdapter{}
}

func (k *KafkaAdapter) Publish(topic string, message []byte) error {
    fmt.Printf("[Kafka] Publishing to %s: %s\n", topic, string(message))
    return nil
}

func (k *KafkaAdapter) Subscribe(topic string, handler func([]byte)) error {
    fmt.Printf("[Kafka] Subscribing to %s\n", topic)
    // 模拟接收消息
    go func() {
        time.Sleep(time.Second)
        handler([]byte("Message from Kafka"))
    }()
    return nil
}

func (k *KafkaAdapter) Close() error {
    fmt.Println("[Kafka] Closing connection")
    return nil
}

// Redis Pub/Sub 适配器
type RedisPubSubAdapter struct {
    // Redis 连接
    client interface{}
}

func NewRedisPubSubAdapter(addr string) *RedisPubSubAdapter {
    fmt.Printf("Connecting to Redis: %s\n", addr)
    return &RedisPubSubAdapter{}
}

func (r *RedisPubSubAdapter) Publish(topic string, message []byte) error {
    fmt.Printf("[Redis] Publishing to %s: %s\n", topic, string(message))
    return nil
}

func (r *RedisPubSubAdapter) Subscribe(topic string, handler func([]byte)) error {
    fmt.Printf("[Redis] Subscribing to %s\n", topic)
    // 模拟接收消息
    go func() {
        time.Sleep(time.Second)
        handler([]byte("Message from Redis"))
    }()
    return nil
}

func (r *RedisPubSubAdapter) Close() error {
    fmt.Println("[Redis] Closing connection")
    return nil
}

// 消息服务（客户端代码）
type MessageService struct {
    mq MessageQueue
}

func NewMessageService(mq MessageQueue) *MessageService {
    return &MessageService{mq: mq}
}

func (s *MessageService) SendMessage(topic string, message string) {
    err := s.mq.Publish(topic, []byte(message))
    if err != nil {
        fmt.Printf("Failed to publish: %v\n", err)
    }
}

func (s *MessageService) ReceiveMessages(topic string) {
    s.mq.Subscribe(topic, func(message []byte) {
        fmt.Printf("Received: %s\n", string(message))
    })
}

func main() {
    fmt.Println("=== Using RabbitMQ ===")
    rabbitmq := NewRabbitMQAdapter("amqp://localhost")
    service := NewMessageService(rabbitmq)
    service.SendMessage("orders", "New order created")
    service.ReceiveMessages("orders")
    time.Sleep(2 * time.Second)
    rabbitmq.Close()
    
    fmt.Println("\n=== Using Kafka ===")
    kafka := NewKafkaAdapter([]string{"localhost:9092"})
    service = NewMessageService(kafka)
    service.SendMessage("orders", "New order created")
    service.ReceiveMessages("orders")
    time.Sleep(2 * time.Second)
    kafka.Close()
    
    fmt.Println("\n=== Using Redis ===")
    redis := NewRedisPubSubAdapter("localhost:6379")
    service = NewMessageService(redis)
    service.SendMessage("orders", "New order created")
    service.ReceiveMessages("orders")
    time.Sleep(2 * time.Second)
    redis.Close()
}
```


## 使用建议

### 何时使用

✅ **应该使用**:
- 需要集成第三方库
- 接口不兼容需要转换
- 统一多个实现的接口
- 遗留系统集成
- 不能修改原有代码

❌ **不应该使用**:
- 接口已经兼容
- 可以直接修改原有代码
- 系统简单不需要抽象
- 过度设计

### 最佳实践

1. **优先使用对象适配器（组合）**
   ```go
   // ✅ 推荐：使用组合
   type Adapter struct {
       adaptee *Adaptee
   }
   
   // ❌ 不推荐：Go 不支持继承
   ```

2. **保持适配器简单**
   ```go
   // ✅ 简单的适配器
   func (a *Adapter) Request() string {
       return a.adaptee.SpecificRequest()
   }
   
   // ❌ 复杂的适配器（考虑重构）
   func (a *Adapter) Request() string {
       // 大量的转换逻辑
       // 复杂的数据处理
       // ...
   }
   ```

3. **使用接口定义目标**
   ```go
   // ✅ 使用接口
   type Target interface {
       Request() string
   }
   
   // ❌ 使用具体类型
   type Target struct {
       // ...
   }
   ```

4. **适配器命名清晰**
   ```go
   // ✅ 清晰的命名
   type AlipayAdapter struct { }
   type MySQLAdapter struct { }
   
   // ❌ 不清晰的命名
   type Adapter1 struct { }
   type MyAdapter struct { }
   ```

5. **处理错误和边界情况**
   ```go
   func (a *Adapter) Request() (string, error) {
       if a.adaptee == nil {
           return "", errors.New("adaptee is nil")
       }
       
       result := a.adaptee.SpecificRequest()
       
       // 验证结果
       if result == "" {
           return "", errors.New("empty result")
       }
       
       return result, nil
   }
   ```

## 与其他模式的关系

### 适配器 vs 装饰器

| 特性 | 适配器模式 | 装饰器模式 |
|------|-----------|-----------|
| 目的 | 接口转换 | 功能增强 |
| 接口 | 改变接口 | 保持接口 |
| 层数 | 通常一层 | 可以多层 |
| 使用场景 | 接口不兼容 | 动态添加功能 |

```go
// 适配器：改变接口
type Adapter struct {
    adaptee *OldInterface
}
func (a *Adapter) NewMethod() { // 新接口
    a.adaptee.OldMethod() // 旧接口
}

// 装饰器：保持接口
type Decorator struct {
    component Component
}
func (d *Decorator) Method() { // 相同接口
    // 添加功能
    d.component.Method() // 相同接口
}
```

### 适配器 vs 代理

| 特性 | 适配器模式 | 代理模式 |
|------|-----------|---------|
| 目的 | 接口转换 | 访问控制 |
| 接口 | 不同接口 | 相同接口 |
| 关注点 | 兼容性 | 控制和优化 |

```go
// 适配器：不同接口
type Adapter struct {
    adaptee *Adaptee
}
func (a *Adapter) TargetMethod() {
    a.adaptee.AdapteeMethod()
}

// 代理：相同接口
type Proxy struct {
    subject Subject
}
func (p *Proxy) Request() {
    // 访问控制
    p.subject.Request()
}
```

### 适配器 vs 外观

| 特性 | 适配器模式 | 外观模式 |
|------|-----------|---------|
| 目的 | 接口转换 | 简化接口 |
| 对象数量 | 一对一 | 一对多 |
| 接口 | 转换接口 | 提供新接口 |

```go
// 适配器：一对一转换
type Adapter struct {
    adaptee *Adaptee
}

// 外观：一对多简化
type Facade struct {
    subsystem1 *Subsystem1
    subsystem2 *Subsystem2
    subsystem3 *Subsystem3
}
```

## 常见问题

### Q1: 什么时候使用适配器模式？

当你需要使用一个现有的类，但它的接口与你需要的不匹配时。特别是在集成第三方库或遗留系统时。

### Q2: 适配器会影响性能吗？

会有轻微的性能开销（额外的方法调用），但通常可以忽略不计。如果性能关键，可以考虑直接使用被适配者。

### Q3: 一个适配器可以适配多个被适配者吗？

可以，但要注意保持适配器的简单性。如果逻辑太复杂，考虑使用多个适配器。

### Q4: 适配器应该添加额外功能吗？

适配器应该专注于接口转换，不应该添加额外的业务逻辑。如果需要添加功能，考虑使用装饰器模式。

### Q5: Go 语言中如何实现类适配器？

Go 不支持继承，但可以使用嵌入（embedding）实现类似效果：

```go
type Adapter struct {
    *Adaptee  // 嵌入
}

func (a *Adapter) Request() string {
    return a.SpecificRequest()  // 直接调用嵌入类型的方法
}
```

## 总结

适配器模式是一个非常实用的结构型模式，特别适合集成第三方库和遗留系统。它通过接口转换实现了代码的复用和解耦。

**记住**:
- ✅ 优先使用对象适配器（组合）
- ✅ 保持适配器简单和专注
- ✅ 使用接口定义目标
- ✅ 适配器命名清晰
- ✅ 处理错误和边界情况
- ❌ 不要在适配器中添加业务逻辑
- ❌ 不要过度使用适配器

## 推荐阅读

### 书籍
- 《设计模式：可复用面向对象软件的基础》- GoF
- 《Head First 设计模式》
- 《Go 语言设计模式》

### 开源项目参考
- Go 标准库的 `io` 包（各种 Reader/Writer 适配器）
- `database/sql` 包（数据库驱动适配器）
- gRPC 的各种适配器实现

### 相关文章
- [Go by Example: Interfaces](https://gobyexample.com/interfaces)
- [Effective Go](https://golang.org/doc/effective_go)
- [Go Design Patterns](https://github.com/tmrts/go-patterns)
