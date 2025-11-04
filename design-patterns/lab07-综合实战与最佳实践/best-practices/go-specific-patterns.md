# Go 语言特有模式

## 概述

Go 语言虽然不是传统的面向对象语言，但它提供了独特的特性来实现设计模式。本文档介绍 Go 语言中的惯用模式和最佳实践，以及如何用 Go 的方式实现经典设计模式。

## Go 语言的特点

### 1. 接口的隐式实现

**特点**: 无需显式声明实现关系

**优势**:
- 更灵活的多态
- 解耦接口定义和实现
- 易于适配现有代码

**示例**:
```go
// 定义接口
type Writer interface {
    Write([]byte) (int, error)
}

// 任何实现了 Write 方法的类型都自动实现了 Writer 接口
type FileWriter struct{}

func (f *FileWriter) Write(data []byte) (int, error) {
    // 实现
    return len(data), nil
}

// 无需显式声明 FileWriter implements Writer
var w Writer = &FileWriter{}
```

### 2. 组合优于继承

**特点**: 使用嵌入实现代码复用

**优势**:
- 避免继承层次过深
- 更灵活的组合
- 清晰的依赖关系

**示例**:
```go
// 基础类型
type Logger struct{}

func (l *Logger) Log(msg string) {
    fmt.Println(msg)
}

// 通过嵌入复用
type Service struct {
    Logger  // 嵌入 Logger
    name string
}

// Service 自动拥有 Log 方法
service := &Service{name: "UserService"}
service.Log("Service started")
```

### 3. 函数作为一等公民

**特点**: 函数可以作为参数、返回值和变量

**优势**:
- 简化策略模式
- 实现高阶函数
- 灵活的回调机制

**示例**:
```go
// 函数类型
type Handler func(ctx *Context) error

// 函数作为参数
func Use(handler Handler) {
    // 使用 handler
}

// 函数作为返回值
func Middleware(next Handler) Handler {
    return func(ctx *Context) error {
        // 前置处理
        err := next(ctx)
        // 后置处理
        return err
    }
}
```

### 4. 并发原语

**特点**: goroutine 和 channel

**优势**:
- 轻量级并发
- 通过通信共享内存
- 简化并发编程

**示例**:
```go
// 使用 channel 实现观察者模式
type EventBus struct {
    subscribers map[string][]chan Event
}

func (b *EventBus) Subscribe(eventType string) <-chan Event {
    ch := make(chan Event)
    b.subscribers[eventType] = append(b.subscribers[eventType], ch)
    return ch
}

func (b *EventBus) Publish(event Event) {
    for _, ch := range b.subscribers[event.Type()] {
        go func(c chan Event) {
            c <- event
        }(ch)
    }
}
```

## Go 语言惯用模式

### 1. Functional Options 模式

**用途**: 优雅地处理可选参数

**传统方式的问题**:
```go
// 参数过多
func NewServer(addr string, port int, timeout time.Duration, maxConn int) *Server

// 使用配置结构体，但需要设置默认值
func NewServer(config ServerConfig) *Server
```

**Functional Options 方式**:
```go
type Server struct {
    addr    string
    port    int
    timeout time.Duration
    maxConn int
}

type Option func(*Server)

func WithPort(port int) Option {
    return func(s *Server) {
        s.port = port
    }
}

func WithTimeout(timeout time.Duration) Option {
    return func(s *Server) {
        s.timeout = timeout
    }
}

func NewServer(addr string, opts ...Option) *Server {
    s := &Server{
        addr:    addr,
        port:    8080,      // 默认值
        timeout: 30 * time.Second,
        maxConn: 100,
    }
    
    for _, opt := range opts {
        opt(s)
    }
    
    return s
}

// 使用
server := NewServer("localhost",
    WithPort(9090),
    WithTimeout(60*time.Second),
)
```

**优点**:
- 清晰的默认值
- 灵活的参数组合
- 易于扩展
- 向后兼容

### 2. 接口适配模式

**用途**: 将函数适配为接口

**示例**:
```go
// 接口
type Handler interface {
    Handle(ctx *Context) error
}

// 函数类型
type HandlerFunc func(ctx *Context) error

// 适配器方法
func (f HandlerFunc) Handle(ctx *Context) error {
    return f(ctx)
}

// 使用
var handler Handler = HandlerFunc(func(ctx *Context) error {
    // 处理逻辑
    return nil
})
```

**应用**: http.HandlerFunc, sort.Interface

### 3. 错误处理模式

**用途**: 优雅地处理错误

**基本模式**:
```go
func DoSomething() error {
    if err := step1(); err != nil {
        return fmt.Errorf("step1 failed: %w", err)
    }
    
    if err := step2(); err != nil {
        return fmt.Errorf("step2 failed: %w", err)
    }
    
    return nil
}
```

**错误包装**:
```go
// 自定义错误类型
type ValidationError struct {
    Field string
    Err   error
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation failed for %s: %v", e.Field, e.Err)
}

func (e *ValidationError) Unwrap() error {
    return e.Err
}

// 使用
if err := validate(data); err != nil {
    return &ValidationError{Field: "email", Err: err}
}
```

**错误检查**:
```go
// 使用 errors.Is 和 errors.As
if errors.Is(err, ErrNotFound) {
    // 处理未找到错误
}

var validationErr *ValidationError
if errors.As(err, &validationErr) {
    // 处理验证错误
}
```

### 4. Context 模式

**用途**: 传递请求范围的值、取消信号和截止时间

**基本用法**:
```go
func DoWork(ctx context.Context) error {
    // 检查取消
    select {
    case <-ctx.Done():
        return ctx.Err()
    default:
    }
    
    // 获取值
    userID := ctx.Value("userID").(string)
    
    // 执行工作
    return nil
}

// 使用
ctx := context.Background()
ctx = context.WithValue(ctx, "userID", "123")
ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
defer cancel()

DoWork(ctx)
```

**最佳实践**:
- Context 作为第一个参数
- 不要存储 Context
- 不要传递 nil Context
- 只用于请求范围的值

### 5. 并发模式

#### Worker Pool 模式

**用途**: 限制并发数量

```go
func WorkerPool(jobs <-chan Job, results chan<- Result, numWorkers int) {
    var wg sync.WaitGroup
    
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for job := range jobs {
                result := process(job)
                results <- result
            }
        }()
    }
    
    wg.Wait()
    close(results)
}
```

#### Pipeline 模式

**用途**: 数据流处理

```go
func Generator(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        for _, n := range nums {
            out <- n
        }
        close(out)
    }()
    return out
}

func Square(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            out <- n * n
        }
        close(out)
    }()
    return out
}

// 使用
nums := Generator(1, 2, 3, 4)
squares := Square(nums)
for s := range squares {
    fmt.Println(s)
}
```

#### Fan-Out/Fan-In 模式

**用途**: 并行处理和结果合并

```go
func FanOut(in <-chan int, n int) []<-chan int {
    outs := make([]<-chan int, n)
    for i := 0; i < n; i++ {
        outs[i] = process(in)
    }
    return outs
}

func FanIn(channels ...<-chan int) <-chan int {
    out := make(chan int)
    var wg sync.WaitGroup
    
    for _, ch := range channels {
        wg.Add(1)
        go func(c <-chan int) {
            defer wg.Done()
            for n := range c {
                out <- n
            }
        }(ch)
    }
    
    go func() {
        wg.Wait()
        close(out)
    }()
    
    return out
}
```

### 6. 资源管理模式

#### defer 模式

**用途**: 确保资源释放

```go
func ReadFile(filename string) ([]byte, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, err
    }
    defer file.Close()  // 确保文件关闭
    
    return io.ReadAll(file)
}
```

#### sync.Pool 模式

**用途**: 对象复用

```go
var bufferPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

func ProcessData(data []byte) {
    buf := bufferPool.Get().(*bytes.Buffer)
    defer bufferPool.Put(buf)
    
    buf.Reset()
    buf.Write(data)
    // 处理数据
}
```

## Go 实现经典模式的特色

### 1. 单例模式

**Go 方式**: 使用 sync.Once

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

**优点**:
- 线程安全
- 延迟初始化
- 简洁明了

### 2. 工厂模式

**Go 方式**: 使用函数和接口

```go
type Service interface {
    Do() error
}

func NewService(serviceType string) Service {
    switch serviceType {
    case "a":
        return &ServiceA{}
    case "b":
        return &ServiceB{}
    default:
        return &ServiceA{}
    }
}
```

**优点**:
- 简单直接
- 无需复杂的类层次
- 利用接口的灵活性

### 3. 策略模式

**Go 方式**: 使用函数类型

```go
// 策略类型
type Strategy func(data []int) []int

// 具体策略
var (
    QuickSort Strategy = func(data []int) []int {
        // 快速排序
        return data
    }
    
    BubbleSort Strategy = func(data []int) []int {
        // 冒泡排序
        return data
    }
)

// 使用
func Sort(data []int, strategy Strategy) []int {
    return strategy(data)
}
```

**优点**:
- 无需定义接口
- 更简洁
- 易于使用

### 4. 装饰器模式

**Go 方式**: 使用函数包装

```go
type Handler func(ctx *Context) error

func LoggingDecorator(next Handler) Handler {
    return func(ctx *Context) error {
        log.Println("Before")
        err := next(ctx)
        log.Println("After")
        return err
    }
}

func AuthDecorator(next Handler) Handler {
    return func(ctx *Context) error {
        // 认证逻辑
        return next(ctx)
    }
}

// 使用
handler := LoggingDecorator(AuthDecorator(businessHandler))
```

**优点**:
- 函数式风格
- 易于组合
- 清晰的调用链

## 最佳实践

### 1. 接口设计

**原则**:
- 接口应该小而专注
- 优先使用标准库接口
- 接受接口，返回具体类型

**示例**:
```go
// 好的接口设计
type Reader interface {
    Read(p []byte) (n int, err error)
}

// 接受接口
func Process(r io.Reader) error {
    // 处理
}

// 返回具体类型
func NewService() *Service {
    return &Service{}
}
```

### 2. 错误处理

**原则**:
- 总是检查错误
- 提供上下文信息
- 使用错误包装

**示例**:
```go
func DoWork() error {
    if err := step1(); err != nil {
        return fmt.Errorf("step1 failed: %w", err)
    }
    return nil
}
```

### 3. 并发安全

**原则**:
- 使用 mutex 保护共享状态
- 优先使用 channel 通信
- 避免数据竞争

**示例**:
```go
type SafeCounter struct {
    mu    sync.Mutex
    count int
}

func (c *SafeCounter) Inc() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.count++
}
```

### 4. 资源管理

**原则**:
- 使用 defer 确保清理
- 及时释放资源
- 避免资源泄漏

**示例**:
```go
func Process() error {
    file, err := os.Open("file.txt")
    if err != nil {
        return err
    }
    defer file.Close()
    
    // 处理文件
    return nil
}
```

## 常见陷阱

### 1. 过度使用接口

**问题**: 为每个类型定义接口

**解决方案**: 只在需要多态时使用接口

### 2. 忽略错误

**问题**: 不检查错误返回值

**解决方案**: 总是检查并处理错误

### 3. goroutine 泄漏

**问题**: goroutine 无法退出

**解决方案**: 使用 context 控制生命周期

### 4. 不当的锁使用

**问题**: 死锁或数据竞争

**解决方案**: 仔细设计锁的粒度和顺序

## 总结

Go 语言的设计模式实现有其独特之处：

1. **接口隐式实现**: 更灵活的多态
2. **组合优于继承**: 更清晰的代码结构
3. **函数是一等公民**: 简化某些模式
4. **并发原语**: 提供新的设计思路
5. **简洁性**: 避免过度设计

记住：**在 Go 中，简单和清晰比复杂的模式更重要。选择最简单有效的解决方案。**
