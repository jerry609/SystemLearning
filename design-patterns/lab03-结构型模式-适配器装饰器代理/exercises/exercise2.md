# 练习 2: 实现 HTTP 请求装饰器链

## 难度
⭐⭐⭐ (困难)

## 学习目标
- 掌握装饰器模式的实现
- 理解中间件的设计思想
- 学会动态添加功能
- 实践责任链模式的应用

## 问题描述

实现一个灵活的 HTTP 客户端装饰器系统，支持为 HTTP 请求动态添加各种功能，如日志记录、重试机制、超时控制、缓存、认证等。装饰器应该可以任意组合，并且不影响原有的 HTTP 客户端功能。

## 功能要求

1. **基础 HTTP 客户端接口**
   - 定义统一的 `HTTPClient` 接口
   - 支持 GET、POST、PUT、DELETE 等方法
   - 返回标准的响应结构

2. **日志装饰器**
   - 记录请求的 URL、方法、耗时
   - 记录响应状态码
   - 支持自定义日志格式

3. **重试装饰器**
   - 支持配置重试次数
   - 支持配置重试间隔
   - 支持指定哪些状态码需要重试
   - 使用指数退避策略

4. **超时装饰器**
   - 支持设置请求超时时间
   - 超时后自动取消请求
   - 返回超时错误

5. **缓存装饰器**
   - 缓存 GET 请求的响应
   - 支持设置缓存过期时间
   - 支持缓存键的自定义
   - 支持缓存失效策略

6. **认证装饰器**
   - 自动添加认证头
   - 支持 Bearer Token
   - 支持 Basic Auth
   - 支持自定义认证方式

7. **限流装饰器**
   - 限制请求频率
   - 使用令牌桶算法
   - 支持配置 QPS

## 输入输出示例

### 示例 1: 基本装饰器组合
**代码**:
```go
// 创建基础客户端
baseClient := NewBaseHTTPClient()

// 添加日志装饰器
loggedClient := NewLoggingDecorator(baseClient)

// 添加重试装饰器
retriedClient := NewRetryDecorator(loggedClient, RetryConfig{
    MaxRetries: 3,
    RetryDelay: time.Second,
})

// 添加超时装饰器
client := NewTimeoutDecorator(retriedClient, 5*time.Second)

// 发送请求
resp, err := client.Get("https://api.example.com/users")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Status: %d\n", resp.StatusCode)
```

**输出**:
```
[2024-01-15 10:30:00] GET https://api.example.com/users
[2024-01-15 10:30:00] Attempt 1 failed: connection timeout
[2024-01-15 10:30:01] Attempt 2 failed: connection timeout
[2024-01-15 10:30:03] Attempt 3 succeeded
[2024-01-15 10:30:03] Response: 200 OK (Duration: 3.2s)
Status: 200
```

### 示例 2: 带缓存和认证
**代码**:
```go
// 创建完整的客户端链
client := NewBaseHTTPClient()
client = NewAuthDecorator(client, "Bearer", "your-token-here")
client = NewCacheDecorator(client, CacheConfig{
    TTL:      5 * time.Minute,
    MaxSize:  100,
})
client = NewLoggingDecorator(client)

// 第一次请求（从服务器获取）
resp1, _ := client.Get("https://api.example.com/users/123")
fmt.Println("First request:", resp1.StatusCode)

// 第二次请求（从缓存获取）
resp2, _ := client.Get("https://api.example.com/users/123")
fmt.Println("Second request:", resp2.StatusCode)
```

**输出**:
```
[2024-01-15 10:30:00] GET https://api.example.com/users/123
[2024-01-15 10:30:00] Cache miss, fetching from server
[2024-01-15 10:30:01] Response: 200 OK (Duration: 1.2s)
First request: 200

[2024-01-15 10:30:02] GET https://api.example.com/users/123
[2024-01-15 10:30:02] Cache hit
[2024-01-15 10:30:02] Response: 200 OK (Duration: 0.001s)
Second request: 200
```

### 示例 3: 限流装饰器
**代码**:
```go
// 创建限流客户端（每秒最多 2 个请求）
client := NewBaseHTTPClient()
client = NewRateLimitDecorator(client, RateLimitConfig{
    RequestsPerSecond: 2,
})
client = NewLoggingDecorator(client)

// 快速发送 5 个请求
for i := 0; i < 5; i++ {
    start := time.Now()
    resp, _ := client.Get(fmt.Sprintf("https://api.example.com/item/%d", i))
    fmt.Printf("Request %d: %d (waited: %v)\n", i, resp.StatusCode, time.Since(start))
}
```

**输出**:
```
[2024-01-15 10:30:00.000] GET https://api.example.com/item/0
Request 0: 200 (waited: 100ms)

[2024-01-15 10:30:00.100] GET https://api.example.com/item/1
Request 1: 200 (waited: 100ms)

[2024-01-15 10:30:00.500] Rate limit: waiting 400ms
[2024-01-15 10:30:00.900] GET https://api.example.com/item/2
Request 2: 200 (waited: 500ms)

[2024-01-15 10:30:01.000] Rate limit: waiting 100ms
[2024-01-15 10:30:01.100] GET https://api.example.com/item/3
Request 3: 200 (waited: 200ms)

[2024-01-15 10:30:01.500] Rate limit: waiting 400ms
[2024-01-15 10:30:01.900] GET https://api.example.com/item/4
Request 4: 200 (waited: 500ms)
```

## 接口定义

```go
// HTTPClient 统一的 HTTP 客户端接口
type HTTPClient interface {
    Get(url string) (*Response, error)
    Post(url string, body []byte) (*Response, error)
    Put(url string, body []byte) (*Response, error)
    Delete(url string) (*Response, error)
}

// Response HTTP 响应结构
type Response struct {
    StatusCode int
    Headers    map[string]string
    Body       []byte
    Duration   time.Duration
}

// RetryConfig 重试配置
type RetryConfig struct {
    MaxRetries     int
    RetryDelay     time.Duration
    RetryableStatus []int // 需要重试的状态码
}

// CacheConfig 缓存配置
type CacheConfig struct {
    TTL     time.Duration
    MaxSize int
}

// RateLimitConfig 限流配置
type RateLimitConfig struct {
    RequestsPerSecond float64
}
```

## 提示

💡 **提示 1**: 装饰器基础结构
```go
type LoggingDecorator struct {
    client HTTPClient
    logger *log.Logger
}

func NewLoggingDecorator(client HTTPClient) *LoggingDecorator {
    return &LoggingDecorator{
        client: client,
        logger: log.New(os.Stdout, "[HTTP] ", log.LstdFlags),
    }
}

func (d *LoggingDecorator) Get(url string) (*Response, error) {
    start := time.Now()
    d.logger.Printf("GET %s", url)
    
    resp, err := d.client.Get(url)
    
    duration := time.Since(start)
    if err != nil {
        d.logger.Printf("Error: %v (Duration: %v)", err, duration)
        return nil, err
    }
    
    d.logger.Printf("Response: %d (Duration: %v)", resp.StatusCode, duration)
    return resp, nil
}
```

💡 **提示 2**: 重试装饰器实现
```go
type RetryDecorator struct {
    client HTTPClient
    config RetryConfig
}

func (d *RetryDecorator) Get(url string) (*Response, error) {
    var lastErr error
    
    for attempt := 0; attempt <= d.config.MaxRetries; attempt++ {
        if attempt > 0 {
            // 指数退避
            delay := d.config.RetryDelay * time.Duration(1<<uint(attempt-1))
            time.Sleep(delay)
        }
        
        resp, err := d.client.Get(url)
        if err == nil && !d.shouldRetry(resp.StatusCode) {
            return resp, nil
        }
        
        lastErr = err
    }
    
    return nil, fmt.Errorf("max retries exceeded: %w", lastErr)
}

func (d *RetryDecorator) shouldRetry(statusCode int) bool {
    for _, code := range d.config.RetryableStatus {
        if code == statusCode {
            return true
        }
    }
    return statusCode >= 500
}
```

💡 **提示 3**: 缓存装饰器实现
```go
type CacheDecorator struct {
    client HTTPClient
    cache  map[string]*cacheEntry
    mu     sync.RWMutex
    config CacheConfig
}

type cacheEntry struct {
    response  *Response
    expiresAt time.Time
}

func (d *CacheDecorator) Get(url string) (*Response, error) {
    // 检查缓存
    d.mu.RLock()
    if entry, ok := d.cache[url]; ok {
        if time.Now().Before(entry.expiresAt) {
            d.mu.RUnlock()
            return entry.response, nil
        }
    }
    d.mu.RUnlock()
    
    // 缓存未命中，发送请求
    resp, err := d.client.Get(url)
    if err != nil {
        return nil, err
    }
    
    // 存入缓存
    d.mu.Lock()
    d.cache[url] = &cacheEntry{
        response:  resp,
        expiresAt: time.Now().Add(d.config.TTL),
    }
    d.mu.Unlock()
    
    return resp, nil
}
```

💡 **提示 4**: 限流装饰器实现（令牌桶算法）
```go
type RateLimitDecorator struct {
    client  HTTPClient
    limiter *rate.Limiter
}

func NewRateLimitDecorator(client HTTPClient, config RateLimitConfig) *RateLimitDecorator {
    return &RateLimitDecorator{
        client:  client,
        limiter: rate.NewLimiter(rate.Limit(config.RequestsPerSecond), 1),
    }
}

func (d *RateLimitDecorator) Get(url string) (*Response, error) {
    // 等待令牌
    if err := d.limiter.Wait(context.Background()); err != nil {
        return nil, err
    }
    
    return d.client.Get(url)
}
```

## 评分标准

- [ ] **接口设计 (20%)**
  - 统一的 HTTPClient 接口
  - 清晰的配置结构
  - 合理的响应结构

- [ ] **装饰器实现 (40%)**
  - 实现至少 4 个装饰器
  - 装饰器可以任意组合
  - 保持接口一致性

- [ ] **功能完整性 (25%)**
  - 日志、重试、超时功能正确
  - 缓存功能正确
  - 限流功能正确

- [ ] **代码质量 (15%)**
  - 代码结构清晰
  - 线程安全
  - 适当的错误处理

## 扩展挑战

如果你完成了基本要求，可以尝试以下扩展功能：

1. **断路器装饰器**
   ```go
   type CircuitBreakerDecorator struct {
       client       HTTPClient
       failureCount int
       threshold    int
       state        CircuitState // Open, HalfOpen, Closed
   }
   ```

2. **指标收集装饰器**
   ```go
   type MetricsDecorator struct {
       client       HTTPClient
       totalRequests int64
       failedRequests int64
       avgDuration   time.Duration
   }
   ```

3. **请求去重装饰器**
   ```go
   type DeduplicationDecorator struct {
       client   HTTPClient
       inflight map[string]*sync.WaitGroup
   }
   ```

4. **压缩装饰器**
   ```go
   type CompressionDecorator struct {
       client HTTPClient
       method string // gzip, deflate
   }
   ```

## 依赖安装

本练习需要使用 `golang.org/x/time/rate` 包，请先安装：

```bash
go get golang.org/x/time/rate
```

或者在项目目录下初始化 Go 模块：

```bash
go mod init exercise2
go mod tidy
```

## 参考资源

- [装饰器模式详解](../theory/02-decorator.md)
- [Go context 包](https://pkg.go.dev/context)
- [Go rate 包](https://pkg.go.dev/golang.org/x/time/rate)

## 提交要求

1. 实现 `HTTPClient` 接口和基础客户端
2. 实现至少 4 个装饰器
3. 编写测试用例验证功能
4. 提供完整的使用示例
5. 添加必要的注释和文档

---

**预计完成时间**: 2-3 小时  
**难度评估**: 困难  
**重点考察**: 装饰器模式、中间件设计、并发安全
