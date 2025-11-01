# HTTP 中间件系统

## 项目背景

在现代 Web 应用开发中，中间件是一种强大的设计模式，用于在请求到达业务处理器之前或之后执行通用的处理逻辑。本项目实现了一个完整的 HTTP 中间件系统，展示了装饰器模式、责任链模式和代理模式在实际应用中的综合运用。

中间件系统可以用于：
- 日志记录：记录每个请求的详细信息
- 身份认证：验证用户身份和权限
- 限流控制：防止服务过载
- 错误恢复：捕获 panic 并优雅处理
- CORS 处理：支持跨域请求
- 请求追踪：为每个请求分配唯一 ID
- 性能监控：统计请求处理时间

## 功能列表

- [x] 日志中间件：记录请求和响应信息
- [x] 认证中间件：支持 Token 和 Basic Auth
- [x] 限流中间件：基于 IP 的请求频率限制
- [x] 恢复中间件：捕获 panic 并返回友好错误
- [x] CORS 中间件：处理跨域请求
- [x] 追踪中间件：为每个请求分配唯一 ID
- [x] 压缩中间件：支持 gzip 响应压缩
- [x] 超时中间件：设置请求处理超时
- [x] 中间件链：灵活组合多个中间件
- [x] 响应包装器：捕获响应状态码和大小

## 技术栈

- Go 1.21+
- 标准库 net/http
- 标准库 testing

## 设计模式应用

| 模式 | 应用位置 | 作用 |
|------|----------|------|
| 装饰器模式 | middleware.go:15-200 | 动态为 HTTP 处理器添加功能 |
| 责任链模式 | middleware.go:205-215 | 按顺序执行多个中间件 |
| 代理模式 | middleware.go:220-280 | ResponseWriter 包装器拦截响应 |

## 项目结构

```
http-middleware-system/
├── README.md              # 项目说明文档
├── middleware.go          # 中间件核心实现
├── middleware_test.go     # 单元测试
└── main.go               # 示例应用
```

## 核心概念

### 1. 中间件类型定义

```go
type Middleware func(http.Handler) http.Handler
```

中间件是一个函数，接收一个 HTTP 处理器，返回一个装饰后的处理器。

### 2. 中间件链

```go
func Chain(handler http.Handler, middlewares ...Middleware) http.Handler
```

将多个中间件按顺序组合成一个处理链。

### 3. 响应包装器

```go
type ResponseWriter struct {
    http.ResponseWriter
    StatusCode int
    BytesWritten int
}
```

包装标准的 ResponseWriter，用于捕获响应状态码和大小。

## 运行方式

### 运行示例应用

```bash
# 进入项目目录
cd design-patterns/lab03-结构型模式-适配器装饰器代理/project/http-middleware-system

# 运行主程序
go run main.go middleware.go

# 在另一个终端测试
curl -H "Authorization: Bearer secret-token" http://localhost:8080/api/users
curl -H "Authorization: Bearer secret-token" http://localhost:8080/api/products
curl http://localhost:8080/health
```

### 运行测试

```bash
# 运行所有测试
go test -v

# 运行测试并显示覆盖率
go test -v -cover

# 生成覆盖率报告
go test -coverprofile=coverage.out
go tool cover -html=coverage.out
```

## 预期输出

### 成功请求

```
2024/01/01 10:00:00 [TRACE] Request ID: req-1234567890
2024/01/01 10:00:00 [LOG] Started GET /api/users
2024/01/01 10:00:00 [AUTH] Token authentication successful
2024/01/01 10:00:00 [LOG] Completed GET /api/users - Status: 200 - Size: 45 bytes - Duration: 2.5ms
```

### 认证失败

```
2024/01/01 10:00:01 [TRACE] Request ID: req-1234567891
2024/01/01 10:00:01 [LOG] Started GET /api/users
2024/01/01 10:00:01 [AUTH] Authentication failed: missing or invalid token
2024/01/01 10:00:01 [LOG] Completed GET /api/users - Status: 401 - Size: 13 bytes - Duration: 0.5ms
```

### 限流触发

```
2024/01/01 10:00:02 [RATE_LIMIT] Request count for 127.0.0.1: 11/10
2024/01/01 10:00:02 [RATE_LIMIT] Rate limit exceeded for 127.0.0.1
```

## 中间件使用示例

### 基础使用

```go
// 单个中间件
handler := LoggingMiddleware()(http.HandlerFunc(YourHandler))

// 多个中间件链
handler := Chain(
    http.HandlerFunc(YourHandler),
    RecoveryMiddleware(),
    LoggingMiddleware(),
    AuthMiddleware("secret-token"),
)
```

### 自定义中间件

```go
func CustomMiddleware() Middleware {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            // 前置处理
            log.Println("Before handler")
            
            // 调用下一个处理器
            next.ServeHTTP(w, r)
            
            // 后置处理
            log.Println("After handler")
        })
    }
}
```

### 条件中间件

```go
func ConditionalMiddleware(condition bool, middleware Middleware) Middleware {
    return func(next http.Handler) http.Handler {
        if condition {
            return middleware(next)
        }
        return next
    }
}
```

## 扩展建议

### 1. 缓存中间件

实现基于内存或 Redis 的响应缓存：

```go
func CacheMiddleware(ttl time.Duration) Middleware {
    cache := make(map[string]cachedResponse)
    // 实现缓存逻辑
}
```

### 2. 指标收集中间件

集成 Prometheus 收集请求指标：

```go
func MetricsMiddleware(registry *prometheus.Registry) Middleware {
    // 实现指标收集
}
```

### 3. 请求体验证中间件

验证请求体的格式和内容：

```go
func ValidationMiddleware(schema interface{}) Middleware {
    // 实现请求验证
}
```

### 4. 会话管理中间件

实现用户会话管理：

```go
func SessionMiddleware(store SessionStore) Middleware {
    // 实现会话管理
}
```

### 5. API 版本控制中间件

根据请求头路由到不同版本的处理器：

```go
func VersionMiddleware(versions map[string]http.Handler) Middleware {
    // 实现版本控制
}
```

### 6. 内容协商中间件

根据 Accept 头返回不同格式的响应：

```go
func ContentNegotiationMiddleware() Middleware {
    // 实现内容协商
}
```

## 最佳实践

### 1. 中间件顺序

中间件的执行顺序很重要，推荐顺序：

1. Recovery（最外层，捕获所有 panic）
2. Logging（记录所有请求）
3. Tracing（请求追踪）
4. CORS（跨域处理）
5. Authentication（身份认证）
6. RateLimit（限流）
7. Timeout（超时控制）
8. Compression（响应压缩）
9. 业务处理器

### 2. 错误处理

中间件应该优雅地处理错误，避免影响其他中间件：

```go
defer func() {
    if err := recover(); err != nil {
        log.Printf("Middleware panic: %v", err)
        // 返回错误响应
    }
}()
```

### 3. 性能考虑

- 避免在中间件中执行耗时操作
- 使用连接池管理外部资源
- 合理设置缓存和超时时间
- 使用 sync.Pool 复用对象

### 4. 可测试性

- 每个中间件应该是独立可测试的
- 使用 httptest 包进行测试
- 模拟外部依赖

### 5. 配置管理

使用配置结构体而不是全局变量：

```go
type Config struct {
    LogLevel    string
    RateLimit   int
    AuthToken   string
}

func NewMiddleware(config Config) Middleware {
    // 使用配置创建中间件
}
```

## 常见问题

### Q1: 中间件的执行顺序是什么？

A: 中间件按照 Chain 函数中的顺序从左到右包装，但执行时是从外到内。例如：

```go
Chain(handler, A, B, C)
// 执行顺序: A -> B -> C -> handler -> C -> B -> A
```

### Q2: 如何在中间件之间传递数据？

A: 使用 context.Context：

```go
ctx := context.WithValue(r.Context(), "user", user)
r = r.WithContext(ctx)
next.ServeHTTP(w, r)
```

### Q3: 如何跳过某些路径的中间件？

A: 在中间件内部检查路径：

```go
if r.URL.Path == "/health" {
    next.ServeHTTP(w, r)
    return
}
```

### Q4: 如何处理中间件中的异步操作？

A: 使用 goroutine 和 channel，但要注意不要在异步操作中写入 ResponseWriter。

## 参考资料

- [Go HTTP Middleware Pattern](https://www.alexedwards.net/blog/making-and-using-middleware)
- [Gorilla Handlers](https://github.com/gorilla/handlers)
- [Negroni Middleware](https://github.com/urfave/negroni)
- [Chi Router Middleware](https://github.com/go-chi/chi)

## 总结

本项目展示了如何使用装饰器模式和责任链模式构建一个灵活、可扩展的 HTTP 中间件系统。通过组合不同的中间件，可以轻松实现复杂的请求处理逻辑，同时保持代码的清晰和可维护性。

中间件模式的核心优势：
- **关注点分离**：每个中间件专注于单一职责
- **可复用性**：中间件可以在不同的路由中复用
- **可组合性**：灵活组合多个中间件
- **可测试性**：每个中间件独立测试
- **可扩展性**：易于添加新的中间件
