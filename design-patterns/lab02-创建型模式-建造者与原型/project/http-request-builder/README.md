# HTTP 请求构建器实战项目

## 项目背景

在实际开发中，我们经常需要构建复杂的 HTTP 请求。直接使用 `http.NewRequest` 需要处理很多细节，代码冗长且容易出错。本项目实现一个优雅的 HTTP 请求构建器，使用建造者模式简化 HTTP 请求的创建过程。

## 功能列表

- [x] 支持所有 HTTP 方法（GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS）
- [x] 支持设置请求头（单个和批量）
- [x] 支持查询参数（单个和批量）
- [x] 支持多种请求体格式（JSON, Form, Raw）
- [x] 支持认证（Bearer Token, Basic Auth, Custom）
- [x] 支持超时设置
- [x] 支持重试机制
- [x] 链式调用 API
- [x] 完整的单元测试

## 技术栈

- Go 1.21+
- 标准库 `net/http`
- 标准库 `encoding/json`
- 标准库 `testing`

## 设计模式应用

| 模式 | 应用位置 | 作用 |
|------|----------|------|
| 建造者模式 | `RequestBuilder` | 使用链式调用构建复杂的 HTTP 请求 |
| Functional Options | `NewClient` | 配置 HTTP 客户端的可选参数 |

## 项目结构

```
http-request-builder/
├── README.md           # 项目说明（本文件）
├── builder.go          # 请求构建器核心代码
└── builder_test.go     # 单元测试
```

## 运行方式

### 运行示例

```bash
# 运行主程序
go run builder.go
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

## 使用示例

### 示例 1: 简单的 GET 请求

```go
client := NewHTTPClient()

req, err := client.NewRequest().
    GET().
    URL("https://api.example.com/users").
    Query("page", "1").
    Query("limit", "10").
    Build()

if err != nil {
    log.Fatal(err)
}

resp, err := client.Do(req)
```

### 示例 2: POST JSON 请求

```go
userData := map[string]interface{}{
    "name":  "John Doe",
    "email": "john@example.com",
    "age":   30,
}

req, err := client.NewRequest().
    POST().
    URL("https://api.example.com/users").
    BodyJSON(userData).
    BearerToken("your-token-here").
    Build()
```

### 示例 3: 带认证和自定义头的请求

```go
req, err := client.NewRequest().
    GET().
    URL("https://api.example.com/profile").
    BearerToken("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...").
    Header("X-Request-ID", "12345").
    Header("Accept", "application/json").
    Timeout(60 * time.Second).
    Build()
```

### 示例 4: 批量设置参数

```go
headers := map[string]string{
    "X-API-Key":    "api-key-123",
    "X-Request-ID": "req-456",
    "Accept":       "application/json",
}

queries := map[string]string{
    "sort":   "created_at",
    "order":  "desc",
    "filter": "active",
}

req, err := client.NewRequest().
    GET().
    URL("https://api.example.com/items").
    Headers(headers).
    Queries(queries).
    Build()
```

## 核心 API

### RequestBuilder 方法

- `GET()`, `POST()`, `PUT()`, `DELETE()`, `PATCH()` - 设置 HTTP 方法
- `URL(url string)` - 设置请求 URL
- `Header(key, value string)` - 添加请求头
- `Headers(headers map[string]string)` - 批量添加请求头
- `Query(key, value string)` - 添加查询参数
- `Queries(queries map[string]string)` - 批量添加查询参数
- `Body(body []byte)` - 设置原始请求体
- `BodyJSON(data interface{})` - 设置 JSON 请求体
- `BodyForm(data map[string]string)` - 设置表单请求体
- `BearerToken(token string)` - 设置 Bearer Token
- `BasicAuth(username, password string)` - 设置基本认证
- `Timeout(timeout time.Duration)` - 设置超时时间
- `Build()` - 构建最终的 HTTP 请求

### HTTPClient 方法

- `NewRequest()` - 创建新的请求构建器
- `Do(req *http.Request)` - 执行 HTTP 请求
- `Get(url string)` - 快捷 GET 请求
- `Post(url string, body interface{})` - 快捷 POST 请求

## 预期输出

运行 `go run builder.go` 将输出：

```
=== HTTP 请求构建器实战项目 ===

示例 1: 简单的 GET 请求
GET https://api.example.com/users?page=1&limit=10
Headers:
  Accept: application/json
Timeout: 30s

示例 2: POST JSON 请求
POST https://api.example.com/users
Headers:
  Authorization: Bearer your-token-here
  Content-Type: application/json
Body: {"age":30,"email":"john@example.com","name":"John Doe"}
Timeout: 30s

示例 3: 带认证和自定义头的请求
GET https://api.example.com/profile
Headers:
  Accept: application/json
  Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
  X-Request-ID: 12345
Timeout: 1m0s

=== 示例结束 ===

建造者模式的优势:
✅ 链式调用，代码优雅
✅ 参数清晰，易于理解
✅ 灵活配置，按需设置
✅ 易于扩展，添加新功能简单
```

## 扩展建议

1. **添加重试机制**
   ```go
   req, err := client.NewRequest().
       GET().
       URL("https://api.example.com/data").
       Retry(3, 1*time.Second).
       Build()
   ```

2. **添加请求拦截器**
   ```go
   client.AddInterceptor(func(req *http.Request) error {
       // 在发送请求前修改请求
       req.Header.Set("X-Timestamp", time.Now().String())
       return nil
   })
   ```

3. **添加响应处理器**
   ```go
   resp, err := client.NewRequest().
       GET().
       URL("https://api.example.com/users").
       Build().
       Execute().
       JSON(&users)
   ```

4. **添加请求缓存**
   ```go
   req, err := client.NewRequest().
       GET().
       URL("https://api.example.com/data").
       Cache(5 * time.Minute).
       Build()
   ```

5. **支持文件上传**
   ```go
   req, err := client.NewRequest().
       POST().
       URL("https://api.example.com/upload").
       File("file", "path/to/file.jpg").
       Build()
   ```

## 学习要点

1. **建造者模式的应用**
   - 如何使用链式调用构建复杂对象
   - 如何设计清晰的 API
   - 如何处理可选参数

2. **HTTP 请求的构建**
   - 如何设置请求方法、URL、头部
   - 如何处理查询参数和请求体
   - 如何实现认证

3. **代码组织**
   - 如何分离关注点
   - 如何编写可测试的代码
   - 如何提供友好的 API

## 常见问题

### Q1: 为什么使用建造者模式而不是直接使用 http.NewRequest？

建造者模式提供了更友好的 API，支持链式调用，代码更清晰易读。同时，它封装了常见的操作（如设置 JSON 请求体、认证等），减少重复代码。

### Q2: 如何处理请求错误？

在 `Build()` 方法中返回 error，调用者需要检查错误。对于执行请求的错误，在 `Do()` 方法中返回。

### Q3: 是否支持并发请求？

是的，每个 `RequestBuilder` 实例是独立的，可以安全地在多个 goroutine 中使用。

### Q4: 如何添加自定义的请求处理逻辑？

可以通过扩展 `HTTPClient` 添加拦截器或中间件功能。

## 参考资源

- [Go net/http 包文档](https://pkg.go.dev/net/http)
- [建造者模式详解](../../theory/01-builder.md)
- [HTTP 协议规范](https://developer.mozilla.org/en-US/docs/Web/HTTP)

---

**项目难度**: 中等  
**预计完成时间**: 2-3 小时  
**适合人群**: 有 Go 基础，想学习设计模式的开发者
