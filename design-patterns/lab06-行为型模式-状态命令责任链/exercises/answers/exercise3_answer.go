package main

// 练习 3: 实现请求处理中间件系统 - 参考答案
//
// 设计思路:
// 1. 使用责任链模式组织中间件
// 2. 每个中间件可以在调用下一个中间件前后执行逻辑
// 3. 中间件可以短路处理，中断链的传递
//
// 使用的设计模式: 责任链模式
// 模式应用位置: Middleware 接口及其实现类

import (
	"fmt"
	"strings"
	"time"
)

// Request HTTP 请求
type Request struct {
	Method    string
	Path      string
	Headers   map[string]string
	Body      string
	User      string
	StartTime time.Time
}

// Response HTTP 响应
type Response struct {
	StatusCode int
	Body       string
	Headers    map[string]string
}

// Middleware 中间件接口
type Middleware interface {
	SetNext(middleware Middleware) Middleware
	Handle(req *Request, res *Response) error
}

// BaseMiddleware 基础中间件
type BaseMiddleware struct {
	next Middleware
}

func (m *BaseMiddleware) SetNext(middleware Middleware) Middleware {
	m.next = middleware
	return middleware
}

func (m *BaseMiddleware) Handle(req *Request, res *Response) error {
	if m.next != nil {
		return m.next.Handle(req, res)
	}
	return nil
}

// LoggerMiddleware 日志中间件
type LoggerMiddleware struct {
	BaseMiddleware
}

func NewLoggerMiddleware() *LoggerMiddleware {
	return &LoggerMiddleware{}
}

func (m *LoggerMiddleware) Handle(req *Request, res *Response) error {
	req.StartTime = time.Now()
	fmt.Printf("[Logger] %s %s - 开始处理\n", req.Method, req.Path)
	
	err := m.BaseMiddleware.Handle(req, res)
	
	duration := time.Since(req.StartTime)
	fmt.Printf("[Logger] %s %s - 完成处理，耗时: %v，状态码: %d\n",
		req.Method, req.Path, duration, res.StatusCode)
	
	return err
}

// AuthMiddleware 认证中间件
type AuthMiddleware struct {
	BaseMiddleware
	validTokens map[string]string
}

func NewAuthMiddleware() *AuthMiddleware {
	return &AuthMiddleware{
		validTokens: map[string]string{
			"Bearer token123": "user1",
			"Bearer token456": "user2",
			"Bearer token789": "admin",
		},
	}
}

func (m *AuthMiddleware) Handle(req *Request, res *Response) error {
	fmt.Println("[Auth] 验证用户身份")
	
	token := req.Headers["Authorization"]
	if token == "" {
		fmt.Println("[Auth] ✗ 未提供认证令牌")
		res.StatusCode = 401
		res.Body = "Unauthorized: Missing token"
		return fmt.Errorf("missing authorization token")
	}
	
	user, ok := m.validTokens[token]
	if !ok {
		fmt.Println("[Auth] ✗ 无效的认证令牌")
		res.StatusCode = 401
		res.Body = "Unauthorized: Invalid token"
		return fmt.Errorf("invalid authorization token")
	}
	
	req.User = user
	fmt.Printf("[Auth] ✓ 用户认证成功: %s\n", user)
	
	return m.BaseMiddleware.Handle(req, res)
}

// RateLimitMiddleware 限流中间件
type RateLimitMiddleware struct {
	BaseMiddleware
	requestCounts map[string]int
	limit         int
}

func NewRateLimitMiddleware(limit int) *RateLimitMiddleware {
	return &RateLimitMiddleware{
		requestCounts: make(map[string]int),
		limit:         limit,
	}
}

func (m *RateLimitMiddleware) Handle(req *Request, res *Response) error {
	fmt.Println("[RateLimit] 检查请求频率")
	
	user := req.User
	if user == "" {
		user = "anonymous"
	}
	
	m.requestCounts[user]++
	count := m.requestCounts[user]
	
	fmt.Printf("[RateLimit] 用户 %s 的请求次数: %d/%d\n", user, count, m.limit)
	
	if count > m.limit {
		fmt.Printf("[RateLimit] ✗ 用户 %s 超过请求限制\n", user)
		res.StatusCode = 429
		res.Body = "Too Many Requests"
		return fmt.Errorf("rate limit exceeded")
	}
	
	fmt.Println("[RateLimit] ✓ 请求频率正常")
	
	return m.BaseMiddleware.Handle(req, res)
}

// CacheMiddleware 缓存中间件
type CacheMiddleware struct {
	BaseMiddleware
	cache map[string]string
}

func NewCacheMiddleware() *CacheMiddleware {
	return &CacheMiddleware{
		cache: make(map[string]string),
	}
}

func (m *CacheMiddleware) Handle(req *Request, res *Response) error {
	if req.Method != "GET" {
		return m.BaseMiddleware.Handle(req, res)
	}
	
	fmt.Println("[Cache] 检查缓存")
	
	cacheKey := req.Path
	if cached, ok := m.cache[cacheKey]; ok {
		fmt.Println("[Cache] ✓ 命中缓存")
		res.StatusCode = 200
		res.Body = cached
		if res.Headers == nil {
			res.Headers = make(map[string]string)
		}
		res.Headers["X-Cache"] = "HIT"
		return nil
	}
	
	fmt.Println("[Cache] ✗ 未命中缓存")
	
	err := m.BaseMiddleware.Handle(req, res)
	
	if err == nil && res.StatusCode == 200 {
		m.cache[cacheKey] = res.Body
		fmt.Println("[Cache] 保存到缓存")
	}
	
	return err
}

// CompressionMiddleware 压缩中间件
type CompressionMiddleware struct {
	BaseMiddleware
}

func NewCompressionMiddleware() *CompressionMiddleware {
	return &CompressionMiddleware{}
}

func (m *CompressionMiddleware) Handle(req *Request, res *Response) error {
	err := m.BaseMiddleware.Handle(req, res)
	
	if err == nil && len(res.Body) > 100 {
		fmt.Println("[Compression] 压缩响应内容")
		if res.Headers == nil {
			res.Headers = make(map[string]string)
		}
		res.Headers["Content-Encoding"] = "gzip"
	}
	
	return err
}

// ErrorHandlerMiddleware 错误处理中间件
type ErrorHandlerMiddleware struct {
	BaseMiddleware
}

func NewErrorHandlerMiddleware() *ErrorHandlerMiddleware {
	return &ErrorHandlerMiddleware{}
}

func (m *ErrorHandlerMiddleware) Handle(req *Request, res *Response) error {
	err := m.BaseMiddleware.Handle(req, res)
	
	if err != nil {
		fmt.Printf("[ErrorHandler] 捕获错误: %v\n", err)
		if res.StatusCode == 0 {
			res.StatusCode = 500
			res.Body = "Internal Server Error"
		}
	}
	
	return nil // 错误已处理，不再向上传递
}

// BusinessHandler 业务处理器
type BusinessHandler struct {
	BaseMiddleware
}

func NewBusinessHandler() *BusinessHandler {
	return &BusinessHandler{}
}

func (h *BusinessHandler) Handle(req *Request, res *Response) error {
	fmt.Println("[Business] 处理业务逻辑")
	
	time.Sleep(50 * time.Millisecond)
	
	res.StatusCode = 200
	res.Body = fmt.Sprintf("Success: Processed %s %s for user %s", req.Method, req.Path, req.User)
	
	fmt.Println("[Business] ✓ 业务处理完成")
	
	return nil
}

// App 应用
type App struct {
	middlewares []Middleware
}

func NewApp() *App {
	return &App{
		middlewares: make([]Middleware, 0),
	}
}

func (a *App) Use(middleware Middleware) {
	a.middlewares = append(a.middlewares, middleware)
}

func (a *App) Handle(req *Request) *Response {
	res := &Response{}
	
	if len(a.middlewares) == 0 {
		return res
	}
	
	// 构建中间件链
	for i := 0; i < len(a.middlewares)-1; i++ {
		a.middlewares[i].SetNext(a.middlewares[i+1])
	}
	
	// 执行中间件链
	a.middlewares[0].Handle(req, res)
	
	return res
}

func main() {
	fmt.Println("=== 练习3: 请求处理中间件系统 ===\n")

	// 场景1: 正常请求
	fmt.Println("--- 场景1: 正常的 GET 请求 ---")
	app1 := NewApp()
	app1.Use(NewLoggerMiddleware())
	app1.Use(NewAuthMiddleware())
	app1.Use(NewBusinessHandler())
	
	req1 := &Request{
		Method: "GET",
		Path:   "/api/users",
		Headers: map[string]string{
			"Authorization": "Bearer token123",
		},
	}
	
	res1 := app1.Handle(req1)
	fmt.Printf("\n响应: %d - %s\n", res1.StatusCode, res1.Body)
	fmt.Println(strings.Repeat("=", 60))

	// 场景2: 未认证的请求
	fmt.Println("\n--- 场景2: 未认证的请求 ---")
	req2 := &Request{
		Method:  "GET",
		Path:    "/api/users",
		Headers: map[string]string{},
	}
	
	res2 := app1.Handle(req2)
	fmt.Printf("\n响应: %d - %s\n", res2.StatusCode, res2.Body)
	fmt.Println(strings.Repeat("=", 60))

	// 场景3: 限流
	fmt.Println("\n--- 场景3: 超过请求限制 ---")
	app2 := NewApp()
	app2.Use(NewLoggerMiddleware())
	app2.Use(NewAuthMiddleware())
	app2.Use(NewRateLimitMiddleware(3))
	app2.Use(NewBusinessHandler())
	
	for i := 1; i <= 4; i++ {
		fmt.Printf("\n第 %d 次请求:\n", i)
		req := &Request{
			Method: "GET",
			Path:   "/api/data",
			Headers: map[string]string{
				"Authorization": "Bearer token456",
			},
		}
		res := app2.Handle(req)
		fmt.Printf("响应: %d - %s\n", res.StatusCode, res.Body)
	}
	fmt.Println(strings.Repeat("=", 60))

	// 场景4: 缓存
	fmt.Println("\n--- 场景4: 缓存测试 ---")
	app3 := NewApp()
	app3.Use(NewLoggerMiddleware())
	app3.Use(NewAuthMiddleware())
	app3.Use(NewCacheMiddleware())
	app3.Use(NewBusinessHandler())
	
	fmt.Println("第一次请求:")
	req4a := &Request{
		Method: "GET",
		Path:   "/api/products",
		Headers: map[string]string{
			"Authorization": "Bearer token789",
		},
	}
	res4a := app3.Handle(req4a)
	fmt.Printf("响应: %d - %s\n", res4a.StatusCode, res4a.Body)
	
	fmt.Println("\n第二次请求（应该命中缓存）:")
	req4b := &Request{
		Method: "GET",
		Path:   "/api/products",
		Headers: map[string]string{
			"Authorization": "Bearer token789",
		},
	}
	res4b := app3.Handle(req4b)
	fmt.Printf("响应: %d - %s\n", res4b.StatusCode, res4b.Body)
	if res4b.Headers["X-Cache"] == "HIT" {
		fmt.Println("✓ 缓存命中")
	}

	fmt.Println("\n=== 练习完成 ===")
}

// 可能的优化方向:
// 1. 实现条件中间件（根据路径或方法选择性应用）
// 2. 支持中间件分组
// 3. 添加异步中间件支持
// 4. 实现中间件优先级
//
// 变体实现:
// 1. 使用函数类型简化中间件定义
// 2. 使用 context 传递请求上下文
// 3. 支持中间件的热重载
