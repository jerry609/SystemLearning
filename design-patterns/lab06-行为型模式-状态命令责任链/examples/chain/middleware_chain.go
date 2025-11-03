package main

import (
	"fmt"
	"strings"
	"time"
)

// HTTP 中间件链示例
// 本示例展示了责任链模式在 HTTP 中间件中的应用

// Request HTTP 请求
type Request struct {
	Method  string
	Path    string
	Headers map[string]string
	Body    string
	User    string
	StartTime time.Time
}

// Response HTTP 响应
type Response struct {
	StatusCode int
	Body       string
	Headers    map[string]string
}

// Handler 处理者接口
type Handler interface {
	SetNext(handler Handler) Handler
	Handle(req *Request, res *Response) error
}

// BaseHandler 基础处理者
type BaseHandler struct {
	next Handler
}

func (h *BaseHandler) SetNext(handler Handler) Handler {
	h.next = handler
	return handler
}

func (h *BaseHandler) Handle(req *Request, res *Response) error {
	if h.next != nil {
		return h.next.Handle(req, res)
	}
	return nil
}

// LoggerMiddleware 日志中间件
type LoggerMiddleware struct {
	BaseHandler
}

func NewLoggerMiddleware() *LoggerMiddleware {
	return &LoggerMiddleware{}
}

func (m *LoggerMiddleware) Handle(req *Request, res *Response) error {
	req.StartTime = time.Now()
	fmt.Printf("[Logger] %s %s - 开始处理\n", req.Method, req.Path)
	
	// 调用下一个处理者
	err := m.BaseHandler.Handle(req, res)
	
	duration := time.Since(req.StartTime)
	fmt.Printf("[Logger] %s %s - 完成处理，耗时: %v，状态码: %d\n", 
		req.Method, req.Path, duration, res.StatusCode)
	
	return err
}

// AuthMiddleware 认证中间件
type AuthMiddleware struct {
	BaseHandler
	validTokens map[string]string
}

func NewAuthMiddleware() *AuthMiddleware {
	return &AuthMiddleware{
		validTokens: map[string]string{
			"token123": "user1",
			"token456": "user2",
			"token789": "admin",
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
	
	// 调用下一个处理者
	return m.BaseHandler.Handle(req, res)
}

// RateLimitMiddleware 限流中间件
type RateLimitMiddleware struct {
	BaseHandler
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
	
	// 调用下一个处理者
	return m.BaseHandler.Handle(req, res)
}

// ValidationMiddleware 验证中间件
type ValidationMiddleware struct {
	BaseHandler
}

func NewValidationMiddleware() *ValidationMiddleware {
	return &ValidationMiddleware{}
}

func (m *ValidationMiddleware) Handle(req *Request, res *Response) error {
	fmt.Println("[Validation] 验证请求参数")
	
	// 验证请求方法
	validMethods := []string{"GET", "POST", "PUT", "DELETE"}
	methodValid := false
	for _, method := range validMethods {
		if req.Method == method {
			methodValid = true
			break
		}
	}
	
	if !methodValid {
		fmt.Printf("[Validation] ✗ 无效的请求方法: %s\n", req.Method)
		res.StatusCode = 405
		res.Body = "Method Not Allowed"
		return fmt.Errorf("invalid method")
	}
	
	// 验证路径
	if req.Path == "" {
		fmt.Println("[Validation] ✗ 路径不能为空")
		res.StatusCode = 400
		res.Body = "Bad Request: Empty path"
		return fmt.Errorf("empty path")
	}
	
	// POST/PUT 请求需要有 Body
	if (req.Method == "POST" || req.Method == "PUT") && req.Body == "" {
		fmt.Println("[Validation] ✗ POST/PUT 请求需要提供 Body")
		res.StatusCode = 400
		res.Body = "Bad Request: Missing body"
		return fmt.Errorf("missing body")
	}
	
	fmt.Println("[Validation] ✓ 请求参数验证通过")
	
	// 调用下一个处理者
	return m.BaseHandler.Handle(req, res)
}

// CORSMiddleware CORS 中间件
type CORSMiddleware struct {
	BaseHandler
	allowedOrigins []string
}

func NewCORSMiddleware(origins []string) *CORSMiddleware {
	return &CORSMiddleware{
		allowedOrigins: origins,
	}
}

func (m *CORSMiddleware) Handle(req *Request, res *Response) error {
	fmt.Println("[CORS] 处理跨域请求")
	
	origin := req.Headers["Origin"]
	if origin == "" {
		fmt.Println("[CORS] 非跨域请求，跳过")
		return m.BaseHandler.Handle(req, res)
	}
	
	allowed := false
	for _, allowedOrigin := range m.allowedOrigins {
		if origin == allowedOrigin || allowedOrigin == "*" {
			allowed = true
			break
		}
	}
	
	if !allowed {
		fmt.Printf("[CORS] ✗ 不允许的来源: %s\n", origin)
		res.StatusCode = 403
		res.Body = "Forbidden: Origin not allowed"
		return fmt.Errorf("origin not allowed")
	}
	
	fmt.Printf("[CORS] ✓ 允许来源: %s\n", origin)
	if res.Headers == nil {
		res.Headers = make(map[string]string)
	}
	res.Headers["Access-Control-Allow-Origin"] = origin
	
	// 调用下一个处理者
	return m.BaseHandler.Handle(req, res)
}

// BusinessHandler 业务处理器
type BusinessHandler struct {
	BaseHandler
}

func NewBusinessHandler() *BusinessHandler {
	return &BusinessHandler{}
}

func (h *BusinessHandler) Handle(req *Request, res *Response) error {
	fmt.Println("[Business] 处理业务逻辑")
	
	// 模拟业务处理
	time.Sleep(50 * time.Millisecond)
	
	res.StatusCode = 200
	res.Body = fmt.Sprintf("Success: Processed %s %s for user %s", req.Method, req.Path, req.User)
	
	fmt.Println("[Business] ✓ 业务处理完成")
	
	return nil
}

func main() {
	fmt.Println("=== 责任链模式示例 - HTTP 中间件链 ===\n")

	// 构建中间件链
	logger := NewLoggerMiddleware()
	cors := NewCORSMiddleware([]string{"http://example.com", "http://localhost:3000"})
	validation := NewValidationMiddleware()
	auth := NewAuthMiddleware()
	rateLimit := NewRateLimitMiddleware(3)
	business := NewBusinessHandler()

	// 链接中间件
	logger.SetNext(cors).
		SetNext(validation).
		SetNext(auth).
		SetNext(rateLimit).
		SetNext(business)

	// 场景1: 正常请求
	fmt.Println("--- 场景1: 正常的 GET 请求 ---")
	req1 := &Request{
		Method: "GET",
		Path:   "/api/users",
		Headers: map[string]string{
			"Authorization": "token123",
			"Origin":        "http://example.com",
		},
	}
	res1 := &Response{}
	
	logger.Handle(req1, res1)
	fmt.Printf("\n响应: %d - %s\n", res1.StatusCode, res1.Body)
	if len(res1.Headers) > 0 {
		fmt.Println("响应头:")
		for k, v := range res1.Headers {
			fmt.Printf("  %s: %s\n", k, v)
		}
	}
	fmt.Println(strings.Repeat("=", 60))

	// 场景2: 未认证的请求
	fmt.Println("\n--- 场景2: 未认证的请求 ---")
	req2 := &Request{
		Method:  "GET",
		Path:    "/api/users",
		Headers: map[string]string{},
	}
	res2 := &Response{}
	
	logger.Handle(req2, res2)
	fmt.Printf("\n响应: %d - %s\n", res2.StatusCode, res2.Body)
	fmt.Println(strings.Repeat("=", 60))

	// 场景3: 无效的令牌
	fmt.Println("\n--- 场景3: 无效的认证令牌 ---")
	req3 := &Request{
		Method: "GET",
		Path:   "/api/users",
		Headers: map[string]string{
			"Authorization": "invalid_token",
		},
	}
	res3 := &Response{}
	
	logger.Handle(req3, res3)
	fmt.Printf("\n响应: %d - %s\n", res3.StatusCode, res3.Body)
	fmt.Println(strings.Repeat("=", 60))

	// 场景4: 超过限流
	fmt.Println("\n--- 场景4: 超过请求限制 ---")
	for i := 1; i <= 4; i++ {
		fmt.Printf("\n第 %d 次请求:\n", i)
		req := &Request{
			Method: "GET",
			Path:   "/api/data",
			Headers: map[string]string{
				"Authorization": "token456",
			},
		}
		res := &Response{}
		
		logger.Handle(req, res)
		fmt.Printf("响应: %d - %s\n", res.StatusCode, res.Body)
	}
	fmt.Println(strings.Repeat("=", 60))

	// 场景5: POST 请求缺少 Body
	fmt.Println("\n--- 场景5: POST 请求缺少 Body ---")
	req5 := &Request{
		Method: "POST",
		Path:   "/api/users",
		Headers: map[string]string{
			"Authorization": "token789",
		},
		Body: "",
	}
	res5 := &Response{}
	
	logger.Handle(req5, res5)
	fmt.Printf("\n响应: %d - %s\n", res5.StatusCode, res5.Body)
	fmt.Println(strings.Repeat("=", 60))

	// 场景6: 不允许的跨域来源
	fmt.Println("\n--- 场景6: 不允许的跨域来源 ---")
	req6 := &Request{
		Method: "GET",
		Path:   "/api/users",
		Headers: map[string]string{
			"Authorization": "token123",
			"Origin":        "http://evil.com",
		},
	}
	res6 := &Response{}
	
	logger.Handle(req6, res6)
	fmt.Printf("\n响应: %d - %s\n", res6.StatusCode, res6.Body)
	fmt.Println(strings.Repeat("=", 60))

	// 场景7: 完整的 POST 请求
	fmt.Println("\n--- 场景7: 完整的 POST 请求 ---")
	req7 := &Request{
		Method: "POST",
		Path:   "/api/users",
		Headers: map[string]string{
			"Authorization": "token789",
			"Origin":        "http://localhost:3000",
		},
		Body: `{"name": "John", "email": "john@example.com"}`,
	}
	res7 := &Response{}
	
	logger.Handle(req7, res7)
	fmt.Printf("\n响应: %d - %s\n", res7.StatusCode, res7.Body)
	if len(res7.Headers) > 0 {
		fmt.Println("响应头:")
		for k, v := range res7.Headers {
			fmt.Printf("  %s: %s\n", k, v)
		}
	}

	fmt.Println("\n=== 示例结束 ===")
}
