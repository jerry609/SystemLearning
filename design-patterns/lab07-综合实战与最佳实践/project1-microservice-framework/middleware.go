package main

import (
	"fmt"
	"time"
)

// 中间件链 - 装饰器模式 + 责任链模式
//
// 本模块结合装饰器模式和责任链模式实现灵活的中间件系统
// 装饰器模式：动态添加功能
// 责任链模式：按顺序处理请求

// Context 请求上下文
type Context struct {
	Path   string
	Token  string
	UserID string
	Data   map[string]interface{}
}

func NewContext(path string) *Context {
	return &Context{
		Path: path,
		Data: make(map[string]interface{}),
	}
}

// Handler 处理器函数类型
type Handler func(ctx *Context) error

// Middleware 中间件接口
type Middleware interface {
	Process(ctx *Context, next Handler) error
}

// MiddlewareChain 中间件链
type MiddlewareChain struct {
	middlewares []Middleware
	handler     Handler
}

func NewMiddlewareChain(handler Handler) *MiddlewareChain {
	return &MiddlewareChain{
		middlewares: make([]Middleware, 0),
		handler:     handler,
	}
}

// Use 添加中间件
func (c *MiddlewareChain) Use(middleware Middleware) *MiddlewareChain {
	c.middlewares = append(c.middlewares, middleware)
	return c
}

// Execute 执行中间件链
func (c *MiddlewareChain) Execute(ctx *Context) error {
	if len(c.middlewares) == 0 {
		return c.handler(ctx)
	}

	// 构建责任链
	var chain Handler
	chain = c.handler

	// 从后向前包装中间件
	for i := len(c.middlewares) - 1; i >= 0; i-- {
		middleware := c.middlewares[i]
		next := chain
		chain = func(ctx *Context) error {
			return middleware.Process(ctx, next)
		}
	}

	return chain(ctx)
}

// LoggingMiddleware 日志中间件
type LoggingMiddleware struct{}

func (m *LoggingMiddleware) Process(ctx *Context, next Handler) error {
	fmt.Printf("[日志中间件] 请求开始: %s\n", ctx.Path)
	start := time.Now()

	err := next(ctx)

	duration := time.Since(start)
	fmt.Printf("[日志中间件] 请求完成: %s (耗时: %v)\n", ctx.Path, duration)
	return err
}

// AuthMiddleware 认证中间件
type AuthMiddleware struct{}

func (m *AuthMiddleware) Process(ctx *Context, next Handler) error {
	if ctx.Token == "" {
		return fmt.Errorf("未提供认证令牌")
	}

	fmt.Printf("[认证中间件] 验证令牌: %s\n", ctx.Token)

	// 模拟令牌验证
	if ctx.Token == "token-123" {
		ctx.UserID = "user-001"
	} else {
		return fmt.Errorf("无效的令牌")
	}

	return next(ctx)
}

// RateLimitMiddleware 限流中间件
type RateLimitMiddleware struct {
	maxRequests int
	window      time.Duration
	requests    map[string][]time.Time
}

func NewRateLimitMiddleware(maxRequests int, window time.Duration) *RateLimitMiddleware {
	return &RateLimitMiddleware{
		maxRequests: maxRequests,
		window:      window,
		requests:    make(map[string][]time.Time),
	}
}

func (m *RateLimitMiddleware) Process(ctx *Context, next Handler) error {
	fmt.Println("[限流中间件] 检查速率限制")

	userID := ctx.UserID
	if userID == "" {
		userID = "anonymous"
	}

	now := time.Now()
	requests := m.requests[userID]

	// 清理过期的请求记录
	validRequests := make([]time.Time, 0)
	for _, t := range requests {
		if now.Sub(t) < m.window {
			validRequests = append(validRequests, t)
		}
	}

	// 检查是否超过限制
	if len(validRequests) >= m.maxRequests {
		return fmt.Errorf("请求过于频繁，请稍后再试")
	}

	// 记录本次请求
	validRequests = append(validRequests, now)
	m.requests[userID] = validRequests

	return next(ctx)
}

// CacheMiddleware 缓存中间件
type CacheMiddleware struct {
	cache map[string]interface{}
}

func NewCacheMiddleware() *CacheMiddleware {
	return &CacheMiddleware{
		cache: make(map[string]interface{}),
	}
}

func (m *CacheMiddleware) Process(ctx *Context, next Handler) error {
	// 检查缓存
	if cached, ok := m.cache[ctx.Path]; ok {
		fmt.Printf("[缓存中间件] 命中缓存: %s\n", ctx.Path)
		ctx.Data["cached"] = cached
		return nil
	}

	fmt.Printf("[缓存中间件] 未命中缓存: %s\n", ctx.Path)

	// 执行下一个处理器
	err := next(ctx)
	if err != nil {
		return err
	}

	// 缓存结果
	if result, ok := ctx.Data["result"]; ok {
		m.cache[ctx.Path] = result
	}

	return nil
}
