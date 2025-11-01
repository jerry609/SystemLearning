package main

// 练习 2: 实现 HTTP 请求装饰器链 - 参考答案
//
// 依赖安装:
// go get golang.org/x/time/rate
//
// 或者在当前目录初始化 Go 模块:
// go mod init exercise2
// go mod tidy
//
// 设计思路:
// 1. 定义统一的 HTTPClient 接口
// 2. 实现基础的 HTTP 客户端
// 3. 为不同功能创建装饰器（日志、重试、超时、缓存、认证、限流）
// 4. 装饰器可以任意组合，形成装饰器链
//
// 使用的设计模式: 装饰器模式
// 模式应用位置:
// - LoggingDecorator: 添加日志功能
// - RetryDecorator: 添加重试功能
// - TimeoutDecorator: 添加超时控制
// - CacheDecorator: 添加缓存功能
// - AuthDecorator: 添加认证功能
// - RateLimitDecorator: 添加限流功能

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/time/rate"
)

// ============================================================================
// 接口定义
// ============================================================================

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

// ============================================================================
// 基础 HTTP 客户端
// ============================================================================

// BaseHTTPClient 基础 HTTP 客户端
type BaseHTTPClient struct{}

// NewBaseHTTPClient 创建基础 HTTP 客户端
func NewBaseHTTPClient() *BaseHTTPClient {
	return &BaseHTTPClient{}
}

func (c *BaseHTTPClient) Get(url string) (*Response, error) {
	return c.doRequest("GET", url, nil)
}

func (c *BaseHTTPClient) Post(url string, body []byte) (*Response, error) {
	return c.doRequest("POST", url, body)
}

func (c *BaseHTTPClient) Put(url string, body []byte) (*Response, error) {
	return c.doRequest("PUT", url, body)
}

func (c *BaseHTTPClient) Delete(url string) (*Response, error) {
	return c.doRequest("DELETE", url, nil)
}

func (c *BaseHTTPClient) doRequest(method, url string, body []byte) (*Response, error) {
	// 模拟 HTTP 请求
	start := time.Now()
	time.Sleep(100 * time.Millisecond) // 模拟网络延迟

	return &Response{
		StatusCode: 200,
		Headers:    map[string]string{"Content-Type": "application/json"},
		Body:       []byte(`{"status": "success"}`),
		Duration:   time.Since(start),
	}, nil
}

// ============================================================================
// 日志装饰器
// ============================================================================

// LoggingDecorator 日志装饰器
type LoggingDecorator struct {
	client HTTPClient
	logger *log.Logger
}

// NewLoggingDecorator 创建日志装饰器
func NewLoggingDecorator(client HTTPClient) *LoggingDecorator {
	return &LoggingDecorator{
		client: client,
		logger: log.New(os.Stdout, "[HTTP] ", log.LstdFlags),
	}
}

func (d *LoggingDecorator) Get(url string) (*Response, error) {
	return d.logRequest("GET", url, func() (*Response, error) {
		return d.client.Get(url)
	})
}

func (d *LoggingDecorator) Post(url string, body []byte) (*Response, error) {
	return d.logRequest("POST", url, func() (*Response, error) {
		return d.client.Post(url, body)
	})
}

func (d *LoggingDecorator) Put(url string, body []byte) (*Response, error) {
	return d.logRequest("PUT", url, func() (*Response, error) {
		return d.client.Put(url, body)
	})
}

func (d *LoggingDecorator) Delete(url string) (*Response, error) {
	return d.logRequest("DELETE", url, func() (*Response, error) {
		return d.client.Delete(url)
	})
}

func (d *LoggingDecorator) logRequest(method, url string, fn func() (*Response, error)) (*Response, error) {
	d.logger.Printf("%s %s", method, url)
	start := time.Now()

	resp, err := fn()

	duration := time.Since(start)
	if err != nil {
		d.logger.Printf("Error: %v (Duration: %v)", err, duration)
		return nil, err
	}

	d.logger.Printf("Response: %d (Duration: %v)", resp.StatusCode, duration)
	return resp, nil
}

// ============================================================================
// 重试装饰器
// ============================================================================

// RetryConfig 重试配置
type RetryConfig struct {
	MaxRetries      int
	RetryDelay      time.Duration
	RetryableStatus []int
}

// RetryDecorator 重试装饰器
type RetryDecorator struct {
	client HTTPClient
	config RetryConfig
}

// NewRetryDecorator 创建重试装饰器
func NewRetryDecorator(client HTTPClient, config RetryConfig) *RetryDecorator {
	if config.RetryableStatus == nil {
		config.RetryableStatus = []int{500, 502, 503, 504}
	}
	return &RetryDecorator{
		client: client,
		config: config,
	}
}

func (d *RetryDecorator) Get(url string) (*Response, error) {
	return d.retryRequest(func() (*Response, error) {
		return d.client.Get(url)
	})
}

func (d *RetryDecorator) Post(url string, body []byte) (*Response, error) {
	return d.retryRequest(func() (*Response, error) {
		return d.client.Post(url, body)
	})
}

func (d *RetryDecorator) Put(url string, body []byte) (*Response, error) {
	return d.retryRequest(func() (*Response, error) {
		return d.client.Put(url, body)
	})
}

func (d *RetryDecorator) Delete(url string) (*Response, error) {
	return d.retryRequest(func() (*Response, error) {
		return d.client.Delete(url)
	})
}

func (d *RetryDecorator) retryRequest(fn func() (*Response, error)) (*Response, error) {
	var lastErr error

	for attempt := 0; attempt <= d.config.MaxRetries; attempt++ {
		if attempt > 0 {
			// 指数退避
			delay := d.config.RetryDelay * time.Duration(1<<uint(attempt-1))
			fmt.Printf("[Retry] Attempt %d failed, waiting %v\n", attempt, delay)
			time.Sleep(delay)
		}

		resp, err := fn()
		if err == nil && !d.shouldRetry(resp.StatusCode) {
			if attempt > 0 {
				fmt.Printf("[Retry] Attempt %d succeeded\n", attempt+1)
			}
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
	return false
}

// ============================================================================
// 超时装饰器
// ============================================================================

// TimeoutDecorator 超时装饰器
type TimeoutDecorator struct {
	client  HTTPClient
	timeout time.Duration
}

// NewTimeoutDecorator 创建超时装饰器
func NewTimeoutDecorator(client HTTPClient, timeout time.Duration) *TimeoutDecorator {
	return &TimeoutDecorator{
		client:  client,
		timeout: timeout,
	}
}

func (d *TimeoutDecorator) Get(url string) (*Response, error) {
	return d.withTimeout(func() (*Response, error) {
		return d.client.Get(url)
	})
}

func (d *TimeoutDecorator) Post(url string, body []byte) (*Response, error) {
	return d.withTimeout(func() (*Response, error) {
		return d.client.Post(url, body)
	})
}

func (d *TimeoutDecorator) Put(url string, body []byte) (*Response, error) {
	return d.withTimeout(func() (*Response, error) {
		return d.client.Put(url, body)
	})
}

func (d *TimeoutDecorator) Delete(url string) (*Response, error) {
	return d.withTimeout(func() (*Response, error) {
		return d.client.Delete(url)
	})
}

func (d *TimeoutDecorator) withTimeout(fn func() (*Response, error)) (*Response, error) {
	ctx, cancel := context.WithTimeout(context.Background(), d.timeout)
	defer cancel()

	type result struct {
		resp *Response
		err  error
	}

	ch := make(chan result, 1)
	go func() {
		resp, err := fn()
		ch <- result{resp, err}
	}()

	select {
	case res := <-ch:
		return res.resp, res.err
	case <-ctx.Done():
		return nil, fmt.Errorf("request timeout after %v", d.timeout)
	}
}

// ============================================================================
// 缓存装饰器
// ============================================================================

// CacheConfig 缓存配置
type CacheConfig struct {
	TTL     time.Duration
	MaxSize int
}

type cacheEntry struct {
	response  *Response
	expiresAt time.Time
}

// CacheDecorator 缓存装饰器
type CacheDecorator struct {
	client HTTPClient
	cache  map[string]*cacheEntry
	mu     sync.RWMutex
	config CacheConfig
}

// NewCacheDecorator 创建缓存装饰器
func NewCacheDecorator(client HTTPClient, config CacheConfig) *CacheDecorator {
	return &CacheDecorator{
		client: client,
		cache:  make(map[string]*cacheEntry),
		config: config,
	}
}

func (d *CacheDecorator) Get(url string) (*Response, error) {
	// 检查缓存
	d.mu.RLock()
	if entry, ok := d.cache[url]; ok {
		if time.Now().Before(entry.expiresAt) {
			d.mu.RUnlock()
			fmt.Println("[Cache] Cache hit")
			return entry.response, nil
		}
	}
	d.mu.RUnlock()

	fmt.Println("[Cache] Cache miss, fetching from server")

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

func (d *CacheDecorator) Post(url string, body []byte) (*Response, error) {
	// POST 请求不缓存
	return d.client.Post(url, body)
}

func (d *CacheDecorator) Put(url string, body []byte) (*Response, error) {
	// PUT 请求不缓存，并清除相关缓存
	d.mu.Lock()
	delete(d.cache, url)
	d.mu.Unlock()
	return d.client.Put(url, body)
}

func (d *CacheDecorator) Delete(url string) (*Response, error) {
	// DELETE 请求不缓存，并清除相关缓存
	d.mu.Lock()
	delete(d.cache, url)
	d.mu.Unlock()
	return d.client.Delete(url)
}

// ============================================================================
// 认证装饰器
// ============================================================================

// AuthDecorator 认证装饰器
type AuthDecorator struct {
	client    HTTPClient
	authType  string
	authValue string
}

// NewAuthDecorator 创建认证装饰器
func NewAuthDecorator(client HTTPClient, authType, authValue string) *AuthDecorator {
	return &AuthDecorator{
		client:    client,
		authType:  authType,
		authValue: authValue,
	}
}

func (d *AuthDecorator) Get(url string) (*Response, error) {
	fmt.Printf("[Auth] Adding %s authentication\n", d.authType)
	return d.client.Get(url)
}

func (d *AuthDecorator) Post(url string, body []byte) (*Response, error) {
	fmt.Printf("[Auth] Adding %s authentication\n", d.authType)
	return d.client.Post(url, body)
}

func (d *AuthDecorator) Put(url string, body []byte) (*Response, error) {
	fmt.Printf("[Auth] Adding %s authentication\n", d.authType)
	return d.client.Put(url, body)
}

func (d *AuthDecorator) Delete(url string) (*Response, error) {
	fmt.Printf("[Auth] Adding %s authentication\n", d.authType)
	return d.client.Delete(url)
}

// ============================================================================
// 限流装饰器
// ============================================================================

// RateLimitConfig 限流配置
type RateLimitConfig struct {
	RequestsPerSecond float64
}

// RateLimitDecorator 限流装饰器
type RateLimitDecorator struct {
	client  HTTPClient
	limiter *rate.Limiter
}

// NewRateLimitDecorator 创建限流装饰器
func NewRateLimitDecorator(client HTTPClient, config RateLimitConfig) *RateLimitDecorator {
	return &RateLimitDecorator{
		client:  client,
		limiter: rate.NewLimiter(rate.Limit(config.RequestsPerSecond), 1),
	}
}

func (d *RateLimitDecorator) Get(url string) (*Response, error) {
	return d.withRateLimit(func() (*Response, error) {
		return d.client.Get(url)
	})
}

func (d *RateLimitDecorator) Post(url string, body []byte) (*Response, error) {
	return d.withRateLimit(func() (*Response, error) {
		return d.client.Post(url, body)
	})
}

func (d *RateLimitDecorator) Put(url string, body []byte) (*Response, error) {
	return d.withRateLimit(func() (*Response, error) {
		return d.client.Put(url, body)
	})
}

func (d *RateLimitDecorator) Delete(url string) (*Response, error) {
	return d.withRateLimit(func() (*Response, error) {
		return d.client.Delete(url)
	})
}

func (d *RateLimitDecorator) withRateLimit(fn func() (*Response, error)) (*Response, error) {
	// 等待令牌
	if err := d.limiter.Wait(context.Background()); err != nil {
		return nil, err
	}
	return fn()
}

// ============================================================================
// 示例代码
// ============================================================================

func main() {
	fmt.Println("=== HTTP 请求装饰器链示例 ===\n")

	// 示例 1: 基本装饰器组合
	fmt.Println("--- 示例 1: 日志 + 重试 + 超时 ---")
	client1 := NewBaseHTTPClient()
	client1 = NewTimeoutDecorator(client1, 5*time.Second)
	client1 = NewRetryDecorator(client1, RetryConfig{
		MaxRetries: 2,
		RetryDelay: 500 * time.Millisecond,
	})
	client1 = NewLoggingDecorator(client1)

	resp1, _ := client1.Get("https://api.example.com/users")
	fmt.Printf("Status: %d\n\n", resp1.StatusCode)

	// 示例 2: 带缓存和认证
	fmt.Println("--- 示例 2: 认证 + 缓存 + 日志 ---")
	client2 := NewBaseHTTPClient()
	client2 = NewAuthDecorator(client2, "Bearer", "your-token-here")
	client2 = NewCacheDecorator(client2, CacheConfig{
		TTL:     5 * time.Minute,
		MaxSize: 100,
	})
	client2 = NewLoggingDecorator(client2)

	// 第一次请求
	resp2, _ := client2.Get("https://api.example.com/users/123")
	fmt.Printf("First request: %d\n", resp2.StatusCode)

	// 第二次请求（从缓存）
	resp3, _ := client2.Get("https://api.example.com/users/123")
	fmt.Printf("Second request: %d\n\n", resp3.StatusCode)

	// 示例 3: 限流装饰器
	fmt.Println("--- 示例 3: 限流 + 日志 ---")
	client3 := NewBaseHTTPClient()
	client3 = NewRateLimitDecorator(client3, RateLimitConfig{
		RequestsPerSecond: 2,
	})
	client3 = NewLoggingDecorator(client3)

	// 快速发送 3 个请求
	for i := 0; i < 3; i++ {
		start := time.Now()
		resp, _ := client3.Get(fmt.Sprintf("https://api.example.com/item/%d", i))
		fmt.Printf("Request %d: %d (waited: %v)\n", i, resp.StatusCode, time.Since(start))
	}

	fmt.Println("\n=== 示例结束 ===")
}

// 可能的优化方向:
// 1. 实现断路器装饰器
// 2. 添加指标收集装饰器
// 3. 实现请求去重装饰器
// 4. 添加压缩装饰器
// 5. 实现更智能的缓存策略
//
// 变体实现:
// 1. 使用中间件链模式
// 2. 使用函数式选项模式
// 3. 实现装饰器工厂
