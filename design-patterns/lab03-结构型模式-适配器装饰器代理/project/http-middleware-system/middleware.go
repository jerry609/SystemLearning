package main

import (
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Middleware 中间件类型定义
// 接收一个 http.Handler，返回一个装饰后的 http.Handler
type Middleware func(http.Handler) http.Handler

// ResponseWriter 响应包装器
// 用于捕获响应状态码和写入的字节数
type ResponseWriter struct {
	http.ResponseWriter
	StatusCode   int
	BytesWritten int
}

// NewResponseWriter 创建响应包装器
func NewResponseWriter(w http.ResponseWriter) *ResponseWriter {
	return &ResponseWriter{
		ResponseWriter: w,
		StatusCode:     http.StatusOK, // 默认 200
	}
}

// WriteHeader 重写 WriteHeader 方法以捕获状态码
func (rw *ResponseWriter) WriteHeader(statusCode int) {
	rw.StatusCode = statusCode
	rw.ResponseWriter.WriteHeader(statusCode)
}

// Write 重写 Write 方法以捕获写入的字节数
func (rw *ResponseWriter) Write(b []byte) (int, error) {
	n, err := rw.ResponseWriter.Write(b)
	rw.BytesWritten += n
	return n, err
}

// LoggingMiddleware 日志中间件
// 记录每个请求的详细信息，包括方法、路径、状态码、响应大小和处理时间
func LoggingMiddleware() Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()

			// 包装 ResponseWriter 以捕获状态码和大小
			rw := NewResponseWriter(w)

			log.Printf("[LOG] Started %s %s from %s", r.Method, r.URL.Path, r.RemoteAddr)

			// 调用下一个处理器
			next.ServeHTTP(rw, r)

			duration := time.Since(start)
			log.Printf("[LOG] Completed %s %s - Status: %d - Size: %d bytes - Duration: %v",
				r.Method, r.URL.Path, rw.StatusCode, rw.BytesWritten, duration)
		})
	}
}

// AuthMiddleware 认证中间件
// 支持 Bearer Token 认证
func AuthMiddleware(validToken string) Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// 从 Authorization 头获取 token
			authHeader := r.Header.Get("Authorization")
			if authHeader == "" {
				log.Printf("[AUTH] Authentication failed: missing authorization header")
				http.Error(w, "Unauthorized: missing authorization header", http.StatusUnauthorized)
				return
			}

			// 检查 Bearer token 格式
			parts := strings.Split(authHeader, " ")
			if len(parts) != 2 || parts[0] != "Bearer" {
				log.Printf("[AUTH] Authentication failed: invalid authorization format")
				http.Error(w, "Unauthorized: invalid authorization format", http.StatusUnauthorized)
				return
			}

			token := parts[1]
			if token != validToken {
				log.Printf("[AUTH] Authentication failed: invalid token")
				http.Error(w, "Unauthorized: invalid token", http.StatusUnauthorized)
				return
			}

			log.Printf("[AUTH] Token authentication successful")
			// 认证通过，调用下一个处理器
			next.ServeHTTP(w, r)
		})
	}
}

// BasicAuthMiddleware Basic 认证中间件
// 支持用户名密码认证
func BasicAuthMiddleware(username, password string) Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			user, pass, ok := r.BasicAuth()
			if !ok {
				log.Printf("[AUTH] Basic authentication failed: missing credentials")
				w.Header().Set("WWW-Authenticate", `Basic realm="Restricted"`)
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			if user != username || pass != password {
				log.Printf("[AUTH] Basic authentication failed: invalid credentials")
				w.Header().Set("WWW-Authenticate", `Basic realm="Restricted"`)
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			log.Printf("[AUTH] Basic authentication successful for user: %s", user)
			next.ServeHTTP(w, r)
		})
	}
}

// RateLimitMiddleware 限流中间件
// 基于 IP 地址限制请求频率
func RateLimitMiddleware(maxRequests int, window time.Duration) Middleware {
	type client struct {
		requests int
		resetAt  time.Time
		mu       sync.Mutex
	}

	clients := make(map[string]*client)
	var mu sync.RWMutex

	// 定期清理过期的客户端记录
	go func() {
		ticker := time.NewTicker(window)
		defer ticker.Stop()
		for range ticker.C {
			mu.Lock()
			now := time.Now()
			for ip, c := range clients {
				if now.After(c.resetAt) {
					delete(clients, ip)
				}
			}
			mu.Unlock()
		}
	}()

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ip := r.RemoteAddr
			now := time.Now()

			// 获取或创建客户端记录
			mu.Lock()
			c, exists := clients[ip]
			if !exists {
				c = &client{
					requests: 0,
					resetAt:  now.Add(window),
				}
				clients[ip] = c
			}
			mu.Unlock()

			// 检查限流
			c.mu.Lock()
			if now.After(c.resetAt) {
				// 重置窗口
				c.requests = 0
				c.resetAt = now.Add(window)
			}

			if c.requests >= maxRequests {
				c.mu.Unlock()
				log.Printf("[RATE_LIMIT] Rate limit exceeded for %s: %d/%d", ip, c.requests, maxRequests)
				http.Error(w, "Too Many Requests", http.StatusTooManyRequests)
				return
			}

			c.requests++
			currentRequests := c.requests
			c.mu.Unlock()

			log.Printf("[RATE_LIMIT] Request count for %s: %d/%d", ip, currentRequests, maxRequests)
			next.ServeHTTP(w, r)
		})
	}
}

// RecoveryMiddleware 恢复中间件
// 捕获 panic 并返回 500 错误，防止服务器崩溃
func RecoveryMiddleware() Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if err := recover(); err != nil {
					log.Printf("[RECOVERY] Panic recovered: %v", err)
					http.Error(w, "Internal Server Error", http.StatusInternalServerError)
				}
			}()

			// 调用下一个处理器
			next.ServeHTTP(w, r)
		})
	}
}

// CORSMiddleware CORS 中间件
// 处理跨域请求
func CORSMiddleware(allowedOrigins []string) Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			origin := r.Header.Get("Origin")

			// 检查是否允许该来源
			allowed := false
			for _, allowedOrigin := range allowedOrigins {
				if origin == allowedOrigin || allowedOrigin == "*" {
					allowed = true
					w.Header().Set("Access-Control-Allow-Origin", origin)
					break
				}
			}

			if allowed {
				w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
				w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
				w.Header().Set("Access-Control-Max-Age", "3600")
				log.Printf("[CORS] Allowed origin: %s", origin)
			}

			// 处理预检请求
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// TracingMiddleware 追踪中间件
// 为每个请求分配唯一的请求 ID
func TracingMiddleware() Middleware {
	var counter uint64
	var mu sync.Mutex

	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// 生成请求 ID
			mu.Lock()
			counter++
			requestID := fmt.Sprintf("req-%d-%d", time.Now().Unix(), counter)
			mu.Unlock()

			// 将请求 ID 添加到 context
			ctx := context.WithValue(r.Context(), "request_id", requestID)
			r = r.WithContext(ctx)

			// 添加到响应头
			w.Header().Set("X-Request-ID", requestID)

			log.Printf("[TRACE] Request ID: %s", requestID)
			next.ServeHTTP(w, r)
		})
	}
}

// CompressionMiddleware 压缩中间件
// 支持 gzip 响应压缩
func CompressionMiddleware() Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// 检查客户端是否支持 gzip
			if !strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
				next.ServeHTTP(w, r)
				return
			}

			// 创建 gzip writer
			w.Header().Set("Content-Encoding", "gzip")
			gz := gzip.NewWriter(w)
			defer gz.Close()

			// 包装 ResponseWriter
			gzw := &gzipResponseWriter{Writer: gz, ResponseWriter: w}
			log.Printf("[COMPRESSION] Enabled gzip compression for %s", r.URL.Path)

			next.ServeHTTP(gzw, r)
		})
	}
}

// gzipResponseWriter gzip 响应包装器
type gzipResponseWriter struct {
	io.Writer
	http.ResponseWriter
}

func (w *gzipResponseWriter) Write(b []byte) (int, error) {
	return w.Writer.Write(b)
}

// TimeoutMiddleware 超时中间件
// 设置请求处理超时时间
func TimeoutMiddleware(timeout time.Duration) Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// 创建带超时的 context
			ctx, cancel := context.WithTimeout(r.Context(), timeout)
			defer cancel()

			// 使用新的 context
			r = r.WithContext(ctx)

			// 使用 channel 等待处理完成或超时
			done := make(chan struct{})
			go func() {
				next.ServeHTTP(w, r)
				close(done)
			}()

			select {
			case <-done:
				// 处理完成
				return
			case <-ctx.Done():
				// 超时
				log.Printf("[TIMEOUT] Request timeout after %v for %s", timeout, r.URL.Path)
				http.Error(w, "Request Timeout", http.StatusRequestTimeout)
			}
		})
	}
}

// Chain 链式组合中间件
// 从左到右应用中间件（最左边的中间件最外层）
func Chain(handler http.Handler, middlewares ...Middleware) http.Handler {
	// 从后向前应用中间件
	for i := len(middlewares) - 1; i >= 0; i-- {
		handler = middlewares[i](handler)
	}
	return handler
}
