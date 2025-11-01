package main

import (
	"fmt"
	"log"
	"net/http"
	"time"
)

// HTTP 中间件示例
// 本示例展示了装饰器模式在 HTTP 中间件中的应用
// 使用函数式装饰器实现日志、认证、限流等功能

// HandlerFunc 类型定义
type HandlerFunc func(http.ResponseWriter, *http.Request)

// Middleware 中间件类型
// 接收一个 HandlerFunc，返回一个装饰后的 HandlerFunc
type Middleware func(HandlerFunc) HandlerFunc

// LoggingMiddleware 日志中间件
// 记录请求的开始和结束时间
func LoggingMiddleware() Middleware {
	return func(next HandlerFunc) HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			log.Printf("[LOG] Started %s %s", r.Method, r.URL.Path)

			// 调用下一个处理器
			next(w, r)

			duration := time.Since(start)
			log.Printf("[LOG] Completed in %v", duration)
		}
	}
}

// AuthMiddleware 认证中间件
// 验证请求头中的 Authorization token
func AuthMiddleware(validToken string) Middleware {
	return func(next HandlerFunc) HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			token := r.Header.Get("Authorization")
			if token != validToken {
				log.Printf("[AUTH] Unauthorized access attempt")
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}

			log.Printf("[AUTH] Authentication successful")
			// 认证通过，调用下一个处理器
			next(w, r)
		}
	}
}

// RecoveryMiddleware 恢复中间件
// 捕获 panic 并返回 500 错误
func RecoveryMiddleware() Middleware {
	return func(next HandlerFunc) HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if err := recover(); err != nil {
					log.Printf("[RECOVERY] Panic recovered: %v", err)
					http.Error(w, "Internal Server Error", http.StatusInternalServerError)
				}
			}()

			// 调用下一个处理器
			next(w, r)
		}
	}
}

// RateLimitMiddleware 限流中间件
// 限制请求频率
func RateLimitMiddleware(maxRequests int, window time.Duration) Middleware {
	type client struct {
		requests int
		resetAt  time.Time
	}
	clients := make(map[string]*client)

	return func(next HandlerFunc) HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			ip := r.RemoteAddr
			now := time.Now()

			// 检查客户端是否存在或需要重置
			c, exists := clients[ip]
			if !exists || now.After(c.resetAt) {
				clients[ip] = &client{
					requests: 1,
					resetAt:  now.Add(window),
				}
				log.Printf("[RATE_LIMIT] New window for %s: 1/%d", ip, maxRequests)
				next(w, r)
				return
			}

			// 检查是否超过限制
			if c.requests >= maxRequests {
				log.Printf("[RATE_LIMIT] Rate limit exceeded for %s", ip)
				http.Error(w, "Too Many Requests", http.StatusTooManyRequests)
				return
			}

			// 增加请求计数
			c.requests++
			log.Printf("[RATE_LIMIT] Request count for %s: %d/%d", ip, c.requests, maxRequests)
			next(w, r)
		}
	}
}

// CORSMiddleware CORS 中间件
// 处理跨域请求
func CORSMiddleware(allowedOrigins []string) Middleware {
	return func(next HandlerFunc) HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			origin := r.Header.Get("Origin")

			// 检查是否允许该来源
			for _, allowed := range allowedOrigins {
				if origin == allowed {
					w.Header().Set("Access-Control-Allow-Origin", origin)
					w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
					w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
					log.Printf("[CORS] Allowed origin: %s", origin)
					break
				}
			}

			// 处理预检请求
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}

			next(w, r)
		}
	}
}

// Chain 链式组合中间件
// 从右到左应用中间件（最右边的中间件最先执行）
func Chain(handler HandlerFunc, middlewares ...Middleware) HandlerFunc {
	// 从后向前应用中间件
	for i := len(middlewares) - 1; i >= 0; i-- {
		handler = middlewares[i](handler)
	}
	return handler
}

// 业务处理器
func HelloHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello, World! Time: %s\n", time.Now().Format(time.RFC3339))
}

func UserHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "User Profile\n")
}

func PanicHandler(w http.ResponseWriter, r *http.Request) {
	panic("Something went wrong!")
}

func main() {
	fmt.Println("=== HTTP 中间件示例 ===\n")

	// 配置日志
	log.SetFlags(log.Ltime)

	// 示例 1: 基础中间件链
	fmt.Println("示例 1: 基础中间件链")
	fmt.Println("访问 http://localhost:8080/hello")
	fmt.Println("需要 Authorization: secret-token\n")

	handler1 := Chain(
		HelloHandler,
		LoggingMiddleware(),
		AuthMiddleware("secret-token"),
	)

	// 示例 2: 完整的中间件链
	fmt.Println("示例 2: 完整的中间件链（包含恢复、日志、认证、限流、CORS）")
	fmt.Println("访问 http://localhost:8080/user")
	fmt.Println("需要 Authorization: secret-token\n")

	handler2 := Chain(
		UserHandler,
		RecoveryMiddleware(),
		LoggingMiddleware(),
		AuthMiddleware("secret-token"),
		RateLimitMiddleware(10, time.Minute),
		CORSMiddleware([]string{"http://localhost:3000", "http://localhost:8080"}),
	)

	// 示例 3: 测试恢复中间件
	fmt.Println("示例 3: 测试恢复中间件")
	fmt.Println("访问 http://localhost:8080/panic 会触发 panic，但会被恢复\n")

	handler3 := Chain(
		PanicHandler,
		RecoveryMiddleware(),
		LoggingMiddleware(),
	)

	// 注册路由
	http.HandleFunc("/hello", handler1)
	http.HandleFunc("/user", handler2)
	http.HandleFunc("/panic", handler3)

	// 启动服务器
	fmt.Println("服务器启动在 :8080")
	fmt.Println("\n测试命令:")
	fmt.Println("  curl -H \"Authorization: secret-token\" http://localhost:8080/hello")
	fmt.Println("  curl -H \"Authorization: secret-token\" http://localhost:8080/user")
	fmt.Println("  curl http://localhost:8080/panic")
	fmt.Println("\n按 Ctrl+C 停止服务器")
	fmt.Println("\n=== 服务器日志 ===")

	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}
