package main

import (
	"fmt"
	"log"
	"net/http"
	"time"
)

// 业务处理器

// UsersHandler 用户列表处理器
func UsersHandler(w http.ResponseWriter, r *http.Request) {
	users := []string{"Alice", "Bob", "Charlie", "David"}
	fmt.Fprintf(w, "Users: %v\n", users)
}

// ProductsHandler 产品列表处理器
func ProductsHandler(w http.ResponseWriter, r *http.Request) {
	products := []string{"Laptop", "Phone", "Tablet", "Watch"}
	fmt.Fprintf(w, "Products: %v\n", products)
}

// HealthHandler 健康检查处理器
func HealthHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "OK\n")
}

// SlowHandler 慢速处理器（用于测试超时）
func SlowHandler(w http.ResponseWriter, r *http.Request) {
	// 模拟耗时操作
	time.Sleep(3 * time.Second)
	fmt.Fprintf(w, "Slow response\n")
}

// PanicHandler 触发 panic 的处理器（用于测试恢复中间件）
func PanicHandler(w http.ResponseWriter, r *http.Request) {
	panic("Something went wrong!")
}

// LargeResponseHandler 大响应处理器（用于测试压缩）
func LargeResponseHandler(w http.ResponseWriter, r *http.Request) {
	// 生成大量文本
	text := ""
	for i := 0; i < 1000; i++ {
		text += fmt.Sprintf("Line %d: This is a test line for compression middleware.\n", i)
	}
	fmt.Fprint(w, text)
}

func main() {
	fmt.Println("=== HTTP 中间件系统 ===")
	fmt.Println()

	// 配置日志
	log.SetFlags(log.Ltime)

	// 配置参数
	const (
		authToken     = "secret-token"
		rateLimit     = 10
		rateLimitTime = time.Minute
		requestTimeout = 2 * time.Second
	)

	// 允许的跨域来源
	allowedOrigins := []string{
		"http://localhost:3000",
		"http://localhost:8080",
	}

	// 路由 1: /api/users - 完整的中间件链
	// 包含：恢复、日志、追踪、CORS、认证、限流、压缩
	usersHandler := Chain(
		http.HandlerFunc(UsersHandler),
		RecoveryMiddleware(),
		LoggingMiddleware(),
		TracingMiddleware(),
		CORSMiddleware(allowedOrigins),
		AuthMiddleware(authToken),
		RateLimitMiddleware(rateLimit, rateLimitTime),
		CompressionMiddleware(),
	)
	http.Handle("/api/users", usersHandler)

	// 路由 2: /api/products - 类似的中间件链
	productsHandler := Chain(
		http.HandlerFunc(ProductsHandler),
		RecoveryMiddleware(),
		LoggingMiddleware(),
		TracingMiddleware(),
		AuthMiddleware(authToken),
		RateLimitMiddleware(rateLimit, rateLimitTime),
	)
	http.Handle("/api/products", productsHandler)

	// 路由 3: /health - 仅日志中间件（健康检查不需要认证）
	healthHandler := Chain(
		http.HandlerFunc(HealthHandler),
		LoggingMiddleware(),
	)
	http.Handle("/health", healthHandler)

	// 路由 4: /slow - 测试超时中间件
	slowHandler := Chain(
		http.HandlerFunc(SlowHandler),
		RecoveryMiddleware(),
		LoggingMiddleware(),
		TimeoutMiddleware(requestTimeout),
	)
	http.Handle("/slow", slowHandler)

	// 路由 5: /panic - 测试恢复中间件
	panicHandler := Chain(
		http.HandlerFunc(PanicHandler),
		RecoveryMiddleware(),
		LoggingMiddleware(),
	)
	http.Handle("/panic", panicHandler)

	// 路由 6: /large - 测试压缩中间件
	largeHandler := Chain(
		http.HandlerFunc(LargeResponseHandler),
		LoggingMiddleware(),
		CompressionMiddleware(),
	)
	http.Handle("/large", largeHandler)

	// 打印使用说明
	fmt.Println("服务器启动在 :8080")
	fmt.Println()
	fmt.Println("可用的路由:")
	fmt.Println("  GET /api/users     - 用户列表（需要认证，有限流）")
	fmt.Println("  GET /api/products  - 产品列表（需要认证，有限流）")
	fmt.Println("  GET /health        - 健康检查（无需认证）")
	fmt.Println("  GET /slow          - 慢速响应（测试超时，2秒超时）")
	fmt.Println("  GET /panic         - 触发 panic（测试恢复）")
	fmt.Println("  GET /large         - 大响应（测试压缩）")

	fmt.Println()
	fmt.Println("测试命令:")
	fmt.Println("  # 成功请求")
	fmt.Println("  curl -H \"Authorization: Bearer secret-token\" http://localhost:8080/api/users")
	fmt.Println()
	fmt.Println("  # 认证失败")
	fmt.Println("  curl http://localhost:8080/api/users")
	fmt.Println()
	fmt.Println("  # 测试限流（快速发送多个请求）")
	fmt.Println("  for i in {1..15}; do curl -H \"Authorization: Bearer secret-token\" http://localhost:8080/api/users; done")
	fmt.Println()
	fmt.Println("  # 健康检查")
	fmt.Println("  curl http://localhost:8080/health")
	fmt.Println()
	fmt.Println("  # 测试超时")
	fmt.Println("  curl http://localhost:8080/slow")
	fmt.Println()
	fmt.Println("  # 测试恢复")
	fmt.Println("  curl http://localhost:8080/panic")
	fmt.Println()
	fmt.Println("  # 测试压缩")
	fmt.Println("  curl -H \"Accept-Encoding: gzip\" http://localhost:8080/large | gunzip")
	fmt.Println()
	fmt.Println("  # 测试 CORS")
	fmt.Println("  curl -H \"Origin: http://localhost:3000\" -H \"Authorization: Bearer secret-token\" http://localhost:8080/api/users -v")

	fmt.Println()
	fmt.Println("按 Ctrl+C 停止服务器")
	fmt.Println()
	fmt.Println("=== 服务器日志 ===")

	// 启动服务器
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}
