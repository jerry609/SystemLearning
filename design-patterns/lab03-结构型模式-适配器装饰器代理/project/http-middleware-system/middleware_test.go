package main

import (
	"compress/gzip"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// 测试用的简单处理器
func testHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("test response"))
}

// TestResponseWriter 测试响应包装器
func TestResponseWriter(t *testing.T) {
	w := httptest.NewRecorder()
	rw := NewResponseWriter(w)

	// 测试默认状态码
	if rw.StatusCode != http.StatusOK {
		t.Errorf("Expected default status code 200, got %d", rw.StatusCode)
	}

	// 测试 WriteHeader
	rw.WriteHeader(http.StatusCreated)
	if rw.StatusCode != http.StatusCreated {
		t.Errorf("Expected status code 201, got %d", rw.StatusCode)
	}

	// 测试 Write
	data := []byte("test data")
	n, err := rw.Write(data)
	if err != nil {
		t.Errorf("Write failed: %v", err)
	}
	if n != len(data) {
		t.Errorf("Expected to write %d bytes, wrote %d", len(data), n)
	}
	if rw.BytesWritten != len(data) {
		t.Errorf("Expected BytesWritten %d, got %d", len(data), rw.BytesWritten)
	}
}

// TestLoggingMiddleware 测试日志中间件
func TestLoggingMiddleware(t *testing.T) {
	handler := LoggingMiddleware()(http.HandlerFunc(testHandler))

	req := httptest.NewRequest("GET", "/test", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status code 200, got %d", w.Code)
	}

	body := w.Body.String()
	if body != "test response" {
		t.Errorf("Expected body 'test response', got '%s'", body)
	}
}

// TestAuthMiddleware 测试认证中间件
func TestAuthMiddleware(t *testing.T) {
	validToken := "secret-token"
	handler := AuthMiddleware(validToken)(http.HandlerFunc(testHandler))

	tests := []struct {
		name           string
		authHeader     string
		expectedStatus int
	}{
		{
			name:           "Valid token",
			authHeader:     "Bearer secret-token",
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Invalid token",
			authHeader:     "Bearer wrong-token",
			expectedStatus: http.StatusUnauthorized,
		},
		{
			name:           "Missing token",
			authHeader:     "",
			expectedStatus: http.StatusUnauthorized,
		},
		{
			name:           "Invalid format",
			authHeader:     "secret-token",
			expectedStatus: http.StatusUnauthorized,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", "/test", nil)
			if tt.authHeader != "" {
				req.Header.Set("Authorization", tt.authHeader)
			}
			w := httptest.NewRecorder()

			handler.ServeHTTP(w, req)

			if w.Code != tt.expectedStatus {
				t.Errorf("Expected status code %d, got %d", tt.expectedStatus, w.Code)
			}
		})
	}
}

// TestBasicAuthMiddleware 测试 Basic 认证中间件
func TestBasicAuthMiddleware(t *testing.T) {
	handler := BasicAuthMiddleware("admin", "password")(http.HandlerFunc(testHandler))

	tests := []struct {
		name           string
		username       string
		password       string
		expectedStatus int
	}{
		{
			name:           "Valid credentials",
			username:       "admin",
			password:       "password",
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Invalid username",
			username:       "user",
			password:       "password",
			expectedStatus: http.StatusUnauthorized,
		},
		{
			name:           "Invalid password",
			username:       "admin",
			password:       "wrong",
			expectedStatus: http.StatusUnauthorized,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", "/test", nil)
			req.SetBasicAuth(tt.username, tt.password)
			w := httptest.NewRecorder()

			handler.ServeHTTP(w, req)

			if w.Code != tt.expectedStatus {
				t.Errorf("Expected status code %d, got %d", tt.expectedStatus, w.Code)
			}
		})
	}
}

// TestRateLimitMiddleware 测试限流中间件
func TestRateLimitMiddleware(t *testing.T) {
	maxRequests := 3
	window := 100 * time.Millisecond
	handler := RateLimitMiddleware(maxRequests, window)(http.HandlerFunc(testHandler))

	req := httptest.NewRequest("GET", "/test", nil)

	// 发送 maxRequests 个请求，应该都成功
	for i := 0; i < maxRequests; i++ {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Request %d: Expected status code 200, got %d", i+1, w.Code)
		}
	}

	// 第 maxRequests+1 个请求应该被限流
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusTooManyRequests {
		t.Errorf("Expected status code 429, got %d", w.Code)
	}

	// 等待窗口重置
	time.Sleep(window + 10*time.Millisecond)

	// 窗口重置后，请求应该再次成功
	w = httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("After window reset: Expected status code 200, got %d", w.Code)
	}
}

// TestRecoveryMiddleware 测试恢复中间件
func TestRecoveryMiddleware(t *testing.T) {
	panicHandler := func(w http.ResponseWriter, r *http.Request) {
		panic("test panic")
	}

	handler := RecoveryMiddleware()(http.HandlerFunc(panicHandler))

	req := httptest.NewRequest("GET", "/test", nil)
	w := httptest.NewRecorder()

	// 不应该 panic，而是返回 500 错误
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("Expected status code 500, got %d", w.Code)
	}
}

// TestCORSMiddleware 测试 CORS 中间件
func TestCORSMiddleware(t *testing.T) {
	allowedOrigins := []string{"http://localhost:3000", "http://example.com"}
	handler := CORSMiddleware(allowedOrigins)(http.HandlerFunc(testHandler))

	tests := []struct {
		name           string
		origin         string
		method         string
		expectAllowed  bool
		expectedStatus int
	}{
		{
			name:           "Allowed origin",
			origin:         "http://localhost:3000",
			method:         "GET",
			expectAllowed:  true,
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Disallowed origin",
			origin:         "http://evil.com",
			method:         "GET",
			expectAllowed:  false,
			expectedStatus: http.StatusOK,
		},
		{
			name:           "OPTIONS request",
			origin:         "http://localhost:3000",
			method:         "OPTIONS",
			expectAllowed:  true,
			expectedStatus: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, "/test", nil)
			req.Header.Set("Origin", tt.origin)
			w := httptest.NewRecorder()

			handler.ServeHTTP(w, req)

			if w.Code != tt.expectedStatus {
				t.Errorf("Expected status code %d, got %d", tt.expectedStatus, w.Code)
			}

			if tt.expectAllowed {
				allowOrigin := w.Header().Get("Access-Control-Allow-Origin")
				if allowOrigin != tt.origin {
					t.Errorf("Expected Access-Control-Allow-Origin '%s', got '%s'", tt.origin, allowOrigin)
				}
			}
		})
	}
}

// TestTracingMiddleware 测试追踪中间件
func TestTracingMiddleware(t *testing.T) {
	handler := TracingMiddleware()(http.HandlerFunc(testHandler))

	req := httptest.NewRequest("GET", "/test", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	// 检查响应头中是否有请求 ID
	requestID := w.Header().Get("X-Request-ID")
	if requestID == "" {
		t.Error("Expected X-Request-ID header, got empty")
	}

	if !strings.HasPrefix(requestID, "req-") {
		t.Errorf("Expected request ID to start with 'req-', got '%s'", requestID)
	}
}

// TestCompressionMiddleware 测试压缩中间件
func TestCompressionMiddleware(t *testing.T) {
	largeHandler := func(w http.ResponseWriter, r *http.Request) {
		// 生成大量文本
		text := strings.Repeat("test data ", 1000)
		w.Write([]byte(text))
	}

	handler := CompressionMiddleware()(http.HandlerFunc(largeHandler))

	tests := []struct {
		name              string
		acceptEncoding    string
		expectCompression bool
	}{
		{
			name:              "With gzip support",
			acceptEncoding:    "gzip",
			expectCompression: true,
		},
		{
			name:              "Without gzip support",
			acceptEncoding:    "",
			expectCompression: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", "/test", nil)
			if tt.acceptEncoding != "" {
				req.Header.Set("Accept-Encoding", tt.acceptEncoding)
			}
			w := httptest.NewRecorder()

			handler.ServeHTTP(w, req)

			if tt.expectCompression {
				contentEncoding := w.Header().Get("Content-Encoding")
				if contentEncoding != "gzip" {
					t.Errorf("Expected Content-Encoding 'gzip', got '%s'", contentEncoding)
				}

				// 尝试解压缩
				gr, err := gzip.NewReader(w.Body)
				if err != nil {
					t.Errorf("Failed to create gzip reader: %v", err)
				}
				defer gr.Close()

				decompressed, err := io.ReadAll(gr)
				if err != nil {
					t.Errorf("Failed to decompress: %v", err)
				}

				if len(decompressed) == 0 {
					t.Error("Decompressed data is empty")
				}
			} else {
				contentEncoding := w.Header().Get("Content-Encoding")
				if contentEncoding == "gzip" {
					t.Error("Expected no compression, but got gzip")
				}
			}
		})
	}
}

// TestTimeoutMiddleware 测试超时中间件
func TestTimeoutMiddleware(t *testing.T) {
	timeout := 100 * time.Millisecond

	tests := []struct {
		name           string
		handlerDelay   time.Duration
		expectedStatus int
	}{
		{
			name:           "Fast handler",
			handlerDelay:   10 * time.Millisecond,
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Slow handler",
			handlerDelay:   200 * time.Millisecond,
			expectedStatus: http.StatusRequestTimeout,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			slowHandler := func(w http.ResponseWriter, r *http.Request) {
				time.Sleep(tt.handlerDelay)
				w.Write([]byte("response"))
			}

			handler := TimeoutMiddleware(timeout)(http.HandlerFunc(slowHandler))

			req := httptest.NewRequest("GET", "/test", nil)
			w := httptest.NewRecorder()

			handler.ServeHTTP(w, req)

			if w.Code != tt.expectedStatus {
				t.Errorf("Expected status code %d, got %d", tt.expectedStatus, w.Code)
			}
		})
	}
}

// TestChain 测试中间件链
func TestChain(t *testing.T) {
	// 创建一个记录执行顺序的处理器
	var order []string

	middleware1 := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			order = append(order, "m1-before")
			next.ServeHTTP(w, r)
			order = append(order, "m1-after")
		})
	}

	middleware2 := func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			order = append(order, "m2-before")
			next.ServeHTTP(w, r)
			order = append(order, "m2-after")
		})
	}

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		order = append(order, "handler")
		w.WriteHeader(http.StatusOK)
	})

	// 链式组合: middleware1 -> middleware2 -> handler
	chain := Chain(handler, middleware1, middleware2)

	req := httptest.NewRequest("GET", "/test", nil)
	w := httptest.NewRecorder()

	chain.ServeHTTP(w, req)

	// 验证执行顺序
	expectedOrder := []string{
		"m1-before",
		"m2-before",
		"handler",
		"m2-after",
		"m1-after",
	}

	if len(order) != len(expectedOrder) {
		t.Errorf("Expected %d steps, got %d", len(expectedOrder), len(order))
	}

	for i, step := range expectedOrder {
		if i >= len(order) || order[i] != step {
			t.Errorf("Step %d: Expected '%s', got '%s'", i, step, order[i])
		}
	}
}

// TestMultipleMiddlewares 测试多个中间件组合
func TestMultipleMiddlewares(t *testing.T) {
	handler := Chain(
		http.HandlerFunc(testHandler),
		RecoveryMiddleware(),
		LoggingMiddleware(),
		TracingMiddleware(),
		AuthMiddleware("test-token"),
	)

	// 测试成功请求
	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("Authorization", "Bearer test-token")
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status code 200, got %d", w.Code)
	}

	// 检查追踪 ID
	requestID := w.Header().Get("X-Request-ID")
	if requestID == "" {
		t.Error("Expected X-Request-ID header")
	}

	// 测试认证失败
	req2 := httptest.NewRequest("GET", "/test", nil)
	w2 := httptest.NewRecorder()

	handler.ServeHTTP(w2, req2)

	if w2.Code != http.StatusUnauthorized {
		t.Errorf("Expected status code 401, got %d", w2.Code)
	}
}

// BenchmarkLoggingMiddleware 基准测试日志中间件
func BenchmarkLoggingMiddleware(b *testing.B) {
	handler := LoggingMiddleware()(http.HandlerFunc(testHandler))
	req := httptest.NewRequest("GET", "/test", nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
	}
}

// BenchmarkAuthMiddleware 基准测试认证中间件
func BenchmarkAuthMiddleware(b *testing.B) {
	handler := AuthMiddleware("secret-token")(http.HandlerFunc(testHandler))
	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("Authorization", "Bearer secret-token")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
	}
}

// BenchmarkChain 基准测试中间件链
func BenchmarkChain(b *testing.B) {
	handler := Chain(
		http.HandlerFunc(testHandler),
		RecoveryMiddleware(),
		LoggingMiddleware(),
		TracingMiddleware(),
		AuthMiddleware("test-token"),
	)

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("Authorization", "Bearer test-token")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)
	}
}
