package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// HTTP 请求构建器示例
// 展示如何使用建造者模式构建复杂的 HTTP 请求

// HTTPRequest 表示一个 HTTP 请求
type HTTPRequest struct {
	method  string
	url     string
	headers map[string]string
	query   map[string]string
	body    []byte
	timeout time.Duration
}

// HTTPRequestBuilder 是 HTTP 请求构建器
type HTTPRequestBuilder struct {
	request *HTTPRequest
}

// NewHTTPRequestBuilder 创建一个新的 HTTP 请求构建器
func NewHTTPRequestBuilder() *HTTPRequestBuilder {
	return &HTTPRequestBuilder{
		request: &HTTPRequest{
			method:  "GET",
			headers: make(map[string]string),
			query:   make(map[string]string),
			timeout: 30 * time.Second,
		},
	}
}

// Method 设置 HTTP 方法
func (b *HTTPRequestBuilder) Method(method string) *HTTPRequestBuilder {
	b.request.method = strings.ToUpper(method)
	return b
}

// GET 设置为 GET 请求
func (b *HTTPRequestBuilder) GET() *HTTPRequestBuilder {
	return b.Method("GET")
}

// POST 设置为 POST 请求
func (b *HTTPRequestBuilder) POST() *HTTPRequestBuilder {
	return b.Method("POST")
}

// PUT 设置为 PUT 请求
func (b *HTTPRequestBuilder) PUT() *HTTPRequestBuilder {
	return b.Method("PUT")
}

// DELETE 设置为 DELETE 请求
func (b *HTTPRequestBuilder) DELETE() *HTTPRequestBuilder {
	return b.Method("DELETE")
}

// URL 设置请求 URL
func (b *HTTPRequestBuilder) URL(url string) *HTTPRequestBuilder {
	b.request.url = url
	return b
}

// Header 添加请求头
func (b *HTTPRequestBuilder) Header(key, value string) *HTTPRequestBuilder {
	b.request.headers[key] = value
	return b
}

// Headers 批量添加请求头
func (b *HTTPRequestBuilder) Headers(headers map[string]string) *HTTPRequestBuilder {
	for k, v := range headers {
		b.request.headers[k] = v
	}
	return b
}

// Query 添加查询参数
func (b *HTTPRequestBuilder) Query(key, value string) *HTTPRequestBuilder {
	b.request.query[key] = value
	return b
}

// Queries 批量添加查询参数
func (b *HTTPRequestBuilder) Queries(queries map[string]string) *HTTPRequestBuilder {
	for k, v := range queries {
		b.request.query[k] = v
	}
	return b
}

// Body 设置请求体（字节数组）
func (b *HTTPRequestBuilder) Body(body []byte) *HTTPRequestBuilder {
	b.request.body = body
	return b
}

// BodyString 设置请求体（字符串）
func (b *HTTPRequestBuilder) BodyString(body string) *HTTPRequestBuilder {
	b.request.body = []byte(body)
	return b
}

// BodyJSON 设置 JSON 请求体
func (b *HTTPRequestBuilder) BodyJSON(data interface{}) *HTTPRequestBuilder {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return b
	}
	b.request.body = jsonData
	b.Header("Content-Type", "application/json")
	return b
}

// Timeout 设置超时时间
func (b *HTTPRequestBuilder) Timeout(timeout time.Duration) *HTTPRequestBuilder {
	b.request.timeout = timeout
	return b
}

// ContentType 设置 Content-Type
func (b *HTTPRequestBuilder) ContentType(contentType string) *HTTPRequestBuilder {
	return b.Header("Content-Type", contentType)
}

// Authorization 设置 Authorization 头
func (b *HTTPRequestBuilder) Authorization(token string) *HTTPRequestBuilder {
	return b.Header("Authorization", token)
}

// BearerToken 设置 Bearer Token
func (b *HTTPRequestBuilder) BearerToken(token string) *HTTPRequestBuilder {
	return b.Header("Authorization", "Bearer "+token)
}

// BasicAuth 设置基本认证
func (b *HTTPRequestBuilder) BasicAuth(username, password string) *HTTPRequestBuilder {
	auth := username + ":" + password
	return b.Header("Authorization", "Basic "+auth)
}

// Build 构建最终的 HTTP 请求
func (b *HTTPRequestBuilder) Build() (*http.Request, error) {
	// 构建完整的 URL（包含查询参数）
	url := b.request.url
	if len(b.request.query) > 0 {
		queryParts := make([]string, 0, len(b.request.query))
		for k, v := range b.request.query {
			queryParts = append(queryParts, fmt.Sprintf("%s=%s", k, v))
		}
		url += "?" + strings.Join(queryParts, "&")
	}

	// 创建请求体
	var bodyReader io.Reader
	if len(b.request.body) > 0 {
		bodyReader = bytes.NewReader(b.request.body)
	}

	// 创建 HTTP 请求
	req, err := http.NewRequest(b.request.method, url, bodyReader)
	if err != nil {
		return nil, err
	}

	// 设置请求头
	for k, v := range b.request.headers {
		req.Header.Set(k, v)
	}

	return req, nil
}

// Execute 执行请求（仅用于演示）
func (b *HTTPRequestBuilder) Execute() (*http.Response, error) {
	req, err := b.Build()
	if err != nil {
		return nil, err
	}

	client := &http.Client{
		Timeout: b.request.timeout,
	}

	return client.Do(req)
}

// String 返回请求的字符串表示（用于调试）
func (b *HTTPRequestBuilder) String() string {
	url := b.request.url
	if len(b.request.query) > 0 {
		queryParts := make([]string, 0, len(b.request.query))
		for k, v := range b.request.query {
			queryParts = append(queryParts, fmt.Sprintf("%s=%s", k, v))
		}
		url += "?" + strings.Join(queryParts, "&")
	}

	result := fmt.Sprintf("%s %s\n", b.request.method, url)

	if len(b.request.headers) > 0 {
		result += "Headers:\n"
		for k, v := range b.request.headers {
			result += fmt.Sprintf("  %s: %s\n", k, v)
		}
	}

	if len(b.request.body) > 0 {
		result += fmt.Sprintf("Body: %s\n", string(b.request.body))
	}

	result += fmt.Sprintf("Timeout: %v\n", b.request.timeout)

	return result
}

func main() {
	fmt.Println("=== HTTP 请求构建器示例 ===\n")

	// 示例 1: 简单的 GET 请求
	fmt.Println("示例 1: 简单的 GET 请求")
	req1 := NewHTTPRequestBuilder().
		GET().
		URL("https://api.example.com/users").
		String()
	fmt.Println(req1)

	// 示例 2: 带查询参数的 GET 请求
	fmt.Println("示例 2: 带查询参数的 GET 请求")
	req2 := NewHTTPRequestBuilder().
		GET().
		URL("https://api.example.com/users").
		Query("page", "1").
		Query("limit", "10").
		Query("status", "active").
		String()
	fmt.Println(req2)

	// 示例 3: 带认证的 GET 请求
	fmt.Println("示例 3: 带认证的 GET 请求")
	req3 := NewHTTPRequestBuilder().
		GET().
		URL("https://api.example.com/profile").
		BearerToken("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...").
		String()
	fmt.Println(req3)

	// 示例 4: POST JSON 请求
	fmt.Println("示例 4: POST JSON 请求")
	userData := map[string]interface{}{
		"name":  "John Doe",
		"email": "john@example.com",
		"age":   30,
	}
	req4 := NewHTTPRequestBuilder().
		POST().
		URL("https://api.example.com/users").
		BodyJSON(userData).
		BearerToken("your-token-here").
		String()
	fmt.Println(req4)

	// 示例 5: PUT 请求
	fmt.Println("示例 5: PUT 请求")
	updateData := map[string]interface{}{
		"status": "inactive",
	}
	req5 := NewHTTPRequestBuilder().
		PUT().
		URL("https://api.example.com/users/123").
		BodyJSON(updateData).
		BearerToken("your-token-here").
		String()
	fmt.Println(req5)

	// 示例 6: DELETE 请求
	fmt.Println("示例 6: DELETE 请求")
	req6 := NewHTTPRequestBuilder().
		DELETE().
		URL("https://api.example.com/users/123").
		BearerToken("your-token-here").
		String()
	fmt.Println(req6)

	// 示例 7: 自定义请求头和超时
	fmt.Println("示例 7: 自定义请求头和超时")
	req7 := NewHTTPRequestBuilder().
		POST().
		URL("https://api.example.com/data").
		Header("X-Custom-Header", "custom-value").
		Header("X-Request-ID", "12345").
		ContentType("application/json").
		BodyString(`{"key": "value"}`).
		Timeout(60*time.Second).
		String()
	fmt.Println(req7)

	// 示例 8: 批量设置请求头和查询参数
	fmt.Println("示例 8: 批量设置请求头和查询参数")
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
	req8 := NewHTTPRequestBuilder().
		GET().
		URL("https://api.example.com/items").
		Headers(headers).
		Queries(queries).
		String()
	fmt.Println(req8)

	fmt.Println("=== 示例结束 ===")
	fmt.Println("\n建造者模式的优势:")
	fmt.Println("✅ 链式调用，代码优雅")
	fmt.Println("✅ 参数清晰，易于理解")
	fmt.Println("✅ 灵活配置，按需设置")
	fmt.Println("✅ 易于扩展，添加新功能简单")
}
