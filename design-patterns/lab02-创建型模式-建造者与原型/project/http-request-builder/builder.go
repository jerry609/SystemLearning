package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// HTTPClient 是 HTTP 客户端
type HTTPClient struct {
	client *http.Client
}

// NewHTTPClient 创建一个新的 HTTP 客户端
func NewHTTPClient(opts ...ClientOption) *HTTPClient {
	client := &HTTPClient{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}

	for _, opt := range opts {
		opt(client)
	}

	return client
}

// ClientOption 是客户端配置选项
type ClientOption func(*HTTPClient)

// WithTimeout 设置客户端超时时间
func WithTimeout(timeout time.Duration) ClientOption {
	return func(c *HTTPClient) {
		c.client.Timeout = timeout
	}
}

// WithTransport 设置自定义 Transport
func WithTransport(transport *http.Transport) ClientOption {
	return func(c *HTTPClient) {
		c.client.Transport = transport
	}
}

// Request 表示一个 HTTP 请求的配置
type Request struct {
	method  string
	url     string
	headers map[string]string
	query   map[string]string
	body    []byte
	timeout time.Duration
}

// RequestBuilder 是 HTTP 请求构建器
type RequestBuilder struct {
	request *Request
	client  *HTTPClient
}

// NewRequest 创建一个新的请求构建器
func (c *HTTPClient) NewRequest() *RequestBuilder {
	return &RequestBuilder{
		request: &Request{
			method:  "GET",
			headers: make(map[string]string),
			query:   make(map[string]string),
			timeout: c.client.Timeout,
		},
		client: c,
	}
}

// Method 设置 HTTP 方法
func (b *RequestBuilder) Method(method string) *RequestBuilder {
	b.request.method = strings.ToUpper(method)
	return b
}

// GET 设置为 GET 请求
func (b *RequestBuilder) GET() *RequestBuilder {
	return b.Method("GET")
}

// POST 设置为 POST 请求
func (b *RequestBuilder) POST() *RequestBuilder {
	return b.Method("POST")
}

// PUT 设置为 PUT 请求
func (b *RequestBuilder) PUT() *RequestBuilder {
	return b.Method("PUT")
}

// DELETE 设置为 DELETE 请求
func (b *RequestBuilder) DELETE() *RequestBuilder {
	return b.Method("DELETE")
}

// PATCH 设置为 PATCH 请求
func (b *RequestBuilder) PATCH() *RequestBuilder {
	return b.Method("PATCH")
}

// HEAD 设置为 HEAD 请求
func (b *RequestBuilder) HEAD() *RequestBuilder {
	return b.Method("HEAD")
}

// OPTIONS 设置为 OPTIONS 请求
func (b *RequestBuilder) OPTIONS() *RequestBuilder {
	return b.Method("OPTIONS")
}

// URL 设置请求 URL
func (b *RequestBuilder) URL(urlStr string) *RequestBuilder {
	b.request.url = urlStr
	return b
}

// Header 添加请求头
func (b *RequestBuilder) Header(key, value string) *RequestBuilder {
	b.request.headers[key] = value
	return b
}

// Headers 批量添加请求头
func (b *RequestBuilder) Headers(headers map[string]string) *RequestBuilder {
	for k, v := range headers {
		b.request.headers[k] = v
	}
	return b
}

// Query 添加查询参数
func (b *RequestBuilder) Query(key, value string) *RequestBuilder {
	b.request.query[key] = value
	return b
}

// Queries 批量添加查询参数
func (b *RequestBuilder) Queries(queries map[string]string) *RequestBuilder {
	for k, v := range queries {
		b.request.query[k] = v
	}
	return b
}

// Body 设置请求体（字节数组）
func (b *RequestBuilder) Body(body []byte) *RequestBuilder {
	b.request.body = body
	return b
}

// BodyString 设置请求体（字符串）
func (b *RequestBuilder) BodyString(body string) *RequestBuilder {
	b.request.body = []byte(body)
	return b
}

// BodyJSON 设置 JSON 请求体
func (b *RequestBuilder) BodyJSON(data interface{}) *RequestBuilder {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return b
	}
	b.request.body = jsonData
	b.Header("Content-Type", "application/json")
	return b
}

// BodyForm 设置表单请求体
func (b *RequestBuilder) BodyForm(data map[string]string) *RequestBuilder {
	form := url.Values{}
	for k, v := range data {
		form.Set(k, v)
	}
	b.request.body = []byte(form.Encode())
	b.Header("Content-Type", "application/x-www-form-urlencoded")
	return b
}

// Timeout 设置超时时间
func (b *RequestBuilder) Timeout(timeout time.Duration) *RequestBuilder {
	b.request.timeout = timeout
	return b
}

// ContentType 设置 Content-Type
func (b *RequestBuilder) ContentType(contentType string) *RequestBuilder {
	return b.Header("Content-Type", contentType)
}

// Accept 设置 Accept 头
func (b *RequestBuilder) Accept(accept string) *RequestBuilder {
	return b.Header("Accept", accept)
}

// Authorization 设置 Authorization 头
func (b *RequestBuilder) Authorization(token string) *RequestBuilder {
	return b.Header("Authorization", token)
}

// BearerToken 设置 Bearer Token
func (b *RequestBuilder) BearerToken(token string) *RequestBuilder {
	return b.Header("Authorization", "Bearer "+token)
}

// BasicAuth 设置基本认证
func (b *RequestBuilder) BasicAuth(username, password string) *RequestBuilder {
	auth := username + ":" + password
	return b.Header("Authorization", "Basic "+auth)
}

// UserAgent 设置 User-Agent
func (b *RequestBuilder) UserAgent(userAgent string) *RequestBuilder {
	return b.Header("User-Agent", userAgent)
}

// Build 构建最终的 HTTP 请求
func (b *RequestBuilder) Build() (*http.Request, error) {
	// 构建完整的 URL（包含查询参数）
	urlStr := b.request.url
	if len(b.request.query) > 0 {
		u, err := url.Parse(urlStr)
		if err != nil {
			return nil, fmt.Errorf("invalid URL: %w", err)
		}

		q := u.Query()
		for k, v := range b.request.query {
			q.Set(k, v)
		}
		u.RawQuery = q.Encode()
		urlStr = u.String()
	}

	// 创建请求体
	var bodyReader io.Reader
	if len(b.request.body) > 0 {
		bodyReader = bytes.NewReader(b.request.body)
	}

	// 创建 HTTP 请求
	req, err := http.NewRequest(b.request.method, urlStr, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// 设置请求头
	for k, v := range b.request.headers {
		req.Header.Set(k, v)
	}

	// 设置默认的 Accept 头
	if req.Header.Get("Accept") == "" {
		req.Header.Set("Accept", "application/json")
	}

	return req, nil
}

// Execute 执行请求
func (b *RequestBuilder) Execute() (*http.Response, error) {
	req, err := b.Build()
	if err != nil {
		return nil, err
	}

	return b.client.Do(req)
}

// Do 执行 HTTP 请求
func (c *HTTPClient) Do(req *http.Request) (*http.Response, error) {
	return c.client.Do(req)
}

// Get 快捷 GET 请求
func (c *HTTPClient) Get(urlStr string) (*http.Response, error) {
	return c.NewRequest().GET().URL(urlStr).Execute()
}

// Post 快捷 POST 请求
func (c *HTTPClient) Post(urlStr string, body interface{}) (*http.Response, error) {
	return c.NewRequest().POST().URL(urlStr).BodyJSON(body).Execute()
}

// String 返回请求的字符串表示（用于调试）
func (b *RequestBuilder) String() string {
	urlStr := b.request.url
	if len(b.request.query) > 0 {
		u, _ := url.Parse(urlStr)
		q := u.Query()
		for k, v := range b.request.query {
			q.Set(k, v)
		}
		u.RawQuery = q.Encode()
		urlStr = u.String()
	}

	result := fmt.Sprintf("%s %s\n", b.request.method, urlStr)

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
	fmt.Println("=== HTTP 请求构建器实战项目 ===")
	fmt.Println()

	// 创建 HTTP 客户端
	client := NewHTTPClient()

	// 示例 1: 简单的 GET 请求
	fmt.Println("示例 1: 简单的 GET 请求")
	req1 := client.NewRequest().
		GET().
		URL("https://api.example.com/users").
		Query("page", "1").
		Query("limit", "10").
		String()
	fmt.Println(req1)

	// 示例 2: POST JSON 请求
	fmt.Println("示例 2: POST JSON 请求")
	userData := map[string]interface{}{
		"name":  "John Doe",
		"email": "john@example.com",
		"age":   30,
	}
	req2 := client.NewRequest().
		POST().
		URL("https://api.example.com/users").
		BodyJSON(userData).
		BearerToken("your-token-here").
		String()
	fmt.Println(req2)

	// 示例 3: 带认证和自定义头的请求
	fmt.Println("示例 3: 带认证和自定义头的请求")
	req3 := client.NewRequest().
		GET().
		URL("https://api.example.com/profile").
		BearerToken("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...").
		Header("X-Request-ID", "12345").
		Accept("application/json").
		Timeout(60*time.Second).
		String()
	fmt.Println(req3)

	// 示例 4: PUT 请求
	fmt.Println("示例 4: PUT 请求")
	updateData := map[string]interface{}{
		"status": "inactive",
	}
	req4 := client.NewRequest().
		PUT().
		URL("https://api.example.com/users/123").
		BodyJSON(updateData).
		BearerToken("your-token-here").
		String()
	fmt.Println(req4)

	// 示例 5: DELETE 请求
	fmt.Println("示例 5: DELETE 请求")
	req5 := client.NewRequest().
		DELETE().
		URL("https://api.example.com/users/123").
		BearerToken("your-token-here").
		String()
	fmt.Println(req5)

	// 示例 6: 表单请求
	fmt.Println("示例 6: 表单请求")
	formData := map[string]string{
		"username": "john",
		"password": "secret",
	}
	req6 := client.NewRequest().
		POST().
		URL("https://api.example.com/login").
		BodyForm(formData).
		String()
	fmt.Println(req6)

	// 示例 7: 批量设置请求头和查询参数
	fmt.Println("示例 7: 批量设置请求头和查询参数")
	headers := map[string]string{
		"X-API-Key":    "api-key-123",
		"X-Request-ID": "req-456",
	}
	queries := map[string]string{
		"sort":   "created_at",
		"order":  "desc",
		"filter": "active",
	}
	req7 := client.NewRequest().
		GET().
		URL("https://api.example.com/items").
		Headers(headers).
		Queries(queries).
		String()
	fmt.Println(req7)

	fmt.Println("=== 示例结束 ===")
	fmt.Println()
	fmt.Println("建造者模式的优势:")
	fmt.Println("✅ 链式调用，代码优雅")
	fmt.Println("✅ 参数清晰，易于理解")
	fmt.Println("✅ 灵活配置，按需设置")
	fmt.Println("✅ 易于扩展，添加新功能简单")
}
