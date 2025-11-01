package main

import (
	"io"
	"strings"
	"testing"
	"time"
)

// TestNewHTTPClient 测试创建 HTTP 客户端
func TestNewHTTPClient(t *testing.T) {
	client := NewHTTPClient()
	if client == nil {
		t.Fatal("Expected client to be created")
	}
	if client.client == nil {
		t.Fatal("Expected http.Client to be initialized")
	}
}

// TestNewHTTPClientWithOptions 测试使用选项创建客户端
func TestNewHTTPClientWithOptions(t *testing.T) {
	timeout := 60 * time.Second
	client := NewHTTPClient(WithTimeout(timeout))

	if client.client.Timeout != timeout {
		t.Errorf("Expected timeout %v, got %v", timeout, client.client.Timeout)
	}
}

// TestRequestBuilderGET 测试 GET 请求构建
func TestRequestBuilderGET(t *testing.T) {
	client := NewHTTPClient()
	req, err := client.NewRequest().
		GET().
		URL("https://api.example.com/users").
		Build()

	if err != nil {
		t.Fatalf("Failed to build request: %v", err)
	}

	if req.Method != "GET" {
		t.Errorf("Expected method GET, got %s", req.Method)
	}

	if req.URL.String() != "https://api.example.com/users" {
		t.Errorf("Expected URL https://api.example.com/users, got %s", req.URL.String())
	}
}

// TestRequestBuilderPOST 测试 POST 请求构建
func TestRequestBuilderPOST(t *testing.T) {
	client := NewHTTPClient()
	req, err := client.NewRequest().
		POST().
		URL("https://api.example.com/users").
		Build()

	if err != nil {
		t.Fatalf("Failed to build request: %v", err)
	}

	if req.Method != "POST" {
		t.Errorf("Expected method POST, got %s", req.Method)
	}
}

// TestRequestBuilderHeaders 测试设置请求头
func TestRequestBuilderHeaders(t *testing.T) {
	client := NewHTTPClient()
	req, err := client.NewRequest().
		GET().
		URL("https://api.example.com/users").
		Header("X-Custom-Header", "custom-value").
		Header("X-Request-ID", "12345").
		Build()

	if err != nil {
		t.Fatalf("Failed to build request: %v", err)
	}

	if req.Header.Get("X-Custom-Header") != "custom-value" {
		t.Errorf("Expected X-Custom-Header to be custom-value, got %s", req.Header.Get("X-Custom-Header"))
	}

	if req.Header.Get("X-Request-ID") != "12345" {
		t.Errorf("Expected X-Request-ID to be 12345, got %s", req.Header.Get("X-Request-ID"))
	}
}

// TestRequestBuilderBatchHeaders 测试批量设置请求头
func TestRequestBuilderBatchHeaders(t *testing.T) {
	client := NewHTTPClient()
	headers := map[string]string{
		"X-API-Key":    "api-key-123",
		"X-Request-ID": "req-456",
	}

	req, err := client.NewRequest().
		GET().
		URL("https://api.example.com/users").
		Headers(headers).
		Build()

	if err != nil {
		t.Fatalf("Failed to build request: %v", err)
	}

	for k, v := range headers {
		if req.Header.Get(k) != v {
			t.Errorf("Expected header %s to be %s, got %s", k, v, req.Header.Get(k))
		}
	}
}

// TestRequestBuilderQuery 测试查询参数
func TestRequestBuilderQuery(t *testing.T) {
	client := NewHTTPClient()
	req, err := client.NewRequest().
		GET().
		URL("https://api.example.com/users").
		Query("page", "1").
		Query("limit", "10").
		Build()

	if err != nil {
		t.Fatalf("Failed to build request: %v", err)
	}

	query := req.URL.Query()
	if query.Get("page") != "1" {
		t.Errorf("Expected page to be 1, got %s", query.Get("page"))
	}

	if query.Get("limit") != "10" {
		t.Errorf("Expected limit to be 10, got %s", query.Get("limit"))
	}
}

// TestRequestBuilderBatchQueries 测试批量设置查询参数
func TestRequestBuilderBatchQueries(t *testing.T) {
	client := NewHTTPClient()
	queries := map[string]string{
		"sort":   "created_at",
		"order":  "desc",
		"filter": "active",
	}

	req, err := client.NewRequest().
		GET().
		URL("https://api.example.com/users").
		Queries(queries).
		Build()

	if err != nil {
		t.Fatalf("Failed to build request: %v", err)
	}

	query := req.URL.Query()
	for k, v := range queries {
		if query.Get(k) != v {
			t.Errorf("Expected query %s to be %s, got %s", k, v, query.Get(k))
		}
	}
}

// TestRequestBuilderBodyJSON 测试 JSON 请求体
func TestRequestBuilderBodyJSON(t *testing.T) {
	client := NewHTTPClient()
	userData := map[string]interface{}{
		"name":  "John Doe",
		"email": "john@example.com",
	}

	req, err := client.NewRequest().
		POST().
		URL("https://api.example.com/users").
		BodyJSON(userData).
		Build()

	if err != nil {
		t.Fatalf("Failed to build request: %v", err)
	}

	if req.Header.Get("Content-Type") != "application/json" {
		t.Errorf("Expected Content-Type to be application/json, got %s", req.Header.Get("Content-Type"))
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("Failed to read body: %v", err)
	}

	bodyStr := string(body)
	if !strings.Contains(bodyStr, "John Doe") {
		t.Errorf("Expected body to contain 'John Doe', got %s", bodyStr)
	}
}

// TestRequestBuilderBodyForm 测试表单请求体
func TestRequestBuilderBodyForm(t *testing.T) {
	client := NewHTTPClient()
	formData := map[string]string{
		"username": "john",
		"password": "secret",
	}

	req, err := client.NewRequest().
		POST().
		URL("https://api.example.com/login").
		BodyForm(formData).
		Build()

	if err != nil {
		t.Fatalf("Failed to build request: %v", err)
	}

	if req.Header.Get("Content-Type") != "application/x-www-form-urlencoded" {
		t.Errorf("Expected Content-Type to be application/x-www-form-urlencoded, got %s", req.Header.Get("Content-Type"))
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		t.Fatalf("Failed to read body: %v", err)
	}

	bodyStr := string(body)
	if !strings.Contains(bodyStr, "username=john") {
		t.Errorf("Expected body to contain 'username=john', got %s", bodyStr)
	}
}

// TestRequestBuilderBearerToken 测试 Bearer Token
func TestRequestBuilderBearerToken(t *testing.T) {
	client := NewHTTPClient()
	token := "test-token-123"

	req, err := client.NewRequest().
		GET().
		URL("https://api.example.com/profile").
		BearerToken(token).
		Build()

	if err != nil {
		t.Fatalf("Failed to build request: %v", err)
	}

	expectedAuth := "Bearer " + token
	if req.Header.Get("Authorization") != expectedAuth {
		t.Errorf("Expected Authorization to be %s, got %s", expectedAuth, req.Header.Get("Authorization"))
	}
}

// TestRequestBuilderBasicAuth 测试基本认证
func TestRequestBuilderBasicAuth(t *testing.T) {
	client := NewHTTPClient()

	req, err := client.NewRequest().
		GET().
		URL("https://api.example.com/profile").
		BasicAuth("user", "pass").
		Build()

	if err != nil {
		t.Fatalf("Failed to build request: %v", err)
	}

	expectedAuth := "Basic user:pass"
	if req.Header.Get("Authorization") != expectedAuth {
		t.Errorf("Expected Authorization to be %s, got %s", expectedAuth, req.Header.Get("Authorization"))
	}
}

// TestRequestBuilderAllMethods 测试所有 HTTP 方法
func TestRequestBuilderAllMethods(t *testing.T) {
	client := NewHTTPClient()
	methods := []struct {
		builder func(*RequestBuilder) *RequestBuilder
		method  string
	}{
		{(*RequestBuilder).GET, "GET"},
		{(*RequestBuilder).POST, "POST"},
		{(*RequestBuilder).PUT, "PUT"},
		{(*RequestBuilder).DELETE, "DELETE"},
		{(*RequestBuilder).PATCH, "PATCH"},
		{(*RequestBuilder).HEAD, "HEAD"},
		{(*RequestBuilder).OPTIONS, "OPTIONS"},
	}

	for _, m := range methods {
		req, err := m.builder(client.NewRequest()).
			URL("https://api.example.com/test").
			Build()

		if err != nil {
			t.Fatalf("Failed to build %s request: %v", m.method, err)
		}

		if req.Method != m.method {
			t.Errorf("Expected method %s, got %s", m.method, req.Method)
		}
	}
}

// TestRequestBuilderChaining 测试链式调用
func TestRequestBuilderChaining(t *testing.T) {
	client := NewHTTPClient()

	req, err := client.NewRequest().
		POST().
		URL("https://api.example.com/users").
		Header("X-Custom", "value").
		Query("page", "1").
		BodyJSON(map[string]string{"name": "John"}).
		BearerToken("token").
		Timeout(60*time.Second).
		Build()

	if err != nil {
		t.Fatalf("Failed to build request: %v", err)
	}

	if req.Method != "POST" {
		t.Errorf("Expected method POST, got %s", req.Method)
	}

	if req.Header.Get("X-Custom") != "value" {
		t.Error("Expected X-Custom header to be set")
	}

	if req.URL.Query().Get("page") != "1" {
		t.Error("Expected page query parameter to be set")
	}

	if req.Header.Get("Authorization") != "Bearer token" {
		t.Error("Expected Authorization header to be set")
	}
}

// TestRequestBuilderInvalidURL 测试无效 URL
func TestRequestBuilderInvalidURL(t *testing.T) {
	client := NewHTTPClient()

	_, err := client.NewRequest().
		GET().
		URL("://invalid-url").
		Build()

	if err == nil {
		t.Error("Expected error for invalid URL, got nil")
	}
}

// TestRequestBuilderDefaultAcceptHeader 测试默认 Accept 头
func TestRequestBuilderDefaultAcceptHeader(t *testing.T) {
	client := NewHTTPClient()

	req, err := client.NewRequest().
		GET().
		URL("https://api.example.com/users").
		Build()

	if err != nil {
		t.Fatalf("Failed to build request: %v", err)
	}

	if req.Header.Get("Accept") != "application/json" {
		t.Errorf("Expected default Accept header to be application/json, got %s", req.Header.Get("Accept"))
	}
}

// TestRequestBuilderString 测试 String 方法
func TestRequestBuilderString(t *testing.T) {
	client := NewHTTPClient()

	str := client.NewRequest().
		GET().
		URL("https://api.example.com/users").
		Query("page", "1").
		Header("X-Custom", "value").
		String()

	if !strings.Contains(str, "GET") {
		t.Error("Expected string to contain GET")
	}

	if !strings.Contains(str, "https://api.example.com/users") {
		t.Error("Expected string to contain URL")
	}

	if !strings.Contains(str, "X-Custom") {
		t.Error("Expected string to contain header")
	}
}

// BenchmarkRequestBuilder 性能基准测试
func BenchmarkRequestBuilder(b *testing.B) {
	client := NewHTTPClient()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.NewRequest().
			POST().
			URL("https://api.example.com/users").
			Header("X-Custom", "value").
			Query("page", "1").
			BodyJSON(map[string]string{"name": "John"}).
			Build()
	}
}
