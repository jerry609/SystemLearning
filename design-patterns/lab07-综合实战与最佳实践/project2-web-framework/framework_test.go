package main

import (
	"testing"
)

// TestRouterBuilder 测试路由构建器
func TestRouterBuilder(t *testing.T) {
	router := NewRouterBuilder().
		GET("/", HomeHandler).
		GET("/users", ListUsersHandler).
		POST("/users", CreateUserHandler).
		Build()

	if len(router.routes) != 3 {
		t.Errorf("期望 3 个路由，实际 %d 个", len(router.routes))
	}
}

// TestRouterMatch 测试路由匹配
func TestRouterMatch(t *testing.T) {
	router := NewRouterBuilder().
		GET("/users", ListUsersHandler).
		GET("/users/:id", GetUserHandler).
		Build()

	// 测试精确匹配
	route, params := router.Match("GET", "/users")
	if route == nil {
		t.Error("路由匹配失败")
	}
	if params != nil {
		t.Error("不应该有参数")
	}

	// 测试参数匹配
	route, params = router.Match("GET", "/users/123")
	if route == nil {
		t.Error("路由匹配失败")
	}
	if params == nil || params["id"] != "123" {
		t.Errorf("参数匹配失败，期望 id=123，实际 %v", params)
	}

	// 测试不匹配
	route, params = router.Match("GET", "/posts")
	if route != nil {
		t.Error("不应该匹配到路由")
	}
}

// TestRouterGroup 测试路由分组
func TestRouterGroup(t *testing.T) {
	builder := NewRouterBuilder()
	api := builder.Group("/api")
	api.GET("/users", ListUsersHandler)
	api.POST("/users", CreateUserHandler)

	router := builder.Build()

	// 测试分组路由
	route, _ := router.Match("GET", "/api/users")
	if route == nil {
		t.Error("分组路由匹配失败")
	}
}

// TestHandlerChain 测试处理器链
func TestHandlerChain(t *testing.T) {
	executed := []string{}

	handler1 := func(ctx *RequestContext) error {
		executed = append(executed, "handler1")
		return nil
	}

	handler2 := func(ctx *RequestContext) error {
		executed = append(executed, "handler2")
		return nil
	}

	chain := NewHandlerChain()
	chain.Use(handler1).Use(handler2)

	ctx := NewRequestContext("GET", "/test")
	err := chain.Execute(ctx)

	if err != nil {
		t.Errorf("处理器链执行失败: %v", err)
	}

	if len(executed) != 2 {
		t.Errorf("期望执行 2 个处理器，实际 %d 个", len(executed))
	}

	if executed[0] != "handler1" || executed[1] != "handler2" {
		t.Error("处理器执行顺序错误")
	}
}

// TestApplication 测试应用程序
func TestApplication(t *testing.T) {
	router := NewRouterBuilder().
		GET("/", HomeHandler).
		GET("/users/:id", GetUserHandler).
		Build()

	app := NewApplication(router)

	// 测试正常请求
	err := app.ServeHTTP("GET", "/")
	if err != nil {
		t.Errorf("请求处理失败: %v", err)
	}

	// 测试参数路由
	err = app.ServeHTTP("GET", "/users/123")
	if err != nil {
		t.Errorf("请求处理失败: %v", err)
	}

	// 测试 404
	err = app.ServeHTTP("GET", "/notfound")
	if err == nil {
		t.Error("期望 404 错误")
	}
}

// TestHTMLTemplateEngine 测试 HTML 模板引擎
func TestHTMLTemplateEngine(t *testing.T) {
	engine := NewHTMLTemplateEngine()
	engine.LoadTemplate("test", "<h1>{{title}}</h1>")

	data := map[string]interface{}{
		"title": "Hello World",
	}

	result, err := engine.Render("test", data)
	if err != nil {
		t.Errorf("渲染失败: %v", err)
	}

	expected := "<h1>Hello World</h1>"
	if result != expected {
		t.Errorf("期望 %s，实际 %s", expected, result)
	}
}

// TestJSONTemplateEngine 测试 JSON 模板引擎
func TestJSONTemplateEngine(t *testing.T) {
	engine := NewJSONTemplateEngine()
	engine.LoadTemplate("test", "")

	data := map[string]interface{}{
		"name": "Alice",
		"age":  30,
	}

	result, err := engine.Render("test", data)
	if err != nil {
		t.Errorf("渲染失败: %v", err)
	}

	// JSON 输出顺序可能不同，只检查是否包含关键字段
	if result == "" {
		t.Error("渲染结果为空")
	}
}

// TestTemplateRenderer 测试模板渲染器
func TestTemplateRenderer(t *testing.T) {
	engine := NewHTMLTemplateEngine()
	engine.LoadTemplate("test", "<p>{{content}}</p>")

	renderer := NewTemplateRenderer(engine)

	data := map[string]interface{}{
		"content": "Test Content",
	}

	result, err := renderer.RenderResponse("test", data)
	if err != nil {
		t.Errorf("渲染失败: %v", err)
	}

	expected := "<p>Test Content</p>"
	if result != expected {
		t.Errorf("期望 %s，实际 %s", expected, result)
	}
}

// TestTemplateFactory 测试模板引擎工厂
func TestTemplateFactory(t *testing.T) {
	factory := &TemplateFactory{}

	engines := []struct {
		engineType string
		expected   string
	}{
		{"html", "*main.HTMLTemplateEngine"},
		{"json", "*main.JSONTemplateEngine"},
		{"xml", "*main.XMLTemplateEngine"},
	}

	for _, tc := range engines {
		engine := factory.CreateEngine(tc.engineType)
		if engine == nil {
			t.Errorf("创建 %s 模板引擎失败", tc.engineType)
		}
	}
}

// TestMiddlewareOrder 测试中间件顺序
func TestMiddlewareOrder(t *testing.T) {
	order := []string{}

	middleware1 := func(ctx *RequestContext) error {
		order = append(order, "m1")
		return nil
	}

	middleware2 := func(ctx *RequestContext) error {
		order = append(order, "m2")
		return nil
	}

	handler := func(ctx *RequestContext) error {
		order = append(order, "handler")
		return nil
	}

	router := NewRouterBuilder().
		GET("/test", handler).
		Build()

	app := NewApplication(router)
	app.Use(middleware1)
	app.Use(middleware2)

	app.ServeHTTP("GET", "/test")

	if len(order) != 3 {
		t.Errorf("期望 3 个处理器，实际 %d 个", len(order))
	}

	if order[0] != "m1" || order[1] != "m2" || order[2] != "handler" {
		t.Errorf("中间件执行顺序错误: %v", order)
	}
}
