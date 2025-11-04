package main

import (
	"fmt"
)

func main() {
	fmt.Println("=== Web 框架演示 ===\n")

	// 1. 路由构建演示
	demonstrateRouterBuilder()

	// 2. 请求处理演示
	demonstrateRequestHandling()

	// 3. 模板渲染演示
	demonstrateTemplateRendering()

	fmt.Println("\n=== 演示完成 ===")
}

// demonstrateRouterBuilder 演示路由构建
func demonstrateRouterBuilder() {
	fmt.Println("--- 1. 路由构建 ---")

	// 使用建造者模式构建路由
	router := NewRouterBuilder().
		GET("/", HomeHandler).
		GET("/users", ListUsersHandler).
		POST("/users", CreateUserHandler).
		GET("/users/:id", GetUserHandler).
		DELETE("/users/:id", DeleteUserHandler).
		Build()

	// 打印路由
	router.PrintRoutes()

	fmt.Println()
}

// demonstrateRequestHandling 演示请求处理
func demonstrateRequestHandling() {
	fmt.Println("--- 2. 请求处理 ---")

	// 构建路由
	router := NewRouterBuilder().
		GET("/", HomeHandler).
		GET("/users", ListUsersHandler).
		GET("/users/:id", GetUserHandler).
		Build()

	// 创建应用
	app := NewApplication(router)

	// 添加全局中间件
	app.Use(LoggingHandler)
	app.Use(AuthHandler)

	// 处理请求
	requests := []struct {
		method string
		path   string
	}{
		{"GET", "/"},
		{"GET", "/users"},
		{"GET", "/users/123"},
	}

	for _, req := range requests {
		fmt.Printf("\n处理请求: %s %s\n", req.method, req.path)
		if err := app.ServeHTTP(req.method, req.path); err != nil {
			fmt.Printf("错误: %v\n", err)
		}
	}

	fmt.Println()
}

// demonstrateTemplateRendering 演示模板渲染
func demonstrateTemplateRendering() {
	fmt.Println("--- 3. 模板渲染 ---")

	// 1. HTML 模板渲染
	fmt.Println("\nHTML 模板渲染:")
	htmlEngine := NewHTMLTemplateEngine()
	htmlEngine.LoadTemplate("user-list", `<html>
<head><title>{{title}}</title></head>
<body>
<h1>{{title}}</h1>
<ul>
{{items}}
</ul>
</body>
</html>`)

	htmlData := map[string]interface{}{
		"title": "用户列表",
		"items": []string{"Alice", "Bob"},
	}

	htmlResult, err := htmlEngine.Render("user-list", htmlData)
	if err != nil {
		fmt.Printf("渲染失败: %v\n", err)
	} else {
		fmt.Println(htmlResult)
	}

	// 2. JSON 模板渲染
	fmt.Println("\nJSON 模板渲染:")
	jsonEngine := NewJSONTemplateEngine()
	jsonEngine.LoadTemplate("user-list", "")

	jsonData := map[string]interface{}{
		"users": []string{"Alice", "Bob"},
		"total": 2,
	}

	jsonResult, err := jsonEngine.Render("user-list", jsonData)
	if err != nil {
		fmt.Printf("渲染失败: %v\n", err)
	} else {
		fmt.Println(jsonResult)
	}

	// 3. 使用模板渲染器
	fmt.Println("\n使用模板渲染器:")
	renderer := NewTemplateRenderer(htmlEngine)
	result, err := renderer.RenderResponse("user-list", htmlData)
	if err != nil {
		fmt.Printf("渲染失败: %v\n", err)
	} else {
		fmt.Println(result)
	}

	fmt.Println()
}
