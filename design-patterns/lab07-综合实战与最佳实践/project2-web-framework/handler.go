package main

import (
	"fmt"
)

// 请求处理 - 责任链模式
//
// 本模块使用责任链模式处理 HTTP 请求
// 支持中间件链式处理

// RequestContext 请求上下文
type RequestContext struct {
	Method string
	Path   string
	Params map[string]string
	Data   map[string]interface{}
	Status int
	Body   string
}

func NewRequestContext(method string, path string) *RequestContext {
	return &RequestContext{
		Method: method,
		Path:   path,
		Params: make(map[string]string),
		Data:   make(map[string]interface{}),
		Status: 200,
	}
}

// HandlerFunc 处理器函数类型
type HandlerFunc func(ctx *RequestContext) error

// HandlerChain 处理器链
type HandlerChain struct {
	handlers []HandlerFunc
}

func NewHandlerChain() *HandlerChain {
	return &HandlerChain{
		handlers: make([]HandlerFunc, 0),
	}
}

// Use 添加处理器
func (c *HandlerChain) Use(handler HandlerFunc) *HandlerChain {
	c.handlers = append(c.handlers, handler)
	return c
}

// Execute 执行处理器链
func (c *HandlerChain) Execute(ctx *RequestContext) error {
	for _, handler := range c.handlers {
		if err := handler(ctx); err != nil {
			return err
		}
	}
	return nil
}

// LoggingHandler 日志处理器
func LoggingHandler(ctx *RequestContext) error {
	fmt.Printf("[日志] %s %s\n", ctx.Method, ctx.Path)
	return nil
}

// AuthHandler 认证处理器
func AuthHandler(ctx *RequestContext) error {
	// 模拟认证
	fmt.Println("[认证] 验证通过")
	ctx.Data["user"] = "authenticated-user"
	return nil
}

// CORSHandler CORS 处理器
func CORSHandler(ctx *RequestContext) error {
	fmt.Println("[CORS] 设置 CORS 头")
	ctx.Data["cors"] = true
	return nil
}

// CompressionHandler 压缩处理器
func CompressionHandler(ctx *RequestContext) error {
	fmt.Println("[压缩] 启用响应压缩")
	ctx.Data["compression"] = true
	return nil
}

// ErrorHandler 错误处理器
func ErrorHandler(ctx *RequestContext) error {
	// 检查是否有错误
	if err, ok := ctx.Data["error"]; ok {
		fmt.Printf("[错误] 处理错误: %v\n", err)
		ctx.Status = 500
		ctx.Body = fmt.Sprintf("Internal Server Error: %v", err)
		return nil
	}
	return nil
}

// RecoveryHandler 恢复处理器
func RecoveryHandler(next HandlerFunc) HandlerFunc {
	return func(ctx *RequestContext) error {
		defer func() {
			if r := recover(); r != nil {
				fmt.Printf("[恢复] 捕获 panic: %v\n", r)
				ctx.Status = 500
				ctx.Body = "Internal Server Error"
			}
		}()
		return next(ctx)
	}
}

// Application 应用程序
type Application struct {
	router      *Router
	middlewares []HandlerFunc
}

func NewApplication(router *Router) *Application {
	return &Application{
		router:      router,
		middlewares: make([]HandlerFunc, 0),
	}
}

// Use 添加全局中间件
func (app *Application) Use(middleware HandlerFunc) {
	app.middlewares = append(app.middlewares, middleware)
}

// ServeHTTP 处理 HTTP 请求
func (app *Application) ServeHTTP(method string, path string) error {
	// 创建请求上下文
	ctx := NewRequestContext(method, path)

	// 匹配路由
	route, params := app.router.Match(method, path)
	if route == nil {
		ctx.Status = 404
		ctx.Body = "Not Found"
		return fmt.Errorf("路由未找到: %s %s", method, path)
	}

	ctx.Params = params

	// 构建处理器链
	chain := NewHandlerChain()

	// 添加全局中间件
	for _, middleware := range app.middlewares {
		chain.Use(middleware)
	}

	// 添加路由处理器
	chain.Use(route.Handler)

	// 执行处理器链
	return chain.Execute(ctx)
}

// 业务处理器示例

// HomeHandler 首页处理器
func HomeHandler(ctx *RequestContext) error {
	fmt.Println("[业务] 处理首页请求")
	ctx.Body = "Welcome to Web Framework!"
	return nil
}

// ListUsersHandler 用户列表处理器
func ListUsersHandler(ctx *RequestContext) error {
	fmt.Println("[业务] 处理用户列表请求")
	ctx.Body = "User List"
	ctx.Data["users"] = []string{"Alice", "Bob", "Charlie"}
	return nil
}

// GetUserHandler 获取用户处理器
func GetUserHandler(ctx *RequestContext) error {
	userID := ctx.Params["id"]
	fmt.Printf("[业务] 处理用户详情请求，ID: %s\n", userID)
	ctx.Body = fmt.Sprintf("User Detail: %s", userID)
	ctx.Data["user"] = map[string]string{"id": userID, "name": "User " + userID}
	return nil
}

// CreateUserHandler 创建用户处理器
func CreateUserHandler(ctx *RequestContext) error {
	fmt.Println("[业务] 处理创建用户请求")
	ctx.Status = 201
	ctx.Body = "User Created"
	return nil
}

// UpdateUserHandler 更新用户处理器
func UpdateUserHandler(ctx *RequestContext) error {
	userID := ctx.Params["id"]
	fmt.Printf("[业务] 处理更新用户请求，ID: %s\n", userID)
	ctx.Body = fmt.Sprintf("User %s Updated", userID)
	return nil
}

// DeleteUserHandler 删除用户处理器
func DeleteUserHandler(ctx *RequestContext) error {
	userID := ctx.Params["id"]
	fmt.Printf("[业务] 处理删除用户请求，ID: %s\n", userID)
	ctx.Status = 204
	ctx.Body = ""
	return nil
}
