package main

import (
	"fmt"
	"strings"
)

// 路由构建 - 建造者模式
//
// 本模块使用建造者模式构建复杂的路由配置
// 支持链式调用、路由分组和中间件

// Route 路由
type Route struct {
	Method  string
	Path    string
	Handler HandlerFunc
}

// Router 路由器
type Router struct {
	routes      []Route
	middlewares []HandlerFunc
	groups      map[string]*RouterGroup
}

func NewRouter() *Router {
	return &Router{
		routes:      make([]Route, 0),
		middlewares: make([]HandlerFunc, 0),
		groups:      make(map[string]*RouterGroup),
	}
}

// Match 匹配路由
func (r *Router) Match(method string, path string) (*Route, map[string]string) {
	for _, route := range r.routes {
		if route.Method != method {
			continue
		}

		// 精确匹配
		if route.Path == path {
			return &route, nil
		}

		// 参数匹配
		params := r.matchParams(route.Path, path)
		if params != nil {
			return &route, params
		}
	}

	return nil, nil
}

// matchParams 匹配路由参数
func (r *Router) matchParams(pattern string, path string) map[string]string {
	patternParts := strings.Split(strings.Trim(pattern, "/"), "/")
	pathParts := strings.Split(strings.Trim(path, "/"), "/")

	if len(patternParts) != len(pathParts) {
		return nil
	}

	params := make(map[string]string)
	for i, part := range patternParts {
		if strings.HasPrefix(part, ":") {
			// 参数
			paramName := strings.TrimPrefix(part, ":")
			params[paramName] = pathParts[i]
		} else if part != pathParts[i] {
			// 不匹配
			return nil
		}
	}

	return params
}

// RouterBuilder 路由构建器
type RouterBuilder struct {
	router *Router
}

func NewRouterBuilder() *RouterBuilder {
	return &RouterBuilder{
		router: NewRouter(),
	}
}

// GET 注册 GET 路由
func (b *RouterBuilder) GET(path string, handler HandlerFunc) *RouterBuilder {
	b.router.routes = append(b.router.routes, Route{
		Method:  "GET",
		Path:    path,
		Handler: handler,
	})
	return b
}

// POST 注册 POST 路由
func (b *RouterBuilder) POST(path string, handler HandlerFunc) *RouterBuilder {
	b.router.routes = append(b.router.routes, Route{
		Method:  "POST",
		Path:    path,
		Handler: handler,
	})
	return b
}

// PUT 注册 PUT 路由
func (b *RouterBuilder) PUT(path string, handler HandlerFunc) *RouterBuilder {
	b.router.routes = append(b.router.routes, Route{
		Method:  "PUT",
		Path:    path,
		Handler: handler,
	})
	return b
}

// DELETE 注册 DELETE 路由
func (b *RouterBuilder) DELETE(path string, handler HandlerFunc) *RouterBuilder {
	b.router.routes = append(b.router.routes, Route{
		Method:  "DELETE",
		Path:    path,
		Handler: handler,
	})
	return b
}

// Use 添加全局中间件
func (b *RouterBuilder) Use(middleware HandlerFunc) *RouterBuilder {
	b.router.middlewares = append(b.router.middlewares, middleware)
	return b
}

// Group 创建路由分组
func (b *RouterBuilder) Group(prefix string) *RouterGroup {
	group := &RouterGroup{
		prefix:      prefix,
		router:      b.router,
		middlewares: make([]HandlerFunc, 0),
	}
	b.router.groups[prefix] = group
	return group
}

// Build 构建路由器
func (b *RouterBuilder) Build() *Router {
	return b.router
}

// RouterGroup 路由分组
type RouterGroup struct {
	prefix      string
	router      *Router
	middlewares []HandlerFunc
}

// GET 注册 GET 路由
func (g *RouterGroup) GET(path string, handler HandlerFunc) *RouterGroup {
	fullPath := g.prefix + path
	g.router.routes = append(g.router.routes, Route{
		Method:  "GET",
		Path:    fullPath,
		Handler: g.wrapHandler(handler),
	})
	return g
}

// POST 注册 POST 路由
func (g *RouterGroup) POST(path string, handler HandlerFunc) *RouterGroup {
	fullPath := g.prefix + path
	g.router.routes = append(g.router.routes, Route{
		Method:  "POST",
		Path:    fullPath,
		Handler: g.wrapHandler(handler),
	})
	return g
}

// PUT 注册 PUT 路由
func (g *RouterGroup) PUT(path string, handler HandlerFunc) *RouterGroup {
	fullPath := g.prefix + path
	g.router.routes = append(g.router.routes, Route{
		Method:  "PUT",
		Path:    fullPath,
		Handler: g.wrapHandler(handler),
	})
	return g
}

// DELETE 注册 DELETE 路由
func (g *RouterGroup) DELETE(path string, handler HandlerFunc) *RouterGroup {
	fullPath := g.prefix + path
	g.router.routes = append(g.router.routes, Route{
		Method:  "DELETE",
		Path:    fullPath,
		Handler: g.wrapHandler(handler),
	})
	return g
}

// Use 添加分组中间件
func (g *RouterGroup) Use(middleware HandlerFunc) *RouterGroup {
	g.middlewares = append(g.middlewares, middleware)
	return g
}

// wrapHandler 包装处理器，应用分组中间件
func (g *RouterGroup) wrapHandler(handler HandlerFunc) HandlerFunc {
	return func(ctx *RequestContext) error {
		// 执行分组中间件
		for _, middleware := range g.middlewares {
			if err := middleware(ctx); err != nil {
				return err
			}
		}
		// 执行处理器
		return handler(ctx)
	}
}

// PrintRoutes 打印所有路由
func (r *Router) PrintRoutes() {
	fmt.Println("注册的路由:")
	for _, route := range r.routes {
		fmt.Printf("  %s %s\n", route.Method, route.Path)
	}
}
