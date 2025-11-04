# Web 框架项目

## 项目背景

本项目实现了一个简单的 Web 框架，展示了如何使用建造者模式、责任链模式和模板方法模式构建一个灵活且易于扩展的 Web 应用框架。

## 功能列表

- [x] 路由注册和匹配（支持静态路由和参数路由）
- [x] 请求处理链（中间件支持）
- [x] 模板渲染引擎
- [x] 上下文管理
- [x] 完整的测试用例

## 技术栈

- Go 1.21+
- 标准库（无外部依赖）

## 设计模式应用

| 模式 | 应用位置 | 作用 |
|------|----------|------|
| 建造者模式 | router.go:RouterBuilder | 构建复杂的路由配置 |
| 责任链模式 | handler.go:HandlerChain | 处理 HTTP 请求的中间件链 |
| 模板方法模式 | template.go:TemplateEngine | 定义模板渲染的算法骨架 |

## 项目结构

```
project2-web-framework/
├── README.md              # 本文件
├── main.go               # 主程序入口，演示框架使用
├── router.go             # 路由构建（建造者模式）
├── handler.go            # 请求处理（责任链模式）
├── template.go           # 模板渲染（模板方法模式）
└── framework_test.go     # 测试文件
```

## 核心组件说明

### 1. 路由构建 (router.go)

使用**建造者模式**构建复杂的路由配置：

```go
// 路由构建器
type RouterBuilder struct {
    router *Router
}

// 链式调用构建路由
router := NewRouterBuilder().
    GET("/users", listUsersHandler).
    POST("/users", createUserHandler).
    GET("/users/:id", getUserHandler).
    Build()
```

**设计要点**:
- 使用建造者模式简化路由配置
- 支持链式调用
- 支持路由分组和中间件

### 2. 请求处理 (handler.go)

使用**责任链模式**处理 HTTP 请求：

```go
// 处理器链
type HandlerChain struct {
    handlers []Handler
}

// 中间件按顺序处理请求
chain.Use(LoggingHandler).
      Use(AuthHandler).
      Use(BusinessHandler)
```

**设计要点**:
- 责任链模式：灵活的请求处理流程
- 支持中间件
- 易于扩展新的处理器

### 3. 模板渲染 (template.go)

使用**模板方法模式**定义渲染流程：

```go
// 模板引擎（抽象类）
type TemplateEngine interface {
    Render(name string, data interface{}) (string, error)
}

// 具体实现定义渲染细节
type HTMLTemplateEngine struct { ... }
type JSONTemplateEngine struct { ... }
```

**设计要点**:
- 模板方法模式：定义算法骨架
- 子类实现具体步骤
- 支持多种模板格式

## 运行方式

### 运行演示程序

```bash
cd project2-web-framework
go run main.go
```

### 预期输出

```
=== Web 框架演示 ===

--- 1. 路由构建 ---
注册路由: GET /
注册路由: GET /users
注册路由: POST /users
注册路由: GET /users/:id
注册路由: DELETE /users/:id

--- 2. 请求处理 ---
处理请求: GET /
[日志] GET /
[认证] 验证通过
[业务] 处理首页请求
响应: Welcome to Web Framework!

处理请求: GET /users
[日志] GET /users
[认证] 验证通过
[业务] 处理用户列表请求
响应: User List

处理请求: GET /users/123
[日志] GET /users/123
[认证] 验证通过
[业务] 处理用户详情请求，ID: 123
响应: User Detail: 123

--- 3. 模板渲染 ---
HTML 模板渲染:
<html>
<head><title>用户列表</title></head>
<body>
<h1>用户列表</h1>
<ul>
<li>Alice</li>
<li>Bob</li>
</ul>
</body>
</html>

JSON 模板渲染:
{"users":["Alice","Bob"],"total":2}

=== 演示完成 ===
```

### 运行测试

```bash
go test -v
```

## 扩展建议

### 1. 路由功能增强

- 支持正则表达式路由
- 支持路由优先级
- 支持路由版本控制

### 2. 中间件扩展

- CORS 中间件
- 压缩中间件
- 缓存中间件
- 会话管理中间件

### 3. 模板引擎增强

- 支持模板继承
- 支持模板片段
- 支持自定义函数
- 支持模板缓存

### 4. 错误处理

- 统一的错误处理机制
- 自定义错误页面
- 错误日志记录

### 5. 性能优化

- 路由缓存
- 模板预编译
- 连接池管理

## 设计模式总结

### 建造者模式的应用

**优点**:
- 简化复杂对象的构建
- 支持链式调用
- 易于扩展新的配置选项

**使用场景**:
- 路由配置
- 请求构建
- 配置对象创建

### 责任链模式的应用

**优点**:
- 灵活的请求处理流程
- 易于添加和删除处理器
- 处理器之间解耦

**使用场景**:
- HTTP 中间件
- 请求过滤
- 数据验证

### 模板方法模式的应用

**优点**:
- 定义算法骨架
- 子类实现具体步骤
- 代码复用

**使用场景**:
- 模板渲染
- 数据处理流程
- 算法框架

## 学习要点

1. **理解建造者模式**: 如何使用建造者模式简化复杂对象的构建
2. **掌握责任链模式**: 如何使用责任链模式实现灵活的请求处理
3. **理解模板方法模式**: 如何使用模板方法模式定义算法骨架
4. **模式组合**: 如何组合使用多种模式构建完整系统
5. **Go 语言特性**: 如何利用 Go 语言的特性实现设计模式

## 参考资源

- [Gin](https://github.com/gin-gonic/gin) - 高性能 Web 框架
- [Echo](https://github.com/labstack/echo) - 简洁的 Web 框架
- [Chi](https://github.com/go-chi/chi) - 轻量级路由器
- [Gorilla Mux](https://github.com/gorilla/mux) - 强大的路由器

## 总结

本项目展示了如何使用建造者模式、责任链模式和模板方法模式构建一个 Web 框架。通过学习本项目，你应该能够：

1. 理解如何使用建造者模式简化配置
2. 掌握责任链模式在中间件中的应用
3. 理解模板方法模式的使用场景
4. 学会组合使用多种设计模式

继续探索和实践，你会发现设计模式的强大之处！
