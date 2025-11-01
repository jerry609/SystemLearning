# 配置管理器实战项目

## 项目背景

在实际应用中，配置管理是一个常见需求。本项目实现一个线程安全的配置管理器，使用单例模式确保全局只有一个配置实例，支持从文件、环境变量等多种来源加载配置。

## 功能列表

- [x] 单例模式实现
- [x] 支持多种配置源（文件、环境变量、默认值）
- [x] 线程安全的读写操作
- [x] 支持配置热更新
- [x] 支持配置验证
- [x] 完整的单元测试

## 技术栈

- Go 1.21+
- 标准库 `sync`
- 标准库 `os`
- 标准库 `encoding/json`

## 设计模式应用

| 模式 | 应用位置 | 作用 |
|------|----------|------|
| 单例模式 | `Config` | 确保全局只有一个配置实例 |

## 项目结构

```
config-manager/
├── README.md           # 项目说明（本文件）
├── config.go           # 配置管理器核心代码
└── config_test.go      # 单元测试
```

## 运行方式

### 运行示例

```bash
# 运行主程序
go run config.go
```

### 运行测试

```bash
# 运行所有测试
go test -v

# 运行测试并显示覆盖率
go test -v -cover
```

## 使用示例

### 示例 1: 基本使用

```go
config := GetConfig()

// 设置配置
config.Set("app.name", "MyApp")
config.Set("app.version", "1.0.0")
config.Set("server.port", 8080)

// 获取配置
appName := config.GetString("app.name")
port := config.GetInt("server.port")

fmt.Printf("App: %s, Port: %d\n", appName, port)
```

### 示例 2: 从文件加载

```go
config := GetConfig()

// 从 JSON 文件加载
err := config.LoadFromFile("config.json")
if err != nil {
    log.Fatal(err)
}

// 使用配置
dbHost := config.GetString("database.host")
dbPort := config.GetInt("database.port")
```

### 示例 3: 环境变量

```go
config := GetConfig()

// 从环境变量加载
config.LoadFromEnv("APP_")

// 获取配置（优先使用环境变量）
port := config.GetInt("port") // 从 APP_PORT 环境变量读取
```

## 核心 API

### Config 方法

- `GetConfig()` - 获取配置单例实例
- `Set(key string, value interface{})` - 设置配置
- `Get(key string) (interface{}, bool)` - 获取配置
- `GetString(key string) string` - 获取字符串配置
- `GetInt(key string) int` - 获取整数配置
- `GetBool(key string) bool` - 获取布尔配置
- `LoadFromFile(filename string) error` - 从文件加载
- `LoadFromEnv(prefix string)` - 从环境变量加载
- `SaveToFile(filename string) error` - 保存到文件
- `Reload() error` - 重新加载配置

## 预期输出

运行 `go run config.go` 将输出：

```
=== 配置管理器实战项目 ===

示例 1: 基本使用
-------------------
设置配置...
应用名称: MyApp
版本号: 1.0.0
服务器端口: 8080
调试模式: true

示例 2: 验证单例
-------------------
实例 1: 0xc0000a0000
实例 2: 0xc0000a0000
✅ config1 和 config2 是同一个实例

示例 3: 并发访问
-------------------
启动 100 个 goroutine 并发读写...
并发操作完成
配置项数量: 104

=== 示例结束 ===
```

## 扩展建议

1. **支持更多配置格式**
   - YAML
   - TOML
   - INI

2. **配置监听**
   ```go
   config.Watch(func(key string, oldValue, newValue interface{}) {
       fmt.Printf("配置变更: %s = %v -> %v\n", key, oldValue, newValue)
   })
   ```

3. **配置加密**
   ```go
   config.SetEncrypted("database.password", "secret")
   password := config.GetDecrypted("database.password")
   ```

4. **配置分组**
   ```go
   dbConfig := config.GetGroup("database")
   host := dbConfig.GetString("host")
   port := dbConfig.GetInt("port")
   ```

## 学习要点

1. **单例模式的应用**
   - 使用 sync.Once 确保线程安全
   - 全局唯一实例
   - 延迟初始化

2. **并发安全**
   - 使用读写锁
   - 避免竞态条件

3. **配置管理**
   - 多种配置源
   - 配置优先级
   - 配置验证

---

**项目难度**: 中等  
**预计完成时间**: 1-2 小时  
**适合人群**: 有 Go 基础，想学习单例模式的开发者
