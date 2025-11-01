# 日志工厂实战项目

## 项目背景

日志是应用程序的重要组成部分。本项目实现一个灵活的日志系统，使用工厂模式创建不同类型的日志记录器（控制台、文件、远程等），支持不同的日志级别和格式。

## 功能列表

- [x] 工厂模式实现
- [x] 支持多种日志类型（Console、File、Syslog）
- [x] 支持日志级别（DEBUG、INFO、WARN、ERROR）
- [x] 支持自定义格式
- [x] 线程安全
- [x] 完整的单元测试

## 技术栈

- Go 1.21+
- 标准库 `log`
- 标准库 `os`
- 标准库 `fmt`

## 设计模式应用

| 模式 | 应用位置 | 作用 |
|------|----------|------|
| 简单工厂 | `CreateLogger` | 根据类型创建日志记录器 |
| 工厂方法 | `LoggerFactory` | 每种日志类型有自己的工厂 |

## 项目结构

```
logger-factory/
├── README.md           # 项目说明（本文件）
├── logger.go           # 日志记录器接口和实现
├── factory.go          # 工厂实现
└── logger_test.go      # 单元测试
```

## 使用示例

### 示例 1: 简单工厂

```go
// 创建控制台日志记录器
logger, err := CreateLogger("console", &LoggerConfig{
    Level:  "INFO",
    Format: "[%s] %s: %s",
})

logger.Info("Application started")
logger.Error("An error occurred")
```

### 示例 2: 工厂方法

```go
// 使用工厂方法
factory := &FileLoggerFactory{}
logger := factory.CreateLogger(&LoggerConfig{
    Level:    "DEBUG",
    Filename: "app.log",
})

logger.Debug("Debug message")
logger.Info("Info message")
```

### 示例 3: 多个日志记录器

```go
// 同时使用多个日志记录器
consoleLogger, _ := CreateLogger("console", config)
fileLogger, _ := CreateLogger("file", config)

consoleLogger.Info("Log to console")
fileLogger.Info("Log to file")
```

## 核心 API

### Logger 接口

- `Debug(message string)` - 调试日志
- `Info(message string)` - 信息日志
- `Warn(message string)` - 警告日志
- `Error(message string)` - 错误日志
- `SetLevel(level string)` - 设置日志级别

### 工厂方法

- `CreateLogger(loggerType string, config *LoggerConfig) (Logger, error)` - 简单工厂
- `LoggerFactory.CreateLogger(config *LoggerConfig) Logger` - 工厂方法

## 预期输出

```
=== 日志工厂实战项目 ===

示例 1: 控制台日志
-------------------
[2024-01-01 10:00:00] INFO: Application started
[2024-01-01 10:00:00] WARN: This is a warning
[2024-01-01 10:00:00] ERROR: An error occurred

示例 2: 文件日志
-------------------
日志已写入文件: app.log

示例 3: 不同日志级别
-------------------
[DEBUG] Debug message
[INFO] Info message
[WARN] Warning message
[ERROR] Error message

=== 示例结束 ===
```

## 学习要点

1. **工厂模式的应用**
   - 简单工厂 vs 工厂方法
   - 如何设计可扩展的工厂
   - 注册机制

2. **日志系统设计**
   - 日志级别
   - 日志格式
   - 输出目标

---

**项目难度**: 中等  
**预计完成时间**: 1-2 小时  
**适合人群**: 有 Go 基础，想学习工厂模式的开发者
