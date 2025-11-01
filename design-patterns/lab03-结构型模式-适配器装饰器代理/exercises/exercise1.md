# 练习 1: 实现日志系统适配器

## 难度
⭐⭐ (中等)

## 学习目标
- 掌握适配器模式的实现
- 理解接口转换的应用场景
- 学会集成第三方库
- 实践统一接口设计

## 问题描述

你的项目中使用了多个不同的日志库（标准库 log、第三方库 logrus、zap 等），每个库的接口都不相同。现在需要实现一个统一的日志接口，并为不同的日志库创建适配器，使它们都能通过统一的接口使用。

## 功能要求

1. **统一日志接口**
   - 定义标准的日志接口 `Logger`
   - 支持不同日志级别（Debug, Info, Warn, Error）
   - 支持格式化输出
   - 支持结构化日志（键值对）

2. **标准库适配器**
   - 为 Go 标准库 `log` 包创建适配器
   - 实现所有接口方法
   - 保持原有功能

3. **第三方库适配器**
   - 为至少一个第三方日志库创建适配器
   - 支持该库的特性
   - 统一接口调用

4. **日志管理器**
   - 支持运行时切换日志实现
   - 支持多个日志输出（同时写入多个日志）
   - 支持日志级别过滤

## 输入输出示例

### 示例 1: 使用标准库适配器
**代码**:
```go
// 创建标准库适配器
stdLogger := log.New(os.Stdout, "[APP] ", log.LstdFlags)
logger := NewStdLogAdapter(stdLogger)

// 使用统一接口
logger.Info("Application started")
logger.Warn("Low memory warning")
logger.Error("Failed to connect to database")
logger.WithFields(map[string]interface{}{
    "user_id": 123,
    "action":  "login",
}).Info("User logged in")
```

**输出**:
```
[APP] 2024/01/15 10:30:00 [INFO] Application started
[APP] 2024/01/15 10:30:01 [WARN] Low memory warning
[APP] 2024/01/15 10:30:02 [ERROR] Failed to connect to database
[APP] 2024/01/15 10:30:03 [INFO] User logged in user_id=123 action=login
```

### 示例 2: 切换日志实现
**代码**:
```go
// 创建日志管理器
manager := NewLoggerManager()

// 注册不同的日志实现
manager.Register("std", NewStdLogAdapter(stdLogger))
manager.Register("custom", NewCustomLogAdapter())

// 使用标准库
manager.Use("std")
manager.Info("Using standard logger")

// 切换到自定义实现
manager.Use("custom")
manager.Info("Using custom logger")
```

**输出**:
```
[STD] 2024/01/15 10:30:00 [INFO] Using standard logger
[CUSTOM] 2024-01-15T10:30:01+08:00 INFO Using custom logger
```

### 示例 3: 多日志输出
**代码**:
```go
// 创建多个日志适配器
fileLogger := NewFileLogAdapter("app.log")
consoleLogger := NewStdLogAdapter(stdLogger)

// 创建多路日志
multiLogger := NewMultiLogger(fileLogger, consoleLogger)

// 同时写入文件和控制台
multiLogger.Info("This message goes to both file and console")
multiLogger.WithFields(map[string]interface{}{
    "request_id": "abc123",
    "duration":   "150ms",
}).Info("Request completed")
```

**输出**:
```
# 控制台输出
[APP] 2024/01/15 10:30:00 [INFO] This message goes to both file and console
[APP] 2024/01/15 10:30:01 [INFO] Request completed request_id=abc123 duration=150ms

# app.log 文件内容
2024-01-15 10:30:00 [INFO] This message goes to both file and console
2024-01-15 10:30:01 [INFO] Request completed request_id=abc123 duration=150ms
```

## 接口定义

```go
// Logger 统一日志接口
type Logger interface {
    Debug(msg string)
    Info(msg string)
    Warn(msg string)
    Error(msg string)
    
    Debugf(format string, args ...interface{})
    Infof(format string, args ...interface{})
    Warnf(format string, args ...interface{})
    Errorf(format string, args ...interface{})
    
    WithFields(fields map[string]interface{}) Logger
}

// LogLevel 日志级别
type LogLevel int

const (
    DebugLevel LogLevel = iota
    InfoLevel
    WarnLevel
    ErrorLevel
)
```

## 提示

💡 **提示 1**: 标准库适配器实现
```go
type StdLogAdapter struct {
    logger *log.Logger
    fields map[string]interface{}
}

func NewStdLogAdapter(logger *log.Logger) *StdLogAdapter {
    return &StdLogAdapter{
        logger: logger,
        fields: make(map[string]interface{}),
    }
}

func (a *StdLogAdapter) Info(msg string) {
    fieldsStr := a.formatFields()
    a.logger.Printf("[INFO] %s%s", msg, fieldsStr)
}
```

💡 **提示 2**: 实现 WithFields 方法
```go
func (a *StdLogAdapter) WithFields(fields map[string]interface{}) Logger {
    // 创建新实例，避免修改原实例
    newAdapter := &StdLogAdapter{
        logger: a.logger,
        fields: make(map[string]interface{}),
    }
    
    // 复制现有字段
    for k, v := range a.fields {
        newAdapter.fields[k] = v
    }
    
    // 添加新字段
    for k, v := range fields {
        newAdapter.fields[k] = v
    }
    
    return newAdapter
}
```

💡 **提示 3**: 格式化字段输出
```go
func (a *StdLogAdapter) formatFields() string {
    if len(a.fields) == 0 {
        return ""
    }
    
    var parts []string
    for k, v := range a.fields {
        parts = append(parts, fmt.Sprintf("%s=%v", k, v))
    }
    return " " + strings.Join(parts, " ")
}
```

💡 **提示 4**: 多日志输出实现
```go
type MultiLogger struct {
    loggers []Logger
}

func NewMultiLogger(loggers ...Logger) *MultiLogger {
    return &MultiLogger{loggers: loggers}
}

func (m *MultiLogger) Info(msg string) {
    for _, logger := range m.loggers {
        logger.Info(msg)
    }
}
```

## 评分标准

- [ ] **接口设计 (25%)**
  - 统一日志接口设计合理
  - 支持所有必需的日志级别
  - 支持结构化日志

- [ ] **适配器实现 (35%)**
  - 正确实现标准库适配器
  - 实现至少一个第三方库适配器
  - 保持原有库的功能

- [ ] **功能完整性 (25%)**
  - 支持运行时切换
  - 支持多日志输出
  - 支持日志级别过滤

- [ ] **代码质量 (15%)**
  - 代码结构清晰
  - 命名规范
  - 适当的注释

## 扩展挑战

如果你完成了基本要求，可以尝试以下扩展功能：

1. **异步日志**
   ```go
   type AsyncLogger struct {
       logger Logger
       queue  chan logEntry
   }
   
   func (a *AsyncLogger) Info(msg string) {
       a.queue <- logEntry{level: InfoLevel, msg: msg}
   }
   
   func (a *AsyncLogger) start() {
       go func() {
           for entry := range a.queue {
               a.logger.Info(entry.msg)
           }
       }()
   }
   ```

2. **日志采样**
   ```go
   type SamplingLogger struct {
       logger     Logger
       sampleRate float64 // 0.0 - 1.0
   }
   
   func (s *SamplingLogger) Info(msg string) {
       if rand.Float64() < s.sampleRate {
           s.logger.Info(msg)
       }
   }
   ```

3. **日志钩子**
   ```go
   type Hook interface {
       Fire(level LogLevel, msg string, fields map[string]interface{}) error
   }
   
   type HookableLogger struct {
       logger Logger
       hooks  []Hook
   }
   ```

4. **日志轮转**
   ```go
   type RotatingFileAdapter struct {
       filename   string
       maxSize    int64
       maxBackups int
       currentSize int64
   }
   
   func (r *RotatingFileAdapter) rotate() error {
       // 实现日志文件轮转
   }
   ```

## 参考资源

- [Go log 包文档](https://pkg.go.dev/log)
- [适配器模式详解](../theory/01-adapter.md)
- [Logrus 库](https://github.com/sirupsen/logrus)
- [Zap 库](https://github.com/uber-go/zap)

## 提交要求

1. 实现统一的 `Logger` 接口
2. 实现至少 2 个日志适配器
3. 实现日志管理器
4. 编写测试用例验证功能
5. 提供使用示例
6. 添加必要的注释和文档

---

**预计完成时间**: 1.5-2 小时  
**难度评估**: 中等  
**重点考察**: 适配器模式、接口设计、第三方库集成
