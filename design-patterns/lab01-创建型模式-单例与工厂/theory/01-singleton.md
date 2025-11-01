# 单例模式 (Singleton Pattern)

## 定义

单例模式确保一个类只有一个实例，并提供一个全局访问点。

## 意图

- 控制实例数量
- 节省系统资源
- 提供全局访问点

## 类图

```
┌─────────────────────┐
│    Singleton        │
├─────────────────────┤
│ - instance: *Self   │
│ - once: sync.Once   │
├─────────────────────┤
│ + GetInstance()     │
│ + Method1()         │
│ + Method2()         │
└─────────────────────┘
```

## 适用场景

1. **配置管理器**
   - 全局配置只需一份
   - 避免重复读取配置文件

2. **数据库连接池**
   - 管理有限的数据库连接
   - 避免创建过多连接

3. **日志记录器**
   - 统一的日志输出
   - 避免文件冲突

4. **线程池**
   - 管理线程资源
   - 避免线程创建开销

5. **缓存管理器**
   - 全局缓存访问
   - 节省内存

## 实现方式

### 1. 饿汉式 (Eager Initialization)

**特点**: 类加载时就创建实例

```go
type Singleton struct {
    // 字段
}

// 包初始化时创建
var instance = &Singleton{}

func GetInstance() *Singleton {
    return instance
}
```

**优点**:
- ✅ 线程安全
- ✅ 实现简单
- ✅ 无性能问题

**缺点**:
- ❌ 无法延迟初始化
- ❌ 可能浪费资源（如果从不使用）

### 2. 懒汉式 (Lazy Initialization)

**特点**: 第一次使用时才创建实例

```go
type Singleton struct {
    // 字段
}

var instance *Singleton

func GetInstance() *Singleton {
    if instance == nil {
        instance = &Singleton{}
    }
    return instance
}
```

**优点**:
- ✅ 延迟初始化
- ✅ 节省资源

**缺点**:
- ❌ 非线程安全
- ❌ 并发时可能创建多个实例

### 3. 双重检查锁 (Double-Checked Locking)

**特点**: 使用锁保证线程安全，双重检查减少锁开销

```go
type Singleton struct {
    // 字段
}

var (
    instance *Singleton
    mu       sync.Mutex
)

func GetInstance() *Singleton {
    if instance == nil {  // 第一次检查（无锁）
        mu.Lock()
        defer mu.Unlock()
        if instance == nil {  // 第二次检查（有锁）
            instance = &Singleton{}
        }
    }
    return instance
}
```

**优点**:
- ✅ 线程安全
- ✅ 延迟初始化
- ✅ 性能较好

**缺点**:
- ❌ 实现复杂
- ❌ 在某些语言中可能有问题（Go 中没问题）

### 4. sync.Once 实现 (推荐)

**特点**: 使用 Go 标准库的 sync.Once

```go
type Singleton struct {
    // 字段
}

var (
    instance *Singleton
    once     sync.Once
)

func GetInstance() *Singleton {
    once.Do(func() {
        instance = &Singleton{}
    })
    return instance
}
```

**优点**:
- ✅ 线程安全
- ✅ 延迟初始化
- ✅ 性能最优
- ✅ 实现简单
- ✅ Go 语言推荐方式

**缺点**:
- ❌ 无（这是 Go 中的最佳实践）

## 优缺点分析

### 优点

1. **控制实例数量**
   - 确保只有一个实例
   - 节省系统资源

2. **全局访问点**
   - 提供统一的访问方式
   - 方便管理和维护

3. **延迟初始化**
   - 按需创建实例
   - 提高启动速度

### 缺点

1. **违反单一职责原则**
   - 既负责创建实例，又负责业务逻辑

2. **难以测试**
   - 全局状态难以隔离
   - 单元测试困难

3. **可能成为全局状态**
   - 隐藏依赖关系
   - 降低代码可读性

4. **并发问题**
   - 需要考虑线程安全
   - 可能成为性能瓶颈

## 使用建议

### 何时使用

✅ **应该使用**:
- 确实需要全局唯一实例
- 资源共享（如连接池、缓存）
- 控制资源访问（如文件、硬件）

❌ **不应该使用**:
- 只是为了方便访问
- 可以用依赖注入替代
- 需要多个实例的场景

### 最佳实践

1. **使用 sync.Once**
   ```go
   var (
       instance *Singleton
       once     sync.Once
   )
   
   func GetInstance() *Singleton {
       once.Do(func() {
           instance = &Singleton{}
       })
       return instance
   }
   ```

2. **考虑依赖注入**
   ```go
   // 不要这样
   func ProcessData() {
       config := GetConfigInstance()
       // ...
   }
   
   // 应该这样
   func ProcessData(config *Config) {
       // ...
   }
   ```

3. **提供重置方法（测试用）**
   ```go
   func ResetInstance() {
       instance = nil
       once = sync.Once{}
   }
   ```

4. **线程安全的操作**
   ```go
   type Singleton struct {
       mu   sync.RWMutex
       data map[string]interface{}
   }
   
   func (s *Singleton) Set(key string, value interface{}) {
       s.mu.Lock()
       defer s.mu.Unlock()
       s.data[key] = value
   }
   
   func (s *Singleton) Get(key string) (interface{}, bool) {
       s.mu.RLock()
       defer s.mu.RUnlock()
       val, ok := s.data[key]
       return val, ok
   }
   ```

## 实际应用

### 1. 配置管理器

```go
type Config struct {
    AppName string
    Version string
    Debug   bool
    mu      sync.RWMutex
    data    map[string]interface{}
}

var (
    configInstance *Config
    configOnce     sync.Once
)

func GetConfig() *Config {
    configOnce.Do(func() {
        configInstance = &Config{
            data: make(map[string]interface{}),
        }
        // 加载配置文件
        configInstance.load()
    })
    return configInstance
}
```

### 2. 数据库连接池

```go
type DBPool struct {
    pool *sql.DB
    mu   sync.Mutex
}

var (
    dbInstance *DBPool
    dbOnce     sync.Once
)

func GetDBPool() *DBPool {
    dbOnce.Do(func() {
        db, err := sql.Open("mysql", "dsn")
        if err != nil {
            panic(err)
        }
        dbInstance = &DBPool{pool: db}
    })
    return dbInstance
}
```

### 3. 日志记录器

```go
type Logger struct {
    file *os.File
    mu   sync.Mutex
}

var (
    loggerInstance *Logger
    loggerOnce     sync.Once
)

func GetLogger() *Logger {
    loggerOnce.Do(func() {
        file, err := os.OpenFile("app.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
        if err != nil {
            panic(err)
        }
        loggerInstance = &Logger{file: file}
    })
    return loggerInstance
}
```

## 常见问题

### Q1: 单例模式是否违反 SOLID 原则？

是的，单例模式违反了单一职责原则（SRP），因为它既负责创建实例，又负责业务逻辑。但在某些场景下，这种权衡是值得的。

### Q2: 如何测试使用单例的代码？

1. 提供重置方法
2. 使用接口而非具体类型
3. 使用依赖注入
4. 使用测试替身（Test Double）

### Q3: 单例模式在 Go 中是否必要？

不一定。Go 可以使用包级别的变量实现类似效果，但 sync.Once 提供了更好的延迟初始化和线程安全保证。

### Q4: 如何避免单例模式的滥用？

1. 只在真正需要全局唯一实例时使用
2. 优先考虑依赖注入
3. 避免将单例作为全局状态容器
4. 保持单例的职责单一

## 总结

单例模式是一个简单但强大的模式，在 Go 语言中使用 sync.Once 可以轻松实现线程安全的单例。但要注意避免滥用，优先考虑依赖注入等更灵活的方案。

**记住**:
- ✅ 使用 sync.Once 实现
- ✅ 考虑线程安全
- ✅ 提供测试支持
- ❌ 避免滥用
- ❌ 不要作为全局状态容器
