# 缓存系统项目

## 项目背景

本项目实现了一个高性能的缓存系统，展示了如何使用单例模式、代理模式和策略模式构建一个灵活且易于扩展的缓存解决方案。

## 功能列表

- [x] 缓存管理器（单例模式）
- [x] 缓存代理（透明缓存）
- [x] 多种淘汰策略（LRU、LFU、FIFO）
- [x] 过期时间管理
- [x] 缓存统计
- [x] 完整的测试用例

## 技术栈

- Go 1.21+
- 标准库（无外部依赖）

## 设计模式应用

| 模式 | 应用位置 | 作用 |
|------|----------|------|
| 单例模式 | cache.go:CacheManager | 全局唯一的缓存管理器 |
| 代理模式 | proxy.go:CacheProxy | 为数据访问添加缓存层 |
| 策略模式 | strategy.go:EvictionStrategy | 实现不同的缓存淘汰算法 |

## 项目结构

```
project3-cache-system/
├── README.md              # 本文件
├── main.go               # 主程序入口，演示缓存系统使用
├── cache.go              # 缓存管理器（单例模式）
├── proxy.go              # 缓存代理（代理模式）
├── strategy.go           # 淘汰策略（策略模式）
└── cache_test.go         # 测试文件
```

## 核心组件说明

### 1. 缓存管理器 (cache.go)

使用**单例模式**确保全局唯一的缓存管理器：

```go
// 缓存管理器（单例）
type CacheManager struct {
    cache    map[string]*CacheItem
    strategy EvictionStrategy
    maxSize  int
}

// 获取单例实例
func GetCacheManager() *CacheManager {
    once.Do(func() {
        instance = &CacheManager{...}
    })
    return instance
}
```

**设计要点**:
- 使用 sync.Once 确保线程安全
- 全局唯一的缓存实例
- 支持配置最大容量和淘汰策略

### 2. 缓存代理 (proxy.go)

使用**代理模式**为数据访问添加缓存层：

```go
// 数据源接口
type DataSource interface {
    Get(key string) (interface{}, error)
}

// 缓存代理
type CacheProxy struct {
    dataSource DataSource
    cache      *CacheManager
}

// 透明缓存访问
func (p *CacheProxy) Get(key string) (interface{}, error) {
    // 先查缓存
    if value, ok := p.cache.Get(key); ok {
        return value, nil
    }
    // 缓存未命中，从数据源获取
    value, err := p.dataSource.Get(key)
    if err == nil {
        p.cache.Set(key, value, 0)
    }
    return value, err
}
```

**设计要点**:
- 代理模式：透明的缓存访问
- 自动缓存未命中的数据
- 支持缓存穿透保护

### 3. 淘汰策略 (strategy.go)

使用**策略模式**实现多种缓存淘汰算法：

```go
// 淘汰策略接口
type EvictionStrategy interface {
    Evict(cache map[string]*CacheItem) string
    OnAccess(key string)
    OnSet(key string)
}

// 具体策略：LRU、LFU、FIFO
type LRUStrategy struct { ... }
type LFUStrategy struct { ... }
type FIFOStrategy struct { ... }
```

**设计要点**:
- 策略模式：封装不同的淘汰算法
- 运行时切换策略
- 易于扩展新的算法

## 运行方式

### 运行演示程序

```bash
cd project3-cache-system
go run main.go
```

### 预期输出

```
=== 缓存系统演示 ===

--- 1. 基本缓存操作 ---
设置缓存: key1 = value1
设置缓存: key2 = value2
设置缓存: key3 = value3
获取缓存: key1 = value1
获取缓存: key2 = value2
删除缓存: key1
获取缓存: key1 = <nil> (已删除)

--- 2. 缓存淘汰策略 ---
使用 LRU 策略:
  设置: item1, item2, item3, item4
  缓存已满，淘汰: item1 (最久未使用)
  访问: item2
  设置: item5
  缓存已满，淘汰: item3

使用 LFU 策略:
  设置: item1, item2, item3, item4
  访问 item1 3 次
  访问 item2 2 次
  缓存已满，淘汰: item3 (使用频率最低)

使用 FIFO 策略:
  设置: item1, item2, item3, item4
  缓存已满，淘汰: item1 (最先进入)

--- 3. 缓存代理 ---
通过代理获取数据: user:1
  缓存未命中，从数据源加载
  数据: User 1
通过代理获取数据: user:1
  缓存命中
  数据: User 1
通过代理获取数据: user:2
  缓存未命中，从数据源加载
  数据: User 2

--- 4. 缓存过期 ---
设置缓存: temp = data (TTL: 2s)
立即获取: temp = data
等待 3 秒...
过期后获取: temp = <nil> (已过期)

--- 5. 缓存统计 ---
缓存统计:
  总请求: 15
  命中: 8
  未命中: 7
  命中率: 53.33%

=== 演示完成 ===
```

### 运行测试

```bash
go test -v
```

## 扩展建议

### 1. 分布式缓存

- 支持多节点部署
- 一致性哈希
- 数据分片

### 2. 持久化

- 定期快照
- AOF 日志
- 数据恢复

### 3. 高级特性

- 缓存预热
- 缓存雪崩保护
- 缓存击穿保护
- 布隆过滤器

### 4. 监控和管理

- 实时监控
- 性能指标
- 管理接口
- 可视化面板

### 5. 性能优化

- 分段锁
- 无锁数据结构
- 内存池
- 压缩存储

## 设计模式总结

### 单例模式的应用

**优点**:
- 全局唯一实例
- 节省资源
- 统一访问点

**使用场景**:
- 缓存管理器
- 配置管理器
- 日志管理器

### 代理模式的应用

**优点**:
- 透明的缓存访问
- 控制对象访问
- 延迟加载

**使用场景**:
- 缓存代理
- 远程代理
- 虚拟代理

### 策略模式的应用

**优点**:
- 封装算法族
- 运行时切换算法
- 易于扩展

**使用场景**:
- 缓存淘汰
- 负载均衡
- 数据压缩

## 学习要点

1. **理解单例模式**: 如何在 Go 中实现线程安全的单例
2. **掌握代理模式**: 如何使用代理模式添加缓存层
3. **理解策略模式**: 如何使用策略模式实现可插拔的算法
4. **缓存设计**: 理解缓存系统的核心概念和设计要点
5. **性能优化**: 学习缓存系统的性能优化技巧

## 参考资源

- [groupcache](https://github.com/golang/groupcache) - Google 的分布式缓存
- [bigcache](https://github.com/allegro/bigcache) - 高性能缓存
- [go-cache](https://github.com/patrickmn/go-cache) - 内存缓存
- [Redis](https://redis.io/) - 流行的缓存数据库

## 总结

本项目展示了如何使用单例模式、代理模式和策略模式构建一个缓存系统。通过学习本项目，你应该能够：

1. 理解单例模式在缓存管理中的应用
2. 掌握代理模式实现透明缓存
3. 理解策略模式在淘汰算法中的应用
4. 学会设计和实现缓存系统

继续探索和实践，你会发现设计模式的强大之处！
