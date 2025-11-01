# 练习 1: 实现线程安全的缓存管理器

## 难度
⭐⭐ (中等)

## 学习目标
- 掌握单例模式的实现
- 理解线程安全的重要性
- 学会使用 sync.Once
- 实践缓存管理

## 问题描述

实现一个线程安全的缓存管理器，使用单例模式确保全局只有一个缓存实例。该缓存管理器应该支持基本的 CRUD 操作，并能够设置过期时间。

## 功能要求

1. **单例模式**
   - 全局唯一实例
   - 使用 sync.Once 实现
   - 线程安全

2. **基本操作**
   - `Set(key string, value interface{})` - 设置缓存
   - `Get(key string) (interface{}, bool)` - 获取缓存
   - `Delete(key string)` - 删除缓存
   - `Clear()` - 清空所有缓存
   - `Size() int` - 获取缓存数量

3. **过期时间**
   - `SetWithExpiry(key string, value interface{}, duration time.Duration)` - 设置带过期时间的缓存
   - 自动清理过期的缓存项

4. **线程安全**
   - 支持并发读写
   - 使用读写锁优化性能

5. **统计信息**
   - `GetStats() CacheStats` - 获取缓存统计信息
   - 统计命中率、未命中次数等

## 输入输出示例

### 示例 1: 基本操作
**代码**:
```go
cache := GetCacheInstance()

// 设置缓存
cache.Set("user:1", map[string]string{"name": "Alice", "email": "alice@example.com"})
cache.Set("user:2", map[string]string{"name": "Bob", "email": "bob@example.com"})

// 获取缓存
if value, ok := cache.Get("user:1"); ok {
    fmt.Println("Found:", value)
}

// 删除缓存
cache.Delete("user:2")

// 获取大小
fmt.Println("Cache size:", cache.Size())
```

**输出**:
```
Found: map[email:alice@example.com name:Alice]
Cache size: 1
```

### 示例 2: 过期时间
**代码**:
```go
cache := GetCacheInstance()

// 设置 5 秒后过期
cache.SetWithExpiry("session:abc", "user123", 5*time.Second)

// 立即获取
if value, ok := cache.Get("session:abc"); ok {
    fmt.Println("Session found:", value)
}

// 等待 6 秒
time.Sleep(6 * time.Second)

// 再次获取
if _, ok := cache.Get("session:abc"); !ok {
    fmt.Println("Session expired")
}
```

**输出**:
```
Session found: user123
Session expired
```

### 示例 3: 并发访问
**代码**:
```go
cache := GetCacheInstance()
var wg sync.WaitGroup

// 并发写入
for i := 0; i < 100; i++ {
    wg.Add(1)
    go func(index int) {
        defer wg.Done()
        key := fmt.Sprintf("key:%d", index)
        cache.Set(key, index)
    }(i)
}

// 并发读取
for i := 0; i < 100; i++ {
    wg.Add(1)
    go func(index int) {
        defer wg.Done()
        key := fmt.Sprintf("key:%d", index)
        cache.Get(key)
    }(i)
}

wg.Wait()
fmt.Println("Cache size:", cache.Size())
```

**输出**:
```
Cache size: 100
```

## 数据结构

```go
type CacheItem struct {
    Value      interface{}
    ExpireTime time.Time
}

type CacheStats struct {
    Hits       int64
    Misses     int64
    TotalItems int
}

type Cache struct {
    data  map[string]*CacheItem
    mu    sync.RWMutex
    stats CacheStats
}
```

## 提示

💡 **提示 1**: 使用 sync.Once 确保单例
```go
var (
    instance *Cache
    once     sync.Once
)

func GetCacheInstance() *Cache {
    once.Do(func() {
        instance = &Cache{
            data: make(map[string]*CacheItem),
        }
        // 启动清理 goroutine
        go instance.cleanupExpired()
    })
    return instance
}
```

💡 **提示 2**: 使用读写锁优化性能
```go
func (c *Cache) Get(key string) (interface{}, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    item, ok := c.data[key]
    if !ok {
        c.stats.Misses++
        return nil, false
    }
    
    // 检查是否过期
    if !item.ExpireTime.IsZero() && time.Now().After(item.ExpireTime) {
        c.stats.Misses++
        return nil, false
    }
    
    c.stats.Hits++
    return item.Value, true
}
```

💡 **提示 3**: 定期清理过期项
```go
func (c *Cache) cleanupExpired() {
    ticker := time.NewTicker(1 * time.Minute)
    defer ticker.Stop()
    
    for range ticker.C {
        c.mu.Lock()
        now := time.Now()
        for key, item := range c.data {
            if !item.ExpireTime.IsZero() && now.After(item.ExpireTime) {
                delete(c.data, key)
            }
        }
        c.mu.Unlock()
    }
}
```

💡 **提示 4**: 提供测试重置方法
```go
// 仅用于测试
func ResetCacheInstance() {
    instance = nil
    once = sync.Once{}
}
```

## 评分标准

- [ ] **单例实现 (30%)**
  - 使用 sync.Once
  - 全局唯一实例
  - 线程安全

- [ ] **功能完整性 (40%)**
  - 实现所有基本操作
  - 支持过期时间
  - 自动清理过期项
  - 统计信息

- [ ] **线程安全 (20%)**
  - 使用读写锁
  - 支持并发访问
  - 无竞态条件

- [ ] **代码质量 (10%)**
  - 代码结构清晰
  - 命名规范
  - 适当的注释

## 扩展挑战

如果你完成了基本要求，可以尝试以下扩展功能：

1. **LRU 淘汰策略**
   ```go
   type LRUCache struct {
       *Cache
       maxSize int
       lruList *list.List
   }
   
   func (c *LRUCache) Set(key string, value interface{}) {
       if c.Size() >= c.maxSize {
           // 淘汰最久未使用的项
           c.evictLRU()
       }
       c.Cache.Set(key, value)
   }
   ```

2. **持久化**
   ```go
   func (c *Cache) SaveToFile(filename string) error {
       // 序列化到文件
   }
   
   func (c *Cache) LoadFromFile(filename string) error {
       // 从文件加载
   }
   ```

3. **分片锁**
   ```go
   type ShardedCache struct {
       shards []*Cache
       count  int
   }
   
   func (c *ShardedCache) getShard(key string) *Cache {
       hash := fnv.New32()
       hash.Write([]byte(key))
       return c.shards[hash.Sum32()%uint32(c.count)]
   }
   ```

4. **事件通知**
   ```go
   type CacheEvent struct {
       Type  string // "set", "get", "delete"
       Key   string
       Value interface{}
   }
   
   func (c *Cache) Subscribe(handler func(CacheEvent)) {
       // 订阅缓存事件
   }
   ```

## 参考资源

- [sync.Once 文档](https://pkg.go.dev/sync#Once)
- [sync.RWMutex 文档](https://pkg.go.dev/sync#RWMutex)
- [单例模式详解](../theory/01-singleton.md)

## 提交要求

1. 实现 `Cache` 结构体和相关方法
2. 编写单元测试验证功能
3. 编写并发测试验证线程安全
4. 提供使用示例
5. 添加必要的注释和文档

---

**预计完成时间**: 1-2 小时  
**难度评估**: 中等  
**重点考察**: 单例模式、线程安全、并发编程
