# 练习 3: 实现智能数据库连接代理

## 难度
⭐⭐ (中等)

## 学习目标
- 掌握代理模式的实现
- 理解虚拟代理和保护代理
- 学会延迟加载和访问控制
- 实践连接池管理

## 问题描述

实现一个智能的数据库连接代理系统，提供连接池管理、延迟连接、访问控制、查询缓存等功能。代理应该对客户端透明，提供与真实数据库连接相同的接口。

## 功能要求

1. **数据库连接接口**
   - 定义统一的 `DBConnection` 接口
   - 支持基本的 CRUD 操作
   - 支持事务操作

2. **虚拟代理（延迟加载）**
   - 延迟创建数据库连接
   - 首次使用时才建立连接
   - 减少资源占用

3. **保护代理（访问控制）**
   - 实现权限检查
   - 支持只读模式
   - 记录敏感操作

4. **缓存代理**
   - 缓存查询结果
   - 支持缓存失效
   - 提高查询性能

5. **连接池代理**
   - 管理多个数据库连接
   - 自动复用连接
   - 限制最大连接数
   - 连接健康检查

6. **日志代理**
   - 记录所有数据库操作
   - 记录执行时间
   - 记录慢查询

## 输入输出示例

### 示例 1: 虚拟代理（延迟加载）
**代码**:
```go
// 创建虚拟代理，此时不建立连接
proxy := NewLazyDBProxy(DBConfig{
    Host:     "localhost",
    Port:     5432,
    Database: "myapp",
    User:     "admin",
    Password: "secret",
})

fmt.Println("Proxy created, no connection yet")

// 首次查询时才建立连接
result, err := proxy.Query("SELECT * FROM users WHERE id = ?", 1)
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Found user: %v\n", result)
```

**输出**:
```
Proxy created, no connection yet
[LazyProxy] Establishing database connection...
[LazyProxy] Connection established successfully
Found user: map[id:1 name:Alice email:alice@example.com]
```

### 示例 2: 保护代理（访问控制）
**代码**:
```go
// 创建真实连接
realConn := NewRealDBConnection(config)

// 创建保护代理，设置为只读模式
proxy := NewProtectionProxy(realConn, ProtectionConfig{
    ReadOnly: true,
    AllowedTables: []string{"users", "posts"},
})

// 查询操作（允许）
result, _ := proxy.Query("SELECT * FROM users")
fmt.Println("Query succeeded:", len(result))

// 写入操作（拒绝）
err := proxy.Execute("DELETE FROM users WHERE id = 1")
if err != nil {
    fmt.Println("Error:", err)
}

// 访问未授权的表（拒绝）
_, err = proxy.Query("SELECT * FROM admin_logs")
if err != nil {
    fmt.Println("Error:", err)
}
```

**输出**:
```
Query succeeded: 10
Error: operation not allowed: write operations are disabled in read-only mode
Error: access denied: table 'admin_logs' is not in allowed list
```

### 示例 3: 连接池代理
**代码**:
```go
// 创建连接池代理
pool := NewConnectionPoolProxy(PoolConfig{
    MaxConnections: 5,
    MinConnections: 2,
    MaxIdleTime:    5 * time.Minute,
})

// 并发执行查询
var wg sync.WaitGroup
for i := 0; i < 10; i++ {
    wg.Add(1)
    go func(id int) {
        defer wg.Done()
        
        // 从池中获取连接
        conn, err := pool.GetConnection()
        if err != nil {
            log.Printf("Worker %d: failed to get connection: %v", id, err)
            return
        }
        defer pool.ReleaseConnection(conn)
        
        // 执行查询
        result, _ := conn.Query("SELECT * FROM users WHERE id = ?", id)
        fmt.Printf("Worker %d: got %d results\n", id, len(result))
        
        // 模拟处理时间
        time.Sleep(100 * time.Millisecond)
    }(i)
}

wg.Wait()
fmt.Printf("Pool stats: %+v\n", pool.GetStats())
```

**输出**:
```
[Pool] Creating new connection (1/5)
[Pool] Creating new connection (2/5)
[Pool] Creating new connection (3/5)
Worker 0: got 1 results
Worker 1: got 1 results
[Pool] Reusing connection from pool
Worker 2: got 1 results
[Pool] Reusing connection from pool
Worker 3: got 1 results
Worker 4: got 1 results
Worker 5: got 1 results
Worker 6: got 1 results
Worker 7: got 1 results
Worker 8: got 1 results
Worker 9: got 1 results
Pool stats: {TotalConnections:3 ActiveConnections:0 IdleConnections:3 TotalQueries:10}
```

### 示例 4: 组合多个代理
**代码**:
```go
// 创建真实连接
realConn := NewRealDBConnection(config)

// 添加缓存代理
cachedConn := NewCacheProxy(realConn, CacheConfig{
    TTL:     5 * time.Minute,
    MaxSize: 1000,
})

// 添加日志代理
loggedConn := NewLoggingProxy(cachedConn)

// 添加保护代理
protectedConn := NewProtectionProxy(loggedConn, ProtectionConfig{
    ReadOnly: false,
    AllowedTables: []string{"users", "posts", "comments"},
})

// 第一次查询（从数据库）
result1, _ := protectedConn.Query("SELECT * FROM users WHERE age > ?", 18)
fmt.Printf("First query: %d results\n", len(result1))

// 第二次相同查询（从缓存）
result2, _ := protectedConn.Query("SELECT * FROM users WHERE age > ?", 18)
fmt.Printf("Second query: %d results\n", len(result2))
```

**输出**:
```
[Log] Query: SELECT * FROM users WHERE age > ? [18]
[Cache] Cache miss, querying database
[DB] Executing query...
[Log] Query completed in 45ms
First query: 25 results

[Log] Query: SELECT * FROM users WHERE age > ? [18]
[Cache] Cache hit
[Log] Query completed in 0.1ms
Second query: 25 results
```

## 接口定义

```go
// DBConnection 数据库连接接口
type DBConnection interface {
    Query(sql string, args ...interface{}) ([]map[string]interface{}, error)
    Execute(sql string, args ...interface{}) error
    BeginTransaction() (Transaction, error)
    Close() error
}

// Transaction 事务接口
type Transaction interface {
    Commit() error
    Rollback() error
    Query(sql string, args ...interface{}) ([]map[string]interface{}, error)
    Execute(sql string, args ...interface{}) error
}

// DBConfig 数据库配置
type DBConfig struct {
    Host     string
    Port     int
    Database string
    User     string
    Password string
}

// ProtectionConfig 保护代理配置
type ProtectionConfig struct {
    ReadOnly      bool
    AllowedTables []string
    AllowedUsers  []string
}

// PoolConfig 连接池配置
type PoolConfig struct {
    MaxConnections int
    MinConnections int
    MaxIdleTime    time.Duration
}

// PoolStats 连接池统计
type PoolStats struct {
    TotalConnections  int
    ActiveConnections int
    IdleConnections   int
    TotalQueries      int64
}
```

## 提示

💡 **提示 1**: 虚拟代理实现
```go
type LazyDBProxy struct {
    config     DBConfig
    realConn   DBConnection
    mu         sync.Mutex
    connected  bool
}

func (p *LazyDBProxy) Query(sql string, args ...interface{}) ([]map[string]interface{}, error) {
    if err := p.ensureConnected(); err != nil {
        return nil, err
    }
    return p.realConn.Query(sql, args...)
}

func (p *LazyDBProxy) ensureConnected() error {
    p.mu.Lock()
    defer p.mu.Unlock()
    
    if !p.connected {
        fmt.Println("[LazyProxy] Establishing database connection...")
        conn, err := NewRealDBConnection(p.config)
        if err != nil {
            return err
        }
        p.realConn = conn
        p.connected = true
        fmt.Println("[LazyProxy] Connection established successfully")
    }
    
    return nil
}
```

💡 **提示 2**: 保护代理实现
```go
type ProtectionProxy struct {
    realConn DBConnection
    config   ProtectionConfig
}

func (p *ProtectionProxy) Execute(sql string, args ...interface{}) error {
    // 检查只读模式
    if p.config.ReadOnly && p.isWriteOperation(sql) {
        return fmt.Errorf("operation not allowed: write operations are disabled in read-only mode")
    }
    
    // 检查表访问权限
    table := p.extractTableName(sql)
    if !p.isTableAllowed(table) {
        return fmt.Errorf("access denied: table '%s' is not in allowed list", table)
    }
    
    return p.realConn.Execute(sql, args...)
}

func (p *ProtectionProxy) isWriteOperation(sql string) bool {
    sql = strings.ToUpper(strings.TrimSpace(sql))
    return strings.HasPrefix(sql, "INSERT") ||
           strings.HasPrefix(sql, "UPDATE") ||
           strings.HasPrefix(sql, "DELETE") ||
           strings.HasPrefix(sql, "DROP") ||
           strings.HasPrefix(sql, "ALTER")
}
```

💡 **提示 3**: 缓存代理实现
```go
type CacheProxy struct {
    realConn DBConnection
    cache    map[string]*cacheEntry
    mu       sync.RWMutex
    config   CacheConfig
}

type cacheEntry struct {
    result    []map[string]interface{}
    expiresAt time.Time
}

func (p *CacheProxy) Query(sql string, args ...interface{}) ([]map[string]interface{}, error) {
    // 生成缓存键
    key := p.generateCacheKey(sql, args...)
    
    // 检查缓存
    p.mu.RLock()
    if entry, ok := p.cache[key]; ok {
        if time.Now().Before(entry.expiresAt) {
            p.mu.RUnlock()
            return entry.result, nil
        }
    }
    p.mu.RUnlock()
    
    // 执行查询
    result, err := p.realConn.Query(sql, args...)
    if err != nil {
        return nil, err
    }
    
    // 存入缓存
    p.mu.Lock()
    p.cache[key] = &cacheEntry{
        result:    result,
        expiresAt: time.Now().Add(p.config.TTL),
    }
    p.mu.Unlock()
    
    return result, nil
}

func (p *CacheProxy) generateCacheKey(sql string, args ...interface{}) string {
    return fmt.Sprintf("%s:%v", sql, args)
}
```

💡 **提示 4**: 连接池代理实现
```go
type ConnectionPoolProxy struct {
    config    PoolConfig
    pool      chan DBConnection
    active    map[DBConnection]bool
    mu        sync.Mutex
    stats     PoolStats
}

func NewConnectionPoolProxy(config PoolConfig) *ConnectionPoolProxy {
    p := &ConnectionPoolProxy{
        config: config,
        pool:   make(chan DBConnection, config.MaxConnections),
        active: make(map[DBConnection]bool),
    }
    
    // 预创建最小连接数
    for i := 0; i < config.MinConnections; i++ {
        conn := p.createConnection()
        p.pool <- conn
    }
    
    return p
}

func (p *ConnectionPoolProxy) GetConnection() (DBConnection, error) {
    select {
    case conn := <-p.pool:
        // 从池中获取连接
        p.mu.Lock()
        p.active[conn] = true
        p.stats.ActiveConnections++
        p.mu.Unlock()
        return conn, nil
    default:
        // 池为空，创建新连接
        p.mu.Lock()
        if p.stats.TotalConnections < p.config.MaxConnections {
            conn := p.createConnection()
            p.active[conn] = true
            p.stats.TotalConnections++
            p.stats.ActiveConnections++
            p.mu.Unlock()
            return conn, nil
        }
        p.mu.Unlock()
        
        // 达到最大连接数，等待可用连接
        conn := <-p.pool
        p.mu.Lock()
        p.active[conn] = true
        p.stats.ActiveConnections++
        p.mu.Unlock()
        return conn, nil
    }
}

func (p *ConnectionPoolProxy) ReleaseConnection(conn DBConnection) {
    p.mu.Lock()
    delete(p.active, conn)
    p.stats.ActiveConnections--
    p.mu.Unlock()
    
    // 放回池中
    p.pool <- conn
}
```

## 评分标准

- [ ] **接口设计 (20%)**
  - 统一的 DBConnection 接口
  - 清晰的配置结构
  - 合理的事务接口

- [ ] **代理实现 (40%)**
  - 实现虚拟代理
  - 实现保护代理
  - 实现缓存代理
  - 实现连接池代理

- [ ] **功能完整性 (25%)**
  - 延迟加载正确
  - 访问控制正确
  - 缓存功能正确
  - 连接池管理正确

- [ ] **代码质量 (15%)**
  - 代码结构清晰
  - 线程安全
  - 适当的错误处理

## 扩展挑战

如果你完成了基本要求，可以尝试以下扩展功能：

1. **智能代理（自动故障转移）**
   ```go
   type SmartProxy struct {
       primary   DBConnection
       secondary DBConnection
       failover  bool
   }
   ```

2. **分片代理**
   ```go
   type ShardingProxy struct {
       shards []DBConnection
       strategy ShardingStrategy
   }
   ```

3. **读写分离代理**
   ```go
   type ReadWriteSplitProxy struct {
       master DBConnection
       slaves []DBConnection
   }
   ```

4. **监控代理**
   ```go
   type MonitoringProxy struct {
       realConn DBConnection
       metrics  *Metrics
   }
   ```

## 参考资源

- [代理模式详解](../theory/03-proxy.md)
- [Go database/sql 包](https://pkg.go.dev/database/sql)
- [连接池设计](https://en.wikipedia.org/wiki/Connection_pool)

## 提交要求

1. 实现 `DBConnection` 接口和真实连接
2. 实现至少 3 个代理类型
3. 编写测试用例验证功能
4. 提供完整的使用示例
5. 添加必要的注释和文档

---

**预计完成时间**: 2-2.5 小时  
**难度评估**: 中等  
**重点考察**: 代理模式、延迟加载、访问控制、连接池管理
