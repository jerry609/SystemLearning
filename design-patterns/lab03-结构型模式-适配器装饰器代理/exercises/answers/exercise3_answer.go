package main

// 练习 3: 实现智能数据库连接代理 - 参考答案
//
// 设计思路:
// 1. 定义统一的 DBConnection 接口
// 2. 实现真实的数据库连接
// 3. 实现虚拟代理（延迟加载）
// 4. 实现保护代理（访问控制）
// 5. 实现缓存代理（查询缓存）
// 6. 实现连接池代理（连接管理）
//
// 使用的设计模式: 代理模式
// 模式应用位置:
// - LazyDBProxy: 虚拟代理，延迟创建连接
// - ProtectionProxy: 保护代理，访问控制
// - CacheProxy: 缓存代理，查询缓存
// - ConnectionPoolProxy: 连接池代理，连接管理

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// ============================================================================
// 接口定义
// ============================================================================

// DBConnection 数据库连接接口
type DBConnection interface {
	Query(sql string, args ...interface{}) ([]map[string]interface{}, error)
	Execute(sql string, args ...interface{}) error
	Close() error
}

// DBConfig 数据库配置
type DBConfig struct {
	Host     string
	Port     int
	Database string
	User     string
	Password string
}

// ============================================================================
// 真实数据库连接
// ============================================================================

// RealDBConnection 真实数据库连接
type RealDBConnection struct {
	config    DBConfig
	connected bool
}

// NewRealDBConnection 创建真实数据库连接
func NewRealDBConnection(config DBConfig) (*RealDBConnection, error) {
	fmt.Printf("[RealDB] Connecting to %s:%d/%s...\n", config.Host, config.Port, config.Database)
	time.Sleep(100 * time.Millisecond) // 模拟连接延迟

	return &RealDBConnection{
		config:    config,
		connected: true,
	}, nil
}

func (c *RealDBConnection) Query(sql string, args ...interface{}) ([]map[string]interface{}, error) {
	if !c.connected {
		return nil, fmt.Errorf("connection is closed")
	}

	fmt.Printf("[RealDB] Executing query: %s %v\n", sql, args)
	time.Sleep(50 * time.Millisecond) // 模拟查询延迟

	// 模拟查询结果
	result := []map[string]interface{}{
		{"id": 1, "name": "Alice", "email": "alice@example.com"},
		{"id": 2, "name": "Bob", "email": "bob@example.com"},
	}

	return result, nil
}

func (c *RealDBConnection) Execute(sql string, args ...interface{}) error {
	if !c.connected {
		return fmt.Errorf("connection is closed")
	}

	fmt.Printf("[RealDB] Executing: %s %v\n", sql, args)
	time.Sleep(50 * time.Millisecond) // 模拟执行延迟

	return nil
}

func (c *RealDBConnection) Close() error {
	fmt.Println("[RealDB] Closing connection")
	c.connected = false
	return nil
}

// ============================================================================
// 虚拟代理（延迟加载）
// ============================================================================

// LazyDBProxy 虚拟代理
type LazyDBProxy struct {
	config    DBConfig
	realConn  DBConnection
	mu        sync.Mutex
	connected bool
}

// NewLazyDBProxy 创建虚拟代理
func NewLazyDBProxy(config DBConfig) *LazyDBProxy {
	return &LazyDBProxy{
		config:    config,
		connected: false,
	}
}

func (p *LazyDBProxy) Query(sql string, args ...interface{}) ([]map[string]interface{}, error) {
	if err := p.ensureConnected(); err != nil {
		return nil, err
	}
	return p.realConn.Query(sql, args...)
}

func (p *LazyDBProxy) Execute(sql string, args ...interface{}) error {
	if err := p.ensureConnected(); err != nil {
		return err
	}
	return p.realConn.Execute(sql, args...)
}

func (p *LazyDBProxy) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.connected && p.realConn != nil {
		return p.realConn.Close()
	}
	return nil
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

// ============================================================================
// 保护代理（访问控制）
// ============================================================================

// ProtectionConfig 保护代理配置
type ProtectionConfig struct {
	ReadOnly      bool
	AllowedTables []string
}

// ProtectionProxy 保护代理
type ProtectionProxy struct {
	realConn DBConnection
	config   ProtectionConfig
}

// NewProtectionProxy 创建保护代理
func NewProtectionProxy(realConn DBConnection, config ProtectionConfig) *ProtectionProxy {
	return &ProtectionProxy{
		realConn: realConn,
		config:   config,
	}
}

func (p *ProtectionProxy) Query(sql string, args ...interface{}) ([]map[string]interface{}, error) {
	// 检查表访问权限
	table := p.extractTableName(sql)
	if !p.isTableAllowed(table) {
		return nil, fmt.Errorf("access denied: table '%s' is not in allowed list", table)
	}

	return p.realConn.Query(sql, args...)
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

func (p *ProtectionProxy) Close() error {
	return p.realConn.Close()
}

func (p *ProtectionProxy) isWriteOperation(sql string) bool {
	sql = strings.ToUpper(strings.TrimSpace(sql))
	return strings.HasPrefix(sql, "INSERT") ||
		strings.HasPrefix(sql, "UPDATE") ||
		strings.HasPrefix(sql, "DELETE") ||
		strings.HasPrefix(sql, "DROP") ||
		strings.HasPrefix(sql, "ALTER")
}

func (p *ProtectionProxy) extractTableName(sql string) string {
	sql = strings.ToUpper(strings.TrimSpace(sql))
	words := strings.Fields(sql)

	// 简单的表名提取逻辑
	for i, word := range words {
		if word == "FROM" || word == "INTO" || word == "UPDATE" {
			if i+1 < len(words) {
				return strings.ToLower(words[i+1])
			}
		}
	}

	return ""
}

func (p *ProtectionProxy) isTableAllowed(table string) bool {
	if len(p.config.AllowedTables) == 0 {
		return true
	}

	for _, allowed := range p.config.AllowedTables {
		if strings.EqualFold(allowed, table) {
			return true
		}
	}

	return false
}

// ============================================================================
// 缓存代理
// ============================================================================

// CacheConfig 缓存配置
type CacheConfig struct {
	TTL     time.Duration
	MaxSize int
}

type cacheEntry struct {
	result    []map[string]interface{}
	expiresAt time.Time
}

// CacheProxy 缓存代理
type CacheProxy struct {
	realConn DBConnection
	cache    map[string]*cacheEntry
	mu       sync.RWMutex
	config   CacheConfig
}

// NewCacheProxy 创建缓存代理
func NewCacheProxy(realConn DBConnection, config CacheConfig) *CacheProxy {
	return &CacheProxy{
		realConn: realConn,
		cache:    make(map[string]*cacheEntry),
		config:   config,
	}
}

func (p *CacheProxy) Query(sql string, args ...interface{}) ([]map[string]interface{}, error) {
	// 生成缓存键
	key := p.generateCacheKey(sql, args...)

	// 检查缓存
	p.mu.RLock()
	if entry, ok := p.cache[key]; ok {
		if time.Now().Before(entry.expiresAt) {
			p.mu.RUnlock()
			fmt.Println("[Cache] Cache hit")
			return entry.result, nil
		}
	}
	p.mu.RUnlock()

	fmt.Println("[Cache] Cache miss, querying database")

	// 执行查询
	result, err := p.realConn.Query(sql, args...)
	if err != nil {
		return nil, err
	}

	// 存入缓存
	p.mu.Lock()
	if len(p.cache) >= p.config.MaxSize {
		// 简单的缓存淘汰：清空所有缓存
		p.cache = make(map[string]*cacheEntry)
	}
	p.cache[key] = &cacheEntry{
		result:    result,
		expiresAt: time.Now().Add(p.config.TTL),
	}
	p.mu.Unlock()

	return result, nil
}

func (p *CacheProxy) Execute(sql string, args ...interface{}) error {
	// 写操作清除缓存
	p.mu.Lock()
	p.cache = make(map[string]*cacheEntry)
	p.mu.Unlock()

	return p.realConn.Execute(sql, args...)
}

func (p *CacheProxy) Close() error {
	return p.realConn.Close()
}

func (p *CacheProxy) generateCacheKey(sql string, args ...interface{}) string {
	return fmt.Sprintf("%s:%v", sql, args)
}

// ============================================================================
// 连接池代理
// ============================================================================

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

// ConnectionPoolProxy 连接池代理
type ConnectionPoolProxy struct {
	config DBConfig
	pool   chan DBConnection
	active map[DBConnection]bool
	mu     sync.Mutex
	stats  PoolStats
}

// NewConnectionPoolProxy 创建连接池代理
func NewConnectionPoolProxy(config DBConfig, poolConfig PoolConfig) *ConnectionPoolProxy {
	p := &ConnectionPoolProxy{
		config: config,
		pool:   make(chan DBConnection, poolConfig.MaxConnections),
		active: make(map[DBConnection]bool),
	}

	// 预创建最小连接数
	for i := 0; i < poolConfig.MinConnections; i++ {
		conn, _ := p.createConnection()
		p.pool <- conn
		p.stats.TotalConnections++
	}

	fmt.Printf("[Pool] Initialized with %d connections\n", poolConfig.MinConnections)

	return p
}

func (p *ConnectionPoolProxy) GetConnection() (DBConnection, error) {
	select {
	case conn := <-p.pool:
		// 从池中获取连接
		p.mu.Lock()
		p.active[conn] = true
		p.stats.ActiveConnections++
		p.stats.IdleConnections--
		p.mu.Unlock()
		fmt.Println("[Pool] Reusing connection from pool")
		return conn, nil
	default:
		// 池为空，创建新连接
		p.mu.Lock()
		if p.stats.TotalConnections < 5 { // 假设最大连接数为 5
			conn, err := p.createConnection()
			if err != nil {
				p.mu.Unlock()
				return nil, err
			}
			p.active[conn] = true
			p.stats.TotalConnections++
			p.stats.ActiveConnections++
			p.mu.Unlock()
			fmt.Printf("[Pool] Creating new connection (%d/5)\n", p.stats.TotalConnections)
			return conn, nil
		}
		p.mu.Unlock()

		// 达到最大连接数，等待可用连接
		fmt.Println("[Pool] Waiting for available connection...")
		conn := <-p.pool
		p.mu.Lock()
		p.active[conn] = true
		p.stats.ActiveConnections++
		p.stats.IdleConnections--
		p.mu.Unlock()
		return conn, nil
	}
}

func (p *ConnectionPoolProxy) ReleaseConnection(conn DBConnection) {
	p.mu.Lock()
	delete(p.active, conn)
	p.stats.ActiveConnections--
	p.stats.IdleConnections++
	p.mu.Unlock()

	// 放回池中
	p.pool <- conn
}

func (p *ConnectionPoolProxy) GetStats() PoolStats {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.stats
}

func (p *ConnectionPoolProxy) createConnection() (DBConnection, error) {
	return NewRealDBConnection(p.config)
}

// ============================================================================
// 日志代理
// ============================================================================

// LoggingProxy 日志代理
type LoggingProxy struct {
	realConn DBConnection
}

// NewLoggingProxy 创建日志代理
func NewLoggingProxy(realConn DBConnection) *LoggingProxy {
	return &LoggingProxy{realConn: realConn}
}

func (p *LoggingProxy) Query(sql string, args ...interface{}) ([]map[string]interface{}, error) {
	fmt.Printf("[Log] Query: %s %v\n", sql, args)
	start := time.Now()

	result, err := p.realConn.Query(sql, args...)

	duration := time.Since(start)
	if err != nil {
		fmt.Printf("[Log] Query failed in %v: %v\n", duration, err)
	} else {
		fmt.Printf("[Log] Query completed in %v\n", duration)
	}

	return result, err
}

func (p *LoggingProxy) Execute(sql string, args ...interface{}) error {
	fmt.Printf("[Log] Execute: %s %v\n", sql, args)
	start := time.Now()

	err := p.realConn.Execute(sql, args...)

	duration := time.Since(start)
	if err != nil {
		fmt.Printf("[Log] Execute failed in %v: %v\n", duration, err)
	} else {
		fmt.Printf("[Log] Execute completed in %v\n", duration)
	}

	return err
}

func (p *LoggingProxy) Close() error {
	fmt.Println("[Log] Closing connection")
	return p.realConn.Close()
}

// ============================================================================
// 示例代码
// ============================================================================

func main() {
	fmt.Println("=== 智能数据库连接代理示例 ===\n")

	config := DBConfig{
		Host:     "localhost",
		Port:     5432,
		Database: "myapp",
		User:     "admin",
		Password: "secret",
	}

	// 示例 1: 虚拟代理（延迟加载）
	fmt.Println("--- 示例 1: 虚拟代理 ---")
	proxy1 := NewLazyDBProxy(config)
	fmt.Println("Proxy created, no connection yet")

	result1, _ := proxy1.Query("SELECT * FROM users WHERE id = ?", 1)
	fmt.Printf("Found %d users\n\n", len(result1))

	// 示例 2: 保护代理（访问控制）
	fmt.Println("--- 示例 2: 保护代理 ---")
	realConn, _ := NewRealDBConnection(config)
	proxy2 := NewProtectionProxy(realConn, ProtectionConfig{
		ReadOnly:      true,
		AllowedTables: []string{"users", "posts"},
	})

	// 查询操作（允许）
	result2, _ := proxy2.Query("SELECT * FROM users")
	fmt.Printf("Query succeeded: %d results\n", len(result2))

	// 写入操作（拒绝）
	err := proxy2.Execute("DELETE FROM users WHERE id = 1")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	}

	// 访问未授权的表（拒绝）
	_, err = proxy2.Query("SELECT * FROM admin_logs")
	if err != nil {
		fmt.Printf("Error: %v\n\n", err)
	}

	// 示例 3: 缓存代理
	fmt.Println("--- 示例 3: 缓存代理 ---")
	realConn2, _ := NewRealDBConnection(config)
	proxy3 := NewCacheProxy(realConn2, CacheConfig{
		TTL:     5 * time.Minute,
		MaxSize: 100,
	})

	// 第一次查询
	result3, _ := proxy3.Query("SELECT * FROM users WHERE age > ?", 18)
	fmt.Printf("First query: %d results\n", len(result3))

	// 第二次相同查询（从缓存）
	result4, _ := proxy3.Query("SELECT * FROM users WHERE age > ?", 18)
	fmt.Printf("Second query: %d results\n\n", len(result4))

	// 示例 4: 组合多个代理
	fmt.Println("--- 示例 4: 组合代理 ---")
	realConn3, _ := NewRealDBConnection(config)
	cachedConn := NewCacheProxy(realConn3, CacheConfig{
		TTL:     5 * time.Minute,
		MaxSize: 1000,
	})
	loggedConn := NewLoggingProxy(cachedConn)
	protectedConn := NewProtectionProxy(loggedConn, ProtectionConfig{
		ReadOnly:      false,
		AllowedTables: []string{"users", "posts", "comments"},
	})

	result5, _ := protectedConn.Query("SELECT * FROM users WHERE age > ?", 18)
	fmt.Printf("Combined proxy query: %d results\n", len(result5))

	fmt.Println("\n=== 示例结束 ===")
}

// 可能的优化方向:
// 1. 实现智能代理（自动故障转移）
// 2. 实现分片代理
// 3. 实现读写分离代理
// 4. 添加监控代理
// 5. 实现更智能的连接池管理
//
// 变体实现:
// 1. 使用接口组合
// 2. 实现代理工厂
// 3. 添加代理链模式
