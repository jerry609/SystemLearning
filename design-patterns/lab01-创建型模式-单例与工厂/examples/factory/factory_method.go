package main

import (
	"fmt"
	"strings"
	"time"
)

// Database 数据库接口 - 完整版本
type Database interface {
	Connect() error
	Close() error
	Query(sql string) ([]map[string]interface{}, error)
	Execute(sql string) error
	Ping() error
	GetConnectionInfo() string
	IsConnected() bool
}

// BaseDatabase 基础数据库结构（包含公共字段）
type BaseDatabase struct {
	host      string
	port      int
	username  string
	password  string
	database  string
	connected bool
	connTime  time.Time
}

// MySQL 数据库 - 完整实现
type MySQL struct {
	BaseDatabase
	charset string
	timeout time.Duration
}

func (m *MySQL) Connect() error {
	fmt.Printf("正在连接到 MySQL: %s:%d/%s...\n", m.host, m.port, m.database)
	time.Sleep(100 * time.Millisecond) // 模拟连接延迟
	
	if m.host == "" {
		return fmt.Errorf("MySQL: host cannot be empty")
	}
	
	m.connected = true
	m.connTime = time.Now()
	fmt.Printf("✅ MySQL 连接成功 (charset: %s, timeout: %v)\n", m.charset, m.timeout)
	return nil
}

func (m *MySQL) Close() error {
	if !m.connected {
		return fmt.Errorf("MySQL: not connected")
	}
	
	duration := time.Since(m.connTime)
	fmt.Printf("关闭 MySQL 连接 (连接时长: %v)\n", duration.Round(time.Millisecond))
	m.connected = false
	return nil
}

func (m *MySQL) Query(sql string) ([]map[string]interface{}, error) {
	if !m.connected {
		return nil, fmt.Errorf("MySQL: not connected")
	}
	
	fmt.Printf("MySQL 查询: %s\n", sql)
	
	// 模拟查询结果
	results := []map[string]interface{}{
		{"id": 1, "name": "Alice", "email": "alice@example.com"},
		{"id": 2, "name": "Bob", "email": "bob@example.com"},
	}
	
	fmt.Printf("✅ 返回 %d 行数据\n", len(results))
	return results, nil
}

func (m *MySQL) Execute(sql string) error {
	if !m.connected {
		return fmt.Errorf("MySQL: not connected")
	}
	
	fmt.Printf("MySQL 执行: %s\n", sql)
	
	// 模拟执行时间
	time.Sleep(50 * time.Millisecond)
	
	if strings.Contains(strings.ToUpper(sql), "DROP") {
		return fmt.Errorf("MySQL: DROP operation not allowed")
	}
	
	fmt.Println("✅ 执行成功")
	return nil
}

func (m *MySQL) Ping() error {
	if !m.connected {
		return fmt.Errorf("MySQL: not connected")
	}
	fmt.Println("MySQL Ping: OK")
	return nil
}

func (m *MySQL) GetConnectionInfo() string {
	return fmt.Sprintf("MySQL://%s@%s:%d/%s", m.username, m.host, m.port, m.database)
}

func (m *MySQL) IsConnected() bool {
	return m.connected
}

// PostgreSQL 数据库 - 完整实现
type PostgreSQL struct {
	BaseDatabase
	sslMode string
	schema  string
}

func (p *PostgreSQL) Connect() error {
	fmt.Printf("正在连接到 PostgreSQL: %s:%d/%s...\n", p.host, p.port, p.database)
	time.Sleep(100 * time.Millisecond)
	
	if p.host == "" {
		return fmt.Errorf("PostgreSQL: host cannot be empty")
	}
	
	p.connected = true
	p.connTime = time.Now()
	fmt.Printf("✅ PostgreSQL 连接成功 (sslMode: %s, schema: %s)\n", p.sslMode, p.schema)
	return nil
}

func (p *PostgreSQL) Close() error {
	if !p.connected {
		return fmt.Errorf("PostgreSQL: not connected")
	}
	
	duration := time.Since(p.connTime)
	fmt.Printf("关闭 PostgreSQL 连接 (连接时长: %v)\n", duration.Round(time.Millisecond))
	p.connected = false
	return nil
}

func (p *PostgreSQL) Query(sql string) ([]map[string]interface{}, error) {
	if !p.connected {
		return nil, fmt.Errorf("PostgreSQL: not connected")
	}
	
	fmt.Printf("PostgreSQL 查询: %s\n", sql)
	
	results := []map[string]interface{}{
		{"id": 1, "product": "Laptop", "price": 999.99},
		{"id": 2, "product": "Mouse", "price": 29.99},
	}
	
	fmt.Printf("✅ 返回 %d 行数据\n", len(results))
	return results, nil
}

func (p *PostgreSQL) Execute(sql string) error {
	if !p.connected {
		return fmt.Errorf("PostgreSQL: not connected")
	}
	
	fmt.Printf("PostgreSQL 执行: %s\n", sql)
	time.Sleep(50 * time.Millisecond)
	
	if strings.Contains(strings.ToUpper(sql), "DROP") {
		return fmt.Errorf("PostgreSQL: DROP operation not allowed")
	}
	
	fmt.Println("✅ 执行成功")
	return nil
}

func (p *PostgreSQL) Ping() error {
	if !p.connected {
		return fmt.Errorf("PostgreSQL: not connected")
	}
	fmt.Println("PostgreSQL Ping: OK")
	return nil
}

func (p *PostgreSQL) GetConnectionInfo() string {
	return fmt.Sprintf("PostgreSQL://%s@%s:%d/%s", p.username, p.host, p.port, p.database)
}

func (p *PostgreSQL) IsConnected() bool {
	return p.connected
}

// MongoDB 数据库 - 完整实现
type MongoDB struct {
	BaseDatabase
	authSource string
	replicaSet string
}

func (m *MongoDB) Connect() error {
	fmt.Printf("正在连接到 MongoDB: %s:%d/%s...\n", m.host, m.port, m.database)
	time.Sleep(100 * time.Millisecond)
	
	if m.host == "" {
		return fmt.Errorf("MongoDB: host cannot be empty")
	}
	
	m.connected = true
	m.connTime = time.Now()
	fmt.Printf("✅ MongoDB 连接成功 (authSource: %s)\n", m.authSource)
	return nil
}

func (m *MongoDB) Close() error {
	if !m.connected {
		return fmt.Errorf("MongoDB: not connected")
	}
	
	duration := time.Since(m.connTime)
	fmt.Printf("关闭 MongoDB 连接 (连接时长: %v)\n", duration.Round(time.Millisecond))
	m.connected = false
	return nil
}

func (m *MongoDB) Query(sql string) ([]map[string]interface{}, error) {
	if !m.connected {
		return nil, fmt.Errorf("MongoDB: not connected")
	}
	
	fmt.Printf("MongoDB 查询: %s\n", sql)
	
	results := []map[string]interface{}{
		{"_id": "507f1f77bcf86cd799439011", "title": "MongoDB Guide", "views": 1500},
		{"_id": "507f1f77bcf86cd799439012", "title": "NoSQL Basics", "views": 2300},
	}
	
	fmt.Printf("✅ 返回 %d 个文档\n", len(results))
	return results, nil
}

func (m *MongoDB) Execute(sql string) error {
	if !m.connected {
		return fmt.Errorf("MongoDB: not connected")
	}
	
	fmt.Printf("MongoDB 执行: %s\n", sql)
	time.Sleep(50 * time.Millisecond)
	fmt.Println("✅ 执行成功")
	return nil
}

func (m *MongoDB) Ping() error {
	if !m.connected {
		return fmt.Errorf("MongoDB: not connected")
	}
	fmt.Println("MongoDB Ping: OK")
	return nil
}

func (m *MongoDB) GetConnectionInfo() string {
	return fmt.Sprintf("MongoDB://%s@%s:%d/%s", m.username, m.host, m.port, m.database)
}

func (m *MongoDB) IsConnected() bool {
	return m.connected
}

// DatabaseConfig 数据库配置
type DatabaseConfig struct {
	Host     string
	Port     int
	Username string
	Password string
	Database string
}

// DatabaseFactory 数据库工厂接口
type DatabaseFactory interface {
	CreateDatabase(config *DatabaseConfig) (Database, error)
	GetDatabaseType() string
}

// MySQLFactory MySQL 工厂 - 完整实现
type MySQLFactory struct {
	charset string
	timeout time.Duration
}

func (f *MySQLFactory) CreateDatabase(config *DatabaseConfig) (Database, error) {
	if config.Host == "" {
		return nil, fmt.Errorf("MySQL: host is required")
	}
	if config.Port == 0 {
		config.Port = 3306 // 默认端口
	}
	
	return &MySQL{
		BaseDatabase: BaseDatabase{
			host:     config.Host,
			port:     config.Port,
			username: config.Username,
			password: config.Password,
			database: config.Database,
		},
		charset: f.charset,
		timeout: f.timeout,
	}, nil
}

func (f *MySQLFactory) GetDatabaseType() string {
	return "MySQL"
}

// PostgreSQLFactory PostgreSQL 工厂 - 完整实现
type PostgreSQLFactory struct {
	sslMode string
	schema  string
}

func (f *PostgreSQLFactory) CreateDatabase(config *DatabaseConfig) (Database, error) {
	if config.Host == "" {
		return nil, fmt.Errorf("PostgreSQL: host is required")
	}
	if config.Port == 0 {
		config.Port = 5432 // 默认端口
	}
	
	return &PostgreSQL{
		BaseDatabase: BaseDatabase{
			host:     config.Host,
			port:     config.Port,
			username: config.Username,
			password: config.Password,
			database: config.Database,
		},
		sslMode: f.sslMode,
		schema:  f.schema,
	}, nil
}

func (f *PostgreSQLFactory) GetDatabaseType() string {
	return "PostgreSQL"
}

// MongoDBFactory MongoDB 工厂 - 完整实现
type MongoDBFactory struct {
	authSource string
	replicaSet string
}

func (f *MongoDBFactory) CreateDatabase(config *DatabaseConfig) (Database, error) {
	if config.Host == "" {
		return nil, fmt.Errorf("MongoDB: host is required")
	}
	if config.Port == 0 {
		config.Port = 27017 // 默认端口
	}
	
	return &MongoDB{
		BaseDatabase: BaseDatabase{
			host:     config.Host,
			port:     config.Port,
			username: config.Username,
			password: config.Password,
			database: config.Database,
		},
		authSource: f.authSource,
		replicaSet: f.replicaSet,
	}, nil
}

func (f *MongoDBFactory) GetDatabaseType() string {
	return "MongoDB"
}

// 客户端代码 - 完整实现
func executeQuery(factory DatabaseFactory, config *DatabaseConfig, sql string) error {
	fmt.Printf("\n使用 %s 工厂创建数据库连接\n", factory.GetDatabaseType())
	fmt.Println(strings.Repeat("=", 50))
	
	db, err := factory.CreateDatabase(config)
	if err != nil {
		return fmt.Errorf("创建数据库失败: %w", err)
	}
	
	// 连接数据库
	if err := db.Connect(); err != nil {
		return fmt.Errorf("连接失败: %w", err)
	}
	defer db.Close()
	
	// 检查连接
	if err := db.Ping(); err != nil {
		return fmt.Errorf("Ping 失败: %w", err)
	}
	
	// 执行查询
	results, err := db.Query(sql)
	if err != nil {
		return fmt.Errorf("查询失败: %w", err)
	}
	
	// 显示结果
	fmt.Println("\n查询结果:")
	for i, row := range results {
		fmt.Printf("  Row %d: %v\n", i+1, row)
	}
	
	return nil
}

func main() {
	fmt.Println("=== 工厂方法模式示例（完整版）===\n")

	// 示例 1: 使用 MySQL
	fmt.Println("【示例 1: MySQL 数据库】")
	mysqlFactory := &MySQLFactory{
		charset: "utf8mb4",
		timeout: 30 * time.Second,
	}
	
	mysqlConfig := &DatabaseConfig{
		Host:     "localhost",
		Port:     3306,
		Username: "root",
		Password: "password",
		Database: "testdb",
	}
	
	if err := executeQuery(mysqlFactory, mysqlConfig, "SELECT * FROM users WHERE age > 18"); err != nil {
		fmt.Printf("❌ 错误: %v\n", err)
	}

	// 示例 2: 使用 PostgreSQL
	fmt.Println("\n\n【示例 2: PostgreSQL 数据库】")
	pgFactory := &PostgreSQLFactory{
		sslMode: "require",
		schema:  "public",
	}
	
	pgConfig := &DatabaseConfig{
		Host:     "localhost",
		Port:     5432,
		Username: "postgres",
		Password: "password",
		Database: "ordersdb",
	}
	
	if err := executeQuery(pgFactory, pgConfig, "SELECT * FROM orders WHERE status = 'completed'"); err != nil {
		fmt.Printf("❌ 错误: %v\n", err)
	}

	// 示例 3: 使用 MongoDB
	fmt.Println("\n\n【示例 3: MongoDB 数据库】")
	mongoFactory := &MongoDBFactory{
		authSource: "admin",
		replicaSet: "",
	}
	
	mongoConfig := &DatabaseConfig{
		Host:     "localhost",
		Port:     27017,
		Username: "admin",
		Password: "password",
		Database: "productsdb",
	}
	
	if err := executeQuery(mongoFactory, mongoConfig, "db.products.find({category: 'electronics'})"); err != nil {
		fmt.Printf("❌ 错误: %v\n", err)
	}

	// 示例 4: 动态选择工厂
	fmt.Println("\n\n【示例 4: 动态选择数据库类型】")
	fmt.Println(strings.Repeat("=", 50))
	
	dbType := "mysql" // 可以从配置文件或环境变量读取
	var factory DatabaseFactory
	
	switch dbType {
	case "mysql":
		factory = &MySQLFactory{charset: "utf8mb4", timeout: 30 * time.Second}
	case "postgres":
		factory = &PostgreSQLFactory{sslMode: "disable", schema: "public"}
	case "mongodb":
		factory = &MongoDBFactory{authSource: "admin"}
	default:
		fmt.Printf("❌ 不支持的数据库类型: %s\n", dbType)
		return
	}
	
	fmt.Printf("✅ 选择了 %s 数据库\n", factory.GetDatabaseType())
	
	config := &DatabaseConfig{
		Host:     "localhost",
		Username: "user",
		Password: "pass",
		Database: "mydb",
	}
	
	if err := executeQuery(factory, config, "SELECT COUNT(*) FROM users"); err != nil {
		fmt.Printf("❌ 错误: %v\n", err)
	}

	// 示例 5: 错误处理
	fmt.Println("\n\n【示例 5: 错误处理】")
	fmt.Println(strings.Repeat("=", 50))
	
	invalidConfig := &DatabaseConfig{
		Host: "", // 空主机名
	}
	
	_, err := mysqlFactory.CreateDatabase(invalidConfig)
	if err != nil {
		fmt.Printf("✅ 正确捕获错误: %v\n", err)
	}

	fmt.Println("\n\n=== 示例结束 ===")
	fmt.Println("\n工厂方法模式特点:")
	fmt.Println("✅ 优点：")
	fmt.Println("  - 符合开闭原则（添加新数据库只需新增工厂类）")
	fmt.Println("  - 符合单一职责原则（每个工厂只负责创建一种产品）")
	fmt.Println("  - 客户端与具体产品解耦")
	fmt.Println("  - 易于扩展新的数据库类型")
	fmt.Println("  - 支持依赖注入")
	fmt.Println("\n❌ 缺点：")
	fmt.Println("  - 类的数量增加（每个产品需要一个工厂）")
	fmt.Println("  - 增加了系统复杂度")
	fmt.Println("  - 需要更多的代码")
	
	fmt.Println("\n💡 适用场景:")
	fmt.Println("  - 产品类型可能扩展")
	fmt.Println("  - 需要高度可扩展性")
	fmt.Println("  - 框架设计")
	fmt.Println("  - 插件系统")
}
