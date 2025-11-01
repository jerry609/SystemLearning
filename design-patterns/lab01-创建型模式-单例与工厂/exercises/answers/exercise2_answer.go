package main

import (
	"fmt"
	"time"
)

// 练习 2: 数据库连接工厂 - 参考答案

// DBConfig 数据库配置
type DBConfig struct {
	Host     string
	Port     int
	Username string
	Password string
	Database string
	MaxConns int
	Timeout  time.Duration
}

// Database 数据库接口
type Database interface {
	Connect() error
	Close() error
	Query(sql string) ([]map[string]interface{}, error)
	Execute(sql string) error
	Ping() error
	GetType() string
}

// ========== MySQL 实现 ==========

// MySQLDatabase MySQL 数据库
type MySQLDatabase struct {
	config    *DBConfig
	connected bool
}

func (db *MySQLDatabase) Connect() error {
	fmt.Printf("连接到 MySQL: %s:%d/%s\n", db.config.Host, db.config.Port, db.config.Database)
	db.connected = true
	return nil
}

func (db *MySQLDatabase) Close() error {
	fmt.Println("关闭 MySQL 连接")
	db.connected = false
	return nil
}

func (db *MySQLDatabase) Query(sql string) ([]map[string]interface{}, error) {
	fmt.Printf("MySQL 查询: %s\n", sql)
	return []map[string]interface{}{
		{"id": 1, "name": "Alice"},
		{"id": 2, "name": "Bob"},
	}, nil
}

func (db *MySQLDatabase) Execute(sql string) error {
	fmt.Printf("MySQL 执行: %s\n", sql)
	return nil
}

func (db *MySQLDatabase) Ping() error {
	if !db.connected {
		return fmt.Errorf("not connected")
	}
	fmt.Println("MySQL Ping: OK")
	return nil
}

func (db *MySQLDatabase) GetType() string {
	return "MySQL"
}

// ========== PostgreSQL 实现 ==========

// PostgreSQLDatabase PostgreSQL 数据库
type PostgreSQLDatabase struct {
	config    *DBConfig
	connected bool
}

func (db *PostgreSQLDatabase) Connect() error {
	fmt.Printf("连接到 PostgreSQL: %s:%d/%s\n", db.config.Host, db.config.Port, db.config.Database)
	db.connected = true
	return nil
}

func (db *PostgreSQLDatabase) Close() error {
	fmt.Println("关闭 PostgreSQL 连接")
	db.connected = false
	return nil
}

func (db *PostgreSQLDatabase) Query(sql string) ([]map[string]interface{}, error) {
	fmt.Printf("PostgreSQL 查询: %s\n", sql)
	return []map[string]interface{}{
		{"id": 1, "name": "Charlie"},
		{"id": 2, "name": "David"},
	}, nil
}

func (db *PostgreSQLDatabase) Execute(sql string) error {
	fmt.Printf("PostgreSQL 执行: %s\n", sql)
	return nil
}

func (db *PostgreSQLDatabase) Ping() error {
	if !db.connected {
		return fmt.Errorf("not connected")
	}
	fmt.Println("PostgreSQL Ping: OK")
	return nil
}

func (db *PostgreSQLDatabase) GetType() string {
	return "PostgreSQL"
}

// ========== MongoDB 实现 ==========

// MongoDBDatabase MongoDB 数据库
type MongoDBDatabase struct {
	config    *DBConfig
	connected bool
}

func (db *MongoDBDatabase) Connect() error {
	fmt.Printf("连接到 MongoDB: %s:%d/%s\n", db.config.Host, db.config.Port, db.config.Database)
	db.connected = true
	return nil
}

func (db *MongoDBDatabase) Close() error {
	fmt.Println("关闭 MongoDB 连接")
	db.connected = false
	return nil
}

func (db *MongoDBDatabase) Query(sql string) ([]map[string]interface{}, error) {
	fmt.Printf("MongoDB 查询: %s\n", sql)
	return []map[string]interface{}{
		{"_id": "1", "name": "Eve"},
		{"_id": "2", "name": "Frank"},
	}, nil
}

func (db *MongoDBDatabase) Execute(sql string) error {
	fmt.Printf("MongoDB 执行: %s\n", sql)
	return nil
}

func (db *MongoDBDatabase) Ping() error {
	if !db.connected {
		return fmt.Errorf("not connected")
	}
	fmt.Println("MongoDB Ping: OK")
	return nil
}

func (db *MongoDBDatabase) GetType() string {
	return "MongoDB"
}

// ========== 简单工厂 ==========

// CreateDatabase 简单工厂方法
func CreateDatabase(dbType string, config *DBConfig) (Database, error) {
	switch dbType {
	case "mysql":
		return &MySQLDatabase{config: config}, nil
	case "postgres":
		return &PostgreSQLDatabase{config: config}, nil
	case "mongodb":
		return &MongoDBDatabase{config: config}, nil
	default:
		return nil, fmt.Errorf("unsupported database type: %s", dbType)
	}
}

// ========== 工厂方法 ==========

// DatabaseFactory 数据库工厂接口
type DatabaseFactory interface {
	CreateDatabase(config *DBConfig) Database
}

// MySQLFactory MySQL 工厂
type MySQLFactory struct{}

func (f *MySQLFactory) CreateDatabase(config *DBConfig) Database {
	return &MySQLDatabase{config: config}
}

// PostgreSQLFactory PostgreSQL 工厂
type PostgreSQLFactory struct{}

func (f *PostgreSQLFactory) CreateDatabase(config *DBConfig) Database {
	return &PostgreSQLDatabase{config: config}
}

// MongoDBFactory MongoDB 工厂
type MongoDBFactory struct{}

func (f *MongoDBFactory) CreateDatabase(config *DBConfig) Database {
	return &MongoDBDatabase{config: config}
}

// ========== 注册机制 ==========

var dbRegistry = make(map[string]func(*DBConfig) Database)

// RegisterDatabase 注册数据库类型
func RegisterDatabase(name string, factory func(*DBConfig) Database) {
	dbRegistry[name] = factory
}

// CreateDatabaseFromRegistry 从注册表创建数据库
func CreateDatabaseFromRegistry(name string, config *DBConfig) (Database, error) {
	factory, ok := dbRegistry[name]
	if !ok {
		return nil, fmt.Errorf("database type not registered: %s", name)
	}
	return factory(config), nil
}

// ========== 连接池 ==========

// ConnectionPool 连接池
type ConnectionPool struct {
	factory  DatabaseFactory
	config   *DBConfig
	pool     chan Database
	maxConns int
}

// NewConnectionPool 创建连接池
func NewConnectionPool(factory DatabaseFactory, config *DBConfig, maxConns int) *ConnectionPool {
	return &ConnectionPool{
		factory:  factory,
		config:   config,
		pool:     make(chan Database, maxConns),
		maxConns: maxConns,
	}
}

// Get 从连接池获取连接
func (p *ConnectionPool) Get() (Database, error) {
	select {
	case db := <-p.pool:
		return db, nil
	default:
		db := p.factory.CreateDatabase(p.config)
		if err := db.Connect(); err != nil {
			return nil, err
		}
		return db, nil
	}
}

// Put 归还连接到连接池
func (p *ConnectionPool) Put(db Database) {
	select {
	case p.pool <- db:
	default:
		db.Close()
	}
}

func main() {
	fmt.Println("=== 数据库连接工厂 - 参考答案 ===\n")

	// 示例 1: 简单工厂
	fmt.Println("示例 1: 简单工厂")
	fmt.Println("-------------------")

	config := &DBConfig{
		Host:     "localhost",
		Port:     3306,
		Username: "root",
		Password: "password",
		Database: "testdb",
		MaxConns: 10,
		Timeout:  30 * time.Second,
	}

	db1, err := CreateDatabase("mysql", config)
	if err != nil {
		fmt.Println("错误:", err)
	} else {
		db1.Connect()
		results, _ := db1.Query("SELECT * FROM users")
		fmt.Printf("查询结果: %v\n", results)
		db1.Close()
	}
	fmt.Println()

	// 示例 2: 工厂方法
	fmt.Println("示例 2: 工厂方法")
	fmt.Println("-------------------")

	var factory DatabaseFactory

	dbType := "postgres"
	switch dbType {
	case "mysql":
		factory = &MySQLFactory{}
	case "postgres":
		factory = &PostgreSQLFactory{}
	case "mongodb":
		factory = &MongoDBFactory{}
	}

	config2 := &DBConfig{
		Host:     "localhost",
		Port:     5432,
		Username: "postgres",
		Password: "password",
		Database: "testdb",
	}

	db2 := factory.CreateDatabase(config2)
	db2.Connect()
	db2.Ping()
	db2.Execute("INSERT INTO users (name) VALUES ('Test')")
	db2.Close()
	fmt.Println()

	// 示例 3: 注册机制
	fmt.Println("示例 3: 注册机制")
	fmt.Println("-------------------")

	// 注册数据库类型
	RegisterDatabase("mysql", func(config *DBConfig) Database {
		return &MySQLDatabase{config: config}
	})

	RegisterDatabase("postgres", func(config *DBConfig) Database {
		return &PostgreSQLDatabase{config: config}
	})

	RegisterDatabase("mongodb", func(config *DBConfig) Database {
		return &MongoDBDatabase{config: config}
	})

	// 从注册表创建
	db3, err := CreateDatabaseFromRegistry("mongodb", &DBConfig{
		Host:     "localhost",
		Port:     27017,
		Database: "testdb",
	})

	if err != nil {
		fmt.Println("错误:", err)
	} else {
		db3.Connect()
		db3.Query("db.users.find()")
		db3.Close()
	}
	fmt.Println()

	// 示例 4: 连接池
	fmt.Println("示例 4: 连接池")
	fmt.Println("-------------------")

	pool := NewConnectionPool(&MySQLFactory{}, config, 5)

	// 获取连接
	db4, err := pool.Get()
	if err != nil {
		fmt.Println("错误:", err)
	} else {
		fmt.Printf("从连接池获取连接: %s\n", db4.GetType())
		db4.Query("SELECT * FROM products")

		// 归还连接
		pool.Put(db4)
		fmt.Println("连接已归还到连接池")
	}
	fmt.Println()

	// 示例 5: 多种数据库类型
	fmt.Println("示例 5: 多种数据库类型")
	fmt.Println("-------------------")

	databases := []struct {
		dbType string
		config *DBConfig
	}{
		{"mysql", &DBConfig{Host: "localhost", Port: 3306, Database: "mysql_db"}},
		{"postgres", &DBConfig{Host: "localhost", Port: 5432, Database: "postgres_db"}},
		{"mongodb", &DBConfig{Host: "localhost", Port: 27017, Database: "mongo_db"}},
	}

	for _, item := range databases {
		db, err := CreateDatabase(item.dbType, item.config)
		if err != nil {
			fmt.Printf("创建 %s 失败: %v\n", item.dbType, err)
			continue
		}

		fmt.Printf("\n使用 %s:\n", db.GetType())
		db.Connect()
		db.Ping()
		db.Query("SELECT * FROM test")
		db.Close()
	}

	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n实现要点:")
	fmt.Println("✅ 定义统一的 Database 接口")
	fmt.Println("✅ 实现简单工厂模式")
	fmt.Println("✅ 实现工厂方法模式")
	fmt.Println("✅ 支持注册机制，易于扩展")
	fmt.Println("✅ 实现连接池管理")
	fmt.Println("✅ 支持多种数据库类型")
}
