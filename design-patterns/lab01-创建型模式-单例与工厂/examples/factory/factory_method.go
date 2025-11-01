package main

import (
	"fmt"
)

// Database 数据库接口
type Database interface {
	Connect() string
	Query(sql string) string
}

// MySQL 数据库
type MySQL struct {
	host string
	port int
}

func (m *MySQL) Connect() string {
	return fmt.Sprintf("连接到 MySQL: %s:%d", m.host, m.port)
}

func (m *MySQL) Query(sql string) string {
	return fmt.Sprintf("MySQL 执行: %s", sql)
}

// PostgreSQL 数据库
type PostgreSQL struct {
	host string
	port int
}

func (p *PostgreSQL) Connect() string {
	return fmt.Sprintf("连接到 PostgreSQL: %s:%d", p.host, p.port)
}

func (p *PostgreSQL) Query(sql string) string {
	return fmt.Sprintf("PostgreSQL 执行: %s", sql)
}

// MongoDB 数据库
type MongoDB struct {
	host string
	port int
}

func (m *MongoDB) Connect() string {
	return fmt.Sprintf("连接到 MongoDB: %s:%d", m.host, m.port)
}

func (m *MongoDB) Query(sql string) string {
	return fmt.Sprintf("MongoDB 执行: %s", sql)
}

// DatabaseFactory 数据库工厂接口
type DatabaseFactory interface {
	CreateDatabase() Database
}

// MySQLFactory MySQL 工厂
type MySQLFactory struct {
	host string
	port int
}

func (f *MySQLFactory) CreateDatabase() Database {
	return &MySQL{
		host: f.host,
		port: f.port,
	}
}

// PostgreSQLFactory PostgreSQL 工厂
type PostgreSQLFactory struct {
	host string
	port int
}

func (f *PostgreSQLFactory) CreateDatabase() Database {
	return &PostgreSQL{
		host: f.host,
		port: f.port,
	}
}

// MongoDBFactory MongoDB 工厂
type MongoDBFactory struct {
	host string
	port int
}

func (f *MongoDBFactory) CreateDatabase() Database {
	return &MongoDB{
		host: f.host,
		port: f.port,
	}
}

// 客户端代码
func executeQuery(factory DatabaseFactory, sql string) {
	db := factory.CreateDatabase()
	fmt.Println(db.Connect())
	fmt.Println(db.Query(sql))
	fmt.Println()
}

func main() {
	fmt.Println("=== 工厂方法模式示例 ===\n")

	// 使用 MySQL
	mysqlFactory := &MySQLFactory{
		host: "localhost",
		port: 3306,
	}
	executeQuery(mysqlFactory, "SELECT * FROM users")

	// 使用 PostgreSQL
	pgFactory := &PostgreSQLFactory{
		host: "localhost",
		port: 5432,
	}
	executeQuery(pgFactory, "SELECT * FROM orders")

	// 使用 MongoDB
	mongoFactory := &MongoDBFactory{
		host: "localhost",
		port: 27017,
	}
	executeQuery(mongoFactory, "db.products.find()")

	fmt.Println("✅ 工厂方法模式演示完成")
	fmt.Println("\n优点：")
	fmt.Println("  - 符合开闭原则（添加新数据库只需新增工厂类）")
	fmt.Println("  - 单一职责（每个工厂只负责创建一种产品）")
	fmt.Println("  - 客户端与具体产品解耦")
	fmt.Println("\n缺点：")
	fmt.Println("  - 类的数量增加（每个产品需要一个工厂）")
	fmt.Println("  - 增加了系统复杂度")
}
