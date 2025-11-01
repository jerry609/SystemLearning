package main

import (
	"crypto/tls"
	"fmt"
	"time"
)

// Functional Options 模式示例
// 这是 Go 语言中最推荐的建造者模式实现方式

// Server 表示一个服务器配置
type Server struct {
	host        string
	port        int
	timeout     time.Duration
	maxConn     int
	tls         *tls.Config
	enableLog   bool
	logLevel    string
	compression bool
}

// Option 是一个函数类型，用于配置 Server
type Option func(*Server)

// WithHost 设置服务器主机地址
func WithHost(host string) Option {
	return func(s *Server) {
		s.host = host
	}
}

// WithPort 设置服务器端口
func WithPort(port int) Option {
	return func(s *Server) {
		s.port = port
	}
}

// WithTimeout 设置超时时间
func WithTimeout(timeout time.Duration) Option {
	return func(s *Server) {
		s.timeout = timeout
	}
}

// WithMaxConnections 设置最大连接数
func WithMaxConnections(maxConn int) Option {
	return func(s *Server) {
		s.maxConn = maxConn
	}
}

// WithTLS 启用 TLS 配置
func WithTLS(config *tls.Config) Option {
	return func(s *Server) {
		s.tls = config
	}
}

// WithLogging 启用日志记录
func WithLogging(enabled bool, level string) Option {
	return func(s *Server) {
		s.enableLog = enabled
		s.logLevel = level
	}
}

// WithCompression 启用压缩
func WithCompression(enabled bool) Option {
	return func(s *Server) {
		s.compression = enabled
	}
}

// NewServer 创建一个新的服务器实例
// 接受可变数量的 Option 参数
func NewServer(opts ...Option) *Server {
	// 设置默认值
	server := &Server{
		host:        "localhost",
		port:        8080,
		timeout:     30 * time.Second,
		maxConn:     100,
		enableLog:   true,
		logLevel:    "info",
		compression: false,
	}

	// 应用所有选项
	for _, opt := range opts {
		opt(server)
	}

	return server
}

// String 返回服务器配置的字符串表示
func (s *Server) String() string {
	return fmt.Sprintf(
		"Server{host: %s, port: %d, timeout: %v, maxConn: %d, tls: %v, log: %v(%s), compression: %v}",
		s.host, s.port, s.timeout, s.maxConn,
		s.tls != nil, s.enableLog, s.logLevel, s.compression,
	)
}

// Database 配置示例
type Database struct {
	host     string
	port     int
	username string
	password string
	database string
	maxIdle  int
	maxOpen  int
	timeout  time.Duration
}

type DBOption func(*Database)

func DBWithHost(host string) DBOption {
	return func(db *Database) {
		db.host = host
	}
}

func DBWithPort(port int) DBOption {
	return func(db *Database) {
		db.port = port
	}
}

func DBWithCredentials(username, password string) DBOption {
	return func(db *Database) {
		db.username = username
		db.password = password
	}
}

func DBWithDatabase(database string) DBOption {
	return func(db *Database) {
		db.database = database
	}
}

func DBWithPoolSize(maxIdle, maxOpen int) DBOption {
	return func(db *Database) {
		db.maxIdle = maxIdle
		db.maxOpen = maxOpen
	}
}

func DBWithTimeout(timeout time.Duration) DBOption {
	return func(db *Database) {
		db.timeout = timeout
	}
}

func NewDatabase(opts ...DBOption) *Database {
	db := &Database{
		host:     "localhost",
		port:     3306,
		username: "root",
		password: "",
		database: "test",
		maxIdle:  10,
		maxOpen:  100,
		timeout:  30 * time.Second,
	}

	for _, opt := range opts {
		opt(db)
	}

	return db
}

func (db *Database) String() string {
	return fmt.Sprintf(
		"Database{host: %s, port: %d, user: %s, db: %s, pool: %d/%d, timeout: %v}",
		db.host, db.port, db.username, db.database,
		db.maxIdle, db.maxOpen, db.timeout,
	)
}

func main() {
	fmt.Println("=== Functional Options 模式示例 ===\n")

	// 示例 1: 使用默认配置
	fmt.Println("示例 1: 使用默认配置")
	server1 := NewServer()
	fmt.Println(server1)
	fmt.Println()

	// 示例 2: 自定义部分配置
	fmt.Println("示例 2: 自定义部分配置")
	server2 := NewServer(
		WithHost("0.0.0.0"),
		WithPort(9090),
	)
	fmt.Println(server2)
	fmt.Println()

	// 示例 3: 完整的自定义配置
	fmt.Println("示例 3: 完整的自定义配置")
	server3 := NewServer(
		WithHost("192.168.1.100"),
		WithPort(443),
		WithTimeout(60*time.Second),
		WithMaxConnections(1000),
		WithLogging(true, "debug"),
		WithCompression(true),
	)
	fmt.Println(server3)
	fmt.Println()

	// 示例 4: 数据库配置
	fmt.Println("示例 4: 数据库配置")
	db1 := NewDatabase()
	fmt.Println("默认配置:", db1)

	db2 := NewDatabase(
		DBWithHost("db.example.com"),
		DBWithPort(5432),
		DBWithCredentials("admin", "secret"),
		DBWithDatabase("production"),
		DBWithPoolSize(20, 200),
		DBWithTimeout(60*time.Second),
	)
	fmt.Println("自定义配置:", db2)
	fmt.Println()

	// 示例 5: 动态构建选项
	fmt.Println("示例 5: 动态构建选项")
	var opts []Option
	opts = append(opts, WithHost("api.example.com"))
	opts = append(opts, WithPort(8443))

	// 根据条件添加选项
	enableTLS := true
	if enableTLS {
		opts = append(opts, WithTLS(&tls.Config{}))
	}

	enableDebug := true
	if enableDebug {
		opts = append(opts, WithLogging(true, "debug"))
	}

	server4 := NewServer(opts...)
	fmt.Println(server4)

	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n优点:")
	fmt.Println("✅ 向后兼容 - 添加新选项不影响现有代码")
	fmt.Println("✅ 默认值 - 提供合理的默认配置")
	fmt.Println("✅ 可选参数 - 只需指定需要的选项")
	fmt.Println("✅ 类型安全 - 编译时检查")
	fmt.Println("✅ 自文档化 - 选项名称清晰表达意图")
}
