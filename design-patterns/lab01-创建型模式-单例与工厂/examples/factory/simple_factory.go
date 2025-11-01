package main

import (
	"fmt"
)

// Logger 日志接口
type Logger interface {
	Log(message string)
}

// ConsoleLogger 控制台日志
type ConsoleLogger struct{}

func (l *ConsoleLogger) Log(message string) {
	fmt.Printf("[CONSOLE] %s\n", message)
}

// FileLogger 文件日志
type FileLogger struct {
	filename string
}

func (l *FileLogger) Log(message string) {
	fmt.Printf("[FILE:%s] %s\n", l.filename, message)
}

// DatabaseLogger 数据库日志
type DatabaseLogger struct {
	table string
}

func (l *DatabaseLogger) Log(message string) {
	fmt.Printf("[DATABASE:%s] %s\n", l.table, message)
}

// LoggerFactory 简单工厂
type LoggerFactory struct{}

// CreateLogger 根据类型创建日志器
func (f *LoggerFactory) CreateLogger(loggerType string) Logger {
	switch loggerType {
	case "console":
		return &ConsoleLogger{}
	case "file":
		return &FileLogger{filename: "app.log"}
	case "database":
		return &DatabaseLogger{table: "logs"}
	default:
		return &ConsoleLogger{} // 默认返回控制台日志
	}
}

func main() {
	fmt.Println("=== 简单工厂模式示例 ===\n")

	factory := &LoggerFactory{}

	// 创建不同类型的日志器
	consoleLogger := factory.CreateLogger("console")
	consoleLogger.Log("这是控制台日志")

	fileLogger := factory.CreateLogger("file")
	fileLogger.Log("这是文件日志")

	dbLogger := factory.CreateLogger("database")
	dbLogger.Log("这是数据库日志")

	// 使用默认日志器
	defaultLogger := factory.CreateLogger("unknown")
	defaultLogger.Log("未知类型，使用默认日志器")

	fmt.Println("\n✅ 简单工厂模式演示完成")
	fmt.Println("\n优点：")
	fmt.Println("  - 客户端不需要知道具体类名")
	fmt.Println("  - 集中管理对象创建")
	fmt.Println("\n缺点：")
	fmt.Println("  - 违反开闭原则（添加新类型需要修改工厂）")
	fmt.Println("  - 工厂类职责过重")
}
