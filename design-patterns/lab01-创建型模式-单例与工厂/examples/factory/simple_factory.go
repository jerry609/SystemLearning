package main

import (
	"fmt"
	"os"
	"time"
)

// LogLevel 日志级别
type LogLevel int

const (
	DEBUG LogLevel = iota
	INFO
	WARN
	ERROR
)

func (l LogLevel) String() string {
	return [...]string{"DEBUG", "INFO", "WARN", "ERROR"}[l]
}

// Logger 日志接口
type Logger interface {
	Log(level LogLevel, message string) error
	Debug(message string) error
	Info(message string) error
	Warn(message string) error
	Error(message string) error
	Close() error
}

// ConsoleLogger 控制台日志 - 完整实现
type ConsoleLogger struct {
	minLevel LogLevel
	useColor bool
}

func (l *ConsoleLogger) formatMessage(level LogLevel, message string) string {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	
	if l.useColor {
		// ANSI 颜色代码
		colors := map[LogLevel]string{
			DEBUG: "\033[36m", // 青色
			INFO:  "\033[32m", // 绿色
			WARN:  "\033[33m", // 黄色
			ERROR: "\033[31m", // 红色
		}
		reset := "\033[0m"
		return fmt.Sprintf("%s[%s] %s%s: %s", colors[level], timestamp, level, reset, message)
	}
	
	return fmt.Sprintf("[%s] %s: %s", timestamp, level, message)
}

func (l *ConsoleLogger) Log(level LogLevel, message string) error {
	if level < l.minLevel {
		return nil
	}
	fmt.Println(l.formatMessage(level, message))
	return nil
}

func (l *ConsoleLogger) Debug(message string) error {
	return l.Log(DEBUG, message)
}

func (l *ConsoleLogger) Info(message string) error {
	return l.Log(INFO, message)
}

func (l *ConsoleLogger) Warn(message string) error {
	return l.Log(WARN, message)
}

func (l *ConsoleLogger) Error(message string) error {
	return l.Log(ERROR, message)
}

func (l *ConsoleLogger) Close() error {
	return nil
}

// FileLogger 文件日志 - 完整实现
type FileLogger struct {
	filename string
	file     *os.File
	minLevel LogLevel
}

func (l *FileLogger) formatMessage(level LogLevel, message string) string {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	return fmt.Sprintf("[%s] %s: %s\n", timestamp, level, message)
}

func (l *FileLogger) Log(level LogLevel, message string) error {
	if level < l.minLevel {
		return nil
	}
	
	if l.file == nil {
		var err error
		l.file, err = os.OpenFile(l.filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
		if err != nil {
			return fmt.Errorf("failed to open log file: %w", err)
		}
	}
	
	_, err := l.file.WriteString(l.formatMessage(level, message))
	return err
}

func (l *FileLogger) Debug(message string) error {
	return l.Log(DEBUG, message)
}

func (l *FileLogger) Info(message string) error {
	return l.Log(INFO, message)
}

func (l *FileLogger) Warn(message string) error {
	return l.Log(WARN, message)
}

func (l *FileLogger) Error(message string) error {
	return l.Log(ERROR, message)
}

func (l *FileLogger) Close() error {
	if l.file != nil {
		return l.file.Close()
	}
	return nil
}

// DatabaseLogger 数据库日志 - 完整实现（模拟）
type DatabaseLogger struct {
	table    string
	minLevel LogLevel
	buffer   []string // 模拟数据库缓冲区
}

func (l *DatabaseLogger) formatMessage(level LogLevel, message string) string {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	return fmt.Sprintf("INSERT INTO %s (timestamp, level, message) VALUES ('%s', '%s', '%s')",
		l.table, timestamp, level, message)
}

func (l *DatabaseLogger) Log(level LogLevel, message string) error {
	if level < l.minLevel {
		return nil
	}
	
	sql := l.formatMessage(level, message)
	l.buffer = append(l.buffer, sql)
	fmt.Printf("[DATABASE:%s] %s\n", l.table, sql)
	return nil
}

func (l *DatabaseLogger) Debug(message string) error {
	return l.Log(DEBUG, message)
}

func (l *DatabaseLogger) Info(message string) error {
	return l.Log(INFO, message)
}

func (l *DatabaseLogger) Warn(message string) error {
	return l.Log(WARN, message)
}

func (l *DatabaseLogger) Error(message string) error {
	return l.Log(ERROR, message)
}

func (l *DatabaseLogger) Close() error {
	fmt.Printf("[DATABASE:%s] Flushing %d log entries...\n", l.table, len(l.buffer))
	l.buffer = nil
	return nil
}

// LoggerConfig 日志配置
type LoggerConfig struct {
	MinLevel LogLevel
	Filename string
	Table    string
	UseColor bool
}

// LoggerFactory 简单工厂 - 完整实现
type LoggerFactory struct{}

// CreateLogger 根据类型创建日志器（带配置）
func (f *LoggerFactory) CreateLogger(loggerType string, config *LoggerConfig) (Logger, error) {
	if config == nil {
		config = &LoggerConfig{
			MinLevel: INFO,
			Filename: "app.log",
			Table:    "logs",
			UseColor: true,
		}
	}
	
	switch loggerType {
	case "console":
		return &ConsoleLogger{
			minLevel: config.MinLevel,
			useColor: config.UseColor,
		}, nil
	case "file":
		if config.Filename == "" {
			return nil, fmt.Errorf("filename is required for file logger")
		}
		return &FileLogger{
			filename: config.Filename,
			minLevel: config.MinLevel,
		}, nil
	case "database":
		if config.Table == "" {
			return nil, fmt.Errorf("table name is required for database logger")
		}
		return &DatabaseLogger{
			table:    config.Table,
			minLevel: config.MinLevel,
			buffer:   make([]string, 0),
		}, nil
	default:
		return nil, fmt.Errorf("unsupported logger type: %s", loggerType)
	}
}

// CreateLoggerSimple 简化版本（使用默认配置）
func (f *LoggerFactory) CreateLoggerSimple(loggerType string) Logger {
	logger, _ := f.CreateLogger(loggerType, nil)
	return logger
}

func main() {
	fmt.Println("=== 简单工厂模式示例（完整版）===\n")

	factory := &LoggerFactory{}

	// 示例 1: 控制台日志（带颜色）
	fmt.Println("示例 1: 控制台日志（带颜色）")
	fmt.Println("-------------------")
	consoleLogger, err := factory.CreateLogger("console", &LoggerConfig{
		MinLevel: DEBUG,
		UseColor: true,
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	consoleLogger.Debug("这是调试信息")
	consoleLogger.Info("应用程序启动")
	consoleLogger.Warn("这是一个警告")
	consoleLogger.Error("发生了一个错误")
	consoleLogger.Close()
	fmt.Println()

	// 示例 2: 文件日志
	fmt.Println("示例 2: 文件日志")
	fmt.Println("-------------------")
	fileLogger, err := factory.CreateLogger("file", &LoggerConfig{
		MinLevel: INFO,
		Filename: "application.log",
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	fileLogger.Info("日志写入文件")
	fileLogger.Warn("文件日志警告")
	fileLogger.Error("文件日志错误")
	fmt.Println("✅ 日志已写入 application.log")
	fileLogger.Close()
	fmt.Println()

	// 示例 3: 数据库日志
	fmt.Println("示例 3: 数据库日志（模拟）")
	fmt.Println("-------------------")
	dbLogger, err := factory.CreateLogger("database", &LoggerConfig{
		MinLevel: WARN,
		Table:    "application_logs",
	})
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	dbLogger.Info("这条不会记录（级别太低）")
	dbLogger.Warn("数据库警告日志")
	dbLogger.Error("数据库错误日志")
	dbLogger.Close()
	fmt.Println()

	// 示例 4: 使用简化版本
	fmt.Println("示例 4: 使用简化版本（默认配置）")
	fmt.Println("-------------------")
	simpleLogger := factory.CreateLoggerSimple("console")
	simpleLogger.Info("使用默认配置的日志")
	simpleLogger.Close()
	fmt.Println()

	// 示例 5: 错误处理
	fmt.Println("示例 5: 错误处理")
	fmt.Println("-------------------")
	_, err = factory.CreateLogger("unknown", nil)
	if err != nil {
		fmt.Printf("✅ 正确处理了错误: %v\n", err)
	}
	
	_, err = factory.CreateLogger("file", &LoggerConfig{Filename: ""})
	if err != nil {
		fmt.Printf("✅ 正确处理了错误: %v\n", err)
	}
	fmt.Println()

	// 示例 6: 不同日志级别过滤
	fmt.Println("示例 6: 日志级别过滤")
	fmt.Println("-------------------")
	errorOnlyLogger, _ := factory.CreateLogger("console", &LoggerConfig{
		MinLevel: ERROR,
		UseColor: false,
	})
	
	fmt.Println("只记录 ERROR 级别:")
	errorOnlyLogger.Debug("不会显示")
	errorOnlyLogger.Info("不会显示")
	errorOnlyLogger.Warn("不会显示")
	errorOnlyLogger.Error("只有这条会显示")
	errorOnlyLogger.Close()

	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n简单工厂模式特点:")
	fmt.Println("✅ 优点：")
	fmt.Println("  - 客户端不需要知道具体类名")
	fmt.Println("  - 集中管理对象创建")
	fmt.Println("  - 简单易用，适合产品种类少的场景")
	fmt.Println("\n❌ 缺点：")
	fmt.Println("  - 违反开闭原则（添加新类型需要修改工厂）")
	fmt.Println("  - 工厂类职责过重")
	fmt.Println("  - 不易扩展")
	
	fmt.Println("\n💡 改进建议:")
	fmt.Println("  - 使用工厂方法模式提高扩展性")
	fmt.Println("  - 使用注册机制支持动态添加新类型")
	fmt.Println("  - 添加配置验证和错误处理")
}
