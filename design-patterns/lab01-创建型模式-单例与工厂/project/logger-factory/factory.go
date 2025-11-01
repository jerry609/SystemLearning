package main

import (
	"fmt"
	"os"
)

// CreateLogger 简单工厂方法
func CreateLogger(loggerType string, config *LoggerConfig) (Logger, error) {
	switch loggerType {
	case "console":
		return &ConsoleLogger{
			level:  config.Level,
			format: config.Format,
		}, nil
	case "file":
		file, err := os.OpenFile(config.Filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
		if err != nil {
			return nil, err
		}
		return &FileLogger{
			level:    config.Level,
			filename: config.Filename,
			file:     file,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported logger type: %s", loggerType)
	}
}

// LoggerFactory 日志工厂接口
type LoggerFactory interface {
	CreateLogger(config *LoggerConfig) (Logger, error)
}

// ConsoleLoggerFactory 控制台日志工厂
type ConsoleLoggerFactory struct{}

func (f *ConsoleLoggerFactory) CreateLogger(config *LoggerConfig) (Logger, error) {
	return &ConsoleLogger{
		level:  config.Level,
		format: config.Format,
	}, nil
}

// FileLoggerFactory 文件日志工厂
type FileLoggerFactory struct{}

func (f *FileLoggerFactory) CreateLogger(config *LoggerConfig) (Logger, error) {
	file, err := os.OpenFile(config.Filename, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
	if err != nil {
		return nil, err
	}
	return &FileLogger{
		level:    config.Level,
		filename: config.Filename,
		file:     file,
	}, nil
}

func main() {
	fmt.Println("=== 日志工厂实战项目 ===\n")

	// 示例 1: 控制台日志
	fmt.Println("示例 1: 控制台日志")
	fmt.Println("-------------------")
	consoleLogger, _ := CreateLogger("console", &LoggerConfig{
		Level: INFO,
	})
	consoleLogger.Info("Application started")
	consoleLogger.Warn("This is a warning")
	consoleLogger.Error("An error occurred")
	fmt.Println()

	// 示例 2: 文件日志
	fmt.Println("示例 2: 文件日志")
	fmt.Println("-------------------")
	fileLogger, _ := CreateLogger("file", &LoggerConfig{
		Level:    DEBUG,
		Filename: "app.log",
	})
	fileLogger.Info("Log to file")
	fileLogger.Debug("Debug message")
	fmt.Println("日志已写入文件: app.log")
	fmt.Println()

	// 示例 3: 工厂方法
	fmt.Println("示例 3: 工厂方法")
	fmt.Println("-------------------")
	factory := &ConsoleLoggerFactory{}
	logger, _ := factory.CreateLogger(&LoggerConfig{Level: DEBUG})
	logger.Debug("Debug message")
	logger.Info("Info message")

	fmt.Println("\n=== 示例结束 ===")
}
