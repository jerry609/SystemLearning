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

// Logger 日志记录器接口
type Logger interface {
	Debug(message string)
	Info(message string)
	Warn(message string)
	Error(message string)
	SetLevel(level LogLevel)
}

// LoggerConfig 日志配置
type LoggerConfig struct {
	Level    LogLevel
	Format   string
	Filename string
}

// ConsoleLogger 控制台日志记录器
type ConsoleLogger struct {
	level  LogLevel
	format string
}

func (l *ConsoleLogger) log(level LogLevel, levelStr, message string) {
	if level < l.level {
		return
	}
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	fmt.Printf("[%s] %s: %s\n", timestamp, levelStr, message)
}

func (l *ConsoleLogger) Debug(message string) {
	l.log(DEBUG, "DEBUG", message)
}

func (l *ConsoleLogger) Info(message string) {
	l.log(INFO, "INFO", message)
}

func (l *ConsoleLogger) Warn(message string) {
	l.log(WARN, "WARN", message)
}

func (l *ConsoleLogger) Error(message string) {
	l.log(ERROR, "ERROR", message)
}

func (l *ConsoleLogger) SetLevel(level LogLevel) {
	l.level = level
}

// FileLogger 文件日志记录器
type FileLogger struct {
	level    LogLevel
	filename string
	file     *os.File
}

func (l *FileLogger) log(level LogLevel, levelStr, message string) {
	if level < l.level {
		return
	}
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	logLine := fmt.Sprintf("[%s] %s: %s\n", timestamp, levelStr, message)
	if l.file != nil {
		l.file.WriteString(logLine)
	}
}

func (l *FileLogger) Debug(message string) {
	l.log(DEBUG, "DEBUG", message)
}

func (l *FileLogger) Info(message string) {
	l.log(INFO, "INFO", message)
}

func (l *FileLogger) Warn(message string) {
	l.log(WARN, "WARN", message)
}

func (l *FileLogger) Error(message string) {
	l.log(ERROR, "ERROR", message)
}

func (l *FileLogger) SetLevel(level LogLevel) {
	l.level = level
}

func (l *FileLogger) Close() error {
	if l.file != nil {
		return l.file.Close()
	}
	return nil
}
