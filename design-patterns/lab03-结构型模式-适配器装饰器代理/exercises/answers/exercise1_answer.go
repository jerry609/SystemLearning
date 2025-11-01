package main

// 练习 1: 实现日志系统适配器 - 参考答案
//
// 设计思路:
// 1. 定义统一的 Logger 接口，支持不同日志级别和结构化日志
// 2. 为标准库 log 包创建适配器，实现接口转换
// 3. 为自定义日志库创建适配器，展示适配器的灵活性
// 4. 实现日志管理器，支持运行时切换和多日志输出
//
// 使用的设计模式: 适配器模式
// 模式应用位置:
// - StdLogAdapter: 将标准库 log 适配到统一接口
// - CustomLogAdapter: 将自定义日志库适配到统一接口
// - MultiLogger: 组合多个日志适配器

import (
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

// ============================================================================
// 统一日志接口
// ============================================================================

// LogLevel 日志级别
type LogLevel int

const (
	DebugLevel LogLevel = iota
	InfoLevel
	WarnLevel
	ErrorLevel
)

func (l LogLevel) String() string {
	switch l {
	case DebugLevel:
		return "DEBUG"
	case InfoLevel:
		return "INFO"
	case WarnLevel:
		return "WARN"
	case ErrorLevel:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// Logger 统一日志接口
type Logger interface {
	Debug(msg string)
	Info(msg string)
	Warn(msg string)
	Error(msg string)

	Debugf(format string, args ...interface{})
	Infof(format string, args ...interface{})
	Warnf(format string, args ...interface{})
	Errorf(format string, args ...interface{})

	WithFields(fields map[string]interface{}) Logger
}

// ============================================================================
// 标准库适配器
// ============================================================================

// StdLogAdapter 标准库日志适配器
type StdLogAdapter struct {
	logger *log.Logger
	fields map[string]interface{}
	level  LogLevel
}

// NewStdLogAdapter 创建标准库日志适配器
func NewStdLogAdapter(logger *log.Logger) *StdLogAdapter {
	return &StdLogAdapter{
		logger: logger,
		fields: make(map[string]interface{}),
		level:  DebugLevel,
	}
}

// Debug 输出调试日志
func (a *StdLogAdapter) Debug(msg string) {
	if a.level <= DebugLevel {
		a.log(DebugLevel, msg)
	}
}

// Info 输出信息日志
func (a *StdLogAdapter) Info(msg string) {
	if a.level <= InfoLevel {
		a.log(InfoLevel, msg)
	}
}

// Warn 输出警告日志
func (a *StdLogAdapter) Warn(msg string) {
	if a.level <= WarnLevel {
		a.log(WarnLevel, msg)
	}
}

// Error 输出错误日志
func (a *StdLogAdapter) Error(msg string) {
	if a.level <= ErrorLevel {
		a.log(ErrorLevel, msg)
	}
}

// Debugf 格式化输出调试日志
func (a *StdLogAdapter) Debugf(format string, args ...interface{}) {
	if a.level <= DebugLevel {
		a.log(DebugLevel, fmt.Sprintf(format, args...))
	}
}

// Infof 格式化输出信息日志
func (a *StdLogAdapter) Infof(format string, args ...interface{}) {
	if a.level <= InfoLevel {
		a.log(InfoLevel, fmt.Sprintf(format, args...))
	}
}

// Warnf 格式化输出警告日志
func (a *StdLogAdapter) Warnf(format string, args ...interface{}) {
	if a.level <= WarnLevel {
		a.log(WarnLevel, fmt.Sprintf(format, args...))
	}
}

// Errorf 格式化输出错误日志
func (a *StdLogAdapter) Errorf(format string, args ...interface{}) {
	if a.level <= ErrorLevel {
		a.log(ErrorLevel, fmt.Sprintf(format, args...))
	}
}

// WithFields 添加结构化字段
func (a *StdLogAdapter) WithFields(fields map[string]interface{}) Logger {
	newAdapter := &StdLogAdapter{
		logger: a.logger,
		fields: make(map[string]interface{}),
		level:  a.level,
	}

	// 复制现有字段
	for k, v := range a.fields {
		newAdapter.fields[k] = v
	}

	// 添加新字段
	for k, v := range fields {
		newAdapter.fields[k] = v
	}

	return newAdapter
}

// log 内部日志方法
func (a *StdLogAdapter) log(level LogLevel, msg string) {
	fieldsStr := a.formatFields()
	a.logger.Printf("[%s] %s%s", level, msg, fieldsStr)
}

// formatFields 格式化字段
func (a *StdLogAdapter) formatFields() string {
	if len(a.fields) == 0 {
		return ""
	}

	var parts []string
	for k, v := range a.fields {
		parts = append(parts, fmt.Sprintf("%s=%v", k, v))
	}
	return " " + strings.Join(parts, " ")
}

// SetLevel 设置日志级别
func (a *StdLogAdapter) SetLevel(level LogLevel) {
	a.level = level
}

// ============================================================================
// 自定义日志适配器
// ============================================================================

// CustomLogger 自定义日志库（模拟第三方库）
type CustomLogger struct {
	prefix string
}

func NewCustomLogger(prefix string) *CustomLogger {
	return &CustomLogger{prefix: prefix}
}

func (c *CustomLogger) Log(level, message string, fields map[string]interface{}) {
	timestamp := time.Now().Format("2006-01-02T15:04:05-07:00")
	fieldsStr := ""
	if len(fields) > 0 {
		var parts []string
		for k, v := range fields {
			parts = append(parts, fmt.Sprintf("%s=%v", k, v))
		}
		fieldsStr = " " + strings.Join(parts, " ")
	}
	fmt.Printf("[%s] %s %s %s%s\n", c.prefix, timestamp, level, message, fieldsStr)
}

// CustomLogAdapter 自定义日志适配器
type CustomLogAdapter struct {
	logger *CustomLogger
	fields map[string]interface{}
}

// NewCustomLogAdapter 创建自定义日志适配器
func NewCustomLogAdapter(logger *CustomLogger) *CustomLogAdapter {
	return &CustomLogAdapter{
		logger: logger,
		fields: make(map[string]interface{}),
	}
}

func (a *CustomLogAdapter) Debug(msg string) {
	a.logger.Log("DEBUG", msg, a.fields)
}

func (a *CustomLogAdapter) Info(msg string) {
	a.logger.Log("INFO", msg, a.fields)
}

func (a *CustomLogAdapter) Warn(msg string) {
	a.logger.Log("WARN", msg, a.fields)
}

func (a *CustomLogAdapter) Error(msg string) {
	a.logger.Log("ERROR", msg, a.fields)
}

func (a *CustomLogAdapter) Debugf(format string, args ...interface{}) {
	a.logger.Log("DEBUG", fmt.Sprintf(format, args...), a.fields)
}

func (a *CustomLogAdapter) Infof(format string, args ...interface{}) {
	a.logger.Log("INFO", fmt.Sprintf(format, args...), a.fields)
}

func (a *CustomLogAdapter) Warnf(format string, args ...interface{}) {
	a.logger.Log("WARN", fmt.Sprintf(format, args...), a.fields)
}

func (a *CustomLogAdapter) Errorf(format string, args ...interface{}) {
	a.logger.Log("ERROR", fmt.Sprintf(format, args...), a.fields)
}

func (a *CustomLogAdapter) WithFields(fields map[string]interface{}) Logger {
	newAdapter := &CustomLogAdapter{
		logger: a.logger,
		fields: make(map[string]interface{}),
	}

	for k, v := range a.fields {
		newAdapter.fields[k] = v
	}

	for k, v := range fields {
		newAdapter.fields[k] = v
	}

	return newAdapter
}

// ============================================================================
// 日志管理器
// ============================================================================

// LoggerManager 日志管理器
type LoggerManager struct {
	loggers map[string]Logger
	current Logger
}

// NewLoggerManager 创建日志管理器
func NewLoggerManager() *LoggerManager {
	return &LoggerManager{
		loggers: make(map[string]Logger),
	}
}

// Register 注册日志实现
func (m *LoggerManager) Register(name string, logger Logger) {
	m.loggers[name] = logger
}

// Use 使用指定的日志实现
func (m *LoggerManager) Use(name string) error {
	logger, ok := m.loggers[name]
	if !ok {
		return fmt.Errorf("logger '%s' not found", name)
	}
	m.current = logger
	return nil
}

// 实现 Logger 接口，委托给当前日志实现
func (m *LoggerManager) Debug(msg string) {
	if m.current != nil {
		m.current.Debug(msg)
	}
}

func (m *LoggerManager) Info(msg string) {
	if m.current != nil {
		m.current.Info(msg)
	}
}

func (m *LoggerManager) Warn(msg string) {
	if m.current != nil {
		m.current.Warn(msg)
	}
}

func (m *LoggerManager) Error(msg string) {
	if m.current != nil {
		m.current.Error(msg)
	}
}

func (m *LoggerManager) Debugf(format string, args ...interface{}) {
	if m.current != nil {
		m.current.Debugf(format, args...)
	}
}

func (m *LoggerManager) Infof(format string, args ...interface{}) {
	if m.current != nil {
		m.current.Infof(format, args...)
	}
}

func (m *LoggerManager) Warnf(format string, args ...interface{}) {
	if m.current != nil {
		m.current.Warnf(format, args...)
	}
}

func (m *LoggerManager) Errorf(format string, args ...interface{}) {
	if m.current != nil {
		m.current.Errorf(format, args...)
	}
}

func (m *LoggerManager) WithFields(fields map[string]interface{}) Logger {
	if m.current != nil {
		return m.current.WithFields(fields)
	}
	return m
}

// ============================================================================
// 多日志输出
// ============================================================================

// MultiLogger 多日志输出
type MultiLogger struct {
	loggers []Logger
}

// NewMultiLogger 创建多日志输出
func NewMultiLogger(loggers ...Logger) *MultiLogger {
	return &MultiLogger{loggers: loggers}
}

func (m *MultiLogger) Debug(msg string) {
	for _, logger := range m.loggers {
		logger.Debug(msg)
	}
}

func (m *MultiLogger) Info(msg string) {
	for _, logger := range m.loggers {
		logger.Info(msg)
	}
}

func (m *MultiLogger) Warn(msg string) {
	for _, logger := range m.loggers {
		logger.Warn(msg)
	}
}

func (m *MultiLogger) Error(msg string) {
	for _, logger := range m.loggers {
		logger.Error(msg)
	}
}

func (m *MultiLogger) Debugf(format string, args ...interface{}) {
	for _, logger := range m.loggers {
		logger.Debugf(format, args...)
	}
}

func (m *MultiLogger) Infof(format string, args ...interface{}) {
	for _, logger := range m.loggers {
		logger.Infof(format, args...)
	}
}

func (m *MultiLogger) Warnf(format string, args ...interface{}) {
	for _, logger := range m.loggers {
		logger.Warnf(format, args...)
	}
}

func (m *MultiLogger) Errorf(format string, args ...interface{}) {
	for _, logger := range m.loggers {
		logger.Errorf(format, args...)
	}
}

func (m *MultiLogger) WithFields(fields map[string]interface{}) Logger {
	newLoggers := make([]Logger, len(m.loggers))
	for i, logger := range m.loggers {
		newLoggers[i] = logger.WithFields(fields)
	}
	return &MultiLogger{loggers: newLoggers}
}

// ============================================================================
// 示例代码
// ============================================================================

func main() {
	fmt.Println("=== 日志系统适配器示例 ===\n")

	// 示例 1: 使用标准库适配器
	fmt.Println("--- 示例 1: 标准库适配器 ---")
	stdLogger := log.New(os.Stdout, "[APP] ", log.LstdFlags)
	logger := NewStdLogAdapter(stdLogger)

	logger.Info("Application started")
	logger.Warn("Low memory warning")
	logger.Error("Failed to connect to database")
	logger.WithFields(map[string]interface{}{
		"user_id": 123,
		"action":  "login",
	}).Info("User logged in")

	fmt.Println()

	// 示例 2: 使用自定义适配器
	fmt.Println("--- 示例 2: 自定义适配器 ---")
	customLogger := NewCustomLogger("CUSTOM")
	customAdapter := NewCustomLogAdapter(customLogger)

	customAdapter.Info("Using custom logger")
	customAdapter.WithFields(map[string]interface{}{
		"request_id": "abc123",
		"duration":   "150ms",
	}).Info("Request completed")

	fmt.Println()

	// 示例 3: 日志管理器
	fmt.Println("--- 示例 3: 日志管理器 ---")
	manager := NewLoggerManager()
	manager.Register("std", logger)
	manager.Register("custom", customAdapter)

	manager.Use("std")
	manager.Info("Using standard logger")

	manager.Use("custom")
	manager.Info("Using custom logger")

	fmt.Println()

	// 示例 4: 多日志输出
	fmt.Println("--- 示例 4: 多日志输出 ---")
	multiLogger := NewMultiLogger(logger, customAdapter)
	multiLogger.Info("This message goes to both loggers")
	multiLogger.WithFields(map[string]interface{}{
		"event": "system_startup",
	}).Info("System initialized")

	fmt.Println("\n=== 示例结束 ===")
}

// 可能的优化方向:
// 1. 添加日志级别过滤功能
// 2. 实现异步日志输出
// 3. 添加日志轮转功能
// 4. 支持日志采样
// 5. 添加日志钩子机制
//
// 变体实现:
// 1. 使用接口组合而非嵌入
// 2. 使用函数式选项模式配置适配器
// 3. 实现日志缓冲区提高性能
