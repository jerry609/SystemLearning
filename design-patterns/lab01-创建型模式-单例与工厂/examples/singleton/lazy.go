package main

import (
	"fmt"
)

// 懒汉式单例模式示例
// 特点：第一次使用时才创建实例，但非线程安全（仅用于演示）

// Logger 日志记录器
type Logger struct {
	level  string
	output string
}

// 实例变量
var instance *Logger

// GetInstance 获取单例实例（非线程安全版本）
func GetInstance() *Logger {
	if instance == nil {
		fmt.Println("创建新的 Logger 实例...")
		instance = &Logger{
			level:  "INFO",
			output: "console",
		}
	}
	return instance
}

// Log 记录日志
func (l *Logger) Log(message string) {
	fmt.Printf("[%s] %s\n", l.level, message)
}

// SetLevel 设置日志级别
func (l *Logger) SetLevel(level string) {
	l.level = level
}

// GetLevel 获取日志级别
func (l *Logger) GetLevel() string {
	return l.level
}

// SetOutput 设置输出方式
func (l *Logger) SetOutput(output string) {
	l.output = output
}

// GetOutput 获取输出方式
func (l *Logger) GetOutput() string {
	return l.output
}

func main() {
	fmt.Println("=== 懒汉式单例模式示例 ===\n")

	fmt.Println("程序启动，但还没有创建 Logger 实例")
	fmt.Println()

	// 第一次获取实例 - 会创建新实例
	fmt.Println("第一次调用 GetInstance():")
	logger1 := GetInstance()
	fmt.Printf("实例 1: %p\n", logger1)
	fmt.Printf("日志级别: %s\n", logger1.GetLevel())
	fmt.Printf("输出方式: %s\n", logger1.GetOutput())
	fmt.Println()

	// 使用日志
	logger1.Log("应用程序启动")
	logger1.SetLevel("DEBUG")
	logger1.Log("调试信息")
	fmt.Println()

	// 第二次获取实例 - 不会创建新实例
	fmt.Println("第二次调用 GetInstance():")
	logger2 := GetInstance()
	fmt.Printf("实例 2: %p\n", logger2)
	fmt.Printf("日志级别: %s\n", logger2.GetLevel())
	fmt.Println()

	// 验证是同一个实例
	if logger1 == logger2 {
		fmt.Println("✅ logger1 和 logger2 是同一个实例")
	} else {
		fmt.Println("❌ logger1 和 logger2 不是同一个实例")
	}

	// 修改实例 2 的配置
	logger2.SetLevel("ERROR")
	logger2.Log("错误信息")
	fmt.Println()

	// 验证实例 1 的配置也被修改了
	fmt.Printf("实例 1 的日志级别: %s\n", logger1.GetLevel())

	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n懒汉式单例特点:")
	fmt.Println("✅ 延迟初始化 - 第一次使用时才创建")
	fmt.Println("✅ 节省资源 - 不使用就不创建")
	fmt.Println("❌ 非线程安全 - 并发时可能创建多个实例")
	fmt.Println("❌ 需要加锁 - 线程安全版本需要使用互斥锁")
	fmt.Println("\n⚠️  注意: 这个示例是非线程安全的，仅用于演示")
	fmt.Println("    在实际项目中，应该使用 sync.Once 或双重检查锁")
}
