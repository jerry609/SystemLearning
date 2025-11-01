package main

import (
	"fmt"
)

// 饿汉式单例模式示例
// 特点：在包初始化时就创建实例，线程安全，但可能浪费资源

// Config 配置管理器
type Config struct {
	appName string
	version string
	debug   bool
	data    map[string]string
}

// 在包初始化时创建实例
var instance = &Config{
	appName: "MyApp",
	version: "1.0.0",
	debug:   false,
	data:    make(map[string]string),
}

// GetInstance 获取单例实例
func GetInstance() *Config {
	return instance
}

// Get 获取配置项
func (c *Config) Get(key string) string {
	return c.data[key]
}

// Set 设置配置项
func (c *Config) Set(key, value string) {
	c.data[key] = value
}

// GetAppName 获取应用名称
func (c *Config) GetAppName() string {
	return c.appName
}

// GetVersion 获取版本号
func (c *Config) GetVersion() string {
	return c.version
}

// IsDebug 是否调试模式
func (c *Config) IsDebug() bool {
	return c.debug
}

// SetDebug 设置调试模式
func (c *Config) SetDebug(debug bool) {
	c.debug = debug
}

func main() {
	fmt.Println("=== 饿汉式单例模式示例 ===\n")

	// 获取单例实例
	config1 := GetInstance()
	fmt.Printf("实例 1: %p\n", config1)
	fmt.Printf("应用名称: %s\n", config1.GetAppName())
	fmt.Printf("版本号: %s\n", config1.GetVersion())
	fmt.Println()

	// 设置配置
	config1.Set("database", "mysql")
	config1.Set("host", "localhost")
	config1.SetDebug(true)

	// 再次获取实例
	config2 := GetInstance()
	fmt.Printf("实例 2: %p\n", config2)
	fmt.Printf("数据库: %s\n", config2.Get("database"))
	fmt.Printf("主机: %s\n", config2.Get("host"))
	fmt.Printf("调试模式: %v\n", config2.IsDebug())
	fmt.Println()

	// 验证是同一个实例
	if config1 == config2 {
		fmt.Println("✅ config1 和 config2 是同一个实例")
	} else {
		fmt.Println("❌ config1 和 config2 不是同一个实例")
	}

	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n饿汉式单例特点:")
	fmt.Println("✅ 线程安全 - 在包初始化时创建")
	fmt.Println("✅ 实现简单 - 代码量少")
	fmt.Println("✅ 无性能问题 - 不需要加锁")
	fmt.Println("❌ 无法延迟初始化 - 即使不使用也会创建")
	fmt.Println("❌ 可能浪费资源 - 如果实例很大且不常用")
}
