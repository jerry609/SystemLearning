package main

import (
	"fmt"
	"sync"
)

// Config 配置管理器（单例）
type Config struct {
	AppName string
	Version string
	Debug   bool
	mu      sync.RWMutex
	data    map[string]interface{}
}

var (
	instance *Config
	once     sync.Once
)

// GetInstance 获取配置管理器实例
// 使用 sync.Once 确保只初始化一次
func GetInstance() *Config {
	once.Do(func() {
		fmt.Println("初始化配置管理器...")
		instance = &Config{
			AppName: "MyApp",
			Version: "1.0.0",
			Debug:   true,
			data:    make(map[string]interface{}),
		}
	})
	return instance
}

// Set 设置配置项
func (c *Config) Set(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data[key] = value
}

// Get 获取配置项
func (c *Config) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	val, ok := c.data[key]
	return val, ok
}

// 演示单例模式
func main() {
	fmt.Println("=== 单例模式示例 (sync.Once) ===\n")

	// 多次调用 GetInstance，只会初始化一次
	config1 := GetInstance()
	fmt.Printf("config1: %p\n", config1)
	fmt.Printf("AppName: %s, Version: %s\n\n", config1.AppName, config1.Version)

	config2 := GetInstance()
	fmt.Printf("config2: %p\n", config2)
	fmt.Printf("config1 == config2: %v\n\n", config1 == config2)

	// 设置和获取配置
	config1.Set("database.host", "localhost")
	config1.Set("database.port", 3306)

	if host, ok := config2.Get("database.host"); ok {
		fmt.Printf("Database Host: %v\n", host)
	}
	if port, ok := config2.Get("database.port"); ok {
		fmt.Printf("Database Port: %v\n", port)
	}

	// 并发测试
	fmt.Println("\n=== 并发测试 ===")
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			cfg := GetInstance()
			fmt.Printf("Goroutine %d: %p\n", id, cfg)
		}(i)
	}
	wg.Wait()

	fmt.Println("\n✅ 所有 goroutine 获取的都是同一个实例")
}
