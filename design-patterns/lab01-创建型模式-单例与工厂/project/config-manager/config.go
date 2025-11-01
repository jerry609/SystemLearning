package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
)

// Config 配置管理器
type Config struct {
	data map[string]interface{}
	mu   sync.RWMutex
}

var (
	instance *Config
	once     sync.Once
)

// GetConfig 获取配置单例实例
func GetConfig() *Config {
	once.Do(func() {
		instance = &Config{
			data: make(map[string]interface{}),
		}
	})
	return instance
}

// Set 设置配置
func (c *Config) Set(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data[key] = value
}

// Get 获取配置
func (c *Config) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	value, ok := c.data[key]
	return value, ok
}

// GetString 获取字符串配置
func (c *Config) GetString(key string) string {
	if value, ok := c.Get(key); ok {
		if str, ok := value.(string); ok {
			return str
		}
	}
	return ""
}

// GetInt 获取整数配置
func (c *Config) GetInt(key string) int {
	if value, ok := c.Get(key); ok {
		switch v := value.(type) {
		case int:
			return v
		case float64:
			return int(v)
		case string:
			if i, err := strconv.Atoi(v); err == nil {
				return i
			}
		}
	}
	return 0
}

// GetBool 获取布尔配置
func (c *Config) GetBool(key string) bool {
	if value, ok := c.Get(key); ok {
		if b, ok := value.(bool); ok {
			return b
		}
	}
	return false
}

// LoadFromFile 从 JSON 文件加载配置
func (c *Config) LoadFromFile(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	var config map[string]interface{}
	if err := json.Unmarshal(data, &config); err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	for k, v := range config {
		c.data[k] = v
	}

	return nil
}

// LoadFromEnv 从环境变量加载配置
func (c *Config) LoadFromEnv(prefix string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, env := range os.Environ() {
		pair := strings.SplitN(env, "=", 2)
		if len(pair) != 2 {
			continue
		}

		key := pair[0]
		value := pair[1]

		if strings.HasPrefix(key, prefix) {
			configKey := strings.ToLower(strings.TrimPrefix(key, prefix))
			c.data[configKey] = value
		}
	}
}

// SaveToFile 保存配置到文件
func (c *Config) SaveToFile(filename string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	data, err := json.MarshalIndent(c.data, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filename, data, 0644)
}

// Clear 清空所有配置
func (c *Config) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data = make(map[string]interface{})
}

// Size 获取配置项数量
func (c *Config) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.data)
}

func main() {
	fmt.Println("=== 配置管理器实战项目 ===\n")

	// 示例 1: 基本使用
	fmt.Println("示例 1: 基本使用")
	fmt.Println("-------------------")
	config := GetConfig()

	fmt.Println("设置配置...")
	config.Set("app.name", "MyApp")
	config.Set("app.version", "1.0.0")
	config.Set("server.port", 8080)
	config.Set("server.debug", true)

	fmt.Printf("应用名称: %s\n", config.GetString("app.name"))
	fmt.Printf("版本号: %s\n", config.GetString("app.version"))
	fmt.Printf("服务器端口: %d\n", config.GetInt("server.port"))
	fmt.Printf("调试模式: %v\n", config.GetBool("server.debug"))
	fmt.Println()

	// 示例 2: 验证单例
	fmt.Println("示例 2: 验证单例")
	fmt.Println("-------------------")
	config1 := GetConfig()
	config2 := GetConfig()

	fmt.Printf("实例 1: %p\n", config1)
	fmt.Printf("实例 2: %p\n", config2)

	if config1 == config2 {
		fmt.Println("✅ config1 和 config2 是同一个实例")
	}
	fmt.Println()

	// 示例 3: 并发访问
	fmt.Println("示例 3: 并发访问")
	fmt.Println("-------------------")
	var wg sync.WaitGroup

	fmt.Println("启动 100 个 goroutine 并发读写...")
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			key := fmt.Sprintf("key.%d", index)
			config.Set(key, index)
			config.Get(key)
		}(i)
	}

	wg.Wait()
	fmt.Println("并发操作完成")
	fmt.Printf("配置项数量: %d\n", config.Size())

	fmt.Println("\n=== 示例结束 ===")
}
