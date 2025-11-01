package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"
)

// Config 配置管理器（单例）- 完整版本
type Config struct {
	AppName   string
	Version   string
	Debug     bool
	mu        sync.RWMutex
	data      map[string]interface{}
	createdAt time.Time
	accessLog []AccessLog
}

// AccessLog 访问日志
type AccessLog struct {
	Timestamp time.Time
	Operation string
	Key       string
}

var (
	instance *Config
	once     sync.Once
)

// GetInstance 获取配置管理器实例
// 使用 sync.Once 确保只初始化一次
func GetInstance() *Config {
	once.Do(func() {
		fmt.Println("🔧 初始化配置管理器...")
		instance = &Config{
			AppName:   "MyApp",
			Version:   "1.0.0",
			Debug:     true,
			data:      make(map[string]interface{}),
			createdAt: time.Now(),
			accessLog: make([]AccessLog, 0),
		}
		fmt.Println("✅ 配置管理器初始化完成")
	})
	return instance
}

// Set 设置配置项
func (c *Config) Set(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	c.data[key] = value
	c.accessLog = append(c.accessLog, AccessLog{
		Timestamp: time.Now(),
		Operation: "SET",
		Key:       key,
	})
}

// Get 获取配置项
func (c *Config) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	c.accessLog = append(c.accessLog, AccessLog{
		Timestamp: time.Now(),
		Operation: "GET",
		Key:       key,
	})
	
	val, ok := c.data[key]
	return val, ok
}

// GetString 获取字符串配置
func (c *Config) GetString(key string) string {
	if val, ok := c.Get(key); ok {
		if str, ok := val.(string); ok {
			return str
		}
	}
	return ""
}

// GetInt 获取整数配置
func (c *Config) GetInt(key string) int {
	if val, ok := c.Get(key); ok {
		if num, ok := val.(int); ok {
			return num
		}
	}
	return 0
}

// GetBool 获取布尔配置
func (c *Config) GetBool(key string) bool {
	if val, ok := c.Get(key); ok {
		if b, ok := val.(bool); ok {
			return b
		}
	}
	return false
}

// Delete 删除配置项
func (c *Config) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	delete(c.data, key)
	c.accessLog = append(c.accessLog, AccessLog{
		Timestamp: time.Now(),
		Operation: "DELETE",
		Key:       key,
	})
}

// GetAll 获取所有配置
func (c *Config) GetAll() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	// 返回副本，避免外部修改
	result := make(map[string]interface{})
	for k, v := range c.data {
		result[k] = v
	}
	return result
}

// Size 获取配置项数量
func (c *Config) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.data)
}

// Clear 清空所有配置
func (c *Config) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	c.data = make(map[string]interface{})
	c.accessLog = append(c.accessLog, AccessLog{
		Timestamp: time.Now(),
		Operation: "CLEAR",
		Key:       "ALL",
	})
}

// GetAccessLog 获取访问日志
func (c *Config) GetAccessLog() []AccessLog {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	// 返回最近的 10 条日志
	start := 0
	if len(c.accessLog) > 10 {
		start = len(c.accessLog) - 10
	}
	return c.accessLog[start:]
}

// GetUptime 获取运行时间
func (c *Config) GetUptime() time.Duration {
	return time.Since(c.createdAt)
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

// LoadFromFile 从文件加载配置
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

// GetInfo 获取配置管理器信息
func (c *Config) GetInfo() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	return fmt.Sprintf(
		"Config Manager Info:\n"+
			"  App: %s v%s\n"+
			"  Debug: %v\n"+
			"  Items: %d\n"+
			"  Uptime: %v\n"+
			"  Access Log: %d entries",
		c.AppName, c.Version, c.Debug,
		len(c.data), c.GetUptime().Round(time.Second),
		len(c.accessLog),
	)
}

// 演示单例模式
func main() {
	fmt.Println("=== 单例模式示例 (sync.Once) - 完整版 ===\n")

	// 示例 1: 验证单例
	fmt.Println("【示例 1: 验证单例】")
	fmt.Println("-------------------")
	config1 := GetInstance()
	fmt.Printf("config1 地址: %p\n", config1)
	fmt.Printf("AppName: %s, Version: %s, Debug: %v\n", config1.AppName, config1.Version, config1.Debug)

	config2 := GetInstance()
	fmt.Printf("config2 地址: %p\n", config2)
	fmt.Printf("config1 == config2: %v\n", config1 == config2)
	
	if config1 == config2 {
		fmt.Println("✅ 确认是同一个实例")
	}
	fmt.Println()

	// 示例 2: 设置和获取配置
	fmt.Println("【示例 2: 设置和获取配置】")
	fmt.Println("-------------------")
	config1.Set("database.host", "localhost")
	config1.Set("database.port", 3306)
	config1.Set("database.name", "mydb")
	config1.Set("server.timeout", 30)
	config1.Set("server.debug", true)

	fmt.Printf("Database Host: %s\n", config2.GetString("database.host"))
	fmt.Printf("Database Port: %d\n", config2.GetInt("database.port"))
	fmt.Printf("Database Name: %s\n", config2.GetString("database.name"))
	fmt.Printf("Server Timeout: %d\n", config2.GetInt("server.timeout"))
	fmt.Printf("Server Debug: %v\n", config2.GetBool("server.debug"))
	fmt.Printf("配置项总数: %d\n", config2.Size())
	fmt.Println()

	// 示例 3: 并发测试
	fmt.Println("【示例 3: 并发安全测试】")
	fmt.Println("-------------------")
	fmt.Println("启动 100 个 goroutine 并发读写...")
	
	var wg sync.WaitGroup
	
	// 并发写入
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			cfg := GetInstance()
			key := fmt.Sprintf("concurrent.key%d", id)
			cfg.Set(key, id*10)
		}(i)
	}
	
	// 并发读取
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			cfg := GetInstance()
			key := fmt.Sprintf("concurrent.key%d", id)
			cfg.Get(key)
		}(i)
	}
	
	wg.Wait()
	fmt.Printf("✅ 并发操作完成，配置项总数: %d\n", config1.Size())
	fmt.Println()

	// 示例 4: 获取所有配置
	fmt.Println("【示例 4: 获取所有配置】")
	fmt.Println("-------------------")
	allConfig := config1.GetAll()
	fmt.Printf("所有配置项 (%d 个):\n", len(allConfig))
	count := 0
	for k, v := range allConfig {
		if count < 5 { // 只显示前 5 个
			fmt.Printf("  %s = %v\n", k, v)
		}
		count++
	}
	if count > 5 {
		fmt.Printf("  ... 还有 %d 个配置项\n", count-5)
	}
	fmt.Println()

	// 示例 5: 访问日志
	fmt.Println("【示例 5: 访问日志】")
	fmt.Println("-------------------")
	logs := config1.GetAccessLog()
	fmt.Printf("最近 %d 条访问日志:\n", len(logs))
	for i, log := range logs {
		fmt.Printf("  %d. [%s] %s %s\n", 
			i+1, 
			log.Timestamp.Format("15:04:05"), 
			log.Operation, 
			log.Key)
	}
	fmt.Println()

	// 示例 6: 配置管理器信息
	fmt.Println("【示例 6: 配置管理器信息】")
	fmt.Println("-------------------")
	fmt.Println(config1.GetInfo())
	fmt.Println()

	// 示例 7: 文件操作
	fmt.Println("【示例 7: 保存和加载配置】")
	fmt.Println("-------------------")
	
	// 清空并设置新配置
	config1.Clear()
	config1.Set("app.name", "TestApp")
	config1.Set("app.version", "2.0.0")
	config1.Set("app.port", 8080)
	
	// 保存到文件
	filename := "config_test.json"
	if err := config1.SaveToFile(filename); err != nil {
		fmt.Printf("❌ 保存失败: %v\n", err)
	} else {
		fmt.Printf("✅ 配置已保存到 %s\n", filename)
	}
	
	// 清空配置
	config1.Clear()
	fmt.Printf("清空后配置项数量: %d\n", config1.Size())
	
	// 从文件加载
	if err := config1.LoadFromFile(filename); err != nil {
		fmt.Printf("❌ 加载失败: %v\n", err)
	} else {
		fmt.Printf("✅ 配置已从 %s 加载\n", filename)
		fmt.Printf("加载后配置项数量: %d\n", config1.Size())
		fmt.Printf("App Name: %s\n", config1.GetString("app.name"))
		fmt.Printf("App Version: %s\n", config1.GetString("app.version"))
		fmt.Printf("App Port: %d\n", config1.GetInt("app.port"))
	}
	
	// 清理测试文件
	os.Remove(filename)
	fmt.Println()

	// 示例 8: 删除配置
	fmt.Println("【示例 8: 删除配置】")
	fmt.Println("-------------------")
	fmt.Printf("删除前配置项数量: %d\n", config1.Size())
	config1.Delete("app.port")
	fmt.Printf("删除后配置项数量: %d\n", config1.Size())
	fmt.Println()

	fmt.Println("=== 示例结束 ===")
	fmt.Println("\nsync.Once 单例模式特点:")
	fmt.Println("✅ 优点：")
	fmt.Println("  - 线程安全 - sync.Once 保证只初始化一次")
	fmt.Println("  - 延迟初始化 - 第一次调用时才创建")
	fmt.Println("  - 性能最优 - 无锁开销（初始化后）")
	fmt.Println("  - 实现简单 - Go 语言推荐方式")
	fmt.Println("  - 支持并发 - 可以安全地并发访问")
	fmt.Println("\n💡 最佳实践:")
	fmt.Println("  - 使用 sync.RWMutex 保护共享数据")
	fmt.Println("  - 提供类型安全的 Get 方法")
	fmt.Println("  - 添加访问日志用于调试")
	fmt.Println("  - 支持配置持久化")
	fmt.Println("  - 提供清晰的 API")
}
