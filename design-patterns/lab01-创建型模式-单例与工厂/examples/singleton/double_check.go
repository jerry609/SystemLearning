package main

import (
	"fmt"
	"sync"
	"time"
)

// 双重检查锁单例模式示例
// 特点：线程安全，延迟初始化，性能较好

// Cache 缓存管理器
type Cache struct {
	data map[string]interface{}
	mu   sync.RWMutex
}

var (
	instance *Cache
	mu       sync.Mutex
)

// GetInstance 获取单例实例（双重检查锁）
func GetInstance() *Cache {
	// 第一次检查（无锁）
	if instance == nil {
		mu.Lock()
		defer mu.Unlock()
		// 第二次检查（有锁）
		if instance == nil {
			fmt.Println("创建新的 Cache 实例...")
			instance = &Cache{
				data: make(map[string]interface{}),
			}
		}
	}
	return instance
}

// Set 设置缓存
func (c *Cache) Set(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data[key] = value
}

// Get 获取缓存
func (c *Cache) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	value, ok := c.data[key]
	return value, ok
}

// Delete 删除缓存
func (c *Cache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.data, key)
}

// Size 获取缓存大小
func (c *Cache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.data)
}

// Clear 清空缓存
func (c *Cache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data = make(map[string]interface{})
}

func main() {
	fmt.Println("=== 双重检查锁单例模式示例 ===\n")

	// 测试并发访问
	fmt.Println("测试并发访问（10个 goroutine 同时获取实例）:")
	var wg sync.WaitGroup
	instances := make([]*Cache, 10)

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			instances[index] = GetInstance()
			fmt.Printf("Goroutine %d 获取实例: %p\n", index, instances[index])
		}(i)
	}

	wg.Wait()
	fmt.Println()

	// 验证所有实例是否相同
	fmt.Println("验证所有实例是否相同:")
	allSame := true
	firstInstance := instances[0]
	for i := 1; i < len(instances); i++ {
		if instances[i] != firstInstance {
			allSame = false
			break
		}
	}

	if allSame {
		fmt.Println("✅ 所有实例都是同一个对象")
	} else {
		fmt.Println("❌ 存在不同的实例")
	}
	fmt.Println()

	// 使用缓存
	fmt.Println("使用缓存:")
	cache := GetInstance()

	cache.Set("user:1", map[string]string{"name": "Alice", "email": "alice@example.com"})
	cache.Set("user:2", map[string]string{"name": "Bob", "email": "bob@example.com"})
	cache.Set("config:timeout", 30)

	fmt.Printf("缓存大小: %d\n", cache.Size())
	fmt.Println()

	// 获取缓存
	if value, ok := cache.Get("user:1"); ok {
		fmt.Printf("user:1 = %v\n", value)
	}

	if value, ok := cache.Get("config:timeout"); ok {
		fmt.Printf("config:timeout = %v\n", value)
	}
	fmt.Println()

	// 测试并发读写
	fmt.Println("测试并发读写:")
	var wg2 sync.WaitGroup

	// 写入数据
	for i := 0; i < 5; i++ {
		wg2.Add(1)
		go func(index int) {
			defer wg2.Done()
			key := fmt.Sprintf("key:%d", index)
			cache.Set(key, index*10)
			time.Sleep(10 * time.Millisecond)
		}(i)
	}

	// 读取数据
	for i := 0; i < 5; i++ {
		wg2.Add(1)
		go func(index int) {
			defer wg2.Done()
			key := fmt.Sprintf("key:%d", index)
			if value, ok := cache.Get(key); ok {
				fmt.Printf("读取 %s = %v\n", key, value)
			}
			time.Sleep(10 * time.Millisecond)
		}(i)
	}

	wg2.Wait()
	fmt.Println()

	fmt.Printf("最终缓存大小: %d\n", cache.Size())

	// 清空缓存
	cache.Clear()
	fmt.Printf("清空后缓存大小: %d\n", cache.Size())

	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n双重检查锁单例特点:")
	fmt.Println("✅ 线程安全 - 使用互斥锁保护")
	fmt.Println("✅ 延迟初始化 - 第一次使用时才创建")
	fmt.Println("✅ 性能较好 - 双重检查减少锁开销")
	fmt.Println("✅ 支持并发 - 可以安全地并发访问")
	fmt.Println("❌ 实现复杂 - 需要两次检查")
	fmt.Println("\n💡 在 Go 中，推荐使用 sync.Once 而不是双重检查锁")
}
