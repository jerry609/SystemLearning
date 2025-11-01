package main

import (
	"fmt"
	"sync"
	"time"
)

// 练习 1: 线程安全的缓存管理器 - 参考答案

// CacheItem 缓存项
type CacheItem struct {
	Value      interface{}
	ExpireTime time.Time
}

// CacheStats 缓存统计信息
type CacheStats struct {
	Hits       int64
	Misses     int64
	TotalItems int
}

// Cache 缓存管理器
type Cache struct {
	data  map[string]*CacheItem
	mu    sync.RWMutex
	stats CacheStats
}

var (
	instance *Cache
	once     sync.Once
)

// GetCacheInstance 获取缓存单例实例
func GetCacheInstance() *Cache {
	once.Do(func() {
		instance = &Cache{
			data: make(map[string]*CacheItem),
		}
		// 启动清理 goroutine
		go instance.cleanupExpired()
	})
	return instance
}

// Set 设置缓存
func (c *Cache) Set(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.data[key] = &CacheItem{
		Value:      value,
		ExpireTime: time.Time{}, // 零值表示永不过期
	}
}

// SetWithExpiry 设置带过期时间的缓存
func (c *Cache) SetWithExpiry(key string, value interface{}, duration time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.data[key] = &CacheItem{
		Value:      value,
		ExpireTime: time.Now().Add(duration),
	}
}

// Get 获取缓存
func (c *Cache) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	item, ok := c.data[key]
	if !ok {
		c.stats.Misses++
		return nil, false
	}

	// 检查是否过期
	if !item.ExpireTime.IsZero() && time.Now().After(item.ExpireTime) {
		c.stats.Misses++
		return nil, false
	}

	c.stats.Hits++
	return item.Value, true
}

// Delete 删除缓存
func (c *Cache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.data, key)
}

// Clear 清空所有缓存
func (c *Cache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.data = make(map[string]*CacheItem)
	c.stats = CacheStats{}
}

// Size 获取缓存数量
func (c *Cache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return len(c.data)
}

// GetStats 获取缓存统计信息
func (c *Cache) GetStats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	stats := c.stats
	stats.TotalItems = len(c.data)
	return stats
}

// cleanupExpired 定期清理过期项
func (c *Cache) cleanupExpired() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		c.mu.Lock()
		now := time.Now()
		for key, item := range c.data {
			if !item.ExpireTime.IsZero() && now.After(item.ExpireTime) {
				delete(c.data, key)
			}
		}
		c.mu.Unlock()
	}
}

// ResetCacheInstance 重置缓存实例（仅用于测试）
func ResetCacheInstance() {
	instance = nil
	once = sync.Once{}
}

func main() {
	fmt.Println("=== 线程安全的缓存管理器 - 参考答案 ===\n")

	// 示例 1: 基本操作
	fmt.Println("示例 1: 基本操作")
	fmt.Println("-------------------")
	cache := GetCacheInstance()

	// 设置缓存
	cache.Set("user:1", map[string]string{"name": "Alice", "email": "alice@example.com"})
	cache.Set("user:2", map[string]string{"name": "Bob", "email": "bob@example.com"})
	cache.Set("user:3", map[string]string{"name": "Charlie", "email": "charlie@example.com"})

	fmt.Printf("缓存大小: %d\n", cache.Size())

	// 获取缓存
	if value, ok := cache.Get("user:1"); ok {
		fmt.Printf("找到 user:1: %v\n", value)
	}

	if _, ok := cache.Get("user:999"); !ok {
		fmt.Println("未找到 user:999")
	}

	// 删除缓存
	cache.Delete("user:2")
	fmt.Printf("删除 user:2 后，缓存大小: %d\n", cache.Size())
	fmt.Println()

	// 示例 2: 过期时间
	fmt.Println("示例 2: 过期时间")
	fmt.Println("-------------------")

	// 设置 3 秒后过期
	cache.SetWithExpiry("session:abc", "user123", 3*time.Second)
	fmt.Println("设置 session:abc，3 秒后过期")

	// 立即获取
	if value, ok := cache.Get("session:abc"); ok {
		fmt.Printf("立即获取: %v\n", value)
	}

	// 等待 4 秒
	fmt.Println("等待 4 秒...")
	time.Sleep(4 * time.Second)

	// 再次获取
	if _, ok := cache.Get("session:abc"); !ok {
		fmt.Println("session:abc 已过期")
	}
	fmt.Println()

	// 示例 3: 并发访问
	fmt.Println("示例 3: 并发访问")
	fmt.Println("-------------------")

	cache.Clear() // 清空缓存
	var wg sync.WaitGroup

	// 并发写入
	fmt.Println("启动 100 个 goroutine 并发写入...")
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			key := fmt.Sprintf("key:%d", index)
			cache.Set(key, index*10)
		}(i)
	}

	// 并发读取
	fmt.Println("启动 100 个 goroutine 并发读取...")
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			key := fmt.Sprintf("key:%d", index)
			cache.Get(key)
		}(i)
	}

	wg.Wait()
	fmt.Printf("并发操作完成，缓存大小: %d\n", cache.Size())
	fmt.Println()

	// 示例 4: 统计信息
	fmt.Println("示例 4: 统计信息")
	fmt.Println("-------------------")

	// 执行一些操作
	cache.Get("key:1")
	cache.Get("key:2")
	cache.Get("key:999") // 未命中
	cache.Get("key:888") // 未命中

	stats := cache.GetStats()
	fmt.Printf("总项数: %d\n", stats.TotalItems)
	fmt.Printf("命中次数: %d\n", stats.Hits)
	fmt.Printf("未命中次数: %d\n", stats.Misses)
	if stats.Hits+stats.Misses > 0 {
		hitRate := float64(stats.Hits) / float64(stats.Hits+stats.Misses) * 100
		fmt.Printf("命中率: %.2f%%\n", hitRate)
	}
	fmt.Println()

	// 示例 5: 验证单例
	fmt.Println("示例 5: 验证单例")
	fmt.Println("-------------------")

	cache1 := GetCacheInstance()
	cache2 := GetCacheInstance()

	fmt.Printf("cache1 地址: %p\n", cache1)
	fmt.Printf("cache2 地址: %p\n", cache2)

	if cache1 == cache2 {
		fmt.Println("✅ cache1 和 cache2 是同一个实例")
	} else {
		fmt.Println("❌ cache1 和 cache2 不是同一个实例")
	}

	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n实现要点:")
	fmt.Println("✅ 使用 sync.Once 确保单例")
	fmt.Println("✅ 使用 sync.RWMutex 实现读写锁")
	fmt.Println("✅ 支持过期时间和自动清理")
	fmt.Println("✅ 提供统计信息")
	fmt.Println("✅ 线程安全，支持并发访问")
}
