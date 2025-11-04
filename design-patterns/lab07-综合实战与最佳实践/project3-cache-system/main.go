package main

import (
	"fmt"
	"time"
)

func main() {
	fmt.Println("=== 缓存系统演示 ===\n")

	// 1. 基本缓存操作演示
	demonstrateBasicCache()

	// 2. 缓存淘汰策略演示
	demonstrateEvictionStrategies()

	// 3. 缓存代理演示
	demonstrateCacheProxy()

	// 4. 缓存过期演示
	demonstrateCacheExpiration()

	// 5. 缓存统计演示
	demonstrateCacheStats()

	fmt.Println("\n=== 演示完成 ===")
}

// demonstrateBasicCache 演示基本缓存操作
func demonstrateBasicCache() {
	fmt.Println("--- 1. 基本缓存操作 ---")

	cache := GetCacheManager()
	cache.Clear()

	// 设置缓存
	cache.Set("key1", "value1", 0)
	fmt.Println("设置缓存: key1 = value1")

	cache.Set("key2", "value2", 0)
	fmt.Println("设置缓存: key2 = value2")

	cache.Set("key3", "value3", 0)
	fmt.Println("设置缓存: key3 = value3")

	// 获取缓存
	if value, ok := cache.Get("key1"); ok {
		fmt.Printf("获取缓存: key1 = %v\n", value)
	}

	if value, ok := cache.Get("key2"); ok {
		fmt.Printf("获取缓存: key2 = %v\n", value)
	}

	// 删除缓存
	cache.Delete("key1")
	fmt.Println("删除缓存: key1")

	if value, ok := cache.Get("key1"); !ok {
		fmt.Printf("获取缓存: key1 = %v (已删除)\n", value)
	}

	fmt.Println()
}

// demonstrateEvictionStrategies 演示缓存淘汰策略
func demonstrateEvictionStrategies() {
	fmt.Println("--- 2. 缓存淘汰策略 ---")

	// 1. LRU 策略
	fmt.Println("\n使用 LRU 策略:")
	lruCache := NewCacheManager(3, NewLRUStrategy())
	lruCache.Set("item1", "data1", 0)
	lruCache.Set("item2", "data2", 0)
	lruCache.Set("item3", "data3", 0)
	fmt.Println("  设置: item1, item2, item3")

	lruCache.Set("item4", "data4", 0)
	fmt.Println("  设置: item4")
	fmt.Println("  缓存已满，淘汰最久未使用的项")

	lruCache.Get("item2") // 访问 item2
	fmt.Println("  访问: item2")

	lruCache.Set("item5", "data5", 0)
	fmt.Println("  设置: item5")
	fmt.Println("  缓存已满，淘汰最久未使用的项")

	// 2. LFU 策略
	fmt.Println("\n使用 LFU 策略:")
	lfuCache := NewCacheManager(3, NewLFUStrategy())
	lfuCache.Set("item1", "data1", 0)
	lfuCache.Set("item2", "data2", 0)
	lfuCache.Set("item3", "data3", 0)
	fmt.Println("  设置: item1, item2, item3")

	// 访问 item1 多次
	lfuCache.Get("item1")
	lfuCache.Get("item1")
	lfuCache.Get("item1")
	fmt.Println("  访问 item1 3 次")

	// 访问 item2 少量次数
	lfuCache.Get("item2")
	lfuCache.Get("item2")
	fmt.Println("  访问 item2 2 次")

	lfuCache.Set("item4", "data4", 0)
	fmt.Println("  设置: item4")
	fmt.Println("  缓存已满，淘汰使用频率最低的项")

	// 3. FIFO 策略
	fmt.Println("\n使用 FIFO 策略:")
	fifoCache := NewCacheManager(3, NewFIFOStrategy())
	fifoCache.Set("item1", "data1", 0)
	fifoCache.Set("item2", "data2", 0)
	fifoCache.Set("item3", "data3", 0)
	fmt.Println("  设置: item1, item2, item3")

	fifoCache.Set("item4", "data4", 0)
	fmt.Println("  设置: item4")
	fmt.Println("  缓存已满，淘汰最先进入的项")

	fmt.Println()
}

// demonstrateCacheProxy 演示缓存代理
func demonstrateCacheProxy() {
	fmt.Println("--- 3. 缓存代理 ---")

	// 创建数据源
	dataSource := NewMemoryDataSource()
	dataSource.Set("user:1", "User 1")
	dataSource.Set("user:2", "User 2")
	dataSource.Set("user:3", "User 3")

	// 创建缓存代理
	cache := NewCacheManager(10, NewLRUStrategy())
	proxy := NewCacheProxy(dataSource, cache, 0)

	// 第一次访问（缓存未命中）
	fmt.Println("\n通过代理获取数据: user:1")
	value, err := proxy.Get("user:1")
	if err != nil {
		fmt.Printf("  错误: %v\n", err)
	} else {
		fmt.Printf("  数据: %v\n", value)
	}

	// 第二次访问（缓存命中）
	fmt.Println("\n通过代理获取数据: user:1")
	value, err = proxy.Get("user:1")
	if err != nil {
		fmt.Printf("  错误: %v\n", err)
	} else {
		fmt.Printf("  数据: %v\n", value)
	}

	// 访问其他数据
	fmt.Println("\n通过代理获取数据: user:2")
	value, err = proxy.Get("user:2")
	if err != nil {
		fmt.Printf("  错误: %v\n", err)
	} else {
		fmt.Printf("  数据: %v\n", value)
	}

	fmt.Println()
}

// demonstrateCacheExpiration 演示缓存过期
func demonstrateCacheExpiration() {
	fmt.Println("--- 4. 缓存过期 ---")

	cache := NewCacheManager(10, NewLRUStrategy())

	// 设置带过期时间的缓存
	cache.Set("temp", "data", 2*time.Second)
	fmt.Println("设置缓存: temp = data (TTL: 2s)")

	// 立即获取
	if value, ok := cache.Get("temp"); ok {
		fmt.Printf("立即获取: temp = %v\n", value)
	}

	// 等待过期
	fmt.Println("等待 3 秒...")
	time.Sleep(3 * time.Second)

	// 过期后获取
	if value, ok := cache.Get("temp"); !ok {
		fmt.Printf("过期后获取: temp = %v (已过期)\n", value)
	}

	fmt.Println()
}

// demonstrateCacheStats 演示缓存统计
func demonstrateCacheStats() {
	fmt.Println("--- 5. 缓存统计 ---")

	cache := NewCacheManager(10, NewLRUStrategy())

	// 执行一些操作
	cache.Set("key1", "value1", 0)
	cache.Set("key2", "value2", 0)
	cache.Set("key3", "value3", 0)

	cache.Get("key1") // 命中
	cache.Get("key2") // 命中
	cache.Get("key4") // 未命中
	cache.Get("key5") // 未命中

	// 获取统计信息
	stats := cache.GetStats()
	hitRate := cache.HitRate()

	fmt.Println("\n缓存统计:")
	fmt.Printf("  总请求: %d\n", stats.Hits+stats.Misses)
	fmt.Printf("  命中: %d\n", stats.Hits)
	fmt.Printf("  未命中: %d\n", stats.Misses)
	fmt.Printf("  命中率: %.2f%%\n", hitRate*100)

	fmt.Println()
}
