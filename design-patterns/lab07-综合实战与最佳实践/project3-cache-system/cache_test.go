package main

import (
	"testing"
	"time"
)

// TestCacheManager 测试缓存管理器
func TestCacheManager(t *testing.T) {
	cache := NewCacheManager(10, NewLRUStrategy())

	// 测试设置和获取
	cache.Set("key1", "value1", 0)
	value, ok := cache.Get("key1")
	if !ok || value != "value1" {
		t.Errorf("期望 value1，实际 %v", value)
	}

	// 测试删除
	cache.Delete("key1")
	_, ok = cache.Get("key1")
	if ok {
		t.Error("删除后不应该获取到值")
	}
}

// TestCacheExpiration 测试缓存过期
func TestCacheExpiration(t *testing.T) {
	cache := NewCacheManager(10, NewLRUStrategy())

	// 设置带过期时间的缓存
	cache.Set("temp", "data", 100*time.Millisecond)

	// 立即获取应该成功
	_, ok := cache.Get("temp")
	if !ok {
		t.Error("立即获取应该成功")
	}

	// 等待过期
	time.Sleep(200 * time.Millisecond)

	// 过期后获取应该失败
	_, ok = cache.Get("temp")
	if ok {
		t.Error("过期后获取应该失败")
	}
}

// TestLRUStrategy 测试 LRU 策略
func TestLRUStrategy(t *testing.T) {
	cache := NewCacheManager(3, NewLRUStrategy())

	// 填满缓存
	cache.Set("key1", "value1", 0)
	cache.Set("key2", "value2", 0)
	cache.Set("key3", "value3", 0)

	// 访问 key1
	cache.Get("key1")

	// 添加新项，应该淘汰 key2（最久未使用）
	cache.Set("key4", "value4", 0)

	// key2 应该被淘汰
	_, ok := cache.Get("key2")
	if ok {
		t.Error("key2 应该被淘汰")
	}

	// key1 应该还在
	_, ok = cache.Get("key1")
	if !ok {
		t.Error("key1 应该还在")
	}
}

// TestLFUStrategy 测试 LFU 策略
func TestLFUStrategy(t *testing.T) {
	cache := NewCacheManager(3, NewLFUStrategy())

	// 填满缓存
	cache.Set("key1", "value1", 0)
	cache.Set("key2", "value2", 0)
	cache.Set("key3", "value3", 0)

	// 多次访问 key1
	cache.Get("key1")
	cache.Get("key1")
	cache.Get("key1")

	// 访问 key2
	cache.Get("key2")

	// 添加新项，应该淘汰 key3（使用频率最低）
	cache.Set("key4", "value4", 0)

	// key3 应该被淘汰
	_, ok := cache.Get("key3")
	if ok {
		t.Error("key3 应该被淘汰")
	}
}

// TestFIFOStrategy 测试 FIFO 策略
func TestFIFOStrategy(t *testing.T) {
	cache := NewCacheManager(3, NewFIFOStrategy())

	// 填满缓存
	cache.Set("key1", "value1", 0)
	cache.Set("key2", "value2", 0)
	cache.Set("key3", "value3", 0)

	// 添加新项，应该淘汰 key1（最先进入）
	cache.Set("key4", "value4", 0)

	// key1 应该被淘汰
	_, ok := cache.Get("key1")
	if ok {
		t.Error("key1 应该被淘汰")
	}

	// key2 应该还在
	_, ok = cache.Get("key2")
	if !ok {
		t.Error("key2 应该还在")
	}
}

// TestCacheProxy 测试缓存代理
func TestCacheProxy(t *testing.T) {
	dataSource := NewMemoryDataSource()
	dataSource.Set("key1", "value1")

	cache := NewCacheManager(10, NewLRUStrategy())
	proxy := NewCacheProxy(dataSource, cache, 0)

	// 第一次获取（缓存未命中）
	value, err := proxy.Get("key1")
	if err != nil {
		t.Errorf("获取失败: %v", err)
	}
	if value != "value1" {
		t.Errorf("期望 value1，实际 %v", value)
	}

	// 第二次获取（缓存命中）
	value, err = proxy.Get("key1")
	if err != nil {
		t.Errorf("获取失败: %v", err)
	}
	if value != "value1" {
		t.Errorf("期望 value1，实际 %v", value)
	}

	// 验证缓存命中
	stats := cache.GetStats()
	if stats.Hits != 1 {
		t.Errorf("期望 1 次命中，实际 %d 次", stats.Hits)
	}
}

// TestCacheStats 测试缓存统计
func TestCacheStats(t *testing.T) {
	cache := NewCacheManager(10, NewLRUStrategy())

	// 执行一些操作
	cache.Set("key1", "value1", 0)
	cache.Get("key1") // 命中
	cache.Get("key2") // 未命中

	stats := cache.GetStats()

	if stats.Hits != 1 {
		t.Errorf("期望 1 次命中，实际 %d 次", stats.Hits)
	}

	if stats.Misses != 1 {
		t.Errorf("期望 1 次未命中，实际 %d 次", stats.Misses)
	}

	hitRate := cache.HitRate()
	expectedRate := 0.5
	if hitRate != expectedRate {
		t.Errorf("期望命中率 %.2f，实际 %.2f", expectedRate, hitRate)
	}
}

// TestCleanExpired 测试清理过期缓存
func TestCleanExpired(t *testing.T) {
	cache := NewCacheManager(10, NewLRUStrategy())

	// 设置一些缓存，部分带过期时间
	cache.Set("key1", "value1", 0)
	cache.Set("key2", "value2", 100*time.Millisecond)
	cache.Set("key3", "value3", 100*time.Millisecond)

	// 等待过期
	time.Sleep(200 * time.Millisecond)

	// 清理过期缓存
	count := cache.CleanExpired()

	if count != 2 {
		t.Errorf("期望清理 2 个过期缓存，实际 %d 个", count)
	}

	// key1 应该还在
	_, ok := cache.Get("key1")
	if !ok {
		t.Error("key1 应该还在")
	}

	// key2 和 key3 应该被清理
	_, ok = cache.Get("key2")
	if ok {
		t.Error("key2 应该被清理")
	}
}

// TestMemoryDataSource 测试内存数据源
func TestMemoryDataSource(t *testing.T) {
	ds := NewMemoryDataSource()

	// 测试设置和获取
	ds.Set("key1", "value1")
	value, err := ds.Get("key1")
	if err != nil {
		t.Errorf("获取失败: %v", err)
	}
	if value != "value1" {
		t.Errorf("期望 value1，实际 %v", value)
	}

	// 测试删除
	ds.Delete("key1")
	_, err = ds.Get("key1")
	if err == nil {
		t.Error("删除后获取应该失败")
	}
}

// TestStrategyFactory 测试策略工厂
func TestStrategyFactory(t *testing.T) {
	factory := &StrategyFactory{}

	strategies := []string{"lru", "lfu", "fifo", "ttl", "random"}

	for _, strategyType := range strategies {
		strategy := factory.CreateStrategy(strategyType)
		if strategy == nil {
			t.Errorf("创建 %s 策略失败", strategyType)
		}
	}
}

// BenchmarkCacheGet 基准测试：缓存获取
func BenchmarkCacheGet(b *testing.B) {
	cache := NewCacheManager(1000, NewLRUStrategy())
	cache.Set("key", "value", 0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.Get("key")
	}
}

// BenchmarkCacheSet 基准测试：缓存设置
func BenchmarkCacheSet(b *testing.B) {
	cache := NewCacheManager(1000, NewLRUStrategy())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.Set("key", "value", 0)
	}
}
