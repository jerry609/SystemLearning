package main

import (
	"sync"
	"time"
)

// 缓存管理器 - 单例模式
//
// 本模块使用单例模式实现全局唯一的缓存管理器
// 使用 sync.Once 确保线程安全

// CacheItem 缓存项
type CacheItem struct {
	Key        string
	Value      interface{}
	ExpireTime time.Time
	CreateTime time.Time
	AccessTime time.Time
	AccessCount int
}

// IsExpired 检查是否过期
func (item *CacheItem) IsExpired() bool {
	if item.ExpireTime.IsZero() {
		return false
	}
	return time.Now().After(item.ExpireTime)
}

// CacheManager 缓存管理器（单例）
type CacheManager struct {
	mu       sync.RWMutex
	cache    map[string]*CacheItem
	strategy EvictionStrategy
	maxSize  int
	stats    *CacheStats
}

// CacheStats 缓存统计
type CacheStats struct {
	Hits   int64
	Misses int64
	Sets   int64
	Deletes int64
}

var (
	instance *CacheManager
	once     sync.Once
)

// GetCacheManager 获取缓存管理器单例
func GetCacheManager() *CacheManager {
	once.Do(func() {
		instance = &CacheManager{
			cache:    make(map[string]*CacheItem),
			strategy: NewLRUStrategy(),
			maxSize:  100,
			stats:    &CacheStats{},
		}
	})
	return instance
}

// NewCacheManager 创建新的缓存管理器（用于测试）
func NewCacheManager(maxSize int, strategy EvictionStrategy) *CacheManager {
	return &CacheManager{
		cache:    make(map[string]*CacheItem),
		strategy: strategy,
		maxSize:  maxSize,
		stats:    &CacheStats{},
	}
}

// Get 获取缓存
func (m *CacheManager) Get(key string) (interface{}, bool) {
	m.mu.RLock()
	item, ok := m.cache[key]
	m.mu.RUnlock()

	if !ok {
		m.stats.Misses++
		return nil, false
	}

	// 检查是否过期
	if item.IsExpired() {
		m.Delete(key)
		m.stats.Misses++
		return nil, false
	}

	// 更新访问信息
	m.mu.Lock()
	item.AccessTime = time.Now()
	item.AccessCount++
	m.mu.Unlock()

	// 通知策略
	if m.strategy != nil {
		m.strategy.OnAccess(key)
	}

	m.stats.Hits++
	return item.Value, true
}

// Set 设置缓存
func (m *CacheManager) Set(key string, value interface{}, ttl time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// 检查是否需要淘汰
	if len(m.cache) >= m.maxSize {
		if _, exists := m.cache[key]; !exists {
			// 新增项，需要淘汰
			if m.strategy != nil {
				evictKey := m.strategy.Evict(m.cache)
				if evictKey != "" {
					delete(m.cache, evictKey)
				}
			}
		}
	}

	now := time.Now()
	item := &CacheItem{
		Key:        key,
		Value:      value,
		CreateTime: now,
		AccessTime: now,
		AccessCount: 0,
	}

	if ttl > 0 {
		item.ExpireTime = now.Add(ttl)
	}

	m.cache[key] = item
	m.stats.Sets++

	// 通知策略
	if m.strategy != nil {
		m.strategy.OnSet(key)
	}
}

// Delete 删除缓存
func (m *CacheManager) Delete(key string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.cache, key)
	m.stats.Deletes++
}

// Clear 清空缓存
func (m *CacheManager) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.cache = make(map[string]*CacheItem)
}

// Size 获取缓存大小
func (m *CacheManager) Size() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return len(m.cache)
}

// Keys 获取所有键
func (m *CacheManager) Keys() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	keys := make([]string, 0, len(m.cache))
	for key := range m.cache {
		keys = append(keys, key)
	}
	return keys
}

// SetStrategy 设置淘汰策略
func (m *CacheManager) SetStrategy(strategy EvictionStrategy) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.strategy = strategy
}

// GetStats 获取缓存统计
func (m *CacheManager) GetStats() *CacheStats {
	return m.stats
}

// HitRate 获取命中率
func (m *CacheManager) HitRate() float64 {
	total := m.stats.Hits + m.stats.Misses
	if total == 0 {
		return 0
	}
	return float64(m.stats.Hits) / float64(total)
}

// CleanExpired 清理过期缓存
func (m *CacheManager) CleanExpired() int {
	m.mu.Lock()
	defer m.mu.Unlock()

	count := 0
	for key, item := range m.cache {
		if item.IsExpired() {
			delete(m.cache, key)
			count++
		}
	}
	return count
}

// StartCleanupTask 启动定期清理任务
func (m *CacheManager) StartCleanupTask(interval time.Duration) {
	ticker := time.NewTicker(interval)
	go func() {
		for range ticker.C {
			m.CleanExpired()
		}
	}()
}
