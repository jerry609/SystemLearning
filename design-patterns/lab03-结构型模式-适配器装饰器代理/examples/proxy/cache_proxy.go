package main

import (
	"fmt"
	"sync"
	"time"
)

// 缓存代理模式示例
// 本示例展示了如何使用代理模式实现缓存功能，减少对后端服务的重复请求

// DataService - 数据服务接口
type DataService interface {
	GetData(key string) (string, error)
	GetUserProfile(userID int) (*UserProfile, error)
}

// UserProfile - 用户资料
type UserProfile struct {
	ID       int
	Name     string
	Email    string
	CreateAt time.Time
}

// RealDataService - 真实的数据服务（模拟数据库或远程API）
type RealDataService struct{}

func NewRealDataService() *RealDataService {
	return &RealDataService{}
}

func (s *RealDataService) GetData(key string) (string, error) {
	fmt.Printf("📡 [RealService] Fetching data from database for key: %s\n", key)
	// 模拟数据库查询延迟
	time.Sleep(1 * time.Second)
	return fmt.Sprintf("Data for %s (fetched at %s)", key, time.Now().Format("15:04:05")), nil
}

func (s *RealDataService) GetUserProfile(userID int) (*UserProfile, error) {
	fmt.Printf("📡 [RealService] Fetching user profile from database for ID: %d\n", userID)
	// 模拟数据库查询延迟
	time.Sleep(1 * time.Second)
	return &UserProfile{
		ID:       userID,
		Name:     fmt.Sprintf("User%d", userID),
		Email:    fmt.Sprintf("user%d@example.com", userID),
		CreateAt: time.Now(),
	}, nil
}

// CacheEntry - 缓存条目
type CacheEntry struct {
	data      interface{}
	timestamp time.Time
}

// CacheProxy - 缓存代理
type CacheProxy struct {
	service   DataService
	cache     map[string]CacheEntry
	cacheTTL  time.Duration
	mu        sync.RWMutex
	stats     CacheStats
}

// CacheStats - 缓存统计信息
type CacheStats struct {
	Hits   int
	Misses int
	mu     sync.Mutex
}

func NewCacheProxy(service DataService, ttl time.Duration) *CacheProxy {
	return &CacheProxy{
		service:  service,
		cache:    make(map[string]CacheEntry),
		cacheTTL: ttl,
	}
}

func (p *CacheProxy) GetData(key string) (string, error) {
	// 检查缓存（使用读锁）
	p.mu.RLock()
	if entry, ok := p.cache[key]; ok {
		if time.Since(entry.timestamp) < p.cacheTTL {
			p.mu.RUnlock()
			p.recordHit()
			fmt.Printf("✅ [CacheProxy] Cache HIT for key: %s (age: %.1fs)\n", 
				key, time.Since(entry.timestamp).Seconds())
			return entry.data.(string), nil
		}
		fmt.Printf("⏰ [CacheProxy] Cache EXPIRED for key: %s (age: %.1fs)\n", 
			key, time.Since(entry.timestamp).Seconds())
	} else {
		fmt.Printf("❌ [CacheProxy] Cache MISS for key: %s\n", key)
	}
	p.mu.RUnlock()
	
	p.recordMiss()
	
	// 从真实服务获取数据
	data, err := p.service.GetData(key)
	if err != nil {
		return "", err
	}
	
	// 更新缓存（使用写锁）
	p.mu.Lock()
	p.cache[key] = CacheEntry{
		data:      data,
		timestamp: time.Now(),
	}
	p.mu.Unlock()
	
	fmt.Printf("💾 [CacheProxy] Data cached for key: %s\n", key)
	return data, nil
}

func (p *CacheProxy) GetUserProfile(userID int) (*UserProfile, error) {
	key := fmt.Sprintf("user:%d", userID)
	
	// 检查缓存
	p.mu.RLock()
	if entry, ok := p.cache[key]; ok {
		if time.Since(entry.timestamp) < p.cacheTTL {
			p.mu.RUnlock()
			p.recordHit()
			fmt.Printf("✅ [CacheProxy] Cache HIT for user: %d\n", userID)
			return entry.data.(*UserProfile), nil
		}
		fmt.Printf("⏰ [CacheProxy] Cache EXPIRED for user: %d\n", userID)
	} else {
		fmt.Printf("❌ [CacheProxy] Cache MISS for user: %d\n", userID)
	}
	p.mu.RUnlock()
	
	p.recordMiss()
	
	// 从真实服务获取数据
	profile, err := p.service.GetUserProfile(userID)
	if err != nil {
		return nil, err
	}
	
	// 更新缓存
	p.mu.Lock()
	p.cache[key] = CacheEntry{
		data:      profile,
		timestamp: time.Now(),
	}
	p.mu.Unlock()
	
	fmt.Printf("💾 [CacheProxy] User profile cached for ID: %d\n", userID)
	return profile, nil
}

// ClearCache - 清除所有缓存
func (p *CacheProxy) ClearCache() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.cache = make(map[string]CacheEntry)
	fmt.Println("🗑️  [CacheProxy] Cache cleared")
}

// InvalidateKey - 使特定键的缓存失效
func (p *CacheProxy) InvalidateKey(key string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.cache, key)
	fmt.Printf("🗑️  [CacheProxy] Cache invalidated for key: %s\n", key)
}

// GetCacheStats - 获取缓存统计信息
func (p *CacheProxy) GetCacheStats() (hits, misses int, hitRate float64) {
	p.stats.mu.Lock()
	defer p.stats.mu.Unlock()
	
	hits = p.stats.Hits
	misses = p.stats.Misses
	total := hits + misses
	if total > 0 {
		hitRate = float64(hits) / float64(total) * 100
	}
	return
}

func (p *CacheProxy) recordHit() {
	p.stats.mu.Lock()
	defer p.stats.mu.Unlock()
	p.stats.Hits++
}

func (p *CacheProxy) recordMiss() {
	p.stats.mu.Lock()
	defer p.stats.mu.Unlock()
	p.stats.Misses++
}

// 演示函数
func demonstrateCacheProxy() {
	fmt.Println("=== 缓存代理模式示例 ===\n")
	
	// 创建真实服务和缓存代理
	realService := NewRealDataService()
	cacheProxy := NewCacheProxy(realService, 5*time.Second)
	
	// 场景1: 基本缓存功能
	fmt.Println("📋 场景1: 基本缓存功能")
	fmt.Println("---")
	
	// 第一次访问（缓存未命中）
	fmt.Println("\n第一次访问 key1:")
	data, _ := cacheProxy.GetData("key1")
	fmt.Printf("结果: %s\n", data)
	
	// 第二次访问（缓存命中）
	fmt.Println("\n第二次访问 key1:")
	data, _ = cacheProxy.GetData("key1")
	fmt.Printf("结果: %s\n", data)
	
	// 场景2: 缓存过期
	fmt.Println("\n\n📋 场景2: 缓存过期")
	fmt.Println("---")
	fmt.Println("等待缓存过期 (6秒)...")
	time.Sleep(6 * time.Second)
	
	fmt.Println("\n缓存过期后访问 key1:")
	data, _ = cacheProxy.GetData("key1")
	fmt.Printf("结果: %s\n", data)
	
	// 场景3: 用户资料缓存
	fmt.Println("\n\n📋 场景3: 用户资料缓存")
	fmt.Println("---")
	
	fmt.Println("\n第一次获取用户资料:")
	profile, _ := cacheProxy.GetUserProfile(123)
	fmt.Printf("用户信息: ID=%d, Name=%s, Email=%s\n", profile.ID, profile.Name, profile.Email)
	
	fmt.Println("\n第二次获取用户资料 (缓存命中):")
	profile, _ = cacheProxy.GetUserProfile(123)
	fmt.Printf("用户信息: ID=%d, Name=%s, Email=%s\n", profile.ID, profile.Name, profile.Email)
	
	// 场景4: 缓存失效
	fmt.Println("\n\n📋 场景4: 手动使缓存失效")
	fmt.Println("---")
	cacheProxy.InvalidateKey("user:123")
	
	fmt.Println("\n缓存失效后获取用户资料:")
	profile, _ = cacheProxy.GetUserProfile(123)
	fmt.Printf("用户信息: ID=%d, Name=%s, Email=%s\n", profile.ID, profile.Name, profile.Email)
	
	// 场景5: 缓存统计
	fmt.Println("\n\n📋 场景5: 缓存统计信息")
	fmt.Println("---")
	hits, misses, hitRate := cacheProxy.GetCacheStats()
	fmt.Printf("缓存命中: %d 次\n", hits)
	fmt.Printf("缓存未命中: %d 次\n", misses)
	fmt.Printf("命中率: %.2f%%\n", hitRate)
}

// 并发访问演示
func demonstrateConcurrentAccess() {
	fmt.Println("\n\n=== 并发访问缓存代理 ===\n")
	
	realService := NewRealDataService()
	cacheProxy := NewCacheProxy(realService, 10*time.Second)
	
	var wg sync.WaitGroup
	
	// 模拟10个并发请求访问相同的数据
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			data, _ := cacheProxy.GetData("shared-key")
			fmt.Printf("Goroutine %d 获取到数据: %s\n", id, data)
		}(i)
		
		// 稍微错开请求时间
		time.Sleep(100 * time.Millisecond)
	}
	
	wg.Wait()
	
	// 显示统计信息
	hits, misses, hitRate := cacheProxy.GetCacheStats()
	fmt.Printf("\n并发访问统计:\n")
	fmt.Printf("  缓存命中: %d 次\n", hits)
	fmt.Printf("  缓存未命中: %d 次\n", misses)
	fmt.Printf("  命中率: %.2f%%\n", hitRate)
}

func main() {
	// 基本缓存代理演示
	demonstrateCacheProxy()
	
	// 并发访问演示
	demonstrateConcurrentAccess()
	
	fmt.Println("\n=== 示例结束 ===")
}
