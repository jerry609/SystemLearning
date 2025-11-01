package main

import (
	"fmt"
	"sync"
	"time"
)

// RPC 代理模式示例
// 本示例展示了如何使用代理模式封装远程服务调用，提供本地接口并实现缓存、重试等功能

// User - 用户实体
type User struct {
	ID        int
	Name      string
	Email     string
	CreatedAt time.Time
}

// Order - 订单实体
type Order struct {
	ID         int
	UserID     int
	Amount     float64
	Status     string
	CreatedAt  time.Time
}

// UserService - 用户服务接口
type UserService interface {
	GetUser(id int) (*User, error)
	CreateUser(name, email string) (*User, error)
	UpdateUser(id int, name, email string) error
}

// OrderService - 订单服务接口
type OrderService interface {
	GetOrder(id int) (*Order, error)
	CreateOrder(userID int, amount float64) (*Order, error)
	GetUserOrders(userID int) ([]*Order, error)
}

// RemoteUserService - 远程用户服务（模拟RPC调用）
type RemoteUserService struct {
	serverURL string
	nextID    int
	mu        sync.Mutex
}

func NewRemoteUserService(url string) *RemoteUserService {
	return &RemoteUserService{
		serverURL: url,
		nextID:    1,
	}
}

func (s *RemoteUserService) GetUser(id int) (*User, error) {
	fmt.Printf("🌐 [RemoteService] Making RPC call to %s/users/%d\n", s.serverURL, id)
	// 模拟网络延迟
	time.Sleep(500 * time.Millisecond)
	
	// 模拟可能的网络错误
	if id < 0 {
		return nil, fmt.Errorf("invalid user ID")
	}
	
	return &User{
		ID:        id,
		Name:      fmt.Sprintf("User%d", id),
		Email:     fmt.Sprintf("user%d@example.com", id),
		CreatedAt: time.Now(),
	}, nil
}

func (s *RemoteUserService) CreateUser(name, email string) (*User, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	fmt.Printf("🌐 [RemoteService] Making RPC call to %s/users (POST)\n", s.serverURL)
	time.Sleep(500 * time.Millisecond)
	
	user := &User{
		ID:        s.nextID,
		Name:      name,
		Email:     email,
		CreatedAt: time.Now(),
	}
	s.nextID++
	
	return user, nil
}

func (s *RemoteUserService) UpdateUser(id int, name, email string) error {
	fmt.Printf("🌐 [RemoteService] Making RPC call to %s/users/%d (PUT)\n", s.serverURL, id)
	time.Sleep(500 * time.Millisecond)
	return nil
}

// RemoteOrderService - 远程订单服务（模拟RPC调用）
type RemoteOrderService struct {
	serverURL string
	nextID    int
	mu        sync.Mutex
}

func NewRemoteOrderService(url string) *RemoteOrderService {
	return &RemoteOrderService{
		serverURL: url,
		nextID:    1,
	}
}

func (s *RemoteOrderService) GetOrder(id int) (*Order, error) {
	fmt.Printf("🌐 [RemoteService] Making RPC call to %s/orders/%d\n", s.serverURL, id)
	time.Sleep(500 * time.Millisecond)
	
	return &Order{
		ID:        id,
		UserID:    1,
		Amount:    99.99,
		Status:    "completed",
		CreatedAt: time.Now(),
	}, nil
}

func (s *RemoteOrderService) CreateOrder(userID int, amount float64) (*Order, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	fmt.Printf("🌐 [RemoteService] Making RPC call to %s/orders (POST)\n", s.serverURL)
	time.Sleep(500 * time.Millisecond)
	
	order := &Order{
		ID:        s.nextID,
		UserID:    userID,
		Amount:    amount,
		Status:    "pending",
		CreatedAt: time.Now(),
	}
	s.nextID++
	
	return order, nil
}

func (s *RemoteOrderService) GetUserOrders(userID int) ([]*Order, error) {
	fmt.Printf("🌐 [RemoteService] Making RPC call to %s/users/%d/orders\n", s.serverURL, userID)
	time.Sleep(500 * time.Millisecond)
	
	// 模拟返回用户的订单列表
	return []*Order{
		{ID: 1, UserID: userID, Amount: 99.99, Status: "completed", CreatedAt: time.Now()},
		{ID: 2, UserID: userID, Amount: 149.99, Status: "pending", CreatedAt: time.Now()},
	}, nil
}

// UserServiceProxy - 用户服务代理（带缓存和重试）
type UserServiceProxy struct {
	service    UserService
	cache      map[int]*User
	cacheTTL   time.Duration
	cacheTime  map[int]time.Time
	maxRetries int
	retryDelay time.Duration
	mu         sync.RWMutex
	stats      ProxyStats
}

// ProxyStats - 代理统计信息
type ProxyStats struct {
	TotalCalls    int
	CacheHits     int
	RemoteCalls   int
	FailedCalls   int
	RetryAttempts int
	mu            sync.Mutex
}

func NewUserServiceProxy(serverURL string, maxRetries int) *UserServiceProxy {
	return &UserServiceProxy{
		service:    NewRemoteUserService(serverURL),
		cache:      make(map[int]*User),
		cacheTime:  make(map[int]time.Time),
		cacheTTL:   5 * time.Minute,
		maxRetries: maxRetries,
		retryDelay: time.Second,
	}
}

func (p *UserServiceProxy) GetUser(id int) (*User, error) {
	p.recordCall()
	
	// 检查缓存
	p.mu.RLock()
	if user, ok := p.cache[id]; ok {
		if time.Since(p.cacheTime[id]) < p.cacheTTL {
			p.mu.RUnlock()
			p.recordCacheHit()
			fmt.Printf("✅ [Proxy] Cache HIT for user ID: %d\n", id)
			return user, nil
		}
		fmt.Printf("⏰ [Proxy] Cache EXPIRED for user ID: %d\n", id)
	} else {
		fmt.Printf("❌ [Proxy] Cache MISS for user ID: %d\n", id)
	}
	p.mu.RUnlock()
	
	// 带重试的远程调用
	var user *User
	var err error
	
	for attempt := 0; attempt <= p.maxRetries; attempt++ {
		if attempt > 0 {
			p.recordRetry()
			fmt.Printf("🔄 [Proxy] Retry attempt %d/%d for user ID: %d\n", attempt, p.maxRetries, id)
			time.Sleep(p.retryDelay)
		}
		
		p.recordRemoteCall()
		user, err = p.service.GetUser(id)
		
		if err == nil {
			// 缓存成功的结果
			p.mu.Lock()
			p.cache[id] = user
			p.cacheTime[id] = time.Now()
			p.mu.Unlock()
			
			fmt.Printf("💾 [Proxy] User cached for ID: %d\n", id)
			return user, nil
		}
		
		fmt.Printf("⚠️  [Proxy] Remote call failed: %v\n", err)
	}
	
	p.recordFailure()
	return nil, fmt.Errorf("failed after %d retries: %w", p.maxRetries, err)
}

func (p *UserServiceProxy) CreateUser(name, email string) (*User, error) {
	p.recordCall()
	p.recordRemoteCall()
	
	user, err := p.service.CreateUser(name, email)
	if err != nil {
		p.recordFailure()
		return nil, err
	}
	
	// 缓存新创建的用户
	p.mu.Lock()
	p.cache[user.ID] = user
	p.cacheTime[user.ID] = time.Now()
	p.mu.Unlock()
	
	fmt.Printf("💾 [Proxy] New user cached for ID: %d\n", user.ID)
	return user, nil
}

func (p *UserServiceProxy) UpdateUser(id int, name, email string) error {
	p.recordCall()
	p.recordRemoteCall()
	
	err := p.service.UpdateUser(id, name, email)
	if err != nil {
		p.recordFailure()
		return err
	}
	
	// 使缓存失效
	p.mu.Lock()
	delete(p.cache, id)
	delete(p.cacheTime, id)
	p.mu.Unlock()
	
	fmt.Printf("🗑️  [Proxy] Cache invalidated for user ID: %d\n", id)
	return nil
}

// InvalidateCache - 使特定用户的缓存失效
func (p *UserServiceProxy) InvalidateCache(id int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.cache, id)
	delete(p.cacheTime, id)
	fmt.Printf("🗑️  [Proxy] Cache manually invalidated for user ID: %d\n", id)
}

// ClearCache - 清除所有缓存
func (p *UserServiceProxy) ClearCache() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.cache = make(map[int]*User)
	p.cacheTime = make(map[int]time.Time)
	fmt.Println("🗑️  [Proxy] All cache cleared")
}

// GetStats - 获取代理统计信息
func (p *UserServiceProxy) GetStats() ProxyStats {
	p.stats.mu.Lock()
	defer p.stats.mu.Unlock()
	return p.stats
}

func (p *UserServiceProxy) recordCall() {
	p.stats.mu.Lock()
	defer p.stats.mu.Unlock()
	p.stats.TotalCalls++
}

func (p *UserServiceProxy) recordCacheHit() {
	p.stats.mu.Lock()
	defer p.stats.mu.Unlock()
	p.stats.CacheHits++
}

func (p *UserServiceProxy) recordRemoteCall() {
	p.stats.mu.Lock()
	defer p.stats.mu.Unlock()
	p.stats.RemoteCalls++
}

func (p *UserServiceProxy) recordFailure() {
	p.stats.mu.Lock()
	defer p.stats.mu.Unlock()
	p.stats.FailedCalls++
}

func (p *UserServiceProxy) recordRetry() {
	p.stats.mu.Lock()
	defer p.stats.mu.Unlock()
	p.stats.RetryAttempts++
}

// OrderServiceProxy - 订单服务代理
type OrderServiceProxy struct {
	service OrderService
	mu      sync.Mutex
}

func NewOrderServiceProxy(serverURL string) *OrderServiceProxy {
	return &OrderServiceProxy{
		service: NewRemoteOrderService(serverURL),
	}
}

func (p *OrderServiceProxy) GetOrder(id int) (*Order, error) {
	fmt.Printf("📦 [OrderProxy] Forwarding GetOrder request for ID: %d\n", id)
	return p.service.GetOrder(id)
}

func (p *OrderServiceProxy) CreateOrder(userID int, amount float64) (*Order, error) {
	fmt.Printf("📦 [OrderProxy] Forwarding CreateOrder request for user: %d\n", userID)
	return p.service.CreateOrder(userID, amount)
}

func (p *OrderServiceProxy) GetUserOrders(userID int) ([]*Order, error) {
	fmt.Printf("📦 [OrderProxy] Forwarding GetUserOrders request for user: %d\n", userID)
	return p.service.GetUserOrders(userID)
}

// 演示函数
func demonstrateRPCProxy() {
	fmt.Println("=== RPC 代理模式示例 ===\n")
	
	// 创建用户服务代理
	userProxy := NewUserServiceProxy("https://api.example.com", 3)
	
	// 场景1: 基本RPC调用和缓存
	fmt.Println("📋 场景1: 基本RPC调用和缓存")
	fmt.Println("---")
	
	fmt.Println("\n第一次获取用户 (远程调用):")
	user, _ := userProxy.GetUser(1)
	fmt.Printf("用户信息: ID=%d, Name=%s, Email=%s\n", user.ID, user.Name, user.Email)
	
	fmt.Println("\n第二次获取用户 (缓存命中):")
	user, _ = userProxy.GetUser(1)
	fmt.Printf("用户信息: ID=%d, Name=%s, Email=%s\n", user.ID, user.Name, user.Email)
	
	// 场景2: 创建用户
	fmt.Println("\n\n📋 场景2: 创建新用户")
	fmt.Println("---")
	
	newUser, _ := userProxy.CreateUser("Alice", "alice@example.com")
	fmt.Printf("创建的用户: ID=%d, Name=%s, Email=%s\n", newUser.ID, newUser.Name, newUser.Email)
	
	fmt.Println("\n获取刚创建的用户 (缓存命中):")
	user, _ = userProxy.GetUser(newUser.ID)
	fmt.Printf("用户信息: ID=%d, Name=%s, Email=%s\n", user.ID, user.Name, user.Email)
	
	// 场景3: 更新用户（缓存失效）
	fmt.Println("\n\n📋 场景3: 更新用户（缓存失效）")
	fmt.Println("---")
	
	fmt.Println("\n更新用户信息:")
	userProxy.UpdateUser(1, "John Updated", "john.updated@example.com")
	
	fmt.Println("\n更新后获取用户 (缓存已失效，需要远程调用):")
	user, _ = userProxy.GetUser(1)
	fmt.Printf("用户信息: ID=%d, Name=%s, Email=%s\n", user.ID, user.Name, user.Email)
	
	// 场景4: 订单服务代理
	fmt.Println("\n\n📋 场景4: 订单服务代理")
	fmt.Println("---")
	
	orderProxy := NewOrderServiceProxy("https://api.example.com")
	
	fmt.Println("\n创建订单:")
	order, _ := orderProxy.CreateOrder(1, 199.99)
	fmt.Printf("订单信息: ID=%d, UserID=%d, Amount=%.2f, Status=%s\n", 
		order.ID, order.UserID, order.Amount, order.Status)
	
	fmt.Println("\n获取用户的所有订单:")
	orders, _ := orderProxy.GetUserOrders(1)
	for _, o := range orders {
		fmt.Printf("  订单 #%d: Amount=%.2f, Status=%s\n", o.ID, o.Amount, o.Status)
	}
	
	// 场景5: 统计信息
	fmt.Println("\n\n📋 场景5: 代理统计信息")
	fmt.Println("---")
	
	stats := userProxy.GetStats()
	fmt.Printf("总调用次数: %d\n", stats.TotalCalls)
	fmt.Printf("缓存命中: %d\n", stats.CacheHits)
	fmt.Printf("远程调用: %d\n", stats.RemoteCalls)
	fmt.Printf("失败调用: %d\n", stats.FailedCalls)
	fmt.Printf("重试次数: %d\n", stats.RetryAttempts)
	
	if stats.TotalCalls > 0 {
		cacheHitRate := float64(stats.CacheHits) / float64(stats.TotalCalls) * 100
		fmt.Printf("缓存命中率: %.2f%%\n", cacheHitRate)
	}
}

// 并发访问演示
func demonstrateConcurrentRPC() {
	fmt.Println("\n\n=== 并发RPC调用 ===\n")
	
	userProxy := NewUserServiceProxy("https://api.example.com", 3)
	
	var wg sync.WaitGroup
	
	// 模拟多个并发请求
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			// 每个goroutine请求不同的用户
			userID := (id % 3) + 1
			user, err := userProxy.GetUser(userID)
			if err != nil {
				fmt.Printf("Goroutine %d 获取用户失败: %v\n", id, err)
			} else {
				fmt.Printf("Goroutine %d 获取到用户: ID=%d, Name=%s\n", id, user.ID, user.Name)
			}
		}(i)
		
		time.Sleep(100 * time.Millisecond)
	}
	
	wg.Wait()
	
	// 显示统计信息
	stats := userProxy.GetStats()
	fmt.Printf("\n并发访问统计:\n")
	fmt.Printf("  总调用: %d 次\n", stats.TotalCalls)
	fmt.Printf("  缓存命中: %d 次\n", stats.CacheHits)
	fmt.Printf("  远程调用: %d 次\n", stats.RemoteCalls)
	
	if stats.TotalCalls > 0 {
		cacheHitRate := float64(stats.CacheHits) / float64(stats.TotalCalls) * 100
		fmt.Printf("  缓存命中率: %.2f%%\n", cacheHitRate)
	}
}

func main() {
	// 基本RPC代理演示
	demonstrateRPCProxy()
	
	// 并发RPC调用演示
	demonstrateConcurrentRPC()
	
	fmt.Println("\n=== 示例结束 ===")
}
