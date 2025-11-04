package main

import (
	"math/rand"
	"sync"
	"time"
)

// 负载均衡 - 策略模式
//
// 本模块使用策略模式实现多种负载均衡算法
// 支持轮询、随机、加权等策略

// LoadBalancer 负载均衡器接口
type LoadBalancer interface {
	Select(services []Service) Service
}

// RoundRobinBalancer 轮询负载均衡器
type RoundRobinBalancer struct {
	mu      sync.Mutex
	current int
}

func NewRoundRobinBalancer() *RoundRobinBalancer {
	return &RoundRobinBalancer{
		current: 0,
	}
}

func (b *RoundRobinBalancer) Select(services []Service) Service {
	if len(services) == 0 {
		return nil
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	service := services[b.current%len(services)]
	b.current++
	return service
}

// RandomBalancer 随机负载均衡器
type RandomBalancer struct {
	rand *rand.Rand
}

func NewRandomBalancer() *RandomBalancer {
	return &RandomBalancer{
		rand: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (b *RandomBalancer) Select(services []Service) Service {
	if len(services) == 0 {
		return nil
	}

	index := b.rand.Intn(len(services))
	return services[index]
}

// WeightedService 带权重的服务
type WeightedService struct {
	Service
	Weight int
}

// WeightedBalancer 加权负载均衡器
type WeightedBalancer struct {
	mu      sync.Mutex
	current int
}

func NewWeightedBalancer() *WeightedBalancer {
	return &WeightedBalancer{
		current: 0,
	}
}

func (b *WeightedBalancer) Select(services []Service) Service {
	if len(services) == 0 {
		return nil
	}

	// 将服务转换为加权服务
	weightedServices := make([]WeightedService, 0)
	for _, service := range services {
		if ws, ok := service.(*WeightedServiceImpl); ok {
			weightedServices = append(weightedServices, WeightedService{
				Service: ws,
				Weight:  ws.weight,
			})
		} else {
			// 默认权重为 1
			weightedServices = append(weightedServices, WeightedService{
				Service: service,
				Weight:  1,
			})
		}
	}

	// 计算总权重
	totalWeight := 0
	for _, ws := range weightedServices {
		totalWeight += ws.Weight
	}

	if totalWeight == 0 {
		return services[0]
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	// 加权轮询算法
	b.current = (b.current + 1) % totalWeight
	currentWeight := 0

	for _, ws := range weightedServices {
		currentWeight += ws.Weight
		if b.current < currentWeight {
			return ws.Service
		}
	}

	return services[0]
}

// WeightedServiceImpl 带权重的服务实现
type WeightedServiceImpl struct {
	name   string
	addr   string
	weight int
}

func NewWeightedService(name string, addr string, weight int) *WeightedServiceImpl {
	return &WeightedServiceImpl{
		name:   name,
		addr:   addr,
		weight: weight,
	}
}

func (s *WeightedServiceImpl) Name() string {
	return s.name
}

func (s *WeightedServiceImpl) Address() string {
	return s.addr
}

func (s *WeightedServiceImpl) Call(method string, args interface{}) (interface{}, error) {
	return nil, nil
}

func (s *WeightedServiceImpl) Weight() int {
	return s.weight
}

// LeastConnectionBalancer 最少连接负载均衡器
type LeastConnectionBalancer struct {
	mu          sync.Mutex
	connections map[string]int
}

func NewLeastConnectionBalancer() *LeastConnectionBalancer {
	return &LeastConnectionBalancer{
		connections: make(map[string]int),
	}
}

func (b *LeastConnectionBalancer) Select(services []Service) Service {
	if len(services) == 0 {
		return nil
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	// 找到连接数最少的服务
	var selected Service
	minConnections := -1

	for _, service := range services {
		addr := service.Address()
		connections := b.connections[addr]

		if minConnections == -1 || connections < minConnections {
			minConnections = connections
			selected = service
		}
	}

	// 增加连接数
	if selected != nil {
		b.connections[selected.Address()]++
	}

	return selected
}

// ReleaseConnection 释放连接
func (b *LeastConnectionBalancer) ReleaseConnection(service Service) {
	b.mu.Lock()
	defer b.mu.Unlock()

	addr := service.Address()
	if b.connections[addr] > 0 {
		b.connections[addr]--
	}
}

// LoadBalancerFactory 负载均衡器工厂
type LoadBalancerFactory struct{}

func (f *LoadBalancerFactory) CreateBalancer(strategy string) LoadBalancer {
	switch strategy {
	case "round-robin":
		return NewRoundRobinBalancer()
	case "random":
		return NewRandomBalancer()
	case "weighted":
		return NewWeightedBalancer()
	case "least-connection":
		return NewLeastConnectionBalancer()
	default:
		return NewRoundRobinBalancer()
	}
}
