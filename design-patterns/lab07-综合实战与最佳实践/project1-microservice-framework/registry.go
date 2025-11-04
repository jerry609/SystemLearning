package main

import (
	"fmt"
	"sync"
)

// 服务注册与发现 - 工厂模式
//
// 本模块使用工厂模式创建和管理服务实例
// 提供服务注册、注销和发现功能

// Service 服务接口
type Service interface {
	Name() string
	Address() string
	Call(method string, args interface{}) (interface{}, error)
}

// ServiceFactory 服务工厂接口
type ServiceFactory interface {
	CreateService(name string, addr string) Service
}

// HTTPService HTTP 服务实现
type HTTPService struct {
	name string
	addr string
}

func (s *HTTPService) Name() string {
	return s.name
}

func (s *HTTPService) Address() string {
	return s.addr
}

func (s *HTTPService) Call(method string, args interface{}) (interface{}, error) {
	return fmt.Sprintf("HTTP调用 %s.%s(%v)", s.name, method, args), nil
}

// RPCService RPC 服务实现
type RPCService struct {
	name string
	addr string
}

func (s *RPCService) Name() string {
	return s.name
}

func (s *RPCService) Address() string {
	return s.addr
}

func (s *RPCService) Call(method string, args interface{}) (interface{}, error) {
	return fmt.Sprintf("RPC调用 %s.%s(%v)", s.name, method, args), nil
}

// DefaultServiceFactory 默认服务工厂
type DefaultServiceFactory struct {
	serviceType string
}

func NewServiceFactory(serviceType string) ServiceFactory {
	return &DefaultServiceFactory{serviceType: serviceType}
}

func (f *DefaultServiceFactory) CreateService(name string, addr string) Service {
	switch f.serviceType {
	case "http":
		return &HTTPService{name: name, addr: addr}
	case "rpc":
		return &RPCService{name: name, addr: addr}
	default:
		return &HTTPService{name: name, addr: addr}
	}
}

// ServiceRegistry 服务注册中心
type ServiceRegistry struct {
	mu       sync.RWMutex
	services map[string][]Service
	factory  ServiceFactory
}

func NewServiceRegistry(factory ServiceFactory) *ServiceRegistry {
	return &ServiceRegistry{
		services: make(map[string][]Service),
		factory:  factory,
	}
}

// Register 注册服务
func (r *ServiceRegistry) Register(name string, addr string) Service {
	r.mu.Lock()
	defer r.mu.Unlock()

	service := r.factory.CreateService(name, addr)
	r.services[name] = append(r.services[name], service)
	return service
}

// Deregister 注销服务
func (r *ServiceRegistry) Deregister(name string, addr string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	services := r.services[name]
	for i, service := range services {
		if service.Address() == addr {
			r.services[name] = append(services[:i], services[i+1:]...)
			break
		}
	}

	if len(r.services[name]) == 0 {
		delete(r.services, name)
	}
}

// Discover 发现服务
func (r *ServiceRegistry) Discover(name string) []Service {
	r.mu.RLock()
	defer r.mu.RUnlock()

	services := r.services[name]
	result := make([]Service, len(services))
	copy(result, services)
	return result
}

// ListServices 列出所有服务
func (r *ServiceRegistry) ListServices() map[string]int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string]int)
	for name, services := range r.services {
		result[name] = len(services)
	}
	return result
}
