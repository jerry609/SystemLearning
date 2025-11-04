package main

import (
	"container/list"
	"sync"
	"time"
)

// 淘汰策略 - 策略模式
//
// 本模块使用策略模式实现多种缓存淘汰算法
// 支持 LRU、LFU、FIFO 等策略

// EvictionStrategy 淘汰策略接口
type EvictionStrategy interface {
	Evict(cache map[string]*CacheItem) string
	OnAccess(key string)
	OnSet(key string)
}

// LRUStrategy LRU（最近最少使用）策略
type LRUStrategy struct {
	mu    sync.Mutex
	list  *list.List
	items map[string]*list.Element
}

func NewLRUStrategy() *LRUStrategy {
	return &LRUStrategy{
		list:  list.New(),
		items: make(map[string]*list.Element),
	}
}

func (s *LRUStrategy) Evict(cache map[string]*CacheItem) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.list.Len() == 0 {
		return ""
	}

	// 淘汰最久未使用的项（链表尾部）
	elem := s.list.Back()
	if elem != nil {
		key := elem.Value.(string)
		s.list.Remove(elem)
		delete(s.items, key)
		return key
	}

	return ""
}

func (s *LRUStrategy) OnAccess(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 移动到链表头部
	if elem, ok := s.items[key]; ok {
		s.list.MoveToFront(elem)
	}
}

func (s *LRUStrategy) OnSet(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 添加到链表头部
	if elem, ok := s.items[key]; ok {
		s.list.MoveToFront(elem)
	} else {
		elem := s.list.PushFront(key)
		s.items[key] = elem
	}
}

// LFUStrategy LFU（最不经常使用）策略
type LFUStrategy struct {
	mu         sync.Mutex
	frequencies map[string]int
}

func NewLFUStrategy() *LFUStrategy {
	return &LFUStrategy{
		frequencies: make(map[string]int),
	}
}

func (s *LFUStrategy) Evict(cache map[string]*CacheItem) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.frequencies) == 0 {
		return ""
	}

	// 找到使用频率最低的项
	var minKey string
	minFreq := -1

	for key, freq := range s.frequencies {
		if minFreq == -1 || freq < minFreq {
			minFreq = freq
			minKey = key
		}
	}

	if minKey != "" {
		delete(s.frequencies, minKey)
	}

	return minKey
}

func (s *LFUStrategy) OnAccess(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.frequencies[key]++
}

func (s *LFUStrategy) OnSet(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.frequencies[key]; !ok {
		s.frequencies[key] = 0
	}
}

// FIFOStrategy FIFO（先进先出）策略
type FIFOStrategy struct {
	mu    sync.Mutex
	queue []string
}

func NewFIFOStrategy() *FIFOStrategy {
	return &FIFOStrategy{
		queue: make([]string, 0),
	}
}

func (s *FIFOStrategy) Evict(cache map[string]*CacheItem) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.queue) == 0 {
		return ""
	}

	// 淘汰最先进入的项（队列头部）
	key := s.queue[0]
	s.queue = s.queue[1:]
	return key
}

func (s *FIFOStrategy) OnAccess(key string) {
	// FIFO 不关心访问
}

func (s *FIFOStrategy) OnSet(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 检查是否已存在
	for _, k := range s.queue {
		if k == key {
			return
		}
	}

	// 添加到队列尾部
	s.queue = append(s.queue, key)
}

// TTLStrategy TTL（基于时间）策略
type TTLStrategy struct {
	mu sync.Mutex
}

func NewTTLStrategy() *TTLStrategy {
	return &TTLStrategy{}
}

func (s *TTLStrategy) Evict(cache map[string]*CacheItem) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 找到最早过期的项
	var oldestKey string
	var oldestTime time.Time

	for key, item := range cache {
		if !item.ExpireTime.IsZero() {
			if oldestTime.IsZero() || item.ExpireTime.Before(oldestTime) {
				oldestTime = item.ExpireTime
				oldestKey = key
			}
		}
	}

	return oldestKey
}

func (s *TTLStrategy) OnAccess(key string) {
	// TTL 不关心访问
}

func (s *TTLStrategy) OnSet(key string) {
	// TTL 不需要特殊处理
}

// RandomStrategy 随机淘汰策略
type RandomStrategy struct {
	mu sync.Mutex
}

func NewRandomStrategy() *RandomStrategy {
	return &RandomStrategy{}
}

func (s *RandomStrategy) Evict(cache map[string]*CacheItem) string {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 随机选择一个键
	for key := range cache {
		return key
	}

	return ""
}

func (s *RandomStrategy) OnAccess(key string) {
	// 随机策略不关心访问
}

func (s *RandomStrategy) OnSet(key string) {
	// 随机策略不需要特殊处理
}

// StrategyFactory 策略工厂
type StrategyFactory struct{}

func (f *StrategyFactory) CreateStrategy(strategyType string) EvictionStrategy {
	switch strategyType {
	case "lru":
		return NewLRUStrategy()
	case "lfu":
		return NewLFUStrategy()
	case "fifo":
		return NewFIFOStrategy()
	case "ttl":
		return NewTTLStrategy()
	case "random":
		return NewRandomStrategy()
	default:
		return NewLRUStrategy()
	}
}
