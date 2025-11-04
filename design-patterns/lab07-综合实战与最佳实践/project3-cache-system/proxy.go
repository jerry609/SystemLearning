package main

import (
	"fmt"
	"time"
)

// 缓存代理 - 代理模式
//
// 本模块使用代理模式为数据访问添加缓存层
// 提供透明的缓存访问

// DataSource 数据源接口
type DataSource interface {
	Get(key string) (interface{}, error)
	Set(key string, value interface{}) error
	Delete(key string) error
}

// CacheProxy 缓存代理
type CacheProxy struct {
	dataSource DataSource
	cache      *CacheManager
	ttl        time.Duration
}

func NewCacheProxy(dataSource DataSource, cache *CacheManager, ttl time.Duration) *CacheProxy {
	return &CacheProxy{
		dataSource: dataSource,
		cache:      cache,
		ttl:        ttl,
	}
}

// Get 获取数据（带缓存）
func (p *CacheProxy) Get(key string) (interface{}, error) {
	// 1. 先查缓存
	if value, ok := p.cache.Get(key); ok {
		fmt.Printf("[缓存代理] 缓存命中: %s\n", key)
		return value, nil
	}

	fmt.Printf("[缓存代理] 缓存未命中: %s，从数据源加载\n", key)

	// 2. 缓存未命中，从数据源获取
	value, err := p.dataSource.Get(key)
	if err != nil {
		return nil, err
	}

	// 3. 写入缓存
	p.cache.Set(key, value, p.ttl)

	return value, nil
}

// Set 设置数据（写穿透）
func (p *CacheProxy) Set(key string, value interface{}) error {
	// 1. 写入数据源
	if err := p.dataSource.Set(key, value); err != nil {
		return err
	}

	// 2. 更新缓存
	p.cache.Set(key, value, p.ttl)

	return nil
}

// Delete 删除数据
func (p *CacheProxy) Delete(key string) error {
	// 1. 从数据源删除
	if err := p.dataSource.Delete(key); err != nil {
		return err
	}

	// 2. 从缓存删除
	p.cache.Delete(key)

	return nil
}

// MemoryDataSource 内存数据源（模拟数据库）
type MemoryDataSource struct {
	data map[string]interface{}
}

func NewMemoryDataSource() *MemoryDataSource {
	return &MemoryDataSource{
		data: make(map[string]interface{}),
	}
}

func (ds *MemoryDataSource) Get(key string) (interface{}, error) {
	value, ok := ds.data[key]
	if !ok {
		return nil, fmt.Errorf("数据未找到: %s", key)
	}
	return value, nil
}

func (ds *MemoryDataSource) Set(key string, value interface{}) error {
	ds.data[key] = value
	return nil
}

func (ds *MemoryDataSource) Delete(key string) error {
	delete(ds.data, key)
	return nil
}

// LazyLoadProxy 延迟加载代理
type LazyLoadProxy struct {
	dataSource DataSource
	cache      *CacheManager
	loader     func(key string) (interface{}, error)
}

func NewLazyLoadProxy(dataSource DataSource, cache *CacheManager, loader func(key string) (interface{}, error)) *LazyLoadProxy {
	return &LazyLoadProxy{
		dataSource: dataSource,
		cache:      cache,
		loader:     loader,
	}
}

// Get 延迟加载数据
func (p *LazyLoadProxy) Get(key string) (interface{}, error) {
	// 1. 先查缓存
	if value, ok := p.cache.Get(key); ok {
		return value, nil
	}

	// 2. 使用加载器加载数据
	value, err := p.loader(key)
	if err != nil {
		return nil, err
	}

	// 3. 写入缓存
	p.cache.Set(key, value, 0)

	return value, nil
}

// WriteBackProxy 写回代理
type WriteBackProxy struct {
	dataSource DataSource
	cache      *CacheManager
	dirty      map[string]bool
}

func NewWriteBackProxy(dataSource DataSource, cache *CacheManager) *WriteBackProxy {
	return &WriteBackProxy{
		dataSource: dataSource,
		cache:      cache,
		dirty:      make(map[string]bool),
	}
}

// Get 获取数据
func (p *WriteBackProxy) Get(key string) (interface{}, error) {
	// 先查缓存
	if value, ok := p.cache.Get(key); ok {
		return value, nil
	}

	// 从数据源加载
	value, err := p.dataSource.Get(key)
	if err != nil {
		return nil, err
	}

	p.cache.Set(key, value, 0)
	return value, nil
}

// Set 设置数据（只写缓存）
func (p *WriteBackProxy) Set(key string, value interface{}) error {
	// 只写缓存，标记为脏数据
	p.cache.Set(key, value, 0)
	p.dirty[key] = true
	return nil
}

// Flush 刷新脏数据到数据源
func (p *WriteBackProxy) Flush() error {
	for key := range p.dirty {
		value, ok := p.cache.Get(key)
		if !ok {
			continue
		}

		if err := p.dataSource.Set(key, value); err != nil {
			return err
		}

		delete(p.dirty, key)
	}
	return nil
}

// ReadThroughProxy 读穿透代理
type ReadThroughProxy struct {
	dataSource DataSource
	cache      *CacheManager
	ttl        time.Duration
}

func NewReadThroughProxy(dataSource DataSource, cache *CacheManager, ttl time.Duration) *ReadThroughProxy {
	return &ReadThroughProxy{
		dataSource: dataSource,
		cache:      cache,
		ttl:        ttl,
	}
}

// Get 读穿透获取数据
func (p *ReadThroughProxy) Get(key string) (interface{}, error) {
	// 先查缓存
	if value, ok := p.cache.Get(key); ok {
		return value, nil
	}

	// 从数据源加载
	value, err := p.dataSource.Get(key)
	if err != nil {
		return nil, err
	}

	// 写入缓存
	p.cache.Set(key, value, p.ttl)

	return value, nil
}
