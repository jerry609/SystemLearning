package main

import (
"fmt"
"sync"
"time"
)

// DataRepository interface
type DataRepository interface {
Get(key string) (string, error)
Set(key string, value string) error
Delete(key string) error
}

// DatabaseRepository implementation
type DatabaseRepository struct {
data map[string]string
mu   sync.RWMutex
}

func NewDatabaseRepository() *DatabaseRepository {
return &DatabaseRepository{
data: make(map[string]string),
}
}

func (r *DatabaseRepository) Get(key string) (string, error) {
time.Sleep(100 * time.Millisecond)
r.mu.RLock()
defer r.mu.RUnlock()
value, exists := r.data[key]
if !exists {
return "", fmt.Errorf("key not found: %s", key)
}
fmt.Printf("[DB] Get: %s = %s (100ms)\n", key, value)
return value, nil
}

func (r *DatabaseRepository) Set(key string, value string) error {
time.Sleep(150 * time.Millisecond)
r.mu.Lock()
defer r.mu.Unlock()
r.data[key] = value
fmt.Printf("[DB] Set: %s = %s (150ms)\n", key, value)
return nil
}

func (r *DatabaseRepository) Delete(key string) error {
time.Sleep(120 * time.Millisecond)
r.mu.Lock()
defer r.mu.Unlock()
delete(r.data, key)
fmt.Printf("[DB] Delete: %s (120ms)\n", key)
return nil
}

type CacheEntry struct {
value     string
expiresAt time.Time
}

func (e *CacheEntry) IsExpired() bool {
return time.Now().After(e.expiresAt)
}

type CacheDecorator struct {
repository DataRepository
cache      map[string]*CacheEntry
ttl        time.Duration
mu         sync.RWMutex
hits       int
misses     int
}

func NewCacheDecorator(repository DataRepository, ttl time.Duration) *CacheDecorator {
return &CacheDecorator{
repository: repository,
cache:      make(map[string]*CacheEntry),
ttl:        ttl,
}
}

func (d *CacheDecorator) Get(key string) (string, error) {
d.mu.RLock()
entry, exists := d.cache[key]
d.mu.RUnlock()
if exists && !entry.IsExpired() {
d.hits++
fmt.Printf("[CACHE] Hit: %s = %s (0ms)\n", key, entry.value)
return entry.value, nil
}
d.misses++
fmt.Printf("[CACHE] Miss: %s\n", key)
value, err := d.repository.Get(key)
if err != nil {
return "", err
}
d.mu.Lock()
d.cache[key] = &CacheEntry{
value:     value,
expiresAt: time.Now().Add(d.ttl),
}
d.mu.Unlock()
return value, nil
}

func (d *CacheDecorator) Set(key string, value string) error {
err := d.repository.Set(key, value)
if err != nil {
return err
}
d.mu.Lock()
d.cache[key] = &CacheEntry{
value:     value,
expiresAt: time.Now().Add(d.ttl),
}
d.mu.Unlock()
fmt.Printf("[CACHE] Updated cache for: %s\n", key)
return nil
}

func (d *CacheDecorator) Delete(key string) error {
err := d.repository.Delete(key)
if err != nil {
return err
}
d.mu.Lock()
delete(d.cache, key)
d.mu.Unlock()
fmt.Printf("[CACHE] Removed from cache: %s\n", key)
return nil
}

func (d *CacheDecorator) GetStats() (hits, misses int, hitRate float64) {
total := d.hits + d.misses
if total == 0 {
return d.hits, d.misses, 0
}
return d.hits, d.misses, float64(d.hits) / float64(total) * 100
}

func main() {
fmt.Println("=== Cache Decorator Example ===\n")
fmt.Println("--- Example 1: Basic Cache Decorator (TTL: 5s) ---")
db1 := NewDatabaseRepository()
cache1 := NewCacheDecorator(db1, 5*time.Second)
cache1.Set("user:1", "Alice")
cache1.Set("user:2", "Bob")
fmt.Println()
fmt.Println("First read:")
cache1.Get("user:1")
cache1.Get("user:2")
fmt.Println()
fmt.Println("Second read (cache hit):")
cache1.Get("user:1")
cache1.Get("user:2")
fmt.Println()
hits, misses, hitRate := cache1.GetStats()
fmt.Printf("Cache stats: hits=%d, misses=%d, hit rate=%.2f%%\n\n", hits, misses, hitRate)
fmt.Println("=== Example Complete ===")
fmt.Println("\nKey Points:")
fmt.Println("1. Cache decorator transparently adds caching functionality")
fmt.Println("2. Supports different caching strategies (TTL, LRU, etc.)")
fmt.Println("3. Can combine multiple cache decorators for multi-level caching")
fmt.Println("4. Cache decorator significantly improves read performance")
}
