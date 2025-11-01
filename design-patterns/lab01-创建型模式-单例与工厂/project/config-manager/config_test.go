package main

import (
	"sync"
	"testing"
)

func TestGetConfig(t *testing.T) {
	config1 := GetConfig()
	config2 := GetConfig()

	if config1 != config2 {
		t.Error("Expected same instance")
	}
}

func TestSetAndGet(t *testing.T) {
	config := GetConfig()
	config.Clear()

	config.Set("test.key", "test value")

	value, ok := config.Get("test.key")
	if !ok {
		t.Error("Expected key to exist")
	}

	if value != "test value" {
		t.Errorf("Expected 'test value', got '%v'", value)
	}
}

func TestGetString(t *testing.T) {
	config := GetConfig()
	config.Clear()

	config.Set("string.key", "hello")

	value := config.GetString("string.key")
	if value != "hello" {
		t.Errorf("Expected 'hello', got '%s'", value)
	}
}

func TestGetInt(t *testing.T) {
	config := GetConfig()
	config.Clear()

	config.Set("int.key", 42)

	value := config.GetInt("int.key")
	if value != 42 {
		t.Errorf("Expected 42, got %d", value)
	}
}

func TestGetBool(t *testing.T) {
	config := GetConfig()
	config.Clear()

	config.Set("bool.key", true)

	value := config.GetBool("bool.key")
	if !value {
		t.Error("Expected true")
	}
}

func TestConcurrentAccess(t *testing.T) {
	config := GetConfig()
	config.Clear()

	var wg sync.WaitGroup
	iterations := 100

	// Concurrent writes
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			config.Set("concurrent.key", index)
		}(i)
	}

	// Concurrent reads
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			config.Get("concurrent.key")
		}()
	}

	wg.Wait()
}

func TestSize(t *testing.T) {
	config := GetConfig()
	config.Clear()

	config.Set("key1", "value1")
	config.Set("key2", "value2")
	config.Set("key3", "value3")

	size := config.Size()
	if size != 3 {
		t.Errorf("Expected size 3, got %d", size)
	}
}

func TestClear(t *testing.T) {
	config := GetConfig()

	config.Set("key1", "value1")
	config.Set("key2", "value2")

	config.Clear()

	size := config.Size()
	if size != 0 {
		t.Errorf("Expected size 0 after clear, got %d", size)
	}
}
