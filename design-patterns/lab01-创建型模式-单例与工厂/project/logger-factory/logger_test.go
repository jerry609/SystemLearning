package main

import (
	"os"
	"testing"
)

func TestCreateConsoleLogger(t *testing.T) {
	logger, err := CreateLogger("console", &LoggerConfig{
		Level: INFO,
	})

	if err != nil {
		t.Fatalf("Failed to create console logger: %v", err)
	}

	if logger == nil {
		t.Fatal("Expected logger to be created")
	}
}

func TestCreateFileLogger(t *testing.T) {
	filename := "test.log"
	defer os.Remove(filename)

	logger, err := CreateLogger("file", &LoggerConfig{
		Level:    DEBUG,
		Filename: filename,
	})

	if err != nil {
		t.Fatalf("Failed to create file logger: %v", err)
	}

	if logger == nil {
		t.Fatal("Expected logger to be created")
	}

	// Clean up
	if fileLogger, ok := logger.(*FileLogger); ok {
		fileLogger.Close()
	}
}

func TestLoggerLevels(t *testing.T) {
	logger, _ := CreateLogger("console", &LoggerConfig{
		Level: INFO,
	})

	// These should not panic
	logger.Info("Info message")
	logger.Warn("Warn message")
	logger.Error("Error message")
}

func TestSetLevel(t *testing.T) {
	logger, _ := CreateLogger("console", &LoggerConfig{
		Level: ERROR,
	})

	logger.SetLevel(DEBUG)

	// Should not panic
	logger.Debug("Debug message")
}

func TestConsoleLoggerFactory(t *testing.T) {
	factory := &ConsoleLoggerFactory{}
	logger, err := factory.CreateLogger(&LoggerConfig{
		Level: INFO,
	})

	if err != nil {
		t.Fatalf("Failed to create logger: %v", err)
	}

	if logger == nil {
		t.Fatal("Expected logger to be created")
	}
}

func TestFileLoggerFactory(t *testing.T) {
	filename := "test_factory.log"
	defer os.Remove(filename)

	factory := &FileLoggerFactory{}
	logger, err := factory.CreateLogger(&LoggerConfig{
		Level:    DEBUG,
		Filename: filename,
	})

	if err != nil {
		t.Fatalf("Failed to create logger: %v", err)
	}

	if logger == nil {
		t.Fatal("Expected logger to be created")
	}

	// Clean up
	if fileLogger, ok := logger.(*FileLogger); ok {
		fileLogger.Close()
	}
}
