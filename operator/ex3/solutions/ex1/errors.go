package main

import (
	"fmt"
	"time"
)

// RetryableError 可重试错误（临时性错误）
type RetryableError struct {
	Reason string        // 错误原因
	After  time.Duration // 建议重试间隔
}

func (e *RetryableError) Error() string {
	return fmt.Sprintf("retryable error: %s (retry after %v)", e.Reason, e.After)
}

// NewRetryableError 创建可重试错误
func NewRetryableError(reason string, after time.Duration) *RetryableError {
	return &RetryableError{
		Reason: reason,
		After:  after,
	}
}

// PermanentError 永久性错误
type PermanentError struct {
	Reason string // 错误原因
}

func (e *PermanentError) Error() string {
	return fmt.Sprintf("permanent error: %s", e.Reason)
}

// NewPermanentError 创建永久性错误
func NewPermanentError(reason string) *PermanentError {
	return &PermanentError{
		Reason: reason,
	}
}

// IsRetryableError 判断是否为可重试错误
func IsRetryableError(err error) bool {
	_, ok := err.(*RetryableError)
	return ok
}

// IsPermanentError 判断是否为永久性错误
func IsPermanentError(err error) bool {
	_, ok := err.(*PermanentError)
	return ok
}
