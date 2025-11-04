package module1

import (
	"context"
	"errors"
	"io"
	"sync"
	"time"
)

// FlowControlManager 管理文件传输的流量控制
// 这是简化版的 HPFS flowcontrol.go 代码，用于学习练习
type FlowControlManager struct {
	MaxFlow    float64 // 单任务最大流量 (bytes/s)
	TotalFlow  float64 // 全局总流量 (bytes/s)
	BufferSize int     // 缓冲区大小

	// TODO: 标注问题 1 - 全局 channel 共享
	// 问题：这个 channel 被所有并发任务共享，但没有引用计数机制
	// 当第一个任务完成时可能关闭它，导致其他任务 panic
	totalFlowTokenChan chan int

	// TODO: 标注问题 2 - Context 取消时序
	// 问题：全局 context 的生命周期管理不当
	// 当 ctx.Done() 触发时，会立即关闭 totalFlowTokenChan
	ctx    context.Context
	cancel context.CancelFunc

	semaphore chan struct{} // 信号量，限制并发任务数
}

// NewFlowControlManager 创建新的流控管理器
func NewFlowControlManager(maxFlow, totalFlow float64, bufferSize int, maxConcurrent int) *FlowControlManager {
	return &FlowControlManager{
		MaxFlow:            maxFlow,
		TotalFlow:          totalFlow,
		BufferSize:         bufferSize,
		totalFlowTokenChan: make(chan int, 100),
		semaphore:          make(chan struct{}, maxConcurrent),
	}
}

// Start 启动流控管理器
func (f *FlowControlManager) Start() {
	ctx, cancel := context.WithCancel(context.Background())
	f.ctx = ctx
	f.cancel = cancel

	// 启动全局 token 分发 goroutine
	// TODO: 思考：这个 goroutine 什么时候会退出？
	f.dispatchToken(f.TotalFlow, f.totalFlowTokenChan, f.ctx)
}

// Stop 停止流控管理器
func (f *FlowControlManager) Stop() {
	if f.cancel != nil {
		f.cancel() // 触发 ctx.Done()
	}
}

// LimitFlow 对数据传输进行流量控制
// 这是 bug 的核心位置
func (f *FlowControlManager) LimitFlow(reader io.Reader, writer io.Writer) (int64, error) {
	// 获取信号量
	if !f.tryAcquire() {
		return 0, errors.New("throttled: too many concurrent tasks")
	}

	// 创建本地 context
	ctx, cancel := context.WithCancel(context.Background())
	tokensChan := make(chan int, 2)

	// TODO: 标注问题 3 - defer close 的危险
	// 问题：defer 会在函数返回时执行，但此时可能还有其他 goroutine 在使用资源
	defer func() {
		if cancel != nil {
			cancel() // 取消本地 context
		}
		f.release() // 释放信号量
	}()

	// 启动本地 token 分发
	f.dispatchToken(f.MaxFlow, tokensChan, ctx)

	var totalWritten int64
	var err error

	// 主传输循环
	// TODO: 分析这个循环的执行流程，特别是嵌套 select 的行为
out:
	for {
		select {
		case _, ok := <-tokensChan: // 读取本地 token
			if !ok {
				break out
			}

			// TODO: 关键问题点！嵌套 select
			// 问题：当 totalFlowTokenChan 关闭时，这里的 break 只跳出内层 select
			// 代码会继续执行到下面的 Read() 操作，而不是跳出外层循环
			select {
			case _, ok := <-f.totalFlowTokenChan: // 读取全局 token
				if !ok {
					// Channel 已关闭
					// TODO: 这个 break 跳到哪里？是内层 select 还是外层 for？
					break
				}
			case <-f.ctx.Done(): // 全局 context 取消
				err = errors.New("global task stopped")
				break
			case <-ctx.Done(): // 本地 context 取消
				err = errors.New("local task stopped")
				break
			}

		case <-f.ctx.Done():
			err = errors.New("global task stopped")
			break out
		case <-ctx.Done():
			err = errors.New("local task stopped")
			break out
		}

		// 读取数据
		buf := make([]byte, f.BufferSize)
		n, readErr := reader.Read(buf)
		if n > 0 {
			written, writeErr := writer.Write(buf[:n])
			totalWritten += int64(written)
			if writeErr != nil {
				return totalWritten, writeErr
			}
		}

		if readErr == io.EOF {
			break out
		}
		if readErr != nil {
			return totalWritten, readErr
		}
	}

	return totalWritten, err
}

// dispatchToken 定期向 channel 发送 token
func (f *FlowControlManager) dispatchToken(flowRate float64, tokenChan chan int, ctx context.Context) {
	go func(ctx context.Context) {
		// TODO: 标注问题 4 - 无条件关闭 channel
		// 问题：当 ctx.Done() 触发时，立即关闭 channel
		// 没有检查是否还有其他 goroutine 正在使用这个 channel
		defer func() {
			close(tokenChan) // 关闭 channel
		}()

		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				// Context 取消，退出 goroutine
				// TODO: 思考：退出时会发生什么？defer close() 会执行
				return
			case <-ticker.C:
				// 计算应该发送的 token 数量
				tokens := int(flowRate * 0.1) // 100ms 的流量
				for i := 0; i < tokens && i < 10; i++ {
					select {
					case <-ctx.Done():
						return
					case tokenChan <- 1:
						// 发送 token
					default:
						// Channel 满了，跳过
					}
				}
			}
		}
	}(ctx)
}

// tryAcquire 尝试获取信号量
func (f *FlowControlManager) tryAcquire() bool {
	select {
	case f.semaphore <- struct{}{}:
		return true
	default:
		return false
	}
}

// release 释放信号量
func (f *FlowControlManager) release() {
	select {
	case <-f.semaphore:
	default:
	}
}

// ============================================
// 练习任务
// ============================================

/*
任务 1: 标注所有问题点
在上面的代码中找到并标注所有的问题点（已经用 TODO 标记了位置）

任务 2: 回答以下问题
1. 为什么 totalFlowTokenChan 是全局共享的？这样设计的目的是什么？
2. 当第一个任务完成时，会发生什么？
3. 为什么嵌套 select 中的 break 无法阻止 panic？
4. 如果有 2 个并发任务，它们的执行时序是怎样的？

任务 3: 绘制数据流图
绘制一个图表，展示：
- 2 个并发任务（Task A 和 Task B）
- 全局 totalFlowTokenChan
- 本地 tokensChan
- Context 的关系

任务 4: 预测 Bug 触发
假设有以下场景：
- Task A 在 T0 时刻开始
- Task B 在 T1 时刻开始（1秒后）
- Task A 在 T5 时刻完成
- Task B 在 T10 时刻完成

请预测：
1. 在哪个时间点会触发 panic？
2. 为什么？
3. 哪个 goroutine 会 panic？

提示：查看 review/HPFS_BUG_FLOW_DIAGRAM.md 了解详细的时序分析
*/
