package main

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"
)

// 练习 1: 文件压缩策略系统 - 参考答案
//
// 设计思路:
// 1. 定义 CompressionStrategy 接口，封装压缩算法
// 2. 实现多种具体压缩策略（ZIP、GZIP、BZIP2、无压缩）
// 3. 创建 FileCompressor 上下文，管理策略切换
// 4. 实现智能策略选择函数，根据文件特性自动选择最优策略
//
// 使用的设计模式: 策略模式
// 模式应用位置: CompressionStrategy 接口和具体策略类

// 压缩策略接口
type CompressionStrategy interface {
	Compress(filename string, size int) (compressedSize int, duration time.Duration)
	Decompress(filename string) error
	GetName() string
	GetCompressionRatio() float64
	GetSpeed() string
}

// ZIP 压缩策略
type ZIPStrategy struct {
	compressionLevel int
}

func NewZIPStrategy() *ZIPStrategy {
	return &ZIPStrategy{compressionLevel: 5}
}

func (z *ZIPStrategy) Compress(filename string, size int) (int, time.Duration) {
	start := time.Now()
	fmt.Printf("使用 ZIP 算法压缩...\n")
	time.Sleep(100 * time.Millisecond)
	
	compressedSize := int(float64(size) * 0.5) // 50% 压缩率
	duration := time.Since(start)
	
	fmt.Printf("压缩完成: %s.zip (%d 字节)\n", filename, compressedSize)
	fmt.Printf("压缩率: %.1f%%\n", (1-float64(compressedSize)/float64(size))*100)
	fmt.Printf("耗时: %v\n", duration)
	
	return compressedSize, duration
}

func (z *ZIPStrategy) Decompress(filename string) error {
	fmt.Printf("解压缩 %s...\n", filename)
	return nil
}

func (z *ZIPStrategy) GetName() string {
	return "ZIP"
}

func (z *ZIPStrategy) GetCompressionRatio() float64 {
	return 0.5
}

func (z *ZIPStrategy) GetSpeed() string {
	return "快"
}

// GZIP 压缩策略
type GZIPStrategy struct{}

func NewGZIPStrategy() *GZIPStrategy {
	return &GZIPStrategy{}
}

func (g *GZIPStrategy) Compress(filename string, size int) (int, time.Duration) {
	start := time.Now()
	fmt.Printf("使用 GZIP 算法压缩...\n")
	time.Sleep(150 * time.Millisecond)
	
	compressedSize := int(float64(size) * 0.3) // 70% 压缩率
	duration := time.Since(start)
	
	fmt.Printf("压缩完成: %s.gz (%d 字节)\n", filename, compressedSize)
	fmt.Printf("压缩率: %.1f%%\n", (1-float64(compressedSize)/float64(size))*100)
	fmt.Printf("耗时: %v\n", duration)
	
	return compressedSize, duration
}

func (g *GZIPStrategy) Decompress(filename string) error {
	fmt.Printf("解压缩 %s...\n", filename)
	return nil
}

func (g *GZIPStrategy) GetName() string {
	return "GZIP"
}

func (g *GZIPStrategy) GetCompressionRatio() float64 {
	return 0.3
}

func (g *GZIPStrategy) GetSpeed() string {
	return "中等"
}

// BZIP2 压缩策略
type BZIP2Strategy struct{}

func NewBZIP2Strategy() *BZIP2Strategy {
	return &BZIP2Strategy{}
}

func (b *BZIP2Strategy) Compress(filename string, size int) (int, time.Duration) {
	start := time.Now()
	fmt.Printf("使用 BZIP2 算法压缩...\n")
	time.Sleep(200 * time.Millisecond)
	
	compressedSize := int(float64(size) * 0.2) // 80% 压缩率
	duration := time.Since(start)
	
	fmt.Printf("压缩完成: %s.bz2 (%d 字节)\n", filename, compressedSize)
	fmt.Printf("压缩率: %.1f%%\n", (1-float64(compressedSize)/float64(size))*100)
	fmt.Printf("耗时: %v\n", duration)
	
	return compressedSize, duration
}

func (b *BZIP2Strategy) Decompress(filename string) error {
	fmt.Printf("解压缩 %s...\n", filename)
	return nil
}

func (b *BZIP2Strategy) GetName() string {
	return "BZIP2"
}

func (b *BZIP2Strategy) GetCompressionRatio() float64 {
	return 0.2
}

func (b *BZIP2Strategy) GetSpeed() string {
	return "慢"
}

// 无压缩策略
type NoCompressionStrategy struct{}

func NewNoCompressionStrategy() *NoCompressionStrategy {
	return &NoCompressionStrategy{}
}

func (n *NoCompressionStrategy) Compress(filename string, size int) (int, time.Duration) {
	start := time.Now()
	fmt.Printf("直接复制文件（无压缩）...\n")
	time.Sleep(50 * time.Millisecond)
	
	duration := time.Since(start)
	
	fmt.Printf("复制完成: %s.copy (%d 字节)\n", filename, size)
	fmt.Printf("压缩率: 0.0%%\n")
	fmt.Printf("耗时: %v\n", duration)
	
	return size, duration
}

func (n *NoCompressionStrategy) Decompress(filename string) error {
	fmt.Printf("复制 %s...\n", filename)
	return nil
}

func (n *NoCompressionStrategy) GetName() string {
	return "无压缩"
}

func (n *NoCompressionStrategy) GetCompressionRatio() float64 {
	return 1.0
}

func (n *NoCompressionStrategy) GetSpeed() string {
	return "最快"
}

// 文件压缩器（上下文）
type FileCompressor struct {
	strategy CompressionStrategy
}

func NewFileCompressor() *FileCompressor {
	return &FileCompressor{}
}

func (f *FileCompressor) SetStrategy(strategy CompressionStrategy) {
	f.strategy = strategy
	fmt.Printf("\n已选择压缩策略: %s\n", strategy.GetName())
	fmt.Printf("压缩率: %.0f%%, 速度: %s\n", (1-strategy.GetCompressionRatio())*100, strategy.GetSpeed())
}

func (f *FileCompressor) Compress(filename string, size int) (int, time.Duration) {
	if f.strategy == nil {
		fmt.Println("错误: 未设置压缩策略")
		return 0, 0
	}
	
	fmt.Printf("\n压缩文件: %s (%d 字节)\n", filename, size)
	return f.strategy.Compress(filename, size)
}

// 智能策略选择
func SelectOptimalStrategy(filename string, size int, preference string) CompressionStrategy {
	fmt.Println("\n文件分析:")
	fmt.Printf("  文件名: %s\n", filename)
	fmt.Printf("  大小: %d 字节\n", size)
	
	ext := strings.ToLower(filepath.Ext(filename))
	isText := ext == ".txt" || ext == ".log" || ext == ".csv" || ext == ".json" || ext == ".xml"
	
	if isText {
		fmt.Println("  类型: 文本文件")
	} else {
		fmt.Println("  类型: 二进制文件")
	}
	fmt.Printf("  用户偏好: %s\n", preference)
	
	// 小文件不压缩
	if size < 1024 {
		fmt.Println("\n推荐策略: 无压缩 (文件太小)")
		return NewNoCompressionStrategy()
	}
	
	// 根据用户偏好选择
	if preference == "speed" {
		if isText {
			fmt.Println("\n推荐策略: GZIP (文本文件 + 速度优先)")
			return NewGZIPStrategy()
		}
		fmt.Println("\n推荐策略: ZIP (速度优先)")
		return NewZIPStrategy()
	}
	
	if preference == "ratio" {
		fmt.Println("\n推荐策略: BZIP2 (压缩率优先)")
		return NewBZIP2Strategy()
	}
	
	// 默认策略
	if isText {
		fmt.Println("\n推荐策略: GZIP (文本文件)")
		return NewGZIPStrategy()
	}
	
	fmt.Println("\n推荐策略: ZIP (通用)")
	return NewZIPStrategy()
}

func main() {
	fmt.Println("=== 文件压缩策略系统 ===")
	
	compressor := NewFileCompressor()
	
	// 场景 1: 手动选择 ZIP 策略
	fmt.Println("\n【场景 1: 手动选择 ZIP 策略】")
	compressor.SetStrategy(NewZIPStrategy())
	compressor.Compress("document.txt", 1024)
	
	// 场景 2: 手动选择 GZIP 策略
	fmt.Println("\n\n【场景 2: 手动选择 GZIP 策略】")
	compressor.SetStrategy(NewGZIPStrategy())
	compressor.Compress("data.log", 10240)
	
	// 场景 3: 自动选择策略（文本文件 + 速度优先）
	fmt.Println("\n\n【场景 3: 自动选择策略（文本文件 + 速度优先）】")
	strategy := SelectOptimalStrategy("document.txt", 10240, "speed")
	compressor.SetStrategy(strategy)
	compressor.Compress("document.txt", 10240)
	
	// 场景 4: 自动选择策略（大文件 + 压缩率优先）
	fmt.Println("\n\n【场景 4: 自动选择策略（大文件 + 压缩率优先）】")
	strategy = SelectOptimalStrategy("archive.bin", 102400, "ratio")
	compressor.SetStrategy(strategy)
	compressor.Compress("archive.bin", 102400)
	
	// 场景 5: 小文件自动选择无压缩
	fmt.Println("\n\n【场景 5: 小文件自动选择无压缩】")
	strategy = SelectOptimalStrategy("small.txt", 512, "speed")
	compressor.SetStrategy(strategy)
	compressor.Compress("small.txt", 512)
	
	fmt.Println("\n=== 示例结束 ===")
}

// 可能的优化方向:
// 1. 添加压缩级别设置（1-9）
// 2. 支持批量压缩多个文件
// 3. 添加压缩统计信息（总压缩率、总耗时）
// 4. 实现加密压缩策略
// 5. 支持分块压缩大文件
//
// 变体实现:
// 1. 使用函数类型代替接口
// 2. 使用工厂模式创建策略
// 3. 添加策略缓存，避免重复创建
