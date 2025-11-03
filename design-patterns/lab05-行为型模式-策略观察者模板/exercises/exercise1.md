# 练习 1: 文件压缩策略系统

## 难度
⭐⭐ (中等)

## 学习目标
- 掌握策略模式的实现
- 理解如何封装算法族
- 学会在运行时动态切换策略
- 理解策略模式与条件语句的区别

## 问题描述

你需要实现一个文件压缩系统，支持多种压缩算法。系统应该能够根据用户的选择或文件特性自动选择合适的压缩策略。

## 功能要求

1. **压缩策略接口**
   - 定义统一的压缩接口
   - 包含压缩和解压缩方法
   - 提供获取压缩率和压缩速度的方法

2. **具体压缩策略**
   - ZIP 压缩：通用压缩，压缩率中等，速度快
   - GZIP 压缩：适合文本文件，压缩率高，速度中等
   - BZIP2 压缩：压缩率最高，速度较慢
   - 无压缩：直接复制，速度最快

3. **压缩上下文**
   - 维护当前使用的压缩策略
   - 提供设置和切换策略的方法
   - 执行压缩和解压缩操作

4. **智能策略选择**
   - 根据文件大小选择策略（小文件不压缩，大文件使用高压缩率）
   - 根据文件类型选择策略（文本文件使用 GZIP，二进制文件使用 ZIP）
   - 根据用户偏好选择策略（速度优先或压缩率优先）

## 输入输出示例

### 示例 1: 手动选择策略
**输入**:
```go
compressor := NewFileCompressor()
compressor.SetStrategy(NewZIPStrategy())
compressor.Compress("document.txt", 1024)
```

**输出**:
```
已选择压缩策略: ZIP
压缩文件: document.txt (1024 字节)
使用 ZIP 算法压缩...
压缩完成: document.txt.zip (512 字节)
压缩率: 50.0%
耗时: 100ms
```

### 示例 2: 自动选择策略
**输入**:
```go
compressor := NewFileCompressor()
strategy := SelectOptimalStrategy("document.txt", 10240, "speed")
compressor.SetStrategy(strategy)
compressor.Compress("document.txt", 10240)
```

**输出**:
```
文件分析:
  文件名: document.txt
  大小: 10240 字节
  类型: 文本文件
  用户偏好: 速度优先

推荐策略: GZIP (文本文件 + 速度优先)

已选择压缩策略: GZIP
压缩文件: document.txt (10240 字节)
使用 GZIP 算法压缩...
压缩完成: document.txt.gz (3072 字节)
压缩率: 70.0%
耗时: 150ms
```

## 提示

💡 **提示 1**: 定义清晰的策略接口
```go
type CompressionStrategy interface {
    Compress(filename string, size int) (compressedSize int, duration time.Duration)
    Decompress(filename string) error
    GetName() string
    GetCompressionRatio() float64
    GetSpeed() string
}
```

💡 **提示 2**: 使用工厂函数创建策略
```go
func SelectOptimalStrategy(filename string, size int, preference string) CompressionStrategy {
    // 根据文件特性和用户偏好选择策略
}
```

💡 **提示 3**: 策略可以包含状态
```go
type ZIPStrategy struct {
    compressionLevel int
    lastCompressedSize int
}
```

## 评分标准

- [ ] **功能完整性 (40%)**
  - 实现所有压缩策略
  - 支持策略切换
  - 实现智能策略选择

- [ ] **代码质量 (30%)**
  - 接口设计合理
  - 代码结构清晰
  - 命名规范

- [ ] **设计模式应用 (20%)**
  - 正确使用策略模式
  - 策略之间可互换
  - 符合开闭原则

- [ ] **扩展性 (10%)**
  - 易于添加新的压缩策略
  - 易于添加新的选择规则

## 扩展挑战

1. **压缩级别**: 为每种压缩策略添加压缩级别设置（1-9）
2. **批量压缩**: 支持批量压缩多个文件
3. **压缩统计**: 记录和显示压缩统计信息（总压缩率、总耗时等）
4. **加密压缩**: 添加加密压缩策略
5. **分块压缩**: 对大文件进行分块压缩

## 相关知识点

- 策略模式的结构和实现
- 接口的设计和使用
- 算法的封装和替换
- 开闭原则的应用
