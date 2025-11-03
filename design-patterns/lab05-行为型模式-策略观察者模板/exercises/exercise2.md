# 练习 2: 股票价格监控系统

## 难度
⭐⭐ (中等)

## 学习目标
- 掌握观察者模式的实现
- 理解一对多的依赖关系
- 学会实现主题和观察者接口
- 理解推模型和拉模型的区别

## 问题描述

你需要实现一个股票价格监控系统。当股票价格发生变化时，系统应该自动通知所有订阅者（邮件通知、短信通知、App 推送等）。系统还应该支持设置价格阈值，只有当价格变化超过阈值时才通知。

## 功能要求

1. **股票主题（Subject）**
   - 维护股票信息（代码、名称、当前价格）
   - 维护观察者列表
   - 提供注册和注销观察者的方法
   - 价格变化时通知所有观察者

2. **观察者接口（Observer）**
   - 定义更新方法
   - 接收股票价格变化通知
   - 获取观察者标识

3. **具体观察者**
   - 邮件通知观察者：发送邮件通知
   - 短信通知观察者：发送短信通知
   - App 推送观察者：推送到移动设备
   - 日志观察者：记录价格变化日志

4. **高级功能**
   - 价格阈值：只有价格变化超过阈值才通知
   - 条件通知：价格上涨/下跌到特定值时通知
   - 通知频率限制：避免频繁通知

## 输入输出示例

### 示例 1: 基本价格监控
**输入**:
```go
stock := NewStock("AAPL", "Apple Inc.", 150.00)

emailObserver := NewEmailObserver("user@example.com")
smsObserver := NewSMSObserver("138****8888")

stock.Attach(emailObserver)
stock.Attach(smsObserver)

stock.SetPrice(155.00)
stock.SetPrice(160.00)
```

**输出**:
```
✓ [邮件通知-user@example.com] 已订阅股票: AAPL

✓ [短信通知-138****8888] 已订阅股票: AAPL

📈 股票价格变化:
  股票: AAPL (Apple Inc.)
  原价格: $150.00
  新价格: $155.00
  涨跌: +$5.00 (+3.33%)

  📧 [邮件通知] 发送到 user@example.com
     AAPL 价格变化: $150.00 -> $155.00 (+3.33%)

  📱 [短信通知] 发送到 138****8888
     AAPL 价格变化: $150.00 -> $155.00 (+3.33%)

📈 股票价格变化:
  股票: AAPL (Apple Inc.)
  原价格: $155.00
  新价格: $160.00
  涨跌: +$5.00 (+3.23%)

  📧 [邮件通知] 发送到 user@example.com
     AAPL 价格变化: $155.00 -> $160.00 (+3.23%)

  📱 [短信通知] 发送到 138****8888
     AAPL 价格变化: $155.00 -> $160.00 (+3.23%)
```

### 示例 2: 条件通知
**输入**:
```go
stock := NewStock("GOOGL", "Google", 140.00)

// 价格上涨超过 5% 时通知
alertObserver := NewPriceAlertObserver("alert@example.com", 5.0)
stock.Attach(alertObserver)

stock.SetPrice(142.00) // 涨幅 1.43%，不通知
stock.SetPrice(148.00) // 涨幅 5.71%，通知
```

**输出**:
```
✓ [价格警报-alert@example.com] 已订阅股票: GOOGL (阈值: 5.0%)

📈 股票价格变化:
  股票: GOOGL (Google)
  原价格: $140.00
  新价格: $142.00
  涨跌: +$2.00 (+1.43%)

  ⚠️  [价格警报] 涨跌幅未达到阈值 (1.43% < 5.0%)

📈 股票价格变化:
  股票: GOOGL (Google)
  原价格: $142.00
  新价格: $148.00
  涨跌: +$6.00 (+4.23%)
  累计涨跌: +$8.00 (+5.71%)

  🚨 [价格警报] 发送到 alert@example.com
     GOOGL 价格上涨超过阈值！
     涨幅: 5.71% (阈值: 5.0%)
```

## 提示

💡 **提示 1**: 定义主题和观察者接口
```go
type Subject interface {
    Attach(observer Observer)
    Detach(observer Observer)
    Notify()
}

type Observer interface {
    Update(stock *Stock)
    GetID() string
}
```

💡 **提示 2**: 使用推模型传递数据
```go
func (s *Stock) Notify() {
    for _, observer := range s.observers {
        observer.Update(s) // 推送股票对象
    }
}
```

💡 **提示 3**: 考虑线程安全
```go
type Stock struct {
    observers []Observer
    mu        sync.RWMutex
}

func (s *Stock) Attach(observer Observer) {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.observers = append(s.observers, observer)
}
```

## 评分标准

- [ ] **功能完整性 (40%)**
  - 实现主题和观察者接口
  - 支持注册和注销观察者
  - 正确通知所有观察者
  - 实现条件通知

- [ ] **代码质量 (30%)**
  - 接口设计合理
  - 代码结构清晰
  - 线程安全

- [ ] **设计模式应用 (20%)**
  - 正确使用观察者模式
  - 主题和观察者松耦合
  - 支持动态添加观察者

- [ ] **用户体验 (10%)**
  - 通知信息清晰
  - 输出格式友好

## 扩展挑战

1. **多股票监控**: 支持同时监控多只股票
2. **历史记录**: 记录价格变化历史
3. **图表展示**: 生成价格变化图表
4. **智能预警**: 基于历史数据预测价格趋势
5. **通知分组**: 支持将观察者分组，批量通知

## 相关知识点

- 观察者模式的结构和实现
- 一对多依赖关系
- 推模型 vs 拉模型
- 线程安全的观察者模式
- 事件驱动架构
