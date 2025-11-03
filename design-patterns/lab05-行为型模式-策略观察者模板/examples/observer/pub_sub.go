package main

import (
	"fmt"
	"sync"
	"time"
)

// 消息
type Message struct {
	Topic     string
	Content   interface{}
	Timestamp time.Time
}

// 订阅者接口
type Subscriber interface {
	OnMessage(msg Message)
	GetID() string
}

// 发布-订阅系统
type PubSubSystem struct {
	subscribers map[string][]Subscriber // topic -> subscribers
	mu          sync.RWMutex
	msgChan     chan Message
	stopChan    chan struct{}
}

func NewPubSubSystem() *PubSubSystem {
	ps := &PubSubSystem{
		subscribers: make(map[string][]Subscriber),
		msgChan:     make(chan Message, 100),
		stopChan:    make(chan struct{}),
	}
	
	// 启动消息处理协程
	go ps.processMessages()
	
	return ps
}

// 订阅主题
func (ps *PubSubSystem) Subscribe(topic string, subscriber Subscriber) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	
	ps.subscribers[topic] = append(ps.subscribers[topic], subscriber)
	fmt.Printf("✓ [%s] 订阅了主题: %s\n", subscriber.GetID(), topic)
}

// 取消订阅
func (ps *PubSubSystem) Unsubscribe(topic string, subscriber Subscriber) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	
	subscribers := ps.subscribers[topic]
	for i, sub := range subscribers {
		if sub.GetID() == subscriber.GetID() {
			ps.subscribers[topic] = append(subscribers[:i], subscribers[i+1:]...)
			fmt.Printf("✗ [%s] 取消订阅主题: %s\n", subscriber.GetID(), topic)
			break
		}
	}
}

// 发布消息
func (ps *PubSubSystem) Publish(topic string, content interface{}) {
	msg := Message{
		Topic:     topic,
		Content:   content,
		Timestamp: time.Now(),
	}
	
	ps.msgChan <- msg
}

// 处理消息
func (ps *PubSubSystem) processMessages() {
	for {
		select {
		case msg := <-ps.msgChan:
			ps.deliverMessage(msg)
		case <-ps.stopChan:
			return
		}
	}
}

// 投递消息给订阅者
func (ps *PubSubSystem) deliverMessage(msg Message) {
	ps.mu.RLock()
	subscribers := make([]Subscriber, len(ps.subscribers[msg.Topic]))
	copy(subscribers, ps.subscribers[msg.Topic])
	ps.mu.RUnlock()
	
	fmt.Printf("\n📨 发布消息到主题: %s\n", msg.Topic)
	fmt.Printf("   内容: %v\n", msg.Content)
	fmt.Printf("   时间: %s\n", msg.Timestamp.Format("15:04:05"))
	fmt.Printf("   订阅者数量: %d\n\n", len(subscribers))
	
	// 并发投递消息
	var wg sync.WaitGroup
	for _, sub := range subscribers {
		wg.Add(1)
		go func(s Subscriber) {
			defer wg.Done()
			s.OnMessage(msg)
		}(sub)
	}
	wg.Wait()
}

// 停止系统
func (ps *PubSubSystem) Stop() {
	close(ps.stopChan)
	close(ps.msgChan)
}

// 获取主题的订阅者数量
func (ps *PubSubSystem) GetSubscriberCount(topic string) int {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	return len(ps.subscribers[topic])
}

// 新闻订阅者
type NewsSubscriber struct {
	id   string
	name string
}

func NewNewsSubscriber(id, name string) *NewsSubscriber {
	return &NewsSubscriber{id: id, name: name}
}

func (n *NewsSubscriber) OnMessage(msg Message) {
	fmt.Printf("  📰 [新闻订阅者-%s] %s 收到消息\n", n.id, n.name)
	fmt.Printf("     主题: %s\n", msg.Topic)
	fmt.Printf("     内容: %v\n", msg.Content)
}

func (n *NewsSubscriber) GetID() string {
	return n.id
}

// 股票订阅者
type StockSubscriber struct {
	id       string
	investor string
}

func NewStockSubscriber(id, investor string) *StockSubscriber {
	return &StockSubscriber{id: id, investor: investor}
}

func (s *StockSubscriber) OnMessage(msg Message) {
	fmt.Printf("  📈 [股票订阅者-%s] 投资者 %s 收到消息\n", s.id, s.investor)
	fmt.Printf("     主题: %s\n", msg.Topic)
	
	if data, ok := msg.Content.(map[string]interface{}); ok {
		if price, exists := data["price"]; exists {
			fmt.Printf("     股票价格: %v\n", price)
		}
		if change, exists := data["change"]; exists {
			fmt.Printf("     涨跌幅: %v\n", change)
		}
	}
}

func (s *StockSubscriber) GetID() string {
	return s.id
}

// 天气订阅者
type WeatherSubscriber struct {
	id   string
	city string
}

func NewWeatherSubscriber(id, city string) *WeatherSubscriber {
	return &WeatherSubscriber{id: id, city: city}
}

func (w *WeatherSubscriber) OnMessage(msg Message) {
	fmt.Printf("  🌤️  [天气订阅者-%s] %s 市民收到消息\n", w.id, w.city)
	fmt.Printf("     主题: %s\n", msg.Topic)
	
	if data, ok := msg.Content.(map[string]interface{}); ok {
		if temp, exists := data["temperature"]; exists {
			fmt.Printf("     温度: %v°C\n", temp)
		}
		if weather, exists := data["weather"]; exists {
			fmt.Printf("     天气: %v\n", weather)
		}
	}
}

func (w *WeatherSubscriber) GetID() string {
	return w.id
}

// 通用订阅者
type GenericSubscriber struct {
	id   string
	name string
}

func NewGenericSubscriber(id, name string) *GenericSubscriber {
	return &GenericSubscriber{id: id, name: name}
}

func (g *GenericSubscriber) OnMessage(msg Message) {
	fmt.Printf("  📬 [订阅者-%s] %s 收到消息\n", g.id, g.name)
	fmt.Printf("     主题: %s, 内容: %v\n", msg.Topic, msg.Content)
}

func (g *GenericSubscriber) GetID() string {
	return g.id
}

func main() {
	fmt.Println("=== 发布-订阅模式示例 ===\n")
	
	// 创建发布-订阅系统
	pubsub := NewPubSubSystem()
	defer pubsub.Stop()
	
	// 场景 1: 新闻订阅
	fmt.Println("【场景 1: 新闻订阅】\n")
	
	// 创建新闻订阅者
	newsReader1 := NewNewsSubscriber("news-001", "张三")
	newsReader2 := NewNewsSubscriber("news-002", "李四")
	newsReader3 := NewNewsSubscriber("news-003", "王五")
	
	// 订阅不同类型的新闻
	pubsub.Subscribe("news.tech", newsReader1)
	pubsub.Subscribe("news.tech", newsReader2)
	pubsub.Subscribe("news.sports", newsReader2)
	pubsub.Subscribe("news.sports", newsReader3)
	
	// 发布科技新闻
	fmt.Println()
	pubsub.Publish("news.tech", map[string]interface{}{
		"title":   "Go 1.22 正式发布",
		"content": "Go 语言发布了最新版本，带来了许多新特性...",
	})
	time.Sleep(500 * time.Millisecond)
	
	// 发布体育新闻
	pubsub.Publish("news.sports", map[string]interface{}{
		"title":   "世界杯决赛结果",
		"content": "经过激烈角逐，冠军诞生...",
	})
	time.Sleep(500 * time.Millisecond)
	
	// 场景 2: 股票价格订阅
	fmt.Println("\n\n【场景 2: 股票价格订阅】\n")
	
	// 创建股票订阅者
	investor1 := NewStockSubscriber("stock-001", "投资者A")
	investor2 := NewStockSubscriber("stock-002", "投资者B")
	investor3 := NewStockSubscriber("stock-003", "投资者C")
	
	// 订阅不同股票
	pubsub.Subscribe("stock.AAPL", investor1)
	pubsub.Subscribe("stock.AAPL", investor2)
	pubsub.Subscribe("stock.GOOGL", investor2)
	pubsub.Subscribe("stock.GOOGL", investor3)
	
	// 发布苹果股票价格
	fmt.Println()
	pubsub.Publish("stock.AAPL", map[string]interface{}{
		"symbol": "AAPL",
		"price":  175.50,
		"change": "+2.5%",
	})
	time.Sleep(500 * time.Millisecond)
	
	// 发布谷歌股票价格
	pubsub.Publish("stock.GOOGL", map[string]interface{}{
		"symbol": "GOOGL",
		"price":  140.20,
		"change": "-1.2%",
	})
	time.Sleep(500 * time.Millisecond)
	
	// 场景 3: 天气预报订阅
	fmt.Println("\n\n【场景 3: 天气预报订阅】\n")
	
	// 创建天气订阅者
	beijingCitizen := NewWeatherSubscriber("weather-001", "北京")
	shanghaiCitizen := NewWeatherSubscriber("weather-002", "上海")
	
	// 订阅城市天气
	pubsub.Subscribe("weather.beijing", beijingCitizen)
	pubsub.Subscribe("weather.shanghai", shanghaiCitizen)
	
	// 发布北京天气
	fmt.Println()
	pubsub.Publish("weather.beijing", map[string]interface{}{
		"city":        "北京",
		"temperature": 15,
		"weather":     "晴",
		"humidity":    "45%",
	})
	time.Sleep(500 * time.Millisecond)
	
	// 发布上海天气
	pubsub.Publish("weather.shanghai", map[string]interface{}{
		"city":        "上海",
		"temperature": 20,
		"weather":     "多云",
		"humidity":    "60%",
	})
	time.Sleep(500 * time.Millisecond)
	
	// 场景 4: 取消订阅
	fmt.Println("\n\n【场景 4: 取消订阅】\n")
	
	// 投资者A 取消订阅苹果股票
	pubsub.Unsubscribe("stock.AAPL", investor1)
	
	// 再次发布苹果股票价格
	fmt.Println()
	pubsub.Publish("stock.AAPL", map[string]interface{}{
		"symbol": "AAPL",
		"price":  176.80,
		"change": "+0.7%",
	})
	time.Sleep(500 * time.Millisecond)
	
	// 场景 5: 通配符订阅（订阅所有新闻）
	fmt.Println("\n\n【场景 5: 多主题订阅】\n")
	
	// 创建一个订阅所有新闻的订阅者
	allNewsReader := NewGenericSubscriber("all-news-001", "新闻爱好者")
	
	pubsub.Subscribe("news.tech", allNewsReader)
	pubsub.Subscribe("news.sports", allNewsReader)
	pubsub.Subscribe("news.finance", allNewsReader)
	
	// 发布不同类型的新闻
	fmt.Println()
	pubsub.Publish("news.tech", "AI 技术取得重大突破")
	time.Sleep(300 * time.Millisecond)
	
	pubsub.Publish("news.sports", "奥运会即将开幕")
	time.Sleep(300 * time.Millisecond)
	
	pubsub.Publish("news.finance", "股市创新高")
	time.Sleep(300 * time.Millisecond)
	
	// 场景 6: 查看订阅统计
	fmt.Println("\n\n【场景 6: 订阅统计】\n")
	
	topics := []string{
		"news.tech",
		"news.sports",
		"stock.AAPL",
		"stock.GOOGL",
		"weather.beijing",
	}
	
	fmt.Println("各主题的订阅者数量:")
	for _, topic := range topics {
		count := pubsub.GetSubscriberCount(topic)
		fmt.Printf("  %s: %d 个订阅者\n", topic, count)
	}
	
	// 场景 7: 批量发布
	fmt.Println("\n\n【场景 7: 批量发布消息】\n")
	
	fmt.Println("批量发布股票更新...")
	stocks := []struct {
		symbol string
		price  float64
		change string
	}{
		{"AAPL", 177.20, "+0.9%"},
		{"GOOGL", 141.50, "+0.9%"},
		{"MSFT", 380.00, "+1.5%"},
	}
	
	for _, stock := range stocks {
		topic := fmt.Sprintf("stock.%s", stock.symbol)
		pubsub.Publish(topic, map[string]interface{}{
			"symbol": stock.symbol,
			"price":  stock.price,
			"change": stock.change,
		})
		time.Sleep(200 * time.Millisecond)
	}
	
	// 等待所有消息处理完成
	time.Sleep(1 * time.Second)
	
	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n💡 发布-订阅模式的特点:")
	fmt.Println("- 完全解耦: 发布者和订阅者互不知道对方")
	fmt.Println("- 异步通信: 通过消息队列实现异步处理")
	fmt.Println("- 主题分类: 支持按主题订阅和发布")
	fmt.Println("- 可扩展性: 易于添加新的发布者和订阅者")
	fmt.Println("- 并发处理: 支持并发投递消息")
}
