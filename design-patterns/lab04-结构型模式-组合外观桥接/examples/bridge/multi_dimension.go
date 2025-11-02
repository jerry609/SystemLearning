package main

import "fmt"

// 桥接模式示例：多维度变化的消息系统
// 本示例展示了如何使用桥接模式处理消息类型和发送方式两个维度的变化

// Implementor 接口：消息发送器
type MessageSender interface {
	Send(to, content string) error
	GetName() string
}

// ConcreteImplementor A：邮件发送器
type EmailSender struct {
	smtpServer string
	port       int
}

func NewEmailSender(server string, port int) *EmailSender {
	return &EmailSender{
		smtpServer: server,
		port:       port,
	}
}

func (e *EmailSender) Send(to, content string) error {
	fmt.Printf("  📧 [Email] 通过 %s:%d 发送邮件\n", e.smtpServer, e.port)
	fmt.Printf("     收件人: %s\n", to)
	fmt.Printf("     内容: %s\n", content)
	return nil
}

func (e *EmailSender) GetName() string {
	return "Email"
}

// ConcreteImplementor B：短信发送器
type SMSSender struct {
	gateway string
	apiKey  string
}

func NewSMSSender(gateway, apiKey string) *SMSSender {
	return &SMSSender{
		gateway: gateway,
		apiKey:  apiKey,
	}
}

func (s *SMSSender) Send(to, content string) error {
	fmt.Printf("  📱 [SMS] 通过网关 %s 发送短信\n", s.gateway)
	fmt.Printf("     收件人: %s\n", to)
	fmt.Printf("     内容: %s\n", content)
	return nil
}

func (s *SMSSender) GetName() string {
	return "SMS"
}

// ConcreteImplementor C：推送通知发送器
type PushSender struct {
	service string
}

func NewPushSender(service string) *PushSender {
	return &PushSender{
		service: service,
	}
}

func (p *PushSender) Send(to, content string) error {
	fmt.Printf("  🔔 [Push] 通过 %s 发送推送通知\n", p.service)
	fmt.Printf("     收件人: %s\n", to)
	fmt.Printf("     内容: %s\n", content)
	return nil
}

func (p *PushSender) GetName() string {
	return "Push"
}

// ConcreteImplementor D：微信发送器
type WeChatSender struct {
	appID string
}

func NewWeChatSender(appID string) *WeChatSender {
	return &WeChatSender{
		appID: appID,
	}
}

func (w *WeChatSender) Send(to, content string) error {
	fmt.Printf("  💬 [WeChat] 通过应用 %s 发送微信消息\n", w.appID)
	fmt.Printf("     收件人: %s\n", to)
	fmt.Printf("     内容: %s\n", content)
	return nil
}

func (w *WeChatSender) GetName() string {
	return "WeChat"
}

// Abstraction：消息抽象
type Message struct {
	sender  MessageSender
	to      string
	content string
}

func (m *Message) SetSender(sender MessageSender) {
	m.sender = sender
}

func (m *Message) SetRecipient(to string) {
	m.to = to
}

func (m *Message) SetContent(content string) {
	m.content = content
}

// RefinedAbstraction A：普通消息
type NormalMessage struct {
	Message
	subject string
}

func NewNormalMessage(sender MessageSender) *NormalMessage {
	return &NormalMessage{
		Message: Message{sender: sender},
	}
}

func (n *NormalMessage) SetSubject(subject string) {
	n.subject = subject
}

func (n *NormalMessage) Send() error {
	fmt.Println("\n📨 发送普通消息")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  类型: 普通消息\n")
	fmt.Printf("  主题: %s\n", n.subject)
	fmt.Printf("  发送方式: %s\n", n.sender.GetName())
	return n.sender.Send(n.to, n.content)
}

// RefinedAbstraction B：紧急消息
type UrgentMessage struct {
	Message
	priority int
}

func NewUrgentMessage(sender MessageSender) *UrgentMessage {
	return &UrgentMessage{
		Message:  Message{sender: sender},
		priority: 1,
	}
}

func (u *UrgentMessage) SetPriority(priority int) {
	u.priority = priority
}

func (u *UrgentMessage) Send() error {
	fmt.Println("\n🚨 发送紧急消息")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  类型: 紧急消息\n")
	fmt.Printf("  优先级: %d\n", u.priority)
	fmt.Printf("  发送方式: %s\n", u.sender.GetName())
	
	// 紧急消息添加前缀
	urgentContent := fmt.Sprintf("[紧急] %s", u.content)
	return u.sender.Send(u.to, urgentContent)
}

// RefinedAbstraction C：加密消息
type EncryptedMessage struct {
	Message
	encryptionKey string
}

func NewEncryptedMessage(sender MessageSender) *EncryptedMessage {
	return &EncryptedMessage{
		Message:       Message{sender: sender},
		encryptionKey: "default-key",
	}
}

func (e *EncryptedMessage) SetEncryptionKey(key string) {
	e.encryptionKey = key
}

func (e *EncryptedMessage) Send() error {
	fmt.Println("\n🔐 发送加密消息")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  类型: 加密消息\n")
	fmt.Printf("  加密密钥: %s\n", e.encryptionKey)
	fmt.Printf("  发送方式: %s\n", e.sender.GetName())
	
	// 模拟加密
	encryptedContent := fmt.Sprintf("[已加密:%s] %s", e.encryptionKey, e.content)
	return e.sender.Send(e.to, encryptedContent)
}

// RefinedAbstraction D：群发消息
type BroadcastMessage struct {
	Message
	recipients []string
}

func NewBroadcastMessage(sender MessageSender) *BroadcastMessage {
	return &BroadcastMessage{
		Message:    Message{sender: sender},
		recipients: make([]string, 0),
	}
}

func (b *BroadcastMessage) AddRecipient(recipient string) {
	b.recipients = append(b.recipients, recipient)
}

func (b *BroadcastMessage) Send() error {
	fmt.Println("\n📢 发送群发消息")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  类型: 群发消息\n")
	fmt.Printf("  收件人数量: %d\n", len(b.recipients))
	fmt.Printf("  发送方式: %s\n", b.sender.GetName())
	
	for i, recipient := range b.recipients {
		fmt.Printf("\n  [%d/%d] 发送给: %s\n", i+1, len(b.recipients), recipient)
		if err := b.sender.Send(recipient, b.content); err != nil {
			return err
		}
	}
	return nil
}

func main() {
	fmt.Println("=== 桥接模式示例：多维度消息系统 ===")

	// 创建不同的发送器
	emailSender := NewEmailSender("smtp.example.com", 587)
	smsSender := NewSMSSender("sms.gateway.com", "api-key-123")
	pushSender := NewPushSender("Firebase")
	wechatSender := NewWeChatSender("wx-app-001")

	// 场景 1: 普通邮件消息
	normalEmail := NewNormalMessage(emailSender)
	normalEmail.SetSubject("会议通知")
	normalEmail.SetRecipient("zhangsan@example.com")
	normalEmail.SetContent("明天下午 3 点开会")
	normalEmail.Send()

	// 场景 2: 紧急短信消息
	urgentSMS := NewUrgentMessage(smsSender)
	urgentSMS.SetPriority(1)
	urgentSMS.SetRecipient("13800138000")
	urgentSMS.SetContent("服务器故障，请立即处理！")
	urgentSMS.Send()

	// 场景 3: 加密推送消息
	encryptedPush := NewEncryptedMessage(pushSender)
	encryptedPush.SetEncryptionKey("AES-256-KEY")
	encryptedPush.SetRecipient("user-123")
	encryptedPush.SetContent("您的验证码是: 123456")
	encryptedPush.Send()

	// 场景 4: 群发微信消息
	broadcast := NewBroadcastMessage(wechatSender)
	broadcast.AddRecipient("user-001")
	broadcast.AddRecipient("user-002")
	broadcast.AddRecipient("user-003")
	broadcast.SetContent("系统将于今晚 10 点进行维护")
	broadcast.Send()

	// 场景 5: 运行时切换发送方式
	fmt.Println("\n🔄 场景 5: 运行时切换发送方式")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	message := NewNormalMessage(emailSender)
	message.SetSubject("测试消息")
	message.SetRecipient("test@example.com")
	message.SetContent("这是一条测试消息")
	
	fmt.Println("\n初始发送方式: Email")
	message.Send()
	
	fmt.Println("\n切换到 SMS:")
	message.SetSender(smsSender)
	message.SetRecipient("13900139000")
	message.Send()
	
	fmt.Println("\n切换到 Push:")
	message.SetSender(pushSender)
	message.SetRecipient("user-456")
	message.Send()

	fmt.Println("\n=== 示例结束 ===")

	// 说明桥接模式的优势
	fmt.Println("\n💡 桥接模式的优势")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("1. 多维度独立变化")
	fmt.Println("   - 消息类型维度: 普通、紧急、加密、群发")
	fmt.Println("   - 发送方式维度: Email、SMS、Push、WeChat")
	fmt.Println("   - 两个维度可以独立扩展")
	fmt.Println()
	fmt.Println("2. 避免类爆炸")
	fmt.Println("   - 不使用桥接: 4种消息 × 4种方式 = 16个类")
	fmt.Println("   - 使用桥接: 4种消息 + 4种方式 = 8个类")
	fmt.Println()
	fmt.Println("3. 灵活组合")
	fmt.Println("   - 任意消息类型可以使用任意发送方式")
	fmt.Println("   - 运行时可以动态切换发送方式")
	fmt.Println()
	fmt.Println("4. 易于扩展")
	fmt.Println("   - 新增消息类型: 只需添加新的 RefinedAbstraction")
	fmt.Println("   - 新增发送方式: 只需添加新的 ConcreteImplementor")
	fmt.Println("   - 不影响现有代码")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
}

// 输出示例：
// === 桥接模式示例：多维度消息系统 ===
//
// 📨 发送普通消息
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//   类型: 普通消息
//   主题: 会议通知
//   发送方式: Email
//   📧 [Email] 通过 smtp.example.com:587 发送邮件
//      收件人: zhangsan@example.com
//      内容: 明天下午 3 点开会
//
// 🚨 发送紧急消息
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//   类型: 紧急消息
//   优先级: 1
//   发送方式: SMS
//   📱 [SMS] 通过网关 sms.gateway.com 发送短信
//      收件人: 13800138000
//      内容: [紧急] 服务器故障，请立即处理！
//
// 🔐 发送加密消息
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//   类型: 加密消息
//   加密密钥: AES-256-KEY
//   发送方式: Push
//   🔔 [Push] 通过 Firebase 发送推送通知
//      收件人: user-123
//      内容: [已加密:AES-256-KEY] 您的验证码是: 123456
//
// 📢 发送群发消息
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//   类型: 群发消息
//   收件人数量: 3
//   发送方式: WeChat
//
//   [1/3] 发送给: user-001
//   💬 [WeChat] 通过应用 wx-app-001 发送微信消息
//      收件人: user-001
//      内容: 系统将于今晚 10 点进行维护
//
//   [2/3] 发送给: user-002
//   💬 [WeChat] 通过应用 wx-app-001 发送微信消息
//      收件人: user-002
//      内容: 系统将于今晚 10 点进行维护
//
//   [3/3] 发送给: user-003
//   💬 [WeChat] 通过应用 wx-app-001 发送微信消息
//      收件人: user-003
//      内容: 系统将于今晚 10 点进行维护
//
// === 示例结束 ===
