package main

import (
	"fmt"
	"time"
)

// 练习 3: 消息发送系统 (桥接模式) - 参考答案
//
// 设计思路:
// 1. 定义 MessageSender 接口作为实现层，支持多种发送方式
// 2. 实现具体的发送器（Email、SMS、WeChat、DingTalk）
// 3. 定义 Message 基础结构作为抽象层
// 4. 实现具体的消息类型（Text、Image、Video、File）
// 5. 通过组合关系连接抽象和实现，支持运行时切换
//
// 使用的设计模式: 桥接模式 (Bridge Pattern)
// 模式应用位置: Message 抽象层和 MessageSender 实现层

// Implementor 接口: 消息发送器
type MessageSender interface {
	Send(to string, content interface{}) error
	GetName() string
	ValidateRecipient(to string) bool
}

// ConcreteImplementor A: 邮件发送器
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

func (e *EmailSender) Send(to string, content interface{}) error {
	fmt.Printf("  [Email] 通过 %s:%d 发送\n", e.smtpServer, e.port)
	fmt.Printf("  收件人: %s\n", to)
	fmt.Printf("  内容: %v\n", content)
	return nil
}

func (e *EmailSender) GetName() string {
	return "Email"
}

func (e *EmailSender) ValidateRecipient(to string) bool {
	// 简化的邮箱验证
	return len(to) > 0 && contains(to, "@")
}

// ConcreteImplementor B: 短信发送器
type SMSSender struct {
	gateway string
}

func NewSMSSender(gateway string) *SMSSender {
	return &SMSSender{gateway: gateway}
}

func (s *SMSSender) Send(to string, content interface{}) error {
	fmt.Printf("  [SMS] 通过网关 %s 发送\n", s.gateway)
	fmt.Printf("  收件人: %s\n", to)
	fmt.Printf("  内容: %v\n", content)
	return nil
}

func (s *SMSSender) GetName() string {
	return "SMS"
}

func (s *SMSSender) ValidateRecipient(to string) bool {
	// 简化的手机号验证
	return len(to) == 11
}

// ConcreteImplementor C: 微信发送器
type WeChatSender struct {
	appID string
}

func NewWeChatSender(appID string) *WeChatSender {
	return &WeChatSender{appID: appID}
}

func (w *WeChatSender) Send(to string, content interface{}) error {
	fmt.Printf("  [WeChat] 通过应用 %s 发送\n", w.appID)
	fmt.Printf("  收件人: %s\n", to)
	fmt.Printf("  内容: %v\n", content)
	return nil
}

func (w *WeChatSender) GetName() string {
	return "WeChat"
}

func (w *WeChatSender) ValidateRecipient(to string) bool {
	return len(to) > 0
}

// ConcreteImplementor D: 钉钉发送器
type DingTalkSender struct {
	botToken string
}

func NewDingTalkSender(token string) *DingTalkSender {
	return &DingTalkSender{botToken: token}
}

func (d *DingTalkSender) Send(to string, content interface{}) error {
	fmt.Printf("  [DingTalk] 通过机器人发送\n")
	fmt.Printf("  收件人: %s\n", to)
	fmt.Printf("  内容: %v\n", content)
	return nil
}

func (d *DingTalkSender) GetName() string {
	return "DingTalk"
}

func (d *DingTalkSender) ValidateRecipient(to string) bool {
	return len(to) > 0
}

// Abstraction: 消息基类
type Message struct {
	sender    MessageSender
	recipient string
	sentTime  time.Time
	status    string
}

func (m *Message) SetSender(sender MessageSender) {
	m.sender = sender
}

func (m *Message) SetRecipient(recipient string) {
	m.recipient = recipient
}

func (m *Message) GetStatus() string {
	return m.status
}

// RefinedAbstraction A: 文本消息
type TextMessage struct {
	Message
	content string
}

func NewTextMessage(sender MessageSender) *TextMessage {
	return &TextMessage{
		Message: Message{
			sender: sender,
			status: "待发送",
		},
	}
}

func (t *TextMessage) SetContent(content string) {
	t.content = content
}

func (t *TextMessage) Send() error {
	fmt.Println("\n📧 发送文本消息")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  消息类型: 文本消息\n")
	fmt.Printf("  发送方式: %s\n", t.sender.GetName())
	fmt.Printf("  收件人: %s\n", t.recipient)
	fmt.Printf("  内容: %s\n", t.content)
	fmt.Printf("  字数: %d\n", len(t.content))
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	if !t.sender.ValidateRecipient(t.recipient) {
		t.status = "失败"
		return fmt.Errorf("收件人格式无效")
	}
	
	err := t.sender.Send(t.recipient, t.content)
	if err == nil {
		t.status = "已发送"
		t.sentTime = time.Now()
		fmt.Println("  ✅ 发送成功")
	} else {
		t.status = "失败"
	}
	return err
}

// RefinedAbstraction B: 图片消息
type ImageMessage struct {
	Message
	imageURL string
	width    int
	height   int
}

func NewImageMessage(sender MessageSender) *ImageMessage {
	return &ImageMessage{
		Message: Message{
			sender: sender,
			status: "待发送",
		},
	}
}

func (i *ImageMessage) SetImageURL(url string) {
	i.imageURL = url
}

func (i *ImageMessage) SetSize(width, height int) {
	i.width = width
	i.height = height
}

func (i *ImageMessage) Send() error {
	fmt.Println("\n🖼️  发送图片消息")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  消息类型: 图片消息\n")
	fmt.Printf("  发送方式: %s\n", i.sender.GetName())
	fmt.Printf("  收件人: %s\n", i.recipient)
	fmt.Printf("  图片URL: %s\n", i.imageURL)
	fmt.Printf("  尺寸: %dx%d\n", i.width, i.height)
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	content := map[string]interface{}{
		"type":   "image",
		"url":    i.imageURL,
		"width":  i.width,
		"height": i.height,
	}
	
	err := i.sender.Send(i.recipient, content)
	if err == nil {
		i.status = "已发送"
		i.sentTime = time.Now()
		fmt.Println("  ✅ 发送成功")
	} else {
		i.status = "失败"
	}
	return err
}

// RefinedAbstraction C: 视频消息
type VideoMessage struct {
	Message
	videoURL  string
	duration  int
	coverURL  string
}

func NewVideoMessage(sender MessageSender) *VideoMessage {
	return &VideoMessage{
		Message: Message{
			sender: sender,
			status: "待发送",
		},
	}
}

func (v *VideoMessage) SetVideoURL(url string) {
	v.videoURL = url
}

func (v *VideoMessage) SetDuration(duration int) {
	v.duration = duration
}

func (v *VideoMessage) SetCoverURL(url string) {
	v.coverURL = url
}

func (v *VideoMessage) Send() error {
	fmt.Println("\n🎥 发送视频消息")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  消息类型: 视频消息\n")
	fmt.Printf("  发送方式: %s\n", v.sender.GetName())
	fmt.Printf("  收件人: %s\n", v.recipient)
	fmt.Printf("  视频URL: %s\n", v.videoURL)
	fmt.Printf("  时长: %d秒\n", v.duration)
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	content := map[string]interface{}{
		"type":     "video",
		"url":      v.videoURL,
		"duration": v.duration,
		"cover":    v.coverURL,
	}
	
	err := v.sender.Send(v.recipient, content)
	if err == nil {
		v.status = "已发送"
		v.sentTime = time.Now()
		fmt.Println("  ✅ 发送成功")
	} else {
		v.status = "失败"
	}
	return err
}

// RefinedAbstraction D: 文件消息
type FileMessage struct {
	Message
	filePath string
	fileSize int64
	fileType string
}

func NewFileMessage(sender MessageSender) *FileMessage {
	return &FileMessage{
		Message: Message{
			sender: sender,
			status: "待发送",
		},
	}
}

func (f *FileMessage) SetFilePath(path string) {
	f.filePath = path
}

func (f *FileMessage) SetFileSize(size int64) {
	f.fileSize = size
}

func (f *FileMessage) SetFileType(fileType string) {
	f.fileType = fileType
}

func (f *FileMessage) Send() error {
	fmt.Println("\n📎 发送文件消息")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  消息类型: 文件消息\n")
	fmt.Printf("  发送方式: %s\n", f.sender.GetName())
	fmt.Printf("  收件人: %s\n", f.recipient)
	fmt.Printf("  文件: %s\n", f.filePath)
	fmt.Printf("  大小: %d KB\n", f.fileSize)
	fmt.Printf("  类型: %s\n", f.fileType)
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	content := map[string]interface{}{
		"type": "file",
		"path": f.filePath,
		"size": f.fileSize,
	}
	
	err := f.sender.Send(f.recipient, content)
	if err == nil {
		f.status = "已发送"
		f.sentTime = time.Now()
		fmt.Println("  ✅ 发送成功")
	} else {
		f.status = "失败"
	}
	return err
}

// SendToMultiple 批量发送
func (f *FileMessage) SendToMultiple(recipients []string) error {
	fmt.Println("\n📢 批量发送文件消息")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  消息类型: 文件消息\n")
	fmt.Printf("  发送方式: %s\n", f.sender.GetName())
	fmt.Printf("  文件: %s\n", f.filePath)
	fmt.Printf("  大小: %d KB\n", f.fileSize)
	fmt.Printf("  收件人数量: %d\n", len(recipients))
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	success := 0
	failed := 0
	
	for i, recipient := range recipients {
		fmt.Printf("  [%d/%d] 发送给 %s ", i+1, len(recipients), recipient)
		
		content := map[string]interface{}{
			"type": "file",
			"path": f.filePath,
			"size": f.fileSize,
		}
		
		err := f.sender.Send(recipient, content)
		if err == nil {
			fmt.Println("✅")
			success++
		} else {
			fmt.Println("❌")
			failed++
		}
	}
	
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("✅ 批量发送完成: 成功 %d, 失败 %d\n", success, failed)
	
	return nil
}

// 辅助函数
func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func main() {
	fmt.Println("=== 练习 3: 消息发送系统 (桥接模式) ===")

	// 创建不同的发送器
	emailSender := NewEmailSender("smtp.example.com", 587)
	smsSender := NewSMSSender("sms.gateway.com")
	wechatSender := NewWeChatSender("wx-app-001")
	dingTalkSender := NewDingTalkSender("dingtalk-bot-token")

	// 场景 1: 发送文本消息（邮件）
	textEmail := NewTextMessage(emailSender)
	textEmail.SetRecipient("user@example.com")
	textEmail.SetContent("Hello, World!")
	textEmail.Send()

	// 场景 2: 发送文本消息（短信）
	textSMS := NewTextMessage(smsSender)
	textSMS.SetRecipient("13800138000")
	textSMS.SetContent("验证码: 123456")
	textSMS.Send()

	// 场景 3: 发送图片消息（微信）
	imageMsg := NewImageMessage(wechatSender)
	imageMsg.SetRecipient("user-123")
	imageMsg.SetImageURL("https://example.com/image.jpg")
	imageMsg.SetSize(1920, 1080)
	imageMsg.Send()

	// 场景 4: 发送视频消息（钉钉）
	videoMsg := NewVideoMessage(dingTalkSender)
	videoMsg.SetRecipient("group-456")
	videoMsg.SetVideoURL("https://example.com/video.mp4")
	videoMsg.SetDuration(120)
	videoMsg.SetCoverURL("https://example.com/cover.jpg")
	videoMsg.Send()

	// 场景 5: 批量发送文件消息
	fileMsg := NewFileMessage(dingTalkSender)
	fileMsg.SetFilePath("/path/to/report.pdf")
	fileMsg.SetFileSize(2048)
	fileMsg.SetFileType("PDF")
	recipients := []string{"user-001", "user-002", "user-003"}
	fileMsg.SendToMultiple(recipients)

	// 场景 6: 运行时切换发送方式
	fmt.Println("\n🔄 场景 6: 运行时切换发送方式")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	message := NewTextMessage(emailSender)
	message.SetContent("测试消息")
	
	fmt.Println("\n初始发送方式: Email")
	message.SetRecipient("test@example.com")
	message.Send()
	
	fmt.Println("\n切换到 SMS:")
	message.SetSender(smsSender)
	message.SetRecipient("13900139000")
	message.Send()
	
	fmt.Println("\n切换到 WeChat:")
	message.SetSender(wechatSender)
	message.SetRecipient("user-789")
	message.Send()

	fmt.Println("\n=== 示例结束 ===")

	// 说明桥接模式的优势
	fmt.Println("\n💡 桥接模式的优势")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("1. 避免类爆炸")
	fmt.Println("   - 不使用桥接: 4种消息 × 4种方式 = 16个类")
	fmt.Println("   - 使用桥接: 4种消息 + 4种方式 = 8个类")
	fmt.Println()
	fmt.Println("2. 独立扩展")
	fmt.Println("   - 新增消息类型不影响发送方式")
	fmt.Println("   - 新增发送方式不影响消息类型")
	fmt.Println()
	fmt.Println("3. 运行时切换")
	fmt.Println("   - 可以动态改变消息的发送方式")
	fmt.Println("   - 支持灵活的组合")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
}

// 可能的优化方向:
// 1. 实现消息队列，支持异步发送
// 2. 添加失败重试机制（指数退避）
// 3. 实现消息模板和变量替换
// 4. 添加发送统计和报表功能
// 5. 支持消息加密和数字签名
// 6. 实现多发送器组合（主备、负载均衡）
// 7. 添加消息追踪和状态查询
//
// 变体实现:
// 1. 使用工厂模式创建发送器
// 2. 使用策略模式选择发送方式
// 3. 使用观察者模式通知发送状态
// 4. 使用命令模式封装发送操作
