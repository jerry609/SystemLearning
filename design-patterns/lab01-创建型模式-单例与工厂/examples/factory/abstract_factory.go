package main

import (
	"fmt"
)

// 抽象工厂模式示例
// 场景：跨平台 UI 组件库，支持 Windows 和 Mac 两种风格

// ========== 抽象产品 A: Button ==========

// Button 按钮接口
type Button interface {
	Render()
	OnClick()
}

// WindowsButton Windows 风格按钮
type WindowsButton struct {
	text string
}

func (b *WindowsButton) Render() {
	fmt.Printf("渲染 Windows 按钮: [%s]\n", b.text)
}

func (b *WindowsButton) OnClick() {
	fmt.Println("Windows 按钮被点击")
}

// MacButton Mac 风格按钮
type MacButton struct {
	text string
}

func (b *MacButton) Render() {
	fmt.Printf("渲染 Mac 按钮: (%s)\n", b.text)
}

func (b *MacButton) OnClick() {
	fmt.Println("Mac 按钮被点击")
}

// ========== 抽象产品 B: Checkbox ==========

// Checkbox 复选框接口
type Checkbox interface {
	Render()
	Toggle()
}

// WindowsCheckbox Windows 风格复选框
type WindowsCheckbox struct {
	label   string
	checked bool
}

func (c *WindowsCheckbox) Render() {
	status := " "
	if c.checked {
		status = "X"
	}
	fmt.Printf("渲染 Windows 复选框: [%s] %s\n", status, c.label)
}

func (c *WindowsCheckbox) Toggle() {
	c.checked = !c.checked
	fmt.Printf("Windows 复选框状态: %v\n", c.checked)
}

// MacCheckbox Mac 风格复选框
type MacCheckbox struct {
	label   string
	checked bool
}

func (c *MacCheckbox) Render() {
	status := "○"
	if c.checked {
		status = "●"
	}
	fmt.Printf("渲染 Mac 复选框: %s %s\n", status, c.label)
}

func (c *MacCheckbox) Toggle() {
	c.checked = !c.checked
	fmt.Printf("Mac 复选框状态: %v\n", c.checked)
}

// ========== 抽象产品 C: TextField ==========

// TextField 文本框接口
type TextField interface {
	Render()
	SetText(text string)
	GetText() string
}

// WindowsTextField Windows 风格文本框
type WindowsTextField struct {
	text        string
	placeholder string
}

func (t *WindowsTextField) Render() {
	display := t.text
	if display == "" {
		display = t.placeholder
	}
	fmt.Printf("渲染 Windows 文本框: |%s|\n", display)
}

func (t *WindowsTextField) SetText(text string) {
	t.text = text
}

func (t *WindowsTextField) GetText() string {
	return t.text
}

// MacTextField Mac 风格文本框
type MacTextField struct {
	text        string
	placeholder string
}

func (t *MacTextField) Render() {
	display := t.text
	if display == "" {
		display = t.placeholder
	}
	fmt.Printf("渲染 Mac 文本框: <%s>\n", display)
}

func (t *MacTextField) SetText(text string) {
	t.text = text
}

func (t *MacTextField) GetText() string {
	return t.text
}

// ========== 抽象工厂 ==========

// GUIFactory UI 组件工厂接口
type GUIFactory interface {
	CreateButton(text string) Button
	CreateCheckbox(label string) Checkbox
	CreateTextField(placeholder string) TextField
}

// ========== 具体工厂 1: Windows ==========

// WindowsFactory Windows 风格工厂
type WindowsFactory struct{}

func (f *WindowsFactory) CreateButton(text string) Button {
	return &WindowsButton{text: text}
}

func (f *WindowsFactory) CreateCheckbox(label string) Checkbox {
	return &WindowsCheckbox{label: label, checked: false}
}

func (f *WindowsFactory) CreateTextField(placeholder string) TextField {
	return &WindowsTextField{placeholder: placeholder}
}

// ========== 具体工厂 2: Mac ==========

// MacFactory Mac 风格工厂
type MacFactory struct{}

func (f *MacFactory) CreateButton(text string) Button {
	return &MacButton{text: text}
}

func (f *MacFactory) CreateCheckbox(label string) Checkbox {
	return &MacCheckbox{label: label, checked: false}
}

func (f *MacFactory) CreateTextField(placeholder string) TextField {
	return &MacTextField{placeholder: placeholder}
}

// ========== 客户端代码 ==========

// Application 应用程序
type Application struct {
	factory GUIFactory
}

func NewApplication(factory GUIFactory) *Application {
	return &Application{factory: factory}
}

func (app *Application) CreateLoginForm() {
	fmt.Println("\n创建登录表单:")
	fmt.Println("================")

	// 创建组件
	usernameField := app.factory.CreateTextField("请输入用户名")
	passwordField := app.factory.CreateTextField("请输入密码")
	rememberCheckbox := app.factory.CreateCheckbox("记住我")
	loginButton := app.factory.CreateButton("登录")
	cancelButton := app.factory.CreateButton("取消")

	// 渲染组件
	usernameField.Render()
	passwordField.Render()
	rememberCheckbox.Render()
	loginButton.Render()
	cancelButton.Render()

	// 模拟用户交互
	fmt.Println("\n用户交互:")
	usernameField.SetText("alice")
	usernameField.Render()

	passwordField.SetText("******")
	passwordField.Render()

	rememberCheckbox.Toggle()
	rememberCheckbox.Render()

	loginButton.OnClick()
}

func (app *Application) CreateSettingsForm() {
	fmt.Println("\n创建设置表单:")
	fmt.Println("================")

	// 创建组件
	emailField := app.factory.CreateTextField("请输入邮箱")
	notificationCheckbox := app.factory.CreateCheckbox("启用通知")
	darkModeCheckbox := app.factory.CreateCheckbox("深色模式")
	saveButton := app.factory.CreateButton("保存")

	// 渲染组件
	emailField.Render()
	notificationCheckbox.Render()
	darkModeCheckbox.Render()
	saveButton.Render()

	// 模拟用户交互
	fmt.Println("\n用户交互:")
	emailField.SetText("alice@example.com")
	emailField.Render()

	notificationCheckbox.Toggle()
	notificationCheckbox.Render()

	darkModeCheckbox.Toggle()
	darkModeCheckbox.Render()

	saveButton.OnClick()
}

func main() {
	fmt.Println("=== 抽象工厂模式示例 ===")

	// 示例 1: Windows 风格应用
	fmt.Println("\n【示例 1: Windows 风格应用】")
	fmt.Println("================================")
	windowsFactory := &WindowsFactory{}
	windowsApp := NewApplication(windowsFactory)
	windowsApp.CreateLoginForm()
	windowsApp.CreateSettingsForm()

	// 示例 2: Mac 风格应用
	fmt.Println("\n\n【示例 2: Mac 风格应用】")
	fmt.Println("================================")
	macFactory := &MacFactory{}
	macApp := NewApplication(macFactory)
	macApp.CreateLoginForm()
	macApp.CreateSettingsForm()

	// 示例 3: 动态切换风格
	fmt.Println("\n\n【示例 3: 动态切换风格】")
	fmt.Println("================================")

	var factory GUIFactory
	platform := "mac" // 可以从配置文件或环境变量读取

	if platform == "windows" {
		factory = &WindowsFactory{}
		fmt.Println("使用 Windows 风格")
	} else {
		factory = &MacFactory{}
		fmt.Println("使用 Mac 风格")
	}

	app := NewApplication(factory)
	app.CreateLoginForm()

	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n抽象工厂模式特点:")
	fmt.Println("✅ 保证产品族的一致性 - 所有组件风格统一")
	fmt.Println("✅ 易于切换产品族 - 只需更换工厂")
	fmt.Println("✅ 符合开闭原则 - 添加新风格不影响现有代码")
	fmt.Println("✅ 隔离具体类 - 客户端不需要知道具体类名")
	fmt.Println("❌ 难以支持新产品 - 添加新组件需要修改所有工厂")
	fmt.Println("❌ 类的数量多 - 每个产品族都需要一套类")
	fmt.Println("\n适用场景:")
	fmt.Println("• 系统需要多个产品族")
	fmt.Println("• 产品族内的产品需要一起使用")
	fmt.Println("• 需要约束产品的组合使用")
	fmt.Println("• 跨平台应用（如 Windows/Mac/Linux）")
}
