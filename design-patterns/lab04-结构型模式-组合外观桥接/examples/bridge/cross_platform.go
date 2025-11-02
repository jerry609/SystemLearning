package main

import "fmt"

// 桥接模式示例：跨平台 GUI
// 本示例展示了如何使用桥接模式实现跨平台的 GUI 控件

// Implementor 接口：平台实现
type Platform interface {
	DrawButton(x, y, width, height int, text string)
	DrawTextField(x, y, width, height int, text string)
	DrawCheckbox(x, y int, checked bool, label string)
}

// ConcreteImplementor A：Windows 平台
type WindowsPlatform struct{}

func (w *WindowsPlatform) DrawButton(x, y, width, height int, text string) {
	fmt.Printf("  [Windows] 绘制按钮: 位置=(%d,%d), 大小=%dx%d, 文本='%s'\n", 
		x, y, width, height, text)
	fmt.Println("  └─ 使用 Windows 原生控件样式")
}

func (w *WindowsPlatform) DrawTextField(x, y, width, height int, text string) {
	fmt.Printf("  [Windows] 绘制文本框: 位置=(%d,%d), 大小=%dx%d, 内容='%s'\n", 
		x, y, width, height, text)
	fmt.Println("  └─ 使用 Windows 原生输入框样式")
}

func (w *WindowsPlatform) DrawCheckbox(x, y int, checked bool, label string) {
	status := "未选中"
	if checked {
		status = "已选中"
	}
	fmt.Printf("  [Windows] 绘制复选框: 位置=(%d,%d), 状态=%s, 标签='%s'\n", 
		x, y, status, label)
	fmt.Println("  └─ 使用 Windows 原生复选框样式")
}

// ConcreteImplementor B：Linux 平台
type LinuxPlatform struct{}

func (l *LinuxPlatform) DrawButton(x, y, width, height int, text string) {
	fmt.Printf("  [Linux/GTK] 绘制按钮: 位置=(%d,%d), 大小=%dx%d, 文本='%s'\n", 
		x, y, width, height, text)
	fmt.Println("  └─ 使用 GTK 主题样式")
}

func (l *LinuxPlatform) DrawTextField(x, y, width, height int, text string) {
	fmt.Printf("  [Linux/GTK] 绘制文本框: 位置=(%d,%d), 大小=%dx%d, 内容='%s'\n", 
		x, y, width, height, text)
	fmt.Println("  └─ 使用 GTK 输入框样式")
}

func (l *LinuxPlatform) DrawCheckbox(x, y int, checked bool, label string) {
	status := "[ ]"
	if checked {
		status = "[✓]"
	}
	fmt.Printf("  [Linux/GTK] 绘制复选框: 位置=(%d,%d), 状态=%s, 标签='%s'\n", 
		x, y, status, label)
	fmt.Println("  └─ 使用 GTK 复选框样式")
}

// ConcreteImplementor C：macOS 平台
type MacOSPlatform struct{}

func (m *MacOSPlatform) DrawButton(x, y, width, height int, text string) {
	fmt.Printf("  [macOS] 绘制按钮: 位置=(%d,%d), 大小=%dx%d, 文本='%s'\n", 
		x, y, width, height, text)
	fmt.Println("  └─ 使用 Cocoa 控件样式")
}

func (m *MacOSPlatform) DrawTextField(x, y, width, height int, text string) {
	fmt.Printf("  [macOS] 绘制文本框: 位置=(%d,%d), 大小=%dx%d, 内容='%s'\n", 
		x, y, width, height, text)
	fmt.Println("  └─ 使用 Cocoa 输入框样式")
}

func (m *MacOSPlatform) DrawCheckbox(x, y int, checked bool, label string) {
	status := "○"
	if checked {
		status = "●"
	}
	fmt.Printf("  [macOS] 绘制复选框: 位置=(%d,%d), 状态=%s, 标签='%s'\n", 
		x, y, status, label)
	fmt.Println("  └─ 使用 Cocoa 复选框样式")
}

// Abstraction：控件抽象
type Widget struct {
	platform Platform
	x, y     int
}

func (w *Widget) SetPlatform(platform Platform) {
	w.platform = platform
}

func (w *Widget) SetPosition(x, y int) {
	w.x = x
	w.y = y
}

// RefinedAbstraction A：按钮
type Button struct {
	Widget
	width, height int
	text          string
}

func NewButton(text string, platform Platform) *Button {
	return &Button{
		Widget: Widget{platform: platform},
		width:  100,
		height: 30,
		text:   text,
	}
}

func (b *Button) Draw() {
	b.platform.DrawButton(b.x, b.y, b.width, b.height, b.text)
}

func (b *Button) SetSize(width, height int) {
	b.width = width
	b.height = height
}

func (b *Button) SetText(text string) {
	b.text = text
}

// RefinedAbstraction B：文本框
type TextField struct {
	Widget
	width, height int
	text          string
}

func NewTextField(text string, platform Platform) *TextField {
	return &TextField{
		Widget: Widget{platform: platform},
		width:  200,
		height: 25,
		text:   text,
	}
}

func (t *TextField) Draw() {
	t.platform.DrawTextField(t.x, t.y, t.width, t.height, t.text)
}

func (t *TextField) SetSize(width, height int) {
	t.width = width
	t.height = height
}

func (t *TextField) SetText(text string) {
	t.text = text
}

// RefinedAbstraction C：复选框
type Checkbox struct {
	Widget
	checked bool
	label   string
}

func NewCheckbox(label string, platform Platform) *Checkbox {
	return &Checkbox{
		Widget:  Widget{platform: platform},
		checked: false,
		label:   label,
	}
}

func (c *Checkbox) Draw() {
	c.platform.DrawCheckbox(c.x, c.y, c.checked, c.label)
}

func (c *Checkbox) SetChecked(checked bool) {
	c.checked = checked
}

func (c *Checkbox) Toggle() {
	c.checked = !c.checked
}

func main() {
	fmt.Println("=== 桥接模式示例：跨平台 GUI ===\n")

	// 创建不同平台的实现
	windows := &WindowsPlatform{}
	linux := &LinuxPlatform{}
	macos := &MacOSPlatform{}

	// 场景 1: 在 Windows 平台上绘制控件
	fmt.Println("📱 场景 1: Windows 平台")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	btnWindows := NewButton("确定", windows)
	btnWindows.SetPosition(10, 10)
	btnWindows.Draw()
	
	textWindows := NewTextField("请输入用户名", windows)
	textWindows.SetPosition(10, 50)
	textWindows.Draw()
	
	checkWindows := NewCheckbox("记住密码", windows)
	checkWindows.SetPosition(10, 85)
	checkWindows.SetChecked(true)
	checkWindows.Draw()
	
	fmt.Println()

	// 场景 2: 在 Linux 平台上绘制相同的控件
	fmt.Println("🐧 场景 2: Linux 平台")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	btnLinux := NewButton("确定", linux)
	btnLinux.SetPosition(10, 10)
	btnLinux.Draw()
	
	textLinux := NewTextField("请输入用户名", linux)
	textLinux.SetPosition(10, 50)
	textLinux.Draw()
	
	checkLinux := NewCheckbox("记住密码", linux)
	checkLinux.SetPosition(10, 85)
	checkLinux.SetChecked(true)
	checkLinux.Draw()
	
	fmt.Println()

	// 场景 3: 在 macOS 平台上绘制相同的控件
	fmt.Println("🍎 场景 3: macOS 平台")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	btnMac := NewButton("确定", macos)
	btnMac.SetPosition(10, 10)
	btnMac.Draw()
	
	textMac := NewTextField("请输入用户名", macos)
	textMac.SetPosition(10, 50)
	textMac.Draw()
	
	checkMac := NewCheckbox("记住密码", macos)
	checkMac.SetPosition(10, 85)
	checkMac.SetChecked(true)
	checkMac.Draw()
	
	fmt.Println()

	// 场景 4: 运行时切换平台
	fmt.Println("🔄 场景 4: 运行时切换平台")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	btn := NewButton("提交", windows)
	btn.SetPosition(20, 20)
	
	fmt.Println("初始平台: Windows")
	btn.Draw()
	
	fmt.Println("\n切换到 Linux 平台:")
	btn.SetPlatform(linux)
	btn.Draw()
	
	fmt.Println("\n切换到 macOS 平台:")
	btn.SetPlatform(macos)
	btn.Draw()
	
	fmt.Println()

	fmt.Println("=== 示例结束 ===")

	// 说明桥接模式的优势
	fmt.Println("\n💡 桥接模式的优势")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("1. 分离抽象和实现")
	fmt.Println("   - 控件（抽象）和平台（实现）可以独立变化")
	fmt.Println("   - 新增控件不影响平台实现")
	fmt.Println("   - 新增平台不影响控件定义")
	fmt.Println()
	fmt.Println("2. 避免类爆炸")
	fmt.Println("   - 不使用桥接: 3种控件 × 3个平台 = 9个类")
	fmt.Println("   - 使用桥接: 3种控件 + 3个平台 = 6个类")
	fmt.Println()
	fmt.Println("3. 运行时切换实现")
	fmt.Println("   - 可以动态改变控件的平台实现")
	fmt.Println("   - 支持跨平台迁移")
	fmt.Println()
	fmt.Println("4. 符合开闭原则")
	fmt.Println("   - 扩展新控件或新平台不影响现有代码")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
}

// 输出示例：
// === 桥接模式示例：跨平台 GUI ===
//
// 📱 场景 1: Windows 平台
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//   [Windows] 绘制按钮: 位置=(10,10), 大小=100x30, 文本='确定'
//   └─ 使用 Windows 原生控件样式
//   [Windows] 绘制文本框: 位置=(10,50), 大小=200x25, 内容='请输入用户名'
//   └─ 使用 Windows 原生输入框样式
//   [Windows] 绘制复选框: 位置=(10,85), 状态=已选中, 标签='记住密码'
//   └─ 使用 Windows 原生复选框样式
//
// 🐧 场景 2: Linux 平台
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//   [Linux/GTK] 绘制按钮: 位置=(10,10), 大小=100x30, 文本='确定'
//   └─ 使用 GTK 主题样式
//   [Linux/GTK] 绘制文本框: 位置=(10,50), 大小=200x25, 内容='请输入用户名'
//   └─ 使用 GTK 输入框样式
//   [Linux/GTK] 绘制复选框: 位置=(10,85), 状态=[✓], 标签='记住密码'
//   └─ 使用 GTK 复选框样式
//
// 🍎 场景 3: macOS 平台
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//   [macOS] 绘制按钮: 位置=(10,10), 大小=100x30, 文本='确定'
//   └─ 使用 Cocoa 控件样式
//   [macOS] 绘制文本框: 位置=(10,50), 大小=200x25, 内容='请输入用户名'
//   └─ 使用 Cocoa 输入框样式
//   [macOS] 绘制复选框: 位置=(10,85), 状态=●, 标签='记住密码'
//   └─ 使用 Cocoa 复选框样式
//
// 🔄 场景 4: 运行时切换平台
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 初始平台: Windows
//   [Windows] 绘制按钮: 位置=(20,20), 大小=100x30, 文本='提交'
//   └─ 使用 Windows 原生控件样式
//
// 切换到 Linux 平台:
//   [Linux/GTK] 绘制按钮: 位置=(20,20), 大小=100x30, 文本='提交'
//   └─ 使用 GTK 主题样式
//
// 切换到 macOS 平台:
//   [macOS] 绘制按钮: 位置=(20,20), 大小=100x30, 文本='提交'
//   └─ 使用 Cocoa 控件样式
//
// === 示例结束 ===
