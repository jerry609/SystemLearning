package main

import "fmt"

// 练习 2: 智能家居控制器 (外观模式) - 参考答案
//
// 设计思路:
// 1. 实现各个子系统（灯光、空调、窗帘、音响、安防）
// 2. 创建外观类 SmartHomeFacade，封装子系统的复杂操作
// 3. 提供高层接口（场景模式），简化客户端使用
// 4. 每个场景按合理顺序协调多个子系统
//
// 使用的设计模式: 外观模式 (Facade Pattern)
// 模式应用位置: SmartHomeFacade 类

// 子系统 1: 灯光系统
type LightingSystem struct{}

func (l *LightingSystem) TurnOn(room string) {
	fmt.Printf("  [灯光系统] 打开%s灯\n", room)
}

func (l *LightingSystem) TurnOff(room string) {
	fmt.Printf("  [灯光系统] 关闭%s灯\n", room)
}

func (l *LightingSystem) SetBrightness(room string, level int) {
	fmt.Printf("  [灯光系统] %s灯亮度设置为 %d%%\n", room, level)
}

func (l *LightingSystem) SetColor(room string, color string) {
	fmt.Printf("  [灯光系统] %s灯颜色设置为 %s\n", room, color)
}

// 子系统 2: 空调系统
type AirConditioner struct{}

func (a *AirConditioner) TurnOn() {
	fmt.Println("  [空调系统] 开启空调")
}

func (a *AirConditioner) TurnOff() {
	fmt.Println("  [空调系统] 关闭空调")
}

func (a *AirConditioner) SetTemperature(temp int) {
	fmt.Printf("  [空调系统] 设置温度为 %d°C\n", temp)
}

func (a *AirConditioner) SetMode(mode string) {
	fmt.Printf("  [空调系统] 设置模式为 %s\n", mode)
}

// 子系统 3: 窗帘系统
type CurtainSystem struct{}

func (c *CurtainSystem) Open(room string) {
	fmt.Printf("  [窗帘系统] 打开%s窗帘\n", room)
}

func (c *CurtainSystem) Close(room string) {
	fmt.Printf("  [窗帘系统] 关闭%s窗帘\n", room)
}

func (c *CurtainSystem) SetPosition(room string, position int) {
	fmt.Printf("  [窗帘系统] %s窗帘位置设置为 %d%%\n", room, position)
}

// 子系统 4: 音响系统
type AudioSystem struct{}

func (a *AudioSystem) TurnOn() {
	fmt.Println("  [音响系统] 开启音响")
}

func (a *AudioSystem) TurnOff() {
	fmt.Println("  [音响系统] 关闭音响")
}

func (a *AudioSystem) SetVolume(level int) {
	fmt.Printf("  [音响系统] 设置音量为 %d\n", level)
}

func (a *AudioSystem) PlayMusic(song string) {
	fmt.Printf("  [音响系统] 播放音乐: %s\n", song)
}

// 子系统 5: 安防系统
type SecuritySystem struct{}

func (s *SecuritySystem) Arm() {
	fmt.Println("  [安防系统] 系统布防")
}

func (s *SecuritySystem) Disarm() {
	fmt.Println("  [安防系统] 系统撤防")
}

func (s *SecuritySystem) LockDoors() {
	fmt.Println("  [安防系统] 锁定所有门")
}

func (s *SecuritySystem) UnlockDoors() {
	fmt.Println("  [安防系统] 解锁所有门")
}

func (s *SecuritySystem) EnableCameras() {
	fmt.Println("  [安防系统] 启用摄像头")
}

func (s *SecuritySystem) DisableCameras() {
	fmt.Println("  [安防系统] 禁用摄像头")
}

// 外观类: 智能家居控制器
type SmartHomeFacade struct {
	lighting *LightingSystem
	ac       *AirConditioner
	curtain  *CurtainSystem
	audio    *AudioSystem
	security *SecuritySystem
}

func NewSmartHomeFacade() *SmartHomeFacade {
	return &SmartHomeFacade{
		lighting: &LightingSystem{},
		ac:       &AirConditioner{},
		curtain:  &CurtainSystem{},
		audio:    &AudioSystem{},
		security: &SecuritySystem{},
	}
}

// GoodMorning 早安模式
func (s *SmartHomeFacade) GoodMorning() {
	fmt.Println("\n🌅 执行早安模式")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	s.curtain.Open("卧室")
	s.lighting.TurnOn("卧室")
	s.lighting.SetBrightness("卧室", 50)
	s.audio.TurnOn()
	s.audio.PlayMusic("轻音乐")
	s.audio.SetVolume(20)
	s.ac.TurnOn()
	s.ac.SetTemperature(24)
	s.ac.SetMode("制冷")
	
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ 早安模式执行完成\n")
}

// LeaveHome 离家模式
func (s *SmartHomeFacade) LeaveHome() {
	fmt.Println("\n🚪 执行离家模式")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	s.lighting.TurnOff("客厅")
	s.lighting.TurnOff("卧室")
	s.lighting.TurnOff("厨房")
	s.ac.TurnOff()
	s.audio.TurnOff()
	s.curtain.Close("客厅")
	s.curtain.Close("卧室")
	s.security.LockDoors()
	s.security.EnableCameras()
	s.security.Arm()
	
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ 离家模式执行完成，家已安全\n")
}

// ComeHome 回家模式
func (s *SmartHomeFacade) ComeHome() {
	fmt.Println("\n🏠 执行回家模式")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	s.security.Disarm()
	s.security.UnlockDoors()
	s.lighting.TurnOn("客厅")
	s.lighting.SetBrightness("客厅", 80)
	s.ac.TurnOn()
	s.ac.SetTemperature(26)
	s.ac.SetMode("制冷")
	s.curtain.Open("客厅")
	
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ 回家模式执行完成，欢迎回家\n")
}

// MovieMode 观影模式
func (s *SmartHomeFacade) MovieMode() {
	fmt.Println("\n🎬 执行观影模式")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	s.lighting.TurnOff("客厅")
	s.curtain.Close("客厅")
	s.audio.TurnOn()
	s.audio.SetVolume(60)
	s.ac.SetTemperature(25)
	
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ 观影模式执行完成，尽情享受电影吧！\n")
}

// SleepMode 睡眠模式
func (s *SmartHomeFacade) SleepMode() {
	fmt.Println("\n🌙 执行睡眠模式")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	s.lighting.TurnOff("客厅")
	s.lighting.TurnOff("卧室")
	s.lighting.TurnOff("厨房")
	s.audio.TurnOff()
	s.curtain.Close("卧室")
	s.ac.SetTemperature(26)
	s.ac.SetMode("睡眠")
	s.security.LockDoors()
	s.security.Arm()
	
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ 睡眠模式执行完成，晚安\n")
}

// PartyMode 派对模式
func (s *SmartHomeFacade) PartyMode() {
	fmt.Println("\n🎉 执行派对模式")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	s.lighting.TurnOn("客厅")
	s.lighting.SetColor("客厅", "彩色")
	s.lighting.TurnOn("卧室")
	s.lighting.SetColor("卧室", "彩色")
	s.audio.TurnOn()
	s.audio.SetVolume(80)
	s.audio.PlayMusic("派对音乐")
	s.curtain.Open("客厅")
	s.security.Disarm()
	s.security.DisableCameras()
	
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ 派对模式执行完成，尽情狂欢吧！\n")
}

func main() {
	fmt.Println("=== 练习 2: 智能家居控制器 (外观模式) ===")

	// 创建智能家居外观
	smartHome := NewSmartHomeFacade()

	// 场景 1: 早安模式
	smartHome.GoodMorning()

	// 场景 2: 离家模式
	smartHome.LeaveHome()

	// 场景 3: 回家模式
	smartHome.ComeHome()

	// 场景 4: 观影模式
	smartHome.MovieMode()

	// 场景 5: 睡眠模式
	smartHome.SleepMode()

	// 场景 6: 派对模式
	smartHome.PartyMode()

	fmt.Println("=== 示例结束 ===")

	// 说明外观模式的优势
	fmt.Println("\n💡 外观模式的优势")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("1. 简化客户端使用")
	fmt.Println("   - 一个方法调用完成复杂操作")
	fmt.Println("   - 不需要了解各个子系统的细节")
	fmt.Println()
	fmt.Println("2. 降低耦合度")
	fmt.Println("   - 客户端与子系统解耦")
	fmt.Println("   - 子系统变化不影响客户端")
	fmt.Println()
	fmt.Println("3. 更好的分层")
	fmt.Println("   - 外观作为子系统的统一入口")
	fmt.Println("   - 便于系统维护和扩展")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
}

// 可能的优化方向:
// 1. 添加自定义场景功能，允许用户保存和加载场景配置
// 2. 实现定时任务，支持定时执行场景
// 3. 添加条件触发，根据环境自动执行场景
// 4. 实现状态查询，查看所有设备的当前状态
// 5. 添加能源管理，统计和优化能耗
// 6. 支持多用户配置，每个用户有自己的偏好
// 7. 添加语音控制接口
//
// 变体实现:
// 1. 使用配置文件定义场景，而不是硬编码
// 2. 使用命令模式封装场景操作，支持撤销
// 3. 使用观察者模式，设备状态变化时通知外观
// 4. 添加日志记录，记录所有操作历史
