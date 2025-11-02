package main

import (
	"fmt"
	"time"
)

// 外观模式示例：家庭影院系统
// 本示例展示了如何使用外观模式简化复杂子系统的使用

// 子系统 1: DVD 播放器
type DVDPlayer struct{}

func (d *DVDPlayer) On() {
	fmt.Println("  [DVD Player] 开机")
}

func (d *DVDPlayer) Off() {
	fmt.Println("  [DVD Player] 关机")
}

func (d *DVDPlayer) Play(movie string) {
	fmt.Printf("  [DVD Player] 播放电影: %s\n", movie)
}

func (d *DVDPlayer) Stop() {
	fmt.Println("  [DVD Player] 停止播放")
}

func (d *DVDPlayer) Eject() {
	fmt.Println("  [DVD Player] 弹出光盘")
}

// 子系统 2: 投影仪
type Projector struct{}

func (p *Projector) On() {
	fmt.Println("  [Projector] 开机")
}

func (p *Projector) Off() {
	fmt.Println("  [Projector] 关机")
}

func (p *Projector) WideScreenMode() {
	fmt.Println("  [Projector] 设置为宽屏模式")
}

func (p *Projector) NormalMode() {
	fmt.Println("  [Projector] 设置为普通模式")
}

// 子系统 3: 音响系统
type SoundSystem struct{}

func (s *SoundSystem) On() {
	fmt.Println("  [Sound System] 开机")
}

func (s *SoundSystem) Off() {
	fmt.Println("  [Sound System] 关机")
}

func (s *SoundSystem) SetVolume(level int) {
	fmt.Printf("  [Sound System] 设置音量为 %d\n", level)
}

func (s *SoundSystem) SetSurroundSound() {
	fmt.Println("  [Sound System] 开启环绕声")
}

// 子系统 4: 灯光系统
type Lights struct{}

func (l *Lights) On() {
	fmt.Println("  [Lights] 开灯")
}

func (l *Lights) Off() {
	fmt.Println("  [Lights] 关灯")
}

func (l *Lights) Dim(level int) {
	fmt.Printf("  [Lights] 调暗灯光至 %d%%\n", level)
}

// 子系统 5: 屏幕
type Screen struct{}

func (s *Screen) Down() {
	fmt.Println("  [Screen] 放下屏幕")
}

func (s *Screen) Up() {
	fmt.Println("  [Screen] 收起屏幕")
}

// 子系统 6: 爆米花机
type PopcornMaker struct{}

func (p *PopcornMaker) On() {
	fmt.Println("  [Popcorn Maker] 开机")
}

func (p *PopcornMaker) Off() {
	fmt.Println("  [Popcorn Maker] 关机")
}

func (p *PopcornMaker) Pop() {
	fmt.Println("  [Popcorn Maker] 开始制作爆米花")
	time.Sleep(100 * time.Millisecond)
	fmt.Println("  [Popcorn Maker] 爆米花制作完成！")
}

// 外观类：家庭影院
type HomeTheaterFacade struct {
	dvd        *DVDPlayer
	projector  *Projector
	sound      *SoundSystem
	lights     *Lights
	screen     *Screen
	popcorn    *PopcornMaker
}

func NewHomeTheaterFacade() *HomeTheaterFacade {
	return &HomeTheaterFacade{
		dvd:       &DVDPlayer{},
		projector: &Projector{},
		sound:     &SoundSystem{},
		lights:    &Lights{},
		screen:    &Screen{},
		popcorn:   &PopcornMaker{},
	}
}

// WatchMovie 观看电影（简化的高层接口）
func (h *HomeTheaterFacade) WatchMovie(movie string) {
	fmt.Println("\n🎬 准备观看电影...")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	h.popcorn.On()
	h.popcorn.Pop()
	h.lights.Dim(10)
	h.screen.Down()
	h.projector.On()
	h.projector.WideScreenMode()
	h.sound.On()
	h.sound.SetVolume(5)
	h.sound.SetSurroundSound()
	h.dvd.On()
	h.dvd.Play(movie)
	
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ 一切就绪，尽情享受电影吧！\n")
}

// EndMovie 结束电影（简化的高层接口）
func (h *HomeTheaterFacade) EndMovie() {
	fmt.Println("\n🛑 结束观影...")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	
	h.popcorn.Off()
	h.dvd.Stop()
	h.dvd.Eject()
	h.dvd.Off()
	h.sound.Off()
	h.projector.Off()
	h.screen.Up()
	h.lights.On()
	
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ 影院系统已关闭\n")
}

// PauseMovie 暂停电影
func (h *HomeTheaterFacade) PauseMovie() {
	fmt.Println("\n⏸️  暂停电影...")
	h.dvd.Stop()
	h.lights.Dim(50)
	fmt.Println("✅ 已暂停\n")
}

// ResumeMovie 继续播放
func (h *HomeTheaterFacade) ResumeMovie(movie string) {
	fmt.Println("\n▶️  继续播放...")
	h.lights.Dim(10)
	h.dvd.Play(movie)
	fmt.Println("✅ 继续播放\n")
}

func main() {
	fmt.Println("=== 外观模式示例：家庭影院系统 ===")

	// 创建家庭影院外观
	homeTheater := NewHomeTheaterFacade()

	// 使用简化的接口观看电影
	// 不需要了解各个子系统的复杂操作
	homeTheater.WatchMovie("《黑客帝国》")

	// 模拟观影过程
	fmt.Println("... 正在观看电影 ...")
	time.Sleep(200 * time.Millisecond)

	// 暂停电影
	homeTheater.PauseMovie()

	// 模拟休息
	fmt.Println("... 休息一下 ...")
	time.Sleep(200 * time.Millisecond)

	// 继续播放
	homeTheater.ResumeMovie("《黑客帝国》")

	// 模拟继续观影
	fmt.Println("... 继续观看电影 ...")
	time.Sleep(200 * time.Millisecond)

	// 结束电影
	homeTheater.EndMovie()

	fmt.Println("=== 示例结束 ===")

	// 对比：如果没有外观模式，客户端需要这样做：
	fmt.Println("\n💡 对比：没有外观模式的情况")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("客户端需要手动操作每个子系统：")
	fmt.Println("  1. 创建所有子系统对象")
	fmt.Println("  2. 按正确顺序调用每个子系统的方法")
	fmt.Println("  3. 记住复杂的操作步骤")
	fmt.Println("  4. 处理各个子系统之间的依赖关系")
	fmt.Println("\n使用外观模式后：")
	fmt.Println("  ✅ 只需调用 WatchMovie() 和 EndMovie()")
	fmt.Println("  ✅ 不需要了解子系统的细节")
	fmt.Println("  ✅ 代码更简洁、更易维护")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
}

// 输出示例：
// === 外观模式示例：家庭影院系统 ===
//
// 🎬 准备观看电影...
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//   [Popcorn Maker] 开机
//   [Popcorn Maker] 开始制作爆米花
//   [Popcorn Maker] 爆米花制作完成！
//   [Lights] 调暗灯光至 10%
//   [Screen] 放下屏幕
//   [Projector] 开机
//   [Projector] 设置为宽屏模式
//   [Sound System] 开机
//   [Sound System] 设置音量为 5
//   [Sound System] 开启环绕声
//   [DVD Player] 开机
//   [DVD Player] 播放电影: 《黑客帝国》
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ✅ 一切就绪，尽情享受电影吧！
//
// ... 正在观看电影 ...
//
// ⏸️  暂停电影...
//   [DVD Player] 停止播放
//   [Lights] 调暗灯光至 50%
// ✅ 已暂停
//
// ... 休息一下 ...
//
// ▶️  继续播放...
//   [Lights] 调暗灯光至 10%
//   [DVD Player] 播放电影: 《黑客帝国》
// ✅ 继续播放
//
// ... 继续观看电影 ...
//
// 🛑 结束观影...
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//   [Popcorn Maker] 关机
//   [DVD Player] 停止播放
//   [DVD Player] 弹出光盘
//   [DVD Player] 关机
//   [Sound System] 关机
//   [Projector] 关机
//   [Screen] 收起屏幕
//   [Lights] 开灯
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ✅ 影院系统已关闭
//
// === 示例结束 ===
