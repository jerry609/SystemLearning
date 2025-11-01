package main

import "fmt"

// 适配器模式示例：接口适配
// 本示例展示如何使用适配器模式统一不同的接口

// Target - 目标接口（统一的媒体播放器接口）
type MediaPlayer interface {
	Play(audioType string, fileName string)
}

// Adaptee - 被适配者接口（高级媒体播放器）
type AdvancedMediaPlayer interface {
	PlayVLC(fileName string)
	PlayMP4(fileName string)
}

// VLCPlayer - VLC 播放器（具体的被适配者）
type VLCPlayer struct{}

func (v *VLCPlayer) PlayVLC(fileName string) {
	fmt.Printf("Playing VLC file: %s\n", fileName)
}

func (v *VLCPlayer) PlayMP4(fileName string) {
	// VLC 播放器不支持 MP4
}

// MP4Player - MP4 播放器（具体的被适配者）
type MP4Player struct{}

func (m *MP4Player) PlayVLC(fileName string) {
	// MP4 播放器不支持 VLC
}

func (m *MP4Player) PlayMP4(fileName string) {
	fmt.Printf("Playing MP4 file: %s\n", fileName)
}

// MediaAdapter - 媒体适配器
// 将 AdvancedMediaPlayer 接口适配到 MediaPlayer 接口
type MediaAdapter struct {
	advancedPlayer AdvancedMediaPlayer
}

func NewMediaAdapter(audioType string) *MediaAdapter {
	var player AdvancedMediaPlayer

	if audioType == "vlc" {
		player = &VLCPlayer{}
	} else if audioType == "mp4" {
		player = &MP4Player{}
	}

	return &MediaAdapter{advancedPlayer: player}
}

func (a *MediaAdapter) Play(audioType string, fileName string) {
	if audioType == "vlc" {
		a.advancedPlayer.PlayVLC(fileName)
	} else if audioType == "mp4" {
		a.advancedPlayer.PlayMP4(fileName)
	}
}

// AudioPlayer - 音频播放器（使用适配器）
type AudioPlayer struct{}

func (p *AudioPlayer) Play(audioType string, fileName string) {
	// 内置支持 MP3 格式
	if audioType == "mp3" {
		fmt.Printf("Playing MP3 file: %s\n", fileName)
	} else if audioType == "vlc" || audioType == "mp4" {
		// 使用适配器播放其他格式
		adapter := NewMediaAdapter(audioType)
		adapter.Play(audioType, fileName)
	} else {
		fmt.Printf("Invalid media type: %s. Format not supported.\n", audioType)
	}
}

func main() {
	fmt.Println("=== 接口适配器示例 ===\n")

	player := &AudioPlayer{}

	fmt.Println("1. 播放 MP3 文件（内置支持）:")
	player.Play("mp3", "song.mp3")

	fmt.Println("\n2. 播放 VLC 文件（通过适配器）:")
	player.Play("vlc", "movie.vlc")

	fmt.Println("\n3. 播放 MP4 文件（通过适配器）:")
	player.Play("mp4", "video.mp4")

	fmt.Println("\n4. 播放不支持的格式:")
	player.Play("avi", "video.avi")

	fmt.Println("\n=== 示例结束 ===")
}
