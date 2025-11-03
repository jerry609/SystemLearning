package main

import (
	"fmt"
	"math/rand"
	"time"
)

// 游戏角色状态机示例
// 本示例展示了状态模式在游戏开发中的应用

// GameState 游戏状态接口
type GameState interface {
	// Attack 攻击
	Attack(player *Player) string
	// TakeDamage 受到伤害
	TakeDamage(player *Player, damage int)
	// UsePotion 使用药水
	UsePotion(player *Player)
	// Rest 休息
	Rest(player *Player)
	// String 状态名称
	String() string
}

// Player 玩家角色
type Player struct {
	Name      string
	HP        int
	MaxHP     int
	state     GameState
	poisonTurns int // 中毒剩余回合数
}

func NewPlayer(name string, maxHP int) *Player {
	player := &Player{
		Name:  name,
		HP:    maxHP,
		MaxHP: maxHP,
	}
	player.SetState(&NormalState{})
	return player
}

func (p *Player) SetState(state GameState) {
	if p.state != nil && p.state.String() != state.String() {
		fmt.Printf("[%s] 状态变更: %s -> %s\n", p.Name, p.state.String(), state.String())
	}
	p.state = state
}

func (p *Player) GetState() string {
	return p.state.String()
}

func (p *Player) Attack() string {
	return p.state.Attack(p)
}

func (p *Player) TakeDamage(damage int) {
	p.state.TakeDamage(p, damage)
}

func (p *Player) UsePotion() {
	p.state.UsePotion(p)
}

func (p *Player) Rest() {
	p.state.Rest(p)
}

func (p *Player) PrintStatus() {
	fmt.Printf("[%s] HP: %d/%d, 状态: %s", p.Name, p.HP, p.MaxHP, p.GetState())
	if p.poisonTurns > 0 {
		fmt.Printf(" (中毒剩余 %d 回合)", p.poisonTurns)
	}
	fmt.Println()
}

// NormalState 正常状态
type NormalState struct{}

func (s *NormalState) Attack(player *Player) string {
	damage := 20 + rand.Intn(10)
	return fmt.Sprintf("[%s] 发动普通攻击，造成 %d 点伤害", player.Name, damage)
}

func (s *NormalState) TakeDamage(player *Player, damage int) {
	player.HP -= damage
	fmt.Printf("[%s] 受到 %d 点伤害，剩余 HP: %d\n", player.Name, damage, player.HP)

	if player.HP <= 0 {
		player.HP = 0
		player.SetState(&DeadState{})
	} else if player.HP < player.MaxHP/3 {
		player.SetState(&InjuredState{})
	}
}

func (s *NormalState) UsePotion(player *Player) {
	heal := 30
	player.HP += heal
	if player.HP > player.MaxHP {
		player.HP = player.MaxHP
	}
	fmt.Printf("[%s] 使用药水，恢复 %d HP，当前 HP: %d\n", player.Name, heal, player.HP)
}

func (s *NormalState) Rest(player *Player) {
	heal := 10
	player.HP += heal
	if player.HP > player.MaxHP {
		player.HP = player.MaxHP
	}
	fmt.Printf("[%s] 休息恢复，恢复 %d HP，当前 HP: %d\n", player.Name, heal, player.HP)
}

func (s *NormalState) String() string {
	return "正常"
}

// InjuredState 受伤状态
type InjuredState struct{}

func (s *InjuredState) Attack(player *Player) string {
	damage := 10 + rand.Intn(5)
	return fmt.Sprintf("[%s] 带伤攻击，造成 %d 点伤害（攻击力下降）", player.Name, damage)
}

func (s *InjuredState) TakeDamage(player *Player, damage int) {
	// 受伤状态下受到的伤害增加
	actualDamage := int(float64(damage) * 1.2)
	player.HP -= actualDamage
	fmt.Printf("[%s] 受到 %d 点伤害（受伤状态伤害增加），剩余 HP: %d\n", player.Name, actualDamage, player.HP)

	if player.HP <= 0 {
		player.HP = 0
		player.SetState(&DeadState{})
	}
}

func (s *InjuredState) UsePotion(player *Player) {
	heal := 40
	player.HP += heal
	if player.HP > player.MaxHP {
		player.HP = player.MaxHP
	}
	fmt.Printf("[%s] 使用药水，恢复 %d HP，当前 HP: %d\n", player.Name, heal, player.HP)

	if player.HP >= player.MaxHP/3 {
		player.SetState(&NormalState{})
	}
}

func (s *InjuredState) Rest(player *Player) {
	heal := 15
	player.HP += heal
	if player.HP > player.MaxHP {
		player.HP = player.MaxHP
	}
	fmt.Printf("[%s] 休息恢复，恢复 %d HP，当前 HP: %d\n", player.Name, heal, player.HP)

	if player.HP >= player.MaxHP/3 {
		player.SetState(&NormalState{})
	}
}

func (s *InjuredState) String() string {
	return "受伤"
}

// PoisonedState 中毒状态
type PoisonedState struct{}

func (s *PoisonedState) Attack(player *Player) string {
	damage := 15 + rand.Intn(5)
	
	// 中毒持续伤害
	poisonDamage := 5
	player.HP -= poisonDamage
	fmt.Printf("[%s] 中毒持续伤害 %d，剩余 HP: %d\n", player.Name, poisonDamage, player.HP)
	
	player.poisonTurns--
	if player.poisonTurns <= 0 {
		fmt.Printf("[%s] 中毒状态解除\n", player.Name)
		if player.HP < player.MaxHP/3 {
			player.SetState(&InjuredState{})
		} else {
			player.SetState(&NormalState{})
		}
	}
	
	if player.HP <= 0 {
		player.HP = 0
		player.SetState(&DeadState{})
		return fmt.Sprintf("[%s] 因中毒而死亡", player.Name)
	}
	
	return fmt.Sprintf("[%s] 中毒攻击，造成 %d 点伤害（攻击力下降）", player.Name, damage)
}

func (s *PoisonedState) TakeDamage(player *Player, damage int) {
	player.HP -= damage
	fmt.Printf("[%s] 受到 %d 点伤害，剩余 HP: %d\n", player.Name, damage, player.HP)

	if player.HP <= 0 {
		player.HP = 0
		player.SetState(&DeadState{})
	}
}

func (s *PoisonedState) UsePotion(player *Player) {
	// 解毒并恢复
	heal := 25
	player.HP += heal
	if player.HP > player.MaxHP {
		player.HP = player.MaxHP
	}
	player.poisonTurns = 0
	fmt.Printf("[%s] 使用药水，解除中毒并恢复 %d HP，当前 HP: %d\n", player.Name, heal, player.HP)

	if player.HP < player.MaxHP/3 {
		player.SetState(&InjuredState{})
	} else {
		player.SetState(&NormalState{})
	}
}

func (s *PoisonedState) Rest(player *Player) {
	// 休息时中毒持续伤害
	poisonDamage := 5
	player.HP -= poisonDamage
	fmt.Printf("[%s] 休息时中毒持续伤害 %d，剩余 HP: %d\n", player.Name, poisonDamage, player.HP)
	
	player.poisonTurns--
	if player.poisonTurns <= 0 {
		fmt.Printf("[%s] 中毒状态解除\n", player.Name)
		if player.HP < player.MaxHP/3 {
			player.SetState(&InjuredState{})
		} else {
			player.SetState(&NormalState{})
		}
	}
	
	if player.HP <= 0 {
		player.HP = 0
		player.SetState(&DeadState{})
	}
}

func (s *PoisonedState) String() string {
	return "中毒"
}

// StunnedState 眩晕状态
type StunnedState struct {
	duration int
}

func (s *StunnedState) Attack(player *Player) string {
	s.duration--
	if s.duration <= 0 {
		fmt.Printf("[%s] 眩晕状态解除\n", player.Name)
		if player.HP < player.MaxHP/3 {
			player.SetState(&InjuredState{})
		} else {
			player.SetState(&NormalState{})
		}
	}
	return fmt.Sprintf("[%s] 眩晕中，无法攻击", player.Name)
}

func (s *StunnedState) TakeDamage(player *Player, damage int) {
	// 眩晕状态下受到的伤害增加
	actualDamage := int(float64(damage) * 1.5)
	player.HP -= actualDamage
	fmt.Printf("[%s] 受到 %d 点伤害（眩晕状态伤害大幅增加），剩余 HP: %d\n", player.Name, actualDamage, player.HP)

	if player.HP <= 0 {
		player.HP = 0
		player.SetState(&DeadState{})
	}
}

func (s *StunnedState) UsePotion(player *Player) {
	fmt.Printf("[%s] 眩晕中，无法使用药水\n", player.Name)
}

func (s *StunnedState) Rest(player *Player) {
	fmt.Printf("[%s] 眩晕中，无法休息\n", player.Name)
	s.duration--
	if s.duration <= 0 {
		fmt.Printf("[%s] 眩晕状态解除\n", player.Name)
		if player.HP < player.MaxHP/3 {
			player.SetState(&InjuredState{})
		} else {
			player.SetState(&NormalState{})
		}
	}
}

func (s *StunnedState) String() string {
	return "眩晕"
}

// DeadState 死亡状态
type DeadState struct{}

func (s *DeadState) Attack(player *Player) string {
	return fmt.Sprintf("[%s] 已死亡，无法攻击", player.Name)
}

func (s *DeadState) TakeDamage(player *Player, damage int) {
	fmt.Printf("[%s] 已死亡，无法受到伤害\n", player.Name)
}

func (s *DeadState) UsePotion(player *Player) {
	fmt.Printf("[%s] 已死亡，无法使用药水\n", player.Name)
}

func (s *DeadState) Rest(player *Player) {
	fmt.Printf("[%s] 已死亡，无法休息\n", player.Name)
}

func (s *DeadState) String() string {
	return "死亡"
}

func main() {
	fmt.Println("=== 状态模式示例 - 游戏角色状态 ===\n")
	
	rand.Seed(time.Now().UnixNano())

	// 场景1: 正常战斗流程
	fmt.Println("--- 场景1: 正常战斗流程 ---")
	player1 := NewPlayer("勇者", 100)
	player1.PrintStatus()
	fmt.Println()

	fmt.Println(player1.Attack())
	player1.PrintStatus()
	fmt.Println()

	player1.TakeDamage(25)
	player1.PrintStatus()
	fmt.Println()

	player1.TakeDamage(30)
	player1.PrintStatus()
	fmt.Println()

	fmt.Println(player1.Attack())
	player1.PrintStatus()
	fmt.Println()

	player1.UsePotion()
	player1.PrintStatus()
	fmt.Println()

	// 场景2: 中毒状态
	fmt.Println("\n--- 场景2: 中毒状态 ---")
	player2 := NewPlayer("法师", 80)
	player2.PrintStatus()
	fmt.Println()

	// 模拟被毒攻击
	player2.poisonTurns = 3
	player2.SetState(&PoisonedState{})
	player2.PrintStatus()
	fmt.Println()

	for i := 0; i < 3; i++ {
		fmt.Printf("回合 %d:\n", i+1)
		fmt.Println(player2.Attack())
		player2.PrintStatus()
		fmt.Println()
	}

	// 场景3: 使用药水解毒
	fmt.Println("\n--- 场景3: 使用药水解毒 ---")
	player3 := NewPlayer("刺客", 90)
	player3.poisonTurns = 5
	player3.SetState(&PoisonedState{})
	player3.PrintStatus()
	fmt.Println()

	fmt.Println(player3.Attack())
	player3.PrintStatus()
	fmt.Println()

	player3.UsePotion()
	player3.PrintStatus()
	fmt.Println()

	// 场景4: 眩晕状态
	fmt.Println("\n--- 场景4: 眩晕状态 ---")
	player4 := NewPlayer("战士", 120)
	player4.PrintStatus()
	fmt.Println()

	// 模拟被眩晕
	player4.SetState(&StunnedState{duration: 2})
	player4.PrintStatus()
	fmt.Println()

	for i := 0; i < 3; i++ {
		fmt.Printf("回合 %d:\n", i+1)
		fmt.Println(player4.Attack())
		player4.PrintStatus()
		fmt.Println()
	}

	// 场景5: 死亡状态
	fmt.Println("\n--- 场景5: 死亡状态 ---")
	player5 := NewPlayer("弓箭手", 70)
	player5.PrintStatus()
	fmt.Println()

	player5.TakeDamage(50)
	player5.PrintStatus()
	fmt.Println()

	player5.TakeDamage(30)
	player5.PrintStatus()
	fmt.Println()

	fmt.Println(player5.Attack())
	player5.UsePotion()
	player5.Rest()
	fmt.Println()

	fmt.Println("=== 示例结束 ===")
}
