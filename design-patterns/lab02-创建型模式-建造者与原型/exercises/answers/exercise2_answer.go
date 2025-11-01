package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
)

// 练习 2: 游戏角色克隆系统 - 参考答案
//
// 设计思路:
// 1. 定义完整的角色数据结构
// 2. 实现深拷贝方法，递归复制所有引用类型
// 3. 实现浅拷贝方法用于对比
// 4. 提供通用的 gob 序列化克隆方法
//
// 使用的设计模式: 原型模式
// 模式应用位置: Character 的 DeepCopy 和 ShallowCopy 方法

// Skill 表示技能
type Skill struct {
	Name     string
	Level    int
	Cooldown int
	Damage   int
}

// Equipment 表示装备
type Equipment struct {
	Name    string
	Type    string
	Attack  int
	Defense int
	Rarity  string
}

// Item 表示物品
type Item struct {
	Name     string
	Quantity int
	Type     string
}

// Buff 表示增益效果
type Buff struct {
	Name     string
	Duration int
	Effect   string
	Value    int
}

// Character 表示游戏角色
type Character struct {
	Name      string
	Level     int
	Health    int
	Mana      int
	Attack    int
	Defense   int
	Skills    []Skill
	Equipment map[string]Equipment
	Inventory []Item
	Buffs     []Buff
}

// DeepCopy 执行深拷贝
func (c *Character) DeepCopy() *Character {
	// 复制 Skills 切片
	skillsCopy := make([]Skill, len(c.Skills))
	for i, skill := range c.Skills {
		skillsCopy[i] = Skill{
			Name:     skill.Name,
			Level:    skill.Level,
			Cooldown: skill.Cooldown,
			Damage:   skill.Damage,
		}
	}

	// 复制 Equipment map
	equipmentCopy := make(map[string]Equipment)
	for k, v := range c.Equipment {
		equipmentCopy[k] = Equipment{
			Name:    v.Name,
			Type:    v.Type,
			Attack:  v.Attack,
			Defense: v.Defense,
			Rarity:  v.Rarity,
		}
	}

	// 复制 Inventory 切片
	inventoryCopy := make([]Item, len(c.Inventory))
	for i, item := range c.Inventory {
		inventoryCopy[i] = Item{
			Name:     item.Name,
			Quantity: item.Quantity,
			Type:     item.Type,
		}
	}

	// 复制 Buffs 切片
	buffsCopy := make([]Buff, len(c.Buffs))
	for i, buff := range c.Buffs {
		buffsCopy[i] = Buff{
			Name:     buff.Name,
			Duration: buff.Duration,
			Effect:   buff.Effect,
			Value:    buff.Value,
		}
	}

	return &Character{
		Name:      c.Name,
		Level:     c.Level,
		Health:    c.Health,
		Mana:      c.Mana,
		Attack:    c.Attack,
		Defense:   c.Defense,
		Skills:    skillsCopy,
		Equipment: equipmentCopy,
		Inventory: inventoryCopy,
		Buffs:     buffsCopy,
	}
}

// ShallowCopy 执行浅拷贝（用于对比）
func (c *Character) ShallowCopy() *Character {
	return &Character{
		Name:      c.Name,
		Level:     c.Level,
		Health:    c.Health,
		Mana:      c.Mana,
		Attack:    c.Attack,
		Defense:   c.Defense,
		Skills:    c.Skills,    // 只复制切片头
		Equipment: c.Equipment, // 只复制 map 引用
		Inventory: c.Inventory, // 只复制切片头
		Buffs:     c.Buffs,     // 只复制切片头
	}
}

// String 返回角色的字符串表示
func (c *Character) String() string {
	return fmt.Sprintf("Character{Name: %s, Level: %d, Health: %d, Mana: %d, Attack: %d, Defense: %d, Skills: %d, Equipment: %d, Inventory: %d, Buffs: %d}",
		c.Name, c.Level, c.Health, c.Mana, c.Attack, c.Defense,
		len(c.Skills), len(c.Equipment), len(c.Inventory), len(c.Buffs))
}

// DetailedString 返回角色的详细字符串表示
func (c *Character) DetailedString() string {
	result := fmt.Sprintf("=== %s (Level %d) ===\n", c.Name, c.Level)
	result += fmt.Sprintf("Health: %d, Mana: %d\n", c.Health, c.Mana)
	result += fmt.Sprintf("Attack: %d, Defense: %d\n", c.Attack, c.Defense)

	result += "\nSkills:\n"
	for _, skill := range c.Skills {
		result += fmt.Sprintf("  - %s (Lv.%d): Damage=%d, Cooldown=%d\n",
			skill.Name, skill.Level, skill.Damage, skill.Cooldown)
	}

	result += "\nEquipment:\n"
	for slot, equip := range c.Equipment {
		result += fmt.Sprintf("  - %s: %s (%s) [%s]\n",
			slot, equip.Name, equip.Type, equip.Rarity)
	}

	result += "\nInventory:\n"
	for _, item := range c.Inventory {
		result += fmt.Sprintf("  - %s x%d (%s)\n",
			item.Name, item.Quantity, item.Type)
	}

	if len(c.Buffs) > 0 {
		result += "\nBuffs:\n"
		for _, buff := range c.Buffs {
			result += fmt.Sprintf("  - %s: %s +%d (Duration: %d)\n",
				buff.Name, buff.Effect, buff.Value, buff.Duration)
		}
	}

	return result
}

// DeepCopyGob 使用 gob 序列化进行深拷贝
func DeepCopyGob(src, dst interface{}) error {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(src); err != nil {
		return err
	}
	return gob.NewDecoder(&buf).Decode(dst)
}

// CreateWarrior 创建一个战士角色
func CreateWarrior() *Character {
	return &Character{
		Name:    "Warrior",
		Level:   10,
		Health:  100,
		Mana:    50,
		Attack:  30,
		Defense: 20,
		Skills: []Skill{
			{Name: "Slash", Level: 5, Cooldown: 2, Damage: 50},
			{Name: "Block", Level: 3, Cooldown: 5, Damage: 0},
			{Name: "Charge", Level: 4, Cooldown: 10, Damage: 80},
		},
		Equipment: map[string]Equipment{
			"weapon": {Name: "Iron Sword", Type: "weapon", Attack: 15, Defense: 0, Rarity: "common"},
			"armor":  {Name: "Plate Armor", Type: "armor", Attack: 0, Defense: 25, Rarity: "rare"},
			"shield": {Name: "Iron Shield", Type: "shield", Attack: 0, Defense: 10, Rarity: "common"},
		},
		Inventory: []Item{
			{Name: "Health Potion", Quantity: 5, Type: "consumable"},
			{Name: "Mana Potion", Quantity: 3, Type: "consumable"},
			{Name: "Antidote", Quantity: 2, Type: "consumable"},
		},
		Buffs: []Buff{
			{Name: "Strength", Duration: 60, Effect: "Attack", Value: 10},
			{Name: "Defense Up", Duration: 120, Effect: "Defense", Value: 5},
		},
	}
}

// CreateMage 创建一个法师角色
func CreateMage() *Character {
	return &Character{
		Name:    "Mage",
		Level:   10,
		Health:  60,
		Mana:    120,
		Attack:  15,
		Defense: 10,
		Skills: []Skill{
			{Name: "Fireball", Level: 5, Cooldown: 3, Damage: 70},
			{Name: "Ice Bolt", Level: 4, Cooldown: 2, Damage: 50},
			{Name: "Lightning", Level: 6, Cooldown: 8, Damage: 100},
		},
		Equipment: map[string]Equipment{
			"weapon": {Name: "Magic Staff", Type: "weapon", Attack: 20, Defense: 0, Rarity: "rare"},
			"armor":  {Name: "Robe", Type: "armor", Attack: 0, Defense: 8, Rarity: "common"},
		},
		Inventory: []Item{
			{Name: "Mana Potion", Quantity: 10, Type: "consumable"},
			{Name: "Scroll of Teleport", Quantity: 2, Type: "scroll"},
		},
		Buffs: []Buff{
			{Name: "Magic Power", Duration: 90, Effect: "Attack", Value: 15},
		},
	}
}

func main() {
	fmt.Println("=== 游戏角色克隆系统 - 参考答案 ===\n")

	// 示例 1: 深拷贝验证
	fmt.Println("示例 1: 深拷贝验证")
	fmt.Println("-------------------")

	warrior := CreateWarrior()
	fmt.Println("原始角色:")
	fmt.Println(warrior.DetailedString())

	// 执行深拷贝
	warriorClone := warrior.DeepCopy()
	fmt.Println("\n克隆角色:")
	fmt.Println(warriorClone.DetailedString())

	// 修改克隆角色
	fmt.Println("\n修改克隆角色...")
	warriorClone.Name = "Elite Warrior"
	warriorClone.Level = 15
	warriorClone.Health = 150
	warriorClone.Skills[0].Damage = 80
	warriorClone.Equipment["weapon"] = Equipment{
		Name: "Steel Sword", Type: "weapon", Attack: 25, Defense: 0, Rarity: "rare",
	}
	warriorClone.Inventory[0].Quantity = 10

	fmt.Println("\n修改后:")
	fmt.Printf("原始角色: %s (Lv.%d) - Skill[0].Damage=%d, Inventory[0].Quantity=%d\n",
		warrior.Name, warrior.Level, warrior.Skills[0].Damage, warrior.Inventory[0].Quantity)
	fmt.Printf("克隆角色: %s (Lv.%d) - Skill[0].Damage=%d, Inventory[0].Quantity=%d\n",
		warriorClone.Name, warriorClone.Level, warriorClone.Skills[0].Damage, warriorClone.Inventory[0].Quantity)
	fmt.Println("✅ 深拷贝成功，修改克隆对象不影响原始对象！")
	fmt.Println()

	// 示例 2: 浅拷贝对比
	fmt.Println("\n示例 2: 浅拷贝对比")
	fmt.Println("-------------------")

	mage := CreateMage()
	fmt.Printf("原始法师: %s\n", mage)

	// 执行浅拷贝
	mageShallowClone := mage.ShallowCopy()
	fmt.Printf("浅拷贝法师: %s\n", mageShallowClone)

	// 修改浅拷贝的切片元素
	fmt.Println("\n修改浅拷贝的技能伤害...")
	mageShallowClone.Skills[0].Damage = 150

	fmt.Printf("原始法师技能伤害: %d\n", mage.Skills[0].Damage)
	fmt.Printf("浅拷贝法师技能伤害: %d\n", mageShallowClone.Skills[0].Damage)
	fmt.Println("❌ 浅拷贝共享引用类型数据，修改会相互影响！")
	fmt.Println()

	// 示例 3: 使用 gob 进行深拷贝
	fmt.Println("\n示例 3: 使用 gob 进行深拷贝")
	fmt.Println("-------------------")

	warrior2 := CreateWarrior()
	var warrior2Clone Character

	if err := DeepCopyGob(warrior2, &warrior2Clone); err != nil {
		fmt.Println("错误:", err)
	} else {
		fmt.Printf("原始角色: %s\n", warrior2)
		fmt.Printf("克隆角色: %s\n", &warrior2Clone)

		// 修改克隆角色
		warrior2Clone.Name = "Gob Warrior"
		warrior2Clone.Level = 20
		warrior2Clone.Skills[0].Damage = 100

		fmt.Println("\n修改后:")
		fmt.Printf("原始角色: %s (Lv.%d) - Skill[0].Damage=%d\n",
			warrior2.Name, warrior2.Level, warrior2.Skills[0].Damage)
		fmt.Printf("克隆角色: %s (Lv.%d) - Skill[0].Damage=%d\n",
			warrior2Clone.Name, warrior2Clone.Level, warrior2Clone.Skills[0].Damage)
		fmt.Println("✅ 使用 gob 实现深拷贝成功！")
	}
	fmt.Println()

	// 示例 4: 批量克隆（原型模式的典型应用）
	fmt.Println("\n示例 4: 批量克隆")
	fmt.Println("-------------------")

	// 创建敌人原型
	goblinPrototype := &Character{
		Name:    "Goblin",
		Level:   5,
		Health:  50,
		Mana:    20,
		Attack:  15,
		Defense: 5,
		Skills: []Skill{
			{Name: "Bite", Level: 1, Cooldown: 1, Damage: 20},
		},
		Equipment: map[string]Equipment{
			"weapon": {Name: "Club", Type: "weapon", Attack: 5, Defense: 0, Rarity: "common"},
		},
		Inventory: []Item{
			{Name: "Gold Coin", Quantity: 10, Type: "currency"},
		},
	}

	// 克隆多个敌人
	enemies := make([]*Character, 5)
	for i := 0; i < 5; i++ {
		enemies[i] = goblinPrototype.DeepCopy()
		enemies[i].Name = fmt.Sprintf("Goblin-%d", i+1)
		// 可以对每个克隆进行微调
		enemies[i].Health += i * 5
		enemies[i].Attack += i * 2
	}

	fmt.Printf("从原型创建了 %d 个敌人:\n", len(enemies))
	for _, enemy := range enemies {
		fmt.Printf("  - %s: Health=%d, Attack=%d\n", enemy.Name, enemy.Health, enemy.Attack)
	}
	fmt.Println("✅ 原型模式适合批量创建相似对象！")

	fmt.Println("\n=== 示例结束 ===")
}

// 可能的优化方向:
// 1. 实现对象池，复用角色对象
// 2. 添加增量克隆功能
// 3. 实现撤销/重做功能
// 4. 添加角色模板系统
// 5. 性能优化：对于频繁克隆的场景，考虑缓存
//
// 变体实现:
// 1. 使用接口定义 Cloneable，支持多态克隆
// 2. 实现写时复制（Copy-on-Write）优化
// 3. 添加克隆选项，支持部分克隆
