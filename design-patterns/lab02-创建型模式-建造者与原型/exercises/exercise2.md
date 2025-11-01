# 练习 2: 实现游戏角色克隆系统

## 难度
⭐⭐ (中等)

## 学习目标
- 掌握原型模式的实现
- 理解深拷贝和浅拷贝的区别
- 学会处理复杂对象的克隆
- 实践对象状态管理

## 问题描述

实现一个游戏角色克隆系统，支持创建角色的完整副本。系统应该能够正确处理角色的所有属性，包括基本属性、装备、技能和背包物品。克隆后的角色应该是完全独立的，修改克隆角色不应影响原始角色。

## 功能要求

1. **角色基本属性**
   - 名称（Name）
   - 等级（Level）
   - 生命值（Health）
   - 魔法值（Mana）
   - 攻击力（Attack）
   - 防御力（Defense）

2. **复杂属性**
   - 技能列表（Skills）- 切片类型
   - 装备（Equipment）- map 类型
   - 背包物品（Inventory）- 结构体切片
   - 属性加成（Buffs）- 结构体切片

3. **克隆功能**
   - 实现深拷贝方法
   - 实现浅拷贝方法（用于对比）
   - 支持使用 gob 序列化的通用克隆方法

4. **验证功能**
   - 验证克隆对象的独立性
   - 提供对比方法显示差异

## 数据结构

```go
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
```

## 输入输出示例

### 示例 1: 基本克隆
**代码**:
```go
warrior := &Character{
    Name:    "Warrior",
    Level:   10,
    Health:  100,
    Mana:    50,
    Attack:  30,
    Defense: 20,
    Skills: []Skill{
        {Name: "Slash", Level: 5, Cooldown: 2, Damage: 50},
        {Name: "Block", Level: 3, Cooldown: 5, Damage: 0},
    },
    Equipment: map[string]Equipment{
        "weapon": {Name: "Iron Sword", Type: "weapon", Attack: 15, Defense: 0, Rarity: "common"},
        "armor":  {Name: "Plate Armor", Type: "armor", Attack: 0, Defense: 25, Rarity: "rare"},
    },
    Inventory: []Item{
        {Name: "Health Potion", Quantity: 5, Type: "consumable"},
        {Name: "Mana Potion", Quantity: 3, Type: "consumable"},
    },
}

// 深拷贝
clone := warrior.DeepCopy()

// 修改克隆对象
clone.Name = "Elite Warrior"
clone.Level = 15
clone.Skills[0].Damage = 80
clone.Equipment["weapon"] = Equipment{Name: "Steel Sword", Type: "weapon", Attack: 25, Defense: 0, Rarity: "rare"}
clone.Inventory[0].Quantity = 10

// 验证独立性
fmt.Println("Original:", warrior.Name, warrior.Level, warrior.Skills[0].Damage)
fmt.Println("Clone:", clone.Name, clone.Level, clone.Skills[0].Damage)
```

**输出**:
```
Original: Warrior 10 50
Clone: Elite Warrior 15 80
✅ 克隆对象完全独立，修改不会相互影响
```

### 示例 2: 浅拷贝对比
**代码**:
```go
original := CreateWarrior()
shallowClone := original.ShallowCopy()

// 修改浅拷贝的切片元素
shallowClone.Skills[0].Damage = 100

fmt.Println("Original skill damage:", original.Skills[0].Damage)
fmt.Println("Shallow clone skill damage:", shallowClone.Skills[0].Damage)
```

**输出**:
```
Original skill damage: 100
Shallow clone skill damage: 100
❌ 浅拷贝共享引用类型数据，修改会相互影响
```

### 示例 3: 批量克隆
**代码**:
```go
// 创建原型
goblinPrototype := CreateGoblin()

// 克隆多个敌人
enemies := make([]*Character, 10)
for i := 0; i < 10; i++ {
    enemies[i] = goblinPrototype.DeepCopy()
    enemies[i].Name = fmt.Sprintf("Goblin-%d", i+1)
    enemies[i].Health += rand.Intn(20)
}

fmt.Printf("Created %d goblins from prototype\n", len(enemies))
```

**输出**:
```
Created 10 goblins from prototype
Goblin-1: Health=105
Goblin-2: Health=112
...
```

## 提示

💡 **提示 1**: 深拷贝需要递归复制所有引用类型
```go
func (c *Character) DeepCopy() *Character {
    // 复制切片
    skillsCopy := make([]Skill, len(c.Skills))
    for i, skill := range c.Skills {
        skillsCopy[i] = Skill{
            Name:     skill.Name,
            Level:    skill.Level,
            Cooldown: skill.Cooldown,
            Damage:   skill.Damage,
        }
    }
    
    // 复制 map
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
    
    // ... 复制其他字段
}
```

💡 **提示 2**: 使用 gob 实现通用克隆
```go
func DeepCopyGob(src, dst interface{}) error {
    var buf bytes.Buffer
    if err := gob.NewEncoder(&buf).Encode(src); err != nil {
        return err
    }
    return gob.NewDecoder(&buf).Decode(dst)
}
```

💡 **提示 3**: 提供验证方法
```go
func (c *Character) Equals(other *Character) bool {
    // 比较所有字段
    return c.Name == other.Name &&
           c.Level == other.Level &&
           // ... 其他字段
}
```

💡 **提示 4**: 考虑性能优化
```go
// 对于频繁克隆的场景，考虑对象池
type CharacterPool struct {
    pool sync.Pool
}

func (p *CharacterPool) Get() *Character {
    if c := p.pool.Get(); c != nil {
        return c.(*Character)
    }
    return &Character{}
}

func (p *CharacterPool) Put(c *Character) {
    // 重置对象
    c.Reset()
    p.pool.Put(c)
}
```

## 评分标准

- [ ] **功能完整性 (40%)**
  - 实现深拷贝方法
  - 实现浅拷贝方法
  - 正确处理所有数据类型
  - 克隆对象完全独立

- [ ] **代码质量 (30%)**
  - 代码结构清晰
  - 命名规范
  - 适当的注释
  - 错误处理

- [ ] **设计模式应用 (20%)**
  - 正确使用原型模式
  - 提供清晰的 Clone 接口
  - 考虑性能优化

- [ ] **测试覆盖 (10%)**
  - 验证深拷贝的独立性
  - 对比深拷贝和浅拷贝
  - 测试边界情况

## 扩展挑战

如果你完成了基本要求，可以尝试以下扩展功能：

1. **实现角色模板系统**
   ```go
   type CharacterTemplate struct {
       templates map[string]*Character
   }
   
   func (t *CharacterTemplate) Register(name string, prototype *Character) {
       t.templates[name] = prototype
   }
   
   func (t *CharacterTemplate) Create(name string) *Character {
       if prototype, ok := t.templates[name]; ok {
           return prototype.DeepCopy()
       }
       return nil
   }
   ```

2. **实现撤销/重做功能**
   ```go
   type CharacterHistory struct {
       snapshots []*Character
       current   int
   }
   
   func (h *CharacterHistory) Save(c *Character) {
       snapshot := c.DeepCopy()
       h.snapshots = append(h.snapshots[:h.current+1], snapshot)
       h.current++
   }
   
   func (h *CharacterHistory) Undo() *Character {
       if h.current > 0 {
           h.current--
           return h.snapshots[h.current].DeepCopy()
       }
       return nil
   }
   ```

3. **实现增量克隆**
   ```go
   type CloneOption func(*Character)
   
   func WithLevel(level int) CloneOption {
       return func(c *Character) {
           c.Level = level
       }
   }
   
   func (c *Character) CloneWith(opts ...CloneOption) *Character {
       clone := c.DeepCopy()
       for _, opt := range opts {
           opt(clone)
       }
       return clone
   }
   ```

4. **性能基准测试**
   ```go
   func BenchmarkDeepCopyManual(b *testing.B) {
       c := CreateWarrior()
       for i := 0; i < b.N; i++ {
           _ = c.DeepCopy()
       }
   }
   
   func BenchmarkDeepCopyGob(b *testing.B) {
       c := CreateWarrior()
       for i := 0; i < b.N; i++ {
           var clone Character
           _ = DeepCopyGob(c, &clone)
       }
   }
   ```

## 参考资源

- [原型模式详解](../theory/02-prototype.md)
- [Go encoding/gob 包文档](https://pkg.go.dev/encoding/gob)
- [深拷贝 vs 浅拷贝](https://en.wikipedia.org/wiki/Object_copying)

## 提交要求

1. 实现 `Character` 结构体和相关方法
2. 实现深拷贝和浅拷贝方法
3. 编写测试用例验证独立性
4. 提供性能对比
5. 添加必要的注释和文档

---

**预计完成时间**: 1-2 小时  
**难度评估**: 中等  
**重点考察**: 原型模式、深拷贝、对象克隆
