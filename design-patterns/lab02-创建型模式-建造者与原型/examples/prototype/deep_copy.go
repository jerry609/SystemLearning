package main

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"fmt"
)

// 深拷贝示例
// 本示例展示深拷贝的特点：递归复制所有字段，包括引用类型指向的对象

// Address 表示地址信息
type Address struct {
	City    string
	Country string
}

// Person 表示一个人
type Person struct {
	Name    string
	Age     int
	Address *Address
	Hobbies []string
}

// DeepCopy 执行深拷贝（手动实现）
func (p *Person) DeepCopy() *Person {
	// 复制 Address
	var addressCopy *Address
	if p.Address != nil {
		addressCopy = &Address{
			City:    p.Address.City,
			Country: p.Address.Country,
		}
	}

	// 复制 Hobbies 切片
	hobbiesCopy := make([]string, len(p.Hobbies))
	copy(hobbiesCopy, p.Hobbies)

	return &Person{
		Name:    p.Name,
		Age:     p.Age,
		Address: addressCopy,
		Hobbies: hobbiesCopy,
	}
}

// String 返回 Person 的字符串表示
func (p *Person) String() string {
	return fmt.Sprintf("Person{Name: %s, Age: %d, Address: %+v, Hobbies: %v}",
		p.Name, p.Age, p.Address, p.Hobbies)
}

// Document 表示一个文档
type Document struct {
	Title    string
	Content  string
	Tags     []string
	Metadata map[string]string
}

// DeepCopy 执行深拷贝（手动实现）
func (d *Document) DeepCopy() *Document {
	// 复制 Tags 切片
	tagsCopy := make([]string, len(d.Tags))
	copy(tagsCopy, d.Tags)

	// 复制 Metadata map
	metadataCopy := make(map[string]string)
	for k, v := range d.Metadata {
		metadataCopy[k] = v
	}

	return &Document{
		Title:    d.Title,
		Content:  d.Content,
		Tags:     tagsCopy,
		Metadata: metadataCopy,
	}
}

// String 返回 Document 的字符串表示
func (d *Document) String() string {
	return fmt.Sprintf("Document{Title: %s, Content: %s, Tags: %v, Metadata: %v}",
		d.Title, d.Content, d.Tags, d.Metadata)
}

// 通用深拷贝函数 - 使用 encoding/gob
func DeepCopyGob(src, dst interface{}) error {
	var buf bytes.Buffer

	// 编码
	if err := gob.NewEncoder(&buf).Encode(src); err != nil {
		return err
	}

	// 解码
	return gob.NewDecoder(&buf).Decode(dst)
}

// 通用深拷贝函数 - 使用 JSON
func DeepCopyJSON(src, dst interface{}) error {
	data, err := json.Marshal(src)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, dst)
}

// GameCharacter 表示游戏角色（用于演示复杂对象的深拷贝）
type GameCharacter struct {
	Name      string
	Level     int
	Health    int
	Skills    []string
	Equipment map[string]string
	Inventory []Item
}

type Item struct {
	Name     string
	Quantity int
}

// DeepCopy 执行深拷贝
func (g *GameCharacter) DeepCopy() *GameCharacter {
	// 复制 Skills
	skillsCopy := make([]string, len(g.Skills))
	copy(skillsCopy, g.Skills)

	// 复制 Equipment
	equipmentCopy := make(map[string]string)
	for k, v := range g.Equipment {
		equipmentCopy[k] = v
	}

	// 复制 Inventory
	inventoryCopy := make([]Item, len(g.Inventory))
	for i, item := range g.Inventory {
		inventoryCopy[i] = Item{
			Name:     item.Name,
			Quantity: item.Quantity,
		}
	}

	return &GameCharacter{
		Name:      g.Name,
		Level:     g.Level,
		Health:    g.Health,
		Skills:    skillsCopy,
		Equipment: equipmentCopy,
		Inventory: inventoryCopy,
	}
}

func (g *GameCharacter) String() string {
	return fmt.Sprintf("GameCharacter{Name: %s, Level: %d, Health: %d, Skills: %v, Equipment: %v, Inventory: %v}",
		g.Name, g.Level, g.Health, g.Skills, g.Equipment, g.Inventory)
}

func main() {
	fmt.Println("=== 深拷贝示例 ===\n")

	// 示例 1: Person 的深拷贝（手动实现）
	fmt.Println("示例 1: Person 的深拷贝（手动实现）")
	fmt.Println("-------------------")

	original := &Person{
		Name: "Alice",
		Age:  30,
		Address: &Address{
			City:    "Beijing",
			Country: "China",
		},
		Hobbies: []string{"reading", "swimming"},
	}

	fmt.Println("原始对象:", original)

	// 执行深拷贝
	clone := original.DeepCopy()
	fmt.Println("克隆对象:", clone)
	fmt.Println()

	// 修改克隆对象的所有字段
	fmt.Println("修改克隆对象的所有字段:")
	clone.Name = "Bob"
	clone.Age = 25
	clone.Address.City = "Shanghai"
	clone.Address.Country = "China"
	clone.Hobbies[0] = "coding"
	clone.Hobbies = append(clone.Hobbies, "gaming")

	fmt.Println("原始对象:", original)
	fmt.Println("克隆对象:", clone)
	fmt.Println("✅ 完全独立，修改克隆对象不影响原始对象！")
	fmt.Println()

	// 示例 2: Document 的深拷贝
	fmt.Println("\n示例 2: Document 的深拷贝")
	fmt.Println("-------------------")

	doc1 := &Document{
		Title:   "Go Design Patterns",
		Content: "This is a book about design patterns in Go.",
		Tags:    []string{"go", "patterns", "programming"},
		Metadata: map[string]string{
			"author": "Alice",
			"year":   "2024",
		},
	}

	fmt.Println("原始文档:", doc1)

	// 执行深拷贝
	doc2 := doc1.DeepCopy()
	fmt.Println("克隆文档:", doc2)
	fmt.Println()

	// 修改克隆文档
	fmt.Println("修改克隆文档:")
	doc2.Title = "Advanced Go Patterns"
	doc2.Tags[0] = "golang"
	doc2.Tags = append(doc2.Tags, "advanced")
	doc2.Metadata["author"] = "Bob"
	doc2.Metadata["edition"] = "2nd"

	fmt.Println("原始文档:", doc1)
	fmt.Println("克隆文档:", doc2)
	fmt.Println("✅ 完全独立，修改不会相互影响！")
	fmt.Println()

	// 示例 3: 使用 gob 进行深拷贝
	fmt.Println("\n示例 3: 使用 gob 进行深拷贝")
	fmt.Println("-------------------")

	person1 := &Person{
		Name: "Charlie",
		Age:  35,
		Address: &Address{
			City:    "Guangzhou",
			Country: "China",
		},
		Hobbies: []string{"music", "travel"},
	}

	fmt.Println("原始对象:", person1)

	var person2 Person
	if err := DeepCopyGob(person1, &person2); err != nil {
		fmt.Println("错误:", err)
	} else {
		fmt.Println("克隆对象:", &person2)

		// 修改克隆对象
		person2.Name = "David"
		person2.Address.City = "Shenzhen"
		person2.Hobbies[0] = "sports"

		fmt.Println("\n修改后:")
		fmt.Println("原始对象:", person1)
		fmt.Println("克隆对象:", &person2)
		fmt.Println("✅ 使用 gob 实现深拷贝，完全独立！")
	}
	fmt.Println()

	// 示例 4: 使用 JSON 进行深拷贝
	fmt.Println("\n示例 4: 使用 JSON 进行深拷贝")
	fmt.Println("-------------------")

	doc3 := &Document{
		Title:   "Kubernetes Guide",
		Content: "A comprehensive guide to Kubernetes.",
		Tags:    []string{"k8s", "devops", "cloud"},
		Metadata: map[string]string{
			"author": "Eve",
			"year":   "2024",
		},
	}

	fmt.Println("原始文档:", doc3)

	var doc4 Document
	if err := DeepCopyJSON(doc3, &doc4); err != nil {
		fmt.Println("错误:", err)
	} else {
		fmt.Println("克隆文档:", &doc4)

		// 修改克隆文档
		doc4.Title = "Advanced Kubernetes"
		doc4.Tags[0] = "kubernetes"
		doc4.Metadata["author"] = "Frank"

		fmt.Println("\n修改后:")
		fmt.Println("原始文档:", doc3)
		fmt.Println("克隆文档:", &doc4)
		fmt.Println("✅ 使用 JSON 实现深拷贝，完全独立！")
	}
	fmt.Println()

	// 示例 5: 复杂对象的深拷贝
	fmt.Println("\n示例 5: 复杂对象的深拷贝（游戏角色）")
	fmt.Println("-------------------")

	warrior := &GameCharacter{
		Name:   "Warrior",
		Level:  10,
		Health: 100,
		Skills: []string{"slash", "block", "charge"},
		Equipment: map[string]string{
			"weapon": "sword",
			"armor":  "plate",
			"shield": "iron",
		},
		Inventory: []Item{
			{Name: "health potion", Quantity: 5},
			{Name: "mana potion", Quantity: 3},
		},
	}

	fmt.Println("原始角色:", warrior)

	// 克隆角色
	warriorClone := warrior.DeepCopy()
	fmt.Println("克隆角色:", warriorClone)
	fmt.Println()

	// 修改克隆角色
	fmt.Println("修改克隆角色:")
	warriorClone.Name = "Elite Warrior"
	warriorClone.Level = 15
	warriorClone.Health = 150
	warriorClone.Skills[0] = "power slash"
	warriorClone.Skills = append(warriorClone.Skills, "berserk")
	warriorClone.Equipment["weapon"] = "legendary sword"
	warriorClone.Inventory[0].Quantity = 10

	fmt.Println("原始角色:", warrior)
	fmt.Println("克隆角色:", warriorClone)
	fmt.Println("✅ 复杂对象深拷贝成功，完全独立！")

	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n深拷贝总结:")
	fmt.Println("✅ 优点: 完全独立的副本，修改不会相互影响")
	fmt.Println("❌ 缺点: 速度较慢，内存占用多")
	fmt.Println("📌 实现方式:")
	fmt.Println("   1. 手动实现 - 性能最好，但需要维护")
	fmt.Println("   2. gob 序列化 - 通用，性能较好")
	fmt.Println("   3. JSON 序列化 - 简单，但性能较差")
	fmt.Println("📌 适用场景: 需要完全独立的副本，修改克隆对象不应影响原对象")
}
