package main

import (
	"fmt"
)

// 浅拷贝示例
// 本示例展示浅拷贝的特点：只复制值类型和引用，不复制引用指向的对象

// Address 表示地址信息
type Address struct {
	City    string
	Country string
}

// Person 表示一个人
type Person struct {
	Name    string
	Age     int
	Address *Address // 引用类型
	Hobbies []string // 引用类型（切片）
}

// ShallowCopy 执行浅拷贝
// 只复制字段的值，对于引用类型，只复制引用（指针），不复制指向的对象
func (p *Person) ShallowCopy() *Person {
	return &Person{
		Name:    p.Name,
		Age:     p.Age,
		Address: p.Address,   // 只复制指针
		Hobbies: p.Hobbies,   // 只复制切片头
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

// ShallowCopy 执行浅拷贝
func (d *Document) ShallowCopy() *Document {
	return &Document{
		Title:    d.Title,
		Content:  d.Content,
		Tags:     d.Tags,     // 只复制切片头
		Metadata: d.Metadata, // 只复制 map 引用
	}
}

// String 返回 Document 的字符串表示
func (d *Document) String() string {
	return fmt.Sprintf("Document{Title: %s, Content: %s, Tags: %v, Metadata: %v}",
		d.Title, d.Content, d.Tags, d.Metadata)
}

func main() {
	fmt.Println("=== 浅拷贝示例 ===\n")

	// 示例 1: Person 的浅拷贝
	fmt.Println("示例 1: Person 的浅拷贝")
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

	// 执行浅拷贝
	clone := original.ShallowCopy()
	fmt.Println("克隆对象:", clone)
	fmt.Println()

	// 修改克隆对象的值类型字段
	fmt.Println("修改克隆对象的值类型字段（Name 和 Age）:")
	clone.Name = "Bob"
	clone.Age = 25
	fmt.Println("原始对象:", original)
	fmt.Println("克隆对象:", clone)
	fmt.Println("✅ 值类型字段独立，修改克隆对象不影响原始对象")
	fmt.Println()

	// 修改克隆对象的引用类型字段（Address）
	fmt.Println("修改克隆对象的引用类型字段（Address）:")
	clone.Address.City = "Shanghai"
	fmt.Println("原始对象:", original)
	fmt.Println("克隆对象:", clone)
	fmt.Println("❌ 引用类型字段共享，修改克隆对象会影响原始对象！")
	fmt.Println()

	// 修改克隆对象的切片
	fmt.Println("修改克隆对象的切片（Hobbies）:")
	clone.Hobbies[0] = "coding"
	clone.Hobbies = append(clone.Hobbies, "gaming")
	fmt.Println("原始对象:", original)
	fmt.Println("克隆对象:", clone)
	fmt.Println("⚠️  修改切片元素会影响原始对象，但 append 后不影响")
	fmt.Println()

	// 示例 2: Document 的浅拷贝
	fmt.Println("\n示例 2: Document 的浅拷贝")
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

	// 执行浅拷贝
	doc2 := doc1.ShallowCopy()
	fmt.Println("克隆文档:", doc2)
	fmt.Println()

	// 修改克隆文档的值类型字段
	fmt.Println("修改克隆文档的值类型字段:")
	doc2.Title = "Advanced Go Patterns"
	doc2.Content = "This is an advanced book."
	fmt.Println("原始文档:", doc1)
	fmt.Println("克隆文档:", doc2)
	fmt.Println("✅ 值类型字段独立")
	fmt.Println()

	// 修改克隆文档的切片
	fmt.Println("修改克隆文档的切片:")
	doc2.Tags[0] = "golang"
	fmt.Println("原始文档:", doc1)
	fmt.Println("克隆文档:", doc2)
	fmt.Println("❌ 切片元素共享，修改会影响原始文档！")
	fmt.Println()

	// 修改克隆文档的 map
	fmt.Println("修改克隆文档的 map:")
	doc2.Metadata["author"] = "Bob"
	doc2.Metadata["edition"] = "2nd"
	fmt.Println("原始文档:", doc1)
	fmt.Println("克隆文档:", doc2)
	fmt.Println("❌ map 共享，修改会影响原始文档！")
	fmt.Println()

	// 示例 3: 浅拷贝的适用场景
	fmt.Println("\n示例 3: 浅拷贝的适用场景")
	fmt.Println("-------------------")

	// 场景：不可变对象
	type Config struct {
		AppName string
		Version string
		Debug   bool
	}

	config1 := &Config{
		AppName: "MyApp",
		Version: "1.0.0",
		Debug:   false,
	}

	// 浅拷贝足够，因为所有字段都是值类型
	config2 := &Config{
		AppName: config1.AppName,
		Version: config1.Version,
		Debug:   config1.Debug,
	}

	config2.Debug = true
	fmt.Printf("原始配置: %+v\n", config1)
	fmt.Printf("克隆配置: %+v\n", config2)
	fmt.Println("✅ 对于只包含值类型的对象，浅拷贝就足够了")

	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n浅拷贝总结:")
	fmt.Println("✅ 优点: 速度快，内存占用少")
	fmt.Println("❌ 缺点: 引用类型字段共享，修改会相互影响")
	fmt.Println("📌 适用场景: 对象只包含值类型，或者明确希望共享引用类型数据")
}
