package main

import "fmt"

// 练习 1: 菜单系统 (组合模式) - 参考答案
//
// 设计思路:
// 1. 定义统一的 MenuComponent 接口，使菜单项和子菜单可以一致处理
// 2. MenuItem 作为叶子节点，表示具体的菜品
// 3. Menu 作为容器节点，可以包含菜单项或其他子菜单
// 4. 使用递归实现树形结构的遍历和计算
//
// 使用的设计模式: 组合模式 (Composite Pattern)
// 模式应用位置: MenuComponent 接口及其实现类

// MenuComponent 菜单组件接口
type MenuComponent interface {
	GetName() string
	GetPrice() float64
	Display(indent string)
	IsVegetarian() bool
}

// MenuItem 菜单项（叶子节点）
type MenuItem struct {
	name        string
	price       float64
	description string
	vegetarian  bool
}

func NewMenuItem(name string, price float64, description string, vegetarian bool) *MenuItem {
	return &MenuItem{
		name:        name,
		price:       price,
		description: description,
		vegetarian:  vegetarian,
	}
}

func (m *MenuItem) GetName() string {
	return m.name
}

func (m *MenuItem) GetPrice() float64 {
	return m.price
}

func (m *MenuItem) Display(indent string) {
	icon := "🍽️ "
	if m.vegetarian {
		icon = "🥬"
	}
	
	vegLabel := ""
	if m.vegetarian {
		vegLabel = " [素食]"
	}
	
	fmt.Printf("%s%s %s - ¥%.2f%s\n", indent, icon, m.name, m.price, vegLabel)
	fmt.Printf("%s   描述: %s\n", indent, m.description)
}

func (m *MenuItem) IsVegetarian() bool {
	return m.vegetarian
}

// Menu 子菜单（容器节点）
type Menu struct {
	name        string
	description string
	components  []MenuComponent
}

func NewMenu(name, description string) *Menu {
	return &Menu{
		name:        name,
		description: description,
		components:  make([]MenuComponent, 0),
	}
}

func (m *Menu) GetName() string {
	return m.name
}

func (m *Menu) GetPrice() float64 {
	var total float64
	for _, component := range m.components {
		total += component.GetPrice()
	}
	return total
}

func (m *Menu) Display(indent string) {
	fmt.Printf("%s📁 %s - %s (总价: ¥%.2f)\n", indent, m.name, m.description, m.GetPrice())
	for _, component := range m.components {
		component.Display(indent + "  ")
	}
}

func (m *Menu) IsVegetarian() bool {
	// 如果所有子项都是素食，则菜单为素食
	for _, component := range m.components {
		if !component.IsVegetarian() {
			return false
		}
	}
	return len(m.components) > 0
}

// Add 添加组件
func (m *Menu) Add(component MenuComponent) {
	m.components = append(m.components, component)
}

// Remove 移除组件
func (m *Menu) Remove(component MenuComponent) {
	for i, c := range m.components {
		if c == component {
			m.components = append(m.components[:i], m.components[i+1:]...)
			break
		}
	}
}

// GetComponents 获取所有组件
func (m *Menu) GetComponents() []MenuComponent {
	return m.components
}

// Find 查找菜品
func (m *Menu) Find(name string) MenuComponent {
	if m.name == name {
		return m
	}
	
	for _, component := range m.components {
		if component.GetName() == name {
			return component
		}
		
		// 如果是子菜单，递归查找
		if menu, ok := component.(*Menu); ok {
			if found := menu.Find(name); found != nil {
				return found
			}
		}
	}
	
	return nil
}

// GetVegetarianItems 获取所有素食菜品
func (m *Menu) GetVegetarianItems() []MenuComponent {
	items := make([]MenuComponent, 0)
	
	for _, component := range m.components {
		if menuItem, ok := component.(*MenuItem); ok {
			if menuItem.IsVegetarian() {
				items = append(items, menuItem)
			}
		} else if menu, ok := component.(*Menu); ok {
			items = append(items, menu.GetVegetarianItems()...)
		}
	}
	
	return items
}

// CountItems 统计菜品总数
func (m *Menu) CountItems() int {
	count := 0
	
	for _, component := range m.components {
		if _, ok := component.(*MenuItem); ok {
			count++
		} else if menu, ok := component.(*Menu); ok {
			count += menu.CountItems()
		}
	}
	
	return count
}

func main() {
	fmt.Println("=== 练习 1: 菜单系统 (组合模式) ===\n")

	// 创建主菜单
	mainMenu := NewMenu("餐厅菜单", "欢迎光临")

	// 创建子菜单
	sichuanMenu := NewMenu("川菜", "麻辣鲜香")
	cantonMenu := NewMenu("粤菜", "清淡鲜美")
	vegetarianMenu := NewMenu("素菜", "健康养生")

	// 添加川菜
	sichuanMenu.Add(NewMenuItem("宫保鸡丁", 38.0, "经典川菜", false))
	sichuanMenu.Add(NewMenuItem("麻婆豆腐", 28.0, "麻辣豆腐", true))
	sichuanMenu.Add(NewMenuItem("水煮鱼", 68.0, "麻辣鲜香", false))
	sichuanMenu.Add(NewMenuItem("回锅肉", 48.0, "家常川菜", false))

	// 添加粤菜
	cantonMenu.Add(NewMenuItem("白切鸡", 48.0, "清淡鸡肉", false))
	cantonMenu.Add(NewMenuItem("清蒸鱼", 68.0, "新鲜海鱼", false))
	cantonMenu.Add(NewMenuItem("烧鹅", 88.0, "广式烧腊", false))

	// 添加素菜
	vegetarianMenu.Add(NewMenuItem("清炒时蔬", 18.0, "新鲜蔬菜", true))
	vegetarianMenu.Add(NewMenuItem("素炒三丝", 22.0, "营养均衡", true))
	vegetarianMenu.Add(NewMenuItem("香菇青菜", 25.0, "健康美味", true))

	// 构建菜单树
	mainMenu.Add(sichuanMenu)
	mainMenu.Add(cantonMenu)
	mainMenu.Add(vegetarianMenu)

	// 显示完整菜单
	fmt.Println("📋 完整菜单:")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	mainMenu.Display("")
	fmt.Println()

	// 统计信息
	fmt.Println("📊 菜单统计:")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("菜单总价: ¥%.2f\n", mainMenu.GetPrice())
	fmt.Printf("菜品总数: %d\n", mainMenu.CountItems())
	fmt.Println()

	// 查找菜品
	fmt.Println("🔍 查找菜品 '麻婆豆腐':")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	found := mainMenu.Find("麻婆豆腐")
	if found != nil {
		fmt.Printf("找到: %s (价格: ¥%.2f)\n", found.GetName(), found.GetPrice())
	} else {
		fmt.Println("未找到")
	}
	fmt.Println()

	// 列出素食菜品
	fmt.Println("🥬 素食菜品:")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	vegetarianItems := mainMenu.GetVegetarianItems()
	for i, item := range vegetarianItems {
		fmt.Printf("%d. %s (¥%.2f)\n", i+1, item.GetName(), item.GetPrice())
	}
	fmt.Println()

	// 显示单个子菜单
	fmt.Println("📁 川菜菜单:")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	sichuanMenu.Display("")
	fmt.Printf("川菜总价: ¥%.2f\n", sichuanMenu.GetPrice())
	fmt.Println()

	fmt.Println("=== 示例结束 ===")
}

// 可能的优化方向:
// 1. 添加菜品分类标签（如：辣度、口味、过敏原）
// 2. 实现菜品搜索功能（按价格范围、关键词等）
// 3. 支持菜品的启用/禁用状态
// 4. 添加菜品评分和评论功能
// 5. 实现菜单的导出功能（JSON、Markdown 等）
// 6. 支持套餐和折扣计算
// 7. 添加菜品图片和营养信息
//
// 变体实现:
// 1. 使用透明方式: 在 MenuComponent 接口中声明 Add/Remove 方法
// 2. 使用访问者模式: 实现菜单的不同遍历方式
// 3. 使用迭代器模式: 提供统一的遍历接口
// 4. 添加缓存: 缓存计算结果（如总价格）以提高性能
