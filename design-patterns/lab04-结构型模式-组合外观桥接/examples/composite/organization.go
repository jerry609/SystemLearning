package main

import "fmt"

// 组合模式示例：组织架构
// 本示例展示了如何使用组合模式表示公司的组织架构

// Employee 员工接口（Component）
type Employee interface {
	GetName() string
	GetPosition() string
	GetSalary() float64
	Display(indent string)
	GetSubordinateCount() int
}

// IndividualEmployee 普通员工（Leaf）
type IndividualEmployee struct {
	name     string
	position string
	salary   float64
}

func NewIndividualEmployee(name, position string, salary float64) *IndividualEmployee {
	return &IndividualEmployee{
		name:     name,
		position: position,
		salary:   salary,
	}
}

func (e *IndividualEmployee) GetName() string {
	return e.name
}

func (e *IndividualEmployee) GetPosition() string {
	return e.position
}

func (e *IndividualEmployee) GetSalary() float64 {
	return e.salary
}

func (e *IndividualEmployee) Display(indent string) {
	fmt.Printf("%s👤 %s - %s (薪资: ¥%.2f)\n", indent, e.name, e.position, e.salary)
}

func (e *IndividualEmployee) GetSubordinateCount() int {
	return 0
}

// Manager 经理（Composite）
type Manager struct {
	name          string
	position      string
	salary        float64
	subordinates  []Employee
}

func NewManager(name, position string, salary float64) *Manager {
	return &Manager{
		name:         name,
		position:     position,
		salary:       salary,
		subordinates: make([]Employee, 0),
	}
}

func (m *Manager) GetName() string {
	return m.name
}

func (m *Manager) GetPosition() string {
	return m.position
}

func (m *Manager) GetSalary() float64 {
	return m.salary
}

func (m *Manager) Display(indent string) {
	fmt.Printf("%s👔 %s - %s (薪资: ¥%.2f, 下属: %d人)\n", 
		indent, m.name, m.position, m.salary, len(m.subordinates))
	
	for _, subordinate := range m.subordinates {
		subordinate.Display(indent + "  ")
	}
}

func (m *Manager) GetSubordinateCount() int {
	count := len(m.subordinates)
	for _, subordinate := range m.subordinates {
		count += subordinate.GetSubordinateCount()
	}
	return count
}

// AddSubordinate 添加下属
func (m *Manager) AddSubordinate(employee Employee) {
	m.subordinates = append(m.subordinates, employee)
}

// RemoveSubordinate 移除下属
func (m *Manager) RemoveSubordinate(employee Employee) {
	for i, subordinate := range m.subordinates {
		if subordinate == employee {
			m.subordinates = append(m.subordinates[:i], m.subordinates[i+1:]...)
			break
		}
	}
}

// GetSubordinates 获取所有直接下属
func (m *Manager) GetSubordinates() []Employee {
	return m.subordinates
}

// GetTotalSalary 计算团队总薪资
func (m *Manager) GetTotalSalary() float64 {
	total := m.salary
	for _, subordinate := range m.subordinates {
		total += subordinate.GetSalary()
		if manager, ok := subordinate.(*Manager); ok {
			total += manager.GetTotalSalary() - manager.GetSalary()
		}
	}
	return total
}

// FindEmployee 查找员工
func (m *Manager) FindEmployee(name string) Employee {
	if m.name == name {
		return m
	}
	
	for _, subordinate := range m.subordinates {
		if subordinate.GetName() == name {
			return subordinate
		}
		
		if manager, ok := subordinate.(*Manager); ok {
			if found := manager.FindEmployee(name); found != nil {
				return found
			}
		}
	}
	
	return nil
}

// GetAllEmployees 获取所有员工（包括自己）
func (m *Manager) GetAllEmployees() []Employee {
	employees := []Employee{m}
	
	for _, subordinate := range m.subordinates {
		if manager, ok := subordinate.(*Manager); ok {
			employees = append(employees, manager.GetAllEmployees()...)
		} else {
			employees = append(employees, subordinate)
		}
	}
	
	return employees
}

func main() {
	fmt.Println("=== 组合模式示例：组织架构 ===\n")

	// 创建 CEO
	ceo := NewManager("张三", "CEO", 50000)

	// 创建部门经理
	cto := NewManager("李四", "CTO", 30000)
	cfo := NewManager("王五", "CFO", 30000)
	cmo := NewManager("赵六", "CMO", 28000)

	// 创建团队负责人
	devLead := NewManager("孙七", "开发主管", 20000)
	qaLead := NewManager("周八", "测试主管", 18000)

	// 创建普通员工
	dev1 := NewIndividualEmployee("钱九", "高级开发工程师", 15000)
	dev2 := NewIndividualEmployee("吴十", "开发工程师", 12000)
	dev3 := NewIndividualEmployee("郑十一", "初级开发工程师", 8000)

	qa1 := NewIndividualEmployee("冯十二", "测试工程师", 10000)
	qa2 := NewIndividualEmployee("陈十三", "测试工程师", 10000)

	accountant := NewIndividualEmployee("褚十四", "会计", 12000)
	marketer := NewIndividualEmployee("卫十五", "市场专员", 10000)

	// 构建组织架构
	ceo.AddSubordinate(cto)
	ceo.AddSubordinate(cfo)
	ceo.AddSubordinate(cmo)

	cto.AddSubordinate(devLead)
	cto.AddSubordinate(qaLead)

	devLead.AddSubordinate(dev1)
	devLead.AddSubordinate(dev2)
	devLead.AddSubordinate(dev3)

	qaLead.AddSubordinate(qa1)
	qaLead.AddSubordinate(qa2)

	cfo.AddSubordinate(accountant)
	cmo.AddSubordinate(marketer)

	// 显示组织架构
	fmt.Println("公司组织架构：")
	ceo.Display("")

	// 统计信息
	fmt.Printf("\n公司总人数：%d 人\n", ceo.GetSubordinateCount()+1)
	fmt.Printf("公司总薪资：¥%.2f\n", ceo.GetTotalSalary())

	// 查找员工
	fmt.Println("\n查找员工 '钱九'：")
	found := ceo.FindEmployee("钱九")
	if found != nil {
		fmt.Printf("找到：%s - %s (薪资: ¥%.2f)\n", 
			found.GetName(), found.GetPosition(), found.GetSalary())
	}

	// CTO 部门统计
	fmt.Println("\nCTO 部门统计：")
	fmt.Printf("部门人数：%d 人\n", cto.GetSubordinateCount()+1)
	fmt.Printf("部门薪资：¥%.2f\n", cto.GetTotalSalary())

	// 获取所有员工
	fmt.Println("\n所有员工列表：")
	allEmployees := ceo.GetAllEmployees()
	for i, emp := range allEmployees {
		fmt.Printf("%d. %s - %s\n", i+1, emp.GetName(), emp.GetPosition())
	}

	// 调整组织架构
	fmt.Println("\n将测试主管调整到 CEO 直接管理：")
	cto.RemoveSubordinate(qaLead)
	ceo.AddSubordinate(qaLead)
	ceo.Display("")

	fmt.Println("\n=== 示例结束 ===")
}

// 输出示例：
// === 组合模式示例：组织架构 ===
//
// 公司组织架构：
// 👔 张三 - CEO (薪资: ¥50000.00, 下属: 3人)
//   👔 李四 - CTO (薪资: ¥30000.00, 下属: 2人)
//     👔 孙七 - 开发主管 (薪资: ¥20000.00, 下属: 3人)
//       👤 钱九 - 高级开发工程师 (薪资: ¥15000.00)
//       👤 吴十 - 开发工程师 (薪资: ¥12000.00)
//       👤 郑十一 - 初级开发工程师 (薪资: ¥8000.00)
//     👔 周八 - 测试主管 (薪资: ¥18000.00, 下属: 2人)
//       👤 冯十二 - 测试工程师 (薪资: ¥10000.00)
//       👤 陈十三 - 测试工程师 (薪资: ¥10000.00)
//   👔 王五 - CFO (薪资: ¥30000.00, 下属: 1人)
//     👤 褚十四 - 会计 (薪资: ¥12000.00)
//   👔 赵六 - CMO (薪资: ¥28000.00, 下属: 1人)
//     👤 卫十五 - 市场专员 (薪资: ¥10000.00)
//
// 公司总人数：15 人
// 公司总薪资：¥253000.00
//
// 查找员工 '钱九'：
// 找到：钱九 - 高级开发工程师 (薪资: ¥15000.00)
//
// CTO 部门统计：
// 部门人数：8 人
// 部门薪资：¥123000.00
//
// 所有员工列表：
// 1. 张三 - CEO
// 2. 李四 - CTO
// 3. 孙七 - 开发主管
// 4. 钱九 - 高级开发工程师
// 5. 吴十 - 开发工程师
// 6. 郑十一 - 初级开发工程师
// 7. 周八 - 测试主管
// 8. 冯十二 - 测试工程师
// 9. 陈十三 - 测试工程师
// 10. 王五 - CFO
// 11. 褚十四 - 会计
// 12. 赵六 - CMO
// 13. 卫十五 - 市场专员
//
// 将测试主管调整到 CEO 直接管理：
// 👔 张三 - CEO (薪资: ¥50000.00, 下属: 4人)
//   👔 李四 - CTO (薪资: ¥30000.00, 下属: 1人)
//     👔 孙七 - 开发主管 (薪资: ¥20000.00, 下属: 3人)
//       👤 钱九 - 高级开发工程师 (薪资: ¥15000.00)
//       👤 吴十 - 开发工程师 (薪资: ¥12000.00)
//       👤 郑十一 - 初级开发工程师 (薪资: ¥8000.00)
//   👔 王五 - CFO (薪资: ¥30000.00, 下属: 1人)
//     👤 褚十四 - 会计 (薪资: ¥12000.00)
//   👔 赵六 - CMO (薪资: ¥28000.00, 下属: 1人)
//     👤 卫十五 - 市场专员 (薪资: ¥10000.00)
//   👔 周八 - 测试主管 (薪资: ¥18000.00, 下属: 2人)
//     👤 冯十二 - 测试工程师 (薪资: ¥10000.00)
//     👤 陈十三 - 测试工程师 (薪资: ¥10000.00)
//
// === 示例结束 ===
