package main

import (
	"fmt"
	"time"
)

// 审批流程示例
// 本示例展示了责任链模式在审批流程中的应用

// ApprovalRequest 审批请求
type ApprovalRequest struct {
	ID          string
	Title       string
	Amount      float64
	Applicant   string
	Department  string
	Description string
	SubmitTime  time.Time
	ApprovalLog []string
}

func NewApprovalRequest(id, title string, amount float64, applicant, department, description string) *ApprovalRequest {
	return &ApprovalRequest{
		ID:          id,
		Title:       title,
		Amount:      amount,
		Applicant:   applicant,
		Department:  department,
		Description: description,
		SubmitTime:  time.Now(),
		ApprovalLog: make([]string, 0),
	}
}

func (r *ApprovalRequest) AddLog(log string) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	r.ApprovalLog = append(r.ApprovalLog, fmt.Sprintf("[%s] %s", timestamp, log))
}

func (r *ApprovalRequest) PrintLog() {
	fmt.Println("\n审批日志:")
	for _, log := range r.ApprovalLog {
		fmt.Printf("  %s\n", log)
	}
}

// Approver 审批者接口
type Approver interface {
	SetNext(approver Approver) Approver
	Approve(request *ApprovalRequest) (bool, error)
	GetName() string
	GetLevel() string
}

// BaseApprover 基础审批者
type BaseApprover struct {
	next  Approver
	name  string
	level string
}

func (a *BaseApprover) SetNext(approver Approver) Approver {
	a.next = approver
	return approver
}

func (a *BaseApprover) GetName() string {
	return a.name
}

func (a *BaseApprover) GetLevel() string {
	return a.level
}

func (a *BaseApprover) PassToNext(request *ApprovalRequest) (bool, error) {
	if a.next != nil {
		fmt.Printf("[%s] 转交给上级审批\n", a.name)
		request.AddLog(fmt.Sprintf("%s (%s) 转交给上级", a.name, a.level))
		return a.next.Approve(request)
	}
	return false, fmt.Errorf("没有更高级别的审批者")
}

// ManagerApprover 部门经理审批者
type ManagerApprover struct {
	BaseApprover
	approvalLimit float64
}

func NewManagerApprover(name string, limit float64) *ManagerApprover {
	return &ManagerApprover{
		BaseApprover: BaseApprover{
			name:  name,
			level: "部门经理",
		},
		approvalLimit: limit,
	}
}

func (a *ManagerApprover) Approve(request *ApprovalRequest) (bool, error) {
	fmt.Printf("\n[%s - %s] 审批请求: %s (金额: %.2f 元)\n", 
		a.level, a.name, request.Title, request.Amount)
	
	if request.Amount < a.approvalLimit {
		fmt.Printf("[%s] ✓ 审批通过 (权限范围: < %.2f 元)\n", a.name, a.approvalLimit)
		request.AddLog(fmt.Sprintf("%s (%s) 审批通过", a.name, a.level))
		return true, nil
	}
	
	fmt.Printf("[%s] 金额超出权限范围 (%.2f >= %.2f)\n", 
		a.name, request.Amount, a.approvalLimit)
	return a.PassToNext(request)
}

// DirectorApprover 总监审批者
type DirectorApprover struct {
	BaseApprover
	approvalLimit float64
}

func NewDirectorApprover(name string, limit float64) *DirectorApprover {
	return &DirectorApprover{
		BaseApprover: BaseApprover{
			name:  name,
			level: "总监",
		},
		approvalLimit: limit,
	}
}

func (a *DirectorApprover) Approve(request *ApprovalRequest) (bool, error) {
	fmt.Printf("\n[%s - %s] 审批请求: %s (金额: %.2f 元)\n", 
		a.level, a.name, request.Title, request.Amount)
	
	if request.Amount < a.approvalLimit {
		fmt.Printf("[%s] ✓ 审批通过 (权限范围: < %.2f 元)\n", a.name, a.approvalLimit)
		request.AddLog(fmt.Sprintf("%s (%s) 审批通过", a.name, a.level))
		return true, nil
	}
	
	fmt.Printf("[%s] 金额超出权限范围 (%.2f >= %.2f)\n", 
		a.name, request.Amount, a.approvalLimit)
	return a.PassToNext(request)
}

// VPApprover 副总经理审批者
type VPApprover struct {
	BaseApprover
	approvalLimit float64
}

func NewVPApprover(name string, limit float64) *VPApprover {
	return &VPApprover{
		BaseApprover: BaseApprover{
			name:  name,
			level: "副总经理",
		},
		approvalLimit: limit,
	}
}

func (a *VPApprover) Approve(request *ApprovalRequest) (bool, error) {
	fmt.Printf("\n[%s - %s] 审批请求: %s (金额: %.2f 元)\n", 
		a.level, a.name, request.Title, request.Amount)
	
	if request.Amount < a.approvalLimit {
		fmt.Printf("[%s] ✓ 审批通过 (权限范围: < %.2f 元)\n", a.name, a.approvalLimit)
		request.AddLog(fmt.Sprintf("%s (%s) 审批通过", a.name, a.level))
		return true, nil
	}
	
	fmt.Printf("[%s] 金额超出权限范围 (%.2f >= %.2f)\n", 
		a.name, request.Amount, a.approvalLimit)
	return a.PassToNext(request)
}

// CEOApprover 总经理审批者
type CEOApprover struct {
	BaseApprover
	approvalLimit float64
}

func NewCEOApprover(name string, limit float64) *CEOApprover {
	return &CEOApprover{
		BaseApprover: BaseApprover{
			name:  name,
			level: "总经理",
		},
		approvalLimit: limit,
	}
}

func (a *CEOApprover) Approve(request *ApprovalRequest) (bool, error) {
	fmt.Printf("\n[%s - %s] 审批请求: %s (金额: %.2f 元)\n", 
		a.level, a.name, request.Title, request.Amount)
	
	if request.Amount < a.approvalLimit {
		fmt.Printf("[%s] ✓ 审批通过 (权限范围: < %.2f 元)\n", a.name, a.approvalLimit)
		request.AddLog(fmt.Sprintf("%s (%s) 审批通过", a.name, a.level))
		return true, nil
	}
	
	fmt.Printf("[%s] 金额过大，需要董事会审批 (%.2f >= %.2f)\n", 
		a.name, request.Amount, a.approvalLimit)
	request.AddLog(fmt.Sprintf("%s (%s) 建议提交董事会审批", a.name, a.level))
	return false, fmt.Errorf("金额过大，需要董事会审批")
}

// ConditionalApprover 条件审批者（根据部门决定）
type ConditionalApprover struct {
	BaseApprover
	departmentLimits map[string]float64
}

func NewConditionalApprover(name string, limits map[string]float64) *ConditionalApprover {
	return &ConditionalApprover{
		BaseApprover: BaseApprover{
			name:  name,
			level: "条件审批",
		},
		departmentLimits: limits,
	}
}

func (a *ConditionalApprover) Approve(request *ApprovalRequest) (bool, error) {
	fmt.Printf("\n[%s - %s] 审批请求: %s (部门: %s, 金额: %.2f 元)\n", 
		a.level, a.name, request.Title, request.Department, request.Amount)
	
	limit, ok := a.departmentLimits[request.Department]
	if !ok {
		fmt.Printf("[%s] 部门 %s 没有特殊审批权限\n", a.name, request.Department)
		return a.PassToNext(request)
	}
	
	if request.Amount < limit {
		fmt.Printf("[%s] ✓ 审批通过 (部门 %s 权限范围: < %.2f 元)\n", 
			a.name, request.Department, limit)
		request.AddLog(fmt.Sprintf("%s (%s) 基于部门权限审批通过", a.name, a.level))
		return true, nil
	}
	
	fmt.Printf("[%s] 金额超出部门权限范围\n", a.name)
	return a.PassToNext(request)
}

func printRequestInfo(request *ApprovalRequest) {
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Printf("审批请求信息:\n")
	fmt.Printf("  ID: %s\n", request.ID)
	fmt.Printf("  标题: %s\n", request.Title)
	fmt.Printf("  金额: %.2f 元\n", request.Amount)
	fmt.Printf("  申请人: %s\n", request.Applicant)
	fmt.Printf("  部门: %s\n", request.Department)
	fmt.Printf("  说明: %s\n", request.Description)
	fmt.Printf("  提交时间: %s\n", request.SubmitTime.Format("2006-01-02 15:04:05"))
	fmt.Println(strings.Repeat("=", 70))
}

func main() {
	fmt.Println("=== 责任链模式示例 - 审批流程 ===\n")

	// 构建审批链
	manager := NewManagerApprover("张经理", 1000)
	director := NewDirectorApprover("李总监", 5000)
	vp := NewVPApprover("王副总", 10000)
	ceo := NewCEOApprover("刘总经理", 50000)

	manager.SetNext(director).SetNext(vp).SetNext(ceo)

	// 场景1: 小额申请（部门经理审批）
	fmt.Println("--- 场景1: 小额申请 ---")
	req1 := NewApprovalRequest(
		"REQ001",
		"购买办公用品",
		800,
		"小王",
		"技术部",
		"购买键盘、鼠标等办公用品",
	)
	printRequestInfo(req1)
	
	if approved, err := manager.Approve(req1); approved {
		fmt.Println("\n✓ 审批流程完成")
	} else {
		fmt.Printf("\n✗ 审批失败: %v\n", err)
	}
	req1.PrintLog()

	// 场景2: 中等金额（总监审批）
	fmt.Println("\n\n--- 场景2: 中等金额申请 ---")
	req2 := NewApprovalRequest(
		"REQ002",
		"团队建设活动",
		3500,
		"小李",
		"市场部",
		"组织团队户外拓展活动",
	)
	printRequestInfo(req2)
	
	if approved, err := manager.Approve(req2); approved {
		fmt.Println("\n✓ 审批流程完成")
	} else {
		fmt.Printf("\n✗ 审批失败: %v\n", err)
	}
	req2.PrintLog()

	// 场景3: 较大金额（副总审批）
	fmt.Println("\n\n--- 场景3: 较大金额申请 ---")
	req3 := NewApprovalRequest(
		"REQ003",
		"购买服务器设备",
		8000,
		"小张",
		"技术部",
		"购买2台服务器用于生产环境",
	)
	printRequestInfo(req3)
	
	if approved, err := manager.Approve(req3); approved {
		fmt.Println("\n✓ 审批流程完成")
	} else {
		fmt.Printf("\n✗ 审批失败: %v\n", err)
	}
	req3.PrintLog()

	// 场景4: 大额申请（总经理审批）
	fmt.Println("\n\n--- 场景4: 大额申请 ---")
	req4 := NewApprovalRequest(
		"REQ004",
		"市场推广费用",
		25000,
		"小赵",
		"市场部",
		"Q1季度市场推广活动费用",
	)
	printRequestInfo(req4)
	
	if approved, err := manager.Approve(req4); approved {
		fmt.Println("\n✓ 审批流程完成")
	} else {
		fmt.Printf("\n✗ 审批失败: %v\n", err)
	}
	req4.PrintLog()

	// 场景5: 超大金额（需要董事会）
	fmt.Println("\n\n--- 场景5: 超大金额申请 ---")
	req5 := NewApprovalRequest(
		"REQ005",
		"新项目投资",
		80000,
		"小孙",
		"战略部",
		"投资新产品线研发",
	)
	printRequestInfo(req5)
	
	if approved, err := manager.Approve(req5); approved {
		fmt.Println("\n✓ 审批流程完成")
	} else {
		fmt.Printf("\n✗ 审批失败: %v\n", err)
	}
	req5.PrintLog()

	// 场景6: 使用条件审批者
	fmt.Println("\n\n--- 场景6: 部门特殊权限 ---")
	
	// 创建新的审批链，包含条件审批者
	manager2 := NewManagerApprover("张经理", 1000)
	conditional := NewConditionalApprover("财务审批", map[string]float64{
		"财务部": 8000,
		"采购部": 6000,
	})
	director2 := NewDirectorApprover("李总监", 5000)
	vp2 := NewVPApprover("王副总", 10000)
	
	manager2.SetNext(conditional).SetNext(director2).SetNext(vp2)
	
	req6 := NewApprovalRequest(
		"REQ006",
		"财务软件采购",
		7000,
		"小周",
		"财务部",
		"购买财务管理软件",
	)
	printRequestInfo(req6)
	
	if approved, err := manager2.Approve(req6); approved {
		fmt.Println("\n✓ 审批流程完成")
	} else {
		fmt.Printf("\n✗ 审批失败: %v\n", err)
	}
	req6.PrintLog()

	fmt.Println("\n=== 示例结束 ===")
}
