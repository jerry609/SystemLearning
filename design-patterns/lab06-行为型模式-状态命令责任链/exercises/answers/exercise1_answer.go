package main

// 练习 1: 实现文档审批流程 - 参考答案
//
// 设计思路:
// 1. 使用状态模式管理文档的状态转换
// 2. 使用责任链模式组织审批者
// 3. 在文档类中维护审批历史
//
// 使用的设计模式: 状态模式 + 责任链模式
// 模式应用位置:
// - 状态模式: DocumentState 接口及其实现类
// - 责任链模式: Approver 接口及其实现类

import (
	"fmt"
	"time"
)

// DocumentState 文档状态接口
type DocumentState interface {
	Submit(doc *Document) error
	Approve(doc *Document, approver Approver) error
	Reject(doc *Document, approver Approver, reason string) error
	Withdraw(doc *Document) error
	Edit(doc *Document) error
	String() string
}

// Document 文档
type Document struct {
	ID          string
	Title       string
	Amount      float64
	Applicant   string
	Department  string
	state       DocumentState
	history     []string
	createdTime time.Time
}

func NewDocument(id, title string, amount float64, applicant, department string) *Document {
	doc := &Document{
		ID:          id,
		Title:       title,
		Amount:      amount,
		Applicant:   applicant,
		Department:  department,
		history:     make([]string, 0),
		createdTime: time.Now(),
	}
	doc.SetState(&DraftState{})
	return doc
}

func (d *Document) SetState(state DocumentState) {
	if d.state != nil {
		d.AddHistory(fmt.Sprintf("状态变更: %s -> %s", d.state.String(), state.String()))
	}
	d.state = state
	fmt.Printf("[文档 %s] 状态变更: %s\n", d.ID, state.String())
}

func (d *Document) GetState() string {
	return d.state.String()
}

func (d *Document) Submit() error {
	return d.state.Submit(d)
}

func (d *Document) Approve(approver Approver) error {
	return d.state.Approve(d, approver)
}

func (d *Document) Reject(approver Approver, reason string) error {
	return d.state.Reject(d, approver, reason)
}

func (d *Document) Withdraw() error {
	return d.state.Withdraw(d)
}

func (d *Document) Edit() error {
	return d.state.Edit(d)
}

func (d *Document) AddHistory(log string) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	d.history = append(d.history, fmt.Sprintf("[%s] %s", timestamp, log))
}

func (d *Document) PrintHistory() {
	fmt.Println("\n审批历史:")
	for _, log := range d.history {
		fmt.Printf("  %s\n", log)
	}
}

// DraftState 草稿状态
type DraftState struct{}

func (s *DraftState) Submit(doc *Document) error {
	fmt.Printf("[文档 %s] 提交审批\n", doc.ID)
	doc.SetState(&PendingState{})
	doc.AddHistory(fmt.Sprintf("%s 提交审批", doc.Applicant))
	return nil
}

func (s *DraftState) Approve(doc *Document, approver Approver) error {
	return fmt.Errorf("草稿状态不能审批")
}

func (s *DraftState) Reject(doc *Document, approver Approver, reason string) error {
	return fmt.Errorf("草稿状态不能拒绝")
}

func (s *DraftState) Withdraw(doc *Document) error {
	return fmt.Errorf("草稿状态不能撤回")
}

func (s *DraftState) Edit(doc *Document) error {
	fmt.Printf("[文档 %s] 编辑文档\n", doc.ID)
	doc.AddHistory("编辑文档")
	return nil
}

func (s *DraftState) String() string {
	return "草稿"
}

// PendingState 待审批状态
type PendingState struct{}

func (s *PendingState) Submit(doc *Document) error {
	return fmt.Errorf("文档已提交，不能重复提交")
}

func (s *PendingState) Approve(doc *Document, approver Approver) error {
	doc.SetState(&ReviewingState{})
	doc.AddHistory(fmt.Sprintf("开始审批流程"))
	
	// 进入审批链
	approved, err := approver.Approve(doc)
	if err != nil {
		return err
	}
	
	if approved {
		doc.SetState(&ApprovedState{})
		return nil
	}
	
	return nil
}

func (s *PendingState) Reject(doc *Document, approver Approver, reason string) error {
	fmt.Printf("[%s] ✗ 拒绝审批，原因: %s\n", approver.GetName(), reason)
	doc.SetState(&RejectedState{})
	doc.AddHistory(fmt.Sprintf("%s 拒绝审批: %s", approver.GetName(), reason))
	return nil
}

func (s *PendingState) Withdraw(doc *Document) error {
	fmt.Printf("[文档 %s] 撤回申请\n", doc.ID)
	doc.SetState(&WithdrawnState{})
	doc.AddHistory(fmt.Sprintf("%s 撤回申请", doc.Applicant))
	return nil
}

func (s *PendingState) Edit(doc *Document) error {
	return fmt.Errorf("待审批状态不能编辑")
}

func (s *PendingState) String() string {
	return "待审批"
}

// ReviewingState 审批中状态
type ReviewingState struct{}

func (s *ReviewingState) Submit(doc *Document) error {
	return fmt.Errorf("文档审批中，不能提交")
}

func (s *ReviewingState) Approve(doc *Document, approver Approver) error {
	return fmt.Errorf("文档已在审批中")
}

func (s *ReviewingState) Reject(doc *Document, approver Approver, reason string) error {
	fmt.Printf("[%s] ✗ 拒绝审批，原因: %s\n", approver.GetName(), reason)
	doc.SetState(&RejectedState{})
	doc.AddHistory(fmt.Sprintf("%s 拒绝审批: %s", approver.GetName(), reason))
	return nil
}

func (s *ReviewingState) Withdraw(doc *Document) error {
	return fmt.Errorf("文档审批中，不能撤回")
}

func (s *ReviewingState) Edit(doc *Document) error {
	return fmt.Errorf("审批中状态不能编辑")
}

func (s *ReviewingState) String() string {
	return "审批中"
}

// ApprovedState 已通过状态
type ApprovedState struct{}

func (s *ApprovedState) Submit(doc *Document) error {
	return fmt.Errorf("文档已通过，不能提交")
}

func (s *ApprovedState) Approve(doc *Document, approver Approver) error {
	return fmt.Errorf("文档已通过")
}

func (s *ApprovedState) Reject(doc *Document, approver Approver, reason string) error {
	return fmt.Errorf("文档已通过，不能拒绝")
}

func (s *ApprovedState) Withdraw(doc *Document) error {
	return fmt.Errorf("文档已通过，不能撤回")
}

func (s *ApprovedState) Edit(doc *Document) error {
	return fmt.Errorf("已通过状态不能编辑")
}

func (s *ApprovedState) String() string {
	return "已通过"
}

// RejectedState 已拒绝状态
type RejectedState struct{}

func (s *RejectedState) Submit(doc *Document) error {
	return fmt.Errorf("文档已拒绝，不能提交")
}

func (s *RejectedState) Approve(doc *Document, approver Approver) error {
	return fmt.Errorf("文档已拒绝")
}

func (s *RejectedState) Reject(doc *Document, approver Approver, reason string) error {
	return fmt.Errorf("文档已拒绝")
}

func (s *RejectedState) Withdraw(doc *Document) error {
	return fmt.Errorf("文档已拒绝，不能撤回")
}

func (s *RejectedState) Edit(doc *Document) error {
	return fmt.Errorf("已拒绝状态不能编辑")
}

func (s *RejectedState) String() string {
	return "已拒绝"
}

// WithdrawnState 已撤回状态
type WithdrawnState struct{}

func (s *WithdrawnState) Submit(doc *Document) error {
	return fmt.Errorf("文档已撤回，不能提交")
}

func (s *WithdrawnState) Approve(doc *Document, approver Approver) error {
	return fmt.Errorf("文档已撤回")
}

func (s *WithdrawnState) Reject(doc *Document, approver Approver, reason string) error {
	return fmt.Errorf("文档已撤回")
}

func (s *WithdrawnState) Withdraw(doc *Document) error {
	return fmt.Errorf("文档已撤回")
}

func (s *WithdrawnState) Edit(doc *Document) error {
	return fmt.Errorf("已撤回状态不能编辑")
}

func (s *WithdrawnState) String() string {
	return "已撤回"
}

// Approver 审批者接口
type Approver interface {
	SetNext(approver Approver) Approver
	Approve(doc *Document) (bool, error)
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

func (a *BaseApprover) PassToNext(doc *Document) (bool, error) {
	if a.next != nil {
		fmt.Printf("[%s] 转交给上级审批\n", a.name)
		doc.AddHistory(fmt.Sprintf("%s 转交给上级", a.name))
		return a.next.Approve(doc)
	}
	return false, fmt.Errorf("没有更高级别的审批者")
}

// ManagerApprover 部门经理
type ManagerApprover struct {
	BaseApprover
	limit float64
}

func NewManagerApprover(name string, limit float64) *ManagerApprover {
	return &ManagerApprover{
		BaseApprover: BaseApprover{name: name, level: "部门经理"},
		limit:        limit,
	}
}

func (a *ManagerApprover) Approve(doc *Document) (bool, error) {
	fmt.Printf("[%s] 审批文档: %s (金额: %.2f 元)\n", a.name, doc.Title, doc.Amount)
	
	if doc.Amount < a.limit {
		fmt.Printf("[%s] ✓ 审批通过\n", a.name)
		doc.AddHistory(fmt.Sprintf("%s (%s) 审批通过", a.name, a.level))
		return true, nil
	}
	
	fmt.Printf("[%s] 金额超出权限，转交上级\n", a.name)
	return a.PassToNext(doc)
}

// DirectorApprover 总监
type DirectorApprover struct {
	BaseApprover
	limit float64
}

func NewDirectorApprover(name string, limit float64) *DirectorApprover {
	return &DirectorApprover{
		BaseApprover: BaseApprover{name: name, level: "总监"},
		limit:        limit,
	}
}

func (a *DirectorApprover) Approve(doc *Document) (bool, error) {
	fmt.Printf("[%s] 审批文档: %s (金额: %.2f 元)\n", a.name, doc.Title, doc.Amount)
	
	if doc.Amount < a.limit {
		fmt.Printf("[%s] ✓ 审批通过\n", a.name)
		doc.AddHistory(fmt.Sprintf("%s (%s) 审批通过", a.name, a.level))
		return true, nil
	}
	
	fmt.Printf("[%s] 金额超出权限，转交上级\n", a.name)
	return a.PassToNext(doc)
}

// CEOApprover 总经理
type CEOApprover struct {
	BaseApprover
	limit float64
}

func NewCEOApprover(name string, limit float64) *CEOApprover {
	return &CEOApprover{
		BaseApprover: BaseApprover{name: name, level: "总经理"},
		limit:        limit,
	}
}

func (a *CEOApprover) Approve(doc *Document) (bool, error) {
	fmt.Printf("[%s] 审批文档: %s (金额: %.2f 元)\n", a.name, doc.Title, doc.Amount)
	
	if doc.Amount < a.limit {
		fmt.Printf("[%s] ✓ 审批通过\n", a.name)
		doc.AddHistory(fmt.Sprintf("%s (%s) 审批通过", a.name, a.level))
		return true, nil
	}
	
	return false, fmt.Errorf("金额过大，需要董事会审批")
}

func main() {
	fmt.Println("=== 练习1: 文档审批流程 ===\n")

	// 构建审批链
	manager := NewManagerApprover("张经理", 1000)
	director := NewDirectorApprover("李总监", 5000)
	ceo := NewCEOApprover("刘总经理", 10000)
	manager.SetNext(director).SetNext(ceo)

	// 场景1: 小额申请
	fmt.Println("--- 场景1: 小额申请 ---")
	doc1 := NewDocument("DOC001", "采购申请", 800, "张三", "技术部")
	doc1.Submit()
	doc1.Approve(manager)
	doc1.PrintHistory()

	// 场景2: 多级审批
	fmt.Println("\n\n--- 场景2: 多级审批 ---")
	doc2 := NewDocument("DOC002", "设备采购", 3500, "李四", "市场部")
	doc2.Submit()
	doc2.Approve(manager)
	doc2.PrintHistory()

	// 场景3: 审批被拒绝
	fmt.Println("\n\n--- 场景3: 审批被拒绝 ---")
	doc3 := NewDocument("DOC003", "差旅费用", 2000, "王五", "销售部")
	doc3.Submit()
	doc3.Reject(director, "预算不足")
	doc3.PrintHistory()

	// 场景4: 撤回申请
	fmt.Println("\n\n--- 场景4: 撤回申请 ---")
	doc4 := NewDocument("DOC004", "培训费用", 1500, "赵六", "人事部")
	doc4.Submit()
	doc4.Withdraw()
	doc4.PrintHistory()

	// 场景5: 非法操作
	fmt.Println("\n\n--- 场景5: 非法操作测试 ---")
	doc5 := NewDocument("DOC005", "测试文档", 500, "测试", "测试部")
	
	if err := doc5.Approve(manager); err != nil {
		fmt.Printf("错误: %v\n", err)
	}
	
	doc5.Submit()
	if err := doc5.Submit(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}

	fmt.Println("\n=== 练习完成 ===")
}

// 可能的优化方向:
// 1. 添加并行审批功能
// 2. 支持审批委托
// 3. 添加审批时限
// 4. 支持审批历史查询和统计
//
// 变体实现:
// 1. 使用事件驱动模式通知状态变更
// 2. 使用观察者模式监听审批进度
// 3. 添加审批模板，支持不同类型的审批流程
