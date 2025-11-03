package main

import (
	"fmt"
	"time"
)

// 订单状态机示例
// 本示例展示了状态模式在订单管理中的应用

// OrderState 订单状态接口
type OrderState interface {
	// Pay 支付
	Pay(order *Order) error
	// Ship 发货
	Ship(order *Order) error
	// Confirm 确认收货
	Confirm(order *Order) error
	// Cancel 取消订单
	Cancel(order *Order) error
	// String 状态名称
	String() string
}

// Order 订单上下文
type Order struct {
	ID          string
	Amount      float64
	state       OrderState
	history     []string // 状态历史
	createdTime time.Time
}

func NewOrder(id string, amount float64) *Order {
	order := &Order{
		ID:          id,
		Amount:      amount,
		history:     make([]string, 0),
		createdTime: time.Now(),
	}
	order.SetState(&PendingState{})
	return order
}

func (o *Order) SetState(state OrderState) {
	if o.state != nil {
		o.history = append(o.history, fmt.Sprintf("%s -> %s", o.state.String(), state.String()))
	}
	o.state = state
	fmt.Printf("[订单 %s] 状态变更: %s\n", o.ID, state.String())
}

func (o *Order) GetState() string {
	return o.state.String()
}

func (o *Order) Pay() error {
	return o.state.Pay(o)
}

func (o *Order) Ship() error {
	return o.state.Ship(o)
}

func (o *Order) Confirm() error {
	return o.state.Confirm(o)
}

func (o *Order) Cancel() error {
	return o.state.Cancel(o)
}

func (o *Order) PrintHistory() {
	fmt.Printf("\n[订单 %s] 状态历史:\n", o.ID)
	for i, h := range o.history {
		fmt.Printf("  %d. %s\n", i+1, h)
	}
}

// PendingState 待支付状态
type PendingState struct{}

func (s *PendingState) Pay(order *Order) error {
	fmt.Printf("[订单 %s] 支付成功，金额: %.2f 元\n", order.ID, order.Amount)
	order.SetState(&PaidState{})
	return nil
}

func (s *PendingState) Ship(order *Order) error {
	return fmt.Errorf("订单未支付，无法发货")
}

func (s *PendingState) Confirm(order *Order) error {
	return fmt.Errorf("订单未支付，无法确认收货")
}

func (s *PendingState) Cancel(order *Order) error {
	fmt.Printf("[订单 %s] 取消订单\n", order.ID)
	order.SetState(&CancelledState{})
	return nil
}

func (s *PendingState) String() string {
	return "待支付"
}

// PaidState 已支付状态
type PaidState struct{}

func (s *PaidState) Pay(order *Order) error {
	return fmt.Errorf("订单已支付，无需重复支付")
}

func (s *PaidState) Ship(order *Order) error {
	fmt.Printf("[订单 %s] 开始发货\n", order.ID)
	order.SetState(&ShippedState{})
	return nil
}

func (s *PaidState) Confirm(order *Order) error {
	return fmt.Errorf("订单未发货，无法确认收货")
}

func (s *PaidState) Cancel(order *Order) error {
	fmt.Printf("[订单 %s] 申请退款\n", order.ID)
	order.SetState(&RefundingState{})
	return nil
}

func (s *PaidState) String() string {
	return "已支付"
}

// ShippedState 已发货状态
type ShippedState struct{}

func (s *ShippedState) Pay(order *Order) error {
	return fmt.Errorf("订单已支付")
}

func (s *ShippedState) Ship(order *Order) error {
	return fmt.Errorf("订单已发货")
}

func (s *ShippedState) Confirm(order *Order) error {
	fmt.Printf("[订单 %s] 确认收货\n", order.ID)
	order.SetState(&CompletedState{})
	return nil
}

func (s *ShippedState) Cancel(order *Order) error {
	fmt.Printf("[订单 %s] 申请退货\n", order.ID)
	order.SetState(&ReturningState{})
	return nil
}

func (s *ShippedState) String() string {
	return "已发货"
}

// CompletedState 已完成状态
type CompletedState struct{}

func (s *CompletedState) Pay(order *Order) error {
	return fmt.Errorf("订单已完成")
}

func (s *CompletedState) Ship(order *Order) error {
	return fmt.Errorf("订单已完成")
}

func (s *CompletedState) Confirm(order *Order) error {
	return fmt.Errorf("订单已确认收货")
}

func (s *CompletedState) Cancel(order *Order) error {
	fmt.Printf("[订单 %s] 申请售后\n", order.ID)
	order.SetState(&AfterSaleState{})
	return nil
}

func (s *CompletedState) String() string {
	return "已完成"
}

// CancelledState 已取消状态
type CancelledState struct{}

func (s *CancelledState) Pay(order *Order) error {
	return fmt.Errorf("订单已取消，无法支付")
}

func (s *CancelledState) Ship(order *Order) error {
	return fmt.Errorf("订单已取消，无法发货")
}

func (s *CancelledState) Confirm(order *Order) error {
	return fmt.Errorf("订单已取消，无法确认收货")
}

func (s *CancelledState) Cancel(order *Order) error {
	return fmt.Errorf("订单已取消")
}

func (s *CancelledState) String() string {
	return "已取消"
}

// RefundingState 退款中状态
type RefundingState struct{}

func (s *RefundingState) Pay(order *Order) error {
	return fmt.Errorf("订单退款中")
}

func (s *RefundingState) Ship(order *Order) error {
	return fmt.Errorf("订单退款中")
}

func (s *RefundingState) Confirm(order *Order) error {
	fmt.Printf("[订单 %s] 退款完成\n", order.ID)
	order.SetState(&RefundedState{})
	return nil
}

func (s *RefundingState) Cancel(order *Order) error {
	return fmt.Errorf("订单退款中")
}

func (s *RefundingState) String() string {
	return "退款中"
}

// RefundedState 已退款状态
type RefundedState struct{}

func (s *RefundedState) Pay(order *Order) error {
	return fmt.Errorf("订单已退款")
}

func (s *RefundedState) Ship(order *Order) error {
	return fmt.Errorf("订单已退款")
}

func (s *RefundedState) Confirm(order *Order) error {
	return fmt.Errorf("订单已退款")
}

func (s *RefundedState) Cancel(order *Order) error {
	return fmt.Errorf("订单已退款")
}

func (s *RefundedState) String() string {
	return "已退款"
}

// ReturningState 退货中状态
type ReturningState struct{}

func (s *ReturningState) Pay(order *Order) error {
	return fmt.Errorf("订单退货中")
}

func (s *ReturningState) Ship(order *Order) error {
	return fmt.Errorf("订单退货中")
}

func (s *ReturningState) Confirm(order *Order) error {
	fmt.Printf("[订单 %s] 退货完成\n", order.ID)
	order.SetState(&RefundedState{})
	return nil
}

func (s *ReturningState) Cancel(order *Order) error {
	return fmt.Errorf("订单退货中")
}

func (s *ReturningState) String() string {
	return "退货中"
}

// AfterSaleState 售后中状态
type AfterSaleState struct{}

func (s *AfterSaleState) Pay(order *Order) error {
	return fmt.Errorf("订单售后中")
}

func (s *AfterSaleState) Ship(order *Order) error {
	return fmt.Errorf("订单售后中")
}

func (s *AfterSaleState) Confirm(order *Order) error {
	fmt.Printf("[订单 %s] 售后完成\n", order.ID)
	order.SetState(&CompletedState{})
	return nil
}

func (s *AfterSaleState) Cancel(order *Order) error {
	return fmt.Errorf("订单售后中")
}

func (s *AfterSaleState) String() string {
	return "售后中"
}

func main() {
	fmt.Println("=== 状态模式示例 - 订单状态机 ===\n")

	// 场景1: 正常流程
	fmt.Println("--- 场景1: 正常购买流程 ---")
	order1 := NewOrder("ORD001", 299.99)
	fmt.Printf("初始状态: %s\n\n", order1.GetState())

	order1.Pay()
	fmt.Println()

	order1.Ship()
	fmt.Println()

	order1.Confirm()
	order1.PrintHistory()
	fmt.Println()

	// 场景2: 支付前取消
	fmt.Println("\n--- 场景2: 支付前取消订单 ---")
	order2 := NewOrder("ORD002", 199.99)
	order2.Cancel()
	order2.PrintHistory()
	fmt.Println()

	// 场景3: 支付后申请退款
	fmt.Println("\n--- 场景3: 支付后申请退款 ---")
	order3 := NewOrder("ORD003", 399.99)
	order3.Pay()
	fmt.Println()

	order3.Cancel() // 申请退款
	fmt.Println()

	order3.Confirm() // 退款完成
	order3.PrintHistory()
	fmt.Println()

	// 场景4: 发货后申请退货
	fmt.Println("\n--- 场景4: 发货后申请退货 ---")
	order4 := NewOrder("ORD004", 599.99)
	order4.Pay()
	fmt.Println()

	order4.Ship()
	fmt.Println()

	order4.Cancel() // 申请退货
	fmt.Println()

	order4.Confirm() // 退货完成
	order4.PrintHistory()
	fmt.Println()

	// 场景5: 完成后申请售后
	fmt.Println("\n--- 场景5: 完成后申请售后 ---")
	order5 := NewOrder("ORD005", 799.99)
	order5.Pay()
	fmt.Println()

	order5.Ship()
	fmt.Println()

	order5.Confirm()
	fmt.Println()

	order5.Cancel() // 申请售后
	fmt.Println()

	order5.Confirm() // 售后完成
	order5.PrintHistory()
	fmt.Println()

	// 场景6: 非法操作
	fmt.Println("\n--- 场景6: 非法操作测试 ---")
	order6 := NewOrder("ORD006", 99.99)

	// 尝试在未支付时发货
	if err := order6.Ship(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}

	// 尝试重复支付
	order6.Pay()
	if err := order6.Pay(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}

	fmt.Println("\n=== 示例结束 ===")
}
