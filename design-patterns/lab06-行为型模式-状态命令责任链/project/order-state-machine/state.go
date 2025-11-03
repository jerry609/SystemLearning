package main

import (
	"fmt"
	"time"
)

// OrderState 订单状态接口
type OrderState interface {
	Pay(order *Order) error
	Ship(order *Order) error
	Confirm(order *Order) error
	Cancel(order *Order) error
	String() string
}

// Order 订单
type Order struct {
	ID          string
	Amount      float64
	Customer    string
	state       OrderState
	history     []string
	createdTime time.Time
}

func NewOrder(id string, amount float64, customer string) *Order {
	order := &Order{
		ID:          id,
		Amount:      amount,
		Customer:    customer,
		history:     make([]string, 0),
		createdTime: time.Now(),
	}
	order.SetState(&PendingState{})
	return order
}

func (o *Order) SetState(state OrderState) {
	if o.state != nil {
		o.addHistory(fmt.Sprintf("状态变更: %s -> %s", o.state.String(), state.String()))
	}
	o.state = state
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

func (o *Order) addHistory(log string) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	o.history = append(o.history, fmt.Sprintf("[%s] %s", timestamp, log))
}

func (o *Order) GetHistory() []string {
	return o.history
}

// PendingState 待支付状态
type PendingState struct{}

func (s *PendingState) Pay(order *Order) error {
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
	order.SetState(&CancelledState{})
	return nil
}

func (s *PendingState) String() string {
	return "待支付"
}

// PaidState 已支付状态
type PaidState struct{}

func (s *PaidState) Pay(order *Order) error {
	return fmt.Errorf("订单已支付")
}

func (s *PaidState) Ship(order *Order) error {
	order.SetState(&ShippedState{})
	return nil
}

func (s *PaidState) Confirm(order *Order) error {
	return fmt.Errorf("订单未发货，无法确认收货")
}

func (s *PaidState) Cancel(order *Order) error {
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
	order.SetState(&CompletedState{})
	return nil
}

func (s *ShippedState) Cancel(order *Order) error {
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
	order.SetState(&AfterSaleState{})
	return nil
}

func (s *CompletedState) String() string {
	return "已完成"
}

// CancelledState 已取消状态
type CancelledState struct{}

func (s *CancelledState) Pay(order *Order) error {
	return fmt.Errorf("订单已取消")
}

func (s *CancelledState) Ship(order *Order) error {
	return fmt.Errorf("订单已取消")
}

func (s *CancelledState) Confirm(order *Order) error {
	return fmt.Errorf("订单已取消")
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
	order.SetState(&CompletedState{})
	return nil
}

func (s *AfterSaleState) Cancel(order *Order) error {
	return fmt.Errorf("订单售后中")
}

func (s *AfterSaleState) String() string {
	return "售后中"
}
