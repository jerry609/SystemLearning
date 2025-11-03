package main

import (
	"testing"
)

func TestOrderNormalFlow(t *testing.T) {
	order := NewOrder("TEST001", 100.0, "测试用户")
	
	if order.GetState() != "待支付" {
		t.Errorf("初始状态应该是待支付，实际是 %s", order.GetState())
	}
	
	if err := order.Pay(); err != nil {
		t.Errorf("支付失败: %v", err)
	}
	if order.GetState() != "已支付" {
		t.Errorf("支付后状态应该是已支付，实际是 %s", order.GetState())
	}
	
	if err := order.Ship(); err != nil {
		t.Errorf("发货失败: %v", err)
	}
	if order.GetState() != "已发货" {
		t.Errorf("发货后状态应该是已发货，实际是 %s", order.GetState())
	}
	
	if err := order.Confirm(); err != nil {
		t.Errorf("确认收货失败: %v", err)
	}
	if order.GetState() != "已完成" {
		t.Errorf("确认收货后状态应该是已完成，实际是 %s", order.GetState())
	}
}

func TestOrderCancelBeforePay(t *testing.T) {
	order := NewOrder("TEST002", 100.0, "测试用户")
	
	if err := order.Cancel(); err != nil {
		t.Errorf("取消失败: %v", err)
	}
	if order.GetState() != "已取消" {
		t.Errorf("取消后状态应该是已取消，实际是 %s", order.GetState())
	}
}

func TestOrderRefundAfterPay(t *testing.T) {
	order := NewOrder("TEST003", 100.0, "测试用户")
	
	order.Pay()
	
	if err := order.Cancel(); err != nil {
		t.Errorf("申请退款失败: %v", err)
	}
	if order.GetState() != "退款中" {
		t.Errorf("申请退款后状态应该是退款中，实际是 %s", order.GetState())
	}
	
	if err := order.Confirm(); err != nil {
		t.Errorf("确认退款失败: %v", err)
	}
	if order.GetState() != "已退款" {
		t.Errorf("确认退款后状态应该是已退款，实际是 %s", order.GetState())
	}
}

func TestOrderReturnAfterShip(t *testing.T) {
	order := NewOrder("TEST004", 100.0, "测试用户")
	
	order.Pay()
	order.Ship()
	
	if err := order.Cancel(); err != nil {
		t.Errorf("申请退货失败: %v", err)
	}
	if order.GetState() != "退货中" {
		t.Errorf("申请退货后状态应该是退货中，实际是 %s", order.GetState())
	}
	
	if err := order.Confirm(); err != nil {
		t.Errorf("确认退货失败: %v", err)
	}
	if order.GetState() != "已退款" {
		t.Errorf("确认退货后状态应该是已退款，实际是 %s", order.GetState())
	}
}

func TestOrderAfterSale(t *testing.T) {
	order := NewOrder("TEST005", 100.0, "测试用户")
	
	order.Pay()
	order.Ship()
	order.Confirm()
	
	if err := order.Cancel(); err != nil {
		t.Errorf("申请售后失败: %v", err)
	}
	if order.GetState() != "售后中" {
		t.Errorf("申请售后后状态应该是售后中，实际是 %s", order.GetState())
	}
	
	if err := order.Confirm(); err != nil {
		t.Errorf("完成售后失败: %v", err)
	}
	if order.GetState() != "已完成" {
		t.Errorf("完成售后后状态应该是已完成，实际是 %s", order.GetState())
	}
}

func TestInvalidOperations(t *testing.T) {
	order := NewOrder("TEST006", 100.0, "测试用户")
	
	// 未支付时发货
	if err := order.Ship(); err == nil {
		t.Error("未支付时发货应该失败")
	}
	
	// 未支付时确认收货
	if err := order.Confirm(); err == nil {
		t.Error("未支付时确认收货应该失败")
	}
	
	order.Pay()
	
	// 重复支付
	if err := order.Pay(); err == nil {
		t.Error("重复支付应该失败")
	}
	
	// 未发货时确认收货
	if err := order.Confirm(); err == nil {
		t.Error("未发货时确认收货应该失败")
	}
}

func TestOrderHistory(t *testing.T) {
	order := NewOrder("TEST007", 100.0, "测试用户")
	
	order.Pay()
	order.Ship()
	order.Confirm()
	
	history := order.GetHistory()
	if len(history) != 3 {
		t.Errorf("应该有3条历史记录，实际有 %d 条", len(history))
	}
}
