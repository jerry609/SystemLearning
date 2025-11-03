package main

import "fmt"

func printOrderInfo(order *Order) {
	fmt.Printf("\n订单信息:\n")
	fmt.Printf("  ID: %s\n", order.ID)
	fmt.Printf("  金额: %.2f 元\n", order.Amount)
	fmt.Printf("  客户: %s\n", order.Customer)
	fmt.Printf("  当前状态: %s\n", order.GetState())
}

func printHistory(order *Order) {
	fmt.Println("\n订单状态历史:")
	for _, log := range order.GetHistory() {
		fmt.Printf("  %s\n", log)
	}
}

func main() {
	fmt.Println("=== 订单状态机系统 ===\n")

	// 场景1: 正常购买流程
	fmt.Println("--- 场景1: 正常购买流程 ---")
	order1 := NewOrder("ORD001", 299.99, "张三")
	printOrderInfo(order1)
	
	fmt.Println("\n操作: 支付订单")
	order1.Pay()
	fmt.Printf("当前状态: %s\n", order1.GetState())
	
	fmt.Println("\n操作: 发货")
	order1.Ship()
	fmt.Printf("当前状态: %s\n", order1.GetState())
	
	fmt.Println("\n操作: 确认收货")
	order1.Confirm()
	fmt.Printf("当前状态: %s\n", order1.GetState())
	
	printHistory(order1)

	// 场景2: 支付前取消
	fmt.Println("\n\n--- 场景2: 支付前取消订单 ---")
	order2 := NewOrder("ORD002", 199.99, "李四")
	printOrderInfo(order2)
	
	fmt.Println("\n操作: 取消订单")
	order2.Cancel()
	fmt.Printf("当前状态: %s\n", order2.GetState())
	
	printHistory(order2)

	// 场景3: 支付后申请退款
	fmt.Println("\n\n--- 场景3: 支付后申请退款 ---")
	order3 := NewOrder("ORD003", 399.99, "王五")
	printOrderInfo(order3)
	
	fmt.Println("\n操作: 支付订单")
	order3.Pay()
	fmt.Printf("当前状态: %s\n", order3.GetState())
	
	fmt.Println("\n操作: 申请退款")
	order3.Cancel()
	fmt.Printf("当前状态: %s\n", order3.GetState())
	
	fmt.Println("\n操作: 确认退款")
	order3.Confirm()
	fmt.Printf("当前状态: %s\n", order3.GetState())
	
	printHistory(order3)

	// 场景4: 发货后申请退货
	fmt.Println("\n\n--- 场景4: 发货后申请退货 ---")
	order4 := NewOrder("ORD004", 599.99, "赵六")
	printOrderInfo(order4)
	
	fmt.Println("\n操作: 支付订单")
	order4.Pay()
	
	fmt.Println("\n操作: 发货")
	order4.Ship()
	
	fmt.Println("\n操作: 申请退货")
	order4.Cancel()
	fmt.Printf("当前状态: %s\n", order4.GetState())
	
	fmt.Println("\n操作: 确认退货")
	order4.Confirm()
	fmt.Printf("当前状态: %s\n", order4.GetState())
	
	printHistory(order4)

	// 场景5: 完成后申请售后
	fmt.Println("\n\n--- 场景5: 完成后申请售后 ---")
	order5 := NewOrder("ORD005", 799.99, "孙七")
	printOrderInfo(order5)
	
	fmt.Println("\n操作: 支付订单")
	order5.Pay()
	
	fmt.Println("\n操作: 发货")
	order5.Ship()
	
	fmt.Println("\n操作: 确认收货")
	order5.Confirm()
	
	fmt.Println("\n操作: 申请售后")
	order5.Cancel()
	fmt.Printf("当前状态: %s\n", order5.GetState())
	
	fmt.Println("\n操作: 完成售后")
	order5.Confirm()
	fmt.Printf("当前状态: %s\n", order5.GetState())
	
	printHistory(order5)

	// 场景6: 非法操作测试
	fmt.Println("\n\n--- 场景6: 非法操作测试 ---")
	order6 := NewOrder("ORD006", 99.99, "周八")
	printOrderInfo(order6)
	
	fmt.Println("\n尝试在未支付时发货:")
	if err := order6.Ship(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}
	
	fmt.Println("\n支付订单:")
	order6.Pay()
	
	fmt.Println("\n尝试重复支付:")
	if err := order6.Pay(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}
	
	fmt.Println("\n尝试在未发货时确认收货:")
	if err := order6.Confirm(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}

	fmt.Println("\n\n=== 系统演示完成 ===")
}
