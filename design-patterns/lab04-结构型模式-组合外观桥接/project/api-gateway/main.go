package main

import "fmt"

func main() {
	fmt.Println("=== API 网关项目 ===")

	// 创建 API 网关
	gateway := NewAPIGateway()

	// 场景 1: 获取用户完整信息
	userID := "user-123"
	profile, err := gateway.GetUserProfile(userID)
	if err != nil {
		fmt.Printf("获取用户信息失败: %v\n", err)
	} else {
		fmt.Printf("用户资料: %v\n", profile)
	}

	// 场景 2: 下单
	items := []string{"商品A", "商品B"}
	amount := 899.99
	orderID, err := gateway.PlaceOrder(userID, items, amount)
	if err != nil {
		fmt.Printf("下单失败: %v\n", err)
	} else {
		fmt.Printf("订单号: %s\n", orderID)
	}

	// 场景 3: 查询订单状态
	if orderID != "" {
		orderStatus, err := gateway.GetOrderStatus(userID, orderID)
		if err != nil {
			fmt.Printf("查询订单失败: %v\n", err)
		} else {
			fmt.Printf("订单状态: %v\n", orderStatus)
		}
	}

	// 场景 4: 取消订单
	if orderID != "" {
		err = gateway.CancelOrder(userID, orderID)
		if err != nil {
			fmt.Printf("取消订单失败: %v\n", err)
		}
	}

	fmt.Println("=== 项目演示结束 ===")

	// 说明外观模式的优势
	fmt.Println("\n💡 外观模式在 API 网关中的应用")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("1. 简化客户端调用")
	fmt.Println("   - 客户端只需调用网关的高层接口")
	fmt.Println("   - 不需要了解各个微服务的细节")
	fmt.Println()
	fmt.Println("2. 统一入口")
	fmt.Println("   - 所有请求通过网关统一处理")
	fmt.Println("   - 便于实现认证、限流、日志等功能")
	fmt.Println()
	fmt.Println("3. 业务编排")
	fmt.Println("   - 协调多个微服务完成复杂业务")
	fmt.Println("   - 处理服务间的依赖关系")
	fmt.Println()
	fmt.Println("4. 降低耦合")
	fmt.Println("   - 客户端与微服务解耦")
	fmt.Println("   - 微服务的变化不影响客户端")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
}
