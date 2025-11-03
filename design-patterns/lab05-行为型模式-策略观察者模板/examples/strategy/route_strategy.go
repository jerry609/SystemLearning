package main

import (
	"fmt"
	"math"
)

// 策略模式示例：路由策略
// 本示例展示了如何使用策略模式实现多种路由算法

// Location 位置坐标
type Location struct {
	Name      string
	Latitude  float64
	Longitude float64
}

// Route 路由信息
type Route struct {
	Path     []string
	Distance float64
	Duration float64
	Cost     float64
}

// RouteStrategy 路由策略接口
type RouteStrategy interface {
	CalculateRoute(start, end Location) Route
	GetName() string
}

// ShortestPathStrategy 最短路径策略
// 优先考虑距离最短
type ShortestPathStrategy struct{}

func (s *ShortestPathStrategy) CalculateRoute(start, end Location) Route {
	distance := calculateDistance(start, end)
	duration := distance / 60.0 // 假设平均速度 60 km/h
	cost := distance * 0.5       // 假设每公里 0.5 元

	return Route{
		Path:     []string{start.Name, end.Name},
		Distance: distance,
		Duration: duration,
		Cost:     cost,
	}
}

func (s *ShortestPathStrategy) GetName() string {
	return "最短路径"
}

// FastestPathStrategy 最快路径策略
// 优先考虑时间最短
type FastestPathStrategy struct{}

func (f *FastestPathStrategy) CalculateRoute(start, end Location) Route {
	distance := calculateDistance(start, end) * 1.2 // 高速路可能绕远
	duration := distance / 100.0                     // 高速平均速度 100 km/h
	cost := distance*0.5 + 15.0                      // 加上高速费

	return Route{
		Path:     []string{start.Name, "高速入口", "高速出口", end.Name},
		Distance: distance,
		Duration: duration,
		Cost:     cost,
	}
}

func (f *FastestPathStrategy) GetName() string {
	return "最快路径"
}

// CheapestPathStrategy 最省钱路径策略
// 优先考虑费用最低
type CheapestPathStrategy struct{}

func (c *CheapestPathStrategy) CalculateRoute(start, end Location) Route {
	distance := calculateDistance(start, end) * 1.1 // 可能绕路避开收费站
	duration := distance / 50.0                      // 国道平均速度 50 km/h
	cost := distance * 0.3                           // 国道费用更低

	return Route{
		Path:     []string{start.Name, "国道", end.Name},
		Distance: distance,
		Duration: duration,
		Cost:     cost,
	}
}

func (c *CheapestPathStrategy) GetName() string {
	return "最省钱路径"
}

// ScenicPathStrategy 风景路线策略
// 优先考虑沿途风景
type ScenicPathStrategy struct{}

func (s *ScenicPathStrategy) CalculateRoute(start, end Location) Route {
	distance := calculateDistance(start, end) * 1.5 // 风景路线通常较远
	duration := distance / 40.0                      // 慢速欣赏风景
	cost := distance * 0.4

	return Route{
		Path:     []string{start.Name, "景点A", "景点B", "景点C", end.Name},
		Distance: distance,
		Duration: duration,
		Cost:     cost,
	}
}

func (s *ScenicPathStrategy) GetName() string {
	return "风景路线"
}

// AvoidTrafficStrategy 避开拥堵策略
// 实时避开拥堵路段
type AvoidTrafficStrategy struct {
	TrafficLevel int // 拥堵等级 1-5
}

func (a *AvoidTrafficStrategy) CalculateRoute(start, end Location) Route {
	distance := calculateDistance(start, end) * 1.3 // 绕路避开拥堵
	// 根据拥堵等级调整时间
	baseSpeed := 60.0 - float64(a.TrafficLevel)*5.0
	duration := distance / baseSpeed
	cost := distance * 0.5

	return Route{
		Path:     []string{start.Name, "绕行路线", end.Name},
		Distance: distance,
		Duration: duration,
		Cost:     cost,
	}
}

func (a *AvoidTrafficStrategy) GetName() string {
	return "避开拥堵"
}

// RouteNavigator 路由导航器（上下文）
type RouteNavigator struct {
	strategy RouteStrategy
}

// NewRouteNavigator 创建路由导航器
func NewRouteNavigator(strategy RouteStrategy) *RouteNavigator {
	return &RouteNavigator{
		strategy: strategy,
	}
}

// SetStrategy 设置路由策略
func (r *RouteNavigator) SetStrategy(strategy RouteStrategy) {
	r.strategy = strategy
}

// Navigate 执行导航
func (r *RouteNavigator) Navigate(start, end Location) Route {
	if r.strategy == nil {
		return Route{}
	}
	return r.strategy.CalculateRoute(start, end)
}

// GetStrategyName 获取当前策略名称
func (r *RouteNavigator) GetStrategyName() string {
	if r.strategy == nil {
		return "未设置策略"
	}
	return r.strategy.GetName()
}

// calculateDistance 计算两点之间的距离（简化版）
// 使用 Haversine 公式计算球面距离
func calculateDistance(loc1, loc2 Location) float64 {
	const earthRadius = 6371.0 // 地球半径（公里）

	lat1 := loc1.Latitude * math.Pi / 180
	lat2 := loc2.Latitude * math.Pi / 180
	deltaLat := (loc2.Latitude - loc1.Latitude) * math.Pi / 180
	deltaLon := (loc2.Longitude - loc1.Longitude) * math.Pi / 180

	a := math.Sin(deltaLat/2)*math.Sin(deltaLat/2) +
		math.Cos(lat1)*math.Cos(lat2)*
			math.Sin(deltaLon/2)*math.Sin(deltaLon/2)

	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return earthRadius * c
}

// printRoute 打印路由信息
func printRoute(route Route, strategyName string) {
	fmt.Printf("\n【%s】\n", strategyName)
	fmt.Printf("  路径: %v\n", route.Path)
	fmt.Printf("  距离: %.2f 公里\n", route.Distance)
	fmt.Printf("  时间: %.2f 小时 (%.0f 分钟)\n", route.Duration, route.Duration*60)
	fmt.Printf("  费用: ¥%.2f\n", route.Cost)
}

func main() {
	fmt.Println("=== 策略模式示例：路由策略 ===\n")

	// 定义起点和终点
	start := Location{
		Name:      "北京",
		Latitude:  39.9042,
		Longitude: 116.4074,
	}

	end := Location{
		Name:      "上海",
		Latitude:  31.2304,
		Longitude: 121.4737,
	}

	fmt.Printf("起点: %s (%.4f, %.4f)\n", start.Name, start.Latitude, start.Longitude)
	fmt.Printf("终点: %s (%.4f, %.4f)\n", end.Name, end.Latitude, end.Longitude)
	fmt.Println("\n========================================")

	// 创建导航器
	navigator := NewRouteNavigator(nil)

	// 测试不同的路由策略
	strategies := []RouteStrategy{
		&ShortestPathStrategy{},
		&FastestPathStrategy{},
		&CheapestPathStrategy{},
		&ScenicPathStrategy{},
		&AvoidTrafficStrategy{TrafficLevel: 3},
	}

	fmt.Println("\n【所有路由方案对比】")
	for _, strategy := range strategies {
		navigator.SetStrategy(strategy)
		route := navigator.Navigate(start, end)
		printRoute(route, strategy.GetName())
	}

	// 根据用户偏好选择策略
	fmt.Println("\n========================================")
	fmt.Println("\n【根据用户偏好选择路由】")

	// 场景 1: 商务出行，时间优先
	fmt.Println("\n场景 1: 商务出行")
	navigator.SetStrategy(&FastestPathStrategy{})
	route1 := navigator.Navigate(start, end)
	printRoute(route1, "推荐方案: "+navigator.GetStrategyName())

	// 场景 2: 自驾游，风景优先
	fmt.Println("\n场景 2: 自驾游")
	navigator.SetStrategy(&ScenicPathStrategy{})
	route2 := navigator.Navigate(start, end)
	printRoute(route2, "推荐方案: "+navigator.GetStrategyName())

	// 场景 3: 预算有限，费用优先
	fmt.Println("\n场景 3: 预算有限")
	navigator.SetStrategy(&CheapestPathStrategy{})
	route3 := navigator.Navigate(start, end)
	printRoute(route3, "推荐方案: "+navigator.GetStrategyName())

	// 场景 4: 高峰时段，避开拥堵
	fmt.Println("\n场景 4: 高峰时段")
	navigator.SetStrategy(&AvoidTrafficStrategy{TrafficLevel: 4})
	route4 := navigator.Navigate(start, end)
	printRoute(route4, "推荐方案: "+navigator.GetStrategyName())

	// 智能推荐：根据多个因素综合评分
	fmt.Println("\n========================================")
	fmt.Println("\n【智能推荐系统】")

	type RouteScore struct {
		Strategy RouteStrategy
		Route    Route
		Score    float64
	}

	// 用户偏好权重
	preferences := map[string]float64{
		"distance": 0.3, // 距离权重
		"duration": 0.5, // 时间权重
		"cost":     0.2, // 费用权重
	}

	fmt.Printf("\n用户偏好权重:\n")
	fmt.Printf("  距离: %.0f%%\n", preferences["distance"]*100)
	fmt.Printf("  时间: %.0f%%\n", preferences["duration"]*100)
	fmt.Printf("  费用: %.0f%%\n", preferences["cost"]*100)

	// 计算每个策略的综合评分
	var scores []RouteScore
	for _, strategy := range strategies {
		navigator.SetStrategy(strategy)
		route := navigator.Navigate(start, end)

		// 归一化并计算综合评分（越低越好）
		score := route.Distance*preferences["distance"] +
			route.Duration*100*preferences["duration"] +
			route.Cost*preferences["cost"]

		scores = append(scores, RouteScore{
			Strategy: strategy,
			Route:    route,
			Score:    score,
		})
	}

	// 找出评分最低（最优）的方案
	bestScore := scores[0]
	for _, s := range scores {
		if s.Score < bestScore.Score {
			bestScore = s
		}
	}

	fmt.Println("\n所有方案评分:")
	for _, s := range scores {
		fmt.Printf("  %s: %.2f 分", s.Strategy.GetName(), s.Score)
		if s.Strategy.GetName() == bestScore.Strategy.GetName() {
			fmt.Printf(" ⭐ 最优方案")
		}
		fmt.Println()
	}

	fmt.Println("\n最优推荐:")
	printRoute(bestScore.Route, bestScore.Strategy.GetName())

	fmt.Println("\n=== 示例结束 ===")
}
