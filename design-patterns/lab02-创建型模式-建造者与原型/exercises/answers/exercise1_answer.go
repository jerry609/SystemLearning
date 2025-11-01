package main

import (
	"fmt"
	"strings"
)

// 练习 1: SQL 查询构建器 - 参考答案
//
// 设计思路:
// 1. 使用 Query 结构体存储查询的各个部分
// 2. QueryBuilder 提供链式调用的方法
// 3. Build 方法组装最终的 SQL 语句
// 4. 使用参数化查询防止 SQL 注入
//
// 使用的设计模式: 建造者模式
// 模式应用位置: QueryBuilder 使用链式调用构建复杂的 SQL 查询

// Query 表示一个 SQL 查询的各个组成部分
type Query struct {
	table   string
	columns []string
	joins   []Join
	where   []string
	groupBy []string
	having  []string
	orderBy string
	limit   int
	offset  int
	args    []interface{}
}

// Join 表示一个 JOIN 操作
type Join struct {
	joinType  string // INNER, LEFT, RIGHT
	table     string
	condition string
}

// QueryBuilder 是 SQL 查询构建器
type QueryBuilder struct {
	query *Query
}

// NewQueryBuilder 创建一个新的查询构建器
func NewQueryBuilder(table string) *QueryBuilder {
	return &QueryBuilder{
		query: &Query{
			table:   table,
			columns: []string{"*"},
			args:    make([]interface{}, 0),
		},
	}
}

// Select 设置要查询的列
func (b *QueryBuilder) Select(columns ...string) *QueryBuilder {
	b.query.columns = columns
	return b
}

// Where 添加 WHERE 条件
func (b *QueryBuilder) Where(condition string, args ...interface{}) *QueryBuilder {
	b.query.where = append(b.query.where, condition)
	b.query.args = append(b.query.args, args...)
	return b
}

// InnerJoin 添加 INNER JOIN
func (b *QueryBuilder) InnerJoin(table, condition string) *QueryBuilder {
	b.query.joins = append(b.query.joins, Join{
		joinType:  "INNER JOIN",
		table:     table,
		condition: condition,
	})
	return b
}

// LeftJoin 添加 LEFT JOIN
func (b *QueryBuilder) LeftJoin(table, condition string) *QueryBuilder {
	b.query.joins = append(b.query.joins, Join{
		joinType:  "LEFT JOIN",
		table:     table,
		condition: condition,
	})
	return b
}

// RightJoin 添加 RIGHT JOIN
func (b *QueryBuilder) RightJoin(table, condition string) *QueryBuilder {
	b.query.joins = append(b.query.joins, Join{
		joinType:  "RIGHT JOIN",
		table:     table,
		condition: condition,
	})
	return b
}

// GroupBy 添加 GROUP BY
func (b *QueryBuilder) GroupBy(columns ...string) *QueryBuilder {
	b.query.groupBy = append(b.query.groupBy, columns...)
	return b
}

// Having 添加 HAVING 条件
func (b *QueryBuilder) Having(condition string, args ...interface{}) *QueryBuilder {
	b.query.having = append(b.query.having, condition)
	b.query.args = append(b.query.args, args...)
	return b
}

// OrderBy 设置排序
func (b *QueryBuilder) OrderBy(orderBy string) *QueryBuilder {
	b.query.orderBy = orderBy
	return b
}

// Limit 设置返回记录数
func (b *QueryBuilder) Limit(limit int) *QueryBuilder {
	b.query.limit = limit
	return b
}

// Offset 设置偏移量
func (b *QueryBuilder) Offset(offset int) *QueryBuilder {
	b.query.offset = offset
	return b
}

// Build 构建最终的 SQL 语句
func (b *QueryBuilder) Build() string {
	var parts []string

	// SELECT
	parts = append(parts, fmt.Sprintf("SELECT %s", strings.Join(b.query.columns, ", ")))

	// FROM
	parts = append(parts, fmt.Sprintf("FROM %s", b.query.table))

	// JOIN
	for _, join := range b.query.joins {
		parts = append(parts, fmt.Sprintf("%s %s ON %s", join.joinType, join.table, join.condition))
	}

	// WHERE
	if len(b.query.where) > 0 {
		parts = append(parts, fmt.Sprintf("WHERE %s", strings.Join(b.query.where, " AND ")))
	}

	// GROUP BY
	if len(b.query.groupBy) > 0 {
		parts = append(parts, fmt.Sprintf("GROUP BY %s", strings.Join(b.query.groupBy, ", ")))
	}

	// HAVING
	if len(b.query.having) > 0 {
		parts = append(parts, fmt.Sprintf("HAVING %s", strings.Join(b.query.having, " AND ")))
	}

	// ORDER BY
	if b.query.orderBy != "" {
		parts = append(parts, fmt.Sprintf("ORDER BY %s", b.query.orderBy))
	}

	// LIMIT
	if b.query.limit > 0 {
		parts = append(parts, fmt.Sprintf("LIMIT %d", b.query.limit))
	}

	// OFFSET
	if b.query.offset > 0 {
		parts = append(parts, fmt.Sprintf("OFFSET %d", b.query.offset))
	}

	return strings.Join(parts, " ")
}

// GetArgs 返回查询参数
func (b *QueryBuilder) GetArgs() []interface{} {
	return b.query.args
}

func main() {
	fmt.Println("=== SQL 查询构建器 - 参考答案 ===\n")

	// 示例 1: 简单查询
	fmt.Println("示例 1: 简单查询")
	sql1 := NewQueryBuilder("users").
		Select("id", "name", "email").
		Where("age > ?", 18).
		OrderBy("created_at DESC").
		Limit(10).
		Build()
	fmt.Println(sql1)
	fmt.Println()

	// 示例 2: 带 JOIN 的查询
	fmt.Println("示例 2: 带 JOIN 的查询")
	builder2 := NewQueryBuilder("users").
		Select("users.id", "users.name", "orders.total").
		InnerJoin("orders", "users.id = orders.user_id").
		Where("orders.status = ?", "completed").
		GroupBy("users.id").
		Having("COUNT(orders.id) > ?", 5)
	sql2 := builder2.Build()
	fmt.Println(sql2)
	fmt.Printf("参数: %v\n", builder2.GetArgs())
	fmt.Println()

	// 示例 3: 聚合查询
	fmt.Println("示例 3: 聚合查询")
	builder3 := NewQueryBuilder("products").
		Select("category", "COUNT(*) as count", "AVG(price) as avg_price").
		Where("status = ?", "active").
		GroupBy("category").
		OrderBy("count DESC")
	sql3 := builder3.Build()
	fmt.Println(sql3)
	fmt.Printf("参数: %v\n", builder3.GetArgs())
	fmt.Println()

	// 示例 4: 复杂查询
	fmt.Println("示例 4: 复杂查询")
	builder4 := NewQueryBuilder("users").
		Select("users.id", "users.name", "COUNT(orders.id) as order_count", "SUM(orders.total) as total_amount").
		LeftJoin("orders", "users.id = orders.user_id").
		Where("users.status = ?", "active").
		Where("users.created_at > ?", "2024-01-01").
		GroupBy("users.id", "users.name").
		Having("COUNT(orders.id) > ?", 3).
		OrderBy("total_amount DESC").
		Limit(20).
		Offset(10)
	sql4 := builder4.Build()
	fmt.Println(sql4)
	fmt.Printf("参数: %v\n", builder4.GetArgs())
	fmt.Println()

	// 示例 5: 多个 JOIN
	fmt.Println("示例 5: 多个 JOIN")
	sql5 := NewQueryBuilder("users").
		Select("users.name", "orders.id as order_id", "products.name as product_name").
		InnerJoin("orders", "users.id = orders.user_id").
		InnerJoin("order_items", "orders.id = order_items.order_id").
		InnerJoin("products", "order_items.product_id = products.id").
		Where("orders.status = ?", "completed").
		OrderBy("orders.created_at DESC").
		Limit(50).
		Build()
	fmt.Println(sql5)

	fmt.Println("\n=== 示例结束 ===")
}

// 可能的优化方向:
// 1. 添加 SQL 注入防护的验证
// 2. 支持更多的 SQL 操作（UNION, DISTINCT, etc.）
// 3. 添加 SQL 格式化功能
// 4. 支持子查询
// 5. 添加查询缓存
//
// 变体实现:
// 1. 使用接口定义 Builder，支持不同的 SQL 方言
// 2. 添加查询优化器
// 3. 支持 ORM 功能
