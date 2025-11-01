package main

import (
	"fmt"
	"strings"
)

// 链式调用建造者模式示例
// 本示例展示了如何使用链式调用构建复杂的 SQL 查询

// Query 表示一个 SQL 查询
type Query struct {
	table   string
	columns []string
	where   []string
	orderBy string
	limit   int
	offset  int
}

// QueryBuilder 是查询构建器
type QueryBuilder struct {
	query *Query
}

// NewQueryBuilder 创建一个新的查询构建器
func NewQueryBuilder(table string) *QueryBuilder {
	return &QueryBuilder{
		query: &Query{
			table:   table,
			columns: []string{"*"},
		},
	}
}

// Select 设置要查询的列
func (b *QueryBuilder) Select(columns ...string) *QueryBuilder {
	b.query.columns = columns
	return b
}

// Where 添加 WHERE 条件
func (b *QueryBuilder) Where(condition string) *QueryBuilder {
	b.query.where = append(b.query.where, condition)
	return b
}

// OrderBy 设置排序
func (b *QueryBuilder) OrderBy(column string) *QueryBuilder {
	b.query.orderBy = column
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
	sql := fmt.Sprintf("SELECT %s FROM %s",
		strings.Join(b.query.columns, ", "),
		b.query.table)

	if len(b.query.where) > 0 {
		sql += " WHERE " + strings.Join(b.query.where, " AND ")
	}

	if b.query.orderBy != "" {
		sql += " ORDER BY " + b.query.orderBy
	}

	if b.query.limit > 0 {
		sql += fmt.Sprintf(" LIMIT %d", b.query.limit)
	}

	if b.query.offset > 0 {
		sql += fmt.Sprintf(" OFFSET %d", b.query.offset)
	}

	return sql
}

func main() {
	fmt.Println("=== 链式调用建造者模式示例 ===\n")

	// 示例 1: 简单查询
	fmt.Println("示例 1: 简单查询")
	sql1 := NewQueryBuilder("users").
		Select("id", "name", "email").
		Build()
	fmt.Println(sql1)
	fmt.Println()

	// 示例 2: 带条件的查询
	fmt.Println("示例 2: 带条件的查询")
	sql2 := NewQueryBuilder("users").
		Select("id", "name", "email").
		Where("age > 18").
		Where("status = 'active'").
		Build()
	fmt.Println(sql2)
	fmt.Println()

	// 示例 3: 完整的查询
	fmt.Println("示例 3: 完整的查询")
	sql3 := NewQueryBuilder("users").
		Select("id", "name", "email", "created_at").
		Where("age > 18").
		Where("status = 'active'").
		OrderBy("created_at DESC").
		Limit(10).
		Offset(20).
		Build()
	fmt.Println(sql3)
	fmt.Println()

	// 示例 4: 只查询部分字段
	fmt.Println("示例 4: 只查询部分字段")
	sql4 := NewQueryBuilder("products").
		Select("name", "price").
		Where("category = 'electronics'").
		OrderBy("price ASC").
		Limit(5).
		Build()
	fmt.Println(sql4)

	fmt.Println("\n=== 示例结束 ===")
}
