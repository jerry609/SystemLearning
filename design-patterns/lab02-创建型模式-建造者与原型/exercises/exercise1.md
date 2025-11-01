# 练习 1: 实现 SQL 查询构建器

## 难度
⭐⭐ (中等)

## 学习目标
- 掌握建造者模式的实现
- 理解链式调用的设计
- 学会构建复杂对象
- 实践 SQL 查询构建

## 问题描述

实现一个功能完整的 SQL 查询构建器，支持构建复杂的 SQL SELECT 语句。该构建器应该支持链式调用，并能够生成正确的 SQL 语句。

## 功能要求

1. **基本查询功能**
   - 支持 SELECT 指定列
   - 支持 FROM 指定表名
   - 支持 WHERE 条件（支持多个条件）
   - 支持 ORDER BY 排序
   - 支持 LIMIT 限制返回数量
   - 支持 OFFSET 偏移量

2. **高级功能**
   - 支持 JOIN 操作（INNER JOIN, LEFT JOIN, RIGHT JOIN）
   - 支持 GROUP BY 分组
   - 支持 HAVING 条件
   - 支持聚合函数（COUNT, SUM, AVG, MAX, MIN）

3. **安全性**
   - 防止 SQL 注入（使用参数化查询）
   - 验证表名和列名的合法性

4. **易用性**
   - 支持链式调用
   - 提供清晰的 API
   - 生成格式化的 SQL 语句

## 输入输出示例

### 示例 1: 简单查询
**代码**:
```go
sql := NewQueryBuilder("users").
    Select("id", "name", "email").
    Where("age > ?", 18).
    OrderBy("created_at DESC").
    Limit(10).
    Build()
```

**输出**:
```sql
SELECT id, name, email FROM users WHERE age > ? ORDER BY created_at DESC LIMIT 10
```

### 示例 2: 带 JOIN 的查询
**代码**:
```go
sql := NewQueryBuilder("users").
    Select("users.id", "users.name", "orders.total").
    InnerJoin("orders", "users.id = orders.user_id").
    Where("orders.status = ?", "completed").
    GroupBy("users.id").
    Having("COUNT(orders.id) > ?", 5).
    Build()
```

**输出**:
```sql
SELECT users.id, users.name, orders.total FROM users 
INNER JOIN orders ON users.id = orders.user_id 
WHERE orders.status = ? 
GROUP BY users.id 
HAVING COUNT(orders.id) > ?
```

### 示例 3: 聚合查询
**代码**:
```go
sql := NewQueryBuilder("products").
    Select("category", "COUNT(*) as count", "AVG(price) as avg_price").
    Where("status = ?", "active").
    GroupBy("category").
    OrderBy("count DESC").
    Build()
```

**输出**:
```sql
SELECT category, COUNT(*) as count, AVG(price) as avg_price FROM products 
WHERE status = ? 
GROUP BY category 
ORDER BY count DESC
```

## 提示

💡 **提示 1**: 使用结构体存储查询的各个部分
```go
type Query struct {
    table    string
    columns  []string
    joins    []string
    where    []string
    groupBy  []string
    having   []string
    orderBy  string
    limit    int
    offset   int
}
```

💡 **提示 2**: 每个方法返回 `*QueryBuilder` 以支持链式调用
```go
func (b *QueryBuilder) Where(condition string, args ...interface{}) *QueryBuilder {
    b.query.where = append(b.query.where, condition)
    return b
}
```

💡 **提示 3**: 使用 `strings.Join` 组合 SQL 片段
```go
if len(b.query.where) > 0 {
    sql += " WHERE " + strings.Join(b.query.where, " AND ")
}
```

💡 **提示 4**: 考虑使用参数化查询防止 SQL 注入
```go
// 不要直接拼接值
// 错误: Where("age > " + strconv.Itoa(age))
// 正确: Where("age > ?", age)
```

## 评分标准

- [ ] **功能完整性 (40%)**
  - 实现所有基本查询功能
  - 实现至少 2 个高级功能
  - 生成正确的 SQL 语句

- [ ] **代码质量 (30%)**
  - 代码结构清晰
  - 命名规范
  - 适当的注释

- [ ] **设计模式应用 (20%)**
  - 正确使用建造者模式
  - 支持链式调用
  - API 设计合理

- [ ] **安全性 (10%)**
  - 防止 SQL 注入
  - 输入验证

## 扩展挑战

如果你完成了基本要求，可以尝试以下扩展功能：

1. **支持子查询**
   ```go
   subQuery := NewQueryBuilder("orders").
       Select("user_id").
       Where("total > ?", 1000)
   
   mainQuery := NewQueryBuilder("users").
       Select("*").
       Where("id IN (?)", subQuery)
   ```

2. **支持 UNION 操作**
   ```go
   query1 := NewQueryBuilder("users").Select("name").Where("age > ?", 18)
   query2 := NewQueryBuilder("admins").Select("name").Where("active = ?", true)
   
   unionQuery := query1.Union(query2)
   ```

3. **支持 INSERT, UPDATE, DELETE 语句**
   ```go
   insert := NewInsertBuilder("users").
       Columns("name", "email").
       Values("John", "john@example.com")
   
   update := NewUpdateBuilder("users").
       Set("status", "active").
       Where("id = ?", 123)
   
   delete := NewDeleteBuilder("users").
       Where("created_at < ?", "2020-01-01")
   ```

4. **支持事务构建**
   ```go
   tx := NewTransactionBuilder().
       Add(insertQuery).
       Add(updateQuery).
       Build()
   ```

## 参考资源

- [Go database/sql 包文档](https://pkg.go.dev/database/sql)
- [SQL 注入防护](https://owasp.org/www-community/attacks/SQL_Injection)
- [建造者模式详解](../theory/01-builder.md)

## 提交要求

1. 实现 `QueryBuilder` 结构体和相关方法
2. 编写测试用例验证功能
3. 提供使用示例
4. 添加必要的注释和文档

---

**预计完成时间**: 1-2 小时  
**难度评估**: 中等  
**重点考察**: 建造者模式、链式调用、SQL 构建
