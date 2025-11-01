# 练习 2: 实现数据库连接工厂

## 难度
⭐⭐⭐ (中等偏难)

## 学习目标
- 掌握工厂模式的实现
- 理解工厂方法和抽象工厂的区别
- 学会设计可扩展的系统
- 实践数据库连接管理

## 问题描述

实现一个数据库连接工厂系统，支持多种数据库类型（MySQL、PostgreSQL、MongoDB）。使用工厂模式创建不同类型的数据库连接，并提供统一的操作接口。

## 功能要求

1. **数据库接口**
   - `Connect() error` - 连接数据库
   - `Close() error` - 关闭连接
   - `Query(sql string) ([]map[string]interface{}, error)` - 查询数据
   - `Execute(sql string) error` - 执行 SQL
   - `Ping() error` - 检查连接

2. **工厂模式**
   - 简单工厂：根据类型创建数据库连接
   - 工厂方法：每种数据库有自己的工厂
   - 支持注册新的数据库类型

3. **连接池**
   - 支持连接池管理
   - 设置最大连接数
   - 连接复用

4. **配置管理**
   - 支持从配置文件读取
   - 支持环境变量
   - 支持默认配置

5. **错误处理**
   - 连接失败重试
   - 超时处理
   - 错误日志

## 输入输出示例

### 示例 1: 简单工厂
**代码**:
```go
// 创建 MySQL 连接
config := &DBConfig{
    Host:     "localhost",
    Port:     3306,
    Username: "root",
    Password: "password",
    Database: "testdb",
}

db, err := CreateDatabase("mysql", config)
if err != nil {
    log.Fatal(err)
}
defer db.Close()

// 查询数据
results, err := db.Query("SELECT * FROM users")
if err != nil {
    log.Fatal(err)
}

for _, row := range results {
    fmt.Println(row)
}
```

### 示例 2: 工厂方法
**代码**:
```go
// 使用工厂方法
var factory DatabaseFactory

dbType := "postgres"
switch dbType {
case "mysql":
    factory = &MySQLFactory{}
case "postgres":
    factory = &PostgreSQLFactory{}
case "mongodb":
    factory = &MongoDBFactory{}
}

db := factory.CreateDatabase(config)
db.Connect()
```

### 示例 3: 注册机制
**代码**:
```go
// 注册数据库类型
RegisterDatabase("mysql", func(config *DBConfig) Database {
    return &MySQLDatabase{config: config}
})

RegisterDatabase("postgres", func(config *DBConfig) Database {
    return &PostgreSQLDatabase{config: config}
})

// 创建数据库连接
db, err := CreateDatabase("mysql", config)
```

## 数据结构

```go
type DBConfig struct {
    Host     string
    Port     int
    Username string
    Password string
    Database string
    MaxConns int
    Timeout  time.Duration
}

type Database interface {
    Connect() error
    Close() error
    Query(sql string) ([]map[string]interface{}, error)
    Execute(sql string) error
    Ping() error
}

type DatabaseFactory interface {
    CreateDatabase(config *DBConfig) Database
}
```

## 提示

💡 **提示 1**: 定义统一的数据库接口
```go
type Database interface {
    Connect() error
    Close() error
    Query(sql string) ([]map[string]interface{}, error)
    Execute(sql string) error
    Ping() error
}
```

💡 **提示 2**: 实现简单工厂
```go
func CreateDatabase(dbType string, config *DBConfig) (Database, error) {
    switch dbType {
    case "mysql":
        return &MySQLDatabase{config: config}, nil
    case "postgres":
        return &PostgreSQLDatabase{config: config}, nil
    case "mongodb":
        return &MongoDBDatabase{config: config}, nil
    default:
        return nil, fmt.Errorf("unsupported database type: %s", dbType)
    }
}
```

💡 **提示 3**: 使用注册机制
```go
var dbRegistry = make(map[string]func(*DBConfig) Database)

func RegisterDatabase(name string, factory func(*DBConfig) Database) {
    dbRegistry[name] = factory
}

func CreateDatabase(name string, config *DBConfig) (Database, error) {
    factory, ok := dbRegistry[name]
    if !ok {
        return nil, fmt.Errorf("database type not registered: %s", name)
    }
    return factory(config), nil
}
```

💡 **提示 4**: 实现连接池
```go
type ConnectionPool struct {
    factory  DatabaseFactory
    config   *DBConfig
    pool     chan Database
    maxConns int
}

func (p *ConnectionPool) Get() (Database, error) {
    select {
    case db := <-p.pool:
        return db, nil
    default:
        return p.factory.CreateDatabase(p.config), nil
    }
}

func (p *ConnectionPool) Put(db Database) {
    select {
    case p.pool <- db:
    default:
        db.Close()
    }
}
```

## 评分标准

- [ ] **工厂模式实现 (40%)**
  - 实现简单工厂
  - 实现工厂方法
  - 支持注册机制
  - 易于扩展

- [ ] **功能完整性 (30%)**
  - 实现所有数据库操作
  - 支持多种数据库类型
  - 连接池管理
  - 配置管理

- [ ] **代码质量 (20%)**
  - 代码结构清晰
  - 命名规范
  - 错误处理完善
  - 适当的注释

- [ ] **可扩展性 (10%)**
  - 易于添加新数据库类型
  - 符合开闭原则
  - 接口设计合理

## 扩展挑战

如果你完成了基本要求，可以尝试以下扩展功能：

1. **事务支持**
   ```go
   type Transaction interface {
       Begin() error
       Commit() error
       Rollback() error
       Query(sql string) ([]map[string]interface{}, error)
       Execute(sql string) error
   }
   
   func (db *MySQLDatabase) BeginTransaction() (Transaction, error) {
       // 开始事务
   }
   ```

2. **ORM 功能**
   ```go
   type User struct {
       ID    int    `db:"id"`
       Name  string `db:"name"`
       Email string `db:"email"`
   }
   
   func (db *Database) Find(dest interface{}, conditions map[string]interface{}) error {
       // 查询并映射到结构体
   }
   
   func (db *Database) Save(entity interface{}) error {
       // 保存实体
   }
   ```

3. **查询构建器**
   ```go
   type QueryBuilder struct {
       db    Database
       table string
       where []string
       limit int
   }
   
   func (qb *QueryBuilder) Where(condition string) *QueryBuilder {
       qb.where = append(qb.where, condition)
       return qb
   }
   
   func (qb *QueryBuilder) Limit(limit int) *QueryBuilder {
       qb.limit = limit
       return qb
   }
   
   func (qb *QueryBuilder) Execute() ([]map[string]interface{}, error) {
       // 构建并执行查询
   }
   ```

4. **读写分离**
   ```go
   type MasterSlaveDB struct {
       master Database
       slaves []Database
   }
   
   func (db *MasterSlaveDB) Query(sql string) ([]map[string]interface{}, error) {
       // 从从库读取
       slave := db.getRandomSlave()
       return slave.Query(sql)
   }
   
   func (db *MasterSlaveDB) Execute(sql string) error {
       // 写入主库
       return db.master.Execute(sql)
   }
   ```

## 参考资源

- [database/sql 包文档](https://pkg.go.dev/database/sql)
- [工厂模式详解](../theory/02-factory.md)
- [设计原则](../theory/03-design-principles.md)

## 提交要求

1. 实现 `Database` 接口和具体实现
2. 实现工厂模式（简单工厂和工厂方法）
3. 编写单元测试
4. 提供使用示例
5. 添加必要的注释和文档

---

**预计完成时间**: 2-3 小时  
**难度评估**: 中等偏难  
**重点考察**: 工厂模式、接口设计、可扩展性
