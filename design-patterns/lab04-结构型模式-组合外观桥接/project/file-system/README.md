# 文件系统项目

## 项目背景

本项目实现一个简化的文件系统，支持文件和目录的统一管理。使用组合模式构建树形结构，实现文件和目录的一致性操作。

## 功能列表

- [x] 创建文件和目录
- [x] 添加和删除文件/目录
- [x] 显示文件系统结构
- [x] 计算目录大小
- [x] 查找文件/目录
- [x] 列出所有文件
- [x] 复制文件/目录
- [x] 移动文件/目录
- [x] 文件权限管理
- [x] 文件搜索功能

## 技术栈

- Go 1.21+
- 标准库

## 设计模式应用

| 模式 | 应用位置 | 作用 |
|------|----------|------|
| 组合模式 | FileSystemNode 接口及其实现 | 统一处理文件和目录，构建树形结构 |

## 项目结构

```
file-system/
├── README.md           # 项目说明
├── main.go            # 主程序入口
├── filesystem.go      # 文件系统实现
└── filesystem_test.go # 测试文件
```

## 运行方式

```bash
# 运行程序
go run main.go filesystem.go

# 运行测试
go test -v

# 运行测试并显示覆盖率
go test -v -cover
```

## 预期输出

```
=== 文件系统项目 ===

创建文件系统结构...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

文件系统结构:
📁 root/ (15877 KB) [rwx]
  📁 home/ (4613 KB) [rwx]
    📁 user/ (4613 KB) [rwx]
      📁 documents/ (1029 KB) [rwx]
        📄 README.md (5 KB) [rw-]
        📄 report.pdf (1024 KB) [rw-]
      📁 pictures/ (3584 KB) [rwx]
        📄 vacation.jpg (2048 KB) [rw-]
        📄 family.jpg (1536 KB) [rw-]
  📁 var/ (11264 KB) [rwx]
    📁 log/ (11264 KB) [rwx]
      📄 system.log (10240 KB) [rw-]
      📄 error.log (1024 KB) [rw-]

总大小: 15877 KB
文件总数: 6

查找文件 'report.pdf':
找到: report.pdf (大小: 1024 KB, 权限: rw-)

搜索 .jpg 文件:
  - home/user/pictures/vacation.jpg
  - home/user/pictures/family.jpg

复制目录 'documents' 到 'backup':
✅ 复制成功

移动文件 'error.log' 到 'home/user':
✅ 移动成功

最终文件系统结构:
[显示更新后的结构]
```

## 扩展建议

1. **文件内容管理**
   - 支持读写文件内容
   - 支持文件追加
   - 支持文件截断

2. **高级搜索**
   - 按文件大小搜索
   - 按修改时间搜索
   - 按文件类型搜索
   - 正则表达式搜索

3. **文件操作**
   - 支持文件重命名
   - 支持符号链接
   - 支持硬链接

4. **权限系统**
   - 用户和组管理
   - 详细的权限控制
   - ACL 支持

5. **持久化**
   - 保存文件系统到磁盘
   - 从磁盘加载文件系统
   - 支持增量保存

6. **性能优化**
   - 缓存目录大小
   - 索引文件路径
   - 并发访问控制
