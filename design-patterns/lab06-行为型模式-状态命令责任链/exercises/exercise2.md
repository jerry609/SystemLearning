# 练习 2: 实现文本编辑器的撤销重做

## 难度
⭐⭐ (中等)

## 学习目标
- 掌握命令模式的实现
- 理解如何实现撤销/重做功能
- 学会管理命令历史

## 问题描述

设计并实现一个简单的文本编辑器，支持基本的编辑操作和撤销/重做功能。

## 功能要求

### 1. 编辑操作
实现以下编辑命令：
- **InsertCommand**: 在指定位置插入文本
- **DeleteCommand**: 删除指定范围的文本
- **ReplaceCommand**: 替换指定文本
- **CutCommand**: 剪切文本到剪贴板
- **PasteCommand**: 从剪贴板粘贴文本

### 2. 命令管理
- 维护命令历史栈
- 支持撤销 (Undo) 操作
- 支持重做 (Redo) 操作
- 支持查看命令历史

### 3. 宏命令
- 支持将多个命令组合成宏命令
- 宏命令可以一次性执行多个操作
- 宏命令支持撤销和重做

### 4. 限制
- 命令历史最多保存 100 条
- 超过限制时，删除最早的命令
- 执行新命令时，清除当前位置之后的历史

## 输入输出示例

### 示例 1: 基本插入和撤销

**输入**:
```go
editor := NewTextEditor()
manager := NewCommandManager()

manager.Execute(NewInsertCommand(editor, "Hello", 0))
manager.Execute(NewInsertCommand(editor, " World", 5))
manager.Undo()
```

**输出**:
```
执行: 在位置 0 插入 "Hello"
内容: "Hello"

执行: 在位置 5 插入 " World"
内容: "Hello World"

撤销: 在位置 5 插入 " World"
内容: "Hello"
```

### 示例 2: 删除和重做

**输入**:
```go
editor.SetContent("Hello World")
manager.Execute(NewDeleteCommand(editor, 5, 11))
manager.Undo()
manager.Redo()
```

**输出**:
```
执行: 删除位置 5-11 的文本 " World"
内容: "Hello"

撤销: 删除位置 5-11 的文本
内容: "Hello World"

重做: Delete(5-11)
执行: 删除位置 5-11 的文本 " World"
内容: "Hello"
```

### 示例 3: 宏命令

**输入**:
```go
editor.SetContent("hello world")
macro := NewMacroCommand(
    "格式化",
    NewReplaceCommand(editor, " ", "_"),
    NewUpperCaseCommand(editor),
)
manager.Execute(macro)
manager.Undo()
```

**输出**:
```
执行宏命令: 格式化
执行: 将 " " 替换为 "_"
执行: 转换为大写
内容: "HELLO_WORLD"

撤销宏命令: 格式化
撤销: 转换为大写
撤销: 将 " " 替换为 "_"
内容: "hello world"
```

## 提示

💡 **提示 1**: 每个命令对象需要保存执行前的状态，用于撤销操作。

💡 **提示 2**: 使用两个栈或一个数组加索引来管理命令历史和当前位置。

💡 **提示 3**: 宏命令可以使用组合模式，包含多个子命令。

💡 **提示 4**: 注意处理边界情况，如空文本、越界位置等。

💡 **提示 5**: 考虑使用接口定义命令行为，便于扩展新的命令类型。

## 评分标准

- [ ] **功能完整性 (40%)**
  - 实现所有要求的命令
  - 正确实现撤销/重做
  - 宏命令功能正常

- [ ] **代码质量 (30%)**
  - 代码结构清晰
  - 命名规范
  - 适当的注释

- [ ] **设计模式应用 (20%)**
  - 正确使用命令模式
  - 命令历史管理合理
  - 宏命令设计合理

- [ ] **错误处理 (10%)**
  - 处理越界情况
  - 处理空历史的撤销/重做
  - 提供清晰的错误信息

## 扩展挑战

如果你完成了基本要求，可以尝试以下扩展功能：

1. **命令压缩**: 合并连续的相同类型命令（如多次插入同一位置）
2. **命令持久化**: 将命令历史保存到文件，支持恢复
3. **选择操作**: 支持选中文本并对选中内容进行操作
4. **查找替换**: 实现查找和批量替换功能
5. **多光标编辑**: 支持多个光标同时编辑
6. **命令统计**: 统计各类命令的使用频率

## 参考资源

- 命令模式理论文档: `theory/02-command.md`
- 撤销重做示例: `examples/command/undo_redo.go`
- 任务队列示例: `examples/command/task_queue.go`
