package main

import (
	"fmt"
	"strings"
)

// 撤销重做示例
// 本示例展示了命令模式在实现撤销/重做功能中的应用

// Command 命令接口
type Command interface {
	Execute() error
	Undo() error
	String() string
}

// TextEditor 文本编辑器（接收者）
type TextEditor struct {
	content string
}

func NewTextEditor() *TextEditor {
	return &TextEditor{
		content: "",
	}
}

func (e *TextEditor) GetContent() string {
	return e.content
}

func (e *TextEditor) SetContent(content string) {
	e.content = content
}

func (e *TextEditor) Print() {
	fmt.Printf("内容: \"%s\"\n", e.content)
}

// InsertCommand 插入命令
type InsertCommand struct {
	editor   *TextEditor
	text     string
	position int
	backup   string
}

func NewInsertCommand(editor *TextEditor, text string, position int) *InsertCommand {
	return &InsertCommand{
		editor:   editor,
		text:     text,
		position: position,
	}
}

func (c *InsertCommand) Execute() error {
	c.backup = c.editor.GetContent()
	content := c.editor.GetContent()
	
	if c.position > len(content) {
		c.position = len(content)
	}
	
	newContent := content[:c.position] + c.text + content[c.position:]
	c.editor.SetContent(newContent)
	fmt.Printf("执行: 在位置 %d 插入 \"%s\"\n", c.position, c.text)
	return nil
}

func (c *InsertCommand) Undo() error {
	c.editor.SetContent(c.backup)
	fmt.Printf("撤销: 在位置 %d 插入 \"%s\"\n", c.position, c.text)
	return nil
}

func (c *InsertCommand) String() string {
	return fmt.Sprintf("Insert(\"%s\" at %d)", c.text, c.position)
}

// DeleteCommand 删除命令
type DeleteCommand struct {
	editor *TextEditor
	start  int
	end    int
	backup string
}

func NewDeleteCommand(editor *TextEditor, start, end int) *DeleteCommand {
	return &DeleteCommand{
		editor: editor,
		start:  start,
		end:    end,
	}
}

func (c *DeleteCommand) Execute() error {
	c.backup = c.editor.GetContent()
	content := c.editor.GetContent()
	
	if c.start > len(content) {
		c.start = len(content)
	}
	if c.end > len(content) {
		c.end = len(content)
	}
	if c.start > c.end {
		c.start, c.end = c.end, c.start
	}
	
	deletedText := content[c.start:c.end]
	newContent := content[:c.start] + content[c.end:]
	c.editor.SetContent(newContent)
	fmt.Printf("执行: 删除位置 %d-%d 的文本 \"%s\"\n", c.start, c.end, deletedText)
	return nil
}

func (c *DeleteCommand) Undo() error {
	c.editor.SetContent(c.backup)
	fmt.Printf("撤销: 删除位置 %d-%d 的文本\n", c.start, c.end)
	return nil
}

func (c *DeleteCommand) String() string {
	return fmt.Sprintf("Delete(%d-%d)", c.start, c.end)
}

// ReplaceCommand 替换命令
type ReplaceCommand struct {
	editor  *TextEditor
	oldText string
	newText string
	backup  string
}

func NewReplaceCommand(editor *TextEditor, oldText, newText string) *ReplaceCommand {
	return &ReplaceCommand{
		editor:  editor,
		oldText: oldText,
		newText: newText,
	}
}

func (c *ReplaceCommand) Execute() error {
	c.backup = c.editor.GetContent()
	content := c.editor.GetContent()
	newContent := strings.ReplaceAll(content, c.oldText, c.newText)
	c.editor.SetContent(newContent)
	fmt.Printf("执行: 将 \"%s\" 替换为 \"%s\"\n", c.oldText, c.newText)
	return nil
}

func (c *ReplaceCommand) Undo() error {
	c.editor.SetContent(c.backup)
	fmt.Printf("撤销: 将 \"%s\" 替换为 \"%s\"\n", c.oldText, c.newText)
	return nil
}

func (c *ReplaceCommand) String() string {
	return fmt.Sprintf("Replace(\"%s\" -> \"%s\")", c.oldText, c.newText)
}

// UpperCaseCommand 转大写命令
type UpperCaseCommand struct {
	editor *TextEditor
	backup string
}

func NewUpperCaseCommand(editor *TextEditor) *UpperCaseCommand {
	return &UpperCaseCommand{
		editor: editor,
	}
}

func (c *UpperCaseCommand) Execute() error {
	c.backup = c.editor.GetContent()
	content := c.editor.GetContent()
	c.editor.SetContent(strings.ToUpper(content))
	fmt.Println("执行: 转换为大写")
	return nil
}

func (c *UpperCaseCommand) Undo() error {
	c.editor.SetContent(c.backup)
	fmt.Println("撤销: 转换为大写")
	return nil
}

func (c *UpperCaseCommand) String() string {
	return "UpperCase()"
}

// CommandManager 命令管理器
type CommandManager struct {
	history []Command
	current int
}

func NewCommandManager() *CommandManager {
	return &CommandManager{
		history: make([]Command, 0),
		current: 0,
	}
}

func (m *CommandManager) Execute(cmd Command) error {
	if err := cmd.Execute(); err != nil {
		return err
	}
	
	// 清除当前位置之后的历史
	m.history = m.history[:m.current]
	m.history = append(m.history, cmd)
	m.current++
	
	return nil
}

func (m *CommandManager) Undo() error {
	if m.current == 0 {
		return fmt.Errorf("没有可撤销的操作")
	}
	
	m.current--
	cmd := m.history[m.current]
	return cmd.Undo()
}

func (m *CommandManager) Redo() error {
	if m.current >= len(m.history) {
		return fmt.Errorf("没有可重做的操作")
	}
	
	cmd := m.history[m.current]
	if err := cmd.Execute(); err != nil {
		return err
	}
	m.current++
	fmt.Printf("重做: %s\n", cmd.String())
	return nil
}

func (m *CommandManager) CanUndo() bool {
	return m.current > 0
}

func (m *CommandManager) CanRedo() bool {
	return m.current < len(m.history)
}

func (m *CommandManager) PrintHistory() {
	fmt.Println("\n命令历史:")
	for i, cmd := range m.history {
		marker := "  "
		if i == m.current {
			marker = "→ "
		}
		fmt.Printf("%s%d. %s\n", marker, i+1, cmd.String())
	}
	fmt.Printf("当前位置: %d/%d\n", m.current, len(m.history))
}

// MacroCommand 宏命令（组合多个命令）
type MacroCommand struct {
	commands []Command
	name     string
}

func NewMacroCommand(name string, commands ...Command) *MacroCommand {
	return &MacroCommand{
		name:     name,
		commands: commands,
	}
}

func (m *MacroCommand) Execute() error {
	fmt.Printf("执行宏命令: %s\n", m.name)
	for _, cmd := range m.commands {
		if err := cmd.Execute(); err != nil {
			return err
		}
	}
	return nil
}

func (m *MacroCommand) Undo() error {
	fmt.Printf("撤销宏命令: %s\n", m.name)
	// 逆序撤销
	for i := len(m.commands) - 1; i >= 0; i-- {
		if err := m.commands[i].Undo(); err != nil {
			return err
		}
	}
	return nil
}

func (m *MacroCommand) String() string {
	return fmt.Sprintf("Macro(%s, %d commands)", m.name, len(m.commands))
}

func main() {
	fmt.Println("=== 命令模式示例 - 撤销重做 ===\n")

	editor := NewTextEditor()
	manager := NewCommandManager()

	// 场景1: 基本的插入和撤销
	fmt.Println("--- 场景1: 基本的插入和撤销 ---")
	editor.Print()
	fmt.Println()

	manager.Execute(NewInsertCommand(editor, "Hello", 0))
	editor.Print()
	fmt.Println()

	manager.Execute(NewInsertCommand(editor, " World", 5))
	editor.Print()
	fmt.Println()

	manager.Execute(NewInsertCommand(editor, "!", 11))
	editor.Print()
	manager.PrintHistory()
	fmt.Println()

	// 撤销操作
	fmt.Println("撤销最后一个操作:")
	manager.Undo()
	editor.Print()
	fmt.Println()

	fmt.Println("再撤销一个操作:")
	manager.Undo()
	editor.Print()
	manager.PrintHistory()
	fmt.Println()

	// 场景2: 重做操作
	fmt.Println("\n--- 场景2: 重做操作 ---")
	fmt.Println("重做:")
	manager.Redo()
	editor.Print()
	fmt.Println()

	fmt.Println("再重做:")
	manager.Redo()
	editor.Print()
	manager.PrintHistory()
	fmt.Println()

	// 场景3: 删除和替换
	fmt.Println("\n--- 场景3: 删除和替换 ---")
	manager.Execute(NewDeleteCommand(editor, 5, 11))
	editor.Print()
	fmt.Println()

	manager.Execute(NewInsertCommand(editor, " Go", 5))
	editor.Print()
	fmt.Println()

	manager.Execute(NewReplaceCommand(editor, "Hello", "Hi"))
	editor.Print()
	manager.PrintHistory()
	fmt.Println()

	// 撤销几步
	fmt.Println("撤销3次:")
	manager.Undo()
	editor.Print()
	manager.Undo()
	editor.Print()
	manager.Undo()
	editor.Print()
	manager.PrintHistory()
	fmt.Println()

	// 场景4: 在中间位置执行新命令
	fmt.Println("\n--- 场景4: 在中间位置执行新命令（清除后续历史） ---")
	manager.Execute(NewInsertCommand(editor, " Python", 11))
	editor.Print()
	manager.PrintHistory()
	fmt.Println()

	// 场景5: 宏命令
	fmt.Println("\n--- 场景5: 宏命令 ---")
	editor.SetContent("hello world")
	editor.Print()
	fmt.Println()

	macro := NewMacroCommand(
		"格式化文本",
		NewReplaceCommand(editor, " ", "_"),
		NewUpperCaseCommand(editor),
	)

	manager2 := NewCommandManager()
	manager2.Execute(macro)
	editor.Print()
	fmt.Println()

	fmt.Println("撤销宏命令:")
	manager2.Undo()
	editor.Print()
	fmt.Println()

	fmt.Println("重做宏命令:")
	manager2.Redo()
	editor.Print()
	fmt.Println()

	// 场景6: 边界测试
	fmt.Println("\n--- 场景6: 边界测试 ---")
	editor.SetContent("test")
	manager3 := NewCommandManager()
	
	fmt.Println("尝试撤销（无历史）:")
	if err := manager3.Undo(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}
	fmt.Println()

	manager3.Execute(NewInsertCommand(editor, "123", 0))
	editor.Print()
	fmt.Println()

	fmt.Println("尝试重做（无可重做）:")
	if err := manager3.Redo(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}

	fmt.Println("\n=== 示例结束 ===")
}
