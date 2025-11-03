package main

// 练习 2: 实现文本编辑器的撤销重做 - 参考答案
//
// 设计思路:
// 1. 使用命令模式封装每个编辑操作
// 2. 使用命令管理器维护命令历史
// 3. 每个命令保存执行前的状态用于撤销
//
// 使用的设计模式: 命令模式 + 组合模式（宏命令）
// 模式应用位置:
// - 命令模式: Command 接口及其实现类
// - 组合模式: MacroCommand 组合多个命令

import (
	"fmt"
	"strings"
)

// Command 命令接口
type Command interface {
	Execute() error
	Undo() error
	String() string
}

// TextEditor 文本编辑器
type TextEditor struct {
	content   string
	clipboard string
}

func NewTextEditor() *TextEditor {
	return &TextEditor{content: ""}
}

func (e *TextEditor) GetContent() string {
	return e.content
}

func (e *TextEditor) SetContent(content string) {
	e.content = content
}

func (e *TextEditor) GetClipboard() string {
	return e.clipboard
}

func (e *TextEditor) SetClipboard(content string) {
	e.clipboard = content
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
	return &InsertCommand{editor: editor, text: text, position: position}
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
	return &DeleteCommand{editor: editor, start: start, end: end}
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
	return &ReplaceCommand{editor: editor, oldText: oldText, newText: newText}
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

// CutCommand 剪切命令
type CutCommand struct {
	editor *TextEditor
	start  int
	end    int
	backup string
}

func NewCutCommand(editor *TextEditor, start, end int) *CutCommand {
	return &CutCommand{editor: editor, start: start, end: end}
}

func (c *CutCommand) Execute() error {
	c.backup = c.editor.GetContent()
	content := c.editor.GetContent()
	if c.start > len(content) {
		c.start = len(content)
	}
	if c.end > len(content) {
		c.end = len(content)
	}
	cutText := content[c.start:c.end]
	c.editor.SetClipboard(cutText)
	newContent := content[:c.start] + content[c.end:]
	c.editor.SetContent(newContent)
	fmt.Printf("执行: 剪切位置 %d-%d 的文本 \"%s\"\n", c.start, c.end, cutText)
	return nil
}

func (c *CutCommand) Undo() error {
	c.editor.SetContent(c.backup)
	fmt.Printf("撤销: 剪切位置 %d-%d 的文本\n", c.start, c.end)
	return nil
}

func (c *CutCommand) String() string {
	return fmt.Sprintf("Cut(%d-%d)", c.start, c.end)
}

// PasteCommand 粘贴命令
type PasteCommand struct {
	editor   *TextEditor
	position int
	backup   string
}

func NewPasteCommand(editor *TextEditor, position int) *PasteCommand {
	return &PasteCommand{editor: editor, position: position}
}

func (c *PasteCommand) Execute() error {
	c.backup = c.editor.GetContent()
	content := c.editor.GetContent()
	clipboard := c.editor.GetClipboard()
	if c.position > len(content) {
		c.position = len(content)
	}
	newContent := content[:c.position] + clipboard + content[c.position:]
	c.editor.SetContent(newContent)
	fmt.Printf("执行: 在位置 %d 粘贴 \"%s\"\n", c.position, clipboard)
	return nil
}

func (c *PasteCommand) Undo() error {
	c.editor.SetContent(c.backup)
	fmt.Printf("撤销: 在位置 %d 粘贴\n", c.position)
	return nil
}

func (c *PasteCommand) String() string {
	return fmt.Sprintf("Paste(at %d)", c.position)
}

// UpperCaseCommand 转大写命令
type UpperCaseCommand struct {
	editor *TextEditor
	backup string
}

func NewUpperCaseCommand(editor *TextEditor) *UpperCaseCommand {
	return &UpperCaseCommand{editor: editor}
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

// MacroCommand 宏命令
type MacroCommand struct {
	commands []Command
	name     string
}

func NewMacroCommand(name string, commands ...Command) *MacroCommand {
	return &MacroCommand{name: name, commands: commands}
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

// CommandManager 命令管理器
type CommandManager struct {
	history  []Command
	current  int
	maxSize  int
}

func NewCommandManager(maxSize int) *CommandManager {
	return &CommandManager{
		history: make([]Command, 0),
		current: 0,
		maxSize: maxSize,
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
	
	// 限制历史大小
	if len(m.history) > m.maxSize {
		m.history = m.history[1:]
		m.current--
	}
	
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

func main() {
	fmt.Println("=== 练习2: 文本编辑器的撤销重做 ===\n")

	editor := NewTextEditor()
	manager := NewCommandManager(100)

	// 场景1: 基本操作
	fmt.Println("--- 场景1: 基本插入和撤销 ---")
	manager.Execute(NewInsertCommand(editor, "Hello", 0))
	editor.Print()
	manager.Execute(NewInsertCommand(editor, " World", 5))
	editor.Print()
	manager.Undo()
	editor.Print()
	manager.Redo()
	editor.Print()
	fmt.Println()

	// 场景2: 剪切和粘贴
	fmt.Println("--- 场景2: 剪切和粘贴 ---")
	manager.Execute(NewCutCommand(editor, 0, 5))
	editor.Print()
	manager.Execute(NewPasteCommand(editor, 6))
	editor.Print()
	manager.Undo()
	editor.Print()
	fmt.Println()

	// 场景3: 宏命令
	fmt.Println("--- 场景3: 宏命令 ---")
	editor.SetContent("hello world")
	editor.Print()
	macro := NewMacroCommand(
		"格式化",
		NewReplaceCommand(editor, " ", "_"),
		NewUpperCaseCommand(editor),
	)
	manager.Execute(macro)
	editor.Print()
	manager.Undo()
	editor.Print()
	manager.PrintHistory()

	fmt.Println("\n=== 练习完成 ===")
}

// 可能的优化方向:
// 1. 实现命令压缩（合并连续的相同操作）
// 2. 添加命令持久化功能
// 3. 支持选择操作
// 4. 实现查找替换功能
//
// 变体实现:
// 1. 使用函数类型简化简单命令
// 2. 添加命令的优先级
// 3. 支持异步命令执行
