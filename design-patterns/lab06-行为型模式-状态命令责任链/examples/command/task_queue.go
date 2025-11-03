package main

import (
	"fmt"
	"time"
)

// 任务队列示例
// 本示例展示了命令模式在任务队列中的应用

// Command 命令接口
type Command interface {
	Execute() error
	String() string
}

// Task 任务接收者
type Task struct {
	ID   string
	Name string
	Data interface{}
}

// SendEmailCommand 发送邮件命令
type SendEmailCommand struct {
	to      string
	subject string
	body    string
}

func NewSendEmailCommand(to, subject, body string) *SendEmailCommand {
	return &SendEmailCommand{
		to:      to,
		subject: subject,
		body:    body,
	}
}

func (c *SendEmailCommand) Execute() error {
	fmt.Printf("发送邮件:\n")
	fmt.Printf("  收件人: %s\n", c.to)
	fmt.Printf("  主题: %s\n", c.subject)
	fmt.Printf("  内容: %s\n", c.body)
	// 模拟发送邮件
	time.Sleep(100 * time.Millisecond)
	fmt.Println("  ✓ 邮件发送成功")
	return nil
}

func (c *SendEmailCommand) String() string {
	return fmt.Sprintf("SendEmail(to=%s, subject=%s)", c.to, c.subject)
}

// ProcessDataCommand 处理数据命令
type ProcessDataCommand struct {
	data     []int
	operator string
}

func NewProcessDataCommand(data []int, operator string) *ProcessDataCommand {
	return &ProcessDataCommand{
		data:     data,
		operator: operator,
	}
}

func (c *ProcessDataCommand) Execute() error {
	fmt.Printf("处理数据: %v, 操作: %s\n", c.data, c.operator)
	
	var result int
	switch c.operator {
	case "sum":
		for _, v := range c.data {
			result += v
		}
		fmt.Printf("  结果: %d\n", result)
	case "avg":
		sum := 0
		for _, v := range c.data {
			sum += v
		}
		result = sum / len(c.data)
		fmt.Printf("  结果: %d\n", result)
	case "max":
		result = c.data[0]
		for _, v := range c.data {
			if v > result {
				result = v
			}
		}
		fmt.Printf("  结果: %d\n", result)
	}
	
	// 模拟处理时间
	time.Sleep(50 * time.Millisecond)
	fmt.Println("  ✓ 数据处理完成")
	return nil
}

func (c *ProcessDataCommand) String() string {
	return fmt.Sprintf("ProcessData(operator=%s, count=%d)", c.operator, len(c.data))
}

// GenerateReportCommand 生成报告命令
type GenerateReportCommand struct {
	reportType string
	startDate  string
	endDate    string
}

func NewGenerateReportCommand(reportType, startDate, endDate string) *GenerateReportCommand {
	return &GenerateReportCommand{
		reportType: reportType,
		startDate:  startDate,
		endDate:    endDate,
	}
}

func (c *GenerateReportCommand) Execute() error {
	fmt.Printf("生成报告:\n")
	fmt.Printf("  类型: %s\n", c.reportType)
	fmt.Printf("  时间范围: %s ~ %s\n", c.startDate, c.endDate)
	// 模拟生成报告
	time.Sleep(200 * time.Millisecond)
	fmt.Println("  ✓ 报告生成完成")
	return nil
}

func (c *GenerateReportCommand) String() string {
	return fmt.Sprintf("GenerateReport(type=%s)", c.reportType)
}

// BackupDatabaseCommand 备份数据库命令
type BackupDatabaseCommand struct {
	database string
	path     string
}

func NewBackupDatabaseCommand(database, path string) *BackupDatabaseCommand {
	return &BackupDatabaseCommand{
		database: database,
		path:     path,
	}
}

func (c *BackupDatabaseCommand) Execute() error {
	fmt.Printf("备份数据库:\n")
	fmt.Printf("  数据库: %s\n", c.database)
	fmt.Printf("  备份路径: %s\n", c.path)
	// 模拟备份过程
	time.Sleep(300 * time.Millisecond)
	fmt.Println("  ✓ 数据库备份完成")
	return nil
}

func (c *BackupDatabaseCommand) String() string {
	return fmt.Sprintf("BackupDatabase(db=%s)", c.database)
}

// TaskQueue 任务队列（调用者）
type TaskQueue struct {
	queue   []Command
	history []Command
}

func NewTaskQueue() *TaskQueue {
	return &TaskQueue{
		queue:   make([]Command, 0),
		history: make([]Command, 0),
	}
}

func (q *TaskQueue) AddTask(cmd Command) {
	q.queue = append(q.queue, cmd)
	fmt.Printf("✓ 任务已加入队列: %s\n", cmd.String())
}

func (q *TaskQueue) ProcessNext() error {
	if len(q.queue) == 0 {
		return fmt.Errorf("队列为空")
	}
	
	cmd := q.queue[0]
	q.queue = q.queue[1:]
	
	fmt.Printf("\n>>> 执行任务: %s\n", cmd.String())
	startTime := time.Now()
	
	if err := cmd.Execute(); err != nil {
		return err
	}
	
	duration := time.Since(startTime)
	fmt.Printf(">>> 任务完成，耗时: %v\n", duration)
	
	q.history = append(q.history, cmd)
	return nil
}

func (q *TaskQueue) ProcessAll() {
	fmt.Println("\n=== 开始处理所有任务 ===")
	for len(q.queue) > 0 {
		if err := q.ProcessNext(); err != nil {
			fmt.Printf("错误: %v\n", err)
			break
		}
	}
	fmt.Println("=== 所有任务处理完成 ===")
}

func (q *TaskQueue) GetQueueSize() int {
	return len(q.queue)
}

func (q *TaskQueue) GetHistorySize() int {
	return len(q.history)
}

func (q *TaskQueue) PrintStatus() {
	fmt.Printf("\n队列状态: 待处理 %d 个任务，已完成 %d 个任务\n", q.GetQueueSize(), q.GetHistorySize())
}

func (q *TaskQueue) PrintHistory() {
	fmt.Println("\n任务历史:")
	for i, cmd := range q.history {
		fmt.Printf("  %d. %s\n", i+1, cmd.String())
	}
}

// PriorityTaskQueue 优先级任务队列
type PriorityTaskQueue struct {
	highPriority   []Command
	normalPriority []Command
	lowPriority    []Command
	history        []Command
}

func NewPriorityTaskQueue() *PriorityTaskQueue {
	return &PriorityTaskQueue{
		highPriority:   make([]Command, 0),
		normalPriority: make([]Command, 0),
		lowPriority:    make([]Command, 0),
		history:        make([]Command, 0),
	}
}

func (q *PriorityTaskQueue) AddTask(cmd Command, priority string) {
	switch priority {
	case "high":
		q.highPriority = append(q.highPriority, cmd)
		fmt.Printf("✓ 高优先级任务已加入队列: %s\n", cmd.String())
	case "low":
		q.lowPriority = append(q.lowPriority, cmd)
		fmt.Printf("✓ 低优先级任务已加入队列: %s\n", cmd.String())
	default:
		q.normalPriority = append(q.normalPriority, cmd)
		fmt.Printf("✓ 普通优先级任务已加入队列: %s\n", cmd.String())
	}
}

func (q *PriorityTaskQueue) ProcessNext() error {
	var cmd Command
	
	if len(q.highPriority) > 0 {
		cmd = q.highPriority[0]
		q.highPriority = q.highPriority[1:]
	} else if len(q.normalPriority) > 0 {
		cmd = q.normalPriority[0]
		q.normalPriority = q.normalPriority[1:]
	} else if len(q.lowPriority) > 0 {
		cmd = q.lowPriority[0]
		q.lowPriority = q.lowPriority[1:]
	} else {
		return fmt.Errorf("队列为空")
	}
	
	fmt.Printf("\n>>> 执行任务: %s\n", cmd.String())
	if err := cmd.Execute(); err != nil {
		return err
	}
	
	q.history = append(q.history, cmd)
	return nil
}

func (q *PriorityTaskQueue) ProcessAll() {
	fmt.Println("\n=== 开始处理所有任务（按优先级） ===")
	for len(q.highPriority)+len(q.normalPriority)+len(q.lowPriority) > 0 {
		if err := q.ProcessNext(); err != nil {
			fmt.Printf("错误: %v\n", err)
			break
		}
	}
	fmt.Println("=== 所有任务处理完成 ===")
}

func main() {
	fmt.Println("=== 命令模式示例 - 任务队列 ===\n")

	// 场景1: 基本任务队列
	fmt.Println("--- 场景1: 基本任务队列 ---")
	queue := NewTaskQueue()
	
	queue.AddTask(NewSendEmailCommand("user@example.com", "欢迎", "欢迎使用我们的服务"))
	queue.AddTask(NewProcessDataCommand([]int{1, 2, 3, 4, 5}, "sum"))
	queue.AddTask(NewGenerateReportCommand("销售报告", "2024-01-01", "2024-01-31"))
	
	queue.PrintStatus()
	queue.ProcessAll()
	queue.PrintStatus()
	queue.PrintHistory()
	
	// 场景2: 优先级任务队列
	fmt.Println("\n\n--- 场景2: 优先级任务队列 ---")
	priorityQueue := NewPriorityTaskQueue()
	
	priorityQueue.AddTask(NewProcessDataCommand([]int{10, 20, 30}, "avg"), "normal")
	priorityQueue.AddTask(NewBackupDatabaseCommand("production", "/backup/db"), "high")
	priorityQueue.AddTask(NewSendEmailCommand("admin@example.com", "系统通知", "系统将在今晚维护"), "high")
	priorityQueue.AddTask(NewGenerateReportCommand("月度报告", "2024-01-01", "2024-01-31"), "low")
	priorityQueue.AddTask(NewProcessDataCommand([]int{5, 15, 25, 35}, "max"), "normal")
	
	priorityQueue.ProcessAll()
	
	// 场景3: 逐个处理任务
	fmt.Println("\n\n--- 场景3: 逐个处理任务 ---")
	queue2 := NewTaskQueue()
	
	queue2.AddTask(NewSendEmailCommand("team@example.com", "会议通知", "明天下午2点开会"))
	queue2.AddTask(NewProcessDataCommand([]int{100, 200, 300}, "sum"))
	queue2.AddTask(NewBackupDatabaseCommand("test", "/backup/test"))
	
	queue2.PrintStatus()
	
	for queue2.GetQueueSize() > 0 {
		fmt.Println("\n按回车键处理下一个任务...")
		// 在实际应用中，这里可以等待用户输入或其他触发条件
		time.Sleep(500 * time.Millisecond)
		queue2.ProcessNext()
		queue2.PrintStatus()
	}
	
	queue2.PrintHistory()

	fmt.Println("\n=== 示例结束 ===")
}
