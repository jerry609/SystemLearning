package main

import (
	"fmt"
	"time"
)

// Task 任务接口
type Task interface {
	Execute() error
	Undo() error
	String() string
	GetPriority() int
}

// EmailTask 邮件任务
type EmailTask struct {
	to      string
	subject string
	body    string
	backup  string
}

func NewEmailTask(to, subject, body string) *EmailTask {
	return &EmailTask{to: to, subject: subject, body: body}
}

func (t *EmailTask) Execute() error {
	t.backup = fmt.Sprintf("Email sent to %s", t.to)
	fmt.Printf("[邮件任务] 发送邮件到 %s\n", t.to)
	fmt.Printf("[邮件任务] 主题: %s\n", t.subject)
	time.Sleep(50 * time.Millisecond)
	fmt.Println("[邮件任务] ✓ 邮件发送成功")
	return nil
}

func (t *EmailTask) Undo() error {
	fmt.Printf("[邮件任务] 撤销邮件发送: %s\n", t.to)
	return nil
}

func (t *EmailTask) String() string {
	return fmt.Sprintf("发送邮件(%s)", t.to)
}

func (t *EmailTask) GetPriority() int {
	return 2
}

// BackupTask 备份任务
type BackupTask struct {
	database string
	path     string
	backup   string
}

func NewBackupTask(database, path string) *BackupTask {
	return &BackupTask{database: database, path: path}
}

func (t *BackupTask) Execute() error {
	t.backup = fmt.Sprintf("Backup of %s", t.database)
	fmt.Printf("[备份任务] 备份数据库 %s\n", t.database)
	fmt.Printf("[备份任务] 备份路径: %s\n", t.path)
	time.Sleep(100 * time.Millisecond)
	fmt.Println("[备份任务] ✓ 备份完成")
	return nil
}

func (t *BackupTask) Undo() error {
	fmt.Printf("[备份任务] 删除备份: %s\n", t.path)
	return nil
}

func (t *BackupTask) String() string {
	return fmt.Sprintf("数据备份(%s)", t.database)
}

func (t *BackupTask) GetPriority() int {
	return 3
}

// ReportTask 报告任务
type ReportTask struct {
	reportType string
	startDate  string
	endDate    string
	backup     string
}

func NewReportTask(reportType, startDate, endDate string) *ReportTask {
	return &ReportTask{reportType: reportType, startDate: startDate, endDate: endDate}
}

func (t *ReportTask) Execute() error {
	t.backup = fmt.Sprintf("Report %s generated", t.reportType)
	fmt.Printf("[报告任务] 生成报告: %s\n", t.reportType)
	fmt.Printf("[报告任务] 时间范围: %s ~ %s\n", t.startDate, t.endDate)
	time.Sleep(150 * time.Millisecond)
	fmt.Println("[报告任务] ✓ 报告生成完成")
	return nil
}

func (t *ReportTask) Undo() error {
	fmt.Printf("[报告任务] 删除报告: %s\n", t.reportType)
	return nil
}

func (t *ReportTask) String() string {
	return fmt.Sprintf("生成报告(%s)", t.reportType)
}

func (t *ReportTask) GetPriority() int {
	return 1
}

// DataProcessTask 数据处理任务
type DataProcessTask struct {
	operation string
	data      []int
	backup    string
}

func NewDataProcessTask(operation string, data []int) *DataProcessTask {
	return &DataProcessTask{operation: operation, data: data}
}

func (t *DataProcessTask) Execute() error {
	t.backup = fmt.Sprintf("Data processed: %s", t.operation)
	fmt.Printf("[数据处理] 操作: %s, 数据量: %d\n", t.operation, len(t.data))
	time.Sleep(80 * time.Millisecond)
	fmt.Println("[数据处理] ✓ 处理完成")
	return nil
}

func (t *DataProcessTask) Undo() error {
	fmt.Printf("[数据处理] 撤销操作: %s\n", t.operation)
	return nil
}

func (t *DataProcessTask) String() string {
	return fmt.Sprintf("数据处理(%s)", t.operation)
}

func (t *DataProcessTask) GetPriority() int {
	return 2
}

// Scheduler 任务调度器
type Scheduler struct {
	queue   []Task
	history []Task
	current int
}

func NewScheduler() *Scheduler {
	return &Scheduler{
		queue:   make([]Task, 0),
		history: make([]Task, 0),
		current: 0,
	}
}

func (s *Scheduler) Submit(task Task) {
	s.queue = append(s.queue, task)
	fmt.Printf("[调度器] 提交任务: %s\n", task.String())
}

func (s *Scheduler) ExecuteNext() error {
	if len(s.queue) == 0 {
		return fmt.Errorf("队列为空")
	}
	
	task := s.queue[0]
	s.queue = s.queue[1:]
	
	fmt.Printf("\n[调度器] 执行任务: %s\n", task.String())
	startTime := time.Now()
	
	if err := task.Execute(); err != nil {
		return err
	}
	
	duration := time.Since(startTime)
	fmt.Printf("[调度器] 任务完成，耗时: %v\n", duration)
	
	s.history = s.history[:s.current]
	s.history = append(s.history, task)
	s.current++
	
	return nil
}

func (s *Scheduler) ExecuteAll() {
	fmt.Println("\n[调度器] 开始执行所有任务")
	for len(s.queue) > 0 {
		if err := s.ExecuteNext(); err != nil {
			fmt.Printf("[调度器] 错误: %v\n", err)
			break
		}
	}
	fmt.Println("[调度器] 所有任务执行完成")
}

func (s *Scheduler) Undo() error {
	if s.current == 0 {
		return fmt.Errorf("没有可撤销的任务")
	}
	
	s.current--
	task := s.history[s.current]
	fmt.Printf("\n[调度器] 撤销任务: %s\n", task.String())
	return task.Undo()
}

func (s *Scheduler) Redo() error {
	if s.current >= len(s.history) {
		return fmt.Errorf("没有可重做的任务")
	}
	
	task := s.history[s.current]
	fmt.Printf("\n[调度器] 重做任务: %s\n", task.String())
	if err := task.Execute(); err != nil {
		return err
	}
	s.current++
	return nil
}

func (s *Scheduler) GetQueueSize() int {
	return len(s.queue)
}

func (s *Scheduler) GetHistorySize() int {
	return len(s.history)
}

func (s *Scheduler) PrintHistory() {
	fmt.Println("\n任务历史:")
	for i, task := range s.history {
		marker := "  "
		if i == s.current {
			marker = "→ "
		}
		fmt.Printf("%s%d. %s\n", marker, i+1, task.String())
	}
	fmt.Printf("当前位置: %d/%d\n", s.current, len(s.history))
}

// PriorityScheduler 优先级调度器
type PriorityScheduler struct {
	queue   []Task
	history []Task
	current int
}

func NewPriorityScheduler() *PriorityScheduler {
	return &PriorityScheduler{
		queue:   make([]Task, 0),
		history: make([]Task, 0),
		current: 0,
	}
}

func (s *PriorityScheduler) Submit(task Task) {
	s.queue = append(s.queue, task)
	s.sortByPriority()
	fmt.Printf("[优先级调度器] 提交任务: %s (优先级: %d)\n", task.String(), task.GetPriority())
}

func (s *PriorityScheduler) sortByPriority() {
	for i := 0; i < len(s.queue)-1; i++ {
		for j := i + 1; j < len(s.queue); j++ {
			if s.queue[i].GetPriority() < s.queue[j].GetPriority() {
				s.queue[i], s.queue[j] = s.queue[j], s.queue[i]
			}
		}
	}
}

func (s *PriorityScheduler) ExecuteNext() error {
	if len(s.queue) == 0 {
		return fmt.Errorf("队列为空")
	}
	
	task := s.queue[0]
	s.queue = s.queue[1:]
	
	fmt.Printf("\n[优先级调度器] 执行任务: %s (优先级: %d)\n", task.String(), task.GetPriority())
	
	if err := task.Execute(); err != nil {
		return err
	}
	
	s.history = s.history[:s.current]
	s.history = append(s.history, task)
	s.current++
	
	return nil
}

func (s *PriorityScheduler) ExecuteAll() {
	fmt.Println("\n[优先级调度器] 开始执行所有任务（按优先级）")
	for len(s.queue) > 0 {
		if err := s.ExecuteNext(); err != nil {
			fmt.Printf("[优先级调度器] 错误: %v\n", err)
			break
		}
	}
	fmt.Println("[优先级调度器] 所有任务执行完成")
}
