package main

import (
	"testing"
)

func TestSchedulerSubmit(t *testing.T) {
	scheduler := NewScheduler()
	task := NewEmailTask("test@example.com", "测试", "测试邮件")
	
	scheduler.Submit(task)
	
	if scheduler.GetQueueSize() != 1 {
		t.Errorf("队列大小应该是1，实际是 %d", scheduler.GetQueueSize())
	}
}

func TestSchedulerExecute(t *testing.T) {
	scheduler := NewScheduler()
	task := NewEmailTask("test@example.com", "测试", "测试邮件")
	
	scheduler.Submit(task)
	
	if err := scheduler.ExecuteNext(); err != nil {
		t.Errorf("执行任务失败: %v", err)
	}
	
	if scheduler.GetQueueSize() != 0 {
		t.Errorf("执行后队列应该为空，实际大小是 %d", scheduler.GetQueueSize())
	}
	
	if scheduler.GetHistorySize() != 1 {
		t.Errorf("历史记录应该有1条，实际有 %d 条", scheduler.GetHistorySize())
	}
}

func TestSchedulerUndo(t *testing.T) {
	scheduler := NewScheduler()
	task := NewEmailTask("test@example.com", "测试", "测试邮件")
	
	scheduler.Submit(task)
	scheduler.ExecuteNext()
	
	if err := scheduler.Undo(); err != nil {
		t.Errorf("撤销失败: %v", err)
	}
	
	if scheduler.current != 0 {
		t.Errorf("撤销后当前位置应该是0，实际是 %d", scheduler.current)
	}
}

func TestSchedulerRedo(t *testing.T) {
	scheduler := NewScheduler()
	task := NewEmailTask("test@example.com", "测试", "测试邮件")
	
	scheduler.Submit(task)
	scheduler.ExecuteNext()
	scheduler.Undo()
	
	if err := scheduler.Redo(); err != nil {
		t.Errorf("重做失败: %v", err)
	}
	
	if scheduler.current != 1 {
		t.Errorf("重做后当前位置应该是1，实际是 %d", scheduler.current)
	}
}

func TestSchedulerExecuteAll(t *testing.T) {
	scheduler := NewScheduler()
	
	scheduler.Submit(NewEmailTask("test1@example.com", "测试1", "测试邮件1"))
	scheduler.Submit(NewEmailTask("test2@example.com", "测试2", "测试邮件2"))
	scheduler.Submit(NewEmailTask("test3@example.com", "测试3", "测试邮件3"))
	
	scheduler.ExecuteAll()
	
	if scheduler.GetQueueSize() != 0 {
		t.Errorf("执行所有任务后队列应该为空，实际大小是 %d", scheduler.GetQueueSize())
	}
	
	if scheduler.GetHistorySize() != 3 {
		t.Errorf("历史记录应该有3条，实际有 %d 条", scheduler.GetHistorySize())
	}
}

func TestSchedulerEmptyQueue(t *testing.T) {
	scheduler := NewScheduler()
	
	if err := scheduler.ExecuteNext(); err == nil {
		t.Error("空队列执行应该失败")
	}
}

func TestSchedulerUndoEmpty(t *testing.T) {
	scheduler := NewScheduler()
	
	if err := scheduler.Undo(); err == nil {
		t.Error("空历史撤销应该失败")
	}
}

func TestSchedulerRedoEmpty(t *testing.T) {
	scheduler := NewScheduler()
	
	if err := scheduler.Redo(); err == nil {
		t.Error("无可重做时重做应该失败")
	}
}

func TestPriorityScheduler(t *testing.T) {
	scheduler := NewPriorityScheduler()
	
	task1 := NewReportTask("报告", "2024-01-01", "2024-01-31")  // 优先级 1
	task2 := NewEmailTask("test@example.com", "邮件", "内容")    // 优先级 2
	task3 := NewBackupTask("db", "/backup")                    // 优先级 3
	
	scheduler.Submit(task1)
	scheduler.Submit(task2)
	scheduler.Submit(task3)
	
	// 验证队列按优先级排序
	if scheduler.queue[0].GetPriority() != 3 {
		t.Errorf("第一个任务优先级应该是3，实际是 %d", scheduler.queue[0].GetPriority())
	}
	
	if scheduler.queue[1].GetPriority() != 2 {
		t.Errorf("第二个任务优先级应该是2，实际是 %d", scheduler.queue[1].GetPriority())
	}
	
	if scheduler.queue[2].GetPriority() != 1 {
		t.Errorf("第三个任务优先级应该是1，实际是 %d", scheduler.queue[2].GetPriority())
	}
}

func TestTaskPriority(t *testing.T) {
	email := NewEmailTask("test@example.com", "测试", "内容")
	backup := NewBackupTask("db", "/backup")
	report := NewReportTask("报告", "2024-01-01", "2024-01-31")
	
	if email.GetPriority() != 2 {
		t.Errorf("邮件任务优先级应该是2，实际是 %d", email.GetPriority())
	}
	
	if backup.GetPriority() != 3 {
		t.Errorf("备份任务优先级应该是3，实际是 %d", backup.GetPriority())
	}
	
	if report.GetPriority() != 1 {
		t.Errorf("报告任务优先级应该是1，实际是 %d", report.GetPriority())
	}
}
