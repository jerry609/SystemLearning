package main

import "fmt"

func main() {
	fmt.Println("=== 任务调度器系统 ===\n")

	// 场景1: 基本任务执行
	fmt.Println("--- 场景1: 基本任务执行 ---")
	scheduler := NewScheduler()
	
	scheduler.Submit(NewEmailTask("user@example.com", "欢迎", "欢迎使用我们的服务"))
	scheduler.Submit(NewBackupTask("production", "/backup/db"))
	scheduler.Submit(NewReportTask("销售报告", "2024-01-01", "2024-01-31"))
	
	fmt.Printf("\n队列中有 %d 个任务\n", scheduler.GetQueueSize())
	scheduler.ExecuteAll()
	fmt.Printf("\n已完成 %d 个任务\n", scheduler.GetHistorySize())

	// 场景2: 撤销和重做
	fmt.Println("\n\n--- 场景2: 撤销和重做 ---")
	scheduler2 := NewScheduler()
	
	scheduler2.Submit(NewDataProcessTask("导入", []int{1, 2, 3, 4, 5}))
	scheduler2.Submit(NewEmailTask("admin@example.com", "通知", "数据导入完成"))
	
	scheduler2.ExecuteNext()
	scheduler2.ExecuteNext()
	
	scheduler2.PrintHistory()
	
	fmt.Println("\n撤销最后一个任务:")
	scheduler2.Undo()
	
	scheduler2.PrintHistory()
	
	fmt.Println("\n重做任务:")
	scheduler2.Redo()
	
	scheduler2.PrintHistory()

	// 场景3: 优先级调度
	fmt.Println("\n\n--- 场景3: 优先级调度 ---")
	priorityScheduler := NewPriorityScheduler()
	
	priorityScheduler.Submit(NewReportTask("月度报告", "2024-01-01", "2024-01-31"))
	priorityScheduler.Submit(NewBackupTask("production", "/backup/db"))
	priorityScheduler.Submit(NewEmailTask("team@example.com", "会议通知", "明天下午2点开会"))
	priorityScheduler.Submit(NewDataProcessTask("导出", []int{1, 2, 3}))
	
	priorityScheduler.ExecuteAll()

	// 场景4: 批量任务
	fmt.Println("\n\n--- 场景4: 批量任务执行 ---")
	scheduler3 := NewScheduler()
	
	for i := 1; i <= 3; i++ {
		scheduler3.Submit(NewEmailTask(
			fmt.Sprintf("user%d@example.com", i),
			"批量通知",
			fmt.Sprintf("这是第 %d 封邮件", i),
		))
	}
	
	scheduler3.ExecuteAll()

	// 场景5: 错误处理
	fmt.Println("\n\n--- 场景5: 错误处理 ---")
	scheduler4 := NewScheduler()
	
	fmt.Println("尝试执行空队列:")
	if err := scheduler4.ExecuteNext(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}
	
	fmt.Println("\n尝试撤销（无历史）:")
	if err := scheduler4.Undo(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}
	
	scheduler4.Submit(NewEmailTask("test@example.com", "测试", "测试邮件"))
	scheduler4.ExecuteNext()
	
	fmt.Println("\n尝试重做（无可重做）:")
	if err := scheduler4.Redo(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}

	fmt.Println("\n\n=== 系统演示完成 ===")
}
