package main

import "fmt"

func main() {
	fmt.Println("=== 文件系统项目 ===\n")

	// 创建根目录
	root := NewDirectory("root")

	// 创建目录结构
	home := NewDirectory("home")
	user := NewDirectory("user")
	documents := NewDirectory("documents")
	pictures := NewDirectory("pictures")
	
	varDir := NewDirectory("var")
	logDir := NewDirectory("log")

	// 创建文件
	readme := NewFile("README.md", 5)
	report := NewFile("report.pdf", 1024)
	vacation := NewFile("vacation.jpg", 2048)
	family := NewFile("family.jpg", 1536)
	syslog := NewFile("system.log", 10240)
	errlog := NewFile("error.log", 1024)

	// 构建文件系统树
	root.Add(home)
	root.Add(varDir)
	
	home.Add(user)
	user.Add(documents)
	user.Add(pictures)
	
	documents.Add(readme)
	documents.Add(report)
	
	pictures.Add(vacation)
	pictures.Add(family)
	
	varDir.Add(logDir)
	logDir.Add(syslog)
	logDir.Add(errlog)

	fmt.Println("创建文件系统结构...")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

	// 显示文件系统结构
	fmt.Println("文件系统结构:")
	root.Display("")
	fmt.Println()

	// 统计信息
	fmt.Printf("总大小: %d KB\n", root.GetSize())
	fmt.Printf("文件总数: %d\n", root.CountFiles())
	fmt.Println()

	// 查找文件
	fmt.Println("查找文件 'report.pdf':")
	found := root.Find("report.pdf")
	if found != nil {
		fmt.Printf("找到: %s (大小: %d KB, 权限: %s)\n", 
			found.GetName(), found.GetSize(), found.GetPermissions())
	}
	fmt.Println()

	// 搜索文件
	fmt.Println("搜索 .jpg 文件:")
	results := root.Search(".jpg")
	for _, result := range results {
		fmt.Printf("  - %s\n", result.GetPath())
	}
	fmt.Println()

	// 列出所有文件
	fmt.Println("所有文件列表:")
	allFiles := root.ListAllFiles()
	for i, file := range allFiles {
		fmt.Printf("%d. %s\n", i+1, file)
	}
	fmt.Println()

	// 复制目录
	fmt.Println("复制目录 'documents' 到 'backup':")
	backup := documents.Copy().(*Directory)
	backup.name = "backup"
	user.Add(backup)
	fmt.Println("✅ 复制成功\n")

	// 移动文件
	fmt.Println("移动文件 'error.log' 到 'home/user':")
	logDir.Move(errlog, user)
	fmt.Println("✅ 移动成功\n")

	// 显示最终结构
	fmt.Println("最终文件系统结构:")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	root.Display("")
	fmt.Println()

	fmt.Printf("总大小: %d KB\n", root.GetSize())
	fmt.Printf("文件总数: %d\n", root.CountFiles())
	fmt.Println()

	// 权限管理演示
	fmt.Println("权限管理:")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("修改前 - report.pdf 权限: %s\n", report.GetPermissions())
	report.SetPermissions("r--")
	fmt.Printf("修改后 - report.pdf 权限: %s\n", report.GetPermissions())
	fmt.Println()

	fmt.Println("=== 项目演示结束 ===")
}
