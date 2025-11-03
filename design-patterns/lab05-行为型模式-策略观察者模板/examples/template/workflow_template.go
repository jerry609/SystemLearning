package main

import (
	"fmt"
	"strings"
	"time"
)

// 工作流步骤接口
type WorkflowStep interface {
	Initialize() error
	Execute() error
	Validate() error
	Cleanup() error
	GetName() string
}

// 基础工作流步骤
type BaseWorkflowStep struct {
	name string
}

func (b *BaseWorkflowStep) GetName() string {
	return b.name
}

func (b *BaseWorkflowStep) Cleanup() error {
	// 默认不需要清理
	return nil
}

// 工作流模板
type WorkflowTemplate struct {
	step WorkflowStep
}

func NewWorkflowTemplate(step WorkflowStep) *WorkflowTemplate {
	return &WorkflowTemplate{step: step}
}

// 模板方法：执行工作流
func (w *WorkflowTemplate) Run() error {
	fmt.Printf("\n=== 执行工作流: %s ===\n\n", w.step.GetName())
	startTime := time.Now()
	
	// 步骤 1: 初始化
	fmt.Println("步骤 1: 初始化")
	if err := w.step.Initialize(); err != nil {
		return fmt.Errorf("初始化失败: %w", err)
	}
	fmt.Println("✓ 初始化完成\n")
	
	// 步骤 2: 执行
	fmt.Println("步骤 2: 执行任务")
	if err := w.step.Execute(); err != nil {
		return fmt.Errorf("执行失败: %w", err)
	}
	fmt.Println("✓ 执行完成\n")
	
	// 步骤 3: 验证
	fmt.Println("步骤 3: 验证结果")
	if err := w.step.Validate(); err != nil {
		return fmt.Errorf("验证失败: %w", err)
	}
	fmt.Println("✓ 验证通过\n")
	
	// 步骤 4: 清理
	fmt.Println("步骤 4: 清理资源")
	if err := w.step.Cleanup(); err != nil {
		return fmt.Errorf("清理失败: %w", err)
	}
	fmt.Println("✓ 清理完成\n")
	
	duration := time.Since(startTime)
	fmt.Printf("=== 工作流完成 (耗时: %v) ===\n", duration)
	
	return nil
}

// 数据备份工作流
type DataBackupWorkflow struct {
	BaseWorkflowStep
	source      string
	destination string
	fileCount   int
}

func NewDataBackupWorkflow(source, destination string) *DataBackupWorkflow {
	return &DataBackupWorkflow{
		BaseWorkflowStep: BaseWorkflowStep{name: "数据备份"},
		source:           source,
		destination:      destination,
	}
}

func (d *DataBackupWorkflow) Initialize() error {
	fmt.Println("  检查源目录和目标目录...")
	fmt.Printf("  源目录: %s\n", d.source)
	fmt.Printf("  目标目录: %s\n", d.destination)
	time.Sleep(100 * time.Millisecond)
	return nil
}

func (d *DataBackupWorkflow) Execute() error {
	fmt.Println("  开始备份文件...")
	// 模拟备份过程
	for i := 1; i <= 5; i++ {
		fmt.Printf("  备份文件 %d/5...\n", i)
		time.Sleep(100 * time.Millisecond)
	}
	d.fileCount = 5
	return nil
}

func (d *DataBackupWorkflow) Validate() error {
	fmt.Println("  验证备份完整性...")
	fmt.Printf("  已备份 %d 个文件\n", d.fileCount)
	time.Sleep(100 * time.Millisecond)
	return nil
}

func (d *DataBackupWorkflow) Cleanup() error {
	fmt.Println("  清理临时文件...")
	time.Sleep(50 * time.Millisecond)
	return nil
}

// 数据库迁移工作流
type DatabaseMigrationWorkflow struct {
	BaseWorkflowStep
	sourceDB string
	targetDB string
	tables   []string
}

func NewDatabaseMigrationWorkflow(sourceDB, targetDB string, tables []string) *DatabaseMigrationWorkflow {
	return &DatabaseMigrationWorkflow{
		BaseWorkflowStep: BaseWorkflowStep{name: "数据库迁移"},
		sourceDB:         sourceDB,
		targetDB:         targetDB,
		tables:           tables,
	}
}

func (d *DatabaseMigrationWorkflow) Initialize() error {
	fmt.Println("  连接源数据库和目标数据库...")
	fmt.Printf("  源数据库: %s\n", d.sourceDB)
	fmt.Printf("  目标数据库: %s\n", d.targetDB)
	fmt.Printf("  待迁移表: %v\n", d.tables)
	time.Sleep(100 * time.Millisecond)
	return nil
}

func (d *DatabaseMigrationWorkflow) Execute() error {
	fmt.Println("  开始迁移数据...")
	for i, table := range d.tables {
		fmt.Printf("  迁移表 %d/%d: %s\n", i+1, len(d.tables), table)
		time.Sleep(150 * time.Millisecond)
	}
	return nil
}

func (d *DatabaseMigrationWorkflow) Validate() error {
	fmt.Println("  验证数据一致性...")
	for _, table := range d.tables {
		fmt.Printf("  验证表: %s\n", table)
		time.Sleep(100 * time.Millisecond)
	}
	return nil
}

func (d *DatabaseMigrationWorkflow) Cleanup() error {
	fmt.Println("  关闭数据库连接...")
	time.Sleep(50 * time.Millisecond)
	return nil
}

// 代码部署工作流
type CodeDeploymentWorkflow struct {
	BaseWorkflowStep
	environment string
	version     string
	services    []string
}

func NewCodeDeploymentWorkflow(environment, version string, services []string) *CodeDeploymentWorkflow {
	return &CodeDeploymentWorkflow{
		BaseWorkflowStep: BaseWorkflowStep{name: "代码部署"},
		environment:      environment,
		version:          version,
		services:         services,
	}
}

func (c *CodeDeploymentWorkflow) Initialize() error {
	fmt.Println("  准备部署环境...")
	fmt.Printf("  环境: %s\n", c.environment)
	fmt.Printf("  版本: %s\n", c.version)
	fmt.Printf("  服务: %v\n", c.services)
	time.Sleep(100 * time.Millisecond)
	return nil
}

func (c *CodeDeploymentWorkflow) Execute() error {
	fmt.Println("  开始部署服务...")
	for i, service := range c.services {
		fmt.Printf("  部署服务 %d/%d: %s\n", i+1, len(c.services), service)
		time.Sleep(200 * time.Millisecond)
	}
	return nil
}

func (c *CodeDeploymentWorkflow) Validate() error {
	fmt.Println("  验证服务健康状态...")
	for _, service := range c.services {
		fmt.Printf("  检查服务: %s - 健康\n", service)
		time.Sleep(100 * time.Millisecond)
	}
	return nil
}

func (c *CodeDeploymentWorkflow) Cleanup() error {
	fmt.Println("  清理旧版本...")
	time.Sleep(50 * time.Millisecond)
	return nil
}

// 测试执行工作流
type TestExecutionWorkflow struct {
	BaseWorkflowStep
	testSuite  string
	testCases  []string
	passedTests int
}

func NewTestExecutionWorkflow(testSuite string, testCases []string) *TestExecutionWorkflow {
	return &TestExecutionWorkflow{
		BaseWorkflowStep: BaseWorkflowStep{name: "测试执行"},
		testSuite:        testSuite,
		testCases:        testCases,
	}
}

func (t *TestExecutionWorkflow) Initialize() error {
	fmt.Println("  初始化测试环境...")
	fmt.Printf("  测试套件: %s\n", t.testSuite)
	fmt.Printf("  测试用例数: %d\n", len(t.testCases))
	time.Sleep(100 * time.Millisecond)
	return nil
}

func (t *TestExecutionWorkflow) Execute() error {
	fmt.Println("  执行测试用例...")
	t.passedTests = 0
	for i, testCase := range t.testCases {
		fmt.Printf("  运行测试 %d/%d: %s\n", i+1, len(t.testCases), testCase)
		time.Sleep(100 * time.Millisecond)
		t.passedTests++
	}
	return nil
}

func (t *TestExecutionWorkflow) Validate() error {
	fmt.Println("  验证测试结果...")
	fmt.Printf("  通过: %d/%d\n", t.passedTests, len(t.testCases))
	if t.passedTests < len(t.testCases) {
		return fmt.Errorf("有测试用例失败")
	}
	return nil
}

func (t *TestExecutionWorkflow) Cleanup() error {
	fmt.Println("  生成测试报告...")
	fmt.Println("  清理测试数据...")
	time.Sleep(50 * time.Millisecond)
	return nil
}

// 批量工作流执行器
type BatchWorkflowExecutor struct {
	workflows []WorkflowStep
}

func NewBatchWorkflowExecutor() *BatchWorkflowExecutor {
	return &BatchWorkflowExecutor{
		workflows: make([]WorkflowStep, 0),
	}
}

func (b *BatchWorkflowExecutor) AddWorkflow(workflow WorkflowStep) {
	b.workflows = append(b.workflows, workflow)
}

func (b *BatchWorkflowExecutor) ExecuteAll() error {
	fmt.Println("\n=== 批量执行工作流 ===")
	fmt.Printf("总共 %d 个工作流\n", len(b.workflows))
	
	for i, workflow := range b.workflows {
		fmt.Printf("\n[%d/%d] ", i+1, len(b.workflows))
		template := NewWorkflowTemplate(workflow)
		if err := template.Run(); err != nil {
			return fmt.Errorf("工作流 %s 执行失败: %w", workflow.GetName(), err)
		}
	}
	
	fmt.Println("\n=== 所有工作流执行完成 ===")
	return nil
}

func main() {
	fmt.Println("=== 工作流模板模式示例 ===")
	
	// 场景 1: 数据备份工作流
	fmt.Println("\n【场景 1: 数据备份工作流】")
	backupWorkflow := NewDataBackupWorkflow("/data/source", "/backup/destination")
	backupTemplate := NewWorkflowTemplate(backupWorkflow)
	if err := backupTemplate.Run(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}
	
	fmt.Println("\n" + strings.Repeat("=", 60))
	
	// 场景 2: 数据库迁移工作流
	fmt.Println("\n【场景 2: 数据库迁移工作流】")
	migrationWorkflow := NewDatabaseMigrationWorkflow(
		"mysql://source:3306/db",
		"mysql://target:3306/db",
		[]string{"users", "orders", "products"},
	)
	migrationTemplate := NewWorkflowTemplate(migrationWorkflow)
	if err := migrationTemplate.Run(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}
	
	fmt.Println("\n" + strings.Repeat("=", 60))
	
	// 场景 3: 代码部署工作流
	fmt.Println("\n【场景 3: 代码部署工作流】")
	deploymentWorkflow := NewCodeDeploymentWorkflow(
		"production",
		"v1.2.0",
		[]string{"api-service", "web-service", "worker-service"},
	)
	deploymentTemplate := NewWorkflowTemplate(deploymentWorkflow)
	if err := deploymentTemplate.Run(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}
	
	fmt.Println("\n" + strings.Repeat("=", 60))
	
	// 场景 4: 测试执行工作流
	fmt.Println("\n【场景 4: 测试执行工作流】")
	testWorkflow := NewTestExecutionWorkflow(
		"Integration Tests",
		[]string{"TestUserAPI", "TestOrderAPI", "TestPaymentAPI"},
	)
	testTemplate := NewWorkflowTemplate(testWorkflow)
	if err := testTemplate.Run(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}
	
	fmt.Println("\n" + strings.Repeat("=", 60))
	
	// 场景 5: 批量执行工作流
	fmt.Println("\n【场景 5: 批量执行工作流】")
	
	batchExecutor := NewBatchWorkflowExecutor()
	batchExecutor.AddWorkflow(NewDataBackupWorkflow("/data/app1", "/backup/app1"))
	batchExecutor.AddWorkflow(NewDataBackupWorkflow("/data/app2", "/backup/app2"))
	batchExecutor.AddWorkflow(NewTestExecutionWorkflow("Unit Tests", []string{"TestA", "TestB"}))
	
	if err := batchExecutor.ExecuteAll(); err != nil {
		fmt.Printf("错误: %v\n", err)
	}
	
	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n💡 工作流模板的应用场景:")
	fmt.Println("- CI/CD 流程: 构建、测试、部署")
	fmt.Println("- 数据处理: ETL 流程")
	fmt.Println("- 任务调度: 定时任务执行")
	fmt.Println("- 业务流程: 订单处理、审批流程")
}
