package main

import (
	"fmt"
	"strings"
	"time"
)

// 模板方法模式 - 框架模板示例
// 本示例展示了如何使用模板方法模式构建一个测试框架

// TestCase 定义测试用例接口（原语操作）
type TestCase interface {
	SetUp()           // 测试前的准备工作
	RunTest()         // 执行测试
	TearDown()        // 测试后的清理工作
	GetName() string  // 获取测试名称
}

// TestRunner 测试运行器（模板类）
type TestRunner struct {
	testCase TestCase
	result   *TestResult
}

// TestResult 测试结果
type TestResult struct {
	Name      string
	Passed    bool
	Duration  time.Duration
	Error     error
	StartTime time.Time
	EndTime   time.Time
}

// NewTestRunner 创建测试运行器
func NewTestRunner(testCase TestCase) *TestRunner {
	return &TestRunner{
		testCase: testCase,
		result: &TestResult{
			Name: testCase.GetName(),
		},
	}
}

// Run 模板方法 - 定义测试执行的算法骨架
func (r *TestRunner) Run() *TestResult {
	fmt.Printf("\n=== 运行测试: %s ===\n", r.testCase.GetName())
	
	// 记录开始时间
	r.result.StartTime = time.Now()
	
	// 步骤 1: 执行 SetUp（前置处理）
	fmt.Println("\n[1/3] 执行 SetUp...")
	r.beforeTest()
	r.testCase.SetUp()
	
	// 步骤 2: 执行测试（核心步骤）
	fmt.Println("\n[2/3] 执行测试...")
	r.executeTest()
	
	// 步骤 3: 执行 TearDown（后置处理）
	fmt.Println("\n[3/3] 执行 TearDown...")
	r.testCase.TearDown()
	r.afterTest()
	
	// 记录结束时间和持续时间
	r.result.EndTime = time.Now()
	r.result.Duration = r.result.EndTime.Sub(r.result.StartTime)
	
	// 打印测试结果
	r.printResult()
	
	return r.result
}

// beforeTest 钩子方法 - 测试前的公共处理
func (r *TestRunner) beforeTest() {
	fmt.Println("  → 初始化测试环境")
}

// executeTest 执行测试的核心逻辑
func (r *TestRunner) executeTest() {
	defer func() {
		if err := recover(); err != nil {
			r.result.Passed = false
			r.result.Error = fmt.Errorf("测试失败: %v", err)
			fmt.Printf("  ✗ 测试失败: %v\n", err)
		}
	}()
	
	// 调用具体的测试方法
	r.testCase.RunTest()
	
	// 如果没有 panic，则测试通过
	r.result.Passed = true
	fmt.Println("  ✓ 测试通过")
}

// afterTest 钩子方法 - 测试后的公共处理
func (r *TestRunner) afterTest() {
	fmt.Println("  → 清理测试环境")
}

// printResult 打印测试结果
func (r *TestRunner) printResult() {
	fmt.Println("\n=== 测试结果 ===")
	fmt.Printf("测试名称: %s\n", r.result.Name)
	fmt.Printf("测试状态: %s\n", r.getStatusString())
	fmt.Printf("执行时间: %v\n", r.result.Duration)
	if r.result.Error != nil {
		fmt.Printf("错误信息: %v\n", r.result.Error)
	}
	fmt.Println("================")
}

// getStatusString 获取状态字符串
func (r *TestRunner) getStatusString() string {
	if r.result.Passed {
		return "✓ PASSED"
	}
	return "✗ FAILED"
}

// ============ 具体测试用例实现 ============

// DatabaseTest 数据库测试用例
type DatabaseTest struct {
	name       string
	connection interface{}
}

func NewDatabaseTest() *DatabaseTest {
	return &DatabaseTest{
		name: "数据库连接测试",
	}
}

func (t *DatabaseTest) GetName() string {
	return t.name
}

func (t *DatabaseTest) SetUp() {
	fmt.Println("  → 建立数据库连接")
	fmt.Println("  → 创建测试表")
	fmt.Println("  → 插入测试数据")
	t.connection = "mock_db_connection"
	time.Sleep(100 * time.Millisecond) // 模拟耗时操作
}

func (t *DatabaseTest) RunTest() {
	fmt.Println("  → 执行数据库查询")
	fmt.Println("  → 验证查询结果")
	fmt.Println("  → 测试事务回滚")
	time.Sleep(50 * time.Millisecond) // 模拟测试执行
}

func (t *DatabaseTest) TearDown() {
	fmt.Println("  → 删除测试数据")
	fmt.Println("  → 关闭数据库连接")
	t.connection = nil
	time.Sleep(50 * time.Millisecond) // 模拟清理操作
}

// APITest API 测试用例
type APITest struct {
	name   string
	server interface{}
}

func NewAPITest() *APITest {
	return &APITest{
		name: "API 接口测试",
	}
}

func (t *APITest) GetName() string {
	return t.name
}

func (t *APITest) SetUp() {
	fmt.Println("  → 启动测试服务器")
	fmt.Println("  → 初始化测试数据")
	t.server = "mock_server"
	time.Sleep(100 * time.Millisecond)
}

func (t *APITest) RunTest() {
	fmt.Println("  → 发送 GET 请求")
	fmt.Println("  → 验证响应状态码")
	fmt.Println("  → 验证响应数据")
	time.Sleep(50 * time.Millisecond)
}

func (t *APITest) TearDown() {
	fmt.Println("  → 清理测试数据")
	fmt.Println("  → 关闭测试服务器")
	t.server = nil
	time.Sleep(50 * time.Millisecond)
}

// FailingTest 失败的测试用例（演示错误处理）
type FailingTest struct {
	name string
}

func NewFailingTest() *FailingTest {
	return &FailingTest{
		name: "失败测试示例",
	}
}

func (t *FailingTest) GetName() string {
	return t.name
}

func (t *FailingTest) SetUp() {
	fmt.Println("  → 准备测试环境")
}

func (t *FailingTest) RunTest() {
	fmt.Println("  → 执行会失败的测试")
	panic("断言失败: expected 1, got 2")
}

func (t *FailingTest) TearDown() {
	fmt.Println("  → 清理测试环境（即使测试失败也会执行）")
}

// ============ 测试套件 ============

// TestSuite 测试套件
type TestSuite struct {
	tests   []TestCase
	results []*TestResult
}

func NewTestSuite() *TestSuite {
	return &TestSuite{
		tests:   make([]TestCase, 0),
		results: make([]*TestResult, 0),
	}
}

func (s *TestSuite) AddTest(test TestCase) {
	s.tests = append(s.tests, test)
}

func (s *TestSuite) Run() {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("开始运行测试套件")
	fmt.Printf("共 %d 个测试用例\n", len(s.tests))
	fmt.Println(strings.Repeat("=", 60))
	
	startTime := time.Now()
	
	for _, test := range s.tests {
		runner := NewTestRunner(test)
		result := runner.Run()
		s.results = append(s.results, result)
	}
	
	duration := time.Since(startTime)
	
	s.printSummary(duration)
}

func (s *TestSuite) printSummary(duration time.Duration) {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("测试套件执行完成")
	fmt.Println(strings.Repeat("=", 60))
	
	passed := 0
	failed := 0
	
	for _, result := range s.results {
		if result.Passed {
			passed++
		} else {
			failed++
		}
	}
	
	fmt.Printf("\n总计: %d 个测试\n", len(s.results))
	fmt.Printf("通过: %d 个 ✓\n", passed)
	fmt.Printf("失败: %d 个 ✗\n", failed)
	fmt.Printf("总耗时: %v\n", duration)
	
	if failed == 0 {
		fmt.Println("\n🎉 所有测试通过！")
	} else {
		fmt.Println("\n⚠️  部分测试失败，请检查错误信息")
	}
}

func main() {
	fmt.Println("=== 模板方法模式 - 测试框架示例 ===")
	fmt.Println("\n本示例展示了如何使用模板方法模式构建测试框架")
	fmt.Println("模板方法定义了测试执行的标准流程：SetUp → RunTest → TearDown")
	fmt.Println("不同的测试用例只需实现具体的测试步骤")
	
	// 创建测试套件
	suite := NewTestSuite()
	
	// 添加测试用例
	suite.AddTest(NewDatabaseTest())
	suite.AddTest(NewAPITest())
	suite.AddTest(NewFailingTest())
	
	// 运行测试套件
	suite.Run()
	
	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n💡 关键点:")
	fmt.Println("1. TestRunner.Run() 是模板方法，定义了测试执行的算法骨架")
	fmt.Println("2. SetUp、RunTest、TearDown 是原语操作，由具体测试用例实现")
	fmt.Println("3. beforeTest 和 afterTest 是钩子方法，提供了扩展点")
	fmt.Println("4. 即使测试失败，TearDown 也会被执行（通过 defer 实现）")
	fmt.Println("5. 所有测试用例共享相同的执行流程，但实现不同的测试逻辑")
}
