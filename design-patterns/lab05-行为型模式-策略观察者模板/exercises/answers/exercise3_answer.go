package main

import (
	"fmt"
	"strings"
	"time"
)

// 练习 3: 数据导入框架 - 参考答案
//
// 设计思路:
// 1. 定义 DataImporter 接口，包含导入的各个步骤
// 2. 实现 DataImportTemplate 模板类，定义导入流程骨架
// 3. 实现具体导入器（CSV、JSON、XML）
// 4. 使用钩子方法提供扩展点
//
// 使用的设计模式: 模板方法模式
// 模式应用位置: DataImportTemplate.Import() 方法

// 数据记录
type Record map[string]interface{}

// 数据导入器接口
type DataImporter interface {
	ReadFile() ([]byte, error)
	ParseData([]byte) ([]Record, error)
	ValidateData([]Record) ([]Record, []error)
	TransformData([]Record) ([]Record, error)
	
	// 钩子方法
	BeforeImport() error
	AfterImport() error
}

// 基础导入器（提供默认实现）
type BaseImporter struct{}

func (b *BaseImporter) BeforeImport() error {
	return nil
}

func (b *BaseImporter) AfterImport() error {
	return nil
}

// 数据导入模板
type DataImportTemplate struct {
	importer DataImporter
}

func NewDataImportTemplate(importer DataImporter) *DataImportTemplate {
	return &DataImportTemplate{importer: importer}
}

func (t *DataImportTemplate) Import() error {
	fmt.Println("=== 开始数据导入流程 ===\n")
	startTime := time.Now()
	
	// 步骤 0: 前置处理
	fmt.Println("步骤 0: 前置处理")
	if err := t.importer.BeforeImport(); err != nil {
		return fmt.Errorf("前置处理失败: %w", err)
	}
	fmt.Println("✓ 前置处理完成\n")
	
	// 步骤 1: 读取文件
	fmt.Println("步骤 1: 读取文件")
	data, err := t.importer.ReadFile()
	if err != nil {
		return fmt.Errorf("读取文件失败: %w", err)
	}
	fmt.Printf("  读取到 %d 字节数据\n", len(data))
	fmt.Println("✓ 读取完成\n")
	
	// 步骤 2: 解析数据
	fmt.Println("步骤 2: 解析数据")
	records, err := t.importer.ParseData(data)
	if err != nil {
		return fmt.Errorf("解析数据失败: %w", err)
	}
	fmt.Printf("  记录数: %d\n", len(records))
	fmt.Println("✓ 解析完成\n")
	
	// 步骤 3: 验证数据
	fmt.Println("步骤 3: 验证数据")
	validRecords, errors := t.importer.ValidateData(records)
	fmt.Printf("  验证通过: %d/%d\n", len(validRecords), len(records))
	if len(errors) > 0 {
		fmt.Printf("  验证失败: %d 条记录\n", len(errors))
	}
	fmt.Println("✓ 验证完成\n")
	
	// 步骤 4: 转换数据
	fmt.Println("步骤 4: 转换数据")
	transformed, err := t.importer.TransformData(validRecords)
	if err != nil {
		return fmt.Errorf("数据转换失败: %w", err)
	}
	fmt.Printf("  转换 %d 条记录\n", len(transformed))
	fmt.Println("✓ 转换完成\n")
	
	// 步骤 5: 保存数据
	fmt.Println("步骤 5: 保存数据")
	if err := t.saveData(transformed); err != nil {
		return fmt.Errorf("保存数据失败: %w", err)
	}
	fmt.Println("✓ 保存成功\n")
	
	// 步骤 6: 后置处理
	fmt.Println("步骤 6: 后置处理")
	if err := t.importer.AfterImport(); err != nil {
		return fmt.Errorf("后置处理失败: %w", err)
	}
	fmt.Println("✓ 后置处理完成\n")
	
	duration := time.Since(startTime)
	fmt.Printf("=== 数据导入完成 (耗时: %v) ===\n", duration)
	
	// 显示统计信息
	fmt.Println("\n导入统计:")
	fmt.Printf("  总记录数: %d\n", len(records))
	fmt.Printf("  成功: %d\n", len(validRecords))
	fmt.Printf("  失败: %d\n", len(errors))
	if len(records) > 0 {
		fmt.Printf("  成功率: %.0f%%\n", float64(len(validRecords))/float64(len(records))*100)
	}
	
	return nil
}

func (t *DataImportTemplate) saveData(records []Record) error {
	fmt.Println("  保存到数据库...")
	for i := range records {
		fmt.Printf("  插入记录 %d/%d\n", i+1, len(records))
		time.Sleep(50 * time.Millisecond)
	}
	return nil
}

// CSV 导入器
type CSVImporter struct {
	BaseImporter
	filePath string
}

func NewCSVImporter(filePath string) *CSVImporter {
	return &CSVImporter{filePath: filePath}
}

func (c *CSVImporter) BeforeImport() error {
	fmt.Printf("  检查文件是否存在: %s\n", c.filePath)
	fmt.Println("  文件大小: 1024 字节")
	return nil
}

func (c *CSVImporter) ReadFile() ([]byte, error) {
	fmt.Println("  从 CSV 文件读取数据...")
	data := "name,age,email\nAlice,30,alice@example.com\nBob,25,bob@example.com\nCharlie,35,charlie@example.com"
	return []byte(data), nil
}

func (c *CSVImporter) ParseData(data []byte) ([]Record, error) {
	fmt.Println("  解析 CSV 格式...")
	lines := strings.Split(string(data), "\n")
	headers := strings.Split(lines[0], ",")
	
	records := make([]Record, 0)
	for i := 1; i < len(lines); i++ {
		values := strings.Split(lines[i], ",")
		record := make(Record)
		for j, header := range headers {
			record[header] = values[j]
		}
		records = append(records, record)
	}
	
	fmt.Printf("  字段: %s\n", strings.Join(headers, ", "))
	return records, nil
}

func (c *CSVImporter) ValidateData(records []Record) ([]Record, []error) {
	fmt.Println("  验证数据格式...")
	validRecords := make([]Record, 0)
	errors := make([]error, 0)
	
	for i, record := range records {
		fmt.Printf("  验证记录 %d/%d: ✓\n", i+1, len(records))
		validRecords = append(validRecords, record)
	}
	
	return validRecords, errors
}

func (c *CSVImporter) TransformData(records []Record) ([]Record, error) {
	fmt.Println("  转换数据为标准格式...")
	fmt.Println("  应用转换规则: 邮箱小写化")
	
	for _, record := range records {
		if email, ok := record["email"].(string); ok {
			record["email"] = strings.ToLower(email)
		}
		record["imported_at"] = time.Now().Format("2006-01-02 15:04:05")
	}
	
	return records, nil
}

func (c *CSVImporter) AfterImport() error {
	fmt.Println("  生成导入报告...")
	fmt.Println("  清理临时文件...")
	return nil
}

// JSON 导入器
type JSONImporter struct {
	BaseImporter
	apiURL string
}

func NewJSONImporter(apiURL string) *JSONImporter {
	return &JSONImporter{apiURL: apiURL}
}

func (j *JSONImporter) BeforeImport() error {
	fmt.Println("  检查 API 连接...")
	fmt.Printf("  API URL: %s\n", j.apiURL)
	return nil
}

func (j *JSONImporter) ReadFile() ([]byte, error) {
	fmt.Println("  从 API 获取 JSON 数据...")
	data := `[{"name":"Alice","age":30,"email":"alice@example.com"},{"name":"Bob","age":25,"email":"bob@example.com"}]`
	return []byte(data), nil
}

func (j *JSONImporter) ParseData(data []byte) ([]Record, error) {
	fmt.Println("  解析 JSON 格式...")
	// 简化的 JSON 解析
	records := []Record{
		{"name": "Alice", "age": "30", "email": "alice@example.com"},
		{"name": "Bob", "age": "25", "email": "bob@example.com"},
	}
	return records, nil
}

func (j *JSONImporter) ValidateData(records []Record) ([]Record, []error) {
	fmt.Println("  验证数据格式...")
	validRecords := make([]Record, 0)
	errors := make([]error, 0)
	
	for i, record := range records {
		fmt.Printf("  验证记录 %d/%d: ✓\n", i+1, len(records))
		validRecords = append(validRecords, record)
	}
	
	return validRecords, errors
}

func (j *JSONImporter) TransformData(records []Record) ([]Record, error) {
	fmt.Println("  转换数据为标准格式...")
	for _, record := range records {
		record["imported_at"] = time.Now().Format("2006-01-02 15:04:05")
	}
	return records, nil
}

func (j *JSONImporter) AfterImport() error {
	fmt.Println("  发送导入完成通知...")
	return nil
}

func main() {
	fmt.Println("=== 数据导入框架 ===")
	
	// 场景 1: CSV 数据导入
	fmt.Println("\n【场景 1: CSV 数据导入】\n")
	csvImporter := NewCSVImporter("users.csv")
	csvTemplate := NewDataImportTemplate(csvImporter)
	if err := csvTemplate.Import(); err != nil {
		fmt.Printf("导入失败: %v\n", err)
	}
	
	fmt.Println("\n" + strings.Repeat("=", 60))
	
	// 场景 2: JSON 数据导入
	fmt.Println("\n【场景 2: JSON 数据导入】\n")
	jsonImporter := NewJSONImporter("https://api.example.com/users")
	jsonTemplate := NewDataImportTemplate(jsonImporter)
	if err := jsonTemplate.Import(); err != nil {
		fmt.Printf("导入失败: %v\n", err)
	}
	
	fmt.Println("\n=== 示例结束 ===")
}
