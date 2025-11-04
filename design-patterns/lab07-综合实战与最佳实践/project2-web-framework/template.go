package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// 模板渲染 - 模板方法模式
//
// 本模块使用模板方法模式定义模板渲染的算法骨架
// 具体的渲染细节由子类实现

// TemplateEngine 模板引擎接口
type TemplateEngine interface {
	Render(name string, data interface{}) (string, error)
}

// BaseTemplateEngine 基础模板引擎（抽象类）
type BaseTemplateEngine struct {
	templates map[string]string
}

func NewBaseTemplateEngine() *BaseTemplateEngine {
	return &BaseTemplateEngine{
		templates: make(map[string]string),
	}
}

// LoadTemplate 加载模板
func (e *BaseTemplateEngine) LoadTemplate(name string, content string) {
	e.templates[name] = content
}

// GetTemplate 获取模板
func (e *BaseTemplateEngine) GetTemplate(name string) (string, error) {
	template, ok := e.templates[name]
	if !ok {
		return "", fmt.Errorf("模板未找到: %s", name)
	}
	return template, nil
}

// HTMLTemplateEngine HTML 模板引擎
type HTMLTemplateEngine struct {
	*BaseTemplateEngine
}

func NewHTMLTemplateEngine() *HTMLTemplateEngine {
	return &HTMLTemplateEngine{
		BaseTemplateEngine: NewBaseTemplateEngine(),
	}
}

// Render 渲染 HTML 模板
func (e *HTMLTemplateEngine) Render(name string, data interface{}) (string, error) {
	// 1. 获取模板
	template, err := e.GetTemplate(name)
	if err != nil {
		return "", err
	}

	// 2. 解析数据
	dataMap, ok := data.(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("数据格式错误")
	}

	// 3. 替换变量
	result := template
	for key, value := range dataMap {
		placeholder := fmt.Sprintf("{{%s}}", key)
		result = strings.ReplaceAll(result, placeholder, fmt.Sprintf("%v", value))
	}

	// 4. 处理列表
	if items, ok := dataMap["items"].([]string); ok {
		listHTML := ""
		for _, item := range items {
			listHTML += fmt.Sprintf("<li>%s</li>\n", item)
		}
		result = strings.ReplaceAll(result, "{{items}}", listHTML)
	}

	return result, nil
}

// JSONTemplateEngine JSON 模板引擎
type JSONTemplateEngine struct {
	*BaseTemplateEngine
}

func NewJSONTemplateEngine() *JSONTemplateEngine {
	return &JSONTemplateEngine{
		BaseTemplateEngine: NewBaseTemplateEngine(),
	}
}

// Render 渲染 JSON 模板
func (e *JSONTemplateEngine) Render(name string, data interface{}) (string, error) {
	// 1. 获取模板（JSON 模板可能不需要）
	_, err := e.GetTemplate(name)
	if err != nil {
		// JSON 渲染可以不依赖模板
	}

	// 2. 序列化数据
	jsonData, err := json.Marshal(data)
	if err != nil {
		return "", fmt.Errorf("JSON 序列化失败: %v", err)
	}

	return string(jsonData), nil
}

// XMLTemplateEngine XML 模板引擎
type XMLTemplateEngine struct {
	*BaseTemplateEngine
}

func NewXMLTemplateEngine() *XMLTemplateEngine {
	return &XMLTemplateEngine{
		BaseTemplateEngine: NewBaseTemplateEngine(),
	}
}

// Render 渲染 XML 模板
func (e *XMLTemplateEngine) Render(name string, data interface{}) (string, error) {
	// 1. 获取模板
	template, err := e.GetTemplate(name)
	if err != nil {
		return "", err
	}

	// 2. 解析数据
	dataMap, ok := data.(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("数据格式错误")
	}

	// 3. 替换变量
	result := template
	for key, value := range dataMap {
		placeholder := fmt.Sprintf("{{%s}}", key)
		result = strings.ReplaceAll(result, placeholder, fmt.Sprintf("%v", value))
	}

	return result, nil
}

// TemplateRenderer 模板渲染器（使用模板方法模式）
type TemplateRenderer struct {
	engine TemplateEngine
}

func NewTemplateRenderer(engine TemplateEngine) *TemplateRenderer {
	return &TemplateRenderer{
		engine: engine,
	}
}

// RenderResponse 渲染响应（模板方法）
func (r *TemplateRenderer) RenderResponse(name string, data interface{}) (string, error) {
	// 1. 预处理数据
	processedData := r.preprocessData(data)

	// 2. 渲染模板
	result, err := r.engine.Render(name, processedData)
	if err != nil {
		return "", err
	}

	// 3. 后处理结果
	finalResult := r.postprocessResult(result)

	return finalResult, nil
}

// preprocessData 预处理数据（钩子方法）
func (r *TemplateRenderer) preprocessData(data interface{}) interface{} {
	// 可以在这里添加通用的数据预处理逻辑
	return data
}

// postprocessResult 后处理结果（钩子方法）
func (r *TemplateRenderer) postprocessResult(result string) string {
	// 可以在这里添加通用的结果后处理逻辑
	// 例如：压缩、格式化等
	return result
}

// TemplateFactory 模板引擎工厂
type TemplateFactory struct{}

func (f *TemplateFactory) CreateEngine(engineType string) TemplateEngine {
	switch engineType {
	case "html":
		return NewHTMLTemplateEngine()
	case "json":
		return NewJSONTemplateEngine()
	case "xml":
		return NewXMLTemplateEngine()
	default:
		return NewHTMLTemplateEngine()
	}
}
