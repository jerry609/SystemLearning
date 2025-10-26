package main

import "time"

// WebAppSpec 定义期望状态
type WebAppSpec struct {
	Image    string            // 容器镜像
	Replicas int32             // 副本数
	Port     int32             // 服务端口
	Env      map[string]string // 环境变量
}

// Condition 表示资源的状态条件
type Condition struct {
	Type               string    // 条件类型，如 "Ready", "Progressing"
	Status             string    // 条件状态: "True", "False", "Unknown"
	LastTransitionTime time.Time // 最后转换时间
	Reason             string    // 原因
	Message            string    // 详细消息
}

// WebAppStatus 定义实际状态
type WebAppStatus struct {
	Phase              string      // 当前阶段: Pending, Creating, Running, Failed, Deleting
	Message            string      // 状态消息
	Reason             string      // 状态原因
	LastReconcileTime  time.Time   // 最后协调时间
	ObservedGeneration int64       // 观察到的 Generation
	Conditions         []Condition // 状态条件
}

// WebApp 完整资源
type WebApp struct {
	Name              string   // 资源名称
	Namespace         string   // 命名空间
	Generation        int64    // 资源版本号
	DeletionTimestamp *time.Time // 删除时间戳（nil 表示未删除）
	Finalizers        []string // Finalizer 列表
	Spec              WebAppSpec
	Status            WebAppStatus
}

// DeploymentSpec 定义 Deployment 的期望状态
type DeploymentSpec struct {
	Replicas int32             // 副本数
	Image    string            // 容器镜像
	Env      map[string]string // 环境变量
}

// DeploymentStatus 定义 Deployment 的实际状态
type DeploymentStatus struct {
	ReadyReplicas int32 // 就绪的副本数
}

// Deployment 表示 Kubernetes Deployment 资源（简化模型）
type Deployment struct {
	Name      string
	Namespace string
	Spec      DeploymentSpec
	Status    DeploymentStatus
	OwnerReferences []OwnerReference // 所有者引用
}

// ServiceSpec 定义 Service 的期望状态
type ServiceSpec struct {
	Port     int32             // 服务端口
	Selector map[string]string // 选择器
}

// Service 表示 Kubernetes Service 资源（简化模型）
type Service struct {
	Name      string
	Namespace string
	Spec      ServiceSpec
	OwnerReferences []OwnerReference // 所有者引用
}

// OwnerReference 表示资源的所有者引用
type OwnerReference struct {
	Name string // 所有者名称
	UID  string // 所有者 UID
}

// Event 表示 Kubernetes Event
type Event struct {
	Type      string    // 事件类型: "Normal", "Warning"
	Reason    string    // 事件原因
	Message   string    // 事件消息
	Timestamp time.Time // 事件时间
	Object    string    // 关联对象（格式: "namespace/name"）
}

// 状态常量
const (
	PhasePending  = "Pending"  // 初始状态，等待处理
	PhaseCreating = "Creating" // 正在创建依赖资源
	PhaseRunning  = "Running"  // 运行中
	PhaseFailed   = "Failed"   // 失败状态
	PhaseDeleting = "Deleting" // 删除中
)

// 事件类型常量
const (
	EventTypeNormal  = "Normal"
	EventTypeWarning = "Warning"
)
