# 实验1：Kubernetes基础概念与环境搭建

## 🎯 学习目标

通过本实验，你将：
- 理解Kubernetes的核心概念和架构
- 在macOS上搭建多种K8s学习环境
- 掌握kubectl基本命令
- 部署你的第一个应用到K8s集群
- 了解K8s与Docker的关系和区别

## 📚 理论知识学习

### 什么是Kubernetes？

Kubernetes（简称K8s）是一个开源的容器编排平台，用于自动化容器化应用的部署、扩缩容和管理。

**核心价值**：
- **自动化部署**：自动将容器调度到合适的节点
- **自愈能力**：自动重启失败的容器，替换节点
- **水平扩缩容**：根据负载自动增减容器数量
- **服务发现**：自动为服务分配网络和存储
- **滚动更新**：零停机时间的应用更新

### Kubernetes vs Docker

| 特性 | Docker | Kubernetes |
|------|--------|------------|
| **主要功能** | 容器化平台 | 容器编排平台 |
| **作用范围** | 单机容器管理 | 集群级容器管理 |
| **网络** | 简单的桥接网络 | 复杂的集群网络 |
| **存储** | 数据卷管理 | 持久化存储抽象 |
| **高可用** | 不支持 | 内建支持 |
| **扩容** | 手动 | 自动化 |
| **使用场景** | 开发测试 | 生产环境 |

### Kubernetes核心概念

#### 1. 集群架构
```
┌─────────────────────────────────────────┐
│              Kubernetes集群              │
├─────────────────────┬───────────────────┤
│     控制平面         │      工作节点        │
│   (Control Plane)   │   (Worker Node)   │
├─────────────────────┼───────────────────┤
│ • kube-apiserver   │ • kubelet         │
│ • etcd             │ • kube-proxy      │
│ • kube-scheduler   │ • Container       │
│ • kube-controller  │   Runtime         │
└─────────────────────┴───────────────────┘
```

#### 2. 核心对象

| 对象 | 用途 | 类比 |
|------|------|------|
| **Pod** | 最小部署单元，包含一个或多个容器 | 一台虚拟机 |
| **Service** | 为Pod提供稳定的网络访问 | 负载均衡器 |
| **Deployment** | 管理Pod的副本和更新 | 应用管理器 |
| **ConfigMap** | 存储配置信息 | 配置文件 |
| **Secret** | 存储敏感信息 | 密码保险箱 |
| **Namespace** | 逻辑隔离环境 | 项目文件夹 |

## 🔧 环境搭建实践

### 方案一：Docker Desktop内置Kubernetes（推荐入门）

#### 步骤1：启用Kubernetes
1. 打开Docker Desktop
2. 进入Settings（设置）
3. 选择Kubernetes选项卡
4. 勾选"Enable Kubernetes"
5. 点击"Apply & Restart"

#### 步骤2：验证安装
```bash
# 检查集群信息
kubectl cluster-info

# 查看节点状态
kubectl get nodes

# 查看当前上下文
kubectl config current-context
```

**预期输出**：
```
Kubernetes control plane is running at https://kubernetes.docker.internal:6443
CoreDNS is running at https://kubernetes.docker.internal:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

NAME             STATUS   ROLES           AGE   VERSION
docker-desktop   Ready    control-plane   1m    v1.30.2

docker-desktop
```

### 方案二：Minikube（推荐深入学习）

#### 步骤1：安装Minikube
```bash
# 使用Homebrew安装
brew install minikube

# 验证安装
minikube version
```

#### 步骤2：启动Minikube集群
```bash
# 启动集群（使用Docker驱动）
minikube start --driver=docker --memory=4096 --cpus=2

# 查看集群状态
minikube status

# 获取集群信息
kubectl cluster-info
```

#### 步骤3：Minikube常用命令
```bash
# 停止集群
minikube stop

# 删除集群
minikube delete

# 打开Kubernetes Dashboard
minikube dashboard

# 查看Minikube IP
minikube ip

# SSH到Minikube虚拟机
minikube ssh
```

### 方案三：Kind（轻量级方案）

#### 步骤1：安装Kind
```bash
# 安装Kind
brew install kind

# 验证安装
kind version
```

#### 步骤2：创建集群
```bash
# 创建默认集群
kind create cluster

# 创建带自定义名称的集群
kind create cluster --name learning-cluster

# 查看集群列表
kind get clusters

# 删除集群
kind delete cluster --name learning-cluster
```

## 🚀 第一个Kubernetes应用

### 实验任务1：部署Nginx应用

#### 步骤1：创建Pod
```bash
# 创建一个Nginx Pod
kubectl run nginx-pod --image=nginx:latest --port=80

# 查看Pod状态
kubectl get pods

# 查看Pod详细信息
kubectl describe pod nginx-pod
```

#### 步骤2：访问应用
```bash
# 方法1：使用端口转发
kubectl port-forward nginx-pod 8080:80

# 在浏览器中访问 http://localhost:8080
```

#### 步骤3：创建Service
```bash
# 创建Service暴露Pod
kubectl expose pod nginx-pod --type=NodePort --name=nginx-service

# 查看Service
kubectl get services

# 获取访问URL（Minikube）
minikube service nginx-service --url
```

### 实验任务2：使用Deployment管理应用

#### 步骤1：创建Deployment YAML文件
```bash
# 创建目录
mkdir -p ~/k8s-labs/lab01

# 创建nginx-deployment.yaml文件
cat > ~/k8s-labs/lab01/nginx-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.20
        ports:
        - containerPort: 80
EOF
```

#### 步骤2：部署应用
```bash
# 应用配置
kubectl apply -f ~/k8s-labs/lab01/nginx-deployment.yaml

# 查看Deployment
kubectl get deployments

# 查看Pod
kubectl get pods -l app=nginx

# 查看ReplicaSet
kubectl get rs
```

#### 步骤3：应用扩缩容
```bash
# 扩容到5个副本
kubectl scale deployment nginx-deployment --replicas=5

# 查看扩容过程
kubectl get pods -w

# 缩容到2个副本
kubectl scale deployment nginx-deployment --replicas=2
```

## 🔍 kubectl命令详解

### 基础命令结构
```bash
kubectl [command] [TYPE] [NAME] [flags]
```

### 常用命令分类

#### 1. 查看资源
```bash
# 查看所有Pod
kubectl get pods

# 查看所有资源
kubectl get all

# 查看特定命名空间的资源
kubectl get pods -n kube-system

# 以YAML格式输出
kubectl get pod nginx-pod -o yaml

# 以JSON格式输出
kubectl get pod nginx-pod -o json

# 查看资源标签
kubectl get pods --show-labels
```

#### 2. 详细信息
```bash
# 查看Pod详细信息
kubectl describe pod nginx-pod

# 查看事件
kubectl get events

# 查看日志
kubectl logs nginx-pod

# 实时查看日志
kubectl logs -f nginx-pod
```

#### 3. 操作资源
```bash
# 创建资源
kubectl create -f deployment.yaml

# 应用配置（创建或更新）
kubectl apply -f deployment.yaml

# 删除资源
kubectl delete pod nginx-pod
kubectl delete -f deployment.yaml

# 编辑资源
kubectl edit deployment nginx-deployment
```

#### 4. 交互操作
```bash
# 进入Pod
kubectl exec -it nginx-pod -- /bin/bash

# 端口转发
kubectl port-forward pod/nginx-pod 8080:80

# 复制文件
kubectl cp localfile nginx-pod:/path/to/remotefile
```

## 🛠️ 实验练习

### 练习1：探索集群信息
```bash
# 1. 查看集群基本信息
kubectl cluster-info

# 2. 查看集群节点详细信息
kubectl describe nodes

# 3. 查看kube-system命名空间的Pod
kubectl get pods -n kube-system

# 4. 查看集群中的所有命名空间
kubectl get namespaces
```

### 练习2：部署Web应用栈
创建一个包含前端和后端的简单Web应用：

```yaml
# ~/k8s-labs/lab01/web-stack.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: nginx
        image: nginx:alpine
        ports:
        - containerPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
spec:
  selector:
    app: frontend
  ports:
  - port: 80
    targetPort: 80
  type: NodePort
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: api
        image: httpd:alpine
        ports:
        - containerPort: 80
```

部署和测试：
```bash
# 部署应用栈
kubectl apply -f ~/k8s-labs/lab01/web-stack.yaml

# 查看所有资源
kubectl get all

# 测试前端服务
kubectl port-forward service/frontend-service 8080:80
```

### 练习3：资源监控
```bash
# 1. 监控Pod资源使用（需要metrics-server）
kubectl top pods

# 2. 实时查看Pod状态变化
kubectl get pods -w

# 3. 查看Pod重启情况
kubectl get pods -o wide

# 4. 分析Pod日志
kubectl logs deployment/nginx-deployment --all-containers=true
```

## 🧪 进阶实验

### 实验：多环境对比测试

在不同K8s环境中部署相同应用，对比性能和功能差异：

```bash
# 1. 在Docker Desktop K8s中部署
kubectl config use-context docker-desktop
kubectl apply -f ~/k8s-labs/lab01/nginx-deployment.yaml

# 2. 在Minikube中部署
kubectl config use-context minikube
kubectl apply -f ~/k8s-labs/lab01/nginx-deployment.yaml

# 3. 比较两个环境的差异
kubectl get nodes -o wide
kubectl get pods -o wide
```

## 🐛 故障排查

### 常见问题及解决方案

#### 问题1：Pod一直处于Pending状态
```bash
# 检查Pod事件
kubectl describe pod <pod-name>

# 可能原因：
# - 资源不足
# - 镜像拉取失败
# - 节点选择器不匹配
```

#### 问题2：无法访问Service
```bash
# 检查Service配置
kubectl describe service <service-name>

# 检查端点
kubectl get endpoints <service-name>

# 检查标签选择器
kubectl get pods --show-labels
```

#### 问题3：Minikube启动失败
```bash
# 查看详细错误信息
minikube start --v=7

# 重置Minikube
minikube delete
minikube start --driver=docker
```

## 💡 最佳实践

### 1. 资源命名规范
- 使用有意义的名称
- 遵循DNS命名规范（小写字母、数字、连字符）
- 包含环境信息（dev、staging、prod）

### 2. 标签和注解
```yaml
metadata:
  labels:
    app: nginx
    version: v1.0
    environment: dev
  annotations:
    description: "Frontend web server"
    maintainer: "jerry@example.com"
```

### 3. 资源管理
- 设置资源请求和限制
- 使用命名空间隔离不同环境
- 定期清理不需要的资源

## 📝 学习检查

完成本实验后，你应该能够回答：

1. **概念理解**：
   - Kubernetes的主要组件有哪些？
   - Pod和容器的关系是什么？
   - Service的作用是什么？

2. **操作技能**：
   - 如何查看集群中的所有Pod？
   - 如何将应用扩容到5个副本？
   - 如何查看Pod的日志？

3. **实际应用**：
   - 在不同K8s环境中部署应用有什么区别？
   - 如何排查Pod无法启动的问题？

## 🔗 延伸学习

- 阅读Kubernetes官方概念文档
- 了解容器编排的发展历史
- 探索其他容器编排工具（Docker Swarm、Nomad）
- 学习云原生计算基金会（CNCF）生态

## ⏭️ 下一步

完成本实验后，继续学习：
- **实验2**：Pod的创建与管理 - 深入了解K8s最小部署单元
- 探索Pod的生命周期、健康检查、资源管理等高级特性

---

**恭喜完成第一个Kubernetes实验！** 🎉 
现在你已经具备了K8s的基础知识，可以继续探索更深入的内容。 