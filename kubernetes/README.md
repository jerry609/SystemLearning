# Kubernetes 学习体系 - 由浅入深实验指南

## 🎯 学习目标

本学习体系专为想要系统掌握Kubernetes的开发者设计，通过循序渐进的实验和项目，帮助您从零基础到熟练运用Kubernetes进行容器编排和管理。

## 💻 环境要求

### 硬件配置
- **推荐配置**：Apple M3芯片，16GB内存
- **最低配置**：8GB内存，50GB可用存储空间
- **操作系统**：macOS 14.6 或更高版本

### 必需工具

#### 已确认安装
- ✅ Docker v27.3.1
- ✅ kubectl v1.30.5  
- ✅ Homebrew v4.4.24

#### 需要安装
```bash
# 安装 Minikube
brew install minikube

# 安装 Helm
brew install helm

# 安装 Kind
brew install kind

# 安装 k9s（Kubernetes CLI管理工具）
brew install k9s

# 安装 kubectx/kubens（上下文切换工具）
brew install kubectx
```

## 🚀 快速环境搭建

### 方案1：Docker Desktop内置Kubernetes（推荐新手）
```bash
# 在Docker Desktop中启用Kubernetes
# 设置 -> Kubernetes -> Enable Kubernetes
```

### 方案2：Minikube（推荐学习）
```bash
# 启动Minikube
minikube start --driver=docker --memory=4096 --cpus=2

# 验证安装
kubectl cluster-info
```

### 方案3：Kind（推荐CI/CD）
```bash
# 创建集群
kind create cluster --name learning-cluster

# 验证
kubectl get nodes
```

## 📚 学习路径

### 阶段一：Kubernetes基础 (第1-3周)
| 实验 | 主题 | 时间 | 重点内容 |
|------|------|------|----------|
| [Lab01](./lab01-k8s基础概念与环境搭建/) | 基础概念与环境搭建 | 2天 | 集群架构、核心概念、环境配置 |
| [Lab02](./lab02-Pod的创建与管理/) | Pod创建与管理 | 2天 | Pod生命周期、容器管理 |
| [Lab03](./lab03-Service服务发现与负载均衡/) | Service与网络 | 3天 | 服务发现、负载均衡、网络模型 |

### 阶段二：应用部署与管理 (第4-6周)
| 实验 | 主题 | 时间 | 重点内容 |
|------|------|------|----------|
| [Lab04](./lab04-Deployment应用部署与更新/) | 应用部署与更新 | 3天 | 滚动更新、版本管理、扩缩容 |
| [Lab05](./lab05-ConfigMap和Secret配置管理/) | 配置管理 | 2天 | 配置分离、密钥管理 |
| [Lab06](./lab06-数据持久化与存储/) | 数据持久化 | 3天 | 存储卷、有状态应用 |

### 阶段三：高级特性与生产实践 (第7-10周)
| 实验 | 主题 | 时间 | 重点内容 |
|------|------|------|----------|
| [Lab07](./lab07-网络策略与安全/) | 网络与安全 | 4天 | RBAC、网络策略、安全加固 |
| [Lab08](./lab08-监控与日志管理/) | 监控与日志 | 4天 | Prometheus、Grafana、日志收集 |
| [Lab09](./lab09-自动扩缩容HPA-VPA/) | 自动扩缩容 | 3天 | HPA、VPA、资源优化 |
| [Lab10](./lab10-生产环境最佳实践/) | 生产最佳实践 | 5天 | CI/CD、高可用、性能调优 |

## 🛠️ 学习方法建议

### 多工具组合学习
1. **理论学习** + **动手实践** + **项目实战**
2. **本地环境** + **云环境**练习
3. **单集群** + **多集群**管理

### 实践建议
- 每个实验都要亲自动手完成
- 记录遇到的问题和解决方案
- 尝试修改实验参数，观察不同效果
- 建立自己的知识笔记和命令备忘录

### 进阶学习
- 参与开源项目
- 学习云原生生态工具
- 准备CKA/CKAD认证

## 📖 学习资源

### 官方文档
- [Kubernetes官方文档](https://kubernetes.io/docs/)
- [Kubernetes中文文档](https://kubernetes.io/zh-cn/docs/)

### 推荐书籍
- 《Kubernetes权威指南》
- 《Kubernetes in Action》
- 《云原生应用架构指南》

### 在线资源
- [Kubernetes官方教程](https://kubernetes.io/docs/tutorials/)
- [Play with Kubernetes](https://labs.play-with-k8s.com/)
- [Katacoda Kubernetes课程](https://www.katacoda.com/courses/kubernetes)

## 🆘 获取帮助

### 常用命令速查
```bash
# 集群信息
kubectl cluster-info
kubectl get nodes

# 查看所有资源
kubectl get all

# 描述资源详情
kubectl describe <resource-type> <resource-name>

# 查看日志
kubectl logs <pod-name>

# 进入容器
kubectl exec -it <pod-name> -- /bin/bash
```

### 故障排查
1. 检查Pod状态：`kubectl get pods`
2. 查看事件：`kubectl get events`
3. 检查日志：`kubectl logs`
4. 描述资源：`kubectl describe`

## 🎉 开始学习

现在就开始您的Kubernetes学习之旅吧！建议从[Lab01 - Kubernetes基础概念与环境搭建](./lab01-k8s基础概念与环境搭建/)开始。

祝您学习愉快！🚀 