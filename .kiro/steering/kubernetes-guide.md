---
inclusion: manual
---
# Kubernetes 学习模块指南

Kubernetes是本项目的已完成模块，包含完整的从入门到生产环境的学习体系。

## 模块结构

主入口文档: [kubernetes/README.md](mdc:kubernetes/README.md)
学习总结: [kubernetes/项目完成总结.md](mdc:kubernetes/项目完成总结.md)

## 实验室序列 (按学习顺序)

### 基础阶段
1. **lab01-k8s基础概念与环境搭建**: [kubernetes/lab01-k8s基础概念与环境搭建/README.md](mdc:kubernetes/lab01-k8s基础概念与环境搭建/README.md)
   - 集群架构、组件介绍、环境搭建
2. **lab02-Pod的创建与管理**: [kubernetes/lab02-Pod的创建与管理/README.md](mdc:kubernetes/lab02-Pod的创建与管理/README.md)
   - Pod生命周期、资源管理、调试技巧

### 核心功能
3. **lab03-Service服务发现与负载均衡**: [kubernetes/lab03-Service服务发现与负载均衡/README.md](mdc:kubernetes/lab03-Service服务发现与负载均衡/README.md)
   - Service类型、Endpoint管理、Ingress控制器
4. **lab04-Deployment应用部署与更新**: [kubernetes/lab04-Deployment应用部署与更新/README.md](mdc:kubernetes/lab04-Deployment应用部署与更新/README.md)
   - 滚动更新、回滚策略、蓝绿部署

### 配置与存储
5. **lab05-ConfigMap和Secret配置管理**: [kubernetes/lab05-ConfigMap和Secret配置管理/README.md](mdc:kubernetes/lab05-ConfigMap和Secret配置管理/README.md)
   - 配置分离、敏感信息管理、热更新
6. **lab06-数据持久化与存储**: [kubernetes/lab06-数据持久化与存储/README.md](mdc:kubernetes/lab06-数据持久化与存储/README.md)
   - PV/PVC、StorageClass、CSI驱动

### 安全与网络
7. **lab07-网络策略与安全**: [kubernetes/lab07-网络策略与安全/README.md](mdc:kubernetes/lab07-网络策略与安全/README.md)
   - RBAC、Pod安全策略、网络隔离

### 运维监控
8. **lab08-监控与日志管理**: [kubernetes/lab08-监控与日志管理/README.md](mdc:kubernetes/lab08-监控与日志管理/README.md)
   - Prometheus监控、ELK日志栈、链路追踪
9. **lab09-自动扩缩容HPA-VPA**: [kubernetes/lab09-自动扩缩容HPA-VPA/README.md](mdc:kubernetes/lab09-自动扩缩容HPA-VPA/README.md)
   - 水平扩缩容、垂直扩缩容、自定义指标

### 生产实践
10. **lab10-生产环境最佳实践**: [kubernetes/lab10-生产环境最佳实践/README.md](mdc:kubernetes/lab10-生产环境最佳实践/README.md)
    - 高可用部署、灾备策略、性能优化

## Kubernetes 最佳实践

### YAML 配置规范
- 使用声明式配置，避免命令式操作
- 合理设置资源限制和请求
- 使用命名空间进行环境隔离
- 添加标签和注解便于管理

### 安全考虑
- 最小权限原则，使用RBAC
- 定期轮换Secret中的敏感信息
- 启用Pod安全策略或Pod安全标准
- 网络策略实现微分段

### 生产环境部署
- 多副本部署确保高可用
- 使用健康检查和就绪探针
- 配置资源配额防止资源滥用
- 实施监控和告警策略

## 认证考试准备

本模块内容涵盖了以下Kubernetes认证的核心知识点:
- **CKA (Certified Kubernetes Administrator)**: 集群管理、故障排除
- **CKAD (Certified Kubernetes Application Developer)**: 应用部署、配置管理
- **CKS (Certified Kubernetes Security Specialist)**: 安全配置、合规检查

