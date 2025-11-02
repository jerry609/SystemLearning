---
inclusion: manual
---
# Docker 学习模块指南

Docker是本项目的已完成基础模块，涵盖容器化技术的核心概念和实践应用。

## 模块结构

主入口文档: [docker/README.md](mdc:docker/README.md)

## 实验室序列 (按学习顺序)

### 基础入门
1. **lab01-基础概念与安装**: [docker/lab01-基础概念与安装/README.md](mdc:docker/lab01-基础概念与安装/README.md)
   - Docker架构、安装配置、基本概念
2. **lab02-镜像操作与管理**: [docker/lab02-镜像操作与管理/README.md](mdc:docker/lab02-镜像操作与管理/README.md)
   - 镜像拉取、构建、推送、管理

### 容器操作
3. **lab03-容器生命周期管理**: [docker/lab03-容器生命周期管理/README.md](mdc:docker/lab03-容器生命周期管理/README.md)
   - 容器创建、启动、停止、删除、监控
4. **lab04-Dockerfile详解与最佳实践**: [docker/lab04-Dockerfile详解与最佳实践/README.md](mdc:docker/lab04-Dockerfile详解与最佳实践/README.md)
   - Dockerfile语法、指令详解、构建优化

### 高级特性
5. **lab05-多阶段构建与镜像优化**: [docker/lab05-多阶段构建与镜像优化/README.md](mdc:docker/lab05-多阶段构建与镜像优化/README.md)
   - 多阶段构建、镜像层优化、体积缩减
6. **lab06-数据卷与持久化存储**: [docker/lab06-数据卷与持久化存储/README.md](mdc:docker/lab06-数据卷与持久化存储/README.md)
   - Volume管理、数据持久化、备份恢复

### 网络与编排
7. **lab07-Docker网络模式与配置**: [docker/lab07-Docker网络模式与配置/README.md](mdc:docker/lab07-Docker网络模式与配置/README.md)
   - 网络模式、自定义网络、服务发现
8. **lab08-Docker Compose多服务编排**: [docker/lab08-Docker Compose多服务编排/README.md](mdc:docker/lab08-Docker Compose多服务编排/README.md)
   - 多容器应用、服务编排、环境管理

### 集群与生产
9. **lab09-Docker Swarm集群管理**: [docker/lab09-Docker Swarm集群管理/README.md](mdc:docker/lab09-Docker Swarm集群管理/README.md)
   - 集群模式、服务管理、负载均衡
10. **lab10-生产环境部署与监控**: [docker/lab10-生产环境部署与监控/README.md](mdc:docker/lab10-生产环境部署与监控/README.md)
    - 安全配置、监控方案、日志管理

## Docker 最佳实践

### Dockerfile 优化
- 使用多阶段构建减少镜像体积
- 合并RUN指令减少镜像层数
- 使用.dockerignore排除不必要文件
- 选择合适的基础镜像(Alpine vs Ubuntu)

### 安全考虑
- 使用非root用户运行应用
- 定期更新基础镜像修复安全漏洞
- 限制容器资源使用
- 使用secrets管理敏感信息

### 生产环境配置
- 设置合理的资源限制
- 配置健康检查
- 使用标签进行版本管理
- 实施容器监控和日志收集

### 性能优化
- 优化镜像构建缓存
- 使用多阶段构建
- 配置适当的重启策略
- 合理规划存储驱动

## 与Kubernetes的关系

Docker作为容器运行时，是Kubernetes的基础技术栈：
- Docker镜像是Kubernetes Pod的基础
- Docker网络概念延伸到Kubernetes网络模型
- Docker Compose的服务编排思想在Kubernetes中得到扩展
- 容器安全和资源管理在Kubernetes中更加完善

建议学习顺序: Docker基础 → Docker进阶 → Kubernetes入门

