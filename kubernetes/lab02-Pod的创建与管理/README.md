# 实验2：Pod的创建与管理

## 🎯 学习目标

通过本实验，你将：
- 深入理解Pod的概念和结构
- 掌握Pod的创建、查看、调试和删除
- 学习Pod的生命周期和状态管理
- 实践Pod的健康检查和资源管理
- 了解多容器Pod的设计模式
- 掌握Pod故障排查的技巧

## 📚 理论知识学习

### Pod核心概念

Pod是Kubernetes中**最小的可部署和管理单元**，它包含一个或多个紧密相关的容器。

#### Pod的特点
- **共享网络**：Pod内的所有容器共享同一个IP地址和端口空间
- **共享存储**：可以挂载相同的Volume
- **共同调度**：Pod内的容器总是被调度到同一个节点
- **生命周期绑定**：容器与Pod共存亡

#### Pod vs 容器的关系
```
┌─────────────── Pod ───────────────┐
│  ┌─────────────┐  ┌─────────────┐ │
│  │  Container  │  │  Container  │ │
│  │     A       │  │     B       │ │
│  └─────────────┘  └─────────────┘ │
│                                   │
│  共享的网络命名空间和存储卷          │
└───────────────────────────────────┘
```

### Pod生命周期

#### 1. Pod状态（Phase）
| 状态 | 描述 | 条件 |
|------|------|------|
| **Pending** | 等待调度 | Pod已被创建，但容器还未启动 |
| **Running** | 运行中 | 至少有一个容器正在运行 |
| **Succeeded** | 成功完成 | 所有容器都成功终止 |
| **Failed** | 失败 | 所有容器都已终止，至少一个失败 |
| **Unknown** | 未知状态 | 无法获取Pod状态 |

#### 2. 容器状态（State）
| 状态 | 描述 |
|------|------|
| **Waiting** | 等待启动（拉取镜像、等待依赖等） |
| **Running** | 正常运行 |
| **Terminated** | 已终止（成功或失败） |

#### 3. Pod生命周期流程图
```
创建Pod → 调度到节点 → 拉取镜像 → 启动容器 → 运行 → 终止
   ↓         ↓         ↓         ↓        ↓      ↓
Pending → Pending → Pending → Running → Running → Succeeded/Failed
```

## 🔧 Pod创建与基本操作

### 方法一：命令行创建Pod

#### 1. 快速创建Pod
```bash
# 创建一个简单的Nginx Pod
kubectl run nginx-pod --image=nginx:1.20 --port=80

# 创建并立即查看
kubectl run test-pod --image=busybox --restart=Never -- sleep 3600
```

#### 2. 带有标签的Pod
```bash
# 创建带标签的Pod
kubectl run labeled-pod --image=nginx:alpine \
  --labels="app=web,version=v1,env=dev"

# 查看标签
kubectl get pods --show-labels
```

#### 3. 交互式Pod
```bash
# 创建可交互的Pod
kubectl run interactive-pod --image=ubuntu:20.04 \
  --restart=Never -it -- /bin/bash

# 创建临时调试Pod
kubectl run debug-pod --image=busybox \
  --restart=Never --rm -it -- sh
```

### 方法二：YAML文件创建Pod

#### 1. 基本Pod定义
```bash
# 创建实验目录
mkdir -p ~/k8s-labs/lab02

# 创建基本Pod配置
cat > ~/k8s-labs/lab02/basic-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: basic-pod
  labels:
    app: web
    version: v1
  annotations:
    description: "基础Pod示例"
    created-by: "jerry"
spec:
  containers:
  - name: nginx
    image: nginx:1.20-alpine
    ports:
    - containerPort: 80
      name: http
    env:
    - name: ENV_VAR
      value: "development"
  restartPolicy: Always
EOF
```

#### 2. 应用配置
```bash
# 应用Pod配置
kubectl apply -f ~/k8s-labs/lab02/basic-pod.yaml

# 查看Pod详情
kubectl describe pod basic-pod

# 查看Pod日志
kubectl logs basic-pod
```

#### 3. 多容器Pod示例
```bash
cat > ~/k8s-labs/lab02/multi-container-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: multi-container-pod
  labels:
    app: multi-app
spec:
  containers:
  # 主容器：Web服务器
  - name: web-server
    image: nginx:alpine
    ports:
    - containerPort: 80
    volumeMounts:
    - name: shared-data
      mountPath: /usr/share/nginx/html
  
  # 辅助容器：内容生成器
  - name: content-generator
    image: busybox
    command: ["/bin/sh"]
    args:
    - -c
    - |
      while true; do
        echo "<h1>Hello from $(hostname)</h1>" > /data/index.html
        echo "<p>Current time: $(date)</p>" >> /data/index.html
        sleep 30
      done
    volumeMounts:
    - name: shared-data
      mountPath: /data
  
  volumes:
  - name: shared-data
    emptyDir: {}
EOF
```

#### 4. 部署和测试多容器Pod
```bash
# 部署多容器Pod
kubectl apply -f ~/k8s-labs/lab02/multi-container-pod.yaml

# 查看Pod状态
kubectl get pods multi-container-pod

# 查看特定容器的日志
kubectl logs multi-container-pod -c web-server
kubectl logs multi-container-pod -c content-generator

# 测试Web服务
kubectl port-forward multi-container-pod 8080:80
# 在浏览器中访问 http://localhost:8080
```

## 🔍 Pod查看与调试

### 1. 查看Pod信息
```bash
# 查看所有Pod
kubectl get pods

# 查看详细信息
kubectl get pods -o wide

# 查看特定Pod
kubectl describe pod basic-pod

# 以YAML格式查看
kubectl get pod basic-pod -o yaml

# 查看Pod事件
kubectl get events --field-selector involvedObject.name=basic-pod
```

### 2. Pod日志管理
```bash
# 查看Pod日志
kubectl logs basic-pod

# 实时跟踪日志
kubectl logs -f basic-pod

# 查看前100行日志
kubectl logs --tail=100 basic-pod

# 查看最近1小时的日志
kubectl logs --since=1h basic-pod

# 多容器Pod指定容器
kubectl logs multi-container-pod -c web-server

# 查看之前的容器日志（如果重启过）
kubectl logs basic-pod --previous
```

### 3. 进入Pod调试
```bash
# 进入Pod执行命令
kubectl exec basic-pod -- ls -la /usr/share/nginx/html

# 交互式进入Pod
kubectl exec -it basic-pod -- /bin/sh

# 多容器Pod指定容器
kubectl exec -it multi-container-pod -c web-server -- /bin/sh

# 在Pod中运行特定命令
kubectl exec basic-pod -- nginx -t
```

### 4. 文件传输
```bash
# 从Pod复制文件到本地
kubectl cp basic-pod:/etc/nginx/nginx.conf ./nginx.conf

# 从本地复制文件到Pod
kubectl cp ./index.html basic-pod:/usr/share/nginx/html/

# 多容器Pod指定容器
kubectl cp basic-pod:/var/log/nginx/access.log ./access.log -c nginx
```

## 🏥 健康检查配置

Kubernetes提供三种类型的健康检查：

### 1. 存活探针（Liveness Probe）
检查容器是否还在运行，失败时重启容器。

```bash
cat > ~/k8s-labs/lab02/liveness-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: liveness-pod
spec:
  containers:
  - name: nginx
    image: nginx:alpine
    ports:
    - containerPort: 80
    livenessProbe:
      httpGet:
        path: /
        port: 80
      initialDelaySeconds: 30  # 容器启动30秒后开始检查
      periodSeconds: 10        # 每10秒检查一次
      timeoutSeconds: 5        # 检查超时时间5秒
      failureThreshold: 3      # 连续失败3次后重启容器
      successThreshold: 1      # 连续成功1次认为恢复
EOF
```

### 2. 就绪探针（Readiness Probe）
检查容器是否准备好接收流量，失败时从Service中移除。

```bash
cat > ~/k8s-labs/lab02/readiness-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: readiness-pod
spec:
  containers:
  - name: web-app
    image: nginx:alpine
    ports:
    - containerPort: 80
    readinessProbe:
      httpGet:
        path: /health
        port: 80
      initialDelaySeconds: 5
      periodSeconds: 3
      timeoutSeconds: 2
      failureThreshold: 3
      successThreshold: 2
    # 模拟健康检查端点
    lifecycle:
      postStart:
        exec:
          command:
          - /bin/sh
          - -c
          - |
            sleep 10
            echo "OK" > /usr/share/nginx/html/health
EOF
```

### 3. 启动探针（Startup Probe）
检查容器是否已启动，在启动探针成功前，其他探针不会运行。

```bash
cat > ~/k8s-labs/lab02/startup-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: startup-pod
spec:
  containers:
  - name: slow-start-app
    image: nginx:alpine
    ports:
    - containerPort: 80
    startupProbe:
      httpGet:
        path: /
        port: 80
      initialDelaySeconds: 10
      periodSeconds: 5
      timeoutSeconds: 3
      failureThreshold: 12  # 允许启动时间：12 * 5 = 60秒
    livenessProbe:
      httpGet:
        path: /
        port: 80
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /
        port: 80
      periodSeconds: 5
EOF
```

### 4. 部署和测试健康检查
```bash
# 部署健康检查Pod
kubectl apply -f ~/k8s-labs/lab02/liveness-pod.yaml
kubectl apply -f ~/k8s-labs/lab02/readiness-pod.yaml
kubectl apply -f ~/k8s-labs/lab02/startup-pod.yaml

# 观察Pod状态变化
kubectl get pods -w

# 查看Pod详细信息
kubectl describe pod liveness-pod
kubectl describe pod readiness-pod
kubectl describe pod startup-pod

# 模拟健康检查失败
kubectl exec liveness-pod -- rm /usr/share/nginx/html/index.html
```

## ⚙️ 资源管理

### 1. 资源请求与限制
```bash
cat > ~/k8s-labs/lab02/resource-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: resource-pod
spec:
  containers:
  - name: resource-demo
    image: nginx:alpine
    resources:
      requests:
        memory: "64Mi"    # 最少需要64MB内存
        cpu: "250m"       # 最少需要0.25核CPU
      limits:
        memory: "128Mi"   # 最多使用128MB内存
        cpu: "500m"       # 最多使用0.5核CPU
    ports:
    - containerPort: 80
EOF
```

### 2. 服务质量类别（QoS Classes）

Kubernetes根据资源配置将Pod分为三个QoS类别：

| QoS类别 | 条件 | 特点 |
|---------|------|------|
| **Guaranteed** | requests = limits | 最高优先级，最后被驱逐 |
| **Burstable** | 有requests但requests < limits | 中等优先级 |
| **BestEffort** | 没有设置requests和limits | 最低优先级，首先被驱逐 |

```bash
# 创建不同QoS类别的Pod示例
cat > ~/k8s-labs/lab02/qos-examples.yaml << 'EOF'
# Guaranteed QoS
apiVersion: v1
kind: Pod
metadata:
  name: guaranteed-pod
spec:
  containers:
  - name: guaranteed-container
    image: nginx:alpine
    resources:
      requests:
        memory: "100Mi"
        cpu: "100m"
      limits:
        memory: "100Mi"
        cpu: "100m"
---
# Burstable QoS
apiVersion: v1
kind: Pod
metadata:
  name: burstable-pod
spec:
  containers:
  - name: burstable-container
    image: nginx:alpine
    resources:
      requests:
        memory: "50Mi"
        cpu: "50m"
      limits:
        memory: "100Mi"
        cpu: "200m"
---
# BestEffort QoS
apiVersion: v1
kind: Pod
metadata:
  name: besteffort-pod
spec:
  containers:
  - name: besteffort-container
    image: nginx:alpine
EOF
```

### 3. 部署和验证QoS
```bash
# 部署资源管理示例
kubectl apply -f ~/k8s-labs/lab02/resource-pod.yaml
kubectl apply -f ~/k8s-labs/lab02/qos-examples.yaml

# 查看Pod的QoS类别
kubectl describe pod guaranteed-pod | grep "QoS Class"
kubectl describe pod burstable-pod | grep "QoS Class"
kubectl describe pod besteffort-pod | grep "QoS Class"

# 查看Pod资源使用（需要metrics-server）
kubectl top pod resource-pod
```

## 🔄 Pod重启策略

### 重启策略类型
| 策略 | 描述 | 适用场景 |
|------|------|----------|
| **Always** | 总是重启（默认） | 长期运行的服务 |
| **OnFailure** | 只有失败时重启 | 批处理任务 |
| **Never** | 从不重启 | 一次性任务 |

```bash
cat > ~/k8s-labs/lab02/restart-policy-examples.yaml << 'EOF'
# Always重启策略
apiVersion: v1
kind: Pod
metadata:
  name: always-restart-pod
spec:
  restartPolicy: Always
  containers:
  - name: app
    image: busybox
    command: ["/bin/sh"]
    args: ["-c", "echo 'Starting...'; sleep 30; echo 'Crashing...'; exit 1"]
---
# OnFailure重启策略
apiVersion: v1
kind: Pod
metadata:
  name: onfailure-restart-pod
spec:
  restartPolicy: OnFailure
  containers:
  - name: job
    image: busybox
    command: ["/bin/sh"]
    args: ["-c", "echo 'Job starting...'; sleep 20; echo 'Job failed'; exit 1"]
---
# Never重启策略
apiVersion: v1
kind: Pod
metadata:
  name: never-restart-pod
spec:
  restartPolicy: Never
  containers:
  - name: task
    image: busybox
    command: ["/bin/sh"]
    args: ["-c", "echo 'Task completed'; sleep 10; exit 0"]
EOF
```

## 🛠️ 实验练习

### 练习1：Pod生命周期观察
```bash
# 1. 创建一个会自动退出的Pod
kubectl run lifecycle-pod --image=busybox \
  --restart=Never -- sh -c "echo 'Hello'; sleep 60; echo 'Goodbye'"

# 2. 实时观察Pod状态变化
kubectl get pods lifecycle-pod -w

# 3. 查看Pod事件
kubectl describe pod lifecycle-pod

# 4. 查看Pod日志
kubectl logs lifecycle-pod
```

### 练习2：多容器协作示例
创建一个日志收集器Pod：

```bash
cat > ~/k8s-labs/lab02/log-collector-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: log-collector-pod
spec:
  containers:
  # 应用容器：生成日志
  - name: app
    image: busybox
    command: ["/bin/sh"]
    args:
    - -c
    - |
      while true; do
        echo "$(date): Application log entry" >> /var/log/app.log
        sleep 5
      done
    volumeMounts:
    - name: log-volume
      mountPath: /var/log
  
  # 日志收集器：读取并处理日志
  - name: log-collector
    image: busybox
    command: ["/bin/sh"]
    args:
    - -c
    - |
      while true; do
        if [ -f /logs/app.log ]; then
          echo "Collecting logs at $(date):"
          tail -n 1 /logs/app.log
        fi
        sleep 10
      done
    volumeMounts:
    - name: log-volume
      mountPath: /logs
  
  volumes:
  - name: log-volume
    emptyDir: {}
EOF

# 部署并观察
kubectl apply -f ~/k8s-labs/lab02/log-collector-pod.yaml

# 查看应用日志
kubectl logs log-collector-pod -c app

# 查看收集器日志
kubectl logs log-collector-pod -c log-collector -f
```

### 练习3：故障模拟与恢复
```bash
# 1. 创建带健康检查的Pod
kubectl apply -f ~/k8s-labs/lab02/liveness-pod.yaml

# 2. 观察正常状态
kubectl get pods liveness-pod

# 3. 模拟应用故障
kubectl exec liveness-pod -- rm -rf /usr/share/nginx/html/*

# 4. 观察Pod重启过程
kubectl get pods liveness-pod -w

# 5. 查看重启历史
kubectl describe pod liveness-pod
```

## 🧪 进阶实验

### 实验1：Pod设计模式 - Sidecar模式
```bash
cat > ~/k8s-labs/lab02/sidecar-pattern.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: sidecar-pod
spec:
  containers:
  # 主容器：Web应用
  - name: web-app
    image: nginx:alpine
    ports:
    - containerPort: 80
    volumeMounts:
    - name: config
      mountPath: /etc/nginx/conf.d
    - name: logs
      mountPath: /var/log/nginx
  
  # Sidecar容器：配置管理
  - name: config-manager
    image: busybox
    command: ["/bin/sh"]
    args:
    - -c
    - |
      while true; do
        cat > /config/default.conf << EOF
      server {
          listen 80;
          server_name localhost;
          
          location / {
              root /usr/share/nginx/html;
              index index.html;
          }
          
          # 动态配置：$(date)
          location /status {
              return 200 "Status OK at $(date)\n";
              add_header Content-Type text/plain;
          }
      }
      EOF
        echo "Configuration updated at $(date)"
        sleep 300  # 每5分钟更新一次配置
      done
    volumeMounts:
    - name: config
      mountPath: /config
  
  # Sidecar容器：日志处理
  - name: log-processor
    image: busybox
    command: ["/bin/sh"]
    args:
    - -c
    - |
      while true; do
        if [ -f /logs/access.log ]; then
          echo "Processing access logs..."
          grep "GET" /logs/access.log | tail -5
        fi
        sleep 30
      done
    volumeMounts:
    - name: logs
      mountPath: /logs
  
  volumes:
  - name: config
    emptyDir: {}
  - name: logs
    emptyDir: {}
EOF
```

### 实验2：Init容器示例
```bash
cat > ~/k8s-labs/lab02/init-container-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: init-container-pod
spec:
  # Init容器：在主容器启动前运行
  initContainers:
  - name: init-db
    image: busybox
    command: ["/bin/sh"]
    args:
    - -c
    - |
      echo "Initializing database..."
      echo "CREATE DATABASE app;" > /data/init.sql
      echo "Database initialization complete"
    volumeMounts:
    - name: data
      mountPath: /data
  
  - name: init-config
    image: busybox
    command: ["/bin/sh"]
    args:
    - -c
    - |
      echo "Setting up configuration..."
      cat > /config/app.conf << EOF
      database_host=localhost
      database_port=3306
      log_level=info
      EOF
      echo "Configuration setup complete"
    volumeMounts:
    - name: config
      mountPath: /config
  
  # 主容器
  containers:
  - name: app
    image: nginx:alpine
    ports:
    - containerPort: 80
    volumeMounts:
    - name: data
      mountPath: /data
    - name: config
      mountPath: /etc/app
    command: ["/bin/sh"]
    args:
    - -c
    - |
      echo "Main application starting..."
      echo "Database files:"
      ls -la /data/
      echo "Configuration files:"
      ls -la /etc/app/
      nginx -g "daemon off;"
  
  volumes:
  - name: data
    emptyDir: {}
  - name: config
    emptyDir: {}
EOF
```

## 🐛 故障排查指南

### 常见Pod问题及解决方案

#### 1. Pod处于Pending状态
```bash
# 检查Pod事件
kubectl describe pod <pod-name>

# 常见原因和解决方案：
# - 资源不足：检查节点资源
kubectl describe nodes

# - 镜像拉取失败：检查镜像名称和网络
kubectl describe pod <pod-name> | grep -A 10 "Events"

# - 存储卷挂载失败：检查PV/PVC状态
kubectl get pv,pvc
```

#### 2. Pod处于CrashLoopBackOff状态
```bash
# 查看Pod重启历史
kubectl describe pod <pod-name>

# 查看容器日志
kubectl logs <pod-name> --previous

# 检查容器启动命令
kubectl get pod <pod-name> -o yaml | grep -A 10 "command"
```

#### 3. Pod处于ImagePullBackOff状态
```bash
# 检查镜像拉取事件
kubectl describe pod <pod-name> | grep -A 5 "Failed to pull image"

# 解决方案：
# - 检查镜像名称拼写
# - 验证镜像是否存在
# - 检查私有仓库认证
```

#### 4. 健康检查失败
```bash
# 查看探针配置
kubectl describe pod <pod-name> | grep -A 10 "Liveness\|Readiness"

# 手动测试健康检查端点
kubectl exec <pod-name> -- curl localhost:8080/health

# 调整探针参数
kubectl edit pod <pod-name>
```

### 调试工具集合
```bash
# 1. 综合信息收集脚本
cat > ~/k8s-labs/lab02/debug-pod.sh << 'EOF'
#!/bin/bash
POD_NAME=$1

if [ -z "$POD_NAME" ]; then
    echo "Usage: $0 <pod-name>"
    exit 1
fi

echo "=== Pod Basic Info ==="
kubectl get pod $POD_NAME -o wide

echo -e "\n=== Pod Description ==="
kubectl describe pod $POD_NAME

echo -e "\n=== Pod Events ==="
kubectl get events --field-selector involvedObject.name=$POD_NAME

echo -e "\n=== Pod Logs ==="
kubectl logs $POD_NAME --tail=20

echo -e "\n=== Pod YAML ==="
kubectl get pod $POD_NAME -o yaml
EOF

chmod +x ~/k8s-labs/lab02/debug-pod.sh
```

## 💡 最佳实践

### 1. Pod设计原则
- **单一职责**：每个Pod只负责一个主要功能
- **无状态化**：尽量避免在Pod中存储状态
- **优雅终止**：正确处理SIGTERM信号
- **资源限制**：总是设置资源requests和limits

### 2. 标签管理策略
```yaml
metadata:
  labels:
    app: web-server           # 应用名称
    version: v1.2.3          # 版本号
    component: frontend      # 组件类型
    tier: web               # 应用层级
    environment: production  # 环境
    owner: team-alpha       # 负责团队
```

### 3. 健康检查最佳实践
- **合理设置时间**：避免过于频繁的检查
- **区分检查类型**：明确何时使用liveness vs readiness
- **提供健康端点**：应用应该提供专门的健康检查接口
- **考虑启动时间**：为慢启动应用设置合适的startup probe

## 📝 学习检查

完成本实验后，你应该能够回答：

1. **概念理解**：
   - Pod和容器的关系是什么？
   - Pod的生命周期包含哪些阶段？
   - 三种健康检查的区别和用途？

2. **操作技能**：
   - 如何创建多容器Pod？
   - 如何查看Pod的资源使用情况？
   - 如何排查Pod启动失败的问题？

3. **实际应用**：
   - 什么情况下使用多容器Pod？
   - 如何设计合适的资源限制？
   - 如何实现Pod的优雅关闭？

## 🔗 延伸学习

- 学习Pod安全上下文（Security Context）
- 了解Pod亲和性和反亲和性
- 探索Pod Disruption Budget（PDB）
- 研究垂直Pod自动扩缩容（VPA）

## ⏭️ 下一步

完成本实验后，继续学习：
- **实验3**：Service服务发现与负载均衡 - 学习如何暴露和访问Pod
- 探索服务网格和流量管理的概念

---

**恭喜完成Pod管理实验！** 🎉
你现在已经掌握了Kubernetes中最核心的概念，可以继续探索服务发现和网络管理。 