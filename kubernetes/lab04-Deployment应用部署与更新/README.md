# 实验4：Deployment应用部署与更新

## 🎯 学习目标

通过本实验，你将：
- 深入理解Deployment的概念和工作原理
- 掌握应用的声明式部署和管理
- 学习滚动更新和版本回滚策略
- 实践扩缩容和自动扩缩容
- 了解部署策略和最佳实践
- 掌握Deployment故障排查技巧

## 📚 理论知识学习

### Deployment核心概念

Deployment是Kubernetes中用于管理无状态应用的核心控制器，它提供了声明式的应用部署和更新能力。

#### Deployment的职责
- **Pod管理**：确保指定数量的Pod副本运行
- **滚动更新**：零停机时间的应用更新
- **版本控制**：保存更新历史，支持回滚
- **扩缩容**：水平扩展Pod数量
- **自愈能力**：自动替换失败的Pod

### Deployment架构

```
┌─────────── Deployment ───────────┐
│  replicas: 3                     │
│  strategy: RollingUpdate         │
│  ┌─────── ReplicaSet v2 ────────┐ │
│  │  replicas: 3                 │ │
│  │  ┌─────┐ ┌─────┐ ┌─────┐   │ │
│  │  │Pod1 │ │Pod2 │ │Pod3 │   │ │
│  │  └─────┘ └─────┘ └─────┘   │ │
│  └─────────────────────────────┘ │
│  ┌─────── ReplicaSet v1 ────────┐ │
│  │  replicas: 0                 │ │  <- 旧版本（保留历史）
│  └─────────────────────────────┘ │
└─────────────────────────────────────┘
```

### Deployment vs ReplicaSet vs Pod

| 层级 | 职责 | 使用场景 |
|------|------|----------|
| **Pod** | 运行容器 | 单个容器实例 |
| **ReplicaSet** | 管理Pod副本 | 确保Pod数量 |
| **Deployment** | 管理ReplicaSet | 应用部署和更新 |

## 🔧 Deployment基本操作

### 准备工作

```bash
# 创建实验目录
mkdir -p ~/k8s-labs/lab04

# 清理之前的资源（如果有）
kubectl delete deployment --all
kubectl delete service --all
```

### 1. 创建基本Deployment

#### 命令行创建
```bash
# 快速创建Deployment
kubectl create deployment nginx-app --image=nginx:1.20 --replicas=3

# 查看Deployment
kubectl get deployments
kubectl get rs  # ReplicaSet
kubectl get pods

# 查看详细信息
kubectl describe deployment nginx-app
```

#### YAML文件创建
```bash
cat > ~/k8s-labs/lab04/basic-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  labels:
    app: web-app
    version: v1.0
spec:
  replicas: 3                    # Pod副本数
  selector:
    matchLabels:
      app: web-app
  template:                      # Pod模板
    metadata:
      labels:
        app: web-app
        version: v1.0
    spec:
      containers:
      - name: web
        image: nginx:1.20-alpine
        ports:
        - containerPort: 80
          name: http
        env:
        - name: VERSION
          value: "v1.0"
        # 资源限制
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
        # 健康检查
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
EOF

# 应用Deployment
kubectl apply -f ~/k8s-labs/lab04/basic-deployment.yaml

# 查看部署状态
kubectl rollout status deployment/web-app
```

### 2. Deployment状态查看

```bash
# 基本信息
kubectl get deployment web-app
kubectl get deployment web-app -o wide

# 详细描述
kubectl describe deployment web-app

# YAML格式输出
kubectl get deployment web-app -o yaml

# 查看ReplicaSet和Pod
kubectl get rs -l app=web-app
kubectl get pods -l app=web-app

# 查看Events
kubectl get events --field-selector involvedObject.name=web-app
```

## 🔄 滚动更新操作

### 1. 镜像更新

#### 方法一：kubectl set image
```bash
# 更新镜像版本
kubectl set image deployment/web-app web=nginx:1.21-alpine

# 实时观察更新过程
kubectl rollout status deployment/web-app

# 查看更新过程中的Pod变化
kubectl get pods -l app=web-app -w
```

#### 方法二：kubectl patch
```bash
# 使用patch更新
kubectl patch deployment web-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"web","image":"nginx:1.22-alpine"}]}}}}'

# 或者更新环境变量
kubectl patch deployment web-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"web","env":[{"name":"VERSION","value":"v1.2"}]}]}}}}'
```

#### 方法三：kubectl edit
```bash
# 直接编辑Deployment
kubectl edit deployment web-app
# 修改image字段或其他配置
```

#### 方法四：YAML文件更新
```bash
# 修改YAML文件中的镜像版本
sed -i 's/nginx:1.20-alpine/nginx:1.21-alpine/' ~/k8s-labs/lab04/basic-deployment.yaml

# 应用更新
kubectl apply -f ~/k8s-labs/lab04/basic-deployment.yaml
```

### 2. 滚动更新策略配置

```bash
cat > ~/k8s-labs/lab04/rolling-update-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rolling-app
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1      # 更新过程中最多1个Pod不可用
      maxSurge: 2           # 更新过程中最多额外创建2个Pod
  selector:
    matchLabels:
      app: rolling-app
  template:
    metadata:
      labels:
        app: rolling-app
    spec:
      containers:
      - name: app
        image: nginx:1.20-alpine
        ports:
        - containerPort: 80
        # 模拟慢启动
        lifecycle:
          postStart:
            exec:
              command: ["/bin/sh", "-c", "sleep 10"]
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 2
EOF

kubectl apply -f ~/k8s-labs/lab04/rolling-update-deployment.yaml

# 触发滚动更新并观察过程
kubectl set image deployment/rolling-app app=nginx:1.21-alpine

# 在另一个终端观察Pod变化
kubectl get pods -l app=rolling-app -w
```

### 3. 重新创建策略

```bash
cat > ~/k8s-labs/lab04/recreate-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recreate-app
spec:
  replicas: 3
  strategy:
    type: Recreate    # 重新创建策略：先删除所有旧Pod，再创建新Pod
  selector:
    matchLabels:
      app: recreate-app
  template:
    metadata:
      labels:
        app: recreate-app
    spec:
      containers:
      - name: app
        image: nginx:1.20-alpine
        ports:
        - containerPort: 80
EOF

kubectl apply -f ~/k8s-labs/lab04/recreate-deployment.yaml

# 触发更新观察重新创建过程
kubectl set image deployment/recreate-app app=nginx:1.21-alpine
kubectl get pods -l app=recreate-app -w
```

## ⏪ 版本回滚操作

### 1. 查看更新历史

```bash
# 查看部署历史
kubectl rollout history deployment/web-app

# 查看特定版本的详细信息
kubectl rollout history deployment/web-app --revision=2

# 查看当前版本信息
kubectl describe deployment web-app | grep -A 10 "Pod Template"
```

### 2. 回滚操作

```bash
# 回滚到上一个版本
kubectl rollout undo deployment/web-app

# 回滚到特定版本
kubectl rollout undo deployment/web-app --to-revision=1

# 查看回滚状态
kubectl rollout status deployment/web-app

# 验证回滚结果
kubectl describe deployment web-app | grep Image
```

### 3. 暂停和恢复更新

```bash
# 暂停正在进行的更新
kubectl rollout pause deployment/web-app

# 恢复暂停的更新
kubectl rollout resume deployment/web-app

# 重启部署（强制更新所有Pod）
kubectl rollout restart deployment/web-app
```

## 📊 扩缩容操作

### 1. 手动扩缩容

```bash
# 扩容到5个副本
kubectl scale deployment web-app --replicas=5

# 查看扩容过程
kubectl get pods -l app=web-app -w

# 缩容到2个副本
kubectl scale deployment web-app --replicas=2

# 基于条件扩容
kubectl scale deployment web-app --current-replicas=2 --replicas=4
```

### 2. 水平Pod自动扩缩容（HPA）

#### 前置条件：安装metrics-server
```bash
# 检查metrics-server是否安装
kubectl get pods -n kube-system | grep metrics-server

# 如果没有安装，在minikube中启用
minikube addons enable metrics-server

# 或在Docker Desktop中部署
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

#### 创建HPA
```bash
# 为Deployment创建HPA
kubectl autoscale deployment web-app --cpu-percent=50 --min=2 --max=10

# 或使用YAML文件
cat > ~/k8s-labs/lab04/hpa.yaml << 'EOF'
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
EOF

kubectl apply -f ~/k8s-labs/lab04/hpa.yaml

# 查看HPA状态
kubectl get hpa
kubectl describe hpa web-app-hpa
```

#### 负载测试触发自动扩容
```bash
# 创建负载测试Pod
kubectl run load-generator --image=busybox --rm -it -- sh

# 在Pod内执行（生成负载）
while true; do wget -q -O- http://web-app-service/; done

# 在另一个终端观察HPA和Pod变化
watch -n 2 'kubectl get hpa,pods'

# 查看Pod资源使用
kubectl top pods
```

## 🛠️ 实验练习

### 练习1：蓝绿部署模拟

```bash
cat > ~/k8s-labs/lab04/blue-green-deployment.yaml << 'EOF'
# Blue版本 (当前生产版本)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-blue
  labels:
    app: webapp
    version: blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
      version: blue
  template:
    metadata:
      labels:
        app: webapp
        version: blue
    spec:
      containers:
      - name: webapp
        image: nginx:1.20-alpine
        ports:
        - containerPort: 80
        env:
        - name: VERSION
          value: "Blue-v1.0"
        lifecycle:
          postStart:
            exec:
              command:
              - /bin/sh
              - -c
              - |
                echo '<h1 style="color: blue">Blue Version - v1.0</h1>' > /usr/share/nginx/html/index.html
                echo '<p>Current time: '$(date)'</p>' >> /usr/share/nginx/html/index.html
---
# Green版本 (新版本)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-green
  labels:
    app: webapp
    version: green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
      version: green
  template:
    metadata:
      labels:
        app: webapp
        version: green
    spec:
      containers:
      - name: webapp
        image: nginx:1.21-alpine
        ports:
        - containerPort: 80
        env:
        - name: VERSION
          value: "Green-v2.0"
        lifecycle:
          postStart:
            exec:
              command:
              - /bin/sh
              - -c
              - |
                echo '<h1 style="color: green">Green Version - v2.0</h1>' > /usr/share/nginx/html/index.html
                echo '<p>Current time: '$(date)'</p>' >> /usr/share/nginx/html/index.html
---
# Service - 指向Blue版本
apiVersion: v1
kind: Service
metadata:
  name: webapp-service
spec:
  selector:
    app: webapp
    version: blue    # 当前指向blue版本
  ports:
  - port: 80
    targetPort: 80
  type: NodePort
EOF

kubectl apply -f ~/k8s-labs/lab04/blue-green-deployment.yaml

# 测试当前版本
kubectl port-forward service/webapp-service 8080:80
# 访问 http://localhost:8080 查看Blue版本

# 切换到Green版本
kubectl patch service webapp-service -p '{"spec":{"selector":{"version":"green"}}}'

# 再次测试查看版本切换
# 如果有问题，快速切回Blue版本
kubectl patch service webapp-service -p '{"spec":{"selector":{"version":"blue"}}}'
```

### 练习2：金丝雀部署

```bash
cat > ~/k8s-labs/lab04/canary-deployment.yaml << 'EOF'
# 稳定版本 - 90%流量
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-stable
  labels:
    app: webapp
    track: stable
spec:
  replicas: 9    # 90%流量
  selector:
    matchLabels:
      app: webapp
      track: stable
  template:
    metadata:
      labels:
        app: webapp
        track: stable
    spec:
      containers:
      - name: webapp
        image: nginx:1.20-alpine
        ports:
        - containerPort: 80
        lifecycle:
          postStart:
            exec:
              command:
              - /bin/sh
              - -c
              - |
                echo '<h1>Stable Version</h1>' > /usr/share/nginx/html/index.html
                echo '<p>Pod: '$(hostname)'</p>' >> /usr/share/nginx/html/index.html
---
# 金丝雀版本 - 10%流量
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-canary
  labels:
    app: webapp
    track: canary
spec:
  replicas: 1    # 10%流量
  selector:
    matchLabels:
      app: webapp
      track: canary
  template:
    metadata:
      labels:
        app: webapp
        track: canary
    spec:
      containers:
      - name: webapp
        image: nginx:1.21-alpine
        ports:
        - containerPort: 80
        lifecycle:
          postStart:
            exec:
              command:
              - /bin/sh
              - -c
              - |
                echo '<h1 style="color: orange">Canary Version</h1>' > /usr/share/nginx/html/index.html
                echo '<p>Pod: '$(hostname)'</p>' >> /usr/share/nginx/html/index.html
---
# Service - 同时选择两个版本
apiVersion: v1
kind: Service
metadata:
  name: webapp-canary-service
spec:
  selector:
    app: webapp    # 只匹配app，不匹配track
  ports:
  - port: 80
    targetPort: 80
  type: NodePort
EOF

kubectl apply -f ~/k8s-labs/lab04/canary-deployment.yaml

# 测试流量分发
for i in {1..20}; do
  echo "Request $i:"
  kubectl exec -it $(kubectl get pod -l app=dns-test -o jsonpath='{.items[0].metadata.name}') -- wget -qO- webapp-canary-service | grep '<h1>'
  sleep 1
done
```

### 练习3：Deployment配置更新最佳实践

```bash
cat > ~/k8s-labs/lab04/production-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: production-app
  labels:
    app: production-app
    environment: prod
  annotations:
    deployment.kubernetes.io/revision: "1"
    kubernetes.io/change-cause: "Initial deployment with nginx:1.20"
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: production-app
  template:
    metadata:
      labels:
        app: production-app
        environment: prod
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9113"
    spec:
      # 安全上下文
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: webapp
        image: nginx:1.20-alpine
        ports:
        - containerPort: 80
          name: http
        # 资源管理
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        # 环境变量
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        # 健康检查
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
        # 优雅关闭
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 10"]
        # 卷挂载
        volumeMounts:
        - name: config
          mountPath: /etc/nginx/conf.d
          readOnly: true
      # 卷定义
      volumes:
      - name: config
        configMap:
          name: nginx-config
      # Pod调度约束
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - production-app
              topologyKey: kubernetes.io/hostname
      # 容忍度
      tolerations:
      - key: "production"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
EOF
```

## 🧪 进阶实验

### 实验1：多容器Deployment

```bash
cat > ~/k8s-labs/lab04/multi-container-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multi-container-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: multi-container-app
  template:
    metadata:
      labels:
        app: multi-container-app
    spec:
      containers:
      # 主应用容器
      - name: webapp
        image: nginx:alpine
        ports:
        - containerPort: 80
        volumeMounts:
        - name: shared-data
          mountPath: /usr/share/nginx/html
      
      # Sidecar容器：日志收集
      - name: log-collector
        image: fluent/fluent-bit:latest
        env:
        - name: FLUENT_CONF
          value: fluent-bit.conf
        volumeMounts:
        - name: shared-data
          mountPath: /var/log/nginx
        - name: fluent-bit-config
          mountPath: /fluent-bit/etc/
      
      # Init容器：初始化数据
      initContainers:
      - name: init-data
        image: busybox
        command: ['sh', '-c']
        args:
        - |
          echo '<h1>Multi-Container App</h1>' > /data/index.html
          echo '<p>Initialized at: '$(date)'</p>' >> /data/index.html
        volumeMounts:
        - name: shared-data
          mountPath: /data
      
      volumes:
      - name: shared-data
        emptyDir: {}
      - name: fluent-bit-config
        configMap:
          name: fluent-bit-config
EOF
```

### 实验2：Deployment高级调度

```bash
cat > ~/k8s-labs/lab04/advanced-scheduling.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: scheduled-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: scheduled-app
  template:
    metadata:
      labels:
        app: scheduled-app
    spec:
      # 节点选择器
      nodeSelector:
        disktype: ssd
      
      # 节点亲和性
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/arch
                operator: In
                values:
                - amd64
                - arm64
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 1
            preference:
              matchExpressions:
              - key: instance-type
                operator: In
                values:
                - compute-optimized
        
        # Pod亲和性
        podAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 50
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - database
              topologyKey: kubernetes.io/hostname
        
        # Pod反亲和性
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - scheduled-app
            topologyKey: kubernetes.io/hostname
      
      # 容忍度
      tolerations:
      - key: "dedicated"
        operator: "Equal"
        value: "experimental"
        effect: "NoSchedule"
      - key: "experimental"
        operator: "Exists"
        effect: "NoExecute"
        tolerationSeconds: 3600
      
      containers:
      - name: app
        image: nginx:alpine
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
EOF
```

## 🐛 故障排查指南

### 常见Deployment问题

#### 1. Pod无法启动
```bash
# 检查Deployment状态
kubectl describe deployment <deployment-name>

# 查看ReplicaSet状态
kubectl describe rs <replicaset-name>

# 查看Pod详情
kubectl describe pod <pod-name>

# 查看Pod日志
kubectl logs <pod-name>

# 检查镜像拉取
kubectl get events --field-selector involvedObject.name=<pod-name>
```

#### 2. 滚动更新卡住
```bash
# 检查更新状态
kubectl rollout status deployment/<deployment-name>

# 查看更新历史
kubectl rollout history deployment/<deployment-name>

# 检查Pod状态分布
kubectl get pods -l app=<app-label> -o wide

# 强制重新部署
kubectl rollout restart deployment/<deployment-name>
```

#### 3. HPA不工作
```bash
# 检查metrics-server
kubectl get pods -n kube-system | grep metrics-server

# 检查HPA状态
kubectl describe hpa <hpa-name>

# 查看资源使用情况
kubectl top pods
kubectl top nodes

# 检查资源请求设置
kubectl describe deployment <deployment-name> | grep -A 5 "Requests"
```

### 调试工具脚本

```bash
cat > ~/k8s-labs/lab04/deployment-debug.sh << 'EOF'
#!/bin/bash

DEPLOYMENT_NAME=$1
if [ -z "$DEPLOYMENT_NAME" ]; then
    echo "Usage: $0 <deployment-name>"
    exit 1
fi

echo "=== Deployment Debug Information ==="
echo "Deployment: $DEPLOYMENT_NAME"

echo -e "\n1. Deployment Status:"
kubectl get deployment $DEPLOYMENT_NAME -o wide

echo -e "\n2. Deployment Description:"
kubectl describe deployment $DEPLOYMENT_NAME

echo -e "\n3. ReplicaSets:"
kubectl get rs -l app=$(kubectl get deployment $DEPLOYMENT_NAME -o jsonpath='{.spec.selector.matchLabels.app}')

echo -e "\n4. Pods:"
kubectl get pods -l app=$(kubectl get deployment $DEPLOYMENT_NAME -o jsonpath='{.spec.selector.matchLabels.app}') -o wide

echo -e "\n5. Rollout History:"
kubectl rollout history deployment/$DEPLOYMENT_NAME

echo -e "\n6. Events:"
kubectl get events --field-selector involvedObject.name=$DEPLOYMENT_NAME --sort-by='.lastTimestamp'

echo -e "\n7. HPA (if exists):"
kubectl get hpa | grep $DEPLOYMENT_NAME || echo "No HPA found"

echo -e "\n8. Resource Usage:"
kubectl top pods -l app=$(kubectl get deployment $DEPLOYMENT_NAME -o jsonpath='{.spec.selector.matchLabels.app}') 2>/dev/null || echo "Metrics not available"
EOF

chmod +x ~/k8s-labs/lab04/deployment-debug.sh
```

## 💡 最佳实践

### 1. Deployment配置最佳实践

```yaml
# 完整的生产级Deployment配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: production-app
  labels:
    app: production-app
    version: v1.0.0
    environment: production
  annotations:
    kubernetes.io/change-cause: "Deploy version v1.0.0"
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: production-app
  template:
    metadata:
      labels:
        app: production-app
        version: v1.0.0
        environment: production
    spec:
      containers:
      - name: app
        image: myapp:v1.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
        # 始终设置资源限制
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        # 健康检查是必须的
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        # 优雅关闭
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]
```

### 2. 部署策略选择

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **滚动更新** | 零停机、渐进式 | 版本共存期 | 大多数Web应用 |
| **重新创建** | 简单、无版本混合 | 有停机时间 | 单实例应用 |
| **蓝绿部署** | 快速切换、易回滚 | 资源消耗大 | 关键业务系统 |
| **金丝雀部署** | 风险可控、渐进验证 | 复杂度高 | 高风险更新 |

### 3. 监控和告警

```yaml
# 在Deployment中添加监控标签
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
    prometheus.io/path: "/metrics"
```

### 4. 安全考虑

```yaml
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: app
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
```

## 📝 学习检查

完成本实验后，你应该能够回答：

1. **概念理解**：
   - Deployment、ReplicaSet、Pod的关系？
   - 滚动更新的工作原理？
   - 不同部署策略的适用场景？

2. **操作技能**：
   - 如何创建和更新Deployment？
   - 如何配置HPA实现自动扩缩容？
   - 如何排查Deployment部署失败的问题？

3. **实际应用**：
   - 如何设计生产环境的部署策略？
   - 如何实现零停机部署？
   - 如何处理部署失败和回滚？

## 🔗 延伸学习

- 学习StatefulSet用于有状态应用部署
- 了解DaemonSet用于节点级服务部署
- 探索Job和CronJob用于批处理任务
- 研究GitOps和CD流水线集成

## ⏭️ 下一步

完成本实验后，继续学习：
- **实验5**：ConfigMap和Secret配置管理 - 学习应用配置的外部化管理
- 探索配置热更新和敏感信息管理

---

**恭喜完成Deployment实验！** 🎉
你现在已经掌握了Kubernetes应用部署和更新的核心技能，可以继续探索配置管理和存储。 