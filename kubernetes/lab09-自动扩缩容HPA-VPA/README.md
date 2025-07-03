# 实验9：自动扩缩容HPA/VPA

## 🎯 学习目标

通过本实验，你将：
- 深入理解Kubernetes自动扩缩容机制
- 掌握HPA(水平Pod自动扩缩容)配置和使用
- 学习VPA(垂直Pod自动扩缩容)实践
- 了解集群自动扩缩容原理
- 实践多维度指标的扩缩容策略
- 掌握扩缩容的监控和调优

## 📚 理论知识学习

### 扩缩容类型对比

| 类型 | 全称 | 扩缩容方向 | 适用场景 |
|------|------|------------|----------|
| **HPA** | Horizontal Pod Autoscaler | 水平(Pod数量) | CPU/内存密集型应用 |
| **VPA** | Vertical Pod Autoscaler | 垂直(资源配额) | 资源需求变化的应用 |
| **CA** | Cluster Autoscaler | 集群节点 | 节点资源不足时 |

### HPA工作原理

```
┌─── Metrics Server ───┐    ┌─── HPA Controller ───┐
│                      │    │                      │
│  CPU: 80%           │───→│  Target: 50%         │
│  Memory: 60%        │    │  Replicas: 3→5       │
│  Custom: 120%       │    │  Algorithm: Calculate │
└──────────────────────┘    └──────────────────────┘
                                      │
                                      ▼
┌─── Deployment ──────────────────────────────────────┐
│                                                     │
│ ┌─Pod─┐ ┌─Pod─┐ ┌─Pod─┐ ┌─Pod─┐ ┌─Pod─┐          │
│ │ App │ │ App │ │ App │ │ App │ │ App │  ← 扩容     │
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘          │
└─────────────────────────────────────────────────────┘
```

## 🔄 HPA水平扩缩容实践

### 准备工作

```bash
# 创建实验目录
mkdir -p ~/k8s-labs/lab09/{hpa,vpa,monitoring,examples}
cd ~/k8s-labs/lab09

# 确保metrics-server运行正常
kubectl get pods -n kube-system | grep metrics-server
```

### 1. 基础HPA配置

```bash
# 创建测试应用
cat > hpa/demo-app.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: php-apache
spec:
  replicas: 1
  selector:
    matchLabels:
      app: php-apache
  template:
    metadata:
      labels:
        app: php-apache
    spec:
      containers:
      - name: php-apache
        image: k8s.gcr.io/hpa-example
        ports:
        - containerPort: 80
        resources:
          requests:
            cpu: 200m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 256Mi
---
apiVersion: v1
kind: Service
metadata:
  name: php-apache
spec:
  ports:
  - port: 80
  selector:
    app: php-apache
EOF

kubectl apply -f hpa/demo-app.yaml
```

### 2. 创建基于CPU的HPA

```bash
cat > hpa/cpu-hpa.yaml << 'EOF'
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: php-apache-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: php-apache
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 5分钟稳定窗口
      policies:
      - type: Percent
        value: 10        # 每次最多缩容10%
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60   # 1分钟稳定窗口
      policies:
      - type: Percent
        value: 50        # 每次最多扩容50%
        periodSeconds: 60
      - type: Pods
        value: 2         # 每次最多增加2个Pod
        periodSeconds: 60
      selectPolicy: Max  # 选择扩容更多的策略
EOF

kubectl apply -f hpa/cpu-hpa.yaml

# 查看HPA状态
kubectl get hpa
kubectl describe hpa php-apache-hpa
```

### 3. 创建多指标HPA

```bash
cat > hpa/multi-metric-hpa.yaml << 'EOF'
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: multi-metric-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: php-apache
  minReplicas: 2
  maxReplicas: 15
  metrics:
  # CPU指标
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  
  # 内存指标
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70
  
  # 自定义指标示例(需要custom metrics API)
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  
  # 外部指标示例
  - type: External
    external:
      metric:
        name: pubsub.googleapis.com|subscription|num_undelivered_messages
        selector:
          matchLabels:
            resource.labels.subscription_id: "my-subscription"
      target:
        type: AverageValue
        averageValue: "30"
  
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 20
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
EOF

# 注意：这个配置需要custom metrics API支持
# kubectl apply -f hpa/multi-metric-hpa.yaml
```

### 4. 压力测试HPA

```bash
# 创建负载生成器
cat > hpa/load-generator.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: load-generator
spec:
  containers:
  - name: load-generator
    image: busybox
    command: ["/bin/sh"]
    args: ["-c", "while true; do wget -q -O- http://php-apache; done"]
EOF

kubectl apply -f hpa/load-generator.yaml

# 监控HPA行为
watch kubectl get hpa,pods

# 查看详细的扩缩容事件
kubectl get events --field-selector involvedObject.name=php-apache-hpa
```

## 📈 VPA垂直扩缩容实践

### 1. 安装VPA组件

```bash
# 下载VPA
git clone https://github.com/kubernetes/autoscaler.git /tmp/autoscaler
cd /tmp/autoscaler/vertical-pod-autoscaler

# 部署VPA组件
./hack/vpa-up.sh

# 验证VPA组件
kubectl get pods -n kube-system | grep vpa
```

### 2. 创建VPA配置

```bash
cat > vpa/demo-app-vpa.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vpa-demo-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vpa-demo-app
  template:
    metadata:
      labels:
        app: vpa-demo-app
    spec:
      containers:
      - name: app
        image: nginx:alpine
        ports:
        - containerPort: 80
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi
        command: ["/bin/sh"]
        args: ["-c", "while true; do stress --cpu 1 --timeout 10s; sleep 10; done"]
---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: vpa-demo-app
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vpa-demo-app
  updatePolicy:
    updateMode: "Auto"  # Auto, Recreation, Initial, Off
  resourcePolicy:
    containerPolicies:
    - containerName: app
      minAllowed:
        cpu: 50m
        memory: 64Mi
      maxAllowed:
        cpu: 1000m
        memory: 1Gi
      controlledResources: ["cpu", "memory"]
      controlledValues: RequestsAndLimits
EOF

kubectl apply -f vpa/demo-app-vpa.yaml

# 查看VPA建议
kubectl get vpa
kubectl describe vpa vpa-demo-app
```

### 3. VPA模式对比

```bash
cat > vpa/vpa-modes.yaml << 'EOF'
# 模式1: Off - 只提供建议，不自动更新
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: vpa-off-mode
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vpa-demo-app
  updatePolicy:
    updateMode: "Off"
---
# 模式2: Initial - 只在Pod创建时设置资源
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: vpa-initial-mode
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vpa-demo-app
  updatePolicy:
    updateMode: "Initial"
---
# 模式3: Recreation - 删除Pod重建以更新资源
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: vpa-recreation-mode
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vpa-demo-app
  updatePolicy:
    updateMode: "Recreation"
  resourcePolicy:
    containerPolicies:
    - containerName: app
      controlledResources: ["cpu", "memory"]
      maxAllowed:
        cpu: "500m"
        memory: "512Mi"
---
# 模式4: Auto - 自动更新(in-place，需要支持)
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: vpa-auto-mode
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vpa-demo-app
  updatePolicy:
    updateMode: "Auto"
EOF

# kubectl apply -f vpa/vpa-modes.yaml
```

## 🔗 HPA与VPA协同使用

### 1. 组合扩缩容策略

```bash
cat > examples/combined-scaling.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web
        image: nginx:alpine
        ports:
        - containerPort: 80
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
---
# VPA负责垂直扩缩容
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: web-app-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: web
      minAllowed:
        cpu: 50m
        memory: 64Mi
      maxAllowed:
        cpu: 200m      # 限制VPA的最大资源
        memory: 256Mi   # 避免与HPA冲突
---
# HPA负责水平扩缩容
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
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # 较高的阈值，优先使用VPA
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 120  # 更长的稳定期
EOF

kubectl apply -f examples/combined-scaling.yaml
```

### 2. 防止冲突的配置

```bash
cat > examples/conflict-prevention.yaml << 'EOF'
# 策略1: 时间分离 - VPA在业务低峰期运行
apiVersion: batch/v1
kind: CronJob
metadata:
  name: vpa-controller
spec:
  schedule: "0 2 * * *"  # 每天凌晨2点
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: vpa-toggle
            image: bitnami/kubectl
            command:
            - /bin/sh
            - -c
            - |
              # 启用VPA
              kubectl patch vpa web-app-vpa -p '{"spec":{"updatePolicy":{"updateMode":"Auto"}}}'
              sleep 3600  # 运行1小时
              # 禁用VPA
              kubectl patch vpa web-app-vpa -p '{"spec":{"updatePolicy":{"updateMode":"Off"}}}'
          restartPolicy: OnFailure
---
# 策略2: 资源分离 - VPA只管理特定资源
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: memory-only-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: web
      controlledResources: ["memory"]  # 只管理内存
      controlledValues: RequestsAndLimits
EOF

# kubectl apply -f examples/conflict-prevention.yaml
```

## 📊 扩缩容监控与调优

### 1. 监控扩缩容行为

```bash
cat > monitoring/scaling-monitor.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: scaling-monitor-script
data:
  monitor.sh: |
    #!/bin/bash
    
    echo "=== Kubernetes扩缩容监控 ==="
    
    while true; do
      echo "$(date) - 当前状态:"
      
      # HPA状态
      echo "HPA状态:"
      kubectl get hpa --no-headers | while read line; do
        echo "  $line"
      done
      
      # VPA状态
      echo "VPA状态:"
      kubectl get vpa --no-headers | while read line; do
        echo "  $line"
      done
      
      # Pod资源使用
      echo "Pod资源使用:"
      kubectl top pods --no-headers | while read line; do
        echo "  $line"
      done
      
      # 最近的扩缩容事件
      echo "最近的扩缩容事件:"
      kubectl get events --field-selector reason=SuccessfulRescale --sort-by='.lastTimestamp' | tail -5
      
      echo "----------------------------------------"
      sleep 30
    done
---
apiVersion: v1
kind: Pod
metadata:
  name: scaling-monitor
spec:
  containers:
  - name: monitor
    image: bitnami/kubectl
    command: ["/bin/sh", "/scripts/monitor.sh"]
    volumeMounts:
    - name: scripts
      mountPath: /scripts
  volumes:
  - name: scripts
    configMap:
      name: scaling-monitor-script
      defaultMode: 0755
  restartPolicy: Always
EOF

kubectl apply -f monitoring/scaling-monitor.yaml
kubectl logs scaling-monitor -f
```

### 2. 扩缩容性能测试

```bash
cat > examples/scaling-test.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: load-test-config
data:
  load-test.sh: |
    #!/bin/bash
    
    echo "开始扩缩容性能测试..."
    
    # 阶段1: 轻负载
    echo "阶段1: 轻负载 (5分钟)"
    for i in {1..10}; do
      for j in {1..5}; do
        wget -q -O- http://php-apache &
      done
      sleep 30
    done
    
    # 阶段2: 中负载
    echo "阶段2: 中负载 (5分钟)"
    for i in {1..10}; do
      for j in {1..20}; do
        wget -q -O- http://php-apache &
      done
      sleep 30
    done
    
    # 阶段3: 高负载
    echo "阶段3: 高负载 (5分钟)"
    for i in {1..10}; do
      for j in {1..50}; do
        wget -q -O- http://php-apache &
      done
      sleep 30
    done
    
    # 阶段4: 负载骤降
    echo "阶段4: 负载骤降 (观察缩容行为)"
    sleep 600
    
    echo "测试完成"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: scaling-load-test
spec:
  template:
    spec:
      containers:
      - name: load-test
        image: busybox
        command: ["/bin/sh", "/scripts/load-test.sh"]
        volumeMounts:
        - name: scripts
          mountPath: /scripts
      volumes:
      - name: scripts
        configMap:
          name: load-test-config
          defaultMode: 0755
      restartPolicy: Never
  backoffLimit: 1
EOF

kubectl apply -f examples/scaling-test.yaml
```

### 3. 扩缩容指标收集

```bash
cat > monitoring/metrics-collector.yaml << 'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: metrics-collector
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: metrics-collector
rules:
- apiGroups: [""]
  resources: ["pods", "nodes"]
  verbs: ["get", "list"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["pods", "nodes"]
  verbs: ["get", "list"]
- apiGroups: ["autoscaling"]
  resources: ["horizontalpodautoscalers"]
  verbs: ["get", "list"]
- apiGroups: ["autoscaling.k8s.io"]
  resources: ["verticalpodautoscalers"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: metrics-collector
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: metrics-collector
subjects:
- kind: ServiceAccount
  name: metrics-collector
  namespace: default
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: metrics-script
data:
  collect.py: |
    #!/usr/bin/env python3
    import subprocess
    import json
    import time
    import csv
    from datetime import datetime
    
    def get_hpa_metrics():
        result = subprocess.run(['kubectl', 'get', 'hpa', '-o', 'json'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get('items', [])
        return []
    
    def get_pod_metrics():
        result = subprocess.run(['kubectl', 'top', 'pods', '--no-headers'], 
                              capture_output=True, text=True)
        pods = []
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        pods.append({
                            'name': parts[0],
                            'cpu': parts[1],
                            'memory': parts[2]
                        })
        return pods
    
    # 创建CSV文件
    with open('/data/scaling_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'hpa_name', 'current_replicas', 'desired_replicas', 
                     'target_cpu', 'current_cpu', 'pod_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        while True:
            timestamp = datetime.now().isoformat()
            hpa_data = get_hpa_metrics()
            pod_data = get_pod_metrics()
            
            for hpa in hpa_data:
                row = {
                    'timestamp': timestamp,
                    'hpa_name': hpa['metadata']['name'],
                    'current_replicas': hpa.get('status', {}).get('currentReplicas', 0),
                    'desired_replicas': hpa.get('status', {}).get('desiredReplicas', 0),
                    'target_cpu': hpa['spec']['metrics'][0]['resource']['target']['averageUtilization'],
                    'current_cpu': hpa.get('status', {}).get('currentMetrics', [{}])[0].get('resource', {}).get('current', {}).get('averageUtilization', 0),
                    'pod_count': len(pod_data)
                }
                writer.writerow(row)
                csvfile.flush()
            
            time.sleep(30)
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: metrics-collector
spec:
  replicas: 1
  selector:
    matchLabels:
      app: metrics-collector
  template:
    metadata:
      labels:
        app: metrics-collector
    spec:
      serviceAccountName: metrics-collector
      containers:
      - name: collector
        image: python:3.9-alpine
        command: ["python3", "/scripts/collect.py"]
        volumeMounts:
        - name: scripts
          mountPath: /scripts
        - name: data
          mountPath: /data
      volumes:
      - name: scripts
        configMap:
          name: metrics-script
      - name: data
        emptyDir: {}
EOF

kubectl apply -f monitoring/metrics-collector.yaml
```

## 🛠️ 实验练习

### 练习1: 创建智能扩缩容策略

```bash
cat > examples/smart-scaling.yaml << 'EOF'
# 基于时间的预测扩缩容
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: time-based-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 2
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      # 工作时间快速扩容
      - type: Percent
        value: 100
        periodSeconds: 15
      # 非工作时间保守扩容
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      # 保守的缩容策略
      - type: Percent
        value: 10
        periodSeconds: 60
EOF

kubectl apply -f examples/smart-scaling.yaml
```

## 📝 最佳实践总结

### 1. HPA配置建议

- 设置合理的CPU/内存阈值(50-70%)
- 配置适当的最小和最大副本数
- 使用稳定窗口避免频繁扩缩容
- 结合业务特点设计扩缩容行为

### 2. VPA使用指南

- 优先使用"Off"模式观察建议
- 谨慎使用"Auto"模式(可能重启Pod)
- 设置合理的资源边界
- 避免与HPA在相同资源上冲突

### 3. 监控和调优

- 持续监控扩缩容行为和效果
- 根据业务模式调整扩缩容参数
- 定期审查和优化扩缩容策略
- 建立扩缩容的告警机制

### 4. 成本优化

- 合理设置最大副本数控制成本
- 使用VPA优化资源配置减少浪费
- 结合节点自动扩缩容提高资源利用率
- 定期分析扩缩容数据优化配置

## 🎯 实验总结

通过本实验，你已经掌握了：

✅ **HPA配置**: CPU、内存、自定义指标的水平扩缩容
✅ **VPA实践**: 垂直扩缩容的不同模式和策略
✅ **组合策略**: HPA和VPA的协同使用
✅ **监控调优**: 扩缩容行为的监控和性能优化
✅ **最佳实践**: 生产环境的扩缩容配置指南

继续下一个实验：**实验10：生产环境最佳实践** 