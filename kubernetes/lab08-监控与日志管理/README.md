# 实验8：监控与日志管理

## 🎯 学习目标

通过本实验，你将：
- 掌握Kubernetes原生监控工具的使用
- 学习Prometheus+Grafana监控栈部署
- 实践日志收集和分析系统搭建
- 了解分布式追踪和APM系统
- 掌握告警规则配置和故障排查
- 学习监控数据的可视化和分析

## 📚 理论知识学习

### Kubernetes监控架构

```
┌─── Metrics Collection ────────────────────────────────┐
│                                                       │
│ ┌─ Node ─┐ ┌─ Pod ─┐ ┌─ Container ─┐ ┌─ Service ─┐   │
│ │        │ │       │ │             │ │           │   │
│ │Kubelet │→│cAdvisor│→│   Metrics   │→│Prometheus │   │
│ │        │ │       │ │   Server    │ │           │   │
│ └────────┘ └───────┘ └─────────────┘ └───────────┘   │
│                            │                          │
│ ┌─── Grafana Dashboard ────┤                          │
│ │   Visualization          │                          │
│ └──────────────────────────┘                          │
└───────────────────────────────────────────────────────┘

┌─── Logs Collection ───────────────────────────────────┐
│                                                       │
│ ┌─ Pod ─┐   ┌─ Fluentd ─┐   ┌─ Elasticsearch ─┐      │
│ │ App   │──→│  Agent    │──→│    Cluster       │      │
│ │ Logs  │   │           │   │                  │      │
│ └───────┘   └───────────┘   └──────────────────┘      │
│                                       │               │
│ ┌─── Kibana Dashboard ─────────────────┘               │
│ │   Log Analysis                                      │
│ └─────────────────────────────────────────────────────┘
```

## 📊 Kubernetes原生监控

### 准备工作

```bash
# 创建实验目录
mkdir -p ~/k8s-labs/lab08/{metrics,monitoring,logging,examples}
cd ~/k8s-labs/lab08

# 启用metrics-server（如果没有）
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### 1. 资源监控基础

```bash
# 查看节点资源使用情况
kubectl top nodes

# 查看Pod资源使用情况
kubectl top pods --all-namespaces

# 查看特定命名空间的资源使用
kubectl top pods -n kube-system

# 按CPU使用排序
kubectl top pods --all-namespaces --sort-by=cpu

# 按内存使用排序
kubectl top pods --all-namespaces --sort-by=memory
```

### 2. 创建资源监控示例

```bash
cat > metrics/resource-monitor.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: resource-demo
spec:
  replicas: 2
  selector:
    matchLabels:
      app: resource-demo
  template:
    metadata:
      labels:
        app: resource-demo
    spec:
      containers:
      - name: cpu-stress
        image: progrium/stress
        command: ["stress"]
        args: ["--cpu", "1", "--timeout", "3600s"]
        resources:
          requests:
            memory: "100Mi"
            cpu: "100m"
          limits:
            memory: "200Mi"
            cpu: "500m"
      
      - name: memory-stress
        image: progrium/stress
        command: ["stress"]
        args: ["--vm", "1", "--vm-bytes", "50M", "--timeout", "3600s"]
        resources:
          requests:
            memory: "50Mi"
            cpu: "50m"
          limits:
            memory: "100Mi"
            cpu: "200m"
EOF

kubectl apply -f metrics/resource-monitor.yaml

# 监控资源使用情况
watch kubectl top pods -l app=resource-demo
```

## 🔍 Prometheus监控系统

### 1. 部署Prometheus

```bash
cat > monitoring/prometheus-config.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "alerting_rules.yml"
    
    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager:9093
    
    scrape_configs:
      # Kubernetes API Server
      - job_name: 'kubernetes-apiservers'
        kubernetes_sd_configs:
        - role: endpoints
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
        - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
          action: keep
          regex: default;kubernetes;https
      
      # Kubernetes Nodes
      - job_name: 'kubernetes-nodes'
        kubernetes_sd_configs:
        - role: node
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
        - action: labelmap
          regex: __meta_kubernetes_node_label_(.+)
        - target_label: __address__
          replacement: kubernetes.default.svc:443
        - source_labels: [__meta_kubernetes_node_name]
          regex: (.+)
          target_label: __metrics_path__
          replacement: /api/v1/nodes/${1}/proxy/metrics
      
      # Kubernetes Pods
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
        - role: pod
        relabel_configs:
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
          action: keep
          regex: true
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
          action: replace
          target_label: __metrics_path__
          regex: (.+)
        - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
          action: replace
          regex: ([^:]+)(?::\d+)?;(\d+)
          replacement: $1:$2
          target_label: __address__
        - action: labelmap
          regex: __meta_kubernetes_pod_label_(.+)
        - source_labels: [__meta_kubernetes_namespace]
          action: replace
          target_label: kubernetes_namespace
        - source_labels: [__meta_kubernetes_pod_name]
          action: replace
          target_label: kubernetes_pod_name
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      serviceAccountName: prometheus
      containers:
      - name: prometheus
        image: prom/prometheus:v2.40.0
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus/
        - name: storage
          mountPath: /prometheus
        command:
        - '/bin/prometheus'
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus'
        - '--web.console.libraries=/etc/prometheus/console_libraries'
        - '--web.console.templates=/etc/prometheus/consoles'
        - '--storage.tsdb.retention.time=15d'
        - '--web.enable-lifecycle'
        resources:
          requests:
            memory: "400Mi"
            cpu: "200m"
          limits:
            memory: "800Mi"
            cpu: "500m"
      
      volumes:
      - name: config
        configMap:
          name: prometheus-config
      - name: storage
        emptyDir: {}
EOF

# 创建命名空间和ServiceAccount
kubectl create namespace monitoring
kubectl create serviceaccount prometheus -n monitoring

# 创建RBAC权限
cat > monitoring/prometheus-rbac.yaml << 'EOF'
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prometheus
rules:
- apiGroups: [""]
  resources:
  - nodes
  - nodes/proxy
  - services
  - endpoints
  - pods
  verbs: ["get", "list", "watch"]
- apiGroups:
  - extensions
  resources:
  - ingresses
  verbs: ["get", "list", "watch"]
- nonResourceURLs: ["/metrics"]
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: prometheus
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: prometheus
subjects:
- kind: ServiceAccount
  name: prometheus
  namespace: monitoring
EOF

kubectl apply -f monitoring/prometheus-rbac.yaml
kubectl apply -f monitoring/prometheus-config.yaml

# 创建Service
cat > monitoring/prometheus-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: monitoring
spec:
  ports:
  - port: 9090
    targetPort: 9090
  selector:
    app: prometheus
  type: NodePort
EOF

kubectl apply -f monitoring/prometheus-service.yaml
```

### 2. 部署Grafana

```bash
cat > monitoring/grafana.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:9.0.0
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: admin123
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
        resources:
          requests:
            memory: "200Mi"
            cpu: "100m"
          limits:
            memory: "400Mi"
            cpu: "300m"
      
      volumes:
      - name: grafana-storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: monitoring
spec:
  ports:
  - port: 3000
    targetPort: 3000
  selector:
    app: grafana
  type: NodePort
EOF

kubectl apply -f monitoring/grafana.yaml
```

### 3. 应用监控示例

```bash
cat > examples/monitored-app.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sample-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sample-app
  template:
    metadata:
      labels:
        app: sample-app
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: app
        image: nginx:alpine
        ports:
        - containerPort: 80
        - containerPort: 8080
        command: ["/bin/sh"]
        args: ["-c", "nginx && while true; do echo 'sample_metric{job=\"sample-app\"} 1' > /tmp/metrics; nc -l -p 8080 < /tmp/metrics; done"]
        resources:
          requests:
            memory: "50Mi"
            cpu: "50m"
          limits:
            memory: "100Mi"
            cpu: "100m"
---
apiVersion: v1
kind: Service
metadata:
  name: sample-app
spec:
  ports:
  - port: 80
    targetPort: 80
    name: http
  - port: 8080
    targetPort: 8080
    name: metrics
  selector:
    app: sample-app
EOF

kubectl apply -f examples/monitored-app.yaml
```

## 📝 日志管理系统

### 1. 日志收集配置

```bash
cat > logging/fluentd-config.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: kube-system
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      read_from_head true
      <parse>
        @type json
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>
    
    <match **>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name fluentd
      type_name fluentd
    </match>
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      serviceAccountName: fluentd
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch
        env:
        - name: FLUENTD_SYSTEMD_CONF
          value: disable
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
        - name: config
          mountPath: /fluentd/etc/
        resources:
          requests:
            memory: "200Mi"
            cpu: "100m"
          limits:
            memory: "400Mi"
            cpu: "200m"
      
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
      - name: config
        configMap:
          name: fluentd-config
EOF

# 创建ServiceAccount和RBAC
cat > logging/fluentd-rbac.yaml << 'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: fluentd
  namespace: kube-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: fluentd
rules:
- apiGroups: [""]
  resources:
  - pods
  - namespaces
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: fluentd
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: fluentd
subjects:
- kind: ServiceAccount
  name: fluentd
  namespace: kube-system
EOF

kubectl apply -f logging/fluentd-rbac.yaml
# kubectl apply -f logging/fluentd-config.yaml  # 需要先部署Elasticsearch
```

### 2. 简化日志查看

```bash
cat > logging/log-viewer.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: log-demo-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: log-demo
  template:
    metadata:
      labels:
        app: log-demo
    spec:
      containers:
      - name: logger
        image: alpine
        command: ["/bin/sh"]
        args: ["-c", "while true; do echo \"$(date): Log message from $HOSTNAME - Level: INFO\"; sleep 5; echo \"$(date): Error message from $HOSTNAME - Level: ERROR\" >&2; sleep 5; done"]
        env:
        - name: LOG_LEVEL
          value: "INFO"
EOF

kubectl apply -f logging/log-viewer.yaml

# 查看日志的不同方式
kubectl logs deployment/log-demo-app
kubectl logs -l app=log-demo --tail=10
kubectl logs -l app=log-demo -f  # 实时跟踪
kubectl logs -l app=log-demo --since=1h  # 最近1小时的日志
```

### 3. 日志聚合示例

```bash
cat > logging/centralized-logging.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: log-config
data:
  filebeat.yml: |
    filebeat.inputs:
    - type: container
      paths:
        - /var/log/containers/*.log
      processors:
        - add_kubernetes_metadata:
            host: ${NODE_NAME}
            matchers:
            - logs_path:
                logs_path: "/var/log/containers/"
    
    output.console:
      pretty: true
    
    logging.level: info
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: filebeat
spec:
  selector:
    matchLabels:
      app: filebeat
  template:
    metadata:
      labels:
        app: filebeat
    spec:
      containers:
      - name: filebeat
        image: elastic/filebeat:7.15.0
        env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        volumeMounts:
        - name: config
          mountPath: /usr/share/filebeat/filebeat.yml
          subPath: filebeat.yml
        - name: varlog
          mountPath: /var/log
          readOnly: true
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
      
      volumes:
      - name: config
        configMap:
          name: log-config
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
EOF

kubectl apply -f logging/centralized-logging.yaml
```

## 🚨 告警系统配置

### 1. AlertManager配置

```bash
cat > monitoring/alertmanager.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: monitoring
data:
  alertmanager.yml: |
    global:
      smtp_smarthost: 'localhost:587'
      smtp_from: 'alerts@company.com'
    
    route:
      group_by: ['alertname']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'web.hook'
    
    receivers:
    - name: 'web.hook'
      webhook_configs:
      - url: 'http://webhook.example.com/'
      email_configs:
      - to: 'admin@company.com'
        subject: 'K8s Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alertmanager
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: alertmanager
  template:
    metadata:
      labels:
        app: alertmanager
    spec:
      containers:
      - name: alertmanager
        image: prom/alertmanager:v0.24.0
        ports:
        - containerPort: 9093
        volumeMounts:
        - name: config
          mountPath: /etc/alertmanager/
        command:
        - '/bin/alertmanager'
        - '--config.file=/etc/alertmanager/alertmanager.yml'
        - '--storage.path=/alertmanager'
        resources:
          requests:
            memory: "100Mi"
            cpu: "50m"
          limits:
            memory: "200Mi"
            cpu: "100m"
      
      volumes:
      - name: config
        configMap:
          name: alertmanager-config
EOF

kubectl apply -f monitoring/alertmanager.yaml
```

### 2. 告警规则

```bash
cat > monitoring/alert-rules.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: monitoring
data:
  alerting_rules.yml: |
    groups:
    - name: kubernetes-alerts
      rules:
      - alert: PodMemoryUsageHigh
        expr: container_memory_usage_bytes / container_spec_memory_limit_bytes * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Pod memory usage is above 80%"
          description: "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} has been using more than 80% of its memory limit for more than 5 minutes."
      
      - alert: PodCPUUsageHigh
        expr: rate(container_cpu_usage_seconds_total[5m]) * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Pod CPU usage is above 80%"
          description: "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} has been using more than 80% CPU for more than 5 minutes."
      
      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Pod is crash looping"
          description: "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} is restarting frequently."
      
      - alert: NodeNotReady
        expr: kube_node_status_condition{condition="Ready",status="true"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Node is not ready"
          description: "Node {{ $labels.node }} has been not ready for more than 5 minutes."
EOF

kubectl apply -f monitoring/alert-rules.yaml
```

## 🛠️ 实验练习

### 练习1：自定义监控仪表板

```bash
# 创建自定义指标示例
cat > examples/custom-metrics.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: metrics-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: metrics-app
  template:
    metadata:
      labels:
        app: metrics-app
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: app
        image: alpine
        ports:
        - containerPort: 8080
        command: ["/bin/sh"]
        args: ["-c", "while true; do echo \"# HELP http_requests_total Total HTTP requests\n# TYPE http_requests_total counter\nhttp_requests_total{method=\\\"GET\\\",status=\\\"200\\\"} $((RANDOM % 1000))\nhttp_requests_total{method=\\\"POST\\\",status=\\\"200\\\"} $((RANDOM % 500))\nhttp_requests_total{method=\\\"GET\\\",status=\\\"404\\\"} $((RANDOM % 100))\" > /tmp/metrics; nc -l -p 8080 < /tmp/metrics; done"]
EOF

kubectl apply -f examples/custom-metrics.yaml
```

### 练习2：日志分析脚本

```bash
cat > logging/log-analysis.sh << 'EOF'
#!/bin/bash

echo "=== Kubernetes日志分析 ==="

# 统计错误日志
echo "1. 错误日志统计："
kubectl logs --all-containers --all-namespaces --since=1h | grep -i error | wc -l

# 查找最频繁的错误
echo "2. 最频繁的错误："
kubectl logs --all-containers --all-namespaces --since=1h | grep -i error | sort | uniq -c | sort -nr | head -5

# 按命名空间统计日志量
echo "3. 各命名空间日志量："
for ns in $(kubectl get namespaces -o jsonpath='{.items[*].metadata.name}'); do
  count=$(kubectl logs --all-containers -n $ns --since=1h | wc -l)
  echo "$ns: $count lines"
done

# 查找重启频繁的Pod
echo "4. 重启频繁的Pod："
kubectl get pods --all-namespaces --field-selector=status.phase=Running -o custom-columns="NAMESPACE:.metadata.namespace,NAME:.metadata.name,RESTARTS:.status.containerStatuses[*].restartCount" | grep -v '<none>' | sort -k3 -nr | head -10
EOF

chmod +x logging/log-analysis.sh
./logging/log-analysis.sh
```

## 📝 最佳实践总结

### 1. 监控策略

- **多层监控**：基础设施、应用、业务指标
- **告警分级**：关键、警告、信息三个级别
- **历史数据**：保留足够的历史数据用于趋势分析
- **容量规划**：基于监控数据进行容量规划

### 2. 日志管理

- **结构化日志**：使用JSON格式便于检索分析
- **日志分级**：合理设置日志级别
- **日志轮转**：定期清理历史日志文件
- **安全考虑**：避免在日志中记录敏感信息

### 3. 性能调优

- 合理设置抓取间隔和保留时间
- 使用标签选择器减少不必要的监控
- 配置合适的存储和计算资源
- 定期检查和优化查询性能

## 🎯 实验总结

通过本实验，你已经掌握了：

✅ **原生监控**: kubectl top和metrics-server
✅ **Prometheus**: 指标收集和存储
✅ **Grafana**: 数据可视化和仪表板
✅ **日志收集**: Fluentd/Filebeat日志聚合
✅ **告警系统**: AlertManager告警配置
✅ **故障排查**: 基于监控数据的问题诊断

继续下一个实验：**实验9：自动扩缩容HPA/VPA** 