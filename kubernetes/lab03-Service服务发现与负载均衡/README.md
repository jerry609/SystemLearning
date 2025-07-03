# 实验3：Service服务发现与负载均衡

## 🎯 学习目标

通过本实验，你将：
- 理解Service的核心概念和作用
- 掌握四种Service类型及其使用场景
- 学习服务发现机制和DNS解析
- 实践负载均衡和会话亲和性
- 了解Endpoint和EndpointSlice的工作原理
- 掌握Service故障排查技巧

## 📚 理论知识学习

### Service核心概念

Service是Kubernetes中的抽象层，它定义了一组Pod的逻辑集合和访问策略。Service解决了Pod IP地址动态变化的问题，为应用提供稳定的网络入口。

#### 为什么需要Service？

**Pod的局限性**：
- Pod IP是临时的，重启后会变化
- Pod可能因为故障、扩缩容而创建或销毁
- 客户端无法直接知道Pod的IP地址变化

**Service的价值**：
- 提供稳定的虚拟IP（ClusterIP）
- 实现负载均衡和流量分发
- 支持服务发现机制
- 抽象底层Pod的变化

### Service工作原理

```
┌─────────────────── Service ──────────────────┐
│  Virtual IP: 10.96.0.100:80                 │
│  ┌─────────────── kube-proxy ───────────────┐│
│  │        负载均衡算法                        ││
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐   ││
│  │   │  Pod1   │ │  Pod2   │ │  Pod3   │   ││
│  │   │10.244.1 │ │10.244.2 │ │10.244.3 │   ││
│  │   └─────────┘ └─────────┘ └─────────┘   ││
│  └───────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
```

### Service类型详解

| 类型 | 描述 | 使用场景 | 访问方式 |
|------|------|----------|----------|
| **ClusterIP** | 集群内部虚拟IP | 内部服务间通信 | 仅集群内访问 |
| **NodePort** | 在每个节点上开放端口 | 外部访问简单服务 | `<NodeIP>:<NodePort>` |
| **LoadBalancer** | 云平台负载均衡器 | 生产环境外部访问 | 云平台分配的外部IP |
| **ExternalName** | DNS CNAME记录 | 外部服务映射 | DNS解析到外部地址 |

## 🔧 Service创建与配置

### 准备工作：创建测试Pod

```bash
# 创建实验目录
mkdir -p ~/k8s-labs/lab03

# 创建Deployment作为Service的后端
cat > ~/k8s-labs/lab03/web-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-deployment
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
        version: v1
    spec:
      containers:
      - name: web
        image: nginx:alpine
        ports:
        - containerPort: 80
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        # 自定义首页显示Pod信息
        lifecycle:
          postStart:
            exec:
              command:
              - /bin/sh
              - -c
              - |
                echo "<h1>Pod: $POD_NAME</h1>" > /usr/share/nginx/html/index.html
                echo "<p>IP: $POD_IP</p>" >> /usr/share/nginx/html/index.html
                echo "<p>Version: v1</p>" >> /usr/share/nginx/html/index.html
                echo "<p>Timestamp: $(date)</p>" >> /usr/share/nginx/html/index.html
EOF

# 部署测试应用
kubectl apply -f ~/k8s-labs/lab03/web-deployment.yaml

# 验证Pod运行状态
kubectl get pods -l app=web
```

### 1. ClusterIP Service（默认类型）

#### 创建ClusterIP Service
```bash
cat > ~/k8s-labs/lab03/clusterip-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: web-clusterip
  labels:
    app: web
spec:
  type: ClusterIP
  selector:
    app: web
  ports:
  - protocol: TCP
    port: 80          # Service端口
    targetPort: 80    # Pod端口
    name: http
EOF

# 应用Service配置
kubectl apply -f ~/k8s-labs/lab03/clusterip-service.yaml

# 查看Service信息
kubectl get service web-clusterip
kubectl describe service web-clusterip
```

#### 测试ClusterIP Service
```bash
# 方法1：使用kubectl proxy
kubectl proxy --port=8080 &
curl http://localhost:8080/api/v1/namespaces/default/services/web-clusterip/proxy/

# 方法2：创建测试Pod
kubectl run test-pod --image=curlimages/curl -it --rm -- sh
# 在Pod内执行：
# curl web-clusterip
# nslookup web-clusterip

# 方法3：端口转发
kubectl port-forward service/web-clusterip 8081:80
# 在浏览器访问 http://localhost:8081
```

### 2. NodePort Service

#### 创建NodePort Service
```bash
cat > ~/k8s-labs/lab03/nodeport-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: web-nodeport
  labels:
    app: web
spec:
  type: NodePort
  selector:
    app: web
  ports:
  - protocol: TCP
    port: 80          # Service端口
    targetPort: 80    # Pod端口
    nodePort: 30080   # 节点端口（可选，不指定则自动分配）
    name: http
EOF

# 应用NodePort Service
kubectl apply -f ~/k8s-labs/lab03/nodeport-service.yaml

# 查看Service信息
kubectl get service web-nodeport
```

#### 测试NodePort Service
```bash
# 获取节点IP
kubectl get nodes -o wide

# 使用NodePort访问
# Docker Desktop: http://localhost:30080
# Minikube: 
minikube ip  # 获取Minikube IP
# 然后访问 http://<minikube-ip>:30080

# 测试负载均衡
for i in {1..10}; do
  curl http://localhost:30080 2>/dev/null | grep "Pod:"
done
```

### 3. LoadBalancer Service

#### 创建LoadBalancer Service
```bash
cat > ~/k8s-labs/lab03/loadbalancer-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: web-loadbalancer
  labels:
    app: web
spec:
  type: LoadBalancer
  selector:
    app: web
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
    name: http
EOF

# 应用LoadBalancer Service
kubectl apply -f ~/k8s-labs/lab03/loadbalancer-service.yaml

# 查看Service状态
kubectl get service web-loadbalancer
```

**注意**：在本地环境（Docker Desktop、Minikube）中，LoadBalancer类型会显示`<pending>`状态，因为没有云平台提供外部负载均衡器。

#### Minikube中启用LoadBalancer
```bash
# 在Minikube中启用tunnel来模拟LoadBalancer
minikube tunnel

# 在另一个终端查看Service获得的外部IP
kubectl get service web-loadbalancer
```

### 4. ExternalName Service

#### 创建ExternalName Service
```bash
cat > ~/k8s-labs/lab03/externalname-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: external-api
spec:
  type: ExternalName
  externalName: httpbin.org
  ports:
  - port: 80
    protocol: TCP
EOF

# 应用ExternalName Service
kubectl apply -f ~/k8s-labs/lab03/externalname-service.yaml

# 测试外部服务映射
kubectl run test-external --image=curlimages/curl -it --rm -- sh
# 在Pod内执行：
# curl external-api/json
# nslookup external-api
```

## 🔍 服务发现机制

### DNS服务发现

Kubernetes为每个Service自动创建DNS记录：

#### DNS命名规则
```
<service-name>.<namespace>.svc.cluster.local
```

#### 测试DNS解析
```bash
# 创建DNS测试Pod
cat > ~/k8s-labs/lab03/dns-test-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: dns-test
spec:
  containers:
  - name: dns-test
    image: busybox
    command: ["sleep", "3600"]
EOF

kubectl apply -f ~/k8s-labs/lab03/dns-test-pod.yaml

# 进入Pod测试DNS解析
kubectl exec -it dns-test -- sh

# 在Pod内执行以下命令：
# 短名称解析（同命名空间）
nslookup web-clusterip

# 完整域名解析
nslookup web-clusterip.default.svc.cluster.local

# 查看DNS配置
cat /etc/resolv.conf

# 查看所有Service的DNS记录
nslookup kubernetes
```

### 环境变量服务发现

Kubernetes还会为每个Service创建环境变量：

```bash
# 创建Pod查看环境变量
kubectl run env-test --image=busybox --rm -it -- env | grep -E "WEB_CLUSTERIP|KUBERNETES"

# 格式说明：
# {SERVICE_NAME}_SERVICE_HOST=<ClusterIP>
# {SERVICE_NAME}_SERVICE_PORT=<Port>
# {SERVICE_NAME}_PORT=<tcp://ClusterIP:Port>
```

## ⚖️ 负载均衡配置

### 负载均衡算法

Kubernetes Service支持两种负载均衡模式：

#### 1. Round Robin（默认）
```bash
# 测试轮询负载均衡
for i in {1..10}; do
  echo "Request $i:"
  kubectl exec test-pod -- curl -s web-clusterip | grep "Pod:"
  sleep 1
done
```

#### 2. Session Affinity（会话亲和性）
```bash
cat > ~/k8s-labs/lab03/session-affinity-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: web-session-affinity
spec:
  type: ClusterIP
  selector:
    app: web
  sessionAffinity: ClientIP  # 基于客户端IP的会话保持
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800  # 会话保持时间：3小时
  ports:
  - port: 80
    targetPort: 80
EOF

kubectl apply -f ~/k8s-labs/lab03/session-affinity-service.yaml

# 测试会话亲和性
kubectl run session-test --image=curlimages/curl -it --rm -- sh
# 在Pod内多次执行：curl web-session-affinity
# 应该始终访问到同一个Pod
```

### 端口配置

#### 多端口Service
```bash
cat > ~/k8s-labs/lab03/multi-port-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: multi-port-service
spec:
  selector:
    app: web
  ports:
  - name: http
    port: 80
    targetPort: 80
    protocol: TCP
  - name: https
    port: 443
    targetPort: 443
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
EOF
```

#### 命名端口
```bash
cat > ~/k8s-labs/lab03/named-port-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: named-port-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: named-port-app
  template:
    metadata:
      labels:
        app: named-port-app
    spec:
      containers:
      - name: app
        image: nginx:alpine
        ports:
        - containerPort: 80
          name: web-port      # 命名端口
        - containerPort: 8080
          name: admin-port    # 管理端口
---
apiVersion: v1
kind: Service
metadata:
  name: named-port-service
spec:
  selector:
    app: named-port-app
  ports:
  - port: 80
    targetPort: web-port    # 使用端口名称
    name: web
  - port: 8080
    targetPort: admin-port  # 使用端口名称
    name: admin
EOF
```

## 🔍 Endpoint和EndpointSlice

### 理解Endpoint

Endpoint对象存储了Service匹配的Pod的IP地址和端口信息。

```bash
# 查看Service对应的Endpoint
kubectl get endpoints web-clusterip

# 查看详细信息
kubectl describe endpoints web-clusterip

# 以YAML格式查看
kubectl get endpoints web-clusterip -o yaml
```

### 手动管理Endpoint

有时需要为Service手动指定Endpoint（例如，访问外部服务）：

```bash
cat > ~/k8s-labs/lab03/manual-endpoint.yaml << 'EOF'
# Service without selector
apiVersion: v1
kind: Service
metadata:
  name: external-database
spec:
  ports:
  - port: 3306
    targetPort: 3306
---
# Manual Endpoint
apiVersion: v1
kind: Endpoints
metadata:
  name: external-database  # 必须与Service名称一致
subsets:
- addresses:
  - ip: 192.168.1.100     # 外部数据库IP
  - ip: 192.168.1.101     # 外部数据库IP（备用）
  ports:
  - port: 3306
    protocol: TCP
EOF

kubectl apply -f ~/k8s-labs/lab03/manual-endpoint.yaml
kubectl get endpoints external-database
```

### EndpointSlice（Kubernetes 1.17+）

EndpointSlice是Endpoint的改进版本，提供更好的性能和扩展性：

```bash
# 查看EndpointSlice
kubectl get endpointslices

# 查看特定Service的EndpointSlice
kubectl get endpointslices -l kubernetes.io/service-name=web-clusterip

# 详细信息
kubectl describe endpointslice <endpointslice-name>
```

## 🛠️ 实验练习

### 练习1：Service类型对比实验

创建一个comprehensive测试脚本：

```bash
cat > ~/k8s-labs/lab03/service-comparison.sh << 'EOF'
#!/bin/bash

echo "=== Service Types Comparison ==="

# 1. 查看所有Service
echo "1. All Services:"
kubectl get services

echo -e "\n2. ClusterIP Service Details:"
kubectl describe service web-clusterip

echo -e "\n3. NodePort Service Details:"
kubectl describe service web-nodeport

echo -e "\n4. LoadBalancer Service Details:"
kubectl describe service web-loadbalancer

echo -e "\n5. Endpoints:"
kubectl get endpoints

echo -e "\n6. DNS Resolution Test:"
kubectl exec dns-test -- nslookup web-clusterip

echo -e "\n7. Load Balancing Test (ClusterIP):"
for i in {1..5}; do
  echo "Request $i:"
  kubectl exec dns-test -- wget -qO- web-clusterip 2>/dev/null | grep "Pod:" || echo "Failed"
done
EOF

chmod +x ~/k8s-labs/lab03/service-comparison.sh
./~/k8s-labs/lab03/service-comparison.sh
```

### 练习2：服务网格模拟

创建多个相互通信的服务：

```bash
cat > ~/k8s-labs/lab03/service-mesh-demo.yaml << 'EOF'
# Frontend Service
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
      - name: frontend
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
# Backend Service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: httpd:alpine
        ports:
        - containerPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: backend
  ports:
  - port: 80
    targetPort: 80
---
# Database Service (External)
apiVersion: v1
kind: Service
metadata:
  name: database-service
spec:
  type: ExternalName
  externalName: db.example.com
  ports:
  - port: 3306
EOF

kubectl apply -f ~/k8s-labs/lab03/service-mesh-demo.yaml
```

### 练习3：故障转移测试

```bash
# 1. 创建测试脚本
cat > ~/k8s-labs/lab03/failover-test.sh << 'EOF'
#!/bin/bash

echo "Starting failover test..."

# 持续访问Service
while true; do
  RESULT=$(kubectl exec dns-test -- wget -qO- web-clusterip 2>/dev/null | grep "Pod:" || echo "FAILED")
  echo "$(date): $RESULT"
  sleep 2
done
EOF

chmod +x ~/k8s-labs/lab03/failover-test.sh

# 2. 在后台运行测试
./~/k8s-labs/lab03/failover-test.sh &
TEST_PID=$!

# 3. 模拟Pod故障
kubectl get pods -l app=web
kubectl delete pod <pod-name>  # 删除一个Pod

# 4. 观察Service自动恢复

# 5. 停止测试
kill $TEST_PID
```

## 🧪 进阶实验

### 实验1：Headless Service

Headless Service不分配ClusterIP，直接返回Pod IP地址：

```bash
cat > ~/k8s-labs/lab03/headless-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: web-headless
spec:
  clusterIP: None          # Headless Service
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 80
EOF

kubectl apply -f ~/k8s-labs/lab03/headless-service.yaml

# 测试Headless Service DNS解析
kubectl exec dns-test -- nslookup web-headless
# 应该返回所有Pod的IP地址，而不是单一的ClusterIP
```

### 实验2：Service拓扑感知

配置Service只将流量路由到本地节点的Pod：

```bash
cat > ~/k8s-labs/lab03/topology-aware-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: web-topology
  annotations:
    service.kubernetes.io/topology-aware-hints: auto
spec:
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
  topologyKeys:
  - "kubernetes.io/hostname"
  - "*"
EOF
```

### 实验3：Service Quality of Service

为不同的Service配置不同的QoS：

```bash
cat > ~/k8s-labs/lab03/qos-service.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: high-priority-service
  annotations:
    service.kubernetes.io/load-balancer-class: "high-performance"
spec:
  selector:
    app: web
    tier: premium
  ports:
  - port: 80
    targetPort: 80
  type: LoadBalancer
EOF
```

## 🐛 故障排查指南

### 常见Service问题

#### 1. Service无法访问
```bash
# 检查Service配置
kubectl describe service <service-name>

# 检查Endpoint
kubectl get endpoints <service-name>

# 如果Endpoint为空，检查：
# - Service的selector是否匹配Pod的labels
kubectl get pods --show-labels
kubectl get service <service-name> -o yaml | grep -A 5 selector

# - Pod是否处于Ready状态
kubectl get pods -l <label-selector>
```

#### 2. 负载均衡不工作
```bash
# 检查kube-proxy状态
kubectl get pods -n kube-system -l k8s-app=kube-proxy

# 查看kube-proxy日志
kubectl logs -n kube-system <kube-proxy-pod>

# 检查iptables规则（在节点上）
sudo iptables -t nat -L | grep <service-name>
```

#### 3. DNS解析失败
```bash
# 检查CoreDNS状态
kubectl get pods -n kube-system -l k8s-app=kube-dns

# 测试DNS解析
kubectl exec dns-test -- nslookup kubernetes.default.svc.cluster.local

# 检查DNS配置
kubectl exec dns-test -- cat /etc/resolv.conf
```

#### 4. NodePort无法访问
```bash
# 检查节点防火墙设置
# 确保NodePort端口(30000-32767)没有被阻止

# 检查Service的NodePort
kubectl get service <service-name> -o wide

# 在节点上测试本地访问
curl localhost:<nodeport>
```

### 调试工具集

```bash
cat > ~/k8s-labs/lab03/service-debug.sh << 'EOF'
#!/bin/bash

SERVICE_NAME=$1
if [ -z "$SERVICE_NAME" ]; then
    echo "Usage: $0 <service-name>"
    exit 1
fi

echo "=== Service Debug Information ==="
echo "Service: $SERVICE_NAME"

echo -e "\n1. Service Details:"
kubectl describe service $SERVICE_NAME

echo -e "\n2. Endpoints:"
kubectl get endpoints $SERVICE_NAME

echo -e "\n3. Related Pods:"
SELECTOR=$(kubectl get service $SERVICE_NAME -o jsonpath='{.spec.selector}' | tr -d '{}' | tr ',' ' ')
if [ ! -z "$SELECTOR" ]; then
    kubectl get pods -l "$SELECTOR"
else
    echo "No selector found (possibly ExternalName service)"
fi

echo -e "\n4. EndpointSlices:"
kubectl get endpointslices -l kubernetes.io/service-name=$SERVICE_NAME

echo -e "\n5. Service Events:"
kubectl get events --field-selector involvedObject.name=$SERVICE_NAME

echo -e "\n6. DNS Test:"
kubectl run dns-test-temp --image=busybox --rm -it --restart=Never -- nslookup $SERVICE_NAME 2>/dev/null || echo "DNS test failed"
EOF

chmod +x ~/k8s-labs/lab03/service-debug.sh
```

## 💡 最佳实践

### 1. Service设计原则
- **明确端口命名**：为多端口Service命名端口
- **合理选择类型**：根据访问需求选择合适的Service类型
- **健康检查配置**：确保Pod有正确的健康检查
- **标签管理**：使用清晰的标签选择器

### 2. 性能优化
- **会话亲和性**：根据应用需求选择是否启用
- **拓扑感知**：利用拓扑感知减少网络延迟
- **连接池**：应用层实现连接池复用

### 3. 安全考虑
- **网络策略**：使用NetworkPolicy限制Service访问
- **TLS终止**：在Service层或Ingress层处理TLS
- **最小权限**：只暴露必要的端口和服务

### 4. 监控和告警
```yaml
# Service监控标签
metadata:
  labels:
    monitoring: "enabled"
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
    prometheus.io/path: "/metrics"
```

## 📝 学习检查

完成本实验后，你应该能够回答：

1. **概念理解**：
   - Service的四种类型及其使用场景？
   - Endpoint和Service的关系？
   - DNS服务发现如何工作？

2. **操作技能**：
   - 如何创建不同类型的Service？
   - 如何配置会话亲和性？
   - 如何排查Service访问失败的问题？

3. **实际应用**：
   - 什么时候使用Headless Service？
   - 如何实现外部服务的映射？
   - 如何设计微服务间的通信？

## 🔗 延伸学习

- 学习Ingress控制器和高级路由
- 了解Service Mesh（Istio、Linkerd）
- 探索跨集群服务发现
- 研究服务网格的可观测性

## ⏭️ 下一步

完成本实验后，继续学习：
- **实验4**：Deployment应用部署与更新 - 学习应用的生命周期管理
- 探索滚动更新、蓝绿部署等高级部署策略

---

**恭喜完成Service实验！** 🎉
你现在已经掌握了Kubernetes服务发现和负载均衡的核心概念，可以继续探索应用部署和管理。 