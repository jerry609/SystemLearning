# 实验10：生产环境最佳实践

## 🎯 学习目标

通过本实验，你将：
- 掌握Kubernetes生产环境部署的完整流程
- 学习CI/CD管道与GitOps最佳实践
- 实践高可用集群架构设计
- 掌握性能调优和资源优化策略
- 了解安全加固和合规性要求
- 学习故障恢复和灾难备份方案

## 📚 理论知识学习

### 生产环境架构设计

```
┌─── 生产环境架构 ───────────────────────────────────────┐
│                                                       │
│ ┌─ 负载均衡层 ─┐  ┌─ 网关层 ─┐  ┌─ 应用层 ─┐         │
│ │  CloudFlare  │→│ Ingress  │→│ Services  │         │
│ │     /        │  │ Controller│  │    &     │         │
│ │   ALB/NLB    │  │    +     │  │   Pods   │         │
│ └──────────────┘  │   WAF    │  └──────────┘         │
│                   └──────────┘                        │
│                                                       │
│ ┌─ 数据层 ─┐    ┌─ 缓存层 ─┐    ┌─ 消息队列 ─┐       │
│ │ Database │    │  Redis   │    │   Kafka    │       │
│ │ Cluster  │    │ Cluster  │    │  /RabbitMQ │       │
│ └──────────┘    └──────────┘    └────────────┘       │
│                                                       │
│ ┌─ 监控层 ─┐    ┌─ 日志层 ─┐    ┌─ 安全层 ─┐         │
│ │Prometheus│    │   ELK    │    │   RBAC   │         │
│ │ Grafana  │    │  Stack   │    │ Network  │         │
│ │ AlertMgr │    │          │    │ Policies │         │
│ └──────────┘    └──────────┘    └──────────┘         │
└───────────────────────────────────────────────────────┘
```

### 生产环境清单

| 类别 | 组件 | 目的 | 优先级 |
|------|------|------|--------|
| **高可用** | 多Master节点 | 控制平面高可用 | 🔴 必须 |
| **网络** | CNI插件 | 网络通信 | 🔴 必须 |
| **存储** | 持久化存储 | 数据持久化 | 🔴 必须 |
| **监控** | Prometheus+Grafana | 系统监控 | 🟡 重要 |
| **日志** | ELK/EFK Stack | 日志聚合 | 🟡 重要 |
| **安全** | RBAC+NetworkPolicy | 访问控制 | 🔴 必须 |
| **备份** | Velero | 灾难恢复 | 🟡 重要 |

## 🚀 CI/CD与GitOps实践

### 准备工作

```bash
# 创建实验目录
mkdir -p ~/k8s-labs/lab10/{ci-cd,gitops,ha,monitoring,security,backup}
cd ~/k8s-labs/lab10

# 创建生产环境命名空间
kubectl create namespace production
kubectl create namespace staging
kubectl create namespace monitoring
```

### 1. GitOps工作流设计

```bash
cat > gitops/workflow-overview.md << 'EOF'
# GitOps工作流程

## 1. 代码提交流程
```
开发者提交代码 → Git仓库 → 触发CI管道 → 构建Docker镜像 → 推送到镜像仓库
                                  ↓
更新部署清单 → GitOps仓库 → ArgoCD检测变更 → 自动部署到集群
```

## 2. 环境管理策略
- **开发环境**: 自动部署每次提交
- **测试环境**: 自动部署稳定分支
- **预生产环境**: 手动批准部署
- **生产环境**: 蓝绿部署或金丝雀部署

## 3. 回滚策略
- Git回滚: 恢复到之前的部署清单
- Kubernetes回滚: 使用rollout命令
- 镜像回滚: 切换到稳定版本镜像
EOF
```

### 2. GitHub Actions CI/CD管道

```bash
cat > ci-cd/github-actions.yaml << 'EOF'
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run linting
      run: npm run lint
    
    - name: Run unit tests
      run: npm test
    
    - name: Run security scan
      run: npm audit

  build:
    name: Build and Push Image
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop'
    
    steps:
    - uses: actions/checkout@v3
      with:
        repository: your-org/k8s-manifests
        token: ${{ secrets.GITOPS_TOKEN }}
    
    - name: Update staging manifest
      run: |
        NEW_TAG=$(echo $GITHUB_SHA | cut -c1-7)
        sed -i "s|image: .*|image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:develop-${NEW_TAG}|" staging/deployment.yaml
        
        git config user.name "GitHub Actions"
        git config user.email "actions@github.com"
        git add staging/deployment.yaml
        git commit -m "Update staging image to ${NEW_TAG}"
        git push

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v3
      with:
        repository: your-org/k8s-manifests
        token: ${{ secrets.GITOPS_TOKEN }}
    
    - name: Update production manifest
      run: |
        NEW_TAG=$(echo $GITHUB_SHA | cut -c1-7)
        sed -i "s|image: .*|image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main-${NEW_TAG}|" production/deployment.yaml
        
        git config user.name "GitHub Actions"
        git config user.email "actions@github.com"
        git add production/deployment.yaml
        git commit -m "Update production image to ${NEW_TAG}"
        git push
EOF
```

### 3. ArgoCD GitOps部署

```bash
cat > gitops/argocd-install.yaml << 'EOF'
# ArgoCD 安装配置
apiVersion: v1
kind: Namespace
metadata:
  name: argocd
---
# 使用官方安装清单
# kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# ArgoCD 应用配置
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: production-app
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/k8s-manifests
    targetRevision: HEAD
    path: production
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
    retry:
      limit: 3
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
EOF

# 安装ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# 等待ArgoCD就绪
kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n argocd

# 获取初始密码
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d
```

## 🏗️ 高可用架构实践

### 1. 多Master节点配置

```bash
cat > ha/ha-cluster-setup.md << 'EOF'
# 高可用集群搭建

## 1. 负载均衡器配置
使用云负载均衡器或HAProxy为API Server提供高可用：

```yaml
# HAProxy配置示例
global
    daemon

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend k8s-api
    bind *:6443
    default_backend k8s-masters

backend k8s-masters
    balance roundrobin
    server master1 10.0.1.10:6443 check
    server master2 10.0.1.11:6443 check
    server master3 10.0.1.12:6443 check
```

## 2. etcd集群配置
确保etcd运行在奇数节点上(3或5个节点)
```

cat > ha/etcd-backup.sh << 'EOF'
#!/bin/bash
# etcd备份脚本

ETCDCTL_API=3
BACKUP_DIR="/backup/etcd/$(date +%Y%m%d-%H%M%S)"
ENDPOINT="https://127.0.0.1:2379"

mkdir -p $BACKUP_DIR

# 创建etcd快照
etcdctl snapshot save $BACKUP_DIR/etcd-snapshot.db \
  --endpoints=$ENDPOINT \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key

# 验证快照
etcdctl snapshot status $BACKUP_DIR/etcd-snapshot.db --write-out=table

echo "Backup completed: $BACKUP_DIR/etcd-snapshot.db"

# 清理7天前的备份
find /backup/etcd -type d -mtime +7 -exec rm -rf {} \;
EOF

chmod +x ha/etcd-backup.sh
```

### 2. 应用层高可用设计

```bash
cat > ha/app-ha-deployment.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: high-availability-app
  namespace: production
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # 保证可用性
  selector:
    matchLabels:
      app: ha-app
  template:
    metadata:
      labels:
        app: ha-app
    spec:
      # 反亲和性确保Pod分布在不同节点
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - ha-app
            topologyKey: kubernetes.io/hostname
      
      containers:
      - name: app
        image: nginx:1.21-alpine
        ports:
        - containerPort: 80
        
        # 健康检查配置
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        
        # 资源限制
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "500m"
        
        # 优雅关闭
        lifecycle:
          preStop:
            exec:
              command:
              - /bin/sh
              - -c
              - sleep 15
      
      # 容忍度配置
      tolerations:
      - key: node.kubernetes.io/unreachable
        operator: Exists
        effect: NoExecute
        tolerationSeconds: 30
      - key: node.kubernetes.io/not-ready
        operator: Exists
        effect: NoExecute
        tolerationSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: ha-app-service
  namespace: production
spec:
  selector:
    app: ha-app
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
---
# 服务中断预算
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ha-app-pdb
  namespace: production
spec:
  minAvailable: 2  # 至少保持2个Pod运行
  selector:
    matchLabels:
      app: ha-app
EOF
```

## ⚡ 性能调优实践

### 1. 资源配额和限制

```bash
cat > monitoring/resource-quota.yaml << 'EOF'
# 命名空间资源配额
apiVersion: v1
kind: ResourceQuota
metadata:
  name: production-quota
  namespace: production
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "10"
    services: "20"
    secrets: "20"
    configmaps: "20"
---
# CPU限制范围
apiVersion: v1
kind: LimitRange
metadata:
  name: production-limits
  namespace: production
spec:
  limits:
  - default:
      cpu: "500m"
      memory: "512Mi"
    defaultRequest:
      cpu: "100m"
      memory: "128Mi"
    max:
      cpu: "2"
      memory: "4Gi"
    min:
      cpu: "50m"
      memory: "64Mi"
    type: Container
EOF

kubectl apply -f monitoring/resource-quota.yaml
```

### 2. 节点性能优化

```bash
cat > monitoring/node-performance.yaml << 'EOF'
# 节点性能监控DaemonSet
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-performance-monitor
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: node-monitor
  template:
    metadata:
      labels:
        app: node-monitor
    spec:
      hostNetwork: true
      hostPID: true
      containers:
      - name: node-exporter
        image: prom/node-exporter:latest
        args:
        - '--path.procfs=/host/proc'
        - '--path.sysfs=/host/sys'
        - '--path.rootfs=/host/root'
        - '--collector.filesystem.ignored-mount-points'
        - '^/(sys|proc|dev|host|etc|rootfs/var/lib/docker/containers|rootfs/var/lib/docker/overlay2|rootfs/run/docker/netns|rootfs/var/lib/docker/aufs)($$|/)'
        ports:
        - containerPort: 9100
          hostPort: 9100
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
        - name: root
          mountPath: /host/root
          readOnly: true
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
      - name: root
        hostPath:
          path: /
      tolerations:
      - effect: NoSchedule
        operator: Exists
EOF

kubectl apply -f monitoring/node-performance.yaml
```

### 3. 应用性能优化配置

```bash
cat > monitoring/performance-tuning.yaml << 'EOF'
# 性能优化的应用部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: optimized-app
  namespace: production
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  selector:
    matchLabels:
      app: optimized-app
  template:
    metadata:
      labels:
        app: optimized-app
      annotations:
        # 性能相关注解
        cluster-autoscaler.kubernetes.io/safe-to-evict: "true"
    spec:
      # 性能优化配置
      priority: 1000  # 高优先级
      priorityClassName: high-priority
      
      # 调度优化
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: node-type
                operator: In
                values:
                - high-performance
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - optimized-app
              topologyKey: kubernetes.io/hostname
      
      containers:
      - name: app
        image: your-app:optimized
        ports:
        - containerPort: 8080
        
        # JVM优化参数
        env:
        - name: JAVA_OPTS
          value: "-Xms512m -Xmx1g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
        - name: SPRING_PROFILES_ACTIVE
          value: "production"
        
        # 资源精确配置
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        
        # 优化的健康检查
        livenessProbe:
          httpGet:
            path: /actuator/health/liveness
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /actuator/health/readiness
            port: 8080
          initialDelaySeconds: 20
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
---
# 高优先级类
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority
value: 1000
globalDefault: false
description: "High priority class for critical applications"
EOF
```

## 🔒 安全加固配置

### 1. Pod安全标准

```bash
cat > security/pod-security.yaml << 'EOF'
# Pod安全策略
apiVersion: v1
kind: Namespace
metadata:
  name: secure-namespace
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
---
# 安全的应用部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-app
  namespace: secure-namespace
spec:
  replicas: 2
  selector:
    matchLabels:
      app: secure-app
  template:
    metadata:
      labels:
        app: secure-app
    spec:
      serviceAccountName: secure-app-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 3000
        fsGroup: 2000
        seccompProfile:
          type: RuntimeDefault
      
      containers:
      - name: app
        image: nginx:alpine
        ports:
        - containerPort: 8080
        
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1000
          capabilities:
            drop:
            - ALL
            add:
            - NET_BIND_SERVICE
        
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: var-cache
          mountPath: /var/cache/nginx
        - name: var-run
          mountPath: /var/run
      
      volumes:
      - name: tmp
        emptyDir: {}
      - name: var-cache
        emptyDir: {}
      - name: var-run
        emptyDir: {}
---
# 服务账户
apiVersion: v1
kind: ServiceAccount
metadata:
  name: secure-app-sa
  namespace: secure-namespace
automountServiceAccountToken: false
EOF
```

### 2. 网络安全策略

```bash
cat > security/network-policies.yaml << 'EOF'
# 默认拒绝策略
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
---
# 允许应用间通信
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-app-communication
  namespace: production
spec:
  podSelector:
    matchLabels:
      tier: frontend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tier: gateway
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          tier: backend
    ports:
    - protocol: TCP
      port: 8080
  - to: []  # 允许DNS解析
    ports:
    - protocol: UDP
      port: 53
---
# 数据库访问策略
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: database-access
  namespace: production
spec:
  podSelector:
    matchLabels:
      tier: database
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tier: backend
    ports:
    - protocol: TCP
      port: 5432
EOF
```

## 💾 备份与灾难恢复

### 1. Velero备份系统

```bash
cat > backup/velero-setup.sh << 'EOF'
#!/bin/bash

# 安装Velero
curl -fsSL -o velero-v1.12.0-linux-amd64.tar.gz https://github.com/vmware-tanzu/velero/releases/download/v1.12.0/velero-v1.12.0-linux-amd64.tar.gz
tar -xzf velero-v1.12.0-linux-amd64.tar.gz
sudo mv velero-v1.12.0-linux-amd64/velero /usr/local/bin/

# 配置AWS S3存储（示例）
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.8.0 \
  --bucket velero-backups \
  --secret-file ./credentials-velero \
  --backup-location-config region=us-west-2 \
  --snapshot-location-config region=us-west-2

# 创建备份
velero backup create backup-$(date +%Y%m%d) --include-namespaces production

# 定期备份
velero schedule create daily-backup --schedule="0 1 * * *" --include-namespaces production --ttl 720h
EOF

cat > backup/backup-strategy.yaml << 'EOF'
# 备份策略配置
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: production-backup
  namespace: velero
spec:
  schedule: "0 2 * * *"  # 每天凌晨2点
  template:
    includedNamespaces:
    - production
    - monitoring
    excludedResources:
    - events
    - replicationcontrollers
    - endpoints
    storageLocation: default
    volumeSnapshotLocations:
    - default
    ttl: 720h0m0s  # 保留30天
EOF
```

### 2. 灾难恢复演练

```bash
cat > backup/disaster-recovery.sh << 'EOF'
#!/bin/bash

echo "=== 灾难恢复演练 ==="

# 1. 模拟故障
echo "1. 模拟应用故障..."
kubectl delete namespace production

# 2. 检查备份
echo "2. 检查可用备份..."
velero backup get

# 3. 执行恢复
echo "3. 执行恢复操作..."
LATEST_BACKUP=$(velero backup get --output json | jq -r '.items[0].metadata.name')
velero restore create restore-$(date +%Y%m%d-%H%M%S) --from-backup $LATEST_BACKUP

# 4. 验证恢复
echo "4. 验证恢复状态..."
kubectl get pods -n production
kubectl get services -n production

echo "灾难恢复演练完成"
EOF

chmod +x backup/disaster-recovery.sh
```

## 🧪 生产环境实战演练

### 演练1：零停机部署

```bash
cat > ci-cd/zero-downtime-deployment.yaml << 'EOF'
# 零停机部署配置
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zero-downtime-app
  namespace: production
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 50%      # 最多增加50%的Pod
      maxUnavailable: 0   # 不允许不可用的Pod
  selector:
    matchLabels:
      app: zero-downtime-app
  template:
    metadata:
      labels:
        app: zero-downtime-app
    spec:
      containers:
      - name: app
        image: nginx:1.21-alpine
        ports:
        - containerPort: 80
        
        # 关键：精确的健康检查
        readinessProbe:
          httpGet:
            path: /ready
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 2
          timeoutSeconds: 1
          successThreshold: 1
          failureThreshold: 3
        
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        # 优雅关闭
        lifecycle:
          preStop:
            exec:
              command:
              - /bin/sh
              - -c
              - sleep 15  # 等待连接处理完成
        
        resources:
          requests:
            memory: "64Mi"
            cpu: "50m"
          limits:
            memory: "128Mi"
            cpu: "200m"
---
# 服务配置
apiVersion: v1
kind: Service
metadata:
  name: zero-downtime-service
  namespace: production
spec:
  selector:
    app: zero-downtime-app
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
---
# 中断预算
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: zero-downtime-pdb
  namespace: production
spec:
  minAvailable: 3  # 至少保持3个Pod运行
  selector:
    matchLabels:
      app: zero-downtime-app
EOF
```

### 演练2：金丝雀部署

```bash
cat > ci-cd/canary-deployment.yaml << 'EOF'
# 金丝雀部署 - 稳定版本
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-stable
  namespace: production
  labels:
    version: stable
spec:
  replicas: 9  # 90%流量
  selector:
    matchLabels:
      app: canary-app
      version: stable
  template:
    metadata:
      labels:
        app: canary-app
        version: stable
    spec:
      containers:
      - name: app
        image: nginx:1.20-alpine
        ports:
        - containerPort: 80
        env:
        - name: VERSION
          value: "stable"
---
# 金丝雀部署 - 新版本
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-canary
  namespace: production
  labels:
    version: canary
spec:
  replicas: 1  # 10%流量
  selector:
    matchLabels:
      app: canary-app
      version: canary
  template:
    metadata:
      labels:
        app: canary-app
        version: canary
    spec:
      containers:
      - name: app
        image: nginx:1.21-alpine
        ports:
        - containerPort: 80
        env:
        - name: VERSION
          value: "canary"
---
# 统一服务
apiVersion: v1
kind: Service
metadata:
  name: canary-service
  namespace: production
spec:
  selector:
    app: canary-app
  ports:
  - port: 80
    targetPort: 80
EOF
```

## 📊 生产环境监控仪表板

```bash
cat > monitoring/production-dashboard.json << 'EOF'
{
  "dashboard": {
    "title": "Production Environment Overview",
    "panels": [
      {
        "title": "Cluster Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"kubernetes-nodes\"}",
            "legendFormat": "Node Status"
          }
        ]
      },
      {
        "title": "Application Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Request Rate"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th Percentile Latency"
          }
        ]
      },
      {
        "title": "Resource Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total[5m])",
            "legendFormat": "CPU Usage"
          },
          {
            "expr": "container_memory_usage_bytes",
            "legendFormat": "Memory Usage"
          }
        ]
      }
    ]
  }
}
EOF
```

## 🎯 生产环境检查清单

### 部署前检查

```bash
cat > production-checklist.md << 'EOF'
# 生产环境部署检查清单

## 🔍 安全检查
- [ ] RBAC权限配置正确
- [ ] NetworkPolicy网络隔离
- [ ] Pod安全上下文配置
- [ ] Secret加密存储
- [ ] 镜像安全扫描通过
- [ ] 准入控制器配置

## 🏗️ 高可用检查
- [ ] 多副本部署(最少3个)
- [ ] Pod反亲和性配置
- [ ] 健康检查配置完整
- [ ] 服务中断预算设置
- [ ] 负载均衡配置正确
- [ ] 数据持久化方案

## ⚡ 性能检查
- [ ] 资源请求和限制设置
- [ ] HPA自动扩缩容配置
- [ ] 镜像优化(多阶段构建)
- [ ] JVM参数调优
- [ ] 数据库连接池优化
- [ ] 缓存策略配置

## 📊 监控检查
- [ ] Prometheus监控配置
- [ ] Grafana仪表板创建
- [ ] 告警规则设置
- [ ] 日志收集配置
- [ ] 链路追踪集成
- [ ] 性能指标暴露

## 💾 备份检查
- [ ] Velero备份策略
- [ ] 数据备份自动化
- [ ] 配置文件版本控制
- [ ] 灾难恢复方案测试
- [ ] RTO/RPO目标定义
- [ ] 备份验证流程

## 🚀 部署检查
- [ ] CI/CD管道测试
- [ ] 金丝雀/蓝绿部署策略
- [ ] 回滚方案准备
- [ ] 零停机部署验证
- [ ] 环境配置同步
- [ ] 依赖服务检查
EOF
```

## 🎓 学习检查

完成本实验后，你应该能够：

1. **生产环境架构设计**：
   - 设计高可用的Kubernetes集群
   - 规划应用的部署架构
   - 制定容灾备份策略

2. **CI/CD与GitOps**：
   - 实现完整的CI/CD管道
   - 配置GitOps工作流
   - 实现自动化部署和回滚

3. **性能与安全优化**：
   - 进行资源调优和性能监控
   - 实施安全策略和合规要求
   - 配置生产级监控告警

4. **运维最佳实践**：
   - 执行零停机部署
   - 处理生产环境故障
   - 进行灾难恢复演练

## 🎉 恭喜完成Kubernetes学习体系

你已经完成了从入门到生产环境的完整Kubernetes学习路径！现在你具备了：

- **基础能力**：Pod、Service、Deployment等核心概念
- **进阶技能**：存储、网络、安全配置
- **高级实践**：监控、扩缩容、生产环境最佳实践

继续保持学习，关注Kubernetes生态的最新发展！

---

**🎯 下一步建议**：
- 深入学习Istio服务网格
- 探索Kubernetes Operator开发
- 参与开源社区贡献
- 准备CKA/CKS认证考试 