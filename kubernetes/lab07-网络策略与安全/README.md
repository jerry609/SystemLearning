# 实验7：网络策略与安全

## 🎯 学习目标

通过本实验，你将：
- 深入理解Kubernetes网络模型和安全机制
- 掌握NetworkPolicy的创建和应用
- 学习RBAC权限控制系统
- 实践Pod安全策略和准入控制
- 了解服务网格安全最佳实践
- 掌握安全扫描和漏洞管理

## 📚 理论知识学习

### Kubernetes网络安全架构

```
┌─── Cluster Network ────────────────────────────────────┐
│                                                        │
│ ┌─ Namespace A ─┐    ┌─ Namespace B ─┐                │
│ │               │    │               │                │
│ │ Pod A1 ←──────┼────┼→ Pod B1       │ NetworkPolicy │
│ │   ↕           │    │   ↕           │                │
│ │ Pod A2        │    │ Pod B2        │                │
│ └───────────────┘    └───────────────┘                │
│                                                        │
│ ┌──────── Ingress Controller ──────────┐               │
│ │  External Traffic → Cluster         │               │
│ └─────────────────────────────────────┘               │
└────────────────────────────────────────────────────────┘
```

### 安全层次结构

| 层级 | 组件 | 功能 |
|------|------|------|
| **集群级** | RBAC, PSP | 用户权限、资源访问控制 |
| **网络级** | NetworkPolicy | 网络流量控制 |
| **Pod级** | SecurityContext | 容器安全配置 |
| **应用级** | Secret, TLS | 数据加密、身份认证 |

## 🛡️ 网络策略实践

### 准备工作

```bash
# 创建实验目录
mkdir -p ~/k8s-labs/lab07/{network-policy,rbac,security,examples}
cd ~/k8s-labs/lab07

# 创建测试命名空间
kubectl create namespace frontend
kubectl create namespace backend
kubectl create namespace database
```

### 1. 默认拒绝所有流量

```bash
cat > network-policy/deny-all.yaml << 'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-traffic
  namespace: backend
spec:
  podSelector: {}  # 选择所有Pod
  policyTypes:
  - Ingress
  - Egress
  # 没有规则 = 拒绝所有
EOF

kubectl apply -f network-policy/deny-all.yaml
```

### 2. 允许特定来源的入口流量

```bash
cat > network-policy/allow-frontend.yaml << 'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: backend
spec:
  podSelector:
    matchLabels:
      app: backend-api
  policyTypes:
  - Ingress
  ingress:
  - from:
    # 允许来自frontend命名空间的流量
    - namespaceSelector:
        matchLabels:
          name: frontend
    # 允许来自特定Pod的流量
    - podSelector:
        matchLabels:
          app: frontend-web
    ports:
    - protocol: TCP
      port: 8080
EOF

kubectl apply -f network-policy/allow-frontend.yaml
```

### 3. 控制出口流量

```bash
cat > network-policy/allow-egress.yaml << 'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-egress-policy
  namespace: backend
spec:
  podSelector:
    matchLabels:
      app: backend-api
  policyTypes:
  - Egress
  egress:
  # 允许访问数据库
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
  
  # 允许DNS解析
  - to: []
    ports:
    - protocol: UDP
      port: 53
  
  # 允许访问外部API
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
        except:
        - 169.254.169.254/32  # 排除元数据服务
    ports:
    - protocol: TCP
      port: 443
EOF

kubectl apply -f network-policy/allow-egress.yaml
```

### 4. 测试网络策略

```bash
# 部署测试应用
cat > examples/test-apps.yaml << 'EOF'
# Frontend应用
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend-web
  namespace: frontend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: frontend-web
  template:
    metadata:
      labels:
        app: frontend-web
    spec:
      containers:
      - name: web
        image: nginx:alpine
        ports:
        - containerPort: 80
---
# Backend应用
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-api
  namespace: backend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: backend-api
  template:
    metadata:
      labels:
        app: backend-api
    spec:
      containers:
      - name: api
        image: nginx:alpine
        ports:
        - containerPort: 8080
        command: ["/bin/sh"]
        args: ["-c", "echo 'Backend API' > /usr/share/nginx/html/index.html && nginx -g 'daemon off;'"]
---
# Database应用
apiVersion: apps/v1
kind: Deployment
metadata:
  name: database
  namespace: database
spec:
  replicas: 1
  selector:
    matchLabels:
      app: database
  template:
    metadata:
      labels:
        app: database
    spec:
      containers:
      - name: db
        image: postgres:13
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_PASSWORD
          value: password123
EOF

kubectl apply -f examples/test-apps.yaml

# 给命名空间打标签
kubectl label namespace frontend name=frontend
kubectl label namespace backend name=backend
kubectl label namespace database name=database
```

## 🔐 RBAC权限控制

### 1. 创建ServiceAccount

```bash
cat > rbac/service-account.yaml << 'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-service-account
  namespace: default
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: read-only-user
  namespace: default
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: admin-user
  namespace: kube-system
EOF

kubectl apply -f rbac/service-account.yaml
```

### 2. 创建Role和ClusterRole

```bash
cat > rbac/roles.yaml << 'EOF'
# 命名空间级别的Role
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: default
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods", "pods/log"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list"]
---
# 集群级别的ClusterRole
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: deployment-manager
rules:
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "watch"]
---
# 只读权限ClusterRole
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cluster-reader
rules:
- apiGroups: [""]
  resources: ["*"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps", "extensions"]
  resources: ["*"]
  verbs: ["get", "list", "watch"]
EOF

kubectl apply -f rbac/roles.yaml
```

### 3. 创建RoleBinding和ClusterRoleBinding

```bash
cat > rbac/bindings.yaml << 'EOF'
# RoleBinding：绑定Role到ServiceAccount
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pod-reader-binding
  namespace: default
subjects:
- kind: ServiceAccount
  name: app-service-account
  namespace: default
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
---
# ClusterRoleBinding：绑定ClusterRole到ServiceAccount
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: deployment-manager-binding
subjects:
- kind: ServiceAccount
  name: admin-user
  namespace: kube-system
roleRef:
  kind: ClusterRole
  name: deployment-manager
  apiGroup: rbac.authorization.k8s.io
---
# 只读用户绑定
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: read-only-binding
subjects:
- kind: ServiceAccount
  name: read-only-user
  namespace: default
roleRef:
  kind: ClusterRole
  name: cluster-reader
  apiGroup: rbac.authorization.k8s.io
EOF

kubectl apply -f rbac/bindings.yaml
```

### 4. 测试RBAC权限

```bash
# 获取ServiceAccount的token
SA_TOKEN=$(kubectl get secret $(kubectl get sa app-service-account -o jsonpath='{.secrets[0].name}') -o jsonpath='{.data.token}' | base64 -d)

# 使用token测试权限
kubectl auth can-i get pods --as=system:serviceaccount:default:app-service-account
kubectl auth can-i create deployments --as=system:serviceaccount:default:app-service-account
kubectl auth can-i delete nodes --as=system:serviceaccount:default:read-only-user
```

## 🛡️ Pod安全策略

### 1. SecurityContext配置

```bash
cat > security/security-context.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 3000
    fsGroup: 2000
  containers:
  - name: app
    image: alpine
    command: ["/bin/sh"]
    args: ["-c", "while true; do whoami; id; ls -la /data; sleep 30; done"]
    
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
        add:
        - NET_BIND_SERVICE
    
    volumeMounts:
    - name: data
      mountPath: /data
    - name: tmp
      mountPath: /tmp
  
  volumes:
  - name: data
    emptyDir: {}
  - name: tmp
    emptyDir: {}
EOF

kubectl apply -f security/security-context.yaml
kubectl logs secure-pod
```

### 2. Pod Security Standards

```bash
cat > security/pss-namespace.yaml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: secure-namespace
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
---
apiVersion: v1
kind: Namespace
metadata:
  name: baseline-namespace
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/audit: baseline
    pod-security.kubernetes.io/warn: baseline
EOF

kubectl apply -f security/pss-namespace.yaml

# 测试在受限命名空间中部署Pod
cat > security/test-restricted-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
  namespace: secure-namespace
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: app
    image: alpine
    command: ["sleep", "3600"]
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    volumeMounts:
    - name: tmp
      mountPath: /tmp
  volumes:
  - name: tmp
    emptyDir: {}
EOF

kubectl apply -f security/test-restricted-pod.yaml
```

## 🔍 安全扫描与审计

### 1. 资源配置检查

```bash
cat > security/security-check.sh << 'EOF'
#!/bin/bash

echo "=== Kubernetes Security Check ==="

# 检查特权容器
echo "1. 检查特权容器："
kubectl get pods --all-namespaces -o jsonpath='{range .items[*]}{.metadata.namespace}{"\t"}{.metadata.name}{"\t"}{.spec.containers[*].securityContext.privileged}{"\n"}{end}' | grep true

# 检查以root用户运行的容器
echo "2. 检查root用户容器："
kubectl get pods --all-namespaces -o jsonpath='{range .items[*]}{.metadata.namespace}{"\t"}{.metadata.name}{"\t"}{.spec.securityContext.runAsUser}{"\n"}{end}' | grep -E "^[^0-9]*\t[^0-9]*\t0$"

# 检查没有资源限制的Pod
echo "3. 检查无资源限制的Pod："
kubectl get pods --all-namespaces -o json | jq -r '.items[] | select(.spec.containers[].resources.limits == null) | "\(.metadata.namespace)\t\(.metadata.name)"'

# 检查NetworkPolicy覆盖情况
echo "4. 检查NetworkPolicy："
kubectl get networkpolicy --all-namespaces

# 检查ServiceAccount权限
echo "5. 检查ServiceAccount："
kubectl get clusterrolebindings -o json | jq -r '.items[] | select(.subjects[]?.kind == "ServiceAccount") | "\(.metadata.name)\t\(.roleRef.name)"'
EOF

chmod +x security/security-check.sh
./security/security-check.sh
```

### 2. 准入控制器

```bash
cat > security/admission-controller.yaml << 'EOF'
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionWebhook
metadata:
  name: security-policy-webhook
webhooks:
- name: security.example.com
  clientConfig:
    service:
      name: security-webhook
      namespace: default
      path: "/validate"
  rules:
  - operations: ["CREATE", "UPDATE"]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods"]
  admissionReviewVersions: ["v1", "v1beta1"]
  sideEffects: None
  failurePolicy: Fail
EOF

# 注意：这个例子需要实际的webhook服务
# kubectl apply -f security/admission-controller.yaml
```

## 🔧 服务网格安全

### 1. mTLS通信

```bash
cat > security/mtls-example.yaml << 'EOF'
# 模拟服务间mTLS配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: tls-config
data:
  nginx.conf: |
    events {}
    http {
        upstream backend {
            server backend-service:8080;
        }
        
        server {
            listen 443 ssl;
            ssl_certificate /etc/ssl/certs/tls.crt;
            ssl_certificate_key /etc/ssl/private/tls.key;
            ssl_client_certificate /etc/ssl/certs/ca.crt;
            ssl_verify_client on;
            
            location / {
                proxy_pass http://backend;
                proxy_ssl_certificate /etc/ssl/certs/client.crt;
                proxy_ssl_certificate_key /etc/ssl/private/client.key;
            }
        }
    }
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-frontend
spec:
  replicas: 1
  selector:
    matchLabels:
      app: secure-frontend
  template:
    metadata:
      labels:
        app: secure-frontend
    spec:
      containers:
      - name: nginx
        image: nginx:alpine
        ports:
        - containerPort: 443
        volumeMounts:
        - name: tls-config
          mountPath: /etc/nginx/
        - name: tls-certs
          mountPath: /etc/ssl/
      
      volumes:
      - name: tls-config
        configMap:
          name: tls-config
      - name: tls-certs
        secret:
          secretName: tls-secret
EOF

# kubectl apply -f security/mtls-example.yaml
```

## 🛠️ 实验练习

### 练习1：多层网络隔离

```bash
# 创建分层网络策略
cat > examples/tiered-security.yaml << 'EOF'
# DMZ区域 - 只允许入口流量
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: dmz-policy
  namespace: frontend
spec:
  podSelector:
    matchLabels:
      tier: dmz
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from: []  # 允许所有入口
    ports:
    - protocol: TCP
      port: 80
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: backend
    ports:
    - protocol: TCP
      port: 8080

---
# 应用层 - 只允许DMZ访问
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: app-tier-policy
  namespace: backend
spec:
  podSelector:
    matchLabels:
      tier: app
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: frontend
      podSelector:
        matchLabels:
          tier: dmz
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
EOF

kubectl apply -f examples/tiered-security.yaml
```

## 📝 最佳实践总结

### 1. 网络安全原则

- **最小权限**：默认拒绝，明确允许
- **网络分段**：使用命名空间和NetworkPolicy隔离
- **流量监控**：记录和分析网络流量
- **定期审计**：检查网络策略有效性

### 2. 访问控制建议

- 为不同角色创建专用ServiceAccount
- 使用Role和ClusterRole实现细粒度权限控制
- 定期审查和清理不必要的权限绑定
- 启用审计日志记录敏感操作

### 3. Pod安全配置

- 禁用特权容器和特权升级
- 使用非root用户运行容器
- 配置只读根文件系统
- 限制容器capabilities

## 🎯 实验总结

通过本实验，你已经掌握了：

✅ **网络策略**: 流量控制和网络隔离
✅ **RBAC权限**: 细粒度访问控制
✅ **Pod安全**: 容器安全配置
✅ **安全扫描**: 漏洞检测和合规检查
✅ **准入控制**: 资源创建时的安全验证
✅ **服务网格**: mTLS和零信任网络

继续下一个实验：**实验8：监控与日志管理** 