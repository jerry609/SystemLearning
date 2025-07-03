# 实验5：ConfigMap和Secret配置管理

## 🎯 学习目标

通过本实验，你将：
- 深入理解Kubernetes配置管理的最佳实践
- 掌握ConfigMap的创建和使用方法
- 学习Secret的安全配置管理
- 实践环境变量、Volume挂载等配置注入方式
- 了解配置热更新和版本管理
- 掌握敏感信息的安全处理

## 📚 理论知识学习

### 配置管理核心概念

在Kubernetes中，应用配置与代码分离是最佳实践，主要通过ConfigMap和Secret来实现。

#### 为什么需要配置分离？
- **环境适配**：同一镜像适用于不同环境
- **配置热更新**：无需重建镜像即可更新配置
- **安全管理**：敏感信息独立存储
- **团队协作**：开发和运维分工明确

### ConfigMap vs Secret

| 特性 | ConfigMap | Secret |
|------|-----------|--------|
| **用途** | 非敏感配置数据 | 敏感信息（密码、密钥等） |
| **存储** | 明文存储 | Base64编码存储 |
| **大小限制** | 1MB | 1MB |
| **访问权限** | 普通权限 | 需要特殊权限 |
| **典型用例** | 配置文件、环境变量 | 数据库密码、API密钥、TLS证书 |

### 配置注入方式

```
┌─────────── Pod ───────────────┐
│                               │
│  ┌─── Container ────┐         │
│  │                  │         │
│  │ ENV_VAR=value ←──┼─────────┼── ConfigMap/Secret
│  │                  │         │   (环境变量)
│  │ /app/config/ ←───┼─────────┼── ConfigMap/Secret
│  │   ├─app.yaml     │         │   (Volume挂载)
│  │   └─db.conf      │         │
│  │                  │         │
│  └──────────────────┘         │
└───────────────────────────────┘
```

## 🗃️ ConfigMap操作实践

### 准备工作

```bash
# 创建实验目录
mkdir -p ~/k8s-labs/lab05/{configs,secrets,examples}
cd ~/k8s-labs/lab05

# 清理之前的资源
kubectl delete configmap --all
kubectl delete secret --all
kubectl delete deployment --all
```

### 1. ConfigMap创建方法

#### 方法一：kubectl create命令

```bash
# 从字面值创建
kubectl create configmap app-config \
  --from-literal=app.name=MyApp \
  --from-literal=app.version=1.0 \
  --from-literal=debug=true

# 查看ConfigMap
kubectl get configmap app-config -o yaml
```

#### 方法二：从文件创建

```bash
# 准备配置文件
cat > configs/app.properties << 'EOF'
server.port=8080
server.host=0.0.0.0
database.url=jdbc:mysql://mysql:3306/myapp
database.driver=com.mysql.cj.jdbc.Driver
logging.level=INFO
cache.enabled=true
cache.ttl=3600
EOF

cat > configs/nginx.conf << 'EOF'
server {
    listen 80;
    server_name localhost;
    
    location / {
        root /usr/share/nginx/html;
        index index.html index.htm;
    }
    
    location /api/ {
        proxy_pass http://backend:8080/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;
}
EOF

# 从文件创建ConfigMap
kubectl create configmap app-properties --from-file=configs/app.properties
kubectl create configmap nginx-config --from-file=nginx.conf=configs/nginx.conf

# 从目录创建（包含所有文件）
kubectl create configmap all-configs --from-file=configs/
```

#### 方法三：YAML文件创建

```bash
cat > examples/configmap-yaml.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: web-config
  labels:
    app: web-app
    component: config
data:
  # 简单键值对
  APP_NAME: "Web Application"
  APP_VERSION: "2.0"
  DEBUG_MODE: "false"
  
  # 多行配置文件
  app.yaml: |
    server:
      port: 8080
      host: "0.0.0.0"
    database:
      host: "mysql-service"
      port: 3306
      name: "webapp"
      pool_size: 10
    logging:
      level: "INFO"
      format: "json"
    
  # HTML模板
  index.html: |
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{APP_NAME}}</title>
    </head>
    <body>
        <h1>Welcome to {{APP_NAME}}</h1>
        <p>Version: {{APP_VERSION}}</p>
    </body>
    </html>
EOF

kubectl apply -f examples/configmap-yaml.yaml
```

### 2. ConfigMap使用方式

#### 环境变量注入

```bash
cat > examples/configmap-env.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-with-config-env
spec:
  replicas: 2
  selector:
    matchLabels:
      app: config-env-demo
  template:
    metadata:
      labels:
        app: config-env-demo
    spec:
      containers:
      - name: app
        image: nginx:alpine
        
        # 方式1：引用整个ConfigMap
        envFrom:
        - configMapRef:
            name: web-config
        
        # 方式2：选择性引用特定键
        env:
        - name: SERVER_PORT
          valueFrom:
            configMapKeyRef:
              name: web-config
              key: server.port
        - name: CUSTOM_MESSAGE
          value: "Hello from $(APP_NAME) v$(APP_VERSION)"
        
        # 验证环境变量的命令
        command: ["/bin/sh"]
        args: ["-c", "while true; do echo 'APP: $(APP_NAME), VERSION: $(APP_VERSION)'; sleep 30; done"]
EOF

kubectl apply -f examples/configmap-env.yaml

# 验证环境变量
kubectl exec deployment/app-with-config-env -- env | grep -E "(APP_|DEBUG_|SERVER_)"
```

#### Volume挂载方式

```bash
cat > examples/configmap-volume.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-with-config-volume
spec:
  replicas: 1
  selector:
    matchLabels:
      app: config-volume-demo
  template:
    metadata:
      labels:
        app: config-volume-demo
    spec:
      containers:
      - name: nginx
        image: nginx:alpine
        ports:
        - containerPort: 80
        
        # 挂载配置文件
        volumeMounts:
        - name: nginx-config-volume
          mountPath: /etc/nginx/conf.d/
          readOnly: true
        - name: app-config-volume
          mountPath: /app/config/
          readOnly: true
        - name: html-volume
          mountPath: /usr/share/nginx/html/
          readOnly: true
      
      volumes:
      # 挂载nginx配置
      - name: nginx-config-volume
        configMap:
          name: nginx-config
          items:
          - key: nginx.conf
            path: default.conf
      
      # 挂载应用配置（整个ConfigMap）
      - name: app-config-volume
        configMap:
          name: web-config
      
      # 挂载HTML文件（选择特定key）
      - name: html-volume
        configMap:
          name: web-config
          items:
          - key: index.html
            path: index.html
EOF

kubectl apply -f examples/configmap-volume.yaml

# 验证挂载的配置文件
kubectl exec deployment/app-with-config-volume -- ls -la /etc/nginx/conf.d/
kubectl exec deployment/app-with-config-volume -- cat /etc/nginx/conf.d/default.conf
kubectl exec deployment/app-with-config-volume -- ls -la /app/config/
```

## 🔐 Secret操作实践

### 1. Secret创建方法

#### 方法一：kubectl create命令

```bash
# 创建通用Secret
kubectl create secret generic db-secret \
  --from-literal=username=admin \
  --from-literal=password=secretpass123 \
  --from-literal=database=myapp_db

# 创建Docker registry认证Secret
kubectl create secret docker-registry regcred \
  --docker-server=registry.example.com \
  --docker-username=myuser \
  --docker-password=mypass \
  --docker-email=user@example.com

# 创建TLS Secret
kubectl create secret tls tls-secret \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key
```

#### 方法二：从文件创建

```bash
# 准备敏感配置文件
mkdir -p secrets

cat > secrets/db.env << 'EOF'
DB_HOST=mysql.internal
DB_PORT=3306
DB_USER=webapp_user
DB_PASSWORD=ultra_secret_password_123
DB_NAME=production_db
EOF

echo "my-api-key-12345" > secrets/api.key
echo "jwt-secret-key-abcdef" > secrets/jwt.key

# 从文件创建Secret
kubectl create secret generic app-secrets --from-file=secrets/
kubectl create secret generic api-key --from-file=api-key=secrets/api.key
```

#### 方法三：YAML文件创建

```bash
cat > examples/secret-yaml.yaml << 'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: webapp-secrets
  labels:
    app: webapp
type: Opaque
data:
  # Base64编码的值
  username: YWRtaW4=                    # admin
  password: c3VwZXJfc2VjcmV0XzEyMw==     # super_secret_123
  api-key: YWJjZGVmZ2hpams=             # abcdefghijk
stringData:
  # 明文值（自动编码）
  database-url: "postgresql://user:pass@localhost:5432/mydb"
  jwt-secret: "my-jwt-secret-key"
  config.json: |
    {
      "api": {
        "key": "secret-api-key",
        "endpoint": "https://api.internal.com"
      },
      "encryption": {
        "algorithm": "AES-256",
        "key": "encryption-key-here"
      }
    }
EOF

kubectl apply -f examples/secret-yaml.yaml
```

### 2. Secret使用方式

#### 环境变量方式

```bash
cat > examples/secret-env.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-with-secrets
spec:
  replicas: 1
  selector:
    matchLabels:
      app: secret-demo
  template:
    metadata:
      labels:
        app: secret-demo
    spec:
      containers:
      - name: app
        image: alpine
        
        # 从Secret注入环境变量
        env:
        - name: DB_USERNAME
          valueFrom:
            secretKeyRef:
              name: webapp-secrets
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: webapp-secrets
              key: password
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: webapp-secrets
              key: api-key
        
        # 批量注入（谨慎使用）
        envFrom:
        - secretRef:
            name: db-secret
        
        command: ["/bin/sh"]
        args: ["-c", "while true; do echo 'DB User: $DB_USERNAME'; sleep 60; done"]
EOF

kubectl apply -f examples/secret-env.yaml

# 验证（注意：密码不应该在日志中显示）
kubectl exec deployment/app-with-secrets -- printenv | grep -E "(DB_|API_)"
```

#### Volume挂载方式

```bash
cat > examples/secret-volume.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-with-secret-files
spec:
  replicas: 1
  selector:
    matchLabels:
      app: secret-file-demo
  template:
    metadata:
      labels:
        app: secret-file-demo
    spec:
      containers:
      - name: app
        image: nginx:alpine
        
        volumeMounts:
        # 挂载所有secret内容
        - name: secret-volume
          mountPath: /etc/secrets/
          readOnly: true
        
        # 挂载特定secret文件
        - name: config-secret
          mountPath: /app/config/
          readOnly: true
        
        # 挂载到特定文件
        - name: api-key-file
          mountPath: /etc/api/key
          subPath: api-key
          readOnly: true
      
      volumes:
      - name: secret-volume
        secret:
          secretName: webapp-secrets
          defaultMode: 0400  # 只读权限
      
      - name: config-secret
        secret:
          secretName: webapp-secrets
          items:
          - key: config.json
            path: app-config.json
            mode: 0600
      
      - name: api-key-file
        secret:
          secretName: webapp-secrets
          items:
          - key: api-key
            path: api-key
EOF

kubectl apply -f examples/secret-volume.yaml

# 验证挂载的secret文件
kubectl exec deployment/app-with-secret-files -- ls -la /etc/secrets/
kubectl exec deployment/app-with-secret-files -- cat /etc/api/key
```

## 🔄 配置热更新实践

### 1. ConfigMap热更新

```bash
cat > examples/hot-reload-demo.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: reload-config
data:
  message: "Hello World v1.0"
  timestamp: "2024-01-01"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hot-reload-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hot-reload
  template:
    metadata:
      labels:
        app: hot-reload
    spec:
      containers:
      - name: app
        image: alpine
        command: ["/bin/sh"]
        args: ["-c", "while true; do echo '=== $(date) ==='; cat /config/message; cat /config/timestamp; sleep 10; done"]
        
        volumeMounts:
        - name: config
          mountPath: /config
      
      volumes:
      - name: config
        configMap:
          name: reload-config
EOF

kubectl apply -f examples/hot-reload-demo.yaml

# 观察当前输出
kubectl logs deployment/hot-reload-app -f &

# 更新ConfigMap
kubectl patch configmap reload-config -p '{"data":{"message":"Hello World v2.0 - Updated!","timestamp":"2024-12-01"}}'

# 观察配置是否自动更新（通常需要几分钟）
```

### 2. 强制Pod重启更新

```bash
# 方法1：重启Deployment
kubectl rollout restart deployment/hot-reload-app

# 方法2：通过annotation触发更新
kubectl patch deployment hot-reload-app -p '{"spec":{"template":{"metadata":{"annotations":{"kubectl.kubernetes.io/restartedAt":"'$(date +%Y-%m-%dT%H:%M:%S%z)'"}}}}}'

# 方法3：使用configmap hash annotation
CONFIGMAP_HASH=$(kubectl get configmap reload-config -o jsonpath='{.metadata.resourceVersion}')
kubectl patch deployment hot-reload-app -p '{"spec":{"template":{"metadata":{"annotations":{"configmap/reload-config":"'$CONFIGMAP_HASH'"}}}}}'
```

## 🛠️ 实验练习

### 练习1：多环境配置管理

创建开发、测试、生产三套环境的配置：

```bash
# 开发环境
kubectl create namespace dev
kubectl create configmap app-config -n dev \
  --from-literal=env=development \
  --from-literal=debug=true \
  --from-literal=db.host=dev-mysql

# 测试环境
kubectl create namespace test
kubectl create configmap app-config -n test \
  --from-literal=env=testing \
  --from-literal=debug=false \
  --from-literal=db.host=test-mysql

# 生产环境
kubectl create namespace prod
kubectl create configmap app-config -n prod \
  --from-literal=env=production \
  --from-literal=debug=false \
  --from-literal=db.host=prod-mysql-cluster

# 创建对应的Secret
kubectl create secret generic db-creds -n dev --from-literal=password=dev123
kubectl create secret generic db-creds -n test --from-literal=password=test456  
kubectl create secret generic db-creds -n prod --from-literal=password=prod_secure_pass
```

### 练习2：应用配置模板

```bash
cat > examples/app-template.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-template-config
data:
  application.yaml: |
    server:
      port: 8080
    spring:
      datasource:
        url: jdbc:mysql://${DB_HOST}:${DB_PORT}/${DB_NAME}
        username: ${DB_USERNAME}
        password: ${DB_PASSWORD}
      jpa:
        hibernate:
          ddl-auto: ${HIBERNATE_DDL}
    logging:
      level:
        com.myapp: ${LOG_LEVEL}
    app:
      features:
        feature-x: ${FEATURE_X_ENABLED}
        feature-y: ${FEATURE_Y_ENABLED}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spring-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: spring-app
  template:
    metadata:
      labels:
        app: spring-app
    spec:
      containers:
      - name: app
        image: openjdk:11-jre-slim
        env:
        - name: DB_HOST
          value: "mysql-service"
        - name: DB_PORT
          value: "3306"
        - name: DB_NAME
          value: "myapp"
        - name: DB_USERNAME
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
        - name: HIBERNATE_DDL
          value: "validate"
        - name: LOG_LEVEL
          value: "INFO"
        - name: FEATURE_X_ENABLED
          value: "true"
        - name: FEATURE_Y_ENABLED
          value: "false"
        
        volumeMounts:
        - name: config
          mountPath: /app/config
        
        command: ["/bin/sh"]
        args: ["-c", "envsubst < /app/config/application.yaml > /tmp/application.yaml && cat /tmp/application.yaml && sleep 3600"]
      
      volumes:
      - name: config
        configMap:
          name: app-template-config
EOF

kubectl apply -f examples/app-template.yaml
```

## 🚀 进阶实验

### 1. 配置版本管理

```bash
# 创建配置版本管理脚本
cat > scripts/config-versioning.sh << 'EOF'
#!/bin/bash

CONFIG_NAME="versioned-config"
VERSION=${1:-$(date +%Y%m%d-%H%M%S)}

# 创建带版本的ConfigMap
kubectl create configmap "${CONFIG_NAME}-${VERSION}" \
  --from-literal=version=$VERSION \
  --from-literal=config.updated=$(date) \
  --from-literal=message="Configuration version $VERSION"

# 更新应用使用的ConfigMap
kubectl patch deployment config-version-app -p "{
  \"spec\": {
    \"template\": {
      \"spec\": {
        \"volumes\": [{
          \"name\": \"config\",
          \"configMap\": {
            \"name\": \"${CONFIG_NAME}-${VERSION}\"
          }
        }]
      }
    }
  }
}"

echo "Configuration updated to version: $VERSION"
EOF

chmod +x scripts/config-versioning.sh
```

### 2. 配置校验和备份

```bash
cat > scripts/config-backup.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="config-backups/$(date +%Y-%m-%d)"
mkdir -p $BACKUP_DIR

# 备份所有ConfigMap
kubectl get configmap -o yaml > $BACKUP_DIR/configmaps.yaml

# 备份所有Secret（注意安全性）
kubectl get secret -o yaml > $BACKUP_DIR/secrets.yaml

# 创建配置清单
kubectl get configmap,secret --no-headers | awk '{print $1}' > $BACKUP_DIR/inventory.txt

echo "Configuration backed up to: $BACKUP_DIR"
EOF

chmod +x scripts/config-backup.sh
```

## 🔍 故障排查指南

### 常见问题及解决方案

#### 1. ConfigMap/Secret未挂载成功

```bash
# 检查Pod状态
kubectl describe pod <pod-name>

# 查看挂载点
kubectl exec <pod-name> -- ls -la /path/to/mount

# 检查Volume定义
kubectl get pod <pod-name> -o yaml | grep -A 10 volumes
```

#### 2. 环境变量未注入

```bash
# 验证环境变量
kubectl exec <pod-name> -- env

# 检查ConfigMap/Secret是否存在
kubectl get configmap
kubectl get secret

# 查看配置内容
kubectl describe configmap <name>
kubectl describe secret <name>
```

#### 3. 配置更新未生效

```bash
# 检查ConfigMap更新时间
kubectl get configmap <name> -o yaml | grep creationTimestamp

# 强制重启Pod
kubectl delete pod <pod-name>

# 或者重启Deployment
kubectl rollout restart deployment <deployment-name>
```

### 调试命令集合

```bash
# 配置对比
kubectl diff -f updated-config.yaml

# 配置历史
kubectl rollout history deployment <name>

# 实时日志监控
kubectl logs -f deployment/<name> | grep -i config

# 资源使用监控
kubectl top pod | grep <app-name>
```

## 📝 最佳实践总结

### 1. 配置管理原则

- **明确分离**：配置与代码完全分离
- **环境隔离**：不同环境使用不同配置
- **版本控制**：配置变更要有版本记录
- **安全第一**：敏感信息必须使用Secret

### 2. 命名规范

```bash
# ConfigMap命名
app-name-config
app-name-env-config  # 环境特定配置
app-name-feature-config  # 功能特定配置

# Secret命名
app-name-secret
app-name-db-secret
app-name-tls-secret
```

### 3. 配置结构化

```yaml
# 推荐的ConfigMap结构
apiVersion: v1
kind: ConfigMap
metadata:
  name: structured-config
  labels:
    app: myapp
    version: "1.0"
    environment: production
data:
  # 环境变量
  APP_ENV: "production"
  DEBUG_MODE: "false"
  
  # 结构化配置文件
  application.yaml: |
    # YAML格式配置
  
  app.properties: |
    # Properties格式配置
  
  config.json: |
    # JSON格式配置
```

### 4. 安全建议

- Secret使用最小权限原则
- 定期轮换敏感信息
- 避免在日志中输出敏感信息
- 使用RBAC控制配置访问权限

## 🎯 实验总结

通过本实验，你已经掌握了：

✅ **ConfigMap管理**：创建、更新、使用配置数据
✅ **Secret管理**：安全地处理敏感信息
✅ **配置注入**：环境变量和Volume挂载方式
✅ **热更新机制**：实现配置的动态更新
✅ **版本管理**：配置的版本控制和回滚
✅ **故障排查**：常见配置问题的诊断和解决

### 下一步学习建议

1. **学习Helm**：更高级的配置模板管理
2. **外部配置中心**：如Consul、etcd集成
3. **配置加密**：使用external-secrets等工具
4. **GitOps**：配置的版本化管理
5. **策略管理**：OPA/Gatekeeper配置验证

继续下一个实验：**实验6：数据持久化与存储** 