# 实验6：数据持久化与存储

## 🎯 学习目标

通过本实验，你将：
- 理解Kubernetes存储架构和核心概念
- 掌握Volume、PersistentVolume、PersistentVolumeClaim的使用
- 学习StorageClass动态存储分配
- 实践StatefulSet有状态应用管理
- 了解不同存储类型的特点和使用场景
- 掌握存储监控、扩容和备份策略

## 📚 理论知识学习

### Kubernetes存储架构

Kubernetes提供了灵活的存储抽象层，将存储需求与具体的存储实现分离。

#### 存储层次结构
```
┌─────────────────── 应用层 ──────────────────┐
│                Pod                         │
│  ┌─────────────────────────────────────┐   │
│  │           Container                 │   │
│  │  VolumeMounts: /data               │   │
│  └─────────────────────────────────────┘   │
└────────────┬────────────────────────────────┘
             │
┌─────────────▼─── Volume抽象层 ─────────────┐
│               Volume                      │
│  ┌─────────────────────────────────────┐   │
│  │      PersistentVolumeClaim          │   │
│  └─────────────────────────────────────┘   │
└────────────┬────────────────────────────────┘
             │
┌─────────────▼─── 存储管理层 ───────────────┐
│            PersistentVolume               │
│  ┌─────────────────────────────────────┐   │
│  │         StorageClass                │   │
│  └─────────────────────────────────────┘   │
└────────────┬────────────────────────────────┘
             │
┌─────────────▼─── 物理存储层 ───────────────┐
│     Local Disk, NFS, Cloud Storage        │
└───────────────────────────────────────────┘
```

### 核心存储概念

#### 1. Volume类型对比

| Volume类型 | 生命周期 | 数据持久性 | 使用场景 |
|------------|----------|------------|----------|
| **emptyDir** | Pod生命周期 | 临时 | 缓存、临时文件 |
| **hostPath** | 节点生命周期 | 持久（单节点） | 日志收集、系统文件 |
| **PersistentVolume** | 独立于Pod | 持久 | 数据库、文件存储 |
| **configMap/secret** | 配置生命周期 | 配置数据 | 配置文件、密钥 |

#### 2. PV/PVC模式
- **PersistentVolume (PV)**: 集群级别的存储资源
- **PersistentVolumeClaim (PVC)**: 用户对存储的请求
- **StorageClass**: 动态存储分配的模板

## 🔧 Volume基础使用

### 准备工作：创建实验目录

```bash
# 创建实验目录
mkdir -p ~/k8s-labs/lab06
cd ~/k8s-labs/lab06
```

### 1. emptyDir Volume

emptyDir在Pod创建时创建，Pod删除时销毁，主要用于临时存储。

```bash
cat > emptydir-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: emptydir-pod
spec:
  containers:
  # 写入容器
  - name: writer
    image: busybox
    command: ["/bin/sh"]
    args:
    - -c
    - |
      while true; do
        echo "Writing data at $(date)" >> /shared-data/data.log
        sleep 10
      done
    volumeMounts:
    - name: shared-storage
      mountPath: /shared-data
  
  # 读取容器
  - name: reader
    image: busybox
    command: ["/bin/sh"]
    args:
    - -c
    - |
      while true; do
        echo "=== Latest Data ==="
        tail -5 /shared-data/data.log 2>/dev/null || echo "No data yet"
        sleep 15
      done
    volumeMounts:
    - name: shared-storage
      mountPath: /shared-data
  
  volumes:
  - name: shared-storage
    emptyDir: {}
EOF

# 部署并测试
kubectl apply -f emptydir-pod.yaml

# 查看容器日志
kubectl logs emptydir-pod -c writer
kubectl logs emptydir-pod -c reader

# 进入容器查看数据
kubectl exec -it emptydir-pod -c reader -- ls -la /shared-data/
```

### 2. hostPath Volume

hostPath将节点上的文件或目录挂载到Pod中。

```bash
cat > hostpath-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: hostpath-pod
spec:
  containers:
  - name: log-collector
    image: busybox
    command: ["/bin/sh"]
    args:
    - -c
    - |
      while true; do
        echo "Collecting logs at $(date)" >> /host-logs/app.log
        echo "Host info: $(uname -a)" >> /host-logs/system.log
        sleep 30
      done
    volumeMounts:
    - name: host-logs
      mountPath: /host-logs
    securityContext:
      runAsUser: 0  # 以root身份运行以写入主机目录
  
  volumes:
  - name: host-logs
    hostPath:
      path: /tmp/k8s-logs
      type: DirectoryOrCreate  # 如果目录不存在则创建
EOF

kubectl apply -f hostpath-pod.yaml

# 在Docker Desktop中查看主机文件
# macOS: ~/Library/Containers/com.docker.docker/Data/vms/0/tty
# 在Minikube中查看
minikube ssh
ls -la /tmp/k8s-logs/
```

### 3. 配置和密钥Volume

```bash
# 创建ConfigMap
kubectl create configmap app-config \
  --from-literal=database_url="mysql://localhost:3306/app" \
  --from-literal=redis_url="redis://localhost:6379" \
  --from-literal=log_level="info"

# 创建Secret
kubectl create secret generic app-secret \
  --from-literal=db_password="supersecret" \
  --from-literal=api_key="abc123xyz"

cat > config-volume-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: config-volume-pod
spec:
  containers:
  - name: app
    image: busybox
    command: ["/bin/sh"]
    args:
    - -c
    - |
      echo "=== Configuration Files ==="
      echo "Config files:"
      ls -la /etc/config/
      echo "Secret files:"
      ls -la /etc/secrets/
      echo
      echo "=== Configuration Content ==="
      cat /etc/config/database_url
      echo
      echo "=== Environment Variables ==="
      env | grep -E "(DB_|API_)"
      echo
      echo "Application started with configuration"
      sleep 3600
    volumeMounts:
    - name: config-volume
      mountPath: /etc/config
    - name: secret-volume
      mountPath: /etc/secrets
    env:
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: app-secret
          key: db_password
    - name: API_KEY
      valueFrom:
        secretKeyRef:
          name: app-secret
          key: api_key
  
  volumes:
  - name: config-volume
    configMap:
      name: app-config
  - name: secret-volume
    secret:
      secretName: app-secret
      defaultMode: 0400  # 只读权限
EOF

kubectl apply -f config-volume-pod.yaml

# 验证配置挂载
kubectl exec config-volume-pod -- cat /etc/config/database_url
kubectl exec config-volume-pod -- ls -la /etc/secrets/
```

## 💾 PersistentVolume和PersistentVolumeClaim

### 1. 静态PV/PVC配置

#### 创建PersistentVolume
```bash
cat > static-pv.yaml << 'EOF'
apiVersion: v1
kind: PersistentVolume
metadata:
  name: static-pv
  labels:
    type: local
spec:
  capacity:
    storage: 1Gi
  accessModes:
  - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: manual
  hostPath:
    path: /tmp/k8s-static-pv
    type: DirectoryOrCreate
EOF

kubectl apply -f static-pv.yaml
kubectl get pv
```

#### 创建PersistentVolumeClaim
```bash
cat > static-pvc.yaml << 'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: static-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 500Mi
  storageClassName: manual
EOF

kubectl apply -f static-pvc.yaml
kubectl get pvc
kubectl describe pvc static-pvc
```

#### 使用PVC的Pod
```bash
cat > pvc-pod.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: pvc-pod
spec:
  containers:
  - name: app
    image: nginx:alpine
    ports:
    - containerPort: 80
    volumeMounts:
    - name: persistent-storage
      mountPath: /usr/share/nginx/html
    lifecycle:
      postStart:
        exec:
          command:
          - /bin/sh
          - -c
          - |
            echo "<h1>Persistent Storage Demo</h1>" > /usr/share/nginx/html/index.html
            echo "<p>Pod: $HOSTNAME</p>" >> /usr/share/nginx/html/index.html
            echo "<p>Time: $(date)</p>" >> /usr/share/nginx/html/index.html
  
  volumes:
  - name: persistent-storage
    persistentVolumeClaim:
      claimName: static-pvc
EOF

kubectl apply -f pvc-pod.yaml

# 测试持久化存储
kubectl port-forward pvc-pod 8080:80
# 访问 http://localhost:8080

# 删除Pod并重新创建，验证数据持久性
kubectl delete pod pvc-pod
kubectl apply -f pvc-pod.yaml
kubectl port-forward pvc-pod 8080:80
```

### 2. StorageClass动态存储

#### 查看默认StorageClass
```bash
# 查看可用的StorageClass
kubectl get storageclass

# 查看默认StorageClass
kubectl get storageclass -o yaml
```

#### 创建动态PVC
```bash
cat > dynamic-pvc.yaml << 'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dynamic-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi
  # 使用默认StorageClass，会自动创建PV
EOF

kubectl apply -f dynamic-pvc.yaml

# 观察PV自动创建
kubectl get pv,pvc
```

### 3. 存储访问模式

| 访问模式 | 简写 | 描述 |
|----------|------|------|
| ReadWriteOnce | RWO | 单节点读写 |
| ReadOnlyMany | ROX | 多节点只读 |
| ReadWriteMany | RWX | 多节点读写 |

```bash
cat > access-modes-test.yaml << 'EOF'
# ReadWriteOnce示例
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rwo-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
---
# 尝试ReadWriteMany（在单节点集群中可能不支持）
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rwx-pvc
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 1Gi
EOF

kubectl apply -f access-modes-test.yaml
kubectl get pvc
```

## 🏗️ StatefulSet有状态应用

StatefulSet为有状态应用提供稳定的网络标识和存储。

### 1. StatefulSet基础示例

```bash
cat > basic-statefulset.yaml << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: nginx-headless
spec:
  clusterIP: None  # Headless Service
  selector:
    app: nginx-sts
  ports:
  - port: 80
    targetPort: 80
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: nginx-statefulset
spec:
  serviceName: nginx-headless
  replicas: 3
  selector:
    matchLabels:
      app: nginx-sts
  template:
    metadata:
      labels:
        app: nginx-sts
    spec:
      containers:
      - name: nginx
        image: nginx:alpine
        ports:
        - containerPort: 80
        volumeMounts:
        - name: nginx-storage
          mountPath: /usr/share/nginx/html
        lifecycle:
          postStart:
            exec:
              command:
              - /bin/sh
              - -c
              - |
                echo "<h1>StatefulSet Pod: $HOSTNAME</h1>" > /usr/share/nginx/html/index.html
                echo "<p>Persistent storage for $HOSTNAME</p>" >> /usr/share/nginx/html/index.html
  
  volumeClaimTemplates:
  - metadata:
      name: nginx-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 500Mi
EOF

kubectl apply -f basic-statefulset.yaml

# 观察StatefulSet部署过程
kubectl get pods -w -l app=nginx-sts

# 查看Pod和PVC
kubectl get pods,pvc -l app=nginx-sts
```

### 2. 验证StatefulSet特性

```bash
# 1. 稳定的网络身份
kubectl run dns-test --image=busybox -it --rm -- sh
# 在Pod内执行：
# nslookup nginx-statefulset-0.nginx-headless
# nslookup nginx-statefulset-1.nginx-headless

# 2. 有序部署和缩放
kubectl scale statefulset nginx-statefulset --replicas=5
kubectl get pods -w -l app=nginx-sts

kubectl scale statefulset nginx-statefulset --replicas=2
kubectl get pods -w -l app=nginx-sts

# 3. 持久化存储
kubectl exec nginx-statefulset-0 -- echo "Data from Pod 0" > /usr/share/nginx/html/data.txt
kubectl delete pod nginx-statefulset-0
# Pod重新创建后数据仍然存在
kubectl exec nginx-statefulset-0 -- cat /usr/share/nginx/html/data.txt
```

### 3. MongoDB StatefulSet实例

```bash
cat > mongodb-statefulset.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: mongodb-config
data:
  mongodb.conf: |
    storage:
      dbPath: /data/db
    net:
      port: 27017
      bindIp: 0.0.0.0
    replication:
      replSetName: rs0
---
apiVersion: v1
kind: Service
metadata:
  name: mongodb-headless
spec:
  clusterIP: None
  selector:
    app: mongodb
  ports:
  - port: 27017
    targetPort: 27017
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mongodb
spec:
  serviceName: mongodb-headless
  replicas: 3
  selector:
    matchLabels:
      app: mongodb
  template:
    metadata:
      labels:
        app: mongodb
    spec:
      containers:
      - name: mongodb
        image: mongo:5.0
        ports:
        - containerPort: 27017
        env:
        - name: MONGO_INITDB_ROOT_USERNAME
          value: "admin"
        - name: MONGO_INITDB_ROOT_PASSWORD
          value: "password123"
        volumeMounts:
        - name: mongodb-data
          mountPath: /data/db
        - name: mongodb-config
          mountPath: /etc/mongodb
        command:
        - mongod
        - --config=/etc/mongodb/mongodb.conf
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      
      volumes:
      - name: mongodb-config
        configMap:
          name: mongodb-config
  
  volumeClaimTemplates:
  - metadata:
      name: mongodb-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 2Gi
EOF

kubectl apply -f mongodb-statefulset.yaml

# 监控部署过程
kubectl get pods -w -l app=mongodb

# 测试MongoDB连接
kubectl exec -it mongodb-0 -- mongo --username admin --password password123 --authenticationDatabase admin
```

## 📊 存储监控与管理

### 1. 存储使用情况监控

```bash
# 查看PV使用情况
kubectl get pv

# 查看PVC状态
kubectl get pvc --all-namespaces

# 查看存储容量
kubectl describe node | grep -A 5 "Allocated resources"

# 存储监控脚本
cat > storage-monitor.sh << 'EOF'
#!/bin/bash

echo "=== Kubernetes Storage Monitor ==="
echo "Date: $(date)"
echo

echo "=== PersistentVolumes ==="
kubectl get pv -o custom-columns=NAME:.metadata.name,CAPACITY:.spec.capacity.storage,ACCESS:.spec.accessModes[0],RECLAIM:.spec.persistentVolumeReclaimPolicy,STATUS:.status.phase,CLAIM:.spec.claimRef.name

echo -e "\n=== PersistentVolumeClaims ==="
kubectl get pvc --all-namespaces -o custom-columns=NAMESPACE:.metadata.namespace,NAME:.metadata.name,STATUS:.status.phase,VOLUME:.spec.volumeName,CAPACITY:.status.capacity.storage,ACCESS:.spec.accessModes[0]

echo -e "\n=== StorageClasses ==="
kubectl get storageclass

echo -e "\n=== Node Storage Usage ==="
kubectl top nodes 2>/dev/null || echo "Metrics server not available"
EOF

chmod +x storage-monitor.sh
./storage-monitor.sh
```

### 2. 存储扩容

```bash
# 创建可扩容的PVC
cat > expandable-pvc.yaml << 'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: expandable-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
EOF

kubectl apply -f expandable-pvc.yaml

# 扩容PVC（需要StorageClass支持）
kubectl patch pvc expandable-pvc -p '{"spec":{"resources":{"requests":{"storage":"2Gi"}}}}'

# 查看扩容状态
kubectl describe pvc expandable-pvc
```

### 3. 存储回收策略

```bash
cat > reclaim-policy-demo.yaml << 'EOF'
# Retain策略PV
apiVersion: v1
kind: PersistentVolume
metadata:
  name: retain-pv
spec:
  capacity:
    storage: 1Gi
  accessModes:
  - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain  # 手动回收
  storageClassName: manual
  hostPath:
    path: /tmp/retain-pv
---
# Recycle策略PV（已废弃，仅作示例）
apiVersion: v1
kind: PersistentVolume
metadata:
  name: recycle-pv
spec:
  capacity:
    storage: 1Gi
  accessModes:
  - ReadWriteOnce
  persistentVolumeReclaimPolicy: Delete  # 自动删除
  storageClassName: manual
  hostPath:
    path: /tmp/recycle-pv
EOF
```

## 🛠️ 实验练习

### 练习1：多级存储架构

创建一个模拟真实应用的多级存储系统：

```bash
cat > multi-tier-storage.yaml << 'EOF'
# 缓存层：快速临时存储
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cache-tier
spec:
  replicas: 2
  selector:
    matchLabels:
      app: cache
  template:
    metadata:
      labels:
        app: cache
    spec:
      containers:
      - name: redis
        image: redis:alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: cache-data
          mountPath: /data
      volumes:
      - name: cache-data
        emptyDir:
          medium: Memory  # 内存存储
          sizeLimit: 256Mi
---
# 应用层：持久化存储
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: app-tier
spec:
  serviceName: app-headless
  replicas: 2
  selector:
    matchLabels:
      app: application
  template:
    metadata:
      labels:
        app: application
    spec:
      containers:
      - name: app
        image: nginx:alpine
        ports:
        - containerPort: 80
        volumeMounts:
        - name: app-data
          mountPath: /data
        - name: config
          mountPath: /etc/nginx/conf.d
      volumes:
      - name: config
        configMap:
          name: app-config
  volumeClaimTemplates:
  - metadata:
      name: app-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 1Gi
---
# 数据层：大容量存储
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: database-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
EOF
```

### 练习2：存储性能测试

```bash
cat > storage-benchmark.yaml << 'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: storage-benchmark
spec:
  containers:
  - name: benchmark
    image: busybox
    command: ["/bin/sh"]
    args:
    - -c
    - |
      echo "=== Storage Performance Test ==="
      
      # 写入测试
      echo "Write test (1MB x 100):"
      time for i in $(seq 1 100); do
        dd if=/dev/zero of=/test-data/testfile$i bs=1M count=1 2>/dev/null
      done
      
      # 读取测试
      echo "Read test:"
      time cat /test-data/testfile* > /dev/null
      
      # 随机读写测试
      echo "Random I/O test:"
      time for i in $(seq 1 50); do
        dd if=/dev/urandom of=/test-data/random$i bs=1K count=1 2>/dev/null
      done
      
      echo "Test completed"
      sleep 3600
    volumeMounts:
    - name: test-storage
      mountPath: /test-data
  volumes:
  - name: test-storage
    persistentVolumeClaim:
      claimName: benchmark-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: benchmark-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi
EOF

kubectl apply -f storage-benchmark.yaml
kubectl logs storage-benchmark -f
```

## 🧪 进阶实验

### 实验1：存储快照和克隆

```bash
# VolumeSnapshot功能（需要CSI驱动支持）
cat > volume-snapshot.yaml << 'EOF'
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: data-snapshot
spec:
  volumeSnapshotClassName: csi-hostpath-snapclass
  source:
    persistentVolumeClaimName: dynamic-pvc
---
# 从快照恢复
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: restored-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi
  dataSource:
    name: data-snapshot
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
EOF
```

### 实验2：存储备份策略

```bash
cat > backup-cronjob.yaml << 'EOF'
apiVersion: batch/v1
kind: CronJob
metadata:
  name: storage-backup
spec:
  schedule: "0 2 * * *"  # 每天凌晨2点
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: busybox
            command: ["/bin/sh"]
            args:
            - -c
            - |
              BACKUP_DIR="/backup/$(date +%Y%m%d_%H%M%S)"
              mkdir -p $BACKUP_DIR
              
              echo "Starting backup at $(date)"
              cp -r /data/* $BACKUP_DIR/
              
              # 创建备份清单
              echo "Backup created: $(date)" > $BACKUP_DIR/backup.info
              ls -la /data/ >> $BACKUP_DIR/backup.info
              
              echo "Backup completed: $BACKUP_DIR"
              
              # 清理7天前的备份
              find /backup -type d -mtime +7 -exec rm -rf {} \;
            volumeMounts:
            - name: source-data
              mountPath: /data
              readOnly: true
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: source-data
            persistentVolumeClaim:
              claimName: static-pvc
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: backup-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF
```

## 🐛 故障排查指南

### 常见存储问题

#### 1. PVC一直处于Pending状态
```bash
# 检查PVC详情
kubectl describe pvc <pvc-name>

# 常见原因：
# - 没有合适的PV可用
kubectl get pv

# - StorageClass不存在或配置错误
kubectl get storageclass

# - 访问模式不匹配
kubectl describe pv <pv-name> | grep "Access Modes"
```

#### 2. Pod无法挂载存储
```bash
# 检查Pod事件
kubectl describe pod <pod-name>

# 检查节点存储情况
kubectl describe node <node-name> | grep -A 10 "Allocated resources"

# 检查PVC绑定状态
kubectl get pvc
```

#### 3. 存储空间不足
```bash
# 检查PVC使用情况
kubectl get pvc -o custom-columns=NAME:.metadata.name,CAPACITY:.status.capacity.storage,USED:.status.allocatedResources.storage

# 扩容PVC
kubectl patch pvc <pvc-name> -p '{"spec":{"resources":{"requests":{"storage":"<new-size>"}}}}'
```

### 存储诊断脚本

```bash
cat > storage-troubleshoot.sh << 'EOF'
#!/bin/bash

PVC_NAME=$1
if [ -z "$PVC_NAME" ]; then
    echo "Usage: $0 <pvc-name>"
    exit 1
fi

echo "=== Storage Troubleshooting: $PVC_NAME ==="

echo "1. PVC Status:"
kubectl describe pvc $PVC_NAME

echo -e "\n2. Bound PV:"
PV_NAME=$(kubectl get pvc $PVC_NAME -o jsonpath='{.spec.volumeName}')
if [ ! -z "$PV_NAME" ]; then
    kubectl describe pv $PV_NAME
else
    echo "No PV bound to this PVC"
fi

echo -e "\n3. StorageClass:"
SC_NAME=$(kubectl get pvc $PVC_NAME -o jsonpath='{.spec.storageClassName}')
if [ ! -z "$SC_NAME" ]; then
    kubectl describe storageclass $SC_NAME
else
    echo "No StorageClass specified"
fi

echo -e "\n4. Related Events:"
kubectl get events --field-selector involvedObject.name=$PVC_NAME

echo -e "\n5. Pods using this PVC:"
kubectl get pods --all-namespaces -o json | jq -r '.items[] | select(.spec.volumes[]?.persistentVolumeClaim.claimName=="'$PVC_NAME'") | .metadata.namespace + "/" + .metadata.name'
EOF

chmod +x storage-troubleshoot.sh
```

## 💡 最佳实践

### 1. 存储选择指南

```yaml
# 根据用途选择存储类型
spec:
  volumes:
  # 临时数据、缓存
  - name: cache
    emptyDir:
      medium: Memory
      sizeLimit: 1Gi
  
  # 配置文件
  - name: config
    configMap:
      name: app-config
  
  # 持久化数据
  - name: data
    persistentVolumeClaim:
      claimName: app-data-pvc
```

### 2. 资源规划
- **容量规划**：预估存储增长，留出足够空间
- **性能要求**：根据IOPS需求选择存储类型
- **备份策略**：定期备份重要数据
- **监控告警**：设置存储使用率告警

### 3. 安全考虑
- **访问权限**：使用适当的文件权限
- **数据加密**：敏感数据使用加密存储
- **网络隔离**：存储网络与业务网络分离

## 📝 学习检查

完成本实验后，你应该能够回答：

1. **概念理解**：
   - Volume、PV、PVC之间的关系？
   - StatefulSet与Deployment在存储方面的区别？
   - 不同存储回收策略的影响？

2. **操作技能**：
   - 如何创建动态和静态PV？
   - 如何排查存储挂载失败的问题？
   - 如何实现存储的备份和恢复？

3. **实际应用**：
   - 什么场景下使用StatefulSet？
   - 如何设计多层存储架构？
   - 如何优化存储性能？

## 🔗 延伸学习

- 学习CSI (Container Storage Interface)
- 了解云存储服务集成
- 探索分布式存储系统
- 研究存储性能优化技术

## ⏭️ 下一步

完成本实验后，继续学习：
- **实验7**：网络策略与安全 - 学习网络隔离和安全策略
- 探索Kubernetes的安全机制和最佳实践

---

**恭喜完成存储实验！** 🎉
你现在已经掌握了Kubernetes存储的核心概念，可以为各种应用设计合适的存储方案。 