# 实验9：Docker Swarm集群管理

## 学习目标
- 理解Docker Swarm集群架构和原理
- 掌握Swarm集群的创建和管理
- 学会服务的部署、扩展和更新
- 理解服务发现和负载均衡机制
- 掌握集群安全和故障处理

## 理论学习

### Docker Swarm简介
Docker Swarm是Docker官方提供的容器编排和集群管理工具，可以将多个Docker主机组成一个虚拟的Docker主机集群。

### 核心概念

| 概念 | 说明 | 作用 |
|------|------|------|
| Swarm | Docker集群 | 包含多个Docker节点的集群 |
| Node | 集群节点 | 运行Docker Engine的主机 |
| Manager | 管理节点 | 负责集群管理和任务调度 |
| Worker | 工作节点 | 执行任务的节点 |
| Service | 服务 | 在集群中运行的应用定义 |
| Task | 任务 | 服务的一个运行实例 |
| Stack | 堆栈 | 一组相关服务的集合 |

### Swarm架构
```
┌─────────────────────────────────────────────────────────┐
│                    Swarm Cluster                        │
├─────────────────────────────────────────────────────────┤
│  Manager Nodes                  │  Worker Nodes         │
│  ┌─────────────┐                │  ┌─────────────┐      │
│  │   Manager   │ ◄─────────────► │  │   Worker    │      │
│  │   (Leader)  │                │  │             │      │
│  └─────────────┘                │  └─────────────┘      │
│  ┌─────────────┐                │  ┌─────────────┐      │
│  │   Manager   │                │  │   Worker    │      │
│  │ (Follower)  │                │  │             │      │
│  └─────────────┘                │  └─────────────┘      │
└─────────────────────────────────────────────────────────┘
```

### 集群管理原理
- **Raft算法**: 管理节点使用Raft算法保证集群状态一致性
- **任务调度**: Manager节点负责将任务分配给Worker节点
- **服务发现**: 内置DNS和负载均衡
- **Rolling Update**: 支持服务的滚动更新

## 动手实践

### 1. 集群初始化和节点管理

#### 1.1 初始化Swarm集群
```bash
# 在主节点初始化集群
docker swarm init --advertise-addr <MANAGER-IP>

# 输出类似:
# Swarm initialized: current node (xxx) is now a manager.
# To add a worker to this swarm, run the following command:
# docker swarm join --token SWMTKN-1-xxx <MANAGER-IP>:2377

# 查看集群状态
docker info | grep Swarm
docker node ls
```

#### 1.2 添加工作节点
```bash
# 在其他主机上执行（使用初始化时获得的命令）
docker swarm join --token SWMTKN-1-xxx <MANAGER-IP>:2377

# 在管理节点查看所有节点
docker node ls
```

#### 1.3 添加管理节点
```bash
# 获取管理节点加入令牌
docker swarm join-token manager

# 在新节点上执行返回的命令
docker swarm join --token SWMTKN-1-xxx <MANAGER-IP>:2377
```

#### 1.4 节点管理
```bash
# 查看节点详细信息
docker node inspect <NODE-ID>

# 将工作节点提升为管理节点
docker node promote <NODE-ID>

# 将管理节点降级为工作节点
docker node demote <NODE-ID>

# 节点下线（维护模式）
docker node update --availability drain <NODE-ID>

# 节点上线
docker node update --availability active <NODE-ID>

# 移除节点
docker node rm <NODE-ID>
```

### 2. 服务管理

#### 2.1 创建服务
```bash
# 创建简单服务
docker service create --name web --replicas 3 --publish 8080:80 nginx

# 查看服务列表
docker service ls

# 查看服务详情
docker service inspect web

# 查看服务任务（容器实例）
docker service ps web
```

#### 2.2 服务配置选项
```bash
# 带环境变量的服务
docker service create \
  --name app \
  --replicas 2 \
  --env NODE_ENV=production \
  --env PORT=3000 \
  node:alpine node app.js

# 带资源限制的服务
docker service create \
  --name limited-app \
  --replicas 2 \
  --limit-cpu 0.5 \
  --limit-memory 512M \
  --reserve-cpu 0.25 \
  --reserve-memory 256M \
  nginx

# 带挂载卷的服务
docker service create \
  --name web-with-volume \
  --replicas 2 \
  --mount type=volume,source=web-data,target=/usr/share/nginx/html \
  nginx
```

#### 2.3 服务扩展
```bash
# 扩展服务实例
docker service scale web=5

# 同时扩展多个服务
docker service scale web=3 app=4

# 查看扩展结果
docker service ps web
```

#### 2.4 服务更新
```bash
# 更新服务镜像
docker service update --image nginx:1.20 web

# 更新服务端口
docker service update --publish-add 9090:80 web

# 更新服务环境变量
docker service update --env-add NEW_VAR=value web

# 配置滚动更新策略
docker service update \
  --update-delay 10s \
  --update-parallelism 2 \
  --update-failure-action rollback \
  web
```

#### 2.5 服务回滚
```bash
# 回滚到上一个版本
docker service rollback web

# 查看服务更新历史
docker service ps web
```

### 3. 网络管理

#### 3.1 创建Overlay网络
```bash
# 创建overlay网络
docker network create --driver overlay my-overlay

# 查看网络
docker network ls

# 创建带加密的网络
docker network create \
  --driver overlay \
  --opt encrypted \
  secure-network
```

#### 3.2 服务网络配置
```bash
# 在指定网络中创建服务
docker service create \
  --name web \
  --network my-overlay \
  --replicas 3 \
  nginx

# 连接服务到多个网络
docker service update \
  --network-add frontend \
  --network-add backend \
  web
```

### 4. 数据管理

#### 4.1 数据卷管理
```bash
# 创建数据卷
docker volume create --driver local web-data

# 在服务中使用数据卷
docker service create \
  --name web \
  --mount type=volume,source=web-data,target=/data \
  nginx
```

#### 4.2 配置和秘钥管理
```bash
# 创建配置
echo "worker_processes auto;" | docker config create nginx-config -

# 创建秘钥
echo "mypassword" | docker secret create db-password -

# 在服务中使用配置和秘钥
docker service create \
  --name web \
  --config source=nginx-config,target=/etc/nginx/nginx.conf \
  --secret source=db-password,target=/run/secrets/db_password \
  nginx
```

### 5. 服务堆栈（Stack）

#### 5.1 编写Stack文件
```yaml
# docker-stack.yml
version: '3.8'
services:
  web:
    image: nginx:latest
    ports:
      - "80:80"
    networks:
      - webnet
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
  
  redis:
    image: redis:alpine
    networks:
      - webnet
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager

networks:
  webnet:
    driver: overlay

volumes:
  web-data:
```

#### 5.2 部署和管理Stack
```bash
# 部署堆栈
docker stack deploy -c docker-stack.yml mystack

# 查看堆栈列表
docker stack ls

# 查看堆栈服务
docker stack services mystack

# 查看堆栈任务
docker stack ps mystack

# 移除堆栈
docker stack rm mystack
```

## 实验任务

### 任务1：高可用Web应用集群
创建一个包含负载均衡器、Web服务器和数据库的高可用应用：

```yaml
# ha-webapp-stack.yml
version: '3.8'
services:
  # 负载均衡器
  lb:
    image: nginx:alpine
    ports:
      - "80:80"
    configs:
      - nginx-config
    networks:
      - frontend
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.role == manager

  # Web应用
  web:
    image: nginx:latest
    networks:
      - frontend
      - backend
    deploy:
      replicas: 5
      update_config:
        parallelism: 2
        delay: 10s
        failure_action: rollback
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M

  # 数据库
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: webapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - db-data:/var/lib/postgresql/data
    networks:
      - backend
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager
      restart_policy:
        condition: on-failure

  # Redis缓存
  redis:
    image: redis:alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - backend
    deploy:
      replicas: 1
      restart_policy:
        condition: on-failure

configs:
  nginx-config:
    external: true

secrets:
  db_password:
    external: true

networks:
  frontend:
    driver: overlay
  backend:
    driver: overlay

volumes:
  db-data:
  redis-data:
```

```bash
# 创建配置和秘钥
echo "upstream backend {
    server web:80;
}
server {
    listen 80;
    location / {
        proxy_pass http://backend;
    }
}" | docker config create nginx-config -

echo "secretpassword123" | docker secret create db_password -

# 部署堆栈
docker stack deploy -c ha-webapp-stack.yml webapp
```

### 任务2：微服务架构部署
部署一个完整的微服务应用：

```yaml
# microservices-stack.yml
version: '3.8'
services:
  # API网关
  gateway:
    image: nginx:alpine
    ports:
      - "80:80"
    configs:
      - gateway-config
    networks:
      - public
      - internal
    deploy:
      replicas: 2

  # 用户服务
  user-service:
    image: myregistry/user-service:latest
    networks:
      - internal
    environment:
      - DATABASE_URL=postgresql://user:pass@user-db:5432/users
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 30s

  # 订单服务
  order-service:
    image: myregistry/order-service:latest
    networks:
      - internal
    environment:
      - DATABASE_URL=postgresql://user:pass@order-db:5432/orders
      - USER_SERVICE_URL=http://user-service:8080
    deploy:
      replicas: 3

  # 用户数据库
  user-db:
    image: postgres:13
    environment:
      POSTGRES_DB: users
      POSTGRES_USER: user
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - user-db-data:/var/lib/postgresql/data
    networks:
      - internal
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.database == true

  # 订单数据库
  order-db:
    image: postgres:13
    environment:
      POSTGRES_DB: orders
      POSTGRES_USER: user
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - order-db-data:/var/lib/postgresql/data
    networks:
      - internal
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.database == true

configs:
  gateway-config:
    external: true

secrets:
  db_password:
    external: true

networks:
  public:
    driver: overlay
  internal:
    driver: overlay

volumes:
  user-db-data:
  order-db-data:
```

### 任务3：监控和日志系统
部署完整的监控堆栈：

```yaml
# monitoring-stack.yml
version: '3.8'
services:
  # Prometheus监控
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    configs:
      - prometheus-config
    volumes:
      - prometheus-data:/prometheus
    networks:
      - monitoring
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager

  # Grafana可视化
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD__FILE=/run/secrets/grafana_password
    secrets:
      - grafana_password
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - monitoring
    deploy:
      replicas: 1

  # 节点监控
  node-exporter:
    image: prom/node-exporter
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - monitoring
    deploy:
      mode: global

  # 容器监控
  cadvisor:
    image: gcr.io/cadvisor/cadvisor
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:rw
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    networks:
      - monitoring
    deploy:
      mode: global

configs:
  prometheus-config:
    external: true

secrets:
  grafana_password:
    external: true

networks:
  monitoring:
    driver: overlay

volumes:
  prometheus-data:
  grafana-data:
```

## 进阶练习

### 1. 集群升级和维护
```bash
# 节点维护模式
docker node update --availability drain node-1

# 等待任务迁移完成
docker service ps service-name

# 节点升级完成后重新上线
docker node update --availability active node-1
```

### 2. 集群备份和恢复
```bash
# 备份Swarm状态
sudo cp -r /var/lib/docker/swarm /backup/swarm-backup-$(date +%Y%m%d)

# 恢复集群（在新环境中）
sudo docker swarm init --force-new-cluster
```

### 3. 安全配置
```bash
# 轮换加入令牌
docker swarm join-token --rotate worker
docker swarm join-token --rotate manager

# 启用自动锁定
docker swarm update --autolock=true

# 解锁集群
docker swarm unlock
```

### 4. 故障模拟和恢复
```bash
# 模拟节点故障
docker node update --availability drain node-2

# 查看服务重新调度
docker service ps web

# 模拟网络分区
# 使用iptables或网络工具模拟网络故障
```

## 生产环境最佳实践

### 1. 集群规划
- **管理节点数量**: 推荐奇数个管理节点（3或5个）
- **节点分布**: 管理节点分布在不同可用区
- **资源规划**: 合理分配CPU、内存和存储资源

### 2. 安全配置
```bash
# 启用集群自动锁定
docker swarm update --autolock=true

# 定期轮换令牌
docker swarm join-token --rotate worker

# 使用TLS证书
docker swarm ca --rotate
```

### 3. 监控和告警
- 监控集群健康状态
- 监控服务可用性
- 设置资源使用告警
- 配置日志收集

### 4. 备份策略
- 定期备份Swarm状态
- 备份配置和秘钥
- 测试恢复流程

## 故障排查指南

### 常见问题

1. **节点离线**
   ```bash
   # 查看节点状态
   docker node ls
   
   # 检查网络连通性
   ping <NODE-IP>
   
   # 检查Docker守护进程
   systemctl status docker
   ```

2. **服务启动失败**
   ```bash
   # 查看服务任务状态
   docker service ps <SERVICE-NAME>
   
   # 查看任务日志
   docker service logs <SERVICE-NAME>
   
   # 检查资源约束
   docker service inspect <SERVICE-NAME>
   ```

3. **网络连接问题**
   ```bash
   # 检查overlay网络
   docker network ls
   docker network inspect <NETWORK-NAME>
   
   # 测试服务间连通性
   docker exec <CONTAINER-ID> nslookup <SERVICE-NAME>
   ```

### 调试工具
```bash
# 进入服务容器调试
docker exec -it $(docker ps -q -f name=<SERVICE-NAME>) sh

# 查看集群事件
docker system events

# 检查集群状态
docker system df
docker system info
```

## 思考题

1. **为什么Swarm管理节点需要奇数个？Raft算法的作用是什么？**

2. **Swarm的服务发现机制是如何工作的？**

3. **如何在Swarm中实现数据库的高可用部署？**

4. **Swarm与Kubernetes相比有什么优缺点？**

5. **在生产环境中如何保证Swarm集群的安全性？**

## 常用命令速查

### 集群管理
```bash
# 初始化集群
docker swarm init --advertise-addr <IP>

# 查看集群信息
docker info
docker node ls

# 获取加入令牌
docker swarm join-token worker
docker swarm join-token manager

# 离开集群
docker swarm leave --force
```

### 服务管理
```bash
# 创建服务
docker service create --name <NAME> <IMAGE>

# 查看服务
docker service ls
docker service ps <SERVICE>
docker service inspect <SERVICE>

# 更新服务
docker service update <SERVICE>
docker service scale <SERVICE>=<NUM>

# 删除服务
docker service rm <SERVICE>
```

### 堆栈管理
```bash
# 部署堆栈
docker stack deploy -c <FILE> <STACK>

# 查看堆栈
docker stack ls
docker stack services <STACK>
docker stack ps <STACK>

# 删除堆栈
docker stack rm <STACK>
```

### 网络管理
```bash
# 创建网络
docker network create --driver overlay <NETWORK>

# 查看网络
docker network ls
docker network inspect <NETWORK>

# 删除网络
docker network rm <NETWORK>
```

## 参考资料

- [Docker Swarm官方文档](https://docs.docker.com/engine/swarm/)
- [Swarm模式概览](https://docs.docker.com/engine/swarm/key-concepts/)
- [生产环境部署指南](https://docs.docker.com/engine/swarm/admin_guide/)
- [Raft算法详解](https://raft.github.io/)

## 下一步
完成本实验后，继续学习：
- **实验10**: 生产环境部署与监控
- 深入了解容器化应用的生产部署
- 学习企业级容器管理 