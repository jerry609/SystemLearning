# 实验10：生产环境部署与监控

## 学习目标
- 掌握生产环境Docker部署策略
- 学会容器化应用的监控和日志管理
- 理解CI/CD流水线与Docker集成
- 掌握容器安全最佳实践
- 学会性能优化和故障排查

## 理论学习

### 生产环境特点
- **高可用性**: 服务不间断运行
- **可扩展性**: 根据负载自动扩缩容
- **安全性**: 数据保护和访问控制
- **可观测性**: 完整的监控和日志系统
- **可维护性**: 便于更新和故障处理

### 部署架构模式

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| 单机部署 | 简单，成本低 | 小型应用，开发测试 |
| 集群部署 | 高可用，可扩展 | 中大型应用 |
| 微服务架构 | 解耦，独立部署 | 复杂业务系统 |
| 混合云部署 | 灵活，容灾 | 企业级应用 |

## 动手实践

### 1. 生产环境镜像构建

#### 1.1 多阶段构建优化
```dockerfile
# 生产级Node.js应用
FROM node:16-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

FROM node:16-alpine AS production
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

WORKDIR /app
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --chown=nextjs:nodejs . .

USER nextjs
EXPOSE 3000
ENV NODE_ENV=production
CMD ["node", "server.js"]
```

#### 1.2 安全镜像构建
```dockerfile
# 安全的Python应用镜像
FROM python:3.9-slim AS base

# 创建非root用户
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

FROM base AS dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS production
WORKDIR /app
COPY --from=dependencies /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "app.py"]
```

### 2. 生产级Compose配置

#### 2.1 生产环境Stack
```yaml
# production-stack.yml
version: '3.8'
services:
  # 反向代理
  traefik:
    image: traefik:v2.8
    command:
      - "--api.insecure=false"
      - "--providers.docker=true"
      - "--providers.docker.swarmMode=true"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.letsencrypt.acme.httpchallenge=true"
      - "--certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web"
      - "--certificatesresolvers.letsencrypt.acme.email=admin@example.com"
      - "--certificatesresolvers.letsencrypt.acme.storage=/certificates/acme.json"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - traefik-certificates:/certificates
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.role == manager
    networks:
      - traefik-public

  # Web应用
  app:
    image: myregistry/myapp:${APP_VERSION:-latest}
    environment:
      - NODE_ENV=production
      - DATABASE_URL_FILE=/run/secrets/database_url
      - JWT_SECRET_FILE=/run/secrets/jwt_secret
    secrets:
      - database_url
      - jwt_secret
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 30s
        failure_action: rollback
        order: start-first
      restart_policy:
        condition: on-failure
      resources:
        limits:
          cpus: '1'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
      labels:
        - "traefik.enable=true"
        - "traefik.http.routers.app.rule=Host(`app.example.com`)"
        - "traefik.http.routers.app.entrypoints=websecure"
        - "traefik.http.routers.app.tls.certresolver=letsencrypt"
        - "traefik.http.services.app.loadbalancer.server.port=3000"
    networks:
      - traefik-public
      - backend
    depends_on:
      - database
      - redis

  # 数据库
  database:
    image: postgres:13
    environment:
      POSTGRES_DB: ${DB_NAME:-myapp}
      POSTGRES_USER: ${DB_USER:-admin}
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.database == true
      restart_policy:
        condition: on-failure
    networks:
      - backend
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER:-admin}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis缓存
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass-file /run/secrets/redis_password
    secrets:
      - redis_password
    volumes:
      - redis_data:/data
    deploy:
      replicas: 1
      restart_policy:
        condition: on-failure
    networks:
      - backend
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

secrets:
  database_url:
    external: true
  jwt_secret:
    external: true
  db_password:
    external: true
  redis_password:
    external: true

networks:
  traefik-public:
    external: true
  backend:
    driver: overlay

volumes:
  traefik-certificates:
  postgres_data:
  redis_data:
```

### 3. 监控系统部署

#### 3.1 Prometheus监控栈
```yaml
# monitoring-stack.yml
version: '3.8'
services:
  # Prometheus时序数据库
  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    configs:
      - source: prometheus_config
        target: /etc/prometheus/prometheus.yml
    volumes:
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - monitoring
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager

  # Grafana可视化
  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_USER=${GRAFANA_USER:-admin}
      - GF_SECURITY_ADMIN_PASSWORD__FILE=/run/secrets/grafana_password
      - GF_USERS_ALLOW_SIGN_UP=false
    secrets:
      - grafana_password
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    networks:
      - monitoring
    deploy:
      replicas: 1

  # AlertManager告警管理
  alertmanager:
    image: prom/alertmanager:latest
    configs:
      - source: alertmanager_config
        target: /etc/alertmanager/config.yml
    ports:
      - "9093:9093"
    networks:
      - monitoring
    deploy:
      replicas: 1

  # 节点监控
  node-exporter:
    image: prom/node-exporter:latest
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.ignored-mount-points=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - monitoring
    deploy:
      mode: global

  # 容器监控
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:rw
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    ports:
      - "8080:8080"
    networks:
      - monitoring
    deploy:
      mode: global

configs:
  prometheus_config:
    external: true
  alertmanager_config:
    external: true

secrets:
  grafana_password:
    external: true

networks:
  monitoring:
    driver: overlay

volumes:
  prometheus_data:
  grafana_data:
```

#### 3.2 日志收集系统
```yaml
# logging-stack.yml
version: '3.8'
services:
  # Elasticsearch存储
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - node.name=elasticsearch
      - cluster.name=docker-cluster
      - bootstrap.memory_lock=true
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - discovery.type=single-node
    ulimits:
      memlock:
        soft: -1
        hard: -1
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - logging
    deploy:
      replicas: 1
      resources:
        limits:
          memory: 1g

  # Logstash日志处理
  logstash:
    image: docker.elastic.co/logstash/logstash:7.15.0
    configs:
      - source: logstash_config
        target: /usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"
    networks:
      - logging
    depends_on:
      - elasticsearch
    deploy:
      replicas: 1

  # Kibana可视化
  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    networks:
      - logging
    depends_on:
      - elasticsearch
    deploy:
      replicas: 1

  # Filebeat日志收集
  filebeat:
    image: docker.elastic.co/beats/filebeat:7.15.0
    user: root
    volumes:
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    configs:
      - source: filebeat_config
        target: /usr/share/filebeat/filebeat.yml
    networks:
      - logging
    deploy:
      mode: global

configs:
  logstash_config:
    external: true
  filebeat_config:
    external: true

networks:
  logging:
    driver: overlay

volumes:
  elasticsearch_data:
```

### 4. CI/CD流水线

#### 4.1 GitHub Actions配置
```yaml
# .github/workflows/deploy.yml
name: Production Deploy

on:
  push:
    branches: [main]
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v3

      - name: Log in to Container Registry
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
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}

      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  security-scan:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ needs.build.outputs.image-tag }}
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

  deploy:
    needs: [build, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to production
        env:
          DOCKER_HOST: ${{ secrets.DOCKER_HOST }}
          DOCKER_TLS_VERIFY: 1
          DOCKER_CERT_PATH: /tmp/docker-certs
        run: |
          echo "${{ secrets.DOCKER_CA }}" > $DOCKER_CERT_PATH/ca.pem
          echo "${{ secrets.DOCKER_CERT }}" > $DOCKER_CERT_PATH/cert.pem
          echo "${{ secrets.DOCKER_KEY }}" > $DOCKER_CERT_PATH/key.pem
          
          export APP_VERSION=${{ needs.build.outputs.image-tag }}
          docker stack deploy -c production-stack.yml myapp
```

#### 4.2 蓝绿部署脚本
```bash
#!/bin/bash
# blue-green-deploy.sh

APP_NAME="myapp"
NEW_VERSION="$1"
CURRENT_SERVICE="${APP_NAME}_app"
BLUE_SERVICE="${APP_NAME}_app_blue"
GREEN_SERVICE="${APP_NAME}_app_green"

# 检查当前运行的服务
CURRENT_COLOR=$(docker service inspect --format '{{.Spec.Labels.color}}' $CURRENT_SERVICE 2>/dev/null || echo "blue")

if [ "$CURRENT_COLOR" = "blue" ]; then
    NEW_COLOR="green"
    NEW_SERVICE=$GREEN_SERVICE
    OLD_SERVICE=$BLUE_SERVICE
else
    NEW_COLOR="blue"
    NEW_SERVICE=$BLUE_SERVICE
    OLD_SERVICE=$GREEN_SERVICE
fi

echo "Deploying version $NEW_VERSION as $NEW_COLOR"

# 创建新版本服务
docker service create \
  --name $NEW_SERVICE \
  --label color=$NEW_COLOR \
  --replicas 3 \
  --network traefik-public \
  --label "traefik.enable=true" \
  --label "traefik.http.routers.${NEW_SERVICE}.rule=Host(\`app.example.com\`)" \
  myregistry/myapp:$NEW_VERSION

# 等待服务就绪
echo "Waiting for $NEW_SERVICE to be ready..."
while [ $(docker service ls --filter name=$NEW_SERVICE --format "{{.Replicas}}" | cut -d'/' -f1) != "3" ]; do
  sleep 5
done

# 健康检查
echo "Performing health check..."
for i in {1..10}; do
  if curl -f http://app.example.com/health; then
    echo "Health check passed"
    break
  fi
  if [ $i -eq 10 ]; then
    echo "Health check failed, rolling back"
    docker service rm $NEW_SERVICE
    exit 1
  fi
  sleep 10
done

# 切换流量
echo "Switching traffic to $NEW_COLOR"
docker service update --label-add traefik.enable=true $NEW_SERVICE
docker service update --label-rm traefik.enable $OLD_SERVICE || true

# 清理旧服务
echo "Cleaning up old service"
sleep 30
docker service rm $OLD_SERVICE || true

echo "Deployment completed successfully"
```

## 实验任务

### 任务1：完整生产环境部署
部署一个包含所有生产要素的电商应用：

```bash
# 1. 创建Swarm集群
docker swarm init

# 2. 创建网络
docker network create --driver overlay traefik-public
docker network create --driver overlay backend

# 3. 创建秘钥
echo "your-db-password" | docker secret create db_password -
echo "your-jwt-secret" | docker secret create jwt_secret -
echo "postgresql://user:pass@database:5432/ecommerce" | docker secret create database_url -

# 4. 部署应用栈
docker stack deploy -c production-stack.yml ecommerce

# 5. 部署监控栈
docker stack deploy -c monitoring-stack.yml monitoring

# 6. 配置告警规则
docker config create alert_rules - << EOF
groups:
- name: containers
  rules:
  - alert: ContainerDown
    expr: absent(container_last_seen)
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "Container {{ \$labels.name }} is down"
EOF
```

### 任务2：性能测试和优化
```bash
# 使用Apache Bench进行压力测试
ab -n 1000 -c 10 http://app.example.com/

# 使用wrk进行更复杂的测试
wrk -t12 -c400 -d30s --script=post.lua http://app.example.com/api/orders

# 监控资源使用
docker stats
docker service ps ecommerce_app
```

### 任务3：故障恢复演练
```bash
# 模拟节点故障
docker node update --availability drain node-2

# 模拟服务故障
docker service update --replicas 0 ecommerce_app
docker service update --replicas 3 ecommerce_app

# 数据库备份和恢复
docker exec -t postgres_container pg_dump -U user database > backup.sql
cat backup.sql | docker exec -i new_postgres_container psql -U user database
```

## 安全最佳实践

### 1. 镜像安全
```bash
# 使用官方基础镜像
FROM node:16-alpine

# 定期扫描漏洞
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image myapp:latest

# 使用多阶段构建
# 不在镜像中存储敏感信息
# 使用非root用户运行
```

### 2. 运行时安全
```bash
# 使用只读根文件系统
docker service create --read-only myapp

# 限制能力
docker service create --cap-drop ALL --cap-add NET_BIND_SERVICE myapp

# 使用安全配置文件
docker service create --security-opt seccomp=secure.json myapp
```

### 3. 网络安全
```bash
# 使用加密网络
docker network create --opt encrypted overlay secure-net

# 限制端口暴露
# 使用防火墙规则
iptables -A INPUT -p tcp --dport 2376 -j DROP
```

## 性能优化

### 1. 镜像优化
- 使用轻量级基础镜像
- 多阶段构建减少镜像大小
- 合并RUN指令减少层数
- 使用.dockerignore排除不必要文件

### 2. 容器优化
```bash
# 设置适当的资源限制
docker service create \
  --limit-cpu 0.5 \
  --limit-memory 512m \
  --reserve-cpu 0.25 \
  --reserve-memory 256m \
  myapp
```

### 3. 存储优化
```bash
# 使用SSD存储
# 配置存储驱动
echo '{"storage-driver": "overlay2"}' > /etc/docker/daemon.json

# 使用数据卷而非绑定挂载
docker volume create --driver local --opt type=tmpfs app-cache
```

## 思考题

1. **如何设计一个零停机的部署策略？**

2. **生产环境中如何保证数据的安全性和完整性？**

3. **如何设计有效的监控告警体系？**

4. **容器化应用的性能瓶颈通常在哪里？如何优化？**

5. **如何处理生产环境中的紧急故障？**

## 参考资料

- [Docker生产部署指南](https://docs.docker.com/config/containers/live-restore/)
- [容器安全最佳实践](https://docs.docker.com/engine/security/)
- [Prometheus监控指南](https://prometheus.io/docs/guides/docker/)
- [Docker性能调优](https://docs.docker.com/config/containers/resource_constraints/)

## 总结

通过完成这10个实验，你已经掌握了：
- Docker的基础概念和操作
- 镜像构建和优化技巧
- 容器编排和集群管理
- 生产环境部署和监控
- 安全性和性能优化

继续深入学习Kubernetes、服务网格等更高级的容器编排技术！ 