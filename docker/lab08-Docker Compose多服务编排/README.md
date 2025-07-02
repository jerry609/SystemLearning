# 实验8：Docker Compose多服务编排

## 学习目标
- 深入理解Docker Compose的作用和原理
- 掌握docker-compose.yml文件的编写
- 学会多服务应用的编排和管理
- 理解服务发现和负载均衡
- 掌握开发环境和生产环境的部署策略

## 理论学习

### Docker Compose简介
Docker Compose是Docker官方的多容器应用编排工具，使用YAML文件定义和运行多容器Docker应用程序。

### 核心概念

| 概念 | 说明 | 作用 |
|------|------|------|
| Project | 项目，包含多个服务 | 组织相关服务的集合 |
| Service | 服务，定义容器的运行方式 | 应用的组件（如web、db） |
| Container | 容器，服务的运行实例 | 服务的具体执行单元 |

### Compose文件结构
```yaml
version: '3.8'
services:
  web:
    # 服务配置
  db:
    # 服务配置
volumes:
  # 数据卷定义
networks:
  # 网络定义
```

### 版本兼容性

| Compose版本 | Docker版本 | 主要特性 |
|-------------|------------|----------|
| 3.8 | 19.03.0+ | 支持GPU、外部网络等 |
| 3.7 | 18.06.0+ | 支持配置和秘钥 |
| 3.6 | 18.02.0+ | 支持扩展字段 |
| 3.5 | 17.12.0+ | 支持外部网络 |

## 动手实践

### 1. 基础入门

#### 1.1 第一个Compose应用
创建一个简单的Web+Redis应用：

```yaml
# docker-compose.yml
version: '3.8'
services:
  web:
    image: nginx:latest
    ports:
      - "8080:80"
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

```bash
# 启动应用
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs

# 停止应用
docker-compose down
```

#### 1.2 使用环境变量
```yaml
# docker-compose.yml
version: '3.8'
services:
  web:
    image: nginx:${NGINX_VERSION:-latest}
    ports:
      - "${WEB_PORT:-80}:80"
    environment:
      - ENV=${ENVIRONMENT:-development}
```

```bash
# .env文件
NGINX_VERSION=1.20
WEB_PORT=8080
ENVIRONMENT=production
```

### 2. 构建自定义镜像

#### 2.1 基于Dockerfile构建
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        - NODE_ENV=production
    ports:
      - "3000:3000"
```

```dockerfile
# Dockerfile
FROM node:16-alpine
ARG NODE_ENV
ENV NODE_ENV=${NODE_ENV}

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .

EXPOSE 3000
CMD ["npm", "start"]
```

#### 2.2 多阶段构建
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build:
      context: .
      target: production
    ports:
      - "3000:3000"
```

```dockerfile
# Dockerfile
FROM node:16-alpine AS development
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
CMD ["npm", "run", "dev"]

FROM node:16-alpine AS production
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
CMD ["npm", "start"]
```

### 3. 数据持久化

#### 3.1 命名数据卷
```yaml
# docker-compose.yml
version: '3.8'
services:
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

#### 3.2 绑定挂载
```yaml
# docker-compose.yml
version: '3.8'
services:
  web:
    image: nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./html:/usr/share/nginx/html:ro
    ports:
      - "80:80"
```

#### 3.3 临时文件系统
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    image: myapp
    tmpfs:
      - /tmp
      - /var/cache
```

### 4. 网络配置

#### 4.1 自定义网络
```yaml
# docker-compose.yml
version: '3.8'
services:
  frontend:
    image: nginx
    networks:
      - frontend-net
    ports:
      - "80:80"
  
  backend:
    image: node:alpine
    networks:
      - frontend-net
      - backend-net
  
  database:
    image: postgres
    networks:
      - backend-net

networks:
  frontend-net:
    driver: bridge
  backend-net:
    driver: bridge
```

#### 4.2 外部网络
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    image: myapp
    networks:
      - external-network

networks:
  external-network:
    external: true
```

### 5. 完整的LAMP应用

创建一个完整的LAMP（Linux + Apache + MySQL + PHP）应用：

```yaml
# docker-compose.yml
version: '3.8'
services:
  # Apache + PHP服务
  web:
    build:
      context: .
      dockerfile: php.Dockerfile
    ports:
      - "80:80"
    volumes:
      - ./src:/var/www/html
    depends_on:
      - db
    environment:
      - MYSQL_HOST=db
      - MYSQL_USER=root
      - MYSQL_PASSWORD=rootpassword
    networks:
      - lamp-network

  # MySQL数据库
  db:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: rootpassword
      MYSQL_DATABASE: lampdb
      MYSQL_USER: lampuser
      MYSQL_PASSWORD: lamppass
    volumes:
      - mysql_data:/var/lib/mysql
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "3306:3306"
    networks:
      - lamp-network

  # phpMyAdmin管理工具
  phpmyadmin:
    image: phpmyadmin/phpmyadmin
    environment:
      PMA_HOST: db
      PMA_PORT: 3306
      PMA_USER: root
      PMA_PASSWORD: rootpassword
    ports:
      - "8080:80"
    depends_on:
      - db
    networks:
      - lamp-network

volumes:
  mysql_data:

networks:
  lamp-network:
    driver: bridge
```

```dockerfile
# php.Dockerfile
FROM php:8.0-apache

# 安装MySQL扩展
RUN docker-php-ext-install mysqli pdo pdo_mysql

# 启用Apache模块
RUN a2enmod rewrite

# 设置工作目录
WORKDIR /var/www/html

# 复制配置文件
COPY apache-config.conf /etc/apache2/sites-available/000-default.conf
```

```php
<?php
// src/index.php
$host = getenv('MYSQL_HOST') ?: 'localhost';
$user = getenv('MYSQL_USER') ?: 'root';
$pass = getenv('MYSQL_PASSWORD') ?: '';
$db = 'lampdb';

try {
    $pdo = new PDO("mysql:host=$host;dbname=$db", $user, $pass);
    echo "<h1>LAMP Stack 运行成功！</h1>";
    echo "<p>数据库连接正常</p>";
    
    // 显示当前时间
    $stmt = $pdo->query('SELECT NOW() as current_time');
    $result = $stmt->fetch();
    echo "<p>当前时间: " . $result['current_time'] . "</p>";
    
} catch(PDOException $e) {
    echo "连接失败: " . $e->getMessage();
}
?>
```

```sql
-- init.sql
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO users (name, email) VALUES 
('张三', 'zhangsan@example.com'),
('李四', 'lisi@example.com');
```

## 实验任务

### 任务1：微服务电商应用
创建一个包含前端、API、数据库和缓存的电商应用：

```yaml
# docker-compose.yml
version: '3.8'
services:
  # 前端服务
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - api
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    networks:
      - frontend-net

  # API服务
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/ecommerce
      - REDIS_URL=redis://redis:6379
    networks:
      - frontend-net
      - backend-net
    volumes:
      - ./api:/app
    command: python manage.py runserver 0.0.0.0:8000

  # 数据库服务
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: ecommerce
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - backend-net

  # 缓存服务
  redis:
    image: redis:alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - backend-net

  # 消息队列
  rabbitmq:
    image: rabbitmq:3-management
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: admin
    ports:
      - "15672:15672"  # 管理界面
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - backend-net

  # 监控服务
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - monitoring-net

volumes:
  postgres_data:
  redis_data:
  rabbitmq_data:

networks:
  frontend-net:
  backend-net:
  monitoring-net:
```

### 任务2：开发环境配置
创建支持热重载的开发环境：

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  web:
    build:
      context: .
      target: development
    volumes:
      - .:/app
      - /app/node_modules
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - CHOKIDAR_USEPOLLING=true
    command: npm run dev
```

### 任务3：生产环境配置
创建生产环境的配置：

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  web:
    build:
      context: .
      target: production
    restart: unless-stopped
    ports:
      - "80:80"
    environment:
      - NODE_ENV=production
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.web.rule=Host(`example.com`)"
    
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - static_files:/var/www/static
    ports:
      - "443:443"
    restart: unless-stopped

volumes:
  static_files:
```

## 进阶练习

### 1. 服务扩展和负载均衡
```bash
# 扩展Web服务到3个实例
docker-compose up -d --scale web=3

# 使用nginx做负载均衡
```

```yaml
# nginx负载均衡配置
version: '3.8'
services:
  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web
  
  web:
    build: .
    expose:
      - "8000"
```

### 2. 健康检查
```yaml
# docker-compose.yml
version: '3.8'
services:
  web:
    image: nginx
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
  
  db:
    image: postgres
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### 3. 配置和秘钥管理
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    image: myapp
    configs:
      - app_config
    secrets:
      - db_password
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password

configs:
  app_config:
    file: ./config/app.conf

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

### 4. 多环境管理
```bash
# 开发环境
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# 测试环境
docker-compose -f docker-compose.yml -f docker-compose.test.yml up

# 生产环境
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### 5. CI/CD集成
```yaml
# .github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Deploy to production
        run: |
          docker-compose -f docker-compose.prod.yml pull
          docker-compose -f docker-compose.prod.yml up -d
```

## 故障排查指南

### 常见问题

1. **服务启动失败**
   ```bash
   # 查看详细日志
   docker-compose logs service_name
   
   # 查看服务状态
   docker-compose ps
   
   # 进入容器调试
   docker-compose exec service_name sh
   ```

2. **网络连接问题**
   ```bash
   # 查看网络
   docker-compose exec service_name nslookup other_service
   
   # 测试端口连通性
   docker-compose exec service_name telnet other_service 3306
   ```

3. **数据卷问题**
   ```bash
   # 查看数据卷
   docker volume ls
   docker volume inspect volume_name
   
   # 清理未使用的数据卷
   docker-compose down -v
   ```

### 调试技巧

1. **使用override文件**
   ```yaml
   # docker-compose.override.yml
   version: '3.8'
   services:
     web:
       command: tail -f /dev/null  # 保持容器运行
       volumes:
         - .:/app  # 挂载源代码
   ```

2. **环境变量调试**
   ```bash
   # 查看环境变量
   docker-compose config
   
   # 验证环境变量
   docker-compose exec service_name env
   ```

## 最佳实践

### 1. 文件组织
```
project/
├── docker-compose.yml
├── docker-compose.dev.yml
├── docker-compose.prod.yml
├── .env
├── .env.example
├── services/
│   ├── web/
│   │   └── Dockerfile
│   └── api/
│       └── Dockerfile
└── config/
    ├── nginx/
    └── mysql/
```

### 2. 安全考虑
- 使用非root用户运行容器
- 不在镜像中存储敏感信息
- 使用secrets管理密码
- 限制容器资源使用

### 3. 性能优化
- 使用.dockerignore文件
- 多阶段构建减少镜像大小
- 合理设置资源限制
- 使用构建缓存

### 4. 版本管理
- 固定镜像版本标签
- 使用语义化版本
- 定期更新依赖

## 思考题

1. **为什么要使用Docker Compose而不是直接使用docker run命令？**

2. **如何在Compose中实现服务的依赖管理？depends_on有什么限制？**

3. **生产环境中使用Compose有什么注意事项？**

4. **如何实现Compose应用的零停机部署？**

5. **Compose与Kubernetes有什么区别？各自的适用场景是什么？**

## 常用命令速查

### 基本操作
```bash
# 启动所有服务
docker-compose up -d

# 停止所有服务
docker-compose down

# 重启服务
docker-compose restart service_name

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f service_name
```

### 构建和更新
```bash
# 构建服务
docker-compose build

# 强制重新构建
docker-compose build --no-cache

# 拉取最新镜像
docker-compose pull

# 更新服务
docker-compose up -d --force-recreate
```

### 调试和维护
```bash
# 进入容器
docker-compose exec service_name bash

# 运行一次性命令
docker-compose run --rm service_name command

# 查看配置
docker-compose config

# 验证配置文件
docker-compose config --quiet
```

### 扩展和清理
```bash
# 扩展服务
docker-compose up -d --scale service_name=3

# 删除所有容器和网络
docker-compose down

# 删除包括数据卷
docker-compose down -v

# 删除包括镜像
docker-compose down --rmi all
```

## 参考资料

- [Docker Compose官方文档](https://docs.docker.com/compose/)
- [Compose文件参考](https://docs.docker.com/compose/compose-file/)
- [Docker Compose最佳实践](https://docs.docker.com/develop/dev-best-practices/)
- [生产环境部署指南](https://docs.docker.com/compose/production/)

## 下一步
完成本实验后，继续学习：
- **实验9**: Docker Swarm集群管理
- 深入了解容器集群和服务发现
- 学习生产级容器编排 