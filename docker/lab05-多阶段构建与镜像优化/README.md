# 实验5：多阶段构建与镜像优化

## 学习目标
- 深入理解多阶段构建的原理和应用
- 掌握高级镜像优化技巧
- 学会分析和减少镜像大小
- 理解不同构建策略的适用场景

## 理论学习

### 1. 多阶段构建原理
多阶段构建允许在单个Dockerfile中使用多个FROM指令，每个FROM指令开始一个新的构建阶段。

### 2. 构建阶段类型
- **构建阶段**: 编译、打包应用
- **运行阶段**: 运行最终应用
- **测试阶段**: 运行测试和检查
- **调试阶段**: 包含调试工具

### 3. 镜像优化策略
| 策略 | 效果 | 适用场景 |
|------|------|----------|
| 多阶段构建 | 大幅减少镜像大小 | 编译型语言 |
| Alpine基础镜像 | 减少基础镜像大小 | 通用应用 |
| 指令合并 | 减少层数 | 复杂构建 |
| 缓存优化 | 加速构建 | 频繁构建 |

## 动手实践

### 步骤1: React应用多阶段构建

#### 创建React应用
```bash
mkdir react-multistage
cd react-multistage

# 创建package.json
cat > package.json << 'EOF'
{
  "name": "react-docker-app",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test --watchAll=false",
    "eject": "react-scripts eject"
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
EOF

# 创建public目录和文件
mkdir -p public src

cat > public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>React Docker App</title>
</head>
<body>
  <div id="root"></div>
</body>
</html>
EOF

cat > src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
EOF

cat > src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [buildInfo, setBuildInfo] = useState({});

  useEffect(() => {
    setBuildInfo({
      buildTime: process.env.REACT_APP_BUILD_TIME || '未知',
      version: process.env.REACT_APP_VERSION || '1.0.0',
      nodeEnv: process.env.NODE_ENV || 'development'
    });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>React Docker 应用</h1>
        <p>这是一个使用多阶段构建的React应用</p>
        <div className="build-info">
          <h3>构建信息:</h3>
          <p>版本: {buildInfo.version}</p>
          <p>构建时间: {buildInfo.buildTime}</p>
          <p>环境: {buildInfo.nodeEnv}</p>
        </div>
      </header>
    </div>
  );
}

export default App;
EOF

cat > src/App.css << 'EOF'
.App {
  text-align: center;
}

.App-header {
  background-color: #282c34;
  padding: 20px;
  color: white;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.build-info {
  margin-top: 20px;
  padding: 20px;
  border: 1px solid #444;
  border-radius: 8px;
  background-color: #333;
}

.build-info h3 {
  margin-top: 0;
  color: #61dafb;
}

.build-info p {
  margin: 10px 0;
  font-family: monospace;
}
EOF
```

#### 单阶段vs多阶段构建对比
```bash
# 单阶段构建Dockerfile
cat > Dockerfile.single << 'EOF'
FROM node:18-alpine

WORKDIR /app

# 复制所有文件
COPY . .

# 安装依赖
RUN npm install

# 构建应用
RUN npm run build

# 安装serve来提供静态文件
RUN npm install -g serve

# 暴露端口
EXPOSE 3000

# 启动应用
CMD ["serve", "-s", "build", "-l", "3000"]
EOF

# 多阶段构建Dockerfile
cat > Dockerfile << 'EOF'
# 构建阶段
FROM node:18-alpine AS builder

WORKDIR /app

# 复制package文件
COPY package*.json ./

# 安装依赖
RUN npm ci --only=production --silent

# 复制源代码
COPY . .

# 设置构建时环境变量
ARG REACT_APP_BUILD_TIME
ARG REACT_APP_VERSION
ENV REACT_APP_BUILD_TIME=$REACT_APP_BUILD_TIME
ENV REACT_APP_VERSION=$REACT_APP_VERSION

# 构建应用
RUN npm run build

# 生产阶段
FROM nginx:alpine AS production

# 复制自定义nginx配置
COPY --from=builder /app/build /usr/share/nginx/html

# 创建自定义nginx配置
RUN echo 'server { \
    listen 80; \
    location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files $uri $uri/ /index.html; \
    } \
    location /health { \
        access_log off; \
        return 200 "healthy\n"; \
        add_header Content-Type text/plain; \
    } \
}' > /etc/nginx/conf.d/default.conf

# 暴露端口
EXPOSE 80

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/health || exit 1

# 启动nginx
CMD ["nginx", "-g", "daemon off;"]
EOF

# 构建并比较镜像大小
echo "构建单阶段镜像..."
docker build -f Dockerfile.single -t react-app:single .

echo "构建多阶段镜像..."
docker build \
  --build-arg REACT_APP_BUILD_TIME="$(date)" \
  --build-arg REACT_APP_VERSION="1.0.0" \
  -t react-app:multistage .

echo "镜像大小对比:"
docker images react-app
```

### 步骤2: Java应用多阶段构建

#### 创建Java Spring Boot应用
```bash
mkdir java-multistage
cd java-multistage

# 创建Maven项目结构
mkdir -p src/main/java/com/example/demo
mkdir -p src/main/resources

# 创建pom.xml
cat > pom.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.example</groupId>
    <artifactId>demo</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>
    
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.1.0</version>
        <relativePath/>
    </parent>
    
    <properties>
        <java.version>17</java.version>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-actuator</artifactId>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>
EOF

# 创建主应用类
cat > src/main/java/com/example/demo/DemoApplication.java << 'EOF'
package com.example.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

@SpringBootApplication
@RestController
public class DemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }

    @GetMapping("/")
    public Map<String, Object> hello() {
        Map<String, Object> response = new HashMap<>();
        response.put("message", "Hello from Spring Boot in Docker!");
        response.put("timestamp", LocalDateTime.now());
        response.put("version", "1.0.0");
        return response;
    }

    @GetMapping("/info")
    public Map<String, Object> info() {
        Map<String, Object> info = new HashMap<>();
        info.put("application", "Spring Boot Docker Demo");
        info.put("java.version", System.getProperty("java.version"));
        info.put("os.name", System.getProperty("os.name"));
        info.put("available.processors", Runtime.getRuntime().availableProcessors());
        info.put("max.memory", Runtime.getRuntime().maxMemory() / 1024 / 1024 + " MB");
        return info;
    }
}
EOF

# 创建application.properties
cat > src/main/resources/application.properties << 'EOF'
server.port=8080
management.endpoints.web.exposure.include=health,info
management.endpoint.health.show-details=always
EOF
```

#### 优化的Java多阶段Dockerfile
```bash
cat > Dockerfile << 'EOF'
# 构建阶段
FROM maven:3.9-eclipse-temurin-17 AS builder

WORKDIR /app

# 先复制pom.xml，利用Docker缓存
COPY pom.xml .

# 下载依赖（这一层会被缓存）
RUN mvn dependency:go-offline -B

# 复制源代码
COPY src ./src

# 构建应用
RUN mvn clean package -DskipTests -B

# 运行阶段
FROM eclipse-temurin:17-jre-alpine AS runtime

# 安装必要工具
RUN apk add --no-cache curl

# 创建非root用户
RUN addgroup -g 1001 appgroup && \
    adduser -u 1001 -G appgroup -s /bin/sh -D appuser

# 设置工作目录
WORKDIR /app

# 从构建阶段复制jar文件
COPY --from=builder /app/target/*.jar app.jar

# 改变文件所有者
RUN chown appuser:appgroup app.jar

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/actuator/health || exit 1

# JVM优化参数
ENV JAVA_OPTS="-Xms128m -Xmx256m -XX:+UseG1GC -XX:+PrintGCDetails"

# 启动应用
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
EOF

# 构建并测试
docker build -t java-app:optimized .
docker run -d -p 8080:8080 --name java-app java-app:optimized

# 测试API
sleep 30  # 等待应用启动
curl http://localhost:8080/
curl http://localhost:8080/info
curl http://localhost:8080/actuator/health
```

### 步骤3: 构建优化策略

#### 构建缓存优化示例
```bash
mkdir cache-optimization
cd cache-optimization

cat > package.json << 'EOF'
{
  "name": "cache-demo",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.18.2",
    "lodash": "^4.17.21"
  }
}
EOF

cat > app.js << 'EOF'
const express = require('express');
const _ = require('lodash');

const app = express();
const port = 3000;

app.get('/', (req, res) => {
  const data = _.range(1, 11).map(i => ({ id: i, value: `Item ${i}` }));
  res.json({ message: 'Cache optimization demo', data });
});

app.listen(port, '0.0.0.0', () => {
  console.log(`Server running on port ${port}`);
});
EOF

# 优化前的Dockerfile
cat > Dockerfile.bad << 'EOF'
FROM node:18-alpine

WORKDIR /app

# 一次性复制所有文件
COPY . .

# 安装依赖
RUN npm install

# 启动应用
CMD ["node", "app.js"]
EOF

# 优化后的Dockerfile
cat > Dockerfile.good << 'EOF'
FROM node:18-alpine

WORKDIR /app

# 先复制package.json，利用缓存
COPY package*.json ./

# 安装依赖（这一层会被缓存）
RUN npm ci --only=production && npm cache clean --force

# 再复制应用代码
COPY app.js .

# 启动应用
CMD ["node", "app.js"]
EOF

# 测试构建缓存效果
echo "第一次构建（坏的示例）:"
time docker build -f Dockerfile.bad -t cache-demo:bad .

echo "修改应用代码..."
echo "// Updated" >> app.js

echo "第二次构建（坏的示例）:"
time docker build -f Dockerfile.bad -t cache-demo:bad .

# 恢复文件
git checkout app.js 2>/dev/null || sed -i '' '$ d' app.js

echo "第一次构建（好的示例）:"
time docker build -f Dockerfile.good -t cache-demo:good .

echo "修改应用代码..."
echo "// Updated" >> app.js

echo "第二次构建（好的示例）:"
time docker build -f Dockerfile.good -t cache-demo:good .
```

#### 镜像大小分析工具
```bash
# 使用docker history分析镜像层
analyze_image() {
    local image=$1
    echo "=== 分析镜像: $image ==="
    docker history $image --format "table {{.CreatedBy}}\t{{.Size}}"
    echo ""
    echo "总大小:"
    docker images $image --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    echo ""
}

# 分析之前构建的镜像
analyze_image "react-app:single"
analyze_image "react-app:multistage"
analyze_image "java-app:optimized"
```

### 步骤4: 高级优化技巧

#### 使用BuildKit提升构建性能
```bash
# 启用BuildKit
export DOCKER_BUILDKIT=1

# 创建高级Dockerfile
cat > Dockerfile.buildkit << 'EOF'
# syntax=docker/dockerfile:1.4

FROM node:18-alpine AS base

WORKDIR /app

# 使用缓存挂载优化npm安装
FROM base AS deps
COPY package*.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --only=production

FROM base AS build
COPY package*.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci
COPY . .
RUN npm run build

FROM nginx:alpine AS runtime
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
EOF

# 构建时使用缓存挂载
docker build -f Dockerfile.buildkit -t app:buildkit .
```

#### 镜像压缩和清理
```bash
# 创建压缩优化的Dockerfile
cat > Dockerfile.compressed << 'EOF'
FROM alpine:latest

# 在单个RUN指令中执行多个操作
RUN apk add --no-cache \
    python3 \
    py3-pip \
    && pip3 install --no-cache-dir flask gunicorn \
    && adduser -D -s /bin/sh appuser \
    && rm -rf /var/cache/apk/* \
    && rm -rf /tmp/*

WORKDIR /app

COPY app.py .

USER appuser

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
EOF

# 创建简单的Flask应用
cat > app.py << 'EOF'
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({'message': 'Compressed image demo'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
EOF

docker build -f Dockerfile.compressed -t app:compressed .
```

## 实验任务

### 任务1: 全栈应用多阶段构建
```bash
# 创建一个包含前端和后端的全栈应用
mkdir fullstack-app
cd fullstack-app

# 创建前端部分（在frontend目录）
mkdir frontend
cd frontend

cat > package.json << 'EOF'
{
  "name": "frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "axios": "^1.4.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "proxy": "http://backend:5000"
}
EOF

cat > src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get('/api/data')
      .then(response => setData(response.data))
      .catch(error => console.error('Error:', error));
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h1>全栈Docker应用</h1>
      {data ? (
        <div>
          <h2>来自后端的数据:</h2>
          <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
      ) : (
        <p>加载中...</p>
      )}
    </div>
  );
}

export default App;
EOF

cd ..

# 创建后端部分（在backend目录）
mkdir backend
cd backend

cat > requirements.txt << 'EOF'
Flask==2.3.3
Flask-CORS==4.0.0
gunicorn==21.2.0
EOF

cat > app.py << 'EOF'
from flask import Flask, jsonify
from flask_cors import CORS
import datetime

app = Flask(__name__)
CORS(app)

@app.route('/api/data')
def get_data():
    return jsonify({
        'message': '这是来自后端的数据',
        'timestamp': datetime.datetime.now().isoformat(),
        'server': '多阶段构建Docker容器'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
EOF

cd ..

# 创建多阶段Dockerfile
cat > Dockerfile << 'EOF'
# 前端构建阶段
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# 后端构建阶段
FROM python:3.9-slim AS backend-builder
WORKDIR /app/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 最终运行阶段
FROM python:3.9-slim AS runtime

# 安装nginx和Python依赖
RUN apt-get update && \
    apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

# 复制后端代码和依赖
WORKDIR /app
COPY --from=backend-builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY backend/ ./backend/

# 复制前端构建结果到nginx目录
COPY --from=frontend-builder /app/frontend/build /var/www/html

# 配置nginx
RUN echo 'server { \
    listen 80; \
    root /var/www/html; \
    index index.html; \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    location /api/ { \
        proxy_pass http://localhost:5000; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
    } \
}' > /etc/nginx/sites-available/default

# 启动脚本
RUN echo '#!/bin/bash\n\
nginx &\n\
cd /app/backend && python app.py' > /start.sh && \
    chmod +x /start.sh

EXPOSE 80

CMD ["/start.sh"]
EOF

# 构建全栈应用
docker build -t fullstack-app .
```

### 任务2: 微服务构建优化
```bash
# 创建多个微服务的构建配置
mkdir microservices
cd microservices

# 创建用户服务
mkdir user-service
cd user-service

cat > Dockerfile << 'EOF'
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o user-service .

FROM alpine:latest AS runtime
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/user-service .
EXPOSE 8081
CMD ["./user-service"]
EOF

cd ..

# 创建订单服务
mkdir order-service
cd order-service

cat > Dockerfile << 'EOF'
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .

FROM node:18-alpine AS runtime
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodeuser -u 1001
WORKDIR /app
COPY --from=builder /app .
USER nodeuser
EXPOSE 8082
CMD ["node", "server.js"]
EOF

cd ..

# 创建构建所有服务的脚本
cat > build-all.sh << 'EOF'
#!/bin/bash

echo "构建用户服务..."
docker build -t user-service:latest ./user-service

echo "构建订单服务..."
docker build -t order-service:latest ./order-service

echo "构建完成！"
docker images | grep -E "(user-service|order-service)"
EOF

chmod +x build-all.sh
```

## 进阶练习

### 练习1: 构建时优化分析
```bash
# 使用BuildKit分析构建过程
export DOCKER_BUILDKIT=1

# 构建时显示详细信息
docker build --progress=plain -t app:analysis .

# 使用docker-slim优化镜像（如果安装了）
# docker-slim build --target app:analysis --tag app:slim
```

### 练习2: 安全优化
```bash
cat > Dockerfile.secure << 'EOF'
FROM node:18-alpine AS builder

# 使用非root用户构建
RUN addgroup -g 1001 -S builder && \
    adduser -S builder -u 1001 -G builder

USER builder
WORKDIR /app

COPY --chown=builder:builder package*.json ./
RUN npm ci --only=production

COPY --chown=builder:builder . .
RUN npm run build

FROM nginx:alpine AS runtime

# 创建nginx用户
RUN addgroup -g 1001 -S nginx && \
    adduser -S nginx -u 1001 -G nginx

# 复制文件并设置权限
COPY --from=builder --chown=nginx:nginx /app/build /usr/share/nginx/html

# 移除不必要的包
RUN apk del --purge

USER nginx

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
EOF
```

## 性能测试和比较

### 镜像大小比较脚本
```bash
cat > compare-images.sh << 'EOF'
#!/bin/bash

echo "=== 镜像大小比较 ==="
echo "镜像名称                    大小"
echo "--------------------------------"

for image in $(docker images --format "{{.Repository}}:{{.Tag}}" | grep -v "<none>"); do
    size=$(docker images $image --format "{{.Size}}")
    printf "%-25s %s\n" "$image" "$size"
done

echo ""
echo "=== 构建历史分析 ==="
for image in react-app:single react-app:multistage; do
    if docker images $image &>/dev/null; then
        echo "--- $image ---"
        docker history $image --format "table {{.Size}}\t{{.CreatedBy}}" | head -10
        echo ""
    fi
done
EOF

chmod +x compare-images.sh
./compare-images.sh
```

## 思考题

1. 多阶段构建在什么情况下最有效？
2. 如何平衡镜像大小和构建时间？
3. 构建缓存的最佳实践是什么？
4. 如何选择最适合的基础镜像？
5. 在微服务架构中如何优化镜像构建？

## 最佳实践总结

### 1. 多阶段构建
- 将构建工具和运行时环境分离
- 只在最终镜像中包含必要文件
- 使用命名阶段提高可读性

### 2. 构建优化
- 合理安排Dockerfile指令顺序
- 利用构建缓存
- 使用.dockerignore排除文件

### 3. 镜像大小
- 选择合适的基础镜像
- 清理包管理器缓存
- 合并RUN指令

### 4. 安全性
- 使用非root用户
- 定期更新基础镜像
- 移除不必要的工具

## 常用命令速查

```bash
# 多阶段构建
docker build --target <阶段名> -t <镜像名> .    # 构建特定阶段

# 构建分析
docker history <镜像名>                         # 查看镜像历史
docker build --progress=plain -t <镜像名> .     # 显示详细构建过程

# 镜像优化
docker images                                   # 查看镜像大小
docker system df                                # 查看磁盘使用

# BuildKit特性
export DOCKER_BUILDKIT=1                        # 启用BuildKit
docker build --build-arg BUILDKIT_INLINE_CACHE=1  # 内联缓存
```

## 下一步
完成本实验后，进入 `lab06-数据卷与持久化存储` 学习Docker数据管理。

## 参考资料
- [多阶段构建官方文档](https://docs.docker.com/develop/dev-best-practices/#use-multi-stage-builds)
- [BuildKit文档](https://docs.docker.com/buildx/working-with-buildx/)
- [镜像优化最佳实践](https://docs.docker.com/develop/dev-best-practices/) 