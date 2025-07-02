# 实验4：Dockerfile详解与最佳实践

## 学习目标
- 掌握Dockerfile的语法和指令
- 学会构建自定义Docker镜像
- 理解镜像分层和构建缓存机制
- 掌握Dockerfile最佳实践和优化技巧

## 理论学习

### 1. Dockerfile基础
Dockerfile是一个文本文件，包含一系列指令用于自动化构建Docker镜像。

### 2. Dockerfile指令详解
| 指令 | 作用 | 示例 |
|------|------|------|
| FROM | 指定基础镜像 | `FROM ubuntu:20.04` |
| RUN | 执行命令 | `RUN apt-get update` |
| COPY | 复制文件 | `COPY app.py /app/` |
| ADD | 复制文件（支持URL和压缩包） | `ADD file.tar.gz /app/` |
| WORKDIR | 设置工作目录 | `WORKDIR /app` |
| ENV | 设置环境变量 | `ENV NODE_ENV=production` |
| EXPOSE | 声明端口 | `EXPOSE 80` |
| CMD | 默认执行命令 | `CMD ["python", "app.py"]` |
| ENTRYPOINT | 入口点 | `ENTRYPOINT ["python"]` |

### 3. 构建过程
```
Dockerfile → docker build → Docker Image → docker run → Container
```

## 动手实践

### 步骤1: 第一个Dockerfile

#### 创建简单的Web应用
```bash
# 创建项目目录
mkdir -p simple-web-app
cd simple-web-app

# 创建一个简单的HTML文件
cat > index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>My Docker App</title>
</head>
<body>
    <h1>Hello Docker!</h1>
    <p>这是我的第一个Docker应用</p>
    <p>当前时间: <span id="time"></span></p>
    <script>
        document.getElementById('time').innerHTML = new Date().toString();
    </script>
</body>
</html>
EOF
```

#### 编写Dockerfile
```bash
cat > Dockerfile << 'EOF'
# 使用官方nginx镜像作为基础镜像
FROM nginx:alpine

# 维护者信息
LABEL maintainer="your-email@example.com"
LABEL description="Simple web application"

# 将HTML文件复制到nginx默认目录
COPY index.html /usr/share/nginx/html/

# 暴露80端口
EXPOSE 80

# nginx默认会启动，所以不需要CMD
EOF
```

#### 构建镜像
```bash
# 构建镜像
docker build -t simple-web-app .

# 查看构建过程和镜像
docker images simple-web-app

# 运行容器测试
docker run -d -p 8080:80 --name my-web-app simple-web-app

# 测试访问
curl http://localhost:8080
```

### 步骤2: Python应用Dockerfile

#### 创建Python Flask应用
```bash
# 创建新的项目目录
mkdir -p flask-app
cd flask-app

# 创建Flask应用
cat > app.py << 'EOF'
from flask import Flask, jsonify
import os
import socket

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({
        'message': 'Hello from Docker!',
        'hostname': socket.gethostname(),
        'environment': os.environ.get('ENV', 'development')
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
EOF

# 创建requirements.txt
cat > requirements.txt << 'EOF'
Flask==2.3.3
gunicorn==21.2.0
EOF
```

#### 编写Python应用的Dockerfile
```bash
cat > Dockerfile << 'EOF'
# 使用官方Python运行时作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV ENV production

# 安装系统依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装Python包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .

# 创建非root用户
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# 运行应用
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
EOF
```

#### 构建和测试Python应用
```bash
# 构建镜像
docker build -t flask-app .

# 运行容器
docker run -d -p 5000:5000 --name my-flask-app flask-app

# 测试应用
curl http://localhost:5000
curl http://localhost:5000/health

# 查看健康状态
docker ps
```

### 步骤3: 多阶段构建示例

#### 创建Go应用
```bash
mkdir -p go-app
cd go-app

cat > main.go << 'EOF'
package main

import (
    "fmt"
    "log"
    "net/http"
    "os"
)

func main() {
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Hello from Go! Hostname: %s", os.Getenv("HOSTNAME"))
    })
    
    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }
    
    fmt.Printf("Server starting on port %s\n", port)
    log.Fatal(http.ListenAndServe(":"+port, nil))
}
EOF

cat > go.mod << 'EOF'
module go-app

go 1.21
EOF
```

#### 多阶段构建Dockerfile
```bash
cat > Dockerfile << 'EOF'
# 构建阶段
FROM golang:1.21-alpine AS builder

# 安装构建依赖
RUN apk add --no-cache git

# 设置工作目录
WORKDIR /app

# 复制go mod文件
COPY go.mod go.sum ./

# 下载依赖
RUN go mod download

# 复制源代码
COPY . .

# 构建应用
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

# 运行阶段
FROM alpine:latest

# 安装ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# 从构建阶段复制可执行文件
COPY --from=builder /app/main .

# 暴露端口
EXPOSE 8080

# 运行应用
CMD ["./main"]
EOF
```

#### 比较镜像大小
```bash
# 构建多阶段镜像
docker build -t go-app:multistage .

# 构建单阶段镜像对比
cat > Dockerfile.single << 'EOF'
FROM golang:1.21-alpine

WORKDIR /app
COPY . .
RUN go build -o main .

EXPOSE 8080
CMD ["./main"]
EOF

docker build -f Dockerfile.single -t go-app:single .

# 比较镜像大小
docker images go-app
```

### 步骤4: 构建优化技巧

#### .dockerignore文件
```bash
cat > .dockerignore << 'EOF'
# Git
.git
.gitignore

# Documentation
README.md
*.md

# Node modules
node_modules

# Logs
*.log

# OS files
.DS_Store
Thumbs.db

# IDE files
.vscode
.idea
*.swp
*.swo

# Build artifacts
dist/
build/
target/

# Test coverage
coverage/
*.cover
EOF
```

#### 优化的Dockerfile示例
```bash
cat > Dockerfile.optimized << 'EOF'
# 使用特定版本而不是latest
FROM node:18-alpine

# 设置非root用户
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# 设置工作目录
WORKDIR /app

# 先复制依赖文件，利用构建缓存
COPY package*.json ./

# 安装依赖
RUN npm ci --only=production && npm cache clean --force

# 复制应用代码
COPY --chown=nextjs:nodejs . .

# 切换到非root用户
USER nextjs

# 暴露端口
EXPOSE 3000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# 启动应用
CMD ["npm", "start"]
EOF
```

## 实验任务

### 任务1: 构建个人博客应用
```bash
# 1. 创建一个简单的静态博客
mkdir personal-blog
cd personal-blog

# 2. 创建多个HTML页面
cat > index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>我的博客</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header><h1>我的个人博客</h1></header>
    <nav>
        <a href="/">首页</a>
        <a href="/about.html">关于</a>
    </nav>
    <main>
        <article>
            <h2>欢迎来到我的博客</h2>
            <p>这是一个使用Docker部署的静态博客。</p>
        </article>
    </main>
</body>
</html>
EOF

cat > about.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>关于 - 我的博客</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header><h1>关于我</h1></header>
    <nav>
        <a href="/">首页</a>
        <a href="/about.html">关于</a>
    </nav>
    <main>
        <p>我是一个Docker学习者。</p>
    </main>
</body>
</html>
EOF

cat > style.css << 'EOF'
body { font-family: Arial, sans-serif; margin: 40px; }
header { background: #333; color: white; padding: 20px; }
nav { background: #f0f0f0; padding: 10px; }
nav a { margin-right: 10px; text-decoration: none; }
main { margin-top: 20px; }
EOF

# 3. 编写Dockerfile
cat > Dockerfile << 'EOF'
FROM nginx:alpine
COPY . /usr/share/nginx/html/
EXPOSE 80
EOF

# 4. 构建并运行
docker build -t personal-blog .
docker run -d -p 8081:80 --name blog personal-blog
```

### 任务2: 数据库应用Dockerfile
```bash
# 创建一个带初始化脚本的数据库
mkdir custom-postgres
cd custom-postgres

cat > init.sql << 'EOF'
CREATE DATABASE myapp;
\c myapp;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO users (username, email) VALUES 
('alice', 'alice@example.com'),
('bob', 'bob@example.com');
EOF

cat > Dockerfile << 'EOF'
FROM postgres:15-alpine

# 设置环境变量
ENV POSTGRES_DB=myapp
ENV POSTGRES_USER=appuser
ENV POSTGRES_PASSWORD=secret123

# 复制初始化脚本
COPY init.sql /docker-entrypoint-initdb.d/

# 暴露端口
EXPOSE 5432
EOF

# 构建和运行
docker build -t custom-postgres .
docker run -d --name my-postgres custom-postgres
```

### 任务3: 微服务应用构建
```bash
# 创建一个API服务
mkdir api-service
cd api-service

cat > server.js << 'EOF'
const express = require('express');
const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

let users = [
    { id: 1, name: 'Alice', email: 'alice@example.com' },
    { id: 2, name: 'Bob', email: 'bob@example.com' }
];

app.get('/api/users', (req, res) => {
    res.json(users);
});

app.get('/api/users/:id', (req, res) => {
    const user = users.find(u => u.id === parseInt(req.params.id));
    if (!user) return res.status(404).json({ error: 'User not found' });
    res.json(user);
});

app.get('/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

app.listen(port, '0.0.0.0', () => {
    console.log(`API服务运行在端口 ${port}`);
});
EOF

cat > package.json << 'EOF'
{
  "name": "api-service",
  "version": "1.0.0",
  "description": "Simple API service",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "express": "^4.18.2"
  }
}
EOF

cat > Dockerfile << 'EOF'
FROM node:18-alpine

# 创建应用目录
WORKDIR /usr/src/app

# 复制package文件
COPY package*.json ./

# 安装依赖
RUN npm install --only=production

# 添加非root用户
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nodeuser -u 1001

# 复制应用代码
COPY --chown=nodeuser:nodejs . .

# 切换到非root用户
USER nodeuser

# 暴露端口
EXPOSE 3000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# 启动应用
CMD ["node", "server.js"]
EOF

# 构建并测试
docker build -t api-service .
docker run -d -p 3000:3000 --name my-api api-service
curl http://localhost:3000/api/users
```

## 进阶练习

### 练习1: 构建缓存优化
```bash
# 观察构建缓存的效果
time docker build -t cache-test .
time docker build -t cache-test .  # 第二次构建应该更快

# 修改代码后重新构建
echo "console.log('updated');" >> server.js
time docker build -t cache-test .  # 观察哪些层被重新构建
```

### 练习2: 镜像安全扫描
```bash
# 使用docker scout扫描安全漏洞（如果可用）
docker scout cves flask-app

# 检查镜像的安全配置
docker run --rm -it docker/docker-bench-security
```

### 练习3: 构建参数和目标
```bash
cat > Dockerfile.args << 'EOF'
ARG NODE_VERSION=18
FROM node:${NODE_VERSION}-alpine

ARG BUILD_DATE
ARG VERSION
LABEL build-date=$BUILD_DATE
LABEL version=$VERSION

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .

# 开发目标
FROM base AS development
RUN npm install --only=development
CMD ["npm", "run", "dev"]

# 生产目标
FROM base AS production
RUN npm ci --only=production
CMD ["npm", "start"]
EOF

# 使用构建参数
docker build \
  --build-arg NODE_VERSION=18 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VERSION=1.0.0 \
  --target production \
  -t app:prod .
```

## 最佳实践总结

### 1. 镜像优化
- 使用多阶段构建减少镜像大小
- 选择合适的基础镜像（alpine vs slim vs full）
- 合并RUN指令减少层数
- 清理包管理器缓存

### 2. 安全性
- 使用非root用户运行应用
- 定期更新基础镜像
- 不在镜像中存储敏感信息
- 使用.dockerignore排除不必要文件

### 3. 构建效率
- 利用构建缓存，先复制依赖文件
- 使用.dockerignore文件
- 合理安排指令顺序

### 4. 可维护性
- 使用明确的版本标签
- 添加有意义的标签
- 文档化Dockerfile
- 使用健康检查

## 思考题

1. 为什么要使用多阶段构建？有什么优势？
2. CMD和ENTRYPOINT的区别是什么？什么时候使用哪个？
3. COPY和ADD的区别是什么？推荐使用哪个？
4. 如何减少Docker镜像的大小？
5. 构建缓存是如何工作的？如何优化构建缓存？

## 常用命令速查

```bash
# 构建镜像
docker build -t <镜像名> .                 # 构建镜像
docker build -f <Dockerfile> -t <镜像名> . # 指定Dockerfile
docker build --no-cache -t <镜像名> .      # 不使用缓存构建

# 构建参数
--build-arg <参数>=<值>                    # 传递构建参数
--target <阶段名>                          # 构建特定阶段

# 镜像分析
docker history <镜像名>                    # 查看镜像历史
docker inspect <镜像名>                    # 查看镜像详情
```

## 下一步
完成本实验后，进入 `lab05-多阶段构建与镜像优化` 深入学习高级构建技巧。

## 参考资料
- [Dockerfile官方文档](https://docs.docker.com/engine/reference/builder/)
- [Docker最佳实践](https://docs.docker.com/develop/dev-best-practices/)
- [镜像构建优化指南](https://docs.docker.com/develop/dev-best-practices/) 