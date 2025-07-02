# 实验1：Docker基础概念与安装

## 学习目标
- 理解Docker的核心概念
- 掌握Docker的安装和基本配置
- 了解Docker架构和工作原理
- 完成第一个Docker容器运行

## 理论学习

### 1. 什么是Docker？
Docker是一个开源的容器化平台，它可以将应用程序及其依赖项打包到一个轻量级、可移植的容器中。

### 2. 核心概念
- **镜像(Image)**: 只读的模板，用于创建容器
- **容器(Container)**: 镜像的运行实例
- **Dockerfile**: 构建镜像的脚本文件
- **仓库(Repository)**: 存储镜像的地方
- **Docker Engine**: Docker的核心运行时

### 3. Docker vs 虚拟机
| 特性 | Docker容器 | 虚拟机 |
|------|-----------|--------|
| 资源占用 | 低 | 高 |
| 启动速度 | 秒级 | 分钟级 |
| 隔离性 | 进程级别 | 硬件级别 |
| 移植性 | 高 | 中等 |

## 动手实践

### 步骤1: 安装Docker

#### macOS安装
```bash
# 使用Homebrew安装
brew install --cask docker

# 或者下载Docker Desktop
# 访问：https://www.docker.com/products/docker-desktop
```

#### 验证安装
```bash
# 检查Docker版本
docker --version

# 检查Docker信息
docker info

# 检查Docker Compose版本
docker-compose --version
```

### 步骤2: 配置Docker

#### 配置镜像加速器（可选）
```bash
# 创建或编辑daemon.json
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": [
    "https://registry.docker-cn.com",
    "https://docker.mirrors.ustc.edu.cn"
  ]
}
EOF

# 重启Docker服务
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 步骤3: 第一个Docker容器

#### 运行Hello World
```bash
# 运行第一个容器
docker run hello-world

# 查看运行过程
echo "这个命令做了什么？"
echo "1. 在本地查找hello-world镜像"
echo "2. 如果没有找到，从Docker Hub下载"
echo "3. 创建并启动容器"
echo "4. 执行容器中的程序"
echo "5. 程序执行完毕后容器停止"
```

#### 运行交互式容器
```bash
# 运行Ubuntu容器并进入bash
docker run -it ubuntu:20.04 bash

# 在容器内执行以下命令
cat /etc/os-release
ls /
exit

# 查看刚才运行的容器
docker ps -a
```

### 步骤4: 基本Docker命令

```bash
# 查看本地镜像
docker images

# 查看运行中的容器
docker ps

# 查看所有容器（包括停止的）
docker ps -a

# 查看Docker系统信息
docker system df

# 清理未使用的资源
docker system prune
```

## 实验任务

### 任务1: 环境验证
完成以下操作并记录输出：
```bash
# 执行以下命令并截图或记录输出
docker --version
docker info
docker run hello-world
```

### 任务2: 探索容器
```bash
# 运行一个nginx容器
docker run -d -p 8080:80 --name my-nginx nginx

# 访问http://localhost:8080验证nginx运行
# 查看容器日志
docker logs my-nginx

# 进入容器内部
docker exec -it my-nginx bash

# 在容器内查看nginx配置
cat /etc/nginx/nginx.conf

# 退出容器
exit

# 停止并删除容器
docker stop my-nginx
docker rm my-nginx
```

### 任务3: 镜像管理
```bash
# 下载不同版本的ubuntu镜像
docker pull ubuntu:18.04
docker pull ubuntu:20.04
docker pull ubuntu:22.04

# 查看镜像大小差异
docker images ubuntu

# 删除其中一个镜像
docker rmi ubuntu:18.04
```

## 思考题

1. 为什么Docker容器比虚拟机启动更快？
2. Docker镜像的分层结构有什么优势？
3. 容器停止后数据会丢失吗？如何保持数据？
4. Docker Hub是什么？还有哪些镜像仓库？
5. 如何查看一个镜像包含哪些层？

## 常见问题

### Q: Docker Desktop启动失败
A: 确保系统开启了虚拟化功能，检查Hyper-V或VirtualBox配置

### Q: 无法拉取镜像
A: 检查网络连接，配置镜像加速器

### Q: 权限问题
A: 将用户添加到docker组：`sudo usermod -aG docker $USER`

## 下一步
完成本实验后，进入 `lab02-镜像操作与管理` 继续学习Docker镜像的详细操作。

## 参考资料
- [Docker官方文档](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [Docker最佳实践](https://docs.docker.com/develop/dev-best-practices/) 