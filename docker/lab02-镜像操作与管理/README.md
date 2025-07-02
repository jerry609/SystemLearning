# 实验2：镜像操作与管理

## 学习目标
- 深入理解Docker镜像的结构和原理
- 掌握镜像的搜索、下载、查看、删除等操作
- 学会镜像的导入导出和备份
- 理解镜像分层和标签系统

## 理论学习

### 1. 镜像分层结构
Docker镜像采用联合文件系统(UnionFS)，由多个只读层组成：
- **基础层**: 通常是操作系统层
- **依赖层**: 安装的软件包和库
- **应用层**: 应用程序代码
- **配置层**: 配置文件和环境变量

### 2. 镜像标签系统
- **仓库名**: 如 `ubuntu`, `nginx`
- **标签**: 如 `latest`, `20.04`, `v1.0`
- **完整格式**: `registry/repository:tag`

### 3. 镜像存储
- 本地存储在 `/var/lib/docker/`
- 远程存储在镜像仓库（如Docker Hub）

## 动手实践

### 步骤1: 镜像搜索与获取

#### 搜索镜像
```bash
# 在Docker Hub搜索nginx镜像
docker search nginx

# 查看搜索结果的字段含义
echo "NAME: 镜像名称"
echo "DESCRIPTION: 镜像描述"
echo "STARS: 收藏数量"
echo "OFFICIAL: 是否官方镜像"
echo "AUTOMATED: 是否自动构建"

# 搜索特定类型的镜像
docker search --filter stars=1000 nginx
docker search --filter is-official=true ubuntu
```

#### 下载镜像
```bash
# 下载最新版本的nginx镜像
docker pull nginx

# 下载指定版本的镜像
docker pull nginx:1.20
docker pull nginx:alpine

# 下载指定架构的镜像
docker pull --platform linux/arm64 nginx

# 查看下载进度（在新终端中）
docker images nginx
```

### 步骤2: 镜像查看与分析

#### 查看镜像列表
```bash
# 查看所有本地镜像
docker images

# 查看镜像详细信息
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}"

# 只显示镜像ID
docker images -q

# 查看指定镜像
docker images nginx
```

#### 检查镜像详情
```bash
# 查看镜像详细信息
docker inspect nginx

# 查看镜像的层信息
docker history nginx

# 查看镜像的构建信息
docker inspect nginx | grep -A 10 "Config"
```

#### 分析镜像分层
```bash
# 使用dive工具分析镜像（需要先安装）
# brew install dive
# dive nginx

# 或者使用docker history命令
docker history nginx --format "table {{.ID}}\t{{.CreatedBy}}\t{{.Size}}"
```

### 步骤3: 镜像标签管理

#### 创建和管理标签
```bash
# 为镜像创建新标签
docker tag nginx my-nginx:v1.0
docker tag nginx my-nginx:latest

# 查看标签结果
docker images my-nginx

# 创建带仓库前缀的标签
docker tag nginx localhost:5000/my-nginx:v1.0
```

#### 重命名镜像
```bash
# 实际上是创建新标签然后删除旧标签
docker tag nginx:latest nginx:backup
docker rmi nginx:latest
docker tag nginx:backup nginx:latest
docker rmi nginx:backup
```

### 步骤4: 镜像导入导出

#### 导出镜像
```bash
# 将镜像保存为tar文件
docker save nginx > nginx.tar
docker save nginx:latest -o nginx-latest.tar

# 保存多个镜像到一个文件
docker save nginx ubuntu > multi-images.tar

# 压缩导出（节省空间）
docker save nginx | gzip > nginx.tar.gz
```

#### 导入镜像
```bash
# 从tar文件加载镜像
docker load < nginx.tar
docker load -i nginx-latest.tar

# 从压缩文件加载镜像
gunzip -c nginx.tar.gz | docker load

# 验证导入结果
docker images nginx
```

#### 容器导出为镜像
```bash
# 运行一个容器并做一些修改
docker run -it --name test-container ubuntu bash
# 在容器内执行：apt update && apt install -y curl
# exit

# 将容器提交为新镜像
docker commit test-container my-ubuntu:with-curl

# 查看新镜像
docker images my-ubuntu

# 清理测试容器
docker rm test-container
```

### 步骤5: 镜像清理与维护

#### 查看镜像使用情况
```bash
# 查看Docker磁盘使用情况
docker system df

# 查看详细的磁盘使用情况
docker system df -v

# 查看悬挂镜像（dangling images）
docker images -f dangling=true
```

#### 清理镜像
```bash
# 删除单个镜像
docker rmi nginx:alpine

# 删除多个镜像
docker rmi nginx:1.20 ubuntu:18.04

# 强制删除镜像（即使有容器在使用）
docker rmi -f nginx:latest

# 删除所有悬挂镜像
docker image prune

# 删除所有未使用的镜像
docker image prune -a

# 删除7天前创建的镜像
docker image prune -a --filter "until=168h"
```

## 实验任务

### 任务1: 镜像搜索与下载
```bash
# 1. 搜索Python镜像，找出官方镜像和星数最多的镜像
docker search python

# 2. 下载Python的多个版本
docker pull python:3.9
docker pull python:3.10
docker pull python:alpine

# 3. 比较不同版本的镜像大小
docker images python
```

### 任务2: 镜像分析
```bash
# 1. 分析python:3.9镜像的构建历史
docker history python:3.9

# 2. 查看镜像的详细配置信息
docker inspect python:3.9 | grep -A 5 "Env"

# 3. 统计镜像的层数
docker history python:3.9 --format "{{.ID}}" | wc -l
```

### 任务3: 镜像备份与恢复
```bash
# 1. 创建一个测试目录用于存放备份文件
mkdir -p backup-images

# 2. 备份Python镜像
docker save python:3.9 -o backup-images/python-3.9.tar

# 3. 删除原镜像并从备份恢复
docker rmi python:3.9
docker load -i backup-images/python-3.9.tar

# 4. 验证恢复结果
docker images python:3.9
```

### 任务4: 自定义镜像标签
```bash
# 1. 为镜像创建多个有意义的标签
docker tag python:3.9 my-python:dev
docker tag python:3.9 my-python:production
docker tag python:3.9 my-python:v3.9.0

# 2. 查看标签创建结果
docker images my-python

# 3. 模拟镜像版本升级（重新标记latest）
docker tag my-python:v3.9.0 my-python:latest
```

## 进阶练习

### 练习1: 创建轻量级镜像对比
```bash
# 比较不同基础镜像的大小
docker pull python:3.9
docker pull python:3.9-slim
docker pull python:3.9-alpine

# 分析大小差异
docker images python:3.9*

# 运行容器对比功能差异
docker run --rm python:3.9 python --version
docker run --rm python:3.9-alpine python --version
```

### 练习2: 镜像安全扫描
```bash
# 使用docker scout分析镜像安全性（如果可用）
docker scout cves python:3.9

# 或者查看镜像的元数据
docker inspect python:3.9 | grep -i security
```

## 思考题

1. 为什么alpine版本的镜像通常比较小？有什么优缺点？
2. 什么是悬挂镜像？它们是如何产生的？
3. 镜像的分层结构如何影响存储空间和传输效率？
4. 如何选择合适的基础镜像？
5. 镜像标签的最佳实践是什么？

## 常用命令速查

```bash
# 镜像操作
docker search <镜像名>          # 搜索镜像
docker pull <镜像名>            # 下载镜像
docker images                   # 查看本地镜像
docker rmi <镜像名>             # 删除镜像
docker tag <源> <目标>          # 创建标签

# 镜像信息
docker inspect <镜像名>         # 查看镜像详情
docker history <镜像名>         # 查看镜像历史

# 镜像备份
docker save <镜像名> > file.tar # 导出镜像
docker load < file.tar          # 导入镜像

# 镜像清理
docker image prune              # 清理悬挂镜像
docker image prune -a           # 清理所有未使用镜像
docker system df                # 查看磁盘使用情况
```

## 下一步
完成本实验后，进入 `lab03-容器生命周期管理` 学习容器的创建、运行、停止等生命周期管理。

## 参考资料
- [Docker镜像官方文档](https://docs.docker.com/engine/reference/commandline/images/)
- [Docker Hub](https://hub.docker.com/)
- [镜像最佳实践](https://docs.docker.com/develop/dev-best-practices/) 