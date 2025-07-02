# 实验3：容器生命周期管理

## 学习目标
- 掌握容器的完整生命周期：创建、启动、停止、删除
- 理解容器的各种运行模式和参数
- 学会容器的监控、日志查看和故障排查
- 掌握容器资源限制和性能优化

## 理论学习

### 1. 容器生命周期状态
```Created → Running → Paused → Stopped → Removed
    ↓        ↓        ↓        ↓         ↓
  docker   docker   docker   docker   docker
  create   start    pause    stop     rm
```

### 2. 容器运行模式
- **前台模式（默认）**: 容器在前台运行，可以看到输出
- **后台模式（-d）**: 容器在后台运行，返回容器ID
- **交互模式（-i）**: 保持STDIN开放
- **伪终端模式（-t）**: 分配一个伪终端

### 3. 容器vs进程
- 容器本质上是一个进程
- 主进程退出，容器就会停止
- 一个容器通常只运行一个主要服务

## 动手实践

### 步骤1: 容器创建与启动

#### 基本运行方式
```bash
# 运行容器的不同方式
docker run hello-world                    # 前台运行并自动删除
docker run -d nginx                       # 后台运行
docker run -it ubuntu bash                # 交互模式
docker run --rm ubuntu echo "Hello"       # 运行后自动删除

# 查看容器状态
docker ps                                 # 运行中的容器
docker ps -a                              # 所有容器（包括停止的）
```

#### 容器命名和端口映射
```bash
# 指定容器名称和端口映射
docker run -d -p 8080:80 --name web-server nginx

# 访问测试
curl http://localhost:8080

# 查看端口映射
docker port web-server
```

#### 环境变量和工作目录
```bash
# 设置环境变量
docker run -e NODE_ENV=production -e PORT=3000 node:alpine env

# 设置工作目录
docker run -w /app node:alpine pwd

# 组合使用
docker run -it -e USER=docker -w /home/docker ubuntu bash
```

### 步骤2: 容器控制操作

#### 启动和停止
```bash
# 创建但不启动容器
docker create --name test-container nginx

# 启动已创建的容器
docker start test-container

# 停止运行中的容器
docker stop test-container

# 强制停止容器
docker kill test-container

# 重启容器
docker restart test-container
```

#### 暂停和恢复
```bash
# 暂停容器
docker pause test-container

# 查看暂停状态
docker ps

# 恢复容器
docker unpause test-container
```

#### 进入运行中的容器
```bash
# 启动一个nginx容器
docker run -d --name nginx-test nginx

# 在容器中执行命令
docker exec nginx-test ls /etc/nginx

# 进入容器的交互终端
docker exec -it nginx-test bash

# 在容器内执行以下命令
ls /usr/share/nginx/html/
cat /etc/nginx/nginx.conf
exit

# 以root用户身份进入容器
docker exec -it --user root nginx-test bash
```

### 步骤3: 容器监控与日志

#### 查看容器信息
```bash
# 查看容器详细信息
docker inspect nginx-test

# 查看容器资源使用情况
docker stats nginx-test

# 实时监控所有容器
docker stats

# 查看容器进程
docker top nginx-test
```

#### 日志管理
```bash
# 查看容器日志
docker logs nginx-test

# 实时跟踪日志
docker logs -f nginx-test

# 查看最近的10行日志
docker logs --tail 10 nginx-test

# 查看指定时间段的日志
docker logs --since "2023-01-01" --until "2023-12-31" nginx-test

# 添加时间戳
docker logs -t nginx-test
```

#### 容器文件系统操作
```bash
# 从容器复制文件到主机
docker cp nginx-test:/etc/nginx/nginx.conf ./nginx.conf

# 从主机复制文件到容器
echo "Hello Docker" > test.txt
docker cp test.txt nginx-test:/tmp/

# 查看容器文件系统变化
docker diff nginx-test
```

### 步骤4: 容器资源限制

#### 内存限制
```bash
# 限制容器使用512MB内存
docker run -d --memory 512m --name memory-limited nginx

# 查看内存限制
docker inspect memory-limited | grep -i memory

# 测试内存限制（会被kill）
docker run --rm --memory 100m busybox sh -c "
  dd if=/dev/zero of=/tmp/test bs=1M count=200
"
```

#### CPU限制
```bash
# 限制CPU使用（1个CPU的50%）
docker run -d --cpus 0.5 --name cpu-limited nginx

# 限制到特定的CPU核心
docker run -d --cpuset-cpus 0,1 --name cpu-specific nginx

# 查看CPU限制
docker inspect cpu-limited | grep -i cpu
```

#### 其他资源限制
```bash
# 组合限制资源
docker run -d \
  --name resource-limited \
  --memory 256m \
  --cpus 0.5 \
  --pids-limit 100 \
  nginx

# 查看资源使用情况
docker stats resource-limited --no-stream
```

### 步骤5: 容器清理

#### 删除容器
```bash
# 删除停止的容器
docker rm nginx-test

# 强制删除运行中的容器
docker rm -f memory-limited

# 删除多个容器
docker rm cpu-limited cpu-specific resource-limited

# 删除所有停止的容器
docker container prune

# 删除所有容器（危险操作！）
docker rm -f $(docker ps -aq)
```

## 实验任务

### 任务1: 容器运行模式探索
```bash
# 1. 运行一个需要长时间运行的服务
docker run -d --name long-running nginx

# 2. 运行一个一次性任务
docker run --rm alpine echo "Task completed"

# 3. 运行一个交互式容器并安装软件
docker run -it --name interactive ubuntu bash
# 在容器内执行：apt update && apt install -y vim

# 4. 在另一个终端查看容器状态
docker ps
docker stats interactive --no-stream
```

### 任务2: 容器网络访问测试
```bash
# 1. 启动一个web服务器并映射端口
docker run -d -p 3000:80 --name web-app nginx

# 2. 启动另一个容器访问第一个容器
docker run --rm alpine sh -c "
  apk add --no-cache curl && 
  curl http://host.docker.internal:3000
"

# 3. 查看端口映射情况
docker port web-app
```

### 任务3: 容器日志和监控
```bash
# 1. 创建一个生成日志的容器
docker run -d --name log-generator busybox sh -c "
  while true; do 
    echo 'Log entry at $(date)'
    sleep 2
  done
"

# 2. 查看日志
docker logs log-generator
docker logs -f --tail 5 log-generator

# 3. 监控容器资源使用
docker stats log-generator --no-stream

# 4. 停止并清理
docker stop log-generator
docker rm log-generator
```

### 任务4: 容器文件操作
```bash
# 1. 创建一个包含数据的容器
docker run -d --name data-container nginx

# 2. 修改容器内的文件
docker exec data-container sh -c "echo 'Hello from container' > /usr/share/nginx/html/test.html"

# 3. 从容器复制文件
docker cp data-container:/usr/share/nginx/html/test.html ./

# 4. 查看文件内容
cat test.html

# 5. 查看容器文件系统变化
docker diff data-container
```

## 进阶练习

### 练习1: 健康检查
```bash
# 运行带健康检查的容器
docker run -d \
  --name health-check \
  --health-cmd="curl -f http://localhost/ || exit 1" \
  --health-interval=30s \
  --health-timeout=3s \
  --health-retries=3 \
  nginx

# 查看健康状态
docker ps
docker inspect health-check | grep -A 10 Health
```

### 练习2: 容器重启策略
```bash
# 设置不同的重启策略
docker run -d --restart=always --name always-restart nginx
docker run -d --restart=on-failure:3 --name restart-on-failure busybox sh -c "exit 1"
docker run -d --restart=unless-stopped --name unless-stopped nginx

# 测试重启策略
docker stop always-restart
sleep 5
docker ps  # 查看是否自动重启
```

### 练习3: 容器信号处理
```bash
# 创建一个处理信号的容器
docker run -d --name signal-test busybox sh -c "
  trap 'echo Received SIGTERM; exit 0' TERM
  while true; do
    echo 'Running...'
    sleep 1
  done
"

# 发送不同信号
docker kill --signal=USR1 signal-test
docker logs signal-test

# 优雅停止
docker stop signal-test
docker logs signal-test
```

## 故障排查指南

### 常见问题和解决方案

#### 容器无法启动
```bash
# 检查容器日志
docker logs <container-name>

# 检查镜像问题
docker run --rm <image> echo "test"

# 检查端口占用
netstat -tuln | grep <port>
```

#### 容器运行异常
```bash
# 检查容器进程
docker top <container-name>

# 检查资源使用
docker stats <container-name>

# 进入容器调试
docker exec -it <container-name> sh
```

#### 性能问题
```bash
# 查看容器资源限制
docker inspect <container-name> | grep -i -A 5 "resources"

# 监控资源使用
docker stats --no-stream

# 分析容器开销
docker system df
docker system events
```

## 思考题

1. 什么情况下应该使用 `docker run` vs `docker create` + `docker start`？
2. 容器的重启策略有哪些？各自适用于什么场景？
3. 如何优雅地停止一个容器？SIGTERM和SIGKILL的区别是什么？
4. 容器的健康检查机制是如何工作的？
5. 为什么建议在生产环境中限制容器资源？

## 常用命令速查

```bash
# 容器生命周期
docker run <镜像>              # 创建并启动容器
docker create <镜像>           # 创建容器
docker start <容器>            # 启动容器
docker stop <容器>             # 停止容器
docker restart <容器>          # 重启容器
docker pause <容器>            # 暂停容器
docker unpause <容器>          # 恢复容器
docker rm <容器>               # 删除容器

# 容器操作
docker ps                      # 查看运行中容器
docker ps -a                   # 查看所有容器
docker exec -it <容器> bash    # 进入容器
docker logs <容器>             # 查看日志
docker stats <容器>            # 查看资源使用
docker inspect <容器>          # 查看详细信息

# 文件操作
docker cp <源> <目标>          # 复制文件
docker diff <容器>             # 查看文件变化

# 资源限制
--memory <大小>                # 内存限制
--cpus <数量>                  # CPU限制
--restart <策略>               # 重启策略
```

## 下一步
完成本实验后，进入 `lab04-Dockerfile详解与最佳实践` 学习如何构建自定义Docker镜像。

## 参考资料
- [Docker容器官方文档](https://docs.docker.com/engine/reference/commandline/container/)
- [容器运行时参考](https://docs.docker.com/engine/reference/run/)
- [容器最佳实践](https://docs.docker.com/develop/dev-best-practices/) 