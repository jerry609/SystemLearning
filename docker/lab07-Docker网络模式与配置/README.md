# 实验7：Docker网络模式与配置

## 学习目标
- 深入理解Docker网络架构和模式
- 掌握各种网络模式的应用场景
- 学会自定义网络配置和管理
- 理解容器间通信机制
- 掌握网络故障排查方法

## 理论学习

### Docker网络架构
Docker网络基于Linux网络命名空间和网桥技术，提供了容器间和容器与外部世界的网络连接。

### 网络模式详解

| 网络模式 | 说明 | 使用场景 | 优缺点 |
|---------|------|----------|---------|
| bridge | 默认模式，使用docker0网桥 | 单主机容器通信 | 简单易用，性能一般 |
| host | 直接使用宿主机网络 | 需要高性能网络 | 性能最佳，隔离性差 |
| none | 禁用网络功能 | 安全要求极高 | 完全隔离，无网络功能 |
| container | 共享其他容器网络 | 紧密耦合的容器 | 网络共享，生命周期耦合 |
| 自定义网络 | 用户定义的网络 | 微服务架构 | 灵活可控，功能丰富 |

### 网络组件
- **Docker0网桥**: 默认网桥，连接所有容器
- **veth pair**: 虚拟网络设备对，连接容器和网桥
- **iptables**: 管理网络规则和端口映射
- **DNS**: 容器名称解析服务

## 动手实践

### 1. 网络模式实验

#### 1.1 Bridge模式（默认）
```bash
# 创建容器（默认bridge模式）
docker run -d --name web1 nginx
docker run -d --name web2 nginx

# 查看网络信息
docker network ls
docker network inspect bridge

# 查看容器IP
docker inspect web1 | grep IPAddress
docker inspect web2 | grep IPAddress

# 容器间通信测试
docker exec web1 ping -c 3 172.17.0.3  # web2的IP
```

#### 1.2 Host模式
```bash
# 使用host网络模式
docker run -d --name web-host --network host nginx

# 查看端口（直接使用宿主机80端口）
netstat -tlnp | grep :80

# 访问测试
curl localhost:80
```

#### 1.3 None模式
```bash
# 创建无网络容器
docker run -it --name no-network --network none alpine sh

# 在容器内检查网络（只有lo接口）
ifconfig
```

#### 1.4 Container模式
```bash
# 创建第一个容器
docker run -d --name app1 nginx

# 创建共享网络的容器
docker run -it --name app2 --network container:app1 alpine sh

# 在app2中查看网络（与app1相同）
ifconfig
```

### 2. 自定义网络

#### 2.1 创建自定义bridge网络
```bash
# 创建自定义网络
docker network create --driver bridge my-network

# 查看网络详情
docker network inspect my-network

# 指定子网和网关
docker network create \
  --driver bridge \
  --subnet=192.168.1.0/24 \
  --gateway=192.168.1.1 \
  custom-network
```

#### 2.2 容器连接自定义网络
```bash
# 在自定义网络中运行容器
docker run -d --name db --network my-network postgres:13

# 运行另一个容器并连接到同一网络
docker run -d --name app --network my-network nginx

# 测试DNS解析（容器名直接解析）
docker exec app ping db
docker exec app nslookup db
```

#### 2.3 动态网络连接
```bash
# 运行容器
docker run -d --name web nginx

# 连接到自定义网络
docker network connect my-network web

# 断开网络连接
docker network disconnect bridge web

# 查看容器网络
docker inspect web
```

### 3. 端口映射和发布

#### 3.1 端口映射基础
```bash
# 映射单个端口
docker run -d -p 8080:80 --name web-8080 nginx

# 映射多个端口
docker run -d -p 8081:80 -p 4443:443 --name web-multi nginx

# 映射到指定IP
docker run -d -p 127.0.0.1:8082:80 --name web-local nginx

# 随机端口映射
docker run -d -P --name web-random nginx
docker port web-random
```

#### 3.2 UDP端口映射
```bash
# 映射UDP端口
docker run -d -p 5000:5000/udp --name udp-app alpine nc -ul 5000

# 测试UDP连接
echo "Hello UDP" | nc -u localhost 5000
```

### 4. 高级网络配置

#### 4.1 网络别名
```bash
# 创建带别名的容器
docker run -d --name db --network my-network --network-alias database postgres:13

# 通过别名访问
docker run --rm --network my-network alpine nslookup database
```

#### 4.2 多网络连接
```bash
# 创建多个网络
docker network create frontend
docker network create backend

# 容器连接多个网络
docker run -d --name app nginx
docker network connect frontend app
docker network connect backend app

# 查看网络连接
docker inspect app
```

#### 4.3 网络隔离
```bash
# 创建隔离网络
docker network create --internal isolated-network

# 在隔离网络中运行容器（无法访问外网）
docker run -it --name isolated --network isolated-network alpine sh

# 测试外网连接（会失败）
ping 8.8.8.8
```

## 实验任务

### 任务1：微服务网络架构
创建一个包含前端、后端和数据库的微服务架构，要求：
- 前端可以访问后端
- 后端可以访问数据库
- 数据库与外网隔离
- 前端通过端口映射对外提供服务

```bash
# 创建网络
docker network create frontend-net
docker network create backend-net

# 运行数据库（仅在后端网络）
docker run -d --name postgres-db \
  --network backend-net \
  -e POSTGRES_PASSWORD=secret \
  postgres:13

# 运行后端API（连接两个网络）
docker run -d --name api-server nginx
docker network connect frontend-net api-server
docker network connect backend-net api-server

# 运行前端（仅在前端网络，对外暴露端口）
docker run -d --name frontend \
  --network frontend-net \
  -p 8080:80 \
  nginx

# 验证网络连通性
docker exec frontend ping api-server
docker exec api-server ping postgres-db
```

### 任务2：负载均衡网络
使用多个后端容器实现简单的负载均衡：

```bash
# 创建网络
docker network create lb-network

# 启动多个后端服务
for i in {1..3}; do
  docker run -d --name backend-$i \
    --network lb-network \
    --network-alias backend \
    nginx
done

# 验证DNS轮询
docker run --rm --network lb-network alpine sh -c "
  for i in {1..6}; do
    nslookup backend
    echo '---'
  done
"
```

### 任务3：跨主机网络（Overlay）
如果有多台Docker主机，练习overlay网络：

```bash
# 初始化Swarm（在管理节点）
docker swarm init

# 创建overlay网络
docker network create -d overlay my-overlay

# 创建服务使用overlay网络
docker service create --name web \
  --network my-overlay \
  --replicas 3 \
  nginx
```

### 任务4：网络监控和调试
学习网络故障排查：

```bash
# 安装网络工具
docker run -it --rm --cap-add NET_ADMIN nicolaka/netshoot

# 在容器中进行网络诊断
# tcpdump、nmap、dig、curl等工具

# 查看网络统计
docker exec container_name netstat -i
docker exec container_name ss -tuln

# 抓包分析
docker exec container_name tcpdump -i eth0
```

## 进阶练习

### 1. 自定义网络驱动
```bash
# 使用macvlan驱动
docker network create -d macvlan \
  --subnet=192.168.1.0/24 \
  --gateway=192.168.1.1 \
  -o parent=eth0 \
  macvlan-net

# 使用ipvlan驱动
docker network create -d ipvlan \
  --subnet=192.168.2.0/24 \
  --gateway=192.168.2.1 \
  -o parent=eth0 \
  ipvlan-net
```

### 2. 网络性能测试
```bash
# 创建性能测试脚本
cat > network-test.sh << 'EOF'
#!/bin/bash

# 测试网络延迟
echo "=== 网络延迟测试 ==="
docker exec container1 ping -c 10 container2

# 测试网络带宽
echo "=== 网络带宽测试 ==="
docker exec container1 iperf3 -s &
docker exec container2 iperf3 -c container1 -t 10

EOF

chmod +x network-test.sh
```

### 3. 网络安全配置
```bash
# 禁用容器间通信
docker network create --opt com.docker.network.bridge.enable_icc=false secure-net

# 使用防火墙规则
iptables -I DOCKER-USER -s 172.17.0.0/16 -d 172.17.0.0/16 -j DROP
```

## 故障排查指南

### 常见网络问题

1. **容器无法访问外网**
   ```bash
   # 检查DNS配置
   docker exec container cat /etc/resolv.conf
   
   # 检查路由
   docker exec container ip route
   
   # 检查防火墙
   iptables -L DOCKER-USER
   ```

2. **容器间无法通信**
   ```bash
   # 检查网络连接
   docker network inspect network_name
   
   # 检查容器IP
   docker inspect container_name | grep IPAddress
   
   # 测试连通性
   docker exec container1 ping container2
   ```

3. **端口映射失败**
   ```bash
   # 检查端口占用
   netstat -tlnp | grep port
   
   # 检查iptables规则
   iptables -t nat -L DOCKER
   
   # 检查容器端口
   docker port container_name
   ```

### 网络调试工具

1. **容器内网络诊断**
   ```bash
   # 安装网络工具容器
   docker run -it --rm --net container:target_container nicolaka/netshoot
   
   # 常用诊断命令
   ip addr show
   ip route show
   netstat -tlnp
   ss -tuln
   nslookup hostname
   dig hostname
   tcpdump -i any
   ```

2. **宿主机网络检查**
   ```bash
   # 查看Docker网络接口
   ip addr show docker0
   
   # 查看网桥信息
   brctl show
   
   # 查看iptables规则
   iptables -t nat -L
   iptables -L DOCKER
   ```

## 思考题

1. **为什么Docker默认使用bridge网络模式？它有什么优缺点？**

2. **在什么情况下应该使用host网络模式？有什么安全隐患？**

3. **如何实现容器的网络隔离？有哪些方法？**

4. **overlay网络如何实现跨主机通信？底层原理是什么？**

5. **如何设计一个安全的微服务网络架构？**

## 常用命令速查

### 网络管理
```bash
# 查看所有网络
docker network ls

# 创建网络
docker network create [OPTIONS] NETWORK

# 删除网络
docker network rm NETWORK

# 查看网络详情
docker network inspect NETWORK

# 清理未使用网络
docker network prune
```

### 容器网络操作
```bash
# 连接容器到网络
docker network connect NETWORK CONTAINER

# 断开容器网络连接
docker network disconnect NETWORK CONTAINER

# 查看容器端口映射
docker port CONTAINER

# 指定网络运行容器
docker run --network NETWORK IMAGE
```

### 网络诊断
```bash
# 查看容器网络配置
docker inspect CONTAINER

# 容器内执行网络命令
docker exec CONTAINER ip addr
docker exec CONTAINER ping HOST
docker exec CONTAINER netstat -tlnp
```

## 参考资料

- [Docker官方网络文档](https://docs.docker.com/network/)
- [Docker网络模式详解](https://docs.docker.com/network/drivers/)
- [容器网络接口CNI](https://github.com/containernetworking/cni)
- [Linux网络命名空间](https://man7.org/linux/man-pages/man7/network_namespaces.7.html)

## 下一步
完成本实验后，继续学习：
- **实验8**: Docker Compose多服务编排
- 深入了解容器编排和服务发现
- 学习复杂应用的部署和管理 