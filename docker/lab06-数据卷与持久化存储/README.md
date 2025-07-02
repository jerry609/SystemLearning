# 实验6：数据卷与持久化存储

## 学习目标
- 理解Docker数据持久化的重要性和原理
- 掌握三种数据卷类型的使用方法
- 学会数据卷的管理和备份恢复
- 了解数据卷在不同场景下的最佳实践

## 理论学习

### 1. 数据持久化需求
容器本身是无状态的，容器删除后数据会丢失。数据卷解决了以下问题：
- 数据持久化存储
- 容器间数据共享
- 主机与容器数据交换
- 数据备份和恢复

### 2. Docker存储类型
| 类型 | 位置 | 生命周期 | 性能 | 使用场景 |
|------|------|----------|------|----------|
| 卷(Volume) | Docker管理的目录 | 独立于容器 | 高 | 数据持久化 |
| 绑定挂载(Bind Mount) | 主机任意目录 | 跟随主机 | 高 | 开发调试 |
| 临时文件系统(tmpfs) | 内存 | 容器生命周期 | 最高 | 临时数据 |

### 3. 数据卷特点
- **命名卷**: 由Docker管理，可复用
- **匿名卷**: 自动创建，难以管理
- **绑定挂载**: 直接映射主机目录

## 动手实践

### 步骤1: 数据卷基础操作

#### 卷的创建和管理
```bash
# 创建命名卷
docker volume create my-volume
docker volume create app-data
docker volume create db-data

# 查看所有卷
docker volume ls

# 查看卷详细信息
docker volume inspect my-volume

# 查看卷在主机上的位置
docker volume inspect my-volume | grep Mountpoint
```

#### 使用数据卷运行容器
```bash
# 使用命名卷
docker run -d --name web-with-volume \
  -v my-volume:/var/www/html \
  -p 8080:80 \
  nginx

# 在卷中创建文件
docker exec web-with-volume sh -c "echo 'Hello Volume!' > /var/www/html/index.html"

# 访问验证
curl http://localhost:8080

# 删除容器，数据仍然存在
docker rm -f web-with-volume

# 创建新容器使用同一个卷
docker run -d --name web-restored \
  -v my-volume:/var/www/html \
  -p 8081:80 \
  nginx

# 验证数据依然存在
curl http://localhost:8081
```

#### 匿名卷vs命名卷
```bash
# 创建匿名卷
docker run -d --name test-anonymous \
  -v /data \
  alpine sleep 3600

# 创建命名卷
docker run -d --name test-named \
  -v named-volume:/data \
  alpine sleep 3600

# 查看差异
docker volume ls
docker inspect test-anonymous | grep -A 10 Mounts
docker inspect test-named | grep -A 10 Mounts
```

### 步骤2: 绑定挂载实战

#### 开发环境数据同步
```bash
# 创建本地项目目录
mkdir -p ~/docker-dev-example
cd ~/docker-dev-example

# 创建简单的Web项目
cat > index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>开发测试</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .update-time { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Docker绑定挂载演示</h1>
        <p>这个文件是从主机绑定挂载的</p>
        <p class="update-time">更新时间: <span id="time"></span></p>
    </div>
    <script>
        document.getElementById('time').textContent = new Date().toLocaleString();
    </script>
</body>
</html>
EOF

cat > style.css << 'EOF'
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
.container {
    background: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 10px;
}
EOF

# 使用绑定挂载运行nginx
docker run -d --name dev-server \
  -v $(pwd):/usr/share/nginx/html \
  -p 8082:80 \
  nginx

# 测试访问
curl http://localhost:8082

# 修改文件测试实时更新
echo '<link rel="stylesheet" href="style.css">' >> index.html

# 再次访问查看变化
curl http://localhost:8082
```

#### 配置文件管理
```bash
# 创建自定义nginx配置
mkdir -p ~/nginx-config
cd ~/nginx-config

cat > nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log;
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            root /usr/share/nginx/html;
            index index.html;
        }
        
        location /api/ {
            return 200 "API endpoint working\n";
            add_header Content-Type text/plain;
        }
        
        location /status {
            access_log off;
            return 200 "OK\n";
            add_header Content-Type text/plain;
        }
    }
}
EOF

# 使用自定义配置启动nginx
docker run -d --name custom-nginx \
  -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf \
  -v ~/docker-dev-example:/usr/share/nginx/html \
  -p 8083:80 \
  nginx

# 测试自定义配置
curl http://localhost:8083/api/
curl http://localhost:8083/status
```

### 步骤3: 数据库持久化实战

#### MySQL数据持久化
```bash
# 创建MySQL数据卷
docker volume create mysql-data

# 启动MySQL容器
docker run -d \
  --name mysql-persistent \
  -e MYSQL_ROOT_PASSWORD=secret123 \
  -e MYSQL_DATABASE=testdb \
  -e MYSQL_USER=testuser \
  -e MYSQL_PASSWORD=testpass \
  -v mysql-data:/var/lib/mysql \
  -p 3306:3306 \
  mysql:8.0

# 等待MySQL启动
sleep 30

# 连接并创建测试数据
docker exec -it mysql-persistent mysql -u testuser -ptestpass testdb << 'EOF'
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO users (username, email) VALUES 
('alice', 'alice@example.com'),
('bob', 'bob@example.com'),
('charlie', 'charlie@example.com');

SELECT * FROM users;
EXIT
EOF

# 删除容器但保留数据
docker rm -f mysql-persistent

# 重新启动MySQL容器
docker run -d \
  --name mysql-restored \
  -e MYSQL_ROOT_PASSWORD=secret123 \
  -e MYSQL_DATABASE=testdb \
  -e MYSQL_USER=testuser \
  -e MYSQL_PASSWORD=testpass \
  -v mysql-data:/var/lib/mysql \
  -p 3306:3306 \
  mysql:8.0

# 等待启动并验证数据
sleep 30
docker exec mysql-restored mysql -u testuser -ptestpass testdb -e "SELECT * FROM users;"
```

#### PostgreSQL数据持久化
```bash
# 创建PostgreSQL数据卷和初始化脚本目录
docker volume create postgres-data
mkdir -p ~/postgres-init

# 创建初始化脚本
cat > ~/postgres-init/init.sql << 'EOF'
-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 插入测试数据
INSERT INTO users (username, email, full_name) VALUES 
('admin', 'admin@example.com', 'System Administrator'),
('alice', 'alice@example.com', 'Alice Johnson'),
('bob', 'bob@example.com', 'Bob Smith')
ON CONFLICT (username) DO NOTHING;

-- 创建文章表
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    author_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 插入文章数据
INSERT INTO articles (title, content, author_id) VALUES 
('Docker入门', 'Docker是一个开源的容器化平台...', 1),
('数据持久化', '在Docker中实现数据持久化的方法...', 2)
ON CONFLICT DO NOTHING;
EOF

# 启动PostgreSQL容器
docker run -d \
  --name postgres-persistent \
  -e POSTGRES_DB=appdb \
  -e POSTGRES_USER=appuser \
  -e POSTGRES_PASSWORD=apppass \
  -v postgres-data:/var/lib/postgresql/data \
  -v ~/postgres-init:/docker-entrypoint-initdb.d \
  -p 5432:5432 \
  postgres:15

# 等待启动并验证数据
sleep 20
docker exec postgres-persistent psql -U appuser -d appdb -c "
SELECT u.username, u.email, a.title 
FROM users u 
JOIN articles a ON u.id = a.author_id;
"
```

### 步骤4: 多容器数据共享

#### 应用和数据库数据共享
```bash
# 创建共享数据卷
docker volume create shared-data
docker volume create logs-data

# 创建应用容器（模拟Web应用）
docker run -d \
  --name web-app \
  -v shared-data:/app/data \
  -v logs-data:/app/logs \
  -e APP_ENV=production \
  busybox sh -c "
    while true; do
      echo 'Web app log entry' >> /app/logs/app.log
      echo 'Shared data update' > /app/data/status.txt
      date >> /app/data/status.txt
      sleep 10
    done
  "

# 创建数据处理容器
docker run -d \
  --name data-processor \
  -v shared-data:/data \
  -v logs-data:/logs \
  busybox sh -c "
    while true; do
      echo 'Processing data...' >> /logs/processor.log
      if [ -f /data/status.txt ]; then
        echo 'Found shared data:' >> /logs/processor.log
        cat /data/status.txt >> /logs/processor.log
      fi
      sleep 15
    done
  "

# 创建日志监控容器
docker run -d \
  --name log-monitor \
  -v logs-data:/logs \
  busybox sh -c "
    while true; do
      echo '=== Log Summary ===' >> /logs/monitor.log
      echo 'App log lines:' >> /logs/monitor.log
      wc -l /logs/app.log >> /logs/monitor.log
      echo 'Processor log lines:' >> /logs/monitor.log
      wc -l /logs/processor.log >> /logs/monitor.log
      echo '==================' >> /logs/monitor.log
      sleep 30
    done
  "

# 等待一段时间后查看日志
sleep 60

echo "=== Web App Logs ==="
docker exec web-app tail -5 /app/logs/app.log

echo "=== Data Processor Logs ==="
docker exec data-processor tail -5 /logs/processor.log

echo "=== Monitor Logs ==="
docker exec log-monitor tail -10 /logs/monitor.log

echo "=== Shared Data ==="
docker exec web-app cat /app/data/status.txt
```

### 步骤5: 数据卷备份和恢复

#### 数据卷备份
```bash
# 创建备份目录
mkdir -p ~/docker-backups

# 停止使用数据卷的容器（可选，确保数据一致性）
docker stop web-app data-processor log-monitor

# 备份数据卷到tar文件
backup_volume() {
    local volume_name=$1
    local backup_name=$2
    
    docker run --rm \
      -v ${volume_name}:/data \
      -v ~/docker-backups:/backup \
      alpine tar czf /backup/${backup_name} -C /data .
    
    echo "备份完成: ${backup_name}"
}

# 执行备份
backup_volume "shared-data" "shared-data-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
backup_volume "logs-data" "logs-data-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
backup_volume "mysql-data" "mysql-data-backup-$(date +%Y%m%d-%H%M%S).tar.gz"

# 查看备份文件
ls -lh ~/docker-backups/
```

#### 数据卷恢复
```bash
# 创建新的数据卷用于恢复测试
docker volume create shared-data-restored
docker volume create logs-data-restored

# 恢复数据卷
restore_volume() {
    local volume_name=$1
    local backup_file=$2
    
    docker run --rm \
      -v ${volume_name}:/data \
      -v ~/docker-backups:/backup \
      alpine tar xzf /backup/${backup_file} -C /data
    
    echo "恢复完成: ${volume_name}"
}

# 执行恢复（替换为实际的备份文件名）
SHARED_BACKUP=$(ls ~/docker-backups/shared-data-backup-*.tar.gz | head -1 | xargs basename)
LOGS_BACKUP=$(ls ~/docker-backups/logs-data-backup-*.tar.gz | head -1 | xargs basename)

restore_volume "shared-data-restored" "$SHARED_BACKUP"
restore_volume "logs-data-restored" "$LOGS_BACKUP"

# 验证恢复的数据
docker run --rm \
  -v shared-data-restored:/data \
  alpine cat /data/status.txt

docker run --rm \
  -v logs-data-restored:/logs \
  alpine sh -c "echo '=== App Logs ==='; tail -3 /logs/app.log"
```

### 步骤6: tmpfs临时文件系统

#### 内存中的临时存储
```bash
# 使用tmpfs挂载
docker run -d \
  --name temp-storage-test \
  --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  alpine sh -c "
    while true; do
      echo 'Writing to memory...'
      dd if=/dev/zero of=/tmp/testfile bs=1M count=10 2>/dev/null
      ls -lh /tmp/testfile
      rm /tmp/testfile
      sleep 5
    done
  "

# 查看容器内存使用
docker stats temp-storage-test --no-stream

# 验证tmpfs特性（重启后数据消失）
docker exec temp-storage-test sh -c "echo 'temp data' > /tmp/test.txt && cat /tmp/test.txt"
docker restart temp-storage-test
sleep 2
docker exec temp-storage-test ls -la /tmp/
```

## 实验任务

### 任务1: 完整的Web应用数据管理
```bash
# 创建完整的Web应用栈
mkdir -p ~/web-stack-demo
cd ~/web-stack-demo

# 创建应用代码
cat > app.py << 'EOF'
from flask import Flask, request, jsonify, render_template_string
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)

# 数据库初始化
def init_db():
    conn = sqlite3.connect('/data/app.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Docker数据持久化演示</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            .message { border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
            input, button { padding: 8px; margin: 5px; }
        </style>
    </head>
    <body>
        <h1>消息板</h1>
        <form action="/add" method="post">
            <input type="text" name="message" placeholder="输入消息" required>
            <button type="submit">添加消息</button>
        </form>
        <div id="messages"></div>
        <script>
            function loadMessages() {
                fetch('/messages')
                    .then(response => response.json())
                    .then(data => {
                        const container = document.getElementById('messages');
                        container.innerHTML = data.map(msg => 
                            '<div class="message">' + msg.content + 
                            '<br><small>' + msg.timestamp + '</small></div>'
                        ).join('');
                    });
            }
            loadMessages();
            setInterval(loadMessages, 5000);
        </script>
    </body>
    </html>
    ''')

@app.route('/add', methods=['POST'])
def add_message():
    message = request.form['message']
    conn = sqlite3.connect('/data/app.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO messages (content) VALUES (?)', (message,))
    conn.commit()
    conn.close()
    return redirect('/')

@app.route('/messages')
def get_messages():
    conn = sqlite3.connect('/data/app.db')
    cursor = conn.cursor()
    cursor.execute('SELECT content, timestamp FROM messages ORDER BY timestamp DESC LIMIT 10')
    messages = [{'content': row[0], 'timestamp': row[1]} for row in cursor.fetchall()]
    conn.close()
    return jsonify(messages)

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
EOF

# 创建requirements.txt
echo "Flask==2.3.3" > requirements.txt

# 创建Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
EOF

# 构建应用镜像
docker build -t web-stack-demo .

# 创建数据卷
docker volume create app-data
docker volume create app-logs

# 运行应用
docker run -d \
  --name web-app-demo \
  -v app-data:/data \
  -v app-logs:/app/logs \
  -p 5000:5000 \
  web-stack-demo

# 测试应用
sleep 5
curl -X POST -d "message=Hello from Docker!" http://localhost:5000/add
curl -X POST -d "message=Data persists across restarts!" http://localhost:5000/add
curl http://localhost:5000/messages

# 重启容器验证数据持久化
docker restart web-app-demo
sleep 5
curl http://localhost:5000/messages
```

### 任务2: 开发环境快速搭建
```bash
# 创建开发环境项目
mkdir -p ~/dev-environment
cd ~/dev-environment

# 创建Node.js项目结构
mkdir -p src public

# 创建package.json
cat > package.json << 'EOF'
{
  "name": "dev-environment",
  "version": "1.0.0",
  "description": "Docker开发环境演示",
  "main": "src/server.js",
  "scripts": {
    "start": "node src/server.js",
    "dev": "nodemon src/server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "nodemon": "^3.0.1"
  }
}
EOF

# 创建服务器代码
cat > src/server.js << 'EOF'
const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const port = 3000;

app.use(express.static('public'));
app.use(express.json());

// 数据存储文件
const dataFile = '/data/app-data.json';

// 确保数据文件存在
function ensureDataFile() {
    const dir = path.dirname(dataFile);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
    if (!fs.existsSync(dataFile)) {
        fs.writeFileSync(dataFile, JSON.stringify({ items: [] }));
    }
}

// 读取数据
function readData() {
    ensureDataFile();
    const data = fs.readFileSync(dataFile, 'utf8');
    return JSON.parse(data);
}

// 写入数据
function writeData(data) {
    fs.writeFileSync(dataFile, JSON.stringify(data, null, 2));
}

app.get('/api/items', (req, res) => {
    const data = readData();
    res.json(data.items);
});

app.post('/api/items', (req, res) => {
    const data = readData();
    const newItem = {
        id: Date.now(),
        name: req.body.name,
        created: new Date().toISOString()
    };
    data.items.push(newItem);
    writeData(data);
    res.json(newItem);
});

app.listen(port, '0.0.0.0', () => {
    console.log(`开发服务器运行在 http://localhost:${port}`);
});
EOF

# 创建前端页面
cat > public/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>开发环境演示</title>
    <style>
        body { font-family: Arial; margin: 40px; }
        .item { border: 1px solid #ddd; padding: 10px; margin: 5px 0; }
        input, button { padding: 8px; margin: 5px; }
    </style>
</head>
<body>
    <h1>开发环境数据持久化演示</h1>
    <div>
        <input type="text" id="itemName" placeholder="输入项目名称">
        <button onclick="addItem()">添加项目</button>
    </div>
    <div id="items"></div>

    <script>
        function loadItems() {
            fetch('/api/items')
                .then(response => response.json())
                .then(items => {
                    const container = document.getElementById('items');
                    container.innerHTML = items.map(item => 
                        '<div class="item">' + item.name + 
                        '<br><small>创建时间: ' + new Date(item.created).toLocaleString() + '</small></div>'
                    ).join('');
                });
        }

        function addItem() {
            const name = document.getElementById('itemName').value;
            if (!name) return;

            fetch('/api/items', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name })
            })
            .then(() => {
                document.getElementById('itemName').value = '';
                loadItems();
            });
        }

        // 初始加载
        loadItems();
    </script>
</body>
</html>
EOF

# 创建开发环境Docker配置
cat > Dockerfile.dev << 'EOF'
FROM node:18-alpine

WORKDIR /app

# 安装nodemon用于热重载
RUN npm install -g nodemon

# 暴露端口
EXPOSE 3000

# 启动开发服务器
CMD ["npm", "run", "dev"]
EOF

# 创建数据卷
docker volume create dev-data
docker volume create node-modules

# 运行开发容器
docker run -d \
  --name dev-container \
  -v $(pwd):/app \
  -v node-modules:/app/node_modules \
  -v dev-data:/data \
  -p 3000:3000 \
  -w /app \
  node:18-alpine sh -c "npm install && npm run dev"

# 等待启动
sleep 10

# 测试API
curl -X POST -H "Content-Type: application/json" \
  -d '{"name":"测试项目1"}' http://localhost:3000/api/items

curl -X POST -H "Content-Type: application/json" \
  -d '{"name":"Docker数据卷演示"}' http://localhost:3000/api/items

curl http://localhost:3000/api/items

echo "开发环境已启动，访问 http://localhost:3000 查看效果"
```

## 进阶练习

### 练习1: 数据卷性能测试
```bash
# 创建性能测试脚本
cat > volume-performance-test.sh << 'EOF'
#!/bin/bash

echo "=== Docker数据卷性能测试 ==="

# 创建测试卷
docker volume create test-volume

# 测试函数
test_performance() {
    local mount_type=$1
    local mount_option=$2
    local test_name=$3
    
    echo "测试 $test_name..."
    
    # 写入测试
    write_time=$(docker run --rm $mount_option alpine sh -c "
        time dd if=/dev/zero of=/test/testfile bs=1M count=100 2>&1 | grep real
    " | awk '{print $2}')
    
    # 读取测试
    read_time=$(docker run --rm $mount_option alpine sh -c "
        time dd if=/test/testfile of=/dev/null bs=1M 2>&1 | grep real
    " | awk '{print $2}')
    
    echo "  写入时间: $write_time"
    echo "  读取时间: $read_time"
    echo ""
    
    # 清理测试文件
    docker run --rm $mount_option alpine rm -f /test/testfile
}

# 测试不同挂载类型
test_performance "volume" "-v test-volume:/test" "命名卷"
test_performance "bind" "-v $(pwd)/test-data:/test" "绑定挂载"
test_performance "tmpfs" "--tmpfs /test:rw,size=512m" "tmpfs"

# 清理
docker volume rm test-volume
rm -rf $(pwd)/test-data

echo "性能测试完成！"
EOF

chmod +x volume-performance-test.sh
mkdir -p test-data
./volume-performance-test.sh
```

### 练习2: 数据卷监控和管理
```bash
# 创建数据卷管理脚本
cat > volume-manager.sh << 'EOF'
#!/bin/bash

show_volume_usage() {
    echo "=== Docker数据卷使用情况 ==="
    echo "卷名称                    大小      挂载点"
    echo "-----------------------------------------------"
    
    for volume in $(docker volume ls -q); do
        # 获取卷信息
        mountpoint=$(docker volume inspect $volume | jq -r '.[0].Mountpoint')
        
        # 计算大小（需要sudo权限）
        if [[ -d "$mountpoint" ]]; then
            size=$(sudo du -sh "$mountpoint" 2>/dev/null | cut -f1)
        else
            size="N/A"
        fi
        
        printf "%-25s %-10s %s\n" "$volume" "$size" "$mountpoint"
    done
}

cleanup_unused_volumes() {
    echo "清理未使用的数据卷..."
    docker volume prune -f
    echo "清理完成！"
}

backup_all_volumes() {
    echo "备份所有数据卷..."
    mkdir -p ~/docker-volume-backups
    
    for volume in $(docker volume ls -q); do
        echo "备份卷: $volume"
        docker run --rm \
            -v $volume:/data \
            -v ~/docker-volume-backups:/backup \
            alpine tar czf /backup/${volume}-$(date +%Y%m%d).tar.gz -C /data .
    done
    
    echo "备份完成！文件保存在 ~/docker-volume-backups/"
}

case "$1" in
    "usage")
        show_volume_usage
        ;;
    "cleanup")
        cleanup_unused_volumes
        ;;
    "backup")
        backup_all_volumes
        ;;
    *)
        echo "用法: $0 {usage|cleanup|backup}"
        echo "  usage  - 显示数据卷使用情况"
        echo "  cleanup - 清理未使用的数据卷"
        echo "  backup - 备份所有数据卷"
        ;;
esac
EOF

chmod +x volume-manager.sh

# 使用管理脚本
./volume-manager.sh usage
```

## 故障排查

### 常见问题解决
```bash
# 1. 权限问题排查
check_volume_permissions() {
    local volume_name=$1
    echo "检查卷权限: $volume_name"
    
    docker run --rm -v $volume_name:/data alpine ls -la /data
    docker run --rm -v $volume_name:/data alpine id
}

# 2. 数据卷空间不足
check_volume_space() {
    echo "检查Docker系统空间使用:"
    docker system df -v
}

# 3. 挂载点冲突
check_mount_conflicts() {
    local container_name=$1
    echo "检查容器挂载点: $container_name"
    
    docker inspect $container_name | jq '.[0].Mounts'
}

# 示例使用
check_volume_permissions "mysql-data"
check_volume_space
```

## 思考题

1. 什么时候使用命名卷vs绑定挂载vs tmpfs？
2. 如何确保多容器环境下的数据一致性？
3. 数据卷的备份策略应该如何设计？
4. 在生产环境中如何监控数据卷的使用情况？
5. 容器化数据库的数据持久化有哪些注意事项？

## 最佳实践总结

### 1. 数据卷选择
- **命名卷**: 生产环境数据持久化
- **绑定挂载**: 开发环境代码同步
- **tmpfs**: 临时数据和敏感信息

### 2. 数据管理
- 定期备份重要数据卷
- 监控数据卷空间使用
- 及时清理未使用的卷

### 3. 安全考虑
- 合理设置文件权限
- 避免挂载敏感系统目录
- 使用专用用户运行容器

### 4. 性能优化
- 根据I/O特性选择存储类型
- 避免跨网络的远程挂载
- 合理配置数据卷大小

## 常用命令速查

```bash
# 数据卷管理
docker volume create <卷名>              # 创建数据卷
docker volume ls                         # 列出所有卷
docker volume inspect <卷名>             # 查看卷详情
docker volume rm <卷名>                  # 删除数据卷
docker volume prune                      # 清理未使用的卷

# 挂载选项
-v <卷名>:<容器路径>                      # 命名卷挂载
-v <主机路径>:<容器路径>                  # 绑定挂载
--tmpfs <容器路径>                       # tmpfs挂载
--mount type=volume,source=<卷名>,target=<容器路径>  # 详细挂载语法

# 数据操作
docker cp <文件> <容器>:<路径>            # 复制文件到容器
docker cp <容器>:<路径> <文件>            # 从容器复制文件
```

## 下一步
完成本实验后，进入 `lab07-Docker网络模式与配置` 学习Docker网络管理。

## 参考资料
- [Docker数据卷官方文档](https://docs.docker.com/storage/volumes/)
- [绑定挂载文档](https://docs.docker.com/storage/bind-mounts/)
- [tmpfs挂载文档](https://docs.docker.com/storage/tmpfs/) 