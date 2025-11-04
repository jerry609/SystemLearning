# 数据流图绘制练习

## 目标

通过绘制数据流图，理解 HPFS Bug 中各个组件之间的交互关系。

## 练习 1: 正常流程（单任务）

绘制单个任务的数据流图。

### 组件

- FlowControlManager
- Task (LimitFlow)
- tokensChan (本地)
- totalFlowTokenChan (全局)
- dispatchToken goroutine (本地)
- dispatchToken goroutine (全局)
- Context (本地)
- Context (全局)

### 任务

使用 Mermaid 语法绘制流程图：

```mermaid
graph TD
    FCM[FlowControlManager]
    Task[Task: LimitFlow]
    LocalToken[tokensChan 本地]
    GlobalToken[totalFlowTokenChan 全局]
    LocalDispatch[dispatchToken 本地]
    GlobalDispatch[dispatchToken 全局]
    LocalCtx[Context 本地]
    GlobalCtx[Context 全局]
    
    %% 请完成以下连接关系
    %% 提示：使用箭头表示数据流向
    %% 例如: A -->|发送 token| B
    
    FCM --> Task
    Task --> LocalToken
    
    %% TODO: 添加更多连接
```

**提示**:
- 使用 `-->` 表示数据流
- 使用 `-->|标签|` 添加说明
- 使用 `-.->` 表示控制流

## 练习 2: Bug 流程（并发任务）

绘制两个并发任务的数据流图，展示 bug 触发过程。

```mermaid
graph TD
    FCM[FlowControlManager]
    
    subgraph Task_A[Task A - GMS]
        TaskA[LimitFlow A]
        LocalTokenA[tokensChan A]
        LocalDispatchA[dispatchToken A]
        LocalCtxA[Context A]
    end
    
    subgraph Task_B[Task B - DN]
        TaskB[LimitFlow B]
        LocalTokenB[tokensChan B]
        LocalDispatchB[dispatchToken B]
        LocalCtxB[Context B]
    end
    
    subgraph Global[全局资源]
        GlobalToken[totalFlowTokenChan]
        GlobalDispatch[dispatchToken 全局]
        GlobalCtx[Context 全局]
    end
    
    %% TODO: 添加连接关系
    %% 重点展示：
    %% 1. 两个任务如何共享全局 channel
    %% 2. Context 取消如何传播
    %% 3. Bug 在哪里触发
```

## 练习 3: 时序数据流图

绘制带时序的数据流图，展示 bug 触发的完整过程。

```mermaid
sequenceDiagram
    participant FCM as FlowControlManager
    participant TaskA as Task A (GMS)
    participant TaskB as Task B (DN)
    participant GlobalChan as totalFlowTokenChan
    participant GlobalCtx as Global Context
    
    Note over FCM: T0: 初始化
    FCM->>GlobalChan: 创建 channel
    FCM->>GlobalCtx: 创建 context
    
    Note over TaskA: T1: Task A 启动
    %% TODO: 添加 Task A 的操作
    
    Note over TaskB: T2: Task B 启动
    %% TODO: 添加 Task B 的操作
    
    Note over TaskA,TaskB: T3-T5: 并发传输
    %% TODO: 添加并发传输的交互
    
    Note over TaskA: T6: Task A 完成
    %% TODO: 添加 Task A 清理操作
    
    Note over GlobalCtx: T7: Context 取消
    %% TODO: 添加 Context 取消的影响
    
    Note over TaskB: T8: Task B Panic
    %% TODO: 添加 Panic 触发过程
```

## 练习 4: 状态转换图

绘制 channel 和 context 的状态转换图。

### totalFlowTokenChan 状态转换

```mermaid
stateDiagram-v2
    [*] --> Created: Start()
    Created --> Open: dispatchToken 启动
    Open --> Closed: Context 取消
    Closed --> [*]
    
    %% TODO: 添加更多状态和转换
    %% 考虑：
    %% - 有多少个任务在使用？
    %% - 什么时候应该关闭？
    %% - 关闭后会发生什么？
```

### Context 状态转换

```mermaid
stateDiagram-v2
    [*] --> Active: 创建
    Active --> Cancelled: cancel() 调用
    Cancelled --> [*]
    
    %% TODO: 添加本地和全局 context 的关系
```

## 练习 5: 组件交互图

绘制详细的组件交互图，包含所有关键操作。

```mermaid
graph LR
    subgraph "Task A 生命周期"
        A1[启动] --> A2[获取信号量]
        A2 --> A3[创建本地资源]
        A3 --> A4[启动 dispatchToken]
        A4 --> A5[传输循环]
        A5 --> A6[读取本地 token]
        A6 --> A7[读取全局 token]
        A7 --> A8[传输数据]
        A8 --> A5
        A5 --> A9[完成]
        A9 --> A10[defer 清理]
    end
    
    subgraph "Task B 生命周期"
        B1[启动] --> B2[获取信号量]
        %% TODO: 完成 Task B 的流程
    end
    
    subgraph "全局资源"
        G1[totalFlowTokenChan]
        G2[Global Context]
        G3[Global dispatchToken]
    end
    
    %% TODO: 添加交互关系
    A7 -.->|读取| G1
    B7 -.->|读取| G1
```

## 练习 6: 问题定位图

在数据流图上标注问题点。

使用以下符号：
- 🔴 严重问题
- 🟡 潜在问题
- 🟢 正常操作

```mermaid
graph TD
    Task1[Task 1] -->|读取| GlobalChan[totalFlowTokenChan]
    Task2[Task 2] -->|读取| GlobalChan
    
    GlobalCtx[Global Context] -.->|控制| GlobalDispatch[Global dispatchToken]
    GlobalDispatch -->|发送 token| GlobalChan
    
    Task1 -->|完成| Cleanup1[清理]
    Cleanup1 -.->|可能触发| GlobalCtx
    
    GlobalCtx -.->|取消| GlobalDispatch
    GlobalDispatch -.->|关闭| GlobalChan
    
    GlobalChan -.->|已关闭| Task2
    Task2 -.->|💥| Panic[Panic!]
    
    %% TODO: 在图上标注：
    %% 1. 哪里是 🔴 严重问题？
    %% 2. 哪里是 🟡 潜在问题？
    %% 3. 哪里是 🟢 正常操作？
```

## 练习 7: 对比图

绘制修复前后的对比图。

### 修复前（有 Bug）

```mermaid
graph TD
    %% TODO: 绘制有 bug 的架构
```

### 修复后（方案 A：禁用全局流控）

```mermaid
graph TD
    %% TODO: 绘制修复后的架构
```

### 修复后（方案 B：引用计数）

```mermaid
graph TD
    %% TODO: 绘制使用引用计数的架构
```

## 验证清单

完成练习后，检查你的图表是否：

- [ ] 包含所有关键组件
- [ ] 正确表示数据流向
- [ ] 清晰标注控制流
- [ ] 展示并发关系
- [ ] 标注问题点
- [ ] 易于理解

## 提示

### Mermaid 语法参考

**流程图**:
```mermaid
graph TD
    A[方框] --> B{菱形}
    B -->|是| C[圆角方框]
    B -->|否| D((圆形))
```

**序列图**:
```mermaid
sequenceDiagram
    A->>B: 消息
    B-->>A: 返回
    Note over A,B: 注释
```

**状态图**:
```mermaid
stateDiagram-v2
    [*] --> State1
    State1 --> State2: 转换
    State2 --> [*]
```

## 参考资料

- [Mermaid 文档](https://mermaid.js.org/)
- [HPFS Bug 流程图](../../review/HPFS_BUG_FLOW_DIAGRAM.md)
- [Go Concurrency Patterns](https://go.dev/blog/pipelines)

## 下一步

完成数据流图练习后：
1. 对比你的图表和参考文档
2. 继续模块 2 的练习
3. 开始实现最小复现程序
