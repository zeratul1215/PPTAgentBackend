# 数据库说明（Database）

本文档用通俗的语言说明 `agent_backend` 的数据库里都存了什么、为什么这么设计，以及本地如何启用。

## 一个总原则

数据库**只存"结构化、需要重启后依然存在"的元数据和索引**；真正的大文件都放在磁盘（或以后的对象存储）上，数据库里只保存一个"指路"的路径。

放在磁盘、**不进数据库**的内容包括：

- 页面 PNG 截图、渲染后的 HTML 包
- 原始上传的 PDF / PPTX / PPT
- gpt-image-2 生成的参考图
- 每页的 state JSON、各 step 的中间产物（`turns/turn_xxxx/` 下的文件）

## 配置方式

通过环境变量 `PPT_DATABASE_URL` 启用，例如：

```
PPT_DATABASE_URL="postgresql://pptagent:pptagent@localhost:5432/pptagent"
```

- **不设置**时：`get_pool()` 返回 `None`，全部功能回退到内存 / 磁盘方案，应用照常运行。
- **设置**时：服务启动（`server/app.py` 的 `lifespan`）会调用 `init_db()` 建表；LangGraph 的 checkpoint 表由 `PostgresSaver.setup()` 单独创建。

## 一、我们自己设计的 5 张表（`schema.sql`）

### 1. `users` — 用户表

谁在用这个系统，是所有权的"根"，其他数据都挂在某个用户下面。

| 字段 | 含义 |
| --- | --- |
| `user_id` | 主键 |
| `display_name` | 显示名 |
| `created_at` | 创建时间 |

### 2. `decks` — PPT 文档表

每一份被上传/处理的 PPT 就是一条记录，存这份 PPT 的"档案信息"。以前靠扫描磁盘上的 `project.json` 来获知有哪些文档，现在直接查这张表。

| 字段 | 含义 |
| --- | --- |
| `project_id` | 主键 |
| `user_id` | 所属用户（外键，级联删除） |
| `title` | 标题 |
| `page_count` | 页数 |
| `page_size_pt` | 页面尺寸（JSONB） |
| `source_kind` | 来源类型：`pdf` / `pptx` / `ppt` … |
| `source_path` | 原始上传文件位置 |
| `workspace_root` | 磁盘工作目录 `result/<project_id>/` |
| `created_at` / `updated_at` | 创建 / 更新时间 |

### 3. `sessions` — 对话会话表

每一个聊天线程一条记录。关键是记住了**这个会话当前正在编辑哪份 PPT**（`active_project_id`），这样用户说"改第 5 页"时系统知道指哪份文档。

| 字段 | 含义 |
| --- | --- |
| `session_id` | 主键 |
| `user_id` | 所属用户（外键，级联删除） |
| `active_project_id` | 当前编辑的文档（外键，文档删除时置空） |
| `title` | 会话标题 |
| `created_at` / `last_active_at` | 创建 / 最后活跃时间 |

### 4. `session_decks` — 会话最近用过的文档

一个会话里可能切换过多份 PPT。按"最后使用时间"排序记录，用来支持"切回上一份文档"这类引用。

| 字段 | 含义 |
| --- | --- |
| `session_id` + `project_id` | 联合主键 |
| `last_used_at` | 最后使用时间（用于排序） |

### 5. `turns` — 单页编辑历史

每调用一次 `edit_pages` 改一页，就记一条元数据，用于"这页改了几次 / 回滚 / 审计"。

| 字段 | 含义 |
| --- | --- |
| `turn_id` | 自增主键 |
| `project_id` | 所属文档（外键，级联删除） |
| `session_id` | 来自哪个会话（外键，会话删除时置空） |
| `page_num` | 第几页（1 起） |
| `demand` | 这一页的独立指令 |
| `ok` | 成功 / 失败 |
| `error` | 错误信息 |
| `turn_dir` | 磁盘上的产物目录 `turns/turn_xxxx/` |
| `created_at` | 创建时间 |

## 二、LangGraph 自动管理的 checkpoint 表

`checkpoints`、`checkpoint_writes` 等表**不是我们建的**，由 `PostgresSaver` 通过自己的 `.setup()` 创建和维护。它们存的是**对话的完整状态快照**（agent 的"记忆"：聊天历史、中间变量）。有了它，进程重启后对话还能接着上一轮继续。

## 一句话总结

数据库存三类东西：**谁（users）拥有哪些 PPT（decks）**、**每个对话正在编辑哪份 PPT 及历史（sessions / session_decks）**、**每一页被怎么改过（turns）**，外加 LangGraph 帮我们存的**对话记忆快照**。所有沉重的文件本体都在磁盘，数据库只存元数据和路径。
