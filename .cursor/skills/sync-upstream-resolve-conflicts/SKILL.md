---
name: sync-upstream-resolve-conflicts
description: Syncs code from upstream repository (chenyme/grok2api), resolves merge conflicts; on functional conflicts asks the user which parts to keep (ours/theirs/specific). Optionally adds upstream as a git remote. Use when the user asks to sync upstream, pull from upstream, merge upstream, resolve upstream conflicts, or update from chenyme/grok2api.
---

# 同步上游仓库并解决冲突

本技能用于从上游仓库同步代码并在出现冲突时按用户意愿解决，可选将上游纳入 git remote 管理。

**上游仓库**：`https://github.com/chenyme/grok2api.git`

## 前置检查

- 当前仓库根目录为 grok2api 项目（或用户明确指定了项目路径）。
- 工作区干净或用户同意在未提交更改时暂存/储藏（stash）后再同步。

## 工作流程

### 1. 是否添加 upstream 为 remote

- 执行 `git remote -v`，若尚未存在指向 `github.com/chenyme/grok2api` 的 remote：
  - 询问用户：「是否将上游添加为 remote（例如命名为 `upstream`）？」
  - 若用户同意，执行：`git remote add upstream https://github.com/chenyme/grok2api.git`
- 若已存在，记下其名称（如 `upstream` 或 `origin`），后续用该名称 fetch。

### 2. 拉取上游并合并

```bash
git fetch upstream   # 或用户已有的上游 remote 名称
git merge upstream/master
```

若当前分支不是 `master`，可改为 `upstream/main` 或用户指定的上游分支，与用户确认后再执行。

### 3. 无冲突时

- 合并完成后告知用户同步成功，并提示可执行 `git push`（若需要）。

### 4. 有冲突时：区分并处理

- 运行 `git status` 查看冲突文件列表。
- **非功能性冲突**（仅空白、格式、注释等）：可直接按项目风格选「ours」或「theirs」或简单二选一后标记已解决。
- **功能性冲突**（影响行为、接口、配置逻辑等）：
  - 列出每个冲突文件及冲突片段所在位置（文件名 + 行号或区间）。
  - 对每个功能性冲突说明：
    - **ours**：当前分支的改动；
    - **theirs**：上游的改动。
  - **询问用户**：「在 [文件名] 的 [简要描述] 处，希望保留哪一侧？可选：ours（当前）、theirs（上游）、或说明要保留的具体逻辑（例如保留某函数、某段配置）。」
  - 按用户选择在文件中保留对应内容，删除冲突标记，然后 `git add <文件>`。

### 5. 完成解决

- 所有冲突解决后执行：`git add`（若还有未 add 的冲突文件）→ `git status` 确认无未合并路径 → 询问用户是否立即提交。
- 若用户同意提交，使用一条简洁的提交信息，例如：`Merge upstream, resolve conflicts (keep ours/theirs as requested)` 或用户指定的信息。

## 可选：仅添加 remote 不合并

若用户仅希望「把上游纳入 git remote 管理」、暂不合并：

- 执行 `git remote add upstream https://github.com/chenyme/grok2api.git`（若尚未添加）。
- 告知用户已添加，之后可用 `git fetch upstream` 与 `git merge upstream/<分支>` 自行同步。

## 注意事项

- 不强制覆盖用户未明确选择保留的本地修改；功能性冲突必须经用户确认再保留 ours/theirs 或具体片段。
- 若上游默认分支为 `main` 而非 `master`，以 `git branch -r` 或上游仓库实际分支名为准，并在合并前与用户确认分支名。
