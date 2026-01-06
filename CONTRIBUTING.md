# 贡献指南

首先，感谢你考虑为智能枪械训练监控系统做出贡献！

## 🌟 贡献方式

我们欢迎以下形式的贡献：

- 🐛 **Bug 报告**：发现问题请提交 Issue
- 💡 **功能建议**：提出新功能想法
- 📝 **文档改进**：完善文档和注释
- 🔧 **代码贡献**：修复 Bug 或添加新功能
- 🌍 **翻译**：帮助翻译文档和界面

## 📋 开发流程

### 1. 准备工作

```bash
# Fork 项目到你的 GitHub 账号

# 克隆你 fork 的仓库
git clone git@github.com:YOUR_USERNAME/gun_train.git
cd gun_train

# 添加上游仓库
git remote add upstream git@github.com:itsoso/gun_train.git

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖（包括开发依赖）
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 如果存在
```

### 2. 创建分支

```bash
# 从 main 分支创建新分支
git checkout -b feature/your-feature-name

# 或者修复 Bug
git checkout -b fix/your-bug-fix
```

分支命名规范：
- `feature/xxx` - 新功能
- `fix/xxx` - Bug 修复
- `docs/xxx` - 文档更新
- `refactor/xxx` - 代码重构
- `test/xxx` - 测试相关

### 3. 开发代码

#### 代码规范

**Python 代码**：
- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 规范
- 使用 `black` 进行代码格式化
- 使用 `flake8` 进行代码检查
- 使用 `mypy` 进行类型检查

```bash
# 格式化代码
black .

# 代码检查
flake8 backend/

# 类型检查
mypy backend/
```

**提交信息**：
- 使用清晰的提交信息
- 建议使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范

```
feat: 添加学员训练进度条
fix: 修复摄像头连接超时问题
docs: 更新部署文档
refactor: 重构视频处理模块
test: 添加动作分析单元测试
```

#### 编写测试

所有新功能和 Bug 修复都应该包含测试：

```bash
# 运行测试
pytest

# 运行特定测试
pytest tests/test_action_analyzer.py

# 查看测试覆盖率
pytest --cov=backend --cov-report=html
```

### 4. 提交代码

```bash
# 添加修改的文件
git add .

# 提交（请使用有意义的提交信息）
git commit -m "feat: 添加XX功能"

# 推送到你的仓库
git push origin feature/your-feature-name
```

### 5. 创建 Pull Request

1. 在 GitHub 上打开你 fork 的仓库
2. 点击 "New Pull Request"
3. 选择你的分支
4. 填写 PR 描述：
   - 解决了什么问题
   - 如何测试
   - 相关 Issue（如果有）
5. 提交 PR

## 📝 Pull Request 要求

### PR 标题格式

```
<type>: <description>

例如：
feat: 添加枪支检测模型
fix: 修复WebSocket连接断开问题
docs: 更新API文档
```

### PR 描述模板

```markdown
## 描述
简要说明这个 PR 做了什么。

## 动机和背景
为什么需要这个改动？解决了什么问题？

## 改动内容
- 改动点1
- 改动点2

## 测试
如何测试这些改动？

## 相关 Issue
Fixes #123
Related to #456

## 截图（如果适用）
添加相关截图

## 检查清单
- [ ] 我的代码遵循项目的代码规范
- [ ] 我已经添加了必要的注释
- [ ] 我已经更新了相关文档
- [ ] 我的改动不会产生新的警告
- [ ] 我已经添加了测试用例
- [ ] 所有测试都通过了
```

## 🐛 Bug 报告

提交 Bug 时，请包含以下信息：

**Bug 描述**
清晰简洁地描述 Bug。

**复现步骤**
1. 执行操作 '...'
2. 点击 '....'
3. 滚动到 '....'
4. 看到错误

**预期行为**
清楚描述你期望发生什么。

**实际行为**
清楚描述实际发生了什么。

**截图**
如果适用，添加截图帮助解释问题。

**环境信息**
- OS: [例如 Ubuntu 20.04]
- Python 版本: [例如 3.9.7]
- 浏览器: [例如 Chrome 95]
- 系统版本: [例如 v1.0.0]

**附加信息**
任何其他相关信息。

## 💡 功能建议

提交功能建议时，请包含：

**问题描述**
当前系统的不足或需要改进的地方。

**建议方案**
你希望如何解决这个问题？

**可选方案**
考虑过的其他解决方案。

**附加信息**
任何其他相关信息、截图、参考链接等。

## 📚 文档贡献

文档改进同样重要！包括：

- 修正拼写错误和语法错误
- 添加缺失的文档
- 改进示例代码
- 翻译文档

文档位置：
- `README.md` - 项目主文档
- `docs/` - 详细文档
- 代码注释 - Python docstrings

## 🔍 代码审查

所有提交都会经过代码审查。审查要点：

- ✅ 代码质量和可读性
- ✅ 是否符合项目规范
- ✅ 是否有充足的测试
- ✅ 是否更新了文档
- ✅ 是否有性能问题
- ✅ 是否有安全隐患

## 🎯 优先级

我们特别欢迎以下贡献：

**高优先级**
- 🐛 Critical Bug 修复
- 🔒 安全漏洞修复
- 📖 文档完善

**中优先级**
- ✨ 新功能开发
- ⚡ 性能优化
- ♻️ 代码重构

**低优先级**
- 💄 UI/UX 改进
- 🎨 代码风格调整

## 💬 沟通渠道

有问题或建议？欢迎通过以下方式联系我们：

- **GitHub Issues**: [提交问题](https://github.com/itsoso/gun_train/issues)
- **GitHub Discussions**: [参与讨论](https://github.com/itsoso/gun_train/discussions)
- **邮件**: support@example.com

## 📜 行为准则

参与本项目，请遵守以下准则：

- 🤝 尊重所有贡献者
- 💬 使用友好和包容的语言
- 🎯 专注于对项目最好的方案
- 🙏 感谢其他人的贡献
- 📚 乐于帮助新手

## 🎉 成为贡献者

一旦你的 PR 被合并，你将：

- 出现在项目贡献者列表中
- 获得项目维护者的感谢
- 为开源社区做出贡献

## 📝 许可协议

通过贡献代码，你同意你的贡献将在 [MIT License](LICENSE) 下发布。

---

再次感谢你的贡献！🙏

