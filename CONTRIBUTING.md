# Contributing Guide

感谢你对本项目的关注！欢迎提交 Issue 和 Pull Request。

## 开发环境搭建

```bash
# 1. Fork 并克隆仓库
git clone https://github.com/your-username/deep-learning-classifier.git
cd deep-learning-classifier

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
pip install pytest flake8 black

# 4. 运行测试
pytest tests/ -v
```

## 代码规范

- **Python 版本**: 3.9+
- **代码风格**: 遵循 PEP 8，使用 `black` 格式化
- **行长度**: 最大 120 字符
- **命名规范**: 
  - 函数/变量: `snake_case`
  - 类名: `PascalCase`
  - 常量: `UPPER_SNAKE_CASE`
- **类型注解**: 公共 API 必须添加类型注解
- **文档字符串**: 所有公共函数/类必须有 docstring，包含参数、返回值说明

## 提交 PR 流程

1. **创建分支**: `git checkout -b feature/your-feature-name`
2. **编写代码**: 确保代码符合规范
3. **添加测试**: 新功能必须包含单元测试
4. **运行测试**: `pytest tests/ -v` 确保全部通过
5. **代码检查**: `flake8 src/ --max-line-length=120`
6. **提交代码**: 
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
7. **推送分支**: `git push origin feature/your-feature-name`
8. **创建 PR**: 在 GitHub 上创建 Pull Request

## Commit Message 规范

采用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type 类型**:
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 添加测试
- `chore`: 构建/工具链相关

**示例**:
```
feat(model): add EfficientNet backbone support

- Add EfficientNet-B0/B3 model variants
- Add corresponding config templates
- Add unit tests for new models

Closes #123
```

## 报告 Issue

提交 Issue 时请包含：
- **环境信息**: Python 版本、操作系统、依赖版本
- **复现步骤**: 清晰的复现步骤
- **期望行为**: 你期望的正确结果
- **实际行为**: 实际发生的错误
- **错误日志**: 完整的错误堆栈信息

## 项目结构说明

```
deep-learning-classifier/
├── src/              # 核心代码
├── configs/          # 配置文件
├── scripts/          # 训练/评估脚本
├── app/              # API服务
├── tests/            # 单元测试
├── examples/         # 使用示例
└── .github/          # CI/CD配置
```

如有疑问，欢迎在 Discussion 区提问！
