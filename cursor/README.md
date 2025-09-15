# 中国象棋游戏

一个使用Python Flask后端和原生HTML/CSS/JavaScript前端开发的中国象棋游戏。

## 功能特点

- 🎮 完整的中国象棋规则实现
- 🎨 美观的现代化界面设计
- 📱 响应式设计，支持移动端
- 🔄 实时游戏状态更新
- 📝 走棋记录显示
- 🎯 精确的走棋规则检查

## 技术栈

- **后端**: Python Flask
- **前端**: 原生HTML/CSS/JavaScript
- **通信**: RESTful API

## 快速开始

### 方法一：使用启动脚本（推荐）

```bash
python run.py
```

### 方法二：手动启动

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动服务器：
```bash
python app.py
```

3. 打开浏览器访问：http://localhost:5000

## 游戏规则

### 棋子说明
- **帅/将**: 只能在九宫格内移动，每次只能移动一格
- **仕/士**: 只能在九宫格内斜向移动一格
- **相/象**: 不能过河，斜向移动两格，不能越子
- **马**: 走日字，不能蹩马腿
- **车**: 直线移动，不能越子
- **炮**: 直线移动，吃子时需要翻山
- **兵/卒**: 未过河只能向前，过河后可左右移动

### 操作说明
1. 点击己方棋子进行选择
2. 点击目标位置进行移动
3. 按ESC键取消选择
4. 点击"重新开始"按钮重置游戏

## 项目结构

```
cursor/
├── app.py              # Flask后端服务器
├── requirements.txt    # Python依赖包
├── run.py             # 启动脚本
├── README.md          # 项目说明
└── templates/
    └── index.html     # 前端页面
```

## API接口

- `GET /` - 游戏主页面
- `GET /api/board` - 获取棋盘状态
- `POST /api/move` - 执行移动
- `POST /api/reset` - 重置游戏

## 开发说明

### 后端架构
- 使用面向对象设计，`ChineseChess`类管理游戏状态
- 完整的走棋规则验证
- RESTful API设计

### 前端特性
- 响应式CSS Grid布局
- 现代化UI设计
- 实时游戏状态同步
- 移动端适配

## 浏览器兼容性

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## 许可证

MIT License
