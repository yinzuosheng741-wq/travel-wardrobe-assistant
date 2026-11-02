# 出行衣橱助手

面向 Windows 的本地优先桌面助手，整合衣物管理、库存搭配、旅行准备和服饰知识问答。

![衣橱界面](docs/assets/wardrobe.png)

## 功能

- **管理衣橱**：手动录入或照片识别，搜索、筛选、编辑和删除衣物。
- **搭配与出行**：结合真实库存、个人偏好、季节和可选天气生成搭配及行李建议。
- **洗护与资料问答**：查询带来源及适用条件的服饰知识，支持当前会话上传 PDF、TXT、Markdown。
- **多任务助手**：通过 LangGraph 协调衣橱管理、穿搭出行和衣物知识能力；衣橱变更先预览，确认后写入。
- **本地记忆与备份**：保存偏好、会话和长期记忆，支持记忆查看与删除、衣橱导出及完整数据备份。

![助手界面](docs/assets/chat.png)

## Windows 客户端

从 [最新 Release](https://github.com/yinzuosheng741-wq/travel-wardrobe-assistant/releases/latest) 下载 `TravelWardrobeAssistant-Windows-Portable.zip`，解压后运行 **出行衣橱助手.exe**。无需安装 Python，公开下载无需登录 GitHub；Release 附件不受 Actions 临时产物保留期限制。

请保留完整目录，不要只复制 exe；程序需要放在可写位置。便携包包含 Python、Qt 和运行依赖。首次启动会初始化演示衣物，已有衣橱不会被覆盖。

每个版本同时提供 `.zip.sha256` 校验文件，可用 PowerShell 的 `Get-FileHash -Algorithm SHA256` 核对下载包。历史版本见 [Releases](https://github.com/yinzuosheng741-wq/travel-wardrobe-assistant/releases)。

## 从源码运行

推荐 Windows x64、Python 3.12，在 PowerShell 中执行：

```powershell
git clone https://github.com/yinzuosheng741-wq/travel-wardrobe-assistant.git
cd travel-wardrobe-assistant
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -c portable-lock.txt
.venv\Scripts\python.exe desktop.py
```

也可双击 `Start.vbs` 隐藏控制台启动，或用 `Start.cmd` 查看启动信息。

## 配置服务

首次启动可配置服务，也可选择“暂不配置”使用离线衣橱。之后从“设置 → 服务状态 → 配置模型、天气与定位”修改。

| 能力 | 配置要求 |
| --- | --- |
| AI 对话 | 支持工具调用的 OpenAI 兼容接口、API Key 和模型名称 |
| 图片识别 | 支持图片输入的视觉模型 |
| 语义检索 | Embedding 接口和向量模型；未配置时使用本地 BM25 检索 |
| 天气 | 高德 Web 服务 Key，或可选 Open-Meteo；IP 城市定位默认关闭 |
| 长期记忆 | 默认本地 SQLite，无需云记忆账号，可关闭 |

图片、对话和向量模型名称应分别填写。支持导入已有 `.env`，公开配置示例为 [.env.example](.env.example)。Open-Meteo 免费接口限非商业使用，详见[服务条款](https://open-meteo.com/en/pricing)。

![服务配置](docs/assets/service-setup.png)

## 数据与备份

衣橱、照片、会话和记忆保存在程序目录的 `data/` 中。服务配置位于 `data/config/services.env`，密钥未加密，请勿分享使用过的个人目录。

本地记忆在本机存储和检索；使用远程对话模型时，召回的少量记忆会作为本轮上下文发送给该模型。照片识别、语义检索和天气也会按所选服务发送相应输入。

升级前退出客户端并备份 `data/`。完整备份包含衣橱、关联照片、会话资料和存在的长期记忆，不包含服务密钥、窗口状态或向量缓存。恢复方法见[客户端指南](docs/guides/DESKTOP.md)，服务配置见[配置说明](docs/guides/SETUP.md)。

## 自行打包

```powershell
.venv\Scripts\python.exe -m pip install pyinstaller==6.11.1
.venv\Scripts\python.exe scripts/build_desktop.py
```

生成 `dist/native/出行衣橱助手/` 和 `dist/TravelWardrobeAssistant-Windows-Portable.zip`。若旧输出包含用户数据库或配置，脚本会停止覆盖。请分发干净公开包，保留个人目录作为自用版本。

## 使用边界与许可

搭配基于库存和规则，不保证符合所有审美需求；洗护建议需核对整件衣物洗标与品牌说明。缺少材质或证据时，助手会要求补充信息。出行能力聚焦衣物、天气和行李，不提供订票、酒店预订或路线导航。当前为单用户客户端，不提供自动更新和云同步。

项目代码采用 [MIT 许可证](LICENSE)，第三方组件和演示素材见[许可说明](THIRD_PARTY_NOTICES.md)与[素材来源](assets/demo/SOURCE.md)。反馈问题时不要附带密钥或个人数据，安全问题见[安全说明](SECURITY.md)。
