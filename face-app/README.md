# 面相识别 / 解读网页应用

上传一张正面人脸照片 → 提取面部关键点 → 计算三庭五眼、脸型、五官比例与画面气色观感 →  
用这些特征到**面相文化知识库**里做 RAG 检索 → 交给大模型生成一份个性化解读 → 存入历史记录。

> ⚠️ **产品定位**：传统文化趣味参考。不是算命，不承诺命运结果，不构成健康、婚姻、投资建议。  
> 代码里内置了合规约束（Prompt 规则 + 关键词过滤 + 免责声明强制追加）。

---

## 一、技术选型（本项目实际使用的默认方案）

| 层       | 选型                                             | 说明                                                  |
| ------- | ---------------------------------------------- | --------------------------------------------------- |
| 前端      | HTML + CSS + 原生 JS（4 个页面 + 3 个合规文档页）           | 移动端友好，照片不做本地持久化                                     |
| 后端      | Python 3.10+ / FastAPI                         | RESTful，自带 `/docs` 交互文档                             |
| 数据库     | 开发 SQLite，生产 MySQL / PostgreSQL                | 改 `.env` 里的 `DATABASE_URL` 即可切换                     |
| 人脸检测    | **MediaPipe FaceMesh（本地，默认）**                  | 478 关键点；Tasks/旧 API 自适应，未装时降级 OpenCV Haar + 比例推算    |
| 面相解读    | **豆包（火山方舟）/ MiniMax（Anthropic 协议）/ OpenAI 兼容** | 换 `.env` 里两行即可切换；无密钥时用本地模板兜底                        |
| 知识库 RAG | 切片 + Embedding + 向量检索                          | 默认纯 Python 哈希向量，零依赖；可换 bge / 云 Embedding / ChromaDB |
| 部署      | Docker Compose + Nginx + HTTPS                 | 附带一键部署脚本                                            |

三种能力都是**可插拔**的：换后端只改 `.env` 与 `services/` 下对应文件，业务代码不动。

---

## 二、目录结构

```
face-app/
├── frontend/                     # 前端（浏览器里跑）
│   ├── index.html                # 首页：上传/拍照 + 授权确认弹窗
│   ├── result.html               # 结果页：特征 + 解读 + 知识来源
│   ├── history.html              # 历史记录页
│   ├── login.html                # 登录/注册/游客体验
│   ├── privacy.html              # 隐私政策（合规三件套之一）
│   ├── terms.html                # 用户协议
│   ├── authorization.html        # 人脸信息处理授权书
│   ├── css/style.css
│   └── js/{api,app,result,history,login}.js
├── backend/                      # 后端（服务器上跑）
│   ├── main.py                   # FastAPI 入口：路由、CORS、限流、静态前端、启动初始化
│   ├── config.py                 # 全部配置（读 .env，含安全默认值）
│   ├── routers/                  # auth / face / history / kb / deps
│   ├── services/
│   │   ├── face_detector.py      # 人脸检测（MediaPipe → OpenCV → 云 API 预留）
│   │   ├── feature_extractor.py  # 三庭五眼 / 五官 / 脸型 / 气色 / 对称度
│   │   ├── rag.py                # ★ 知识库：切片 / Embedding / 入库 / 检索
│   │   ├── interpreter.py        # ★ 解读：豆包 / OpenAI 兼容 / 本地模板
│   │   ├── compliance.py         # 合规过滤 + 免责声明
│   │   └── security.py           # bcrypt 密码、JWT、签名 URL
│   ├── models/db.py              # SQLAlchemy 表：users / face_records / kb_documents / kb_chunks
│   ├── schemas/models.py         # Pydantic 请求响应结构
│   ├── knowledge_base/
│   │   ├── docs/默认相学常识.md   # 内置兜底知识（空库时自动导入）
│   │   └── README.md             # ★ 怎么往里加你自己的知识（含 PDF 古籍路线）
│   ├── raw_sources/              # ★ 原始资料投放目录（PDF/TXT/MD）
│   │   └── extracted/            # 从 PDF 抽出的原文（中间产物）
│   ├── scripts/
│   │   ├── auto_build_kb.py      # ★ PDF（含扫描版）→ 大模型整理 → 入库，一条命令
│   │   ├── ingest_local.py       # 已整理好的 md/txt 批量导入
│   │   ├── check_detector.py     # 引擎体检：看装了哪个引擎、这张图能不能检出
│   │   ├── check_llm.py          # 大模型体检：密钥/地址/模型名是否正确
│   │   ├── make_test_face.py     # 生成合成测试人脸（自测用，不涉及真人照片）
│   │   ├── make_test_pdfs.py     # 生成数字版 + 繁体竖排扫描版测试 PDF
│   │   └── selftest.py           # 端到端自测脚本
│   ├── requirements.txt
│   └── .env.example
├── docker/{Dockerfile,docker-compose.yml}
├── deploy/{nginx.conf,deploy.sh}
└── README.md
```

---

## 三、本地运行（5 分钟跑通）

### 1. 装依赖

```bash
cd face-app
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r backend/requirements.txt
```

> **关于 MediaPipe 版本**：MediaPipe 1.x 只提供 Tasks API（旧的 `mp.solutions` 已移除），  
> 首次分析会自动下载 3.6MB 的 `face_landmarker.task` 模型到 `backend/storage/models/`。  
> 离线环境请手动放置该模型，或设 `FACE_LANDMARKER_AUTO_DOWNLOAD=false` 让它走 OpenCV 兜底。  
> 另外 `mediapipe` 会依赖 `opencv-contrib-python`；若它把 cv2 升到 5.x，Haar 兜底引擎会失效  
> （OpenCV 5 移除了 `CascadeClassifier`），此时 MediaPipe 正常工作即可。  
> 用 `python scripts/check_detector.py` 可以随时确认当前哪个引擎在生效。

### 2. 配环境变量

```bash
cd backend
cp .env.example .env      # Windows: copy .env.example .env
```

`.env` 里**唯一需要填的是大模型密钥**（不填也能跑，解读会退化为本地模板）。

**方案 A：豆包（火山方舟，OpenAI 协议）**

```ini
LLM_PROVIDER=doubao
ARK_API_KEY=你的火山方舟密钥
ARK_MODEL=doubao-seed-1-6-250615
```

**方案 B：MiniMax（Anthropic 协议）**

```ini
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-api-xxxxxxxx        # 控制台「账户信息」里生成
ANTHROPIC_BASE_URL=https://api.minimax.cn/anthropic
ANTHROPIC_MODEL=MiniMax-M3.1-Flash-Preview   # M2 / M2.1 / Text-01 等均可用
```

**方案 C：任意 OpenAI 兼容接口**（DeepSeek / Kimi / 通义 / 自建 vLLM）

```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

填完先体检一次，确认密钥、地址、模型名都对：

```bash
python scripts/check_llm.py                                  # 用 .env 的配置发一次最小请求
python scripts/check_llm.py --models MiniMax-M2 MiniMax-Text-01   # 不知道有哪些模型时可逐个试探
```

成功会打印模型返回的一段话；失败会打印具体错误（401 密钥错 / 404 模型名错 / 超时等）。  
最后看 `/api/health` 的 `llm` 段，或分析结果里的 `llm_provider` 字段（应为 `anthropic`/`doubao`/`openai`；  
若是 `template(fallback:...)` 说明调用失败并已降级，后端日志里有原因）。

### 3. 启动后端

```bash
cd backend
uvicorn main:app --reload --port 8000
```

启动后：

- 接口文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/api/health>
- 前端页面：<http://localhost:8000/>（后端已挂载 `frontend/`，一个端口全搞定）

首次启动会自动建表，并在知识库为空时导入 `knowledge_base/docs/` 里的默认相学常识。

### 4. 只跑前端（可选）

想把前端放到别的端口/静态服务器也可以：

```bash
python -m http.server 8080 --directory frontend
```

此时把 `.env` 的 `CORS_ORIGINS` 加上 `http://localhost:8080`（默认已含）。

---

## 四、运行自测清单（逐项验证每个功能）

### 方式 A：一键脚本（推荐）

```bash
cd backend
python scripts/check_detector.py        # 0) 体检：确认引擎与依赖是否就绪
python scripts/make_test_face.py        # 1) 生成合成测试人脸
python scripts/selftest.py --verbose    # 2) 后端已在 8000 端口运行时执行
```

脚本会依次验证并打印 PASS/FAIL：  
`健康检查 → 游客登录 → 未登录被拒 401 → 上传分析 → 特征合理性 → 解读长度与合规 → RAG 命中 →
历史列表 → 记录详情 → 知识库检索接口 → 签名 URL 取图 → 伪造票据被拒 403 → 删除记录 → 删除后 404 → 跨账号隔离`。

### 方式 B：手工逐项验证

| #  | 验证项     | 操作                                              | 期望结果                                                        |
| -- | ------- | ----------------------------------------------- | ----------------------------------------------------------- |
| 1  | 健康检查    | 浏览器打开 `/api/health`                             | `{"status":"ok",...}`，`engines` 里至少一个为 true，`kb.chunks > 0` |
| 2  | 接口文档    | 打开 `/docs`                                      | 能看到 auth / face / history / kb 四组接口                         |
| 3  | 注册登录    | 前端 `/login.html` 注册或点「游客一键体验」                   | 跳转首页，导航栏显示用户名                                               |
| 4  | 上传分析    | `/index.html` 选一张正面照，勾选授权 → 开始分析                | 有进度条与「正在检测…」动画，几秒后跳结果页                                      |
| 5  | 非人脸图片被拒 | 上传风景照/纯色图                                       | 提示「没有检测到清晰的人脸」                                              |
| 6  | 格式/大小校验 | 上传 .gif 或 >10MB 的图                              | 提示仅支持 jpg/png/webp 或超过 10MB                                 |
| 7  | 结果页内容   | 查看结果页                                           | 有照片、脸型、三庭比例条、五官标签、气色、解读正文、免责声明                              |
| 8  | 知识来源可解释 | 结果页底部                                           | 展示引用的知识片段标题、相似度、摘要与检索词                                      |
| 9  | 历史记录    | 打开 `/history.html`                              | 列出记录，可「查看解读」/「删除」                                           |
| 10 | 删除权     | 删除一条记录                                          | 记录消失；后端 `storage/images/` 下对应照片同时被删                         |
| 11 | 权限隔离    | 换一个游客账号访问别人的 `result.html?id=1`                 | 提示记录不存在（404）                                                |
| 12 | 私有照片    | 直接访问 `/api/face/image/1`（无 token）               | 403；带 analyze 返回的签名 token 才能看到                              |
| 13 | 知识库导入   | `python scripts/ingest_local.py --path ../我的资料` | 打印导入文档数/切片数；再跑 `--list` 看统计                                 |
| 14 | 检索调试    | `POST /api/kb/query {"query":"三庭"}`             | 返回命中的知识片段与相似度                                               |
| 15 | 解读来源    | 看记录里的 `llm_provider`                            | 填了密钥为 `doubao`；没填为 `template`                               |
| 16 | 合规文案    | 检查解读文本                                          | 无「改运/必发财/克夫/大凶」等词；结尾有免责声明                                   |
| 17 | 密码安全    | 直接看数据库                                          | `users.password_hash` 是 bcrypt 串（`$2b$...`），不是明文            |
| 18 | 限流      | 快速连发 40 次分析请求                                   | 超过阈值返回 429（默认 30 次/分钟，可配置）                                  |

---

## 五、API 一览

| 方法     | 路径                            | 作用                        | 鉴权                       |
| ------ | ----------------------------- | ------------------------- | ------------------------ |
| POST   | `/api/auth/register`          | 注册                        | —                        |
| POST   | `/api/auth/login`             | 登录                        | —                        |
| POST   | `/api/auth/guest`             | 游客一键体验                    | —                        |
| GET    | `/api/auth/me`                | 当前用户                      | Bearer                   |
| POST   | `/api/face/analyze`           | 面相分析（multipart 上传 `file`） | Bearer                   |
| GET    | `/api/face/image/{id}?token=` | 读取私有照片（签名 URL）            | 签名票据                     |
| GET    | `/api/history`                | 我的历史记录（`limit`/`offset`）  | Bearer                   |
| GET    | `/api/history/{id}`           | 单条详情（特征 + 解读 + 引用片段）      | Bearer                   |
| DELETE | `/api/history/{id}`           | 删除记录及照片                   | Bearer                   |
| POST   | `/api/kb/ingest`              | 导入知识文档（文件或目录）             | Bearer + 可选 `X-KB-Token` |
| POST   | `/api/kb/query`               | 测试检索                      | Bearer                   |
| GET    | `/api/kb/stats`               | 知识库统计                     | Bearer                   |
| GET    | `/api/health`                 | 健康检查                      | —                        |

命令行示例：

```bash
# 1) 拿 token
curl -X POST http://localhost:8000/api/auth/guest

# 2) 分析（把 <TOKEN> 换成上一步的 access_token）
curl -X POST http://localhost:8000/api/face/analyze \
     -H "Authorization: Bearer <TOKEN>" \
     -F "file=@backend/storage/testdata/test_face.jpg"
```

---

## 六、自建知识库怎么接（重点）

详细步骤见 [`backend/knowledge_base/README.md`](backend/knowledge_base/README.md)。分两条路：

**路线 A：手上是整理好的 `.md/.txt`**（最快）

1. 放进 `backend/knowledge_base/docs/`；
2. 导入：`python scripts/ingest_local.py`，或 `POST /api/kb/ingest` 上传文件。

**路线 B：手上是 PDF 古籍（含扫描影印件）**（全自动）

1. 把 PDF 丢进 `backend/raw_sources/`；
2. `pip install pymupdf rapidocr-onnxruntime`（首次）；
3. `python scripts/auto_build_kb.py --all` —— 自动完成：
   - 数字版页面用 **PyMuPDF** 抽文字层；扫描页按 300dpi 栅格化后用**中文 OCR**识别；  
     繁体**竖排**按「列从右到左、列内从上到下」重排阅读顺序；
   - 抽出的原文交给大模型整理：中性现代中文、按 `# 三庭/五眼/五官/气色/脸型/十二宫 + ## 小节`  
     组织、保留三庭五眼/印堂/气色/脸型等关键术语、把「必发财/克夫/大凶/寿元不长」等  
     绝对化与恐吓性表述改写成「传统相学认为……供参考」；
   - 结果写入 `knowledge_base/docs/*.md` 并自动入库。
   没有真古籍也能先验证：`python scripts/make_test_pdfs.py` 生成两份测试 PDF（数字版 + 繁体竖排扫描版）。

想换更强的向量模型：`.env` 里把 `EMBEDDING_PROVIDER` 改成 `bge`（本地 bge-small-zh）或 `openai`（云 Embedding）。

代码里的接入点集中在 `backend/services/rag.py`：`chunk_document()` 负责切片、  
`get_embedder()` 负责向量化、`search()` 负责检索。换向量库也只需改 `search()`，调用方无感。

---

## 七、部署上线

```bash
# 云服务器（Ubuntu 22.04+，1核2G 起步）
sudo bash deploy/deploy.sh your-domain.com
```

脚本会自动：装 Docker → 生成含随机密钥的 `.env` → 写入域名到 Nginx 配置 → 起  
`backend + MySQL + Redis + Nginx` → 健康检查 → 申请 Let's Encrypt 免费证书。

上线后必查：

```bash
curl -fsS https://your-domain.com/api/health       # 期望 {"status":"ok",...}
docker compose -f docker/docker-compose.yml logs -f backend
```

**上线前检查表**

- [ ] `.env` 已填 `ARK_API_KEY`，并已改掉 `JWT_SECRET` / `SIGNED_URL_SECRET`
- [ ] `.env` 已加入 `.gitignore`，`git status` 里看不到密钥
- [ ] MySQL 已设强密码（脚本自动生成）
- [ ] HTTPS 证书已签发，并加了续期定时任务
- [ ] 照片走对象存储私有桶（OSS/COS）+ 签名 URL，数据库只存 key
- [ ] 设置了 `KB_INGEST_TOKEN`，防止任何人往知识库里塞内容
- [ ] `/docs` 已关闭或加访问控制
- [ ] 隐私政策 / 用户协议 / 数据授权书三页已补上运营主体与联系方式
- [ ] 国内域名+服务器已完成 ICP 备案

---

## 八、安全与合规（已实现的部分）

- **上传前授权**：首页勾选协议 + 二次确认弹窗，才允许上传；
- **上传校验**：格式（jpg/png/webp）、魔数、大小（≤10MB）、分辨率、必须含人脸，否则 422；
- **照片私有化**：存私有目录，读取需带 HMAC 签名票据（默认 15 分钟过期），伪造票据 403；
- **删除权**：`DELETE /api/history/{id}` 同时删除数据库记录与照片文件；
- **密码安全**：bcrypt 哈希（环境缺 bcrypt 时降级 PBKDF2），登录失败不区分「用户不存在/密码错」；
- **Token**：JWT + 7 天过期，前端过期自动跳登录；
- **数据隔离**：所有历史接口校验记录归属，跨账号访问统一 404；
- **文案合规**：Prompt 规则 + `services/compliance.py` 关键词替换 + 免责声明强制追加；
- **限流**：写接口默认 30 次/分钟/IP（进程内实现，上量换 Redis）。

---

## 九、扩展路线（架构已预留，无需重写）

- **缓存**：`REDIS_URL` 已预留；热门解读结果可缓存 `(image_sha256 + 版本号)`；
- **异步化**：AI 解读是算力大头，可把 `analyze` 拆成「返回 record_id + 后台任务」两段，前端轮询；
- **多实例**：后端无状态（除本地照片目录），把照片换对象存储后可直接水平扩容，Nginx 上游轮询；
- **向量库升级**：`VECTOR_STORE=chroma` 或换 pgvector，只改 `rag.search()`；
- **监控**：`/api/health` 已可被探针使用，可直接接云监控 / Prometheus；


- **云 API 切换**：`FACE_DETECTOR=cloud` 时只需在 `face_detector._detect_cloud()` 里补厂商 SDK 调用。

---

## 十、常见问题

**Q：没填大模型密钥能用吗？**  
能。解读由本地模板基于「特征 + 知识片段」拼装，结构、合规、RAG 引用都在，只是文采不如大模型。

**Q：我的资料是扫描版古籍 PDF，能用吗？**  
能，全自动。放进 `backend/raw_sources/`，装 `pymupdf` + `rapidocr-onnxruntime`，  
然后 `python scripts/auto_build_kb.py --all`：扫描页自动 OCR（繁体竖排会做阅读顺序重排），  
再由大模型整理成中性 Markdown 并入库。识别不准时优先把 `--dpi` 提到 400。  
详见 [knowledge\_base/README.md](face-app/backend/knowledge_base/README.md) 的「零、手上只有 PDF」。

**Q：MiniMax 的密钥和地址填在哪？**`backend/.env` 里填这三行（`LLM_PROVIDER=anthropic`、`ANTHROPIC_API_KEY=sk-api-...`、  
`ANTHROPIC_BASE_URL=https://api.minimax.cn/anthropic`），模型名默认 `MiniMax-M3.1-Flash-Preview`。  
MiniMax 提供的是 **Anthropic 协议**接口（`/v1/messages`），所以走 `anthropic` 这一路，  
不能填在 `OPENAI_BASE_URL` 里——OpenAI 面走的是 `/chat/completions`，路径不同。  
填完用 `python scripts/check_llm.py` 验证。

**Q：密钥会不会被提交到 Git？**  
`backend/.env` 在 `.gitignore` 里，不会进版本库。但**密钥一旦出现在聊天/截图/工单里就应视为泄露**，  
建议到控制台轮换一次。上传密钥时也别贴进前端代码或 `frontend/` 下的任何文件。

**Q：没装 mediapipe 会怎样？**  
自动降级 OpenCV Haar：能检测人脸外接框，再按人类学比例推算关键点。结果里会标 `estimated: true`，  
结果页给出精度提示，不会假装精确。

**Q：上传自己的照片安全吗？**  
默认 MediaPipe 全部在本地/自有服务器完成，照片不出机器；只有你填了云端大模型密钥时，  
才会把**特征与知识片段**（不含照片本身）发给模型生成文案。

**Q：怎么彻底关掉人脸数据留存？**  
分析完成后不落库即可：把 `routers/face.py` 里创建 `FaceRecord` 的那段删掉/改成不提交，  
并让接口直接返回结果（照片也只放在内存）。保留删除接口，用户仍可随时清除历史。

**Q：Windows 下 `pip install` 报 `PermissionError: [Errno 13]`（某些受限/沙箱环境）？**  
极少数受限环境会给工作区打低完整性标签，而 `tempfile.mkdtemp()` 创建的目录权限过严。  
仓库里带了 `.pysite/sitecustomize.py` 补丁，安装时加一句 `set PYTHONPATH=.pysite` 即可绕过（详见文件顶部注释）。

---

## 十一、许可与责任

本项目为技术实现示例。面相解读内容属于传统文化趣味参考，请勿据以做出任何重大决策。  
正式对外提供服务前，请按当地法律法规（如《个人信息保护法》《数据安全法》）完成合规评估与备案。
