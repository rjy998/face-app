# 知识库接入指南（自建面相知识库）

这份文档回答一件事：**我有一堆面相相关资料，怎么让 AI 的解读建立在它们之上？**

整个知识库的代码集中在 `backend/services/rag.py`，你不需要改动接口、路由或前端。

---

## 零、手上只有 PDF（尤其是扫描版古籍）

把 PDF 丢进 `backend/raw_sources/`，一条命令跑完全程：**抽文字 → 大模型整理 → 入库**。

```bash
cd backend
pip install pymupdf rapidocr-onnxruntime     # 一次性：PDF 解析 + 中文 OCR
python scripts/auto_build_kb.py --all        # 抽取 + 整理 + 入库
```

它会做这些事：

| 页面类型 | 处理方式 |
|---|---|
| 数字版（能选中文字） | PyMuPDF 直接抽文字层，逐页判断、速度极快 |
| 扫描版（整页是图） | 按 `--dpi`（默认 **300**）栅格化后交给中文 OCR，逐页识别 |
| 混合版 | 同一份 PDF 里两种页混着也没问题，逐页自动分流 |
| 繁体竖排 | OCR 后按「**列从右到左、列内从上到下**」重排阅读顺序（横排则按行左到右） |

抽出来的原文写在 `raw_sources/extracted/*.txt`（**中间产物，不要直接入库**——还带着
「必发财/克夫/大凶」这类原文表述）。第二步会把它交给大模型整理成：

```markdown
# 三庭
## 三庭划分
传统相学将面部纵向分为三段：**上庭**自发际至眉……
## 三庭的传统说法
- **三庭均等**：传统相学认为……，仅供文化参考。
```

要求都已写进整理 Prompt：中性现代中文、按 `# 主题（三庭/五眼/五官/气色/脸型/十二宫）+ ## 小节`
组织、**保留三庭五眼/印堂/气色/脸型等关键术语**、绝对化与恐吓性表述改写成
「传统相学认为……供参考」、繁体与 OCR 错字按上下文还原、不要写编辑说明、不要占位小节。

### 内容口径（已确认，改 Prompt 时请保持）

| 主题 | 口径 |
|---|---|
| 部位限定 | **必须保留**（`耳根部有痣` 不能写成 `有痣`，`左耳小` 不能写成 `耳小`）——限定条件是结论成立的前提 |
| 健康类 | 换说法不删除：「耳根部有痣有阑尾炎」→「耳根部有痣者，传统说法认为可留意消化系统的日常」；去掉具体病名 |
| 六亲寿夭 | 换说法不删除：「左耳小主父先亡」→「传统说法提示可多留意父亲的身体健康」 |
| 婚姻家庭 | 换说法：「克夫」→「传统说法认为夫妻关系需多留意」 |
| 生育类 | **不预测性别**，只留「传统说法中是与生育情况相关的观察角度，无科学依据」 |
| 小节标题 | 必须中性，不出现生死/吉凶/病症字眼（`父母寿元观察` → `父母的健康观察角度`） |
| 解读输出 | 只给用户【结论】+【注意事项】，正文不写处理过程、不逐句声明免责；免责声明由系统在末尾统一追加一行（`COMPLIANCE_DISCLAIMER_ENABLED=false` 可关闭） |
| 吉凶口径 | 好的地方在【结论】里讲；**不足之处一律改写成建议**（「宜…」「建议…」），不写「格局稍欠」「易显优柔」这类定性贬评 |
| 部位范围 | **只对照片能测到的部位下判断**：耳朵形态、痣、法令纹、牙齿等无法从人脸关键点得出，相关资料条目会被过滤（`KB_DROP_UNMEASURABLE=false` 可关闭），解读里也不提 |
| 比例测量 | 上庭起点用**像素检测的发际线**（关键点里没有发际线，索引 10 只是"模型看到的额头最高处"，被刘海遮挡会偏小）；上/中庭分界取**眉线**；检测不到或额头被遮住时如实标注"仅供参考"，不假装精确 |

生成物写到 `knowledge_base/docs/*.md`，然后自动入库（与 `ingest_local.py` 同一代码路径）。
中途可以随时停下来看结果：

```bash
python scripts/auto_build_kb.py --extract                 # 只抽文字，先看 OCR 质量
python scripts/auto_build_kb.py --distill --dry-run       # 只打印将发给大模型的 Prompt
python scripts/auto_build_kb.py --distill                 # 整理（抽好的 txt 可反复重跑）
python scripts/auto_build_kb.py --ingest --probe          # 入库并抽查检索命中
python scripts/auto_build_kb.py --list                    # 看各目录现状
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--dpi 400` | 竖排小字识别不准时提高分辨率（**这是最有效的调节项**） |
| `--pages 1-30,45` | 先只处理前几十页，确认效果再全量跑 |
| `--ocr-engine rapidocr` | 指定 OCR 引擎（`auto` 默认按 paddleocr → rapidocr 挑） |
| `--force-ocr` | 忽略文字层、整本都走 OCR（用于文字层是乱码的烂扫描） |
| `--chunk-chars 3000` | 每次交给大模型的原文长度 |
| `--compliance-review` | 整理后再让大模型做一次合规复审（更保险，代价是多一次调用） |
| `--force` | 覆盖已存在的产物（默认跳过，可反复跑不怕浪费） |

**没有真古籍也能验证流程**：`python scripts/make_test_pdfs.py` 会生成一份数字版和一份
繁体竖排扫描版测试 PDF（内容里故意混了「必发财/克夫/大凶/寿元不长」），跑完 `--all`
可以直观看到中性化效果，确认无误后再删掉它们。

### OCR 引擎怎么选

| 引擎 | 安装 | 说明 |
|---|---|---|
| **RapidOCR**（推荐） | `pip install rapidocr-onnxruntime` | PP-OCR 模型的 onnxruntime 版，约 60MB，CPU 可跑，与其它依赖无冲突 |
| PaddleOCR | `pip install paddleocr paddlepaddle` | 官方实现，效果略好；但依赖重（paddlepaddle 100MB+），且要求 `numpy<2.4` + `opencv-contrib 4.10`，与本项目的 `opencv-python-headless<5` 组合冲突，**建议单独建虚拟环境使用** |

两个都装时脚本优先用 PaddleOCR；PaddleOCR 初始化失败（缺依赖、模型下载不通）会自动回退 RapidOCR，
并在日志里说明原因。首次使用任一引擎都会自动下载模型文件（几十 MB）。

---

## 一、两条命令搞定

```bash
cd backend

# 1) 把资料放进去（.md 或 .txt，一个主题一个文件）
#    backend/knowledge_base/docs/ 是你的资料目录

# 2) 导入（向量化 + 入库）
python scripts/ingest_local.py

# 看一下入库结果
python scripts/ingest_local.py --list
```

输出示例：

```
导入结果： {'documents': 3, 'chunks': 42, 'skipped': 1, 'titles': ['三庭五眼', '五官歌诀', '气色笔记']}
知识库统计： {'documents': 4, 'chunks': 54, 'embedding_provider': 'LocalHashEmbedding', ...}
```

导入后**不需要重启服务**，下一次分析就会检索到新知识。

常用参数：

| 命令 | 作用 |
|---|---|
| `python scripts/ingest_local.py --path ../我的资料` | 导入其它目录（递归找 .md/.txt） |
| `python scripts/ingest_local.py --file 三庭.md` | 只导入一个文件 |
| `python scripts/ingest_local.py --source "《神相全编》摘录"` | 标注来源（会记录到 `kb_documents.source`） |
| `python scripts/ingest_local.py --list` | 只查看统计 |
| `python scripts/ingest_local.py --reset` | 清空知识库后重新导入（**会删除全部切片**） |

内容相同的文档会自动跳过（按内容 sha256 去重），所以重复执行是安全的。

---

## 二、通过接口导入（适合运营后台 / 批量上传）

```bash
# 单文件
curl -X POST http://localhost:8000/api/kb/ingest \
     -H "Authorization: Bearer <TOKEN>" \
     -F "files=@我的资料/三庭五眼.md" \
     -F "source=个人整理"

# 服务器上的整个目录
curl -X POST http://localhost:8000/api/kb/ingest \
     -H "Authorization: Bearer <TOKEN>" \
     -F "directory=/data/kb_docs"

# 先测检索效果（调参必备）
curl -X POST http://localhost:8000/api/kb/query \
     -H "Authorization: Bearer <TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"query":"三庭 中庭偏长 面相","top_k":5}'
```

生产环境建议在 `.env` 里设置 `KB_INGEST_TOKEN=一串随机口令`，
这样调用导入接口必须带 `-H "X-KB-Token: 一串随机口令"`，防止别人往知识库里灌内容。

---

## 三、文档怎么写效果最好

程序按 **Markdown 标题分层**切片，因此：

1. **一个主题一个文件**：`三庭.md`、`五眼.md`、`十二宫.md`…
2. **用标题分级**，标题写具体，它会被拼进片段前缀帮助检索：
   ```markdown
   # 三庭总论
   ## 上庭（额部）
   额部饱满开阔者，传统相学认为……
   ## 中庭（眉眼鼻）
   ```
3. **每个小节控制在 200–500 字**。太长会被按句切分（默认 350 字一片，重叠 50 字，
   可在 `.env` 里调 `KB_CHUNK_SIZE` / `KB_CHUNK_OVERLAP`）。
4. **口语化描述 + 关键词都要有**：检索词来自人脸特征标签（如「鹅蛋脸」「中庭偏长」「眼距适中」
   「印堂」「气色 红润」），正文里尽量出现这些词，命中率会明显更高。
5. **合规**：不要写「必定发财」「克夫」这类绝对化/恐吓性内容。
   即使写了，生成层也会被 `services/compliance.py` 替换掉，但原始描述越中性，解读质量越好。

---

## 四、换更好的向量模型（可选，效果提升最明显的一步）

默认是**内置哈希向量**（零依赖、离线、按关键词命中，起步够用）。
想要真正的语义检索，改 `.env` 里一行：

```ini
# 方案 A：本地神经向量（中文效果好，需装依赖，首次会下载约 100MB 模型）
EMBEDDING_PROVIDER=bge
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# 方案 B：云 Embedding（无需本地算力，按量计费）
EMBEDDING_PROVIDER=openai
OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
OPENAI_API_KEY=你的密钥
EMBEDDING_MODEL=你的 embedding 模型名
```

对应依赖：

```bash
pip install sentence-transformers   # 方案 A
```

> ⚠️ **换 Embedding 模型后必须重建索引**，因为新旧向量不在同一空间：
> ```bash
> python scripts/ingest_local.py --reset
> ```

**推荐做法**：知识库定型后再决定向量模型；调整期间用默认 `local` 反复试检索效果
（`POST /api/kb/query`），满意了再换成 `bge` 重建一次。

---

## 五、换向量库（数据量上来之后）

默认向量存在 SQLite 的 `kb_chunks.embedding` 字段里，检索时全量算余弦相似度。
**几千个切片毫无压力**（毫秒级）；到十万级再换：

1. 安装并配置 ChromaDB：`pip install chromadb`，`.env` 里 `VECTOR_STORE=chroma`；
2. 或者换 PostgreSQL + pgvector / 阿里云向量检索。

改动只在 `rag.search()` 一个函数里，函数签名和返回值不变，`interpreter.py` 不用动。

---

## 六、检索是怎么接进解读的

```
人脸照片
  → face_detector：478 个关键点
  → feature_extractor：三庭五眼/脸型/五官/气色 结构化特征
      └─ features["search_terms"] = ["鹅蛋脸", "三庭 均衡", "眼距 适中", "气色 自然", ...]
  → rag.build_queries(features)        # 按部位展开成 8 条定向检索词
  → rag.search_multi(db, queries)      # 每条各取 top-N → 去重 → 自有资料优先排序
  → interpreter.build_messages(...)    # 片段（带出处与小节标题）作为【参考资料】拼进 Prompt
  → 大模型 / 本地模板 生成解读
  → 片段信息随记录一起存入 kb_refs     # 结果页「解读引用的知识片段」可追溯
```

**为什么是「多路检索」而不是一条总检索词**：单条长句（把三庭、眼距、眉、鼻、唇、气色一次拼起来）
向量会被稀释，实测只命中一条与本人无关的片段，解读就会显得「像没喂资料」。
按部位分头检索后，每个部位都能拿到该部位最相关的知识，再合并去重，命中率和相关性都明显更好。

排序上**自有资料优先**，内置兜底文档只补空位：

| 配置 | 默认 | 作用 |
|---|---|---|
| `KB_PER_QUERY` | 3 | 每条检索词取前 N 条 |
| `KB_FALLBACK_TITLE` | 默认相学常识 | 哪篇文档算「兜底知识」（排在自有资料之后） |
| `KB_INCLUDE_FALLBACK` | true | 设为 false 则解读完全基于自建资料（缺主题时该主题没有依据） |

所以调优抓手有四个：
- 知识库本身的质量（**最主要**）；
- `KB_TOP_K` / `KB_PER_QUERY`（给模型多少参考资料，太多会稀释重点）；
- Embedding 模型（语义检索能力）；
- 如果需要某个主题一定被解读到，正文里要出现与检索词相近的说法（见上一节「关键词」）。

---

## 七、常见问题

**Q：扫描版识别得不准 / 整列漏字？**
九成是分辨率不够。把 `--dpi` 提到 350~400 再试（竖排古籍尤其吃分辨率）；
其次可以把扫描件先裁掉黑边、摆正倾斜再放进来。脚本本身也有兜底：
首轮结果太差时会自动提高 DPI 并追加一次二值化预处理，取字数最多的结果。

**Q：整理这一步会不会把原文改错？**
会。大模型整理**必须人工通读**一遍再对外使用（生成文件的头部注释里也写了这句）。
建议先用 `--pages 1-5` 小范围跑，确认风格符合预期再全量；不满意就改 Prompt 重跑，
`--distill --force` 可以随时重新生成，不影响原始 PDF 和抽取文本。

**Q：整理要花多少钱？**
按原文长度算。默认每 3000 字原文调用一次大模型（输出约 1000-1500 字）。
一本 10 万字的古籍 ≈ 35 次调用，用便宜模型成本可忽略；也可以用 `--chunk-chars` 调整批大小。

**Q：我的 PDF 有版权/涉密，能完全离线吗？**
抽取（PyMuPDF + OCR）全程在本机；只有**整理**这一步会把原文发给大模型。
若不能外发，把 `.env` 的 `LLM_PROVIDER` 指向自建的 OpenAI 兼容服务（本地 vLLM/Ollama 均可），
或先把 `raw_sources/extracted/*.txt` 人工整理成 `.md` 放进 `docs/`，跳过整理步骤。

**Q：导入后解读没变化？**
先 `POST /api/kb/query` 用同样的检索词测一下有没有命中；命中了但解读没变，
说明当前用的是大模型（会综合所有资料），或命中片段相似度太低（低于 0.3 基本无参考价值，
说明文档措辞和检索词差太远）。

**Q：能导入 PDF / Word 吗？**
先转成 `.md` / `.txt` 再导入。这样做的好处是标题结构清晰、切片质量高。
批量转换可以用 `pandoc`：`pandoc a.docx -t markdown -o a.md`。

**Q：会把我导入的内容发到云端吗？**
只有命中 top-k 的片段会随 Prompt 发给大模型（不含照片）。如果你要求全程不出内网，
把 `.env` 的 `LLM_PROVIDER` 设为 `template`（不走任何外部接口），或接自建的 OpenAI 兼容 LLM。

**Q：多份资料内容冲突怎么办？**
保留较权威、措辞中性的那份，删除另一份后 `--reset` 重建。
知识库不做自动仲裁，它只负责"把相关资料找出来"，最终表述由生成层决定。

**Q：数据存在哪？**
`kb_documents`（文档）+ `kb_chunks`（切片与向量），SQLite 文件默认在
`backend/face_app.db`；换生产库改 `.env` 的 `DATABASE_URL`。切片的原始资料请保留在
`knowledge_base/docs/` 并纳入你自己的版本管理（建议不要放进 Git 公开仓库）。
