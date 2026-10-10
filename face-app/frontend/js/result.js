/* ==========================================================================
 * result.js —— 结果页：展示照片、结构化特征、解读与知识来源
 * 数据来源：优先用首页传过来的 sessionStorage 结果；刷新页面时回源后端接口。
 * ========================================================================== */

const recordId = Number(new URLSearchParams(location.search).get('id') || 0);
const message = document.getElementById('message');

function showMessage(text, isError) {
  message.innerHTML = '';
  message.appendChild(el('div', `notice${isError ? ' err' : ' ok'}`, text));
}

/** 渲染一条比例条；display 用于替换默认的百分比文案（如五眼用「×1.00 倍」表达） */
function meter(label, ratio, hint, display) {
  const wrap = el('div', 'meter');
  const head = el('div', 'head');
  head.appendChild(el('span', null, label));
  const valueText = display || `${(ratio * 100).toFixed(0)}%`;
  head.appendChild(el('span', null, `${valueText}${hint ? ' · ' + hint : ''}`));
  const bar = el('div', 'bar');
  const fill = document.createElement('i');
  fill.style.width = `${Math.min(100, Math.max(3, ratio * 100))}%`;
  bar.appendChild(fill);
  wrap.appendChild(head);
  wrap.appendChild(bar);
  return wrap;
}

function tag(text, cls) { return el('span', `tag${cls ? ' ' + cls : ''}`, text); }

/* ---------- 渲染 ---------- */
function render(data) {
  const features = data.features || {};

  // 照片与概要
  document.getElementById('photoCard').hidden = false;
  if (data.image_url) document.getElementById('photo').src = data.image_url;
  setText(document.getElementById('faceShape'), `脸型：${(features.face_shape || {}).label || '—'}`);
  setText(
    document.getElementById('metaLine'),
    `分析时间：${new Date(data.created_at).toLocaleString()} · 检测引擎：${data.engine} · 解读来源：${data.llm_provider}`
  );

  const tags = document.getElementById('tags');
  tags.innerHTML = '';
  (features.summary || []).forEach((line) => tags.appendChild(tag(line)));
  if (features.estimated) tags.appendChild(tag('关键点为比例推算，精度有限', 'warn'));

  // 解读
  setText(document.getElementById('reading'), data.reading || '（暂无解读内容）');
  setText(document.getElementById('disclaimer'), data.disclaimer || '');

  // 特征
  document.getElementById('featureCard').hidden = false;
  const warnings = document.getElementById('warnings');
  warnings.innerHTML = '';
  (features.warnings || []).forEach((text) => warnings.appendChild(el('div', 'notice', text)));

  const courts = features.three_courts || {};
  const courtsBox = document.getElementById('courts');
  courtsBox.innerHTML = '';
  ['upper', 'middle', 'lower'].forEach((key) => {
    const part = courts[key] || {};
    const names = { upper: '上庭（额）', middle: '中庭（眉眼鼻）', lower: '下庭（口唇下巴）' };
    if (part.ratio != null) courtsBox.appendChild(meter(names[key], part.ratio, part.label));
  });
  courtsBox.appendChild(el('p', 'muted', courts.judgement || ''));
  // 把「上庭起点怎么来的」如实显示出来：测到发际线 / 没测到
  if (courts.hairline) {
    const measured = (courts.upper || {}).measured;
    const hairlineText = measured
      ? `上庭起点：发际线（${courts.hairline.source || '像素检测'}，置信度 ${courts.hairline.confidence}）`
      : `上庭起点：未测得（${courts.hairline.source || '发际线不可见'}）——上庭高低本次不作判断，露出额头重拍即可测得`;
    courtsBox.appendChild(el('p', 'muted', hairlineText));
  }
  if (courts.note) courtsBox.appendChild(el('p', 'muted', `说明：${courts.note}`));

  const eyes = features.five_eyes || {};
  const eyesBox = document.getElementById('eyes');
  eyesBox.innerHTML = '';
  ['eye_width_ratio', 'inner_gap_ratio', 'outer_gap_ratio'].forEach((key) => {
    const item = eyes[key] || {};
    const names = { eye_width_ratio: '单眼宽度', inner_gap_ratio: '两眼间距', outer_gap_ratio: '眼尾到脸侧' };
    if (item.value != null) {
      // 以「标准眼宽 = 1.00」为基准展示，比百分比更直观
      eyesBox.appendChild(meter(names[key], Math.min(item.value, 2) / 2, item.label, `×${item.value.toFixed(2)} 标准眼宽`));
    }
  });
  eyesBox.appendChild(el('p', 'muted', eyes.judgement || ''));

  const facial = features.facial_features || {};
  const facialBox = document.getElementById('facial');
  facialBox.innerHTML = '';
  const facialNames = { brow: '眉', eye: '眼', nose: '鼻', mouth: '口唇', forehead: '额' };
  Object.keys(facialNames).forEach((key) => {
    const item = facial[key] || {};
    if (item.label) facialBox.appendChild(tag(`${facialNames[key]}：${item.label}`, 'alt'));
  });
  const shape = features.face_shape || {};
  if (shape.label) {
    facialBox.appendChild(
      el('p', 'muted',
        `${shape.label}（${shape.reason || ''}）— 脸长/脸宽 ${shape.length_ratio}，下颌/颧骨 ${shape.jaw_cheek_ratio}，额宽/颧骨 ${shape.forehead_cheek_ratio}`)
    );
  }

  const complexion = features.complexion || {};
  const symmetry = features.symmetry || {};
  const cBox = document.getElementById('complexion');
  cBox.innerHTML = '';
  if (complexion.label) {
    cBox.appendChild(tag(`气色：${complexion.label}`, 'alt'));
    cBox.appendChild(
      el('p', 'muted',
        `明度 ${complexion.brightness ?? '—'} · 红润度 ${complexion.redness ?? '—'} · 暖黄度 ${complexion.yellowness ?? '—'} · 均匀度 ${complexion.evenness ?? '—'}`)
    );
    if (complexion.caveat) cBox.appendChild(el('p', 'muted', complexion.caveat));
  }
  if (symmetry.label) cBox.appendChild(tag(`对称度：${symmetry.label}（${symmetry.score}）`));

  // 知识来源
  const refs = data.kb_refs || [];
  if (refs.length || features.kb_query) {
    document.getElementById('kbCard').hidden = false;
    setText(document.getElementById('kbQuery'), features.kb_query ? `检索词：${features.kb_query}` : '');
    const box = document.getElementById('kbRefs');
    box.innerHTML = '';
    if (!refs.length) box.appendChild(el('p', 'muted', '知识库暂无命中内容，本次解读仅依据人脸特征生成。'));
    refs.forEach((ref) => {
      const item = el('div', 'card');
      item.style.boxShadow = 'none';
      item.style.border = '1px solid var(--line)';
      item.appendChild(el('div', null, `《${ref.doc_title}》 · 相似度 ${ref.score}`));
      item.appendChild(el('p', 'muted', ref.excerpt));
      box.appendChild(item);
    });
  }
}

/* ---------- 删除 ---------- */
document.getElementById('btnDelete').addEventListener('click', async () => {
  if (!confirm('确定删除这条记录吗？照片与解读都会被永久删除，无法恢复。')) return;
  try {
    await Api.deleteHistory(recordId);
    sessionStorage.removeItem('face_app_result');
    location.href = 'history.html';
  } catch (error) {
    showMessage(error.message || '删除失败', true);
  }
});

/* ---------- 加载 ---------- */
(async function load() {
  if (!Auth.requireLogin()) return;
  if (!recordId) {
    showMessage('缺少记录 id，请从首页重新分析', true);
    return;
  }

  const cached = sessionStorage.getItem('face_app_result');
  if (cached) {
    try {
      const parsed = JSON.parse(cached);
      if (Number(parsed.record_id) === recordId) {
        render(parsed);
        return;
      }
    } catch (error) { /* 缓存损坏则忽略，走接口 */ }
  }

  try {
    const data = await Api.historyDetail(recordId);
    render(data);
  } catch (error) {
    showMessage(error.message || '加载失败，请返回历史记录重试', true);
  }
})();
