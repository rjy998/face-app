/* ==========================================================================
 * history.js —— 历史记录列表：分页加载 + 删除
 * ========================================================================== */

const listBox = document.getElementById('list');
const message = document.getElementById('message');
const btnMore = document.getElementById('btnMore');
const PAGE_SIZE = 10;

let offset = 0;
let loaded = 0;

function showMessage(text, isError) {
  message.innerHTML = '';
  message.appendChild(el('div', `notice${isError ? ' err' : ' ok'}`, text));
}

(function renderNav() {
  const holder = document.getElementById('navUser');
  const user = Auth.user;
  if (!user) {
    holder.innerHTML = '<a href="login.html">登录</a>';
    return;
  }
  const link = document.createElement('a');
  link.href = '#';
  link.textContent = `退出（${user.username}）`;
  link.onclick = (event) => { event.preventDefault(); Auth.clear(); location.href = 'login.html'; };
  holder.appendChild(link);
})();

function renderItem(item) {
  const card = el('section', 'card');
  const row = el('div', 'record');

  const img = document.createElement('img');
  img.alt = '历史照片';
  img.loading = 'lazy';
  if (item.image_url) img.src = item.image_url;
  row.appendChild(img);

  const body = el('div', 'body');
  body.appendChild(el('div', 'title', item.face_shape ? `脸型：${item.face_shape}` : '面相解读记录'));
  body.appendChild(el('div', 'desc', `${new Date(item.created_at).toLocaleString()}`));
  if (item.summary) body.appendChild(el('div', 'desc', item.summary));
  if (item.reading_preview) body.appendChild(el('div', 'desc', `${item.reading_preview}…`));

  const actions = el('div', 'row');
  const btnOpen = el('button', 'btn small', '查看解读');
  btnOpen.type = 'button';
  btnOpen.onclick = () => { location.href = `result.html?id=${item.record_id}`; };
  const btnDelete = el('button', 'btn danger small', '删除');
  btnDelete.type = 'button';
  btnDelete.onclick = async () => {
    if (!confirm('删除后照片与解读都会被永久移除，确定继续吗？')) return;
    try {
      await Api.deleteHistory(item.record_id);
      card.remove();
      loaded -= 1;
      if (!listBox.children.length) renderEmpty();
    } catch (error) {
      showMessage(error.message || '删除失败', true);
    }
  };
  actions.appendChild(btnOpen);
  actions.appendChild(btnDelete);
  body.appendChild(actions);

  row.appendChild(body);
  card.appendChild(row);
  return card;
}

function renderEmpty() {
  listBox.innerHTML = '';
  const box = el('div', 'card empty');
  box.appendChild(el('div', null, '还没有分析记录'));
  box.appendChild(el('div', 'muted', '回到首页上传一张正面照片，几秒钟就能得到一份趣味解读。'));
  const btn = el('button', 'btn', '去上传照片');
  btn.type = 'button';
  btn.style.marginTop = '12px';
  btn.onclick = () => { location.href = 'index.html'; };
  box.appendChild(btn);
  listBox.appendChild(box);
}

async function loadPage() {
  btnMore.disabled = true;
  try {
    const items = await Api.history(PAGE_SIZE, offset);
    if (offset === 0 && !items.length) {
      renderEmpty();
      return;
    }
    items.forEach((item) => listBox.appendChild(renderItem(item)));
    loaded += items.length;
    offset += items.length;
    btnMore.hidden = items.length < PAGE_SIZE;
  } catch (error) {
    showMessage(error.message || '加载失败', true);
  } finally {
    btnMore.disabled = false;
  }
}

btnMore.addEventListener('click', loadPage);

(async function init() {
  if (!Auth.requireLogin()) return;
  await loadPage();
})();
