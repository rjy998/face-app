/* ==========================================================================
 * login.js —— 登录 / 注册 / 游客体验
 * ========================================================================== */

let mode = 'login';   // login | register
const redirect = new URLSearchParams(location.search).get('redirect') || 'index.html';

const formTitle = document.getElementById('formTitle');
const username = document.getElementById('username');
const password = document.getElementById('password');
const btnSubmit = document.getElementById('btnSubmit');
const btnSwitch = document.getElementById('btnSwitch');
const message = document.getElementById('message');

// 已登录直接跳回
if (Auth.token) {
  location.replace(redirect);
}

function showMessage(text, isError) {
  message.innerHTML = '';
  message.appendChild(el('div', `notice${isError ? ' err' : ' ok'}`, text));
}

function setMode(next) {
  mode = next;
  const isLogin = mode === 'login';
  formTitle.textContent = isLogin ? '登录' : '注册';
  btnSubmit.textContent = isLogin ? '登录' : '注册并登录';
  btnSwitch.textContent = isLogin ? '没有账号？去注册' : '已有账号？去登录';
  password.setAttribute('autocomplete', isLogin ? 'current-password' : 'new-password');
  message.innerHTML = '';
}

btnSwitch.addEventListener('click', () => setMode(mode === 'login' ? 'register' : 'login'));

async function submit() {
  const name = username.value.trim();
  const pwd = password.value;
  if (!name || !pwd) return showMessage('请填写用户名和密码', true);
  if (pwd.length < 6) return showMessage('密码至少 6 位', true);

  btnSubmit.disabled = true;
  btnSubmit.textContent = '处理中…';
  try {
    const data = mode === 'login' ? await Api.login(name, pwd) : await Api.register(name, pwd);
    Auth.save(data);
    location.replace(redirect);
  } catch (error) {
    showMessage(error.message || '操作失败，请稍后重试', true);
  } finally {
    btnSubmit.disabled = false;
    btnSubmit.textContent = mode === 'login' ? '登录' : '注册并登录';
  }
}

btnSubmit.addEventListener('click', submit);
password.addEventListener('keydown', (event) => { if (event.key === 'Enter') submit(); });

document.getElementById('btnGuest').addEventListener('click', async () => {
  try {
    const data = await Api.guest();
    Auth.save(data);
    location.replace(redirect);
  } catch (error) {
    showMessage(error.message || '游客体验创建失败', true);
  }
});
