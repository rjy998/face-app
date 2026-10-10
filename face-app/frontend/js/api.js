/* ==========================================================================
 * api.js —— 后端接口调用封装（所有页面共用）
 * 说明：token 只存在 localStorage 里，随 Authorization: Bearer 头发送。
 *      人脸照片不做任何本地持久化：仅在内存中预览与上传。
 * ========================================================================== */

/** 同源部署时直接用当前 origin；用 file:// 打开页面时回退到本地后端 */
const API_BASE = (location.protocol === 'file:') ? 'http://localhost:8000' : location.origin;

const TOKEN_KEY = 'face_app_token';
const USER_KEY = 'face_app_user';

const Auth = {
  get token() { return localStorage.getItem(TOKEN_KEY) || ''; },
  get user() {
    try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null'); } catch (e) { return null; }
  },
  save(data) {
    localStorage.setItem(TOKEN_KEY, data.access_token);
    localStorage.setItem(USER_KEY, JSON.stringify(data.user));
  },
  clear() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
  },
  /** 未登录则跳转登录页，并带上回跳地址 */
  requireLogin() {
    if (!this.token) {
      const back = encodeURIComponent(location.pathname + location.search);
      location.href = `login.html?redirect=${back}`;
      return false;
    }
    return true;
  }
};

/** 统一的 fetch 封装：自动带 token、统一错误信息 */
async function request(path, options = {}) {
  const headers = Object.assign({}, options.headers || {});
  if (Auth.token) headers['Authorization'] = `Bearer ${Auth.token}`;
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  const response = await fetch(API_BASE + path, Object.assign({}, options, { headers }));
  if (response.status === 401) {
    Auth.clear();
    const back = encodeURIComponent(location.pathname + location.search);
    location.href = `login.html?redirect=${back}`;
    throw new Error('登录已过期，请重新登录');
  }

  const text = await response.text();
  const data = text ? safeParse(text) : null;
  if (!response.ok) {
    const detail = (data && (data.detail || data.message)) || `请求失败（${response.status}）`;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return data;
}

function safeParse(text) {
  try { return JSON.parse(text); } catch (e) { return { detail: text }; }
}

const Api = {
  register: (username, password) =>
    request('/api/auth/register', { method: 'POST', body: JSON.stringify({ username, password }) }),
  login: (username, password) =>
    request('/api/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) }),
  guest: () => request('/api/auth/guest', { method: 'POST' }),
  me: () => request('/api/auth/me'),

  /** 上传照片做分析；onProgress(percent) 可选 */
  analyze(file, onProgress) {
    return new Promise((resolve, reject) => {
      const form = new FormData();
      form.append('file', file, file.name || 'photo.jpg');
      const xhr = new XMLHttpRequest();
      xhr.open('POST', API_BASE + '/api/face/analyze');
      if (Auth.token) xhr.setRequestHeader('Authorization', `Bearer ${Auth.token}`);
      xhr.upload.onprogress = (event) => {
        if (onProgress && event.lengthComputable) {
          onProgress(Math.round((event.loaded / event.total) * 100));
        }
      };
      xhr.onload = () => {
        const data = xhr.responseText ? safeParse(xhr.responseText) : null;
        if (xhr.status >= 200 && xhr.status < 300) return resolve(data);
        if (xhr.status === 401) {
          Auth.clear();
          location.href = 'login.html';
          return reject(new Error('登录已过期，请重新登录'));
        }
        const detail = (data && data.detail) || `分析失败（${xhr.status}）`;
        reject(new Error(typeof detail === 'string' ? detail : JSON.stringify(detail)));
      };
      xhr.onerror = () => reject(new Error('网络异常，请检查后端服务是否已启动'));
      xhr.send(form);
    });
  },

  history: (limit = 20, offset = 0) => request(`/api/history?limit=${limit}&offset=${offset}`),
  historyDetail: (id) => request(`/api/history/${id}`),
  deleteHistory: (id) => request(`/api/history/${id}`, { method: 'DELETE' }),
  health: () => request('/api/health')
};

/** 小工具：安全地把文本写进元素（避免 XSS） */
function setText(el, text) {
  if (el) el.textContent = text == null ? '' : String(text);
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}
