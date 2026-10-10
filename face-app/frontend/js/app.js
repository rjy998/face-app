/* ==========================================================================
 * app.js —— 首页交互：文件选择 / 摄像头拍照 / 预览 / 授权确认 / 上传分析
 * ========================================================================== */

const state = { file: null, objectUrl: null };

const picker = document.getElementById('picker');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const cameraInput = document.getElementById('cameraInput');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('previewImg');
const consent = document.getElementById('consent');
const btnAnalyze = document.getElementById('btnAnalyze');
const progress = document.getElementById('progress');
const loadingBox = document.getElementById('loadingBox');
const loadingText = document.getElementById('loadingText');
const message = document.getElementById('message');
const consentModal = document.getElementById('consentModal');

// 摄像头相关
const cameraModal = document.getElementById('cameraModal');
const cameraVideo = document.getElementById('cameraVideo');
const cameraMessage = document.getElementById('cameraMessage');
let cameraStream = null;

/* ---------- 导航栏用户信息 ---------- */
(function renderNav() {
  const holder = document.getElementById('navUser');
  const user = Auth.user;
  if (!user) {
    holder.innerHTML = '<a href="login.html">登录 / 注册</a>';
    return;
  }
  const link = document.createElement('a');
  link.href = '#';
  link.textContent = `退出（${user.username}）`;
  link.onclick = (event) => {
    event.preventDefault();
    Auth.clear();
    location.reload();
  };
  holder.appendChild(link);
})();

/* ---------- 选图（文件 / 拖拽 / 粘贴 / 摄像头拍完都走这里） ---------- */
function pickFile(file) {
  hideMessage();
  if (!file) return;

  const allowed = ['image/jpeg', 'image/png', 'image/webp'];
  if (!allowed.includes(file.type)) {
    showMessage('请选择 JPG / PNG / WebP 格式的图片', true);
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showMessage('图片不能超过 10MB，请压缩后重试', true);
    return;
  }

  state.file = file;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  previewImg.src = state.objectUrl;
  // 预览覆盖在原虚线框上（CSS 绝对定位），不额外占位
  preview.hidden = false;
  dropzone.classList.add('filled');
  updateAnalyzeButton();
  picker.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function clearFile() {
  state.file = null;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = null;
  preview.hidden = true;
  previewImg.removeAttribute('src');
  dropzone.classList.remove('filled');
  fileInput.value = '';
  cameraInput.value = '';
  updateAnalyzeButton();
}

/* ---------- 文件选择 / 点击 / 拖拽 / 粘贴 ---------- */
dropzone.addEventListener('click', () => fileInput.click());
previewImg.addEventListener('click', () => fileInput.click());   // 点预览图可换一张
document.getElementById('btnPick').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (event) => pickFile(event.target.files[0]));
cameraInput.addEventListener('change', (event) => pickFile(event.target.files[0]));

// 拖拽监听挂在 picker 上：预览覆盖后依然可以继续拖入新图替换
['dragenter', 'dragover'].forEach((type) =>
  picker.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add('dragover');
  })
);
['dragleave', 'drop'].forEach((type) =>
  picker.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragover');
  })
);
picker.addEventListener('drop', (event) => {
  const file = event.dataTransfer && event.dataTransfer.files[0];
  pickFile(file);
});

// 支持直接粘贴截图（Ctrl+V）
document.addEventListener('paste', (event) => {
  const item = Array.from(event.clipboardData?.items || []).find((it) => it.type.startsWith('image/'));
  if (item) pickFile(item.getAsFile());
});

document.getElementById('btnClear').addEventListener('click', (event) => {
  event.stopPropagation();   // 别触发「点击选择照片」
  clearFile();
});

consent.addEventListener('change', updateAnalyzeButton);

function updateAnalyzeButton() {
  btnAnalyze.disabled = !(state.file && consent.checked);
}

/* ---------- 摄像头拍照 ---------- */
document.getElementById('btnCamera').addEventListener('click', openCamera);
document.getElementById('btnCameraCancel').addEventListener('click', closeCamera);
document.getElementById('btnCameraShot').addEventListener('click', takePhoto);
document.getElementById('btnCameraFallback').addEventListener('click', () =>
  fallbackToSystemCamera('已切换到系统相机 / 文件选择')
);
// 点遮罩或按 Esc 关闭取景窗
cameraModal.addEventListener('click', (event) => {
  if (event.target === cameraModal) closeCamera();
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && cameraModal.classList.contains('on')) closeCamera();
});

async function openCamera() {
  cameraMessage.innerHTML = '';

  // 浏览器要求安全上下文（https 或 localhost）才允许调用摄像头
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    fallbackToSystemCamera('当前环境无法直接调用摄像头（需通过 https 或 http://localhost 访问），已为你打开系统相机 / 文件选择');
    return;
  }

  cameraModal.classList.add('on');
  cameraMessage.appendChild(el('div', 'loading-box', '正在启动摄像头…'));

  // 有些环境（无设备、驱动异常、权限弹窗被挂起）会让 getUserMedia 长时间不返回，
  // 这里加个超时提示，避免用户对着转圈干等。
  const slowTimer = setTimeout(() => {
    if (!cameraStream) {
      cameraMessage.innerHTML = '';
      cameraMessage.appendChild(el('div', 'notice err', '摄像头启动较慢或未响应。请检查是否已允许摄像头权限，或直接点下面的按钮改用文件选择。'));
    }
  }, 10000);

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    cameraMessage.innerHTML = '';
    cameraVideo.srcObject = cameraStream;
    await cameraVideo.play();
  } catch (error) {
    cameraMessage.innerHTML = '';
    cameraMessage.appendChild(el('div', 'notice err', describeCameraError(error)));
    cameraMessage.appendChild(el('div', 'muted', '也可以点下面的按钮改用系统相机或直接选择文件。'));
  } finally {
    clearTimeout(slowTimer);
  }
}

function describeCameraError(error) {
  const name = error && error.name;
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return '摄像头权限被拒绝：请点击浏览器地址栏左侧的权限图标，把「摄像头」改为允许后重试。';
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return '没有检测到摄像头设备，请改用「从文件选择」上传照片。';
  }
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return '摄像头被其他程序占用（如会议、直播软件），请关闭后重试。';
  }
  if (name === 'OverconstrainedError') {
    return '摄像头不支持所需的分辨率，请重试或改用文件选择。';
  }
  return `无法打开摄像头：${(error && error.message) || '未知错误'}`;
}

function closeCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());  // 及时释放摄像头
    cameraStream = null;
  }
  if (cameraVideo) cameraVideo.srcObject = null;
  cameraModal.classList.remove('on');
  cameraMessage.innerHTML = '';
}

function takePhoto() {
  if (!cameraStream || !cameraVideo.videoWidth) {
    cameraMessage.innerHTML = '';
    cameraMessage.appendChild(el('div', 'notice err', '画面还没准备好，请稍等一秒再点「拍摄」。'));
    return;
  }
  const canvas = document.createElement('canvas');
  canvas.width = cameraVideo.videoWidth;
  canvas.height = cameraVideo.videoHeight;
  // 取景窗是镜像显示（照镜子的直觉），这里按摄像头原始画面保存，不做翻转
  canvas.getContext('2d').drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(
    (blob) => {
      if (!blob) {
        cameraMessage.innerHTML = '';
        cameraMessage.appendChild(el('div', 'notice err', '拍摄失败，请重试或改用文件选择。'));
        return;
      }
      const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const file = new File([blob], `camera-${stamp}.jpg`, { type: 'image/jpeg' });
      closeCamera();
      pickFile(file);
    },
    'image/jpeg',
    0.92
  );
}

/** 无法直接调用摄像头时，退化为 <input capture>（手机上会打开系统相机） */
function fallbackToSystemCamera(text) {
  closeCamera();
  showMessage(text, false);
  cameraInput.click();
}

/* ---------- 提示信息 ---------- */
function showMessage(text, isError) {
  message.innerHTML = '';
  const node = el('div', `notice${isError ? ' err' : ' ok'}`, text);
  message.appendChild(node);
}
function hideMessage() { message.innerHTML = ''; }

/* ---------- 分析流程 ---------- */
btnAnalyze.addEventListener('click', () => {
  if (!Auth.requireLogin()) return;
  if (!state.file) return showMessage('请先选择或拍摄一张照片', true);
  if (!consent.checked) return showMessage('请先勾选同意授权说明', true);
  consentModal.classList.add('on');       // 上传前二次确认（合规要求）
});

document.getElementById('btnCancelConsent').addEventListener('click', () => {
  consentModal.classList.remove('on');
});

document.getElementById('btnAgree').addEventListener('click', async () => {
  consentModal.classList.remove('on');
  await runAnalyze();
});

async function runAnalyze() {
  btnAnalyze.disabled = true;
  hideMessage();
  progress.classList.add('on');
  progress.firstElementChild.style.width = '0%';
  loadingBox.hidden = false;

  const steps = [
    '正在上传照片…',
    '正在检测人脸关键点…',
    '正在计算三庭五眼与脸型…',
    '正在检索面相知识库…',
    '正在生成解读…'
  ];
  let stepIndex = 0;
  loadingText.textContent = steps[0];
  const timer = setInterval(() => {
    stepIndex = Math.min(stepIndex + 1, steps.length - 1);
    loadingText.textContent = steps[stepIndex];
  }, 1400);

  try {
    const result = await Api.analyze(state.file, (percent) => {
      progress.firstElementChild.style.width = `${Math.max(percent, 8)}%`;
      if (percent >= 100) loadingText.textContent = steps[1];
    });
    // 结果暂存到 sessionStorage，结果页可直接渲染（刷新时会回源后端接口）
    sessionStorage.setItem('face_app_result', JSON.stringify(result));
    location.href = `result.html?id=${result.record_id}`;
  } catch (error) {
    showMessage(error.message || '分析失败，请稍后重试', true);
    btnAnalyze.disabled = false;
  } finally {
    clearInterval(timer);
    progress.classList.remove('on');
    loadingBox.hidden = true;
  }
}
