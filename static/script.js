/**
 * 花朵识别应用 - 前端交互逻辑
 */

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const clearBtn = document.getElementById('clearBtn');
const fileInput = document.getElementById('fileInput');
const recognizeBtn = document.getElementById('recognizeBtn');
const resultSection = document.getElementById('resultSection');
const resultCard = document.getElementById('resultCard');
const resultIcon = document.getElementById('resultIcon');
const resultTitle = document.getElementById('resultTitle');
const resultMessage = document.getElementById('resultMessage');
const similarityBar = document.getElementById('similarityBar');
const similarityValue = document.getElementById('similarityValue');
const barFill = document.getElementById('barFill');
const learnCard = document.getElementById('learnCard');
const flowerName = document.getElementById('flowerName');
const learnBtn = document.getElementById('learnBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const categoriesGrid = document.getElementById('categoriesGrid');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');

// State
let currentImageData = null;
let isRequesting = false;  // 防止并发请求

// ============================================
// Toast 通知
// ============================================
function showToast(message, type = 'info') {
    toastMessage.textContent = message;
    toast.className = 'toast';
    if (type === 'success') toast.classList.add('success');
    if (type === 'error') toast.classList.add('error');

    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => toast.classList.remove('show'), 3000);
}

// ============================================
// 图片处理
// ============================================
function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        showToast('请上传图片文件', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        currentImageData = e.target.result;
        previewImage.src = currentImageData;
        uploadPlaceholder.style.display = 'none';
        previewContainer.style.display = 'inline-block';
        recognizeBtn.disabled = false;
        hideResult();
    };
    reader.readAsDataURL(file);
}

// ============================================
// 结果显示
// ============================================
function hideResult() {
    resultSection.style.display = 'none';
}

function clearImage() {
    currentImageData = null;
    previewImage.src = '';
    uploadPlaceholder.style.display = 'flex';
    previewContainer.style.display = 'none';
    fileInput.value = '';
    recognizeBtn.disabled = true;
    hideResult();
    learnCard.style.display = 'none';
}

function showRecognizedResult(category, similarity) {
    resultSection.style.display = 'block';
    learnCard.style.display = similarity < 1.0 ? 'block' : 'none';

    resultCard.className = 'result-card recognized';
    resultIcon.textContent = getFlowerEmoji(category);
    resultTitle.textContent = category;
    resultMessage.textContent = `我认出来了！这好像是「${category}」哦~`;

    const percent = Math.round(similarity * 100);
    similarityValue.textContent = `${percent}%`;
    barFill.style.width = `${percent}%`;
    similarityBar.style.display = 'block';
}

function showUnknownResult(similarity) {
    resultSection.style.display = 'block';
    learnCard.style.display = 'block';

    resultCard.className = 'result-card unknown';
    resultIcon.textContent = '🤔';
    resultTitle.textContent = '我还不认识这朵花';
    resultMessage.textContent = '我还没学会这种花，请先教我这是什么花！';

    const percent = Math.round(similarity * 100);
    similarityValue.textContent = `${percent}%`;
    barFill.style.width = `${percent}%`;
    similarityBar.style.display = 'block';
}

function getFlowerEmoji(category) {
    const flowerMap = {
        '梅花': '🌺',
        '樱花': '🌸',
        '桃花': '🌺',
        '郁金香': '🌷',
        '玫瑰': '🌹',
        '向日葵': '🌻',
        '菊花': '🌼',
        '牡丹': '🌺',
        '荷花': '🪷',
        '兰花': '💐'
    };
    return flowerMap[category] || '🌼';
}

// ============================================
// API 调用
// ============================================
async function recognizeFlower(retry = true) {
    if (!currentImageData) return;
    if (isRequesting) {
        showToast('正在处理中，请稍候', 'info');
        return;
    }

    isRequesting = true;
    showLoading(true);
    hideResult();

    let timeoutId = null;

    try {
        timeoutId = setTimeout(() => {
            showToast('请求超时，请稍后重试', 'error');
            showLoading(false);
            isRequesting = false;
        }, 30000);

        const response = await fetch('/api/recognize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: currentImageData })
        });

        clearTimeout(timeoutId);
        timeoutId = null;

        const data = await response.json();

        if (data.success) {
            if (data.recognized) {
                showRecognizedResult(data.category, data.similarity);
            } else {
                showUnknownResult(data.similarity);
            }
        } else {
            showToast(data.error || '识别失败', 'error');
        }
    } catch (error) {
        console.error('识别错误:', error);
        if (timeoutId) clearTimeout(timeoutId);
        if (retry) {
            await new Promise(r => setTimeout(r, 2000));
            recognizeFlower(false);
        } else {
            showToast('网络错误，请稍后重试', 'error');
        }
    } finally {
        showLoading(false);
        isRequesting = false;
    }
}

async function learnFlower(retry = true) {
    const name = flowerName.value.trim();

    if (!name) {
        showToast('请输入花的名字', 'error');
        flowerName.focus();
        return;
    }

    if (!currentImageData) {
        showToast('没有图片', 'error');
        return;
    }

    if (isRequesting) {
        showToast('正在处理中，请稍候', 'info');
        return;
    }

    isRequesting = true;
    showLoading(true);

    let timeoutId = null;

    try {
        timeoutId = setTimeout(() => {
            showToast('请求超时，请稍后重试', 'error');
            showLoading(false);
            isRequesting = false;
        }, 30000);

        const response = await fetch('/api/learn', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: currentImageData,
                category: name
            })
        });

        clearTimeout(timeoutId);
        timeoutId = null;

        const data = await response.json();

        if (data.success) {
            showToast(data.message, 'success');
            learnCard.style.display = 'none';
            flowerName.value = '';
            loadCategories();
        } else {
            showToast(data.error || '学习失败', 'error');
        }
    } catch (error) {
        console.error('学习错误:', error);
        if (timeoutId) clearTimeout(timeoutId);
        if (retry) {
            await new Promise(r => setTimeout(r, 2000));
            learnFlower(false);
        } else {
            showToast('网络错误，请稍后重试', 'error');
        }
    } finally {
        showLoading(false);
        isRequesting = false;
    }
}

async function loadCategories() {
    try {
        const response = await fetch('/api/categories');
        const data = await response.json();

        if (data.success) {
            renderCategories(data.categories);
        }
    } catch (error) {
        console.error('加载类别失败:', error);
    }
}

function renderCategories(categories) {
    const entries = Object.entries(categories);

    if (entries.length === 0) {
        categoriesGrid.innerHTML = '<div class="category-loading">还没有学习任何花朵，快上传图片让我学习吧！</div>';
        return;
    }

    categoriesGrid.innerHTML = entries.map(([name, info]) => `
        <div class="category-card">
            <div class="category-icon">${getFlowerEmoji(name)}</div>
            <div class="category-name">${name}</div>
        </div>
    `).join('');
}

function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

// ============================================
// 事件绑定
// ============================================

// 上传区域点击
uploadArea.addEventListener('click', () => {
    if (!previewContainer.style.display || previewContainer.style.display === 'none') {
        fileInput.click();
    }
});

// 文件选择
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

// 拖拽上传
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

// 清除按钮
clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearImage();
});

// 识别按钮
recognizeBtn.addEventListener('click', recognizeFlower);

// 学习按钮
learnBtn.addEventListener('click', learnFlower);

// 键盘事件
flowerName.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') learnFlower();
});

// ============================================
// 后端连接检测
// ============================================
async function checkBackendConnection() {
    try {
        const response = await fetch('/api/categories', {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            return true;
        }
        return false;
    } catch (error) {
        return false;
    }
}

// 预热后端连接（防止冷启动）
async function warmupBackend() {
    try {
        await fetch('/api/categories', { method: 'GET' });
    } catch (e) {
        // 忽略预热失败
    }
}

async function init() {
    try {
        const response = await fetch('/api/categories');
        const data = await response.json();

        if (data.success) {
            renderCategories(data.categories);
        } else {
            categoriesGrid.innerHTML = '<div class="category-loading">加载失败</div>';
        }
    } catch (error) {
        console.error('加载类别失败:', error);
        categoriesGrid.innerHTML = '<div class="category-loading">后端服务未启动</div>';
    }
}

// ============================================
// 初始化
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    init();
});
