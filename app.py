"""
花朵识别应用 - Flask后端
使用简单特征比对算法，适合教学
"""

from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from PIL import Image
import base64
import io
import json
from datetime import datetime
import cv2
from skimage.feature import hog

import threading
import requests
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'learned'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB


def keep_alive():
    """定时访问自己防止休眠"""
    while True:
        try:
            # 访问 Render 的健康检查端点或首页
            requests.get('https://flower-see.onrender.com/', timeout=10)
            print('[KeepAlive] 自唤醒成功')
        except Exception as e:
            print(f'[KeepAlive] 自唤醒失败: {e}')
        time.sleep(10 * 60)  # 每10分钟


# 启动后台守护线程（仅在非 debug 模式）
if not app.debug:
    keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
    keep_alive_thread.start()

# 相似度阈值，低于此值认为"没学会"
SIMILARITY_THRESHOLD = 0.80


def extract_features(image_path):
    """
    提取图片特征：使用HOG特征 + 颜色直方图
    HOG对形状和纹理敏感，适合区分不同类型花朵
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        img_array = np.array(img)

        # HOG 特征提取
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )

        # 颜色直方图特征 (HSV颜色空间更稳定)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

        # 归一化
        hist_h = hist_h / (hist_h.sum() + 1e-6)
        hist_s = hist_s / (hist_s.sum() + 1e-6)
        hist_v = hist_v / (hist_v.sum() + 1e-6)

        # 合并特征
        features = np.concatenate([hog_features, hist_h, hist_s, hist_v])

        return features
    except Exception as e:
        print(f"特征提取错误: {e}")
        return None


def extract_features_from_base64(base64_str):
    """从Base64字符串提取特征"""
    try:
        # 去除前缀
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]

        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img = img.resize((128, 128), Image.Resampling.LANCZOS)
        img_array = np.array(img)

        # HOG 特征提取
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )

        # 颜色直方图特征 (HSV)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

        # 归一化
        hist_h = hist_h / (hist_h.sum() + 1e-6)
        hist_s = hist_s / (hist_s.sum() + 1e-6)
        hist_v = hist_v / (hist_v.sum() + 1e-6)

        # 合并特征
        features = np.concatenate([hog_features, hist_h, hist_s, hist_v])

        return features
    except Exception as e:
        print(f"Base64特征提取错误: {e}")
        return None


def cosine_similarity(a, b):
    """计算余弦相似度"""
    if a is None or b is None:
        return 0.0
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_all_learned_categories():
    """获取所有已学习的类别"""
    categories = {}
    base_path = app.config['UPLOAD_FOLDER']

    if not os.path.exists(base_path):
        return categories

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            categories[folder] = []

            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and file != 'meta.json':
                    file_path = os.path.join(folder_path, file)
                    features = extract_features(file_path)
                    if features is not None:
                        categories[folder].append({
                            'path': file_path,
                            'features': features.tolist(),
                            'name': file
                        })

    return categories


def recognize_flower(image_data):
    """识别花朵"""
    # 提取上传图片的特征
    features = extract_features_from_base64(image_data)
    if features is None:
        return None, 0.0

    all_categories = get_all_learned_categories()

    if not all_categories:
        return None, 0.0

    # 计算与每个类别的相似度
    best_match = None
    best_similarity = 0.0

    for category, samples in all_categories.items():
        for sample in samples:
            sim = cosine_similarity(features, np.array(sample['features']))
            if sim > best_similarity:
                best_similarity = sim
                best_match = category

    # 只有超过阈值才返回结果
    if best_similarity >= SIMILARITY_THRESHOLD:
        return best_match, best_similarity
    else:
        return None, best_similarity


def save_learned_flower(category_name, image_data):
    """保存学习的新花朵"""
    # 创建类别文件夹
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], category_name)
    os.makedirs(folder_path, exist_ok=True)

    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{timestamp}.jpg'
    file_path = os.path.join(folder_path, filename)

    # 解码并保存图片
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img.save(file_path, 'JPEG', quality=95)

    return file_path


@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/learned/<path:filename>')
def serve_learned(filename):
    """提供learned文件夹中的图片访问"""
    import os
    from flask import send_from_directory
    # 安全验证：确保文件在learned目录内
    safe_path = os.path.normpath(filename)
    if safe_path.startswith(os.pardir):
        return jsonify({'error': 'Invalid path'}), 403
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_path)
    if os.path.exists(full_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], safe_path)
    return jsonify({'error': 'File not found', 'path': full_path}), 404


@app.route('/api/debug/files')
def debug_files():
    """调试：列出learned文件夹中的文件"""
    import os
    learned_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(learned_dir):
        return jsonify({'exists': False, 'files': []})
    files = []
    for root, dirs, filenames in os.walk(learned_dir):
        for f in filenames:
            full_path = os.path.join(root, f)
            files.append({
                'path': full_path,
                'web_path': full_path.replace('\\', '/')
            })
    return jsonify({'exists': True, 'files': files})


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """识别花朵API"""
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': '没有图片数据'})

    image_data = data['image']
    threshold = float(data.get('threshold', SIMILARITY_THRESHOLD))
    category, similarity = recognize_flower(image_data)

    if category and similarity >= threshold:
        return jsonify({
            'success': True,
            'recognized': True,
            'category': category,
            'similarity': float(similarity)
        })
    else:
        return jsonify({
            'success': True,
            'recognized': False,
            'category': None,
            'similarity': float(similarity),
            'message': '相似度太低，未能识别出这是什么花，请先教我这是什么花！'
        })


@app.route('/api/learn', methods=['POST'])
def learn():
    """学习新花朵API"""
    data = request.get_json()

    if not data or 'image' not in data or 'category' not in data:
        return jsonify({'success': False, 'error': '缺少必要参数'})

    image_data = data['image']
    category_name = data['category'].strip()

    if not category_name:
        return jsonify({'success': False, 'error': '请输入花的名字'})

    # 验证类别名不包含特殊字符
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        if char in category_name:
            return jsonify({'success': False, 'error': '名字不能包含特殊字符'})

    try:
        file_path = save_learned_flower(category_name, image_data)
        return jsonify({
            'success': True,
            'message': f'学会了！以后我能识别"{category_name}"了！',
            'path': file_path
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'保存失败: {str(e)}'})


@app.route('/api/categories', methods=['GET'])
def list_categories():
    """获取所有已学习的类别"""
    all_categories = get_all_learned_categories()

    result = {}
    for cat, items in all_categories.items():
        result[cat] = {
            'count': len(items),
            'source': 'learned'
        }

    return jsonify({
        'success': True,
        'categories': result
    })


@app.route('/api/category/<category_name>', methods=['GET'])
def get_category_images(category_name):
    """获取某个类别的所有图片"""
    all_categories = get_all_learned_categories()

    images = []
    for cat, items in all_categories.items():
        if cat == category_name:
            for item in items:
                # 统一路径分隔符为正斜杠
                web_path = item['path'].replace('\\', '/')
                images.append({
                    'name': item['name'],
                    'path': web_path,
                    'source': 'learned'
                })

    return jsonify({
        'success': True,
        'category': category_name,
        'images': images
    })


@app.route('/api/category/<category_name>/delete', methods=['POST'])
def delete_category_image(category_name):
    """删除某个类别中的单张图片"""
    data = request.get_json()
    if not data or 'path' not in data:
        return jsonify({'success': False, 'error': '缺少图片路径'})

    file_path = data['path'].replace('/', '\\')

    # 安全检查：只允许删除 learned 文件夹下的图片
    if not file_path.startswith(app.config['UPLOAD_FOLDER']):
        return jsonify({'success': False, 'error': '只能删除学习文件夹中的图片'})

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': True, 'message': '删除成功'})
        else:
            return jsonify({'success': False, 'error': '文件不存在'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'删除失败: {str(e)}'})


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print('=' * 50)
    print('Flower Recognition App Starting...')
    print('=' * 50)
    print('Access: http://localhost:5000')
    print('Image storage: learned/ folder')
    print('=' * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
