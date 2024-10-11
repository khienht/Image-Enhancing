import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Khởi tạo Flask app
app = Flask(__name__)

# Thư mục để lưu trữ ảnh upload
UPLOAD_FOLDER = 'static/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Hàm xử lý ảnh: denoising, sharpening, edge detection
def process_image(image_path):
    img = cv2.imread(image_path)

    # Lưu ảnh gốc
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original.jpg')
    cv2.imwrite(original_path, img)
    
    # Bước 1: Denoising / Smoothing
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Bước 2: Sharpening
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel_sharpening)

    # Bước 3: Edge Detection
    # Sobel
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobelx, sobely)

    # Chuyển kết quả Sobel về kiểu uint8 để có thể hiển thị
    sobel = np.uint8(np.absolute(sobel_magnitude))

    # Prewitt
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    prewittx = cv2.filter2D(img, cv2.CV_64F, kernelx)
    prewitty = cv2.filter2D(img, cv2.CV_64F, kernely)
    prewitt_magnitude = cv2.magnitude(prewittx, prewitty)

    # Chuyển kết quả Prewitt về kiểu uint8 để có thể hiển thị
    prewitt = np.uint8(np.absolute(prewitt_magnitude))

    # Canny Edge
    canny = cv2.Canny(img, 100, 200)

    # Lưu ảnh xử lý vào thư mục static/images/
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'denoised.jpg'), denoised)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'sharpened.jpg'), sharpened)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'sobel.jpg'), sobel)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'prewitt.jpg'), prewitt)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'canny.jpg'), canny)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Xử lý ảnh
            process_image(filepath)

            return redirect(url_for('result'))
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
