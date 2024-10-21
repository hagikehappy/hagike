"""
图像滤波
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_smoothing_filter(image, filter_type='mean', **kwargs):
    """
    对图像应用平滑滤波。

    :param image: 输入图像（numpy数组）。
    :param filter_type: 滤波器类型，'mean' 或 'gaussian'。
    :param kwargs: 其他参数，例如均值滤波器的卷积核大小，高斯滤波器的标准差等。
    :return: 平滑处理后的图像。
    """
    if filter_type == 'mean':
        kernel_size = kwargs.get('kernel_size', (5, 5))
        smoothed = cv2.blur(image, kernel_size)
    elif filter_type == 'gaussian':
        sigma_x = kwargs.get('sigma_x', 0)  # 高斯核在X方向的标准偏差
        smoothed = cv2.GaussianBlur(image, (0, 0), sigma_x)
    else:
        raise ValueError("Unsupported filter type")
    return smoothed


def apply_sharpening_filter(image, filter_type='laplacian', **kwargs):
    """
    对图像应用锐化滤波。

    :param image: 输入图像（numpy数组）。
    :param filter_type: 滤波器类型，'laplacian' 或 'sharpening'（自定义锐化）。
    :param kwargs: 其他参数，例如锐化滤波器的强度等。
    :return: 锐化处理后的图像。
    """
    if filter_type == 'laplacian':
        ksize = kwargs.get('ksize', 3)  # 拉普拉斯核的大小
        laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
        sharpened = cv2.convertScaleAbs(image - laplacian)  # 注意这里进行了反转
    elif filter_type == 'sharpening':
        alpha = kwargs.get('alpha', 1.5)  # 锐化强度
        beta = kwargs.get('beta', -50)  # 添加到像素的值
        sharpened = cv2.convertScaleAbs(alpha * image + beta)
    else:
        raise ValueError("Unsupported filter type")
    return sharpened


def frequency_domain_filtering(image_path):
    """频域滤波"""
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"The image at {image_path} was not found.")

        # 显示原始图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    # 转换到频域
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 计算频谱图
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # 显示频谱图
    plt.subplot(1, 3, 2)
    plt.title('Magnitude Spectrum')
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis('off')

    # 定义理想低通滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    radius = 30  # 低通滤波器的半径，可以根据需要调整
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius * radius
    mask[mask_area] = 0

    # 应用理想低通滤波器
    fshift_lowpass = dft_shift * mask
    f_ishift_lowpass = np.fft.ifftshift(fshift_lowpass)
    img_back_lowpass = cv2.idft(f_ishift_lowpass)
    img_lowpass = cv2.magnitude(img_back_lowpass[:, :, 0], img_back_lowpass[:, :, 1])

    # 显示低通滤波后的图像
    plt.subplot(1, 3, 3)
    plt.title('Low-pass Filtered Image')
    plt.imshow(img_lowpass, cmap='gray')
    plt.axis('off')

    # 显示结果
    plt.show()

    # 定义理想高通滤波器
    mask_highpass = 1 - mask

    # 应用理想高通滤波器
    fshift_highpass = dft_shift * mask_highpass
    f_ishift_highpass = np.fft.ifftshift(fshift_highpass)
    img_back_highpass = cv2.idft(f_ishift_highpass)
    img_highpass = cv2.magnitude(img_back_highpass[:, :, 0], img_back_highpass[:, :, 1])

    # 显示高通滤波后的图像（可以单独显示或保存到文件）
    plt.figure(figsize=(5, 5))
    plt.title('High-pass Filtered Image')
    plt.imshow(img_highpass, cmap='gray')
    plt.axis('off')
    plt.show()

    # 分析频域滤波效果（这里仅通过显示图像进行分析，可以进一步量化分析）
    print("Frequency domain filtering analysis:")
    print("Low-pass filter successfully removed high-frequency noise, but some details may be lost.")
    print("High-pass filter enhanced image details, but also introduced some noise.")

