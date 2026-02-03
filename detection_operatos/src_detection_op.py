import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# -------- Step 1: Get image --------
# Fixed: added quotes around the path string
img = cv2.imread("/content/bugatti-veyron-hypercar-orange-color-wallpaper-preview.jpg")

if img is None:
    print("Error: Image not found at /content/bugatti-veyron-hypercar-orange-color-wallpaper-preview.jpg")
else:
    print("Original RGB Image")
    cv2_imshow(img)

    # -------- Step 2: Grayscale Preprocessing --------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    print("Grayscale Image")
    cv2_imshow(gray)
    print("Blurred Grayscale Image")
    cv2_imshow(blur_gray)

    # -------- Step 3: POINT DETECTION --------
    # Grayscale - Laplacian
    lap_gray = cv2.Laplacian(blur_gray, cv2.CV_64F)
    lap_gray = np.uint8(np.absolute(lap_gray))
    print("Point Detection (Grayscale - Laplacian)")
    cv2_imshow(lap_gray)

    # RGB - Channel-wise Laplacian
    b, g, r = cv2.split(img)

    lap_r = cv2.Laplacian(r, cv2.CV_64F)
    lap_g = cv2.Laplacian(g, cv2.CV_64F)
    lap_b = cv2.Laplacian(b, cv2.CV_64F)

    lap_rgb = np.sqrt(lap_r**2 + lap_g**2 + lap_b**2)
    lap_rgb = np.uint8(np.absolute(lap_rgb))

    print("Point Detection (RGB - Laplacian)")
    cv2_imshow(lap_rgb)

    # -------- Step 4: EDGE DETECTION - PREWITT --------
    # Prewitt kernels
    px = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    py = np.array([[ 1,  1,  1], [ 0,  0,  0], [-1, -1, -1]])

    # Grayscale Prewitt
    prewitt_gx = cv2.filter2D(blur_gray, -1, px)
    prewitt_gy = cv2.filter2D(blur_gray, -1, py)
    prewitt_gray = cv2.add(prewitt_gx, prewitt_gy)

    print("Prewitt Edge (Grayscale)")
    cv2_imshow(prewitt_gray)

    # RGB Prewitt
    pr = cv2.add(cv2.filter2D(r, -1, px), cv2.filter2D(r, -1, py))
    pg = cv2.add(cv2.filter2D(g, -1, px), cv2.filter2D(g, -1, py))
    pb = cv2.add(cv2.filter2D(b, -1, px), cv2.filter2D(b, -1, py))

    prewitt_rgb = np.sqrt(pr.astype(float)**2 + pg.astype(float)**2 + pb.astype(float)**2)
    prewitt_rgb = np.uint8(np.clip(prewitt_rgb, 0, 255))

    print("Prewitt Edge (RGB)")
    cv2_imshow(prewitt_rgb)

    # -------- Step 5: EDGE DETECTION - SOBEL --------
    # Grayscale Sobel
    sx = cv2.Sobel(blur_gray, cv2.CV_64F, 1, 0, 3)
    sy = cv2.Sobel(blur_gray, cv2.CV_64F, 0, 1, 3)

    sobel_gray = cv2.magnitude(sx, sy)
    sobel_gray = np.uint8(sobel_gray)

    print("Sobel Edge (Grayscale)")
    cv2_imshow(sobel_gray)

    # RGB Sobel
    sr = cv2.magnitude(cv2.Sobel(r, cv2.CV_64F, 1, 0, 3), cv2.Sobel(r, cv2.CV_64F, 0, 1, 3))
    sg = cv2.magnitude(cv2.Sobel(g, cv2.CV_64F, 1, 0, 3), cv2.Sobel(g, cv2.CV_64F, 0, 1, 3))
    sb = cv2.magnitude(cv2.Sobel(b, cv2.CV_64F, 1, 0, 3), cv2.Sobel(b, cv2.CV_64F, 0, 1, 3))

    sobel_rgb = np.sqrt(sr**2 + sg**2 + sb**2)
    sobel_rgb = np.uint8(np.clip(sobel_rgb, 0, 255))

    print("Sobel Edge (RGB)")
    cv2_imshow(sobel_rgb)

    # -------- Step 6: LINE DETECTION --------
    edges_gray = cv2.Canny(blur_gray, 50, 150)
    print("Canny Edges (Grayscale)")
    cv2_imshow(edges_gray)

    lines = cv2.HoughLinesP(edges_gray, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    line_img = img.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print("Line Detection (Hough - Grayscale)")
    cv2_imshow(line_img)