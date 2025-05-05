import os
import cv2

IMGS_DIR = 'aula29\imgs'
OUT_DIR = os.path.join(IMGS_DIR, 'rotacionadas')
os.makedirs(OUT_DIR, exist_ok=True)

img_files = sorted([
    f for f in os.listdir(IMGS_DIR)
    if os.path.isfile(os.path.join(IMGS_DIR, f))
       and f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

angulo = 45
escala = 1.0

for fname in img_files:
    path = os.path.join(IMGS_DIR, fname)
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR)  

    h, w = img.shape[:2]
    centro = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(centro, angulo, escala)

    rotated = cv2.warpAffine(img, M, (w, h))

    out_path = os.path.join(OUT_DIR, fname)
    cv2.imwrite(out_path, rotated)

    cv2.imshow('Rotacionada', rotated)
    cv2.waitKey(0)

cv2.destroyAllWindows()
