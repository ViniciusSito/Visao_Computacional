import os
import cv2
import numpy as np

IMGS_DIR = r'aula29\imgs'
OUT_DIR = os.path.join(IMGS_DIR, 'transladadas')
os.makedirs(OUT_DIR, exist_ok=True)

tx, ty = 100, 50

img_files = sorted([
    f for f in os.listdir(IMGS_DIR)
    if os.path.isfile(os.path.join(IMGS_DIR, f))
       and f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

for fname in img_files:
    path = os.path.join(IMGS_DIR, fname)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Use IMREAD_COLOR se quiser cores
    if img is None:
        print(f"Falha ao carregar {fname}, pulando…")
        continue

    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    
    h, w = img.shape[:2]
    img_trans = cv2.warpAffine(img, M, (w, h))

    out_path = os.path.join(OUT_DIR, fname)
    cv2.imwrite(out_path, img_trans)

    cv2.imshow('Transladada', img_trans)
    cv2.waitKey(0)

cv2.destroyAllWindows()
print(f'Todas as imagens foram transladadas em ({tx},{ty}) e salvas em "{OUT_DIR}".')
