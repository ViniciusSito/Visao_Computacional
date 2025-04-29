import os
import cv2
import numpy as np

"""

"""

def aplicar_shi_tomasi(imagem_gray, max_corners, quality_level, min_distance):
    cantos = cv2.goodFeaturesToTrack(
        imagem_gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        useHarrisDetector=False
    )
    img_out = imagem_gray.copy()
    if cantos is not None:
        for c in np.int32(cantos):
            x, y = c.ravel()
            cv2.circle(img_out, (x, y), 3, 255, -1)
    return img_out


IMGS_DIR = os.path.join(os.path.dirname(__file__), 'imgs')


img_files = sorted([
    f for f in os.listdir(IMGS_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])


imagens = []
for fname in img_files:
    path = os.path.join(IMGS_DIR, fname)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"[Aviso] não consegui carregar {fname}, pulando.")
        continue
    imagens.append((fname, gray))


configs = [
    (50,  0.01, 10),
    (100, 0.05, 20),
    (150, 0.10, 30),
]
for fname, img_gray in imagens:
    for maxC, ql, md in configs:
        out = aplicar_shi_tomasi(
            img_gray,
            max_corners   = maxC,
            quality_level = ql,
            min_distance  = md
        )
        title = f'{fname} — MC={maxC}, QL={ql:.2f}, MD={md}'
        cv2.imshow(title, out)

cv2.waitKey(0)
cv2.destroyAllWindows()
