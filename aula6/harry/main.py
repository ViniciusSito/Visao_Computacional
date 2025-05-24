import os
import cv2
import numpy as np

def detectar_cantos_harris(image, blockSize, ksize, k):
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    harris    = cv2.cornerHarris(image, blockSize=blockSize, ksize=ksize, k=k)
    harris    = cv2.dilate(harris, None)
    img_color[harris > 0.01 * harris.max()] = [0, 0, 255]
    return img_color


IMGS_DIR = os.path.join(os.path.dirname(__file__), 'imgs')

img_files = sorted([
    f for f in os.listdir(IMGS_DIR)
    if os.path.isfile(os.path.join(IMGS_DIR, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

imgs = []
for fname in img_files:
    path = os.path.join(IMGS_DIR, fname)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Falha ao carregar {fname}, pulando…")
        continue
    imgs.append((fname, gray))

if not imgs:
    print("Nenhuma imagem válida encontrada em", IMGS_DIR)
    exit()

variacoes = [
    (2, 3, 0.04),
    (3, 5, 0.05),
    (4, 7, 0.06),
]

for i, (fname, img) in enumerate(imgs, start=1):
    for bs, ks, k in variacoes:
        out = detectar_cantos_harris(img, blockSize=bs, ksize=ks, k=k)
        janela = f'{fname} — BS={bs}, KS={ks}, k={k}'
        cv2.imshow(janela, out)

cv2.waitKey(0)
cv2.destroyAllWindows()
