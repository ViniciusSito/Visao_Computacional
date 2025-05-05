import os
import cv2

IMGS_DIR = r'aula29\imgs'
OUT_DIR = os.path.join(IMGS_DIR, 'escaladas')
os.makedirs(OUT_DIR, exist_ok=True)

sx, sy = 1.5, 1.5

img_files = sorted([
    f for f in os.listdir(IMGS_DIR)
    if os.path.isfile(os.path.join(IMGS_DIR, f))
       and f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

for fname in img_files:
    path = os.path.join(IMGS_DIR, fname)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Falha ao carregar {fname}, pulando…")
        continue

    img_escalada = cv2.resize(
        img,
        None,
        fx=sx,
        fy=sy,
        interpolation=cv2.INTER_LINEAR
    )

    out_path = os.path.join(OUT_DIR, fname)
    cv2.imwrite(out_path, img_escalada)
    cv2.imshow('Imagem Escalada', img_escalada)
    cv2.waitKey(0)

cv2.destroyAllWindows()
print(f'Todas as imagens foram escaladas em ({sx}x, {sy}x) e salvas em "{OUT_DIR}".')
