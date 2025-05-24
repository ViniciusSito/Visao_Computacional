import os
import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMGS_DIR = os.path.join(BASE_DIR, 'imgs')

IMAGE_PATH = os.path.join(IMGS_DIR, 'imgs.jpg')

imagem = cv2.imread(IMAGE_PATH)

h, w = imagem.shape[:2]

pts_origem = np.float32([
    [0,   0],
    [w-1, 0],
    [0,   h-1],
    [w-1, h-1],
])

margem = 0.3      
dx = int(w * margem)
pts_destino = np.float32([
    [dx,     0],
    [w-1-dx, 0],
    [0,      h-1],
    [w-1,    h-1],
])

M = cv2.getPerspectiveTransform(pts_origem, pts_destino)
imagem_trapezio = cv2.warpPerspective(imagem, M, (w, h))

cv2.imshow('Imagem em Trapézio', imagem_trapezio)
cv2.imwrite(os.path.join(IMGS_DIR, 'exemplo_trapezio.jpg'), imagem_trapezio)
cv2.waitKey(0)
cv2.destroyAllWindows()
