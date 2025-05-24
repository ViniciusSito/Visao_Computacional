import os
import cv2
import numpy as np


BASE_DIR    = os.path.dirname(__file__)        
IMGS_FOLDER = os.path.join(BASE_DIR, 'imgs')   

nome_arquivo = 'img.jpg'

caminho_img = os.path.join(IMGS_FOLDER, nome_arquivo)

imagem = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(imagem, None)

imagem_color = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
imagem_sift  = cv2.drawKeypoints(
    imagem_color,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
cv2.imshow('SIFT - Keypoints', imagem_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()
