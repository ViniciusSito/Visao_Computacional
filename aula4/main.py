import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMGS_DIR = os.path.join(os.path.dirname(__file__), 'imgs')
IMAGE_FILENAME = 'imgss.jpg' 
SOBEL_KSIZE = 3
CANNY_THRESHOLD_1 = 100
CANNY_THRESHOLD_2 = 200

image_path = os.path.join(IMGS_DIR, IMAGE_FILENAME)

if not os.path.exists(IMGS_DIR):
    print(f"Erro: A pasta de imagens '{IMGS_DIR}' não foi encontrada.")
    print("Crie uma pasta chamada 'imgs' no mesmo diretório do script e coloque sua imagem lá.")
    exit()

imagem_colorida = cv2.imread(image_path)
if imagem_colorida is None:
    print(f"Erro: Não foi possível carregar a imagem '{image_path}'.")
    print(f"Certifique-se de que o arquivo '{IMAGE_FILENAME}' existe dentro da pasta '{IMGS_DIR}'.")
    exit()

imagem_cinza = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)

sobel_x_64f = cv2.Sobel(imagem_cinza, cv2.CV_64F, 1, 0, ksize=SOBEL_KSIZE)
sobel_y_64f = cv2.Sobel(imagem_cinza, cv2.CV_64F, 0, 1, ksize=SOBEL_KSIZE)
sobel_magnitude = cv2.magnitude(sobel_x_64f, sobel_y_64f)

kernel_prewitt_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype=np.float32)
kernel_prewitt_y = np.array([[1, 1, 1],
                             [0, 0, 0],
                             [-1, -1, -1]], dtype=np.float32)

prewitt_x_64f = cv2.filter2D(imagem_cinza, cv2.CV_64F, kernel_prewitt_x)
prewitt_y_64f = cv2.filter2D(imagem_cinza, cv2.CV_64F, kernel_prewitt_y)
prewitt_soma_xy = prewitt_x_64f + prewitt_y_64f

bordas_canny = cv2.Canny(imagem_cinza, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

cv2.imwrite('sobel_aula4.jpg', cv2.convertScaleAbs(sobel_magnitude))
cv2.imwrite('prewitt_aula4.jpg', cv2.convertScaleAbs(prewitt_soma_xy))
cv2.imwrite('canny_aula4.jpg', bordas_canny)

print("Imagens processadas salvas como: sobel_aula4.jpg, prewitt_aula4.jpg, canny_aula4.jpg")

plt.figure(figsize=(10, 5))

plt.subplot(1, 4, 1)
plt.imshow(imagem_cinza, cmap='gray')
plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 4, 2)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title('Sobel')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 4, 3)
plt.imshow(prewitt_soma_xy, cmap='gray')
plt.title('Prewitt (X+Y)')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 4, 4)
plt.imshow(bordas_canny, cmap='gray')
plt.title('Canny')
plt.xticks([]), plt.yticks([])

plt.suptitle('Comparação de Detecção de Bordas (Aula 4 Simplificado)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()