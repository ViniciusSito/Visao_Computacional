import os
import cv2
import numpy as np


IMGS_DIR = os.path.join(os.path.dirname(__file__), 'imgs')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output_aula_features')
IMAGE_FILENAME = 'imgss.png' 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
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
print(f"Imagem '{IMAGE_FILENAME}' carregada com sucesso.")

print("\nAplicando SIFT...")
try:
    sift = cv2.SIFT_create()

    keypoints_sift, descriptors_sift = sift.detectAndCompute(imagem_cinza, None)

    imagem_sift_vis = cv2.drawKeypoints(imagem_colorida, keypoints_sift, None,
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('SIFT Keypoints', imagem_sift_vis)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'sift_keypoints.jpg'), imagem_sift_vis)
    print(f"SIFT: {len(keypoints_sift)} pontos-chave detectados. Imagem salva em '{OUTPUT_DIR}'.")

except cv2.error as e:
    print(f"Erro ao usar SIFT: {e}")
    print("SIFT pode não estar disponível na sua instalação do OpenCV. "
          "Tente instalar 'opencv-contrib-python' ou verifique a versão do OpenCV.")
    imagem_sift_vis = None # Para evitar erro no waitKey se SIFT falhar

print("\nNota sobre SURF: SURF é patenteado e geralmente não está disponível nas compilações padrão do OpenCV.")
print("Ele foi removido dos módulos contrib do OpenCV devido a questões de patente.")

print("\nAplicando ORB...")
try:
    orb = cv2.ORB_create(nfeatures=500) 

    keypoints_orb, descriptors_orb = orb.detectAndCompute(imagem_cinza, None)

    imagem_orb_vis = cv2.drawKeypoints(imagem_colorida, keypoints_orb, None,
                                     color=(0, 255, 0), flags=0) # flags=0 é o padrão
    cv2.imshow('ORB Keypoints', imagem_orb_vis)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'orb_keypoints.jpg'), imagem_orb_vis)
    print(f"ORB: {len(keypoints_orb)} pontos-chave detectados. Imagem salva em '{OUTPUT_DIR}'.")

except cv2.error as e:
    print(f"Erro ao usar ORB: {e}")
    imagem_orb_vis = None 

if imagem_sift_vis is not None or imagem_orb_vis is not None:
    print("\nPressione qualquer tecla nas janelas das imagens para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("\nNenhuma imagem com keypoints foi gerada para exibição.")

print("\nProcessamento concluído.")