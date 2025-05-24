import os
import cv2
import numpy as np

# --- Configurações ---
IMGS_DIR = os.path.join(os.path.dirname(__file__), 'imgs')
LARGURA_REDIMENSIONADA = 400  # Nova largura da imagem
ALTURA_REDIMENSIONADA = 300   # Nova altura da imagem
ANGULO_ROTACAO = 45           # Ângulo em graus para rotacionar
KERNEL_BLUR_SIZE = (15, 15)   # Tamanho do kernel para o GaussianBlur (deve ser ímpar)

# Lista os arquivos de imagem no diretório, priorizando jpg/jpeg, depois png

if not os.path.isdir(IMGS_DIR):
    print(f"Erro: O diretório de imagens '{IMGS_DIR}' não foi encontrado.")
    print("Crie uma pasta chamada 'imgs' no mesmo diretório do script e coloque suas imagens lá.")
    img_files = []
else:
    img_files = sorted([
        f for f in os.listdir(IMGS_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) # Mantém png, mas prioriza jpg/jpeg se precisar escolher
    ])

# --- Funções de Processamento de Imagem ---

def carregar_imagem(caminho_imagem):
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_COLOR)
    if imagem is None:
        print(f"Erro: Não foi possível carregar a imagem em '{caminho_imagem}'.")
        print("Verifique se o arquivo existe e está no caminho correto.")
        return None
    return imagem

def aplicar_blur(imagem, kernel_size):
    """Aplica o filtro Gaussian Blur na imagem."""
    if imagem is None:
        return None
    imagem_blur = cv2.GaussianBlur(imagem, kernel_size, 0)
    return imagem_blur

def redimensionar_imagem(imagem, largura, altura):
    """Redimensiona a imagem para as dimensões especificadas."""
    if imagem is None:
        return None
    imagem_redimensionada = cv2.resize(imagem, (largura, altura))
    return imagem_redimensionada

def rotacionar_imagem(imagem, angulo):
    """Rotaciona a imagem pelo ângulo especificado em torno do centro,
    ajustando o tamanho da tela para evitar cortes."""

    # 1. Validação da Imagem de Entrada
    if imagem is None:
        print("Erro: Imagem de entrada é None.")
        return None # Retorna None se a imagem não for válida

    # A forma (shape) de uma imagem colorida carregada pelo OpenCV é (altura, largura, canais).
    (h, w) = imagem.shape[:2]

    # Usamos divisão inteira (//) para garantir coordenadas de pixel inteiras.
    centro = (w // 2, h // 2)

    #   [[cos(theta), -sin(theta), (1-cos(theta))*centro_x + sin(theta)*centro_y],
    #   [sin(theta),  cos(theta), -sin(theta)*centro_x + (1-cos(theta))*centro_y]]

    matriz_rotacao = cv2.getRotationMatrix2D(centro, angulo, 1.0)

    # nova_largura = largura_original * cos_theta + altura_original * sin_theta
    # nova_altura  = largura_original * sin_theta + altura_original * cos_theta

    # matriz_rotacao[0,0] é cos(theta) e matriz_rotacao[0,1] é -sin(theta) para a transformação.

    cos_theta = np.abs(matriz_rotacao[0, 0]) # Componente alfa da matriz
    sin_theta = np.abs(matriz_rotacao[0, 1]) # Componente beta da matriz

    nova_largura = int((h * sin_theta) + (w * cos_theta))
    nova_altura = int((h * cos_theta) + (w * sin_theta))

    # Adicionamos a diferença entre o centro da nova tela e o centro da imagem original.
    # matriz_rotacao[0, 2] (tx) controla a translação em x.
    # matriz_rotacao[1, 2] (ty) controla a translação em y.
    matriz_rotacao[0, 2] += (nova_largura / 2) - centro[0]
    matriz_rotacao[1, 2] += (nova_altura / 2) - centro[1]

    # Esta função mapeia os pixels da imagem de origem para a imagem de destino

    imagem_rotacionada = cv2.warpAffine(imagem, matriz_rotacao, (nova_largura, nova_altura))

    # 8. Retorno da Imagem Rotacionada
    return imagem_rotacionada

def exibir_imagens(titulos_e_imagens):
    """Exibe múltiplas imagens em janelas separadas."""
    for titulo, imagem in titulos_e_imagens:
        if imagem is not None:
            cv2.imshow(titulo, imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Fluxo Principal do Projeto ---
if __name__ == "__main__":
    if not img_files:
        print("Nenhuma imagem (.jpg, .jpeg, .png) encontrada na pasta 'imgs'.")
        print(f"Verifique o diretório: {os.path.abspath(IMGS_DIR)}")
    else:
        # Pega o primeiro arquivo da lista (você pode adicionar lógica para escolher outro)
        nome_arquivo_imagem = img_files[0]
        caminho_completo_imagem = os.path.join(IMGS_DIR, nome_arquivo_imagem)
        print(f"Carregando imagem: {caminho_completo_imagem}")

        imagem_original = carregar_imagem(caminho_completo_imagem)

        if imagem_original is not None:
            imagem_com_blur = aplicar_blur(imagem_original.copy(), KERNEL_BLUR_SIZE)
            imagem_para_redimensionar = imagem_original.copy()
            imagem_redimensionada = redimensionar_imagem(imagem_para_redimensionar, LARGURA_REDIMENSIONADA, ALTURA_REDIMENSIONADA)
            
            imagem_para_rotacionar = imagem_redimensionada.copy() if imagem_redimensionada is not None else imagem_original.copy()
            imagem_rotacionada = rotacionar_imagem(imagem_para_rotacionar, ANGULO_ROTACAO)

            imagens_para_exibir = [
                ("1. Imagem Original", imagem_original),
                ("2. Imagem com Blur", imagem_com_blur),
                ("3. Imagem Redimensionada", imagem_redimensionada),
                ("4. Imagem Rotacionada (a partir da redimensionada)", imagem_rotacionada)
            ]
            exibir_imagens(imagens_para_exibir)
        else:
            print("Encerrando o programa devido a erro no carregamento da imagem.")