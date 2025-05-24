import os
import cv2
import numpy as np


IMGS_DIR = os.path.join(os.path.dirname(__file__), 'imgs')

THRESH_BINARIO_VALOR = 127
ADAPTIVE_THRESH_BLOCK_SIZE = 11 
ADAPTIVE_THRESH_C = 2 
REGION_GROWING_SEED = (100, 100) 
REGION_GROWING_SIMILARITY_THRESH = 200 

# Lista os arquivos de imagem no diretório.
if not os.path.isdir(IMGS_DIR):
    print(f"Erro: O diretório de imagens '{IMGS_DIR}' não foi encontrado.")
    print("Crie uma pasta chamada 'imgs' no mesmo diretório do script e coloque suas imagens lá.")
    img_files = []
else:
    img_files = sorted([
        f for f in os.listdir(IMGS_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

def carregar_imagem(caminho_imagem):
    """Carrega uma imagem do caminho especificado."""
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_COLOR)
    if imagem is None:
        print(f"Erro: Não foi possível carregar a imagem em '{caminho_imagem}'.")
        print("Verifique se o arquivo existe e está no caminho correto.")
        return None
    return imagem

def exibir_imagens(titulos_e_imagens):
    """Exibe múltiplas imagens em janelas separadas."""
    for titulo, imagem in titulos_e_imagens:
        if imagem is not None:
            cv2.imshow(titulo, imagem)
    print("\nPressione qualquer tecla nas janelas das imagens para fechar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Funções de Segmentação ---

def aplicar_thresholding_binario(imagem_colorida, limiar):
    """Aplica thresholding binário a uma imagem."""
    if imagem_colorida is None: return None
    imagem_cinza = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)
    # O primeiro valor retornado é o limiar efetivamente usado (útil com THRESH_OTSU)
    _, imagem_binaria = cv2.threshold(imagem_cinza, limiar, 255, cv2.THRESH_BINARY)
    return imagem_binaria

def aplicar_thresholding_adaptativo(imagem_colorida, block_size, C):
    """Aplica thresholding adaptativo gaussiano a uma imagem."""
    if imagem_colorida is None: return None
    imagem_cinza = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)
    imagem_adaptativa = cv2.adaptiveThreshold(imagem_cinza, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, block_size, C)
    return imagem_adaptativa

def simular_region_growing(imagem_cinza, seed_coords, similaridade_limite, iteracoes=5):
    """Simula um crescimento de região básico a partir de uma semente."""
    if imagem_cinza is None: return None
    if not (0 <= seed_coords[0] < imagem_cinza.shape[0] and \
            0 <= seed_coords[1] < imagem_cinza.shape[1]):
        print(f"Erro: Coordenadas da semente {seed_coords} estão fora dos limites da imagem.")
        return None

    mascara = np.zeros_like(imagem_cinza) # Cria uma máscara preta do mesmo tamanho
    mascara[seed_coords] = 255 # Define o pixel da semente como branco na máscara

    # Simulação de crescimento usando dilatação e checagem de intensidade
    # Nota: Este é um exemplo muito simplificado de crescimento de região.
    # Algoritmos reais são mais sofisticados.
    for _ in range(iteracoes):
        mascara_dilatada = cv2.dilate(mascara, None, iterations=1)
        # Adiciona à máscara apenas os pixels dilatados que estão na imagem original
        # e atendem ao critério de similaridade (intensidade > limite neste caso)
        novos_pixels = (mascara_dilatada > 0) & (imagem_cinza > similaridade_limite)
        mascara[novos_pixels] = 255
    return mascara

def aplicar_watershed_simplificado(imagem_colorida, imagem_cinza_original):

    if imagem_colorida is None or imagem_cinza_original is None:
        print("Erro Watershed: imagem_colorida ou imagem_cinza_original é None.")
        return None

    # 1. Verificar o tipo da imagem de entrada (deve ser CV_8UC3)
    if not (imagem_colorida.ndim == 3 and imagem_colorida.shape[2] == 3 and imagem_colorida.dtype == np.uint8):
        print(f"Erro Watershed: A imagem colorida de entrada não é CV_8UC3 (8-bit, 3-canais). Tipo atual: {imagem_colorida.dtype}, Dims: {imagem_colorida.ndim}")
        # Tentar converter se for um caso comum, como BGRA ou Cinza para BGR
        if imagem_colorida.ndim == 2 and imagem_colorida.dtype == np.uint8: # Cinza
            imagem_colorida = cv2.cvtColor(imagem_colorida, cv2.COLOR_GRAY2BGR)
            print("Info Watershed: Imagem cinza convertida para BGR.")
        elif imagem_colorida.ndim == 3 and imagem_colorida.shape[2] == 4 and imagem_colorida.dtype == np.uint8: # BGRA
            imagem_colorida = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGRA2BGR)
            print("Info Watershed: Imagem BGRA convertida para BGR.")
        else:
            print("Erro Watershed: Não foi possível converter a imagem de entrada para o formato CV_8UC3 necessário.")
            return None

    # Esta é uma simplificação apenas para evitar o erro de tipo e seguir a ideia básica da aula.
    marcadores_para_algoritmo = imagem_cinza_original.astype(np.int32)

    # A função cv2.watershed modifica a imagem de marcadores_para_algoritmo diretamente.
    cv2.watershed(imagem_colorida, marcadores_para_algoritmo)

    # Para exibir, convertemos os marcadores (que agora contêm -1 para os limites) para uint8.
    # O valor -1 se tornará 255 (branco) devido ao overflow na conversão para tipo não sinalizado.
    imagem_watershed_visualizacao = marcadores_para_algoritmo.astype(np.uint8)

    # imagem_colorida_com_bordas[marcadores_para_algoritmo == -1] = [0, 0, 255] # Limites em vermelho

    return imagem_watershed_visualizacao


# --- Fluxo Principal do Projeto ---
if __name__ == "__main__":
    if not img_files:
        print(f"Nenhuma imagem (.jpg, .jpeg, .png) encontrada na pasta '{IMGS_DIR}'.")
        print(f"Verifique o diretório: {os.path.abspath(IMGS_DIR)}")
    else:
        nome_arquivo_imagem = img_files[0] # Pega o primeiro arquivo
        caminho_completo_imagem = os.path.join(IMGS_DIR, nome_arquivo_imagem)
        print(f"Carregando imagem: {caminho_completo_imagem}")

        imagem_original = carregar_imagem(caminho_completo_imagem)

        if imagem_original is not None:
            # Converter para escala de cinza uma vez para reuso
            imagem_cinza_original = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)

            # 1. Thresholding Binário
            img_thresh_binario = aplicar_thresholding_binario(imagem_original.copy(), THRESH_BINARIO_VALOR)

            # 2. Thresholding Adaptativo
            img_thresh_adaptativo = aplicar_thresholding_adaptativo(imagem_original.copy(),
                                                                   ADAPTIVE_THRESH_BLOCK_SIZE,
                                                                   ADAPTIVE_THRESH_C)
            # 3. Simulação de Crescimento de Região
            img_region_growing = simular_region_growing(imagem_cinza_original.copy(),
                                                        REGION_GROWING_SEED,
                                                        REGION_GROWING_SIMILARITY_THRESH)

            # 4. Watershed Simplificado (como no exemplo do texto)
            img_watershed = aplicar_watershed_simplificado(imagem_original.copy(),
                                                          imagem_cinza_original.copy())

            # Preparar para exibir
            imagens_para_exibir = [
                ("1. Imagem Original", imagem_original),
                ("1a. Imagem em Cinza", imagem_cinza_original),
                ("2. Thresholding Binario", img_thresh_binario),
                ("3. Thresholding Adaptativo", img_thresh_adaptativo),
                ("4. Crescimento de Regiao (Simples)", img_region_growing),
                ("5. Watershed (Simplificado)", img_watershed)
            ]
            exibir_imagens(imagens_para_exibir)
        else:
            print("Encerrando o programa devido a erro no carregamento da imagem.")