import cv2
import PoseModulo as pm
import time
import numpy as np

# Importar modulo de detecção de pose/ Carregar vídeo
detector = pm.DetectorPose()
# Inicializar Web-Cam e coletar cada Frame
capt = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#capt = cv2.VideoCapture('Recursos/RD.mp4')
cont = 0
direcao = 0
tAnterior = 0

while True:
    # Exibir vídeo
    success, img = capt.read()

    # Exibir a imagem em 25% de seu tamanho original
    img = cv2.resize(img, (0, 0), None, 1.75, 1.75)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Corrigir FPS
    #tAtual = time.time()
    #fps = 1 / (tAtual - tAnterior)
    #tAnterior = tAtual

    # Carregar imagem
    # img = cv2.imread('Recursos/Cross.jpg')

    # Localizar pose (False, remover as marcações desnecessárias com FALSE)
    img = detector.acharPose(img, False)

    # Achar os valores das landmarks
    lmLista = detector.acharPosicao(img, False)
    print(lmLista)

    # Verificar a existencia da lista e selecionar pontos desejados (no caso os braços )
    if len(lmLista) != 0:
        # Braço Esquerdo
        angulo = detector.acharangulo(img, 11, 13, 15)
        # Braço Direito
        angulo = detector.acharangulo(img, 12, 14, 16)

        # Numpy para converter o ângulo (min = 110 , max = 315)
        porcentagem = np.interp(angulo, (130, 300), (0, 100))
        print(angulo, porcentagem)

        #Barra de com porcentagem do ângulo
        barra = np.interp(angulo, (130,320), (650,100))
        # print(angulo, porcentagem)

        # Contar Número de Repeticões
        cor = (0, 255, 0)
        if porcentagem == 100:
            cor = (0, 0, 255)
            if direcao == 0:
                cont += 0.5
                direcao = 1
        if porcentagem == 0:
            cor = (255, 0, 0)
            if direcao == 1:
                cont += 0.5
                direcao = 0
        print(cont)

        #Design da barra de porcentagem
        cv2.rectangle(img, (150, 480), (200, 720), cor, 2)
        cv2.rectangle(img, (150, int(barra)), (200, 720), cor, cv2.FILLED)
        cv2.putText(img, f'{int(porcentagem)} %', (100, 50), cv2.FONT_HERSHEY_PLAIN, 4, cor, 4)

        # Design do Contador
        cv2.rectangle(img, (0, 570), (150, 720), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, str(int(cont)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 15)

    cv2.imshow("Gustavo AI Personal", img)
    cv2.waitKey(1)
