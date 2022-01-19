# Módulo para detectar e manipular a pose
import cv2
import mediapipe as mp
import math
import time


# Classe/Construtor para permitir manipular os objetos
class DetectorPose():

    def __init__(self, mode=False, modelComp=1, smoothLm=True, seg=False, smoothSeg=True, detectionCon=0.5,
                 trackCon=0.5):

        self.mode = mode
        self.modelComp = modelComp
        self.smoothLm = smoothLm
        self.seg = seg
        self.smoothSeg = smoothSeg
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode, self.modelComp, self.smoothLm, self.seg, self.smoothSeg,
                                     self.detectionCon, self.trackCon)

    # Método para localizar a pose
    def acharPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.resultado = self.pose.process(imgRGB)
        if self.resultado.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.resultado.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    # Método para localizar os pontos do mediapipe
    def acharPosicao(self, img, draw=True):
        self.lmLista = []
        if self.resultado.pose_landmarks:
            for id, landmark in enumerate(self.resultado.pose_landmarks.landmark):
                altura, largura, channel = img.shape
                # print(id, landmark)
                cx, cy = int(landmark.x * largura), int(landmark.y * altura)
                self.lmLista.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmLista

    # Método para Localizar o Ângulo
    def acharangulo(self, img, ponto1, ponto2, ponto3, draw=True):

        # Achar as landmarks
        x1, y1 = self.lmLista[ponto1][1:]
        x2, y2 = self.lmLista[ponto2][1:]
        x3, y3 = self.lmLista[ponto3][1:]

        # Calcular o ângulo entre duas linhas com 3 pontos
        angulo = math.degrees(math.atan2(y3 - y2, y3 - y2) - math.atan2(y1 - y2, x1 - x2))

        if angulo < 0:
            angulo += 360
        # print(angulo)

        # Design
        if draw:
            # Marcação das linhas
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 0, 0), 3)

            # Marcação dos pontos
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 255, 0), 2)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 255, 0), 2)
            cv2.circle(img, (x3, y3), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 255, 0), 2)
            cv2.putText(img, str(int(angulo))  + "o" , (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        return angulo


# MAIN
def main():
    capt = cv2.VideoCapture('RD.mp4')
    tAnterior = 0
    detector = DetectorPose()

    while True:
        success, img = capt.read()
        img = detector.acharPose(img)
        lmLista = detector.acharPosicao(img)

        # Apontar o indice para rastrear o ponto desejado, ex: print(lmLista[14])
        if len(lmLista) != 0:
            print(lmLista)
            # Exemplo de demarcação do ponto:
            # cv2.circle(img, (lmLista[14][1], lmLista[14][2]), 15, (0, 0, 255), cv2.FILLED)

        tAtual = time.time()
        fps = 1 / (tAtual - tAnterior)
        tAnterior = tAtual

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Imagem", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
