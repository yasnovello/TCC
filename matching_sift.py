#SIFT, FAST, HOG, BRIEF, ORB,SURF


# importing required libraries SIFT
import os
import time
from unittest import result
import cv2

# ler a impressão latente SIFT
resultados = []
entrada = None

for inicio in[inicio for inicio in os.listdir("SOCOFing/Testes/medium/")]:
    sample = cv2.imread("SOCOFing/Testes/medium/"+inicio)
    sample = cv2.resize(sample, None, fx=4, fy=4)
    entrada = inicio
    print(entrada)
    # cv2.imshow("Result",sample)
    # cv2.waitKey(5)

    # melhor score
    best_score = 0
    filename = None
    # imagem de melhor matching
    image = None
    # pontos chaves para a correspondencia da impressão
    # kp1 = ponto chave 1
    # kp2 = ponto chave 2
    # mp  = ponto de correspondencia
    kp1, kp2, mp = None, None, None
    counter = 0
    inicio = time.time()
   

    for file in[file for file in os.listdir("SOCOFing/Testes/Real/")]:
        if counter % 2 == 0:
            print(f"{counter} - {file}")
        counter += 1
        
        fingerprint_image = cv2.imread("SOCOFing/Testes/Real/"+file)
        fingerprint_image_bw = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)
        #Iniciação do SIFT
        sift = cv2.SIFT_create()
        # Encontre keypoints nas duas imagens
        keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

    # Crie o matcher Flann Based Matcher (KNN)
        matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10},
                                        {}).knnMatch(descriptors_1, descriptors_2, k=2)
        match_points = []
        fim = time.time()
        #Selecionando o melhor match
        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                match_points.append(p)
        keypoints = 0
        if len(keypoints_1) < len(keypoints_2):
            keypoints = len(keypoints_1)
        else:
            keypoints = len(keypoints_2)

        if len(match_points)/keypoints*100 > best_score:
            best_score = len(match_points)/keypoints*100
            filename = file
            image = fingerprint_image
            kp1, kp2, mp = keypoints_1, keypoints_2, match_points
    tempo = fim-inicio
    b_score = str(best_score)
    # result = cv2.drawMatches(sample, kp1, image, kp2, mp, None),
    # result = cv2.resize(result, None, fx=4, fy=4)
    resultado = [entrada,filename,b_score,tempo]
    resultados.append(resultado)
    # print("Best Match:" + filename),
    # print("Score"+str(best_score)),
    # print(fim-inicio), result = cv2.drawMatches(sample, kp1, image, kp2, mp, None),
    # result = cv2.resize(result, None, fx=4, fy=4)
    #cv2.imshow("Result", resultados)
    # print("Results", resultado)
    # cv2.waitKey(5)
    # cv2.destroyAllWindows()
print("Resultados", resultados,"\n")
cv2.waitKey(5)
cv2.destroyAllWindows()
        
