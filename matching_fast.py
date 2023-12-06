# -*- coding: utf-8 -*-
"""Matching FAST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WM80sV1AG81oTK5gOw1diFXSdJ69OuEO
"""

# importing required libraries SIFT
import os
import time
from unittest import result
import cv2


"""# Lendo impressão digital do dataset"""

# ler a impressão latente SIFT
resultados = []
entrada = None
contadorTeste = 0
for inicio in[inicio for inicio in os.listdir("SOCOFing/Testes/medium/")]:
    sample = cv2.imread("SOCOFing/Testes/medium/"+inicio)
    sample = cv2.resize(sample, None, fx=4, fy=4)
    sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    entrada = inicio
   
    print(entrada)

    FLANN_INDEX_LSH = 6
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

    files = [file for file in os.listdir("SOCOFing/Testes/Real/")]
    counter = 0
    inicio = time.time()

    for file in files:
        if counter % 2 == 0:
            print(f"{counter} - {file}")

        counter += 1

        fingerprint_image = cv2.imread("SOCOFing/Testes/Real/"+file)
        fingerprint_image_bw = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)

        # Criação do detector FAST
        fast = cv2.FastFeatureDetector_create()

        # Encontra as keypoints nas duas imagens
        kp1 = fast.detect(sample_gray, None)
        kp2 = fast.detect(fingerprint_image_bw, None)

        # Criação do objeto descritor BRIEF(Binary Robust Independent Elementary Features )
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        # Computa os descritores BRIEF
        keypoints_1, descriptors_1 = brief.compute(sample_gray, kp1)
        keypoints_2, descriptors_2 = brief.compute(fingerprint_image_bw, kp2)

        index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2

        search_params = dict(checks = 50)
        matches = cv2.FlannBasedMatcher(index_params, search_params).knnMatch(descriptors_1, descriptors_2, k=2)

        match_points = []
        fim = time.time()
        #Selecionando o melhor match
        for m_n in matches:
            if len(m_n) != 2:
                continue
            (p,q) = m_n

            if p.distance < 0.6*q.distance:
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
    contadorTeste +=1
    tempo = fim-inicio
    b_score = str(best_score)
    resultado = [entrada,filename,b_score,tempo]
    resultados.append(resultado)
   
    # print("Best Match: " + str(filename))
    # print("Score: "+str(best_score))
    # print("Tempo de processamento: " + str(fim-inicio))
    # result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
    # result = cv2.resize(result, None, fx=4, fy=4)
    
print("Resultados", resultados,"\n")
print("Quantos dados foram inseridos para teste: ",contadorTeste)
cv2.waitKey(5)
cv2.destroyAllWindows()