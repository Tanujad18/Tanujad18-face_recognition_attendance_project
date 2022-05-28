# -*- coding: utf-8 -*-
"""
Created on Tue May 17 18:39:12 2022

@author: tanuj
"""

import cv2
import face_recognition

imgElon = face_recognition.load_image_file('ImagesBasic/Elon Musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('ImagesBasic/Bill gates.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

imgTanuja = face_recognition.load_image_file('ImagesBasic/Tanuja Dhaybar.jpg')
imgTanuja = cv2.cvtColor(imgTanuja, cv2.COLOR_BGR2RGB)

imgAnushka = face_recognition.load_image_file('ImagesBasic/Anushka Shelke.jpg')
imgAnushka = cv2.cvtColor(imgAnushka, cv2.COLOR_BGR2RGB)

imgRiya = face_recognition.load_image_file('ImagesBasic/Riya Dunung.jpg')
imgRiya = cv2.cvtColor(imgRiya, cv2.COLOR_BGR2RGB)

imgSam = face_recognition.load_image_file('ImagesBasic/Samruddhi Sawant.jpg')
imgSam = cv2.cvtColor(imgSam, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)


faceTanuja = face_recognition.face_locations(imgTanuja)[0]
encodeTanuja = face_recognition.face_encodings(imgTanuja)[0]
cv2.rectangle(imgTanuja, (faceTanuja[3], faceTanuja[0]), (faceTanuja[1], faceTanuja[2]), (255, 0, 255), 2)

faceAnushka = face_recognition.face_locations(imgAnushka)[0]
encodeAnushka = face_recognition.face_encodings(imgAnushka)[0]
cv2.rectangle(imgAnushka, (faceAnushka[3], faceAnushka[0]), (faceAnushka[1], faceAnushka[2]), (255, 0, 255), 2)

faceRiya = face_recognition.face_locations(imgRiya)[0]
encodeRiya = face_recognition.face_encodings(imgRiya)[0]
cv2.rectangle(imgRiya, (faceRiya[3], faceRiya[0]), (faceRiya[1], faceRiya[2]), (255, 0, 255), 2)

faceSam = face_recognition.face_locations(imgSam)[0]
encodeSam = face_recognition.face_encodings(imgSam)[0]
cv2.rectangle(imgSam, (faceSam[3], faceSam[0]), (faceSam[1], faceSam[2]), (255, 0, 255), 2)

results1 = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis1 = face_recognition.face_distance([encodeElon], encodeTest)
print(results1, faceDis1)

results2 = face_recognition.compare_faces([encodeTest], encodeTanuja)
faceDis2 = face_recognition.face_distance([encodeTest], encodeTanuja)
print(results2, faceDis2)

results3 = face_recognition.compare_faces([encodeTanuja], encodeAnushka)
faceDis3 = face_recognition.face_distance([encodeTanuja], encodeAnushka)
print(results3, faceDis3)

results4 = face_recognition.compare_faces([encodeAnushka], encodeRiya)
faceDis4 = face_recognition.face_distance([encodeAnushka], encodeRiya)
print(results4, faceDis4)

results5 = face_recognition.compare_faces([encodeAnushka], encodeSam)
faceDis5 = face_recognition.face_distance([encodeAnushka], encodeSam)
print(results5, faceDis5)

cv2.putText(imgTest, f'{results1} {round(faceDis1[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.putText(imgTanuja, f'{results2} {round(faceDis2[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.putText(imgAnushka, f'{results3} {round(faceDis3[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.putText(imgRiya, f'{results4} {round(faceDis4[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.putText(imgSam, f'{results4} {round(faceDis4[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Elon Musk Test', imgElon)
cv2.imshow('Bill gate Test', imgTest)
cv2.imshow('Tanuja Dhaybar Test', imgTanuja)
cv2.imshow('Anushka Shelke Test', imgAnushka)
cv2.imshow('Riya Dunung Test', imgRiya)
cv2.imshow('Samruddhi Sawant Test', imgSam)
cv2.waitKey(0)