# cmake yes
# dlib yes
# face-recognition yes
# numpy yes
# opencv-python yes
# wheel yes


import cv2
import face_recognition as fr
import numpy as np


# Cargar la imagen de referencia y aprender a reconocerla.
imagen_referencia = fr.load_image_file("pictures/Foto-Brayan-1.jpg")
imagen_prueba = fr.load_image_file("pictures/Foto-Brayan-2.jpg")


# Convertir las imágenes de BGR (OpenCV) a RGB (face_recognition)
imagen_referencia = cv2.cvtColor(imagen_referencia, cv2.COLOR_BGR2RGB)
imagen_prueba = cv2.cvtColor(imagen_prueba, cv2.COLOR_BGR2RGB)


# Localizar la cara de referencia
lugar_cara_referencia = fr.face_locations(imagen_referencia)[0]
# Codificar la cara control
codificacion_cara_referencia = fr.face_encodings(imagen_referencia)[0]


# Localizar la cara de prueba
lugar_cara_prueba = fr.face_locations(imagen_prueba)[0]
# Codificar la cara de prueba
codificacion_cara_prueba = fr.face_encodings(imagen_prueba)[0]


# Mostrar un rectángulo alrededor de la cara de referencia
cv2.rectangle(imagen_referencia, (lugar_cara_referencia[3], lugar_cara_referencia[0]),
                                 (lugar_cara_referencia[1], lugar_cara_referencia[2]),
                                 (0, 255, 0), 2)


# Mostrar un rectángulo alrededor de la cara de prueba
cv2.rectangle(imagen_prueba, (lugar_cara_prueba[3], lugar_cara_prueba[0]),
                             (lugar_cara_prueba[1], lugar_cara_prueba[2]),
                             (0, 255, 0), 2)


# Comparar las caras
resultado = fr.compare_faces([codificacion_cara_referencia], codificacion_cara_prueba)
# Calcular la distancia entre las caras
distancia = fr.face_distance([codificacion_cara_referencia], codificacion_cara_prueba)


# Imprimir el resultado
cv2.putText(imagen_prueba,
            f"Resultado: {resultado[0]} {distancia[0].round(2)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2)


# Redimensionar a la mitad (50%)
# referencia_resized = cv2.resize(imagen_referencia, (0, 0), fx=0.5, fy=0.5)
# prueba_resized = cv2.resize(imagen_prueba, (0, 0), fx=0.2, fy=0.2)
# O a un tamaño específico, ej: 640x480
# referencia_resized = cv2.resize(imagen_referencia, (640, 480))
# prueba_resized = cv2.resize(imagen_prueba, (640, 480))


# Mostrar las imágenes
cv2.imshow("Imagen de referencia", imagen_referencia)
cv2.imshow("Imagen de prueba", imagen_prueba)


# Mantener el programa abierto hasta que se presione una tecla
cv2.waitKey(0)