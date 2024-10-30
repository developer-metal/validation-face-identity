"""
Validador de la identidad mediante recocimiento facial
"""

#Importar librerias
import face_recognition as fr
import cv2
import os
import numpy
from datetime import datetime
import sys
import os
from constantes import NOMBRE_FICHERO, RUTA_FICHERO

class ValidationIdentity:
    def __init__(self,nombre_fichero, ruta):
        self.nombre_fichero = nombre_fichero
        self.ruta = ruta

    def codificar(self,imagenes):
        lista_codificada = []
        for imagen in imagenes:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            codificacion = fr.face_encodings(imagen)[0]
            lista_codificada.append(codificacion)
    
        return lista_codificada



    def captureVideoReal(self, lista_empleados_codificados, nombres_empleados):
        try:
                captura = cv2.VideoCapture(0) # indice de camara , ver con ls -l /dev/video*
                print("Iniciando camara / Por favor espere...")
                #leer imagen de la camara
                exito, imagen = captura.read()
                if not exito:
                   print("No se puede acceder a la cámara")
                   exit()
                else:
                    #reconocer cara en captura
                    cara_captura = fr.face_locations(imagen)
                if len(cara_captura) == 0:
                    print(f"No se ha detectado un rostro en la imagen")
                    exit()
        	
                cara_captura_codificada = fr.face_encodings(imagen, cara_captura)
                # comparar caras
                for caracodif,caraubic in zip(cara_captura_codificada, cara_captura):
                    coincidencias = fr.compare_faces(lista_empleados_codificados, caracodif) #comparar caras
                    distancias = fr.face_distance(lista_empleados_codificados, caracodif) #distancia entre caras
                    indice_coincidencia = numpy.argmin(distancias)
            
        
                #mostrar coincidencias
                if distancias[indice_coincidencia] > 0.6:
                   print(f"No coincide con ninguno de nuestros empleados [Distancia: {round(distancias[indice_coincidencia],2)}]")
                   exit()
           
                else:
                   nombre = nombres_empleados[indice_coincidencia]
                   texto_salida = "Presiona cualquier tecla para salir"
                   y1, x2, y2, x1 = caraubic
                   cv2.rectangle(imagen, (x1,y1 - 40), (x2,y2), (0,255,0),2)
                   rect_height = 35
                   cv2.rectangle(imagen, (x1,y2 + 0), (x2,y2 + rect_height), (0,255,0), cv2.FILLED)
            
                   cv2.putText(imagen,texto_salida,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,cv2.LINE_AA)
            
                   (nombre_ancho,nombre_alto), _ = cv2.getTextSize(nombre,cv2.FONT_HERSHEY_SIMPLEX,0.6,1)
            
                   nombre_x = x1 + (x2 - x1 - nombre_ancho) // 2
                   nombre_y = y2 + (rect_height + nombre_alto) // 2
                   cv2.putText(imagen, nombre, (nombre_x, nombre_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                   print("camera success")
                   self.registrar_ingresos(nombre) # log de empleados
                   cv2.imshow('Imagen Empleado',imagen)
                   cv2.waitKey(0)
                   captura.release()
                   cv2.destroyAllWindows()
        except:
             print(f"Error al validar rostro")
           
            
            
            


    def principal_recognition_image(self):
        ruta = self.ruta
        if not os.path.isdir(ruta):
           print(f"Error: El directorio '{ruta} no existe.'")
           exit()
        
        mis_imagenes = []
        nombres_empleados = []
        lista_empleados = os.listdir(ruta)
        for nombre in lista_empleados:
            imagen_actual = cv2.imread(f'{ruta}/{nombre}')
            if imagen_actual is not None:
               mis_imagenes.append(imagen_actual)
               nombres_empleados.append(os.path.splitext(nombre)[0])
            else:
                print("No se pudo leer la imagen")
                
            
            
        lista_empleados_codificados = self.codificar(mis_imagenes)
        self.captureVideoReal(lista_empleados_codificados, nombres_empleados)




    def registrar_ingresos(self, persona):
        f = open(self.nombre_fichero,'a+')
        lista_datos = f.readlines()
        nombres_registro = []
        for linea in lista_datos:
            ingreso = linea.split(',')
            nombres_registro.append(ingreso[0])
            
            
        if  persona not in nombres_registro:
            hora = datetime.now()
            hora_formato = hora.strftime("%H:%M:%S")
            f.writelines(f"\n{persona}, {hora_formato}")



if __name__ == '__main__':
    empleado = ValidationIdentity(NOMBRE_FICHERO, RUTA_FICHERO)
    empleado.principal_recognition_image()
    
    
    
 
