import numpy as np
import cv2
from shapely.geometry import Polygon
import os

def mask_to_polygons(mask_path):
    '''
    Convierte una máscara de imagen en polígonos. Devuelve dos listas:
    - Lista de polígonos de shapely sin normalizar
    - Lista de polígonos de shapely normalizados (coordenadas entre 0 y 1)

    Args:
        img_path (str): Ruta al archivo de imagen original.
        mask_path (str): Ruta al archivo de la máscara en escala de grises.
    '''
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Calcula los contornos 
    mask = mask.astype(bool)
    
    #contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # convertimos los contornos a polígonos de Label Studio
    polygons = []
    normalized_polygons = []
    for contour in contours:
        
        # Lo meto en un try porque la extraccion de polígonos que hace el opencv a partir de la máscara
        # a veces genera polígonos de menos de 4 vértices, que no tiene sentido por no ser cerrados, 
        # provocando que falle al convertir a polígno de shapely

        try:
            polygon = contour.reshape(-1, 2).tolist()
          
            # normalizamos las coordenadas entre 0 y 1 porque así lo requiere YOLOv8
            normalized_polygon = [[round(coord[0] / mask.shape[1] , 4), round(coord[1] / mask.shape[0] , 4)] for coord in polygon]
        
            # Convertimos a objeto poligono de shapely (sin normalizar)
            polygon_shapely = Polygon(polygon)
            simplified_polygon = polygon_shapely.simplify(0.85, preserve_topology=True)
            polygons.append(simplified_polygon)

            # normalizdos
            normalized_polygons.append(Polygon(normalized_polygon))
          

        except Exception as e:
            pass
        

    return polygons, normalized_polygons

def main():
    # input_dir = './tmp/val_masks'
    # output_dir = './tmp/val_labels_v3'
    # for j in os.listdir(input_dir):
    #     mask_path = os.path.join(input_dir, j)
    #     cv2.imwrite("test_mask.png", mask_generated)
    #     with open('{}.txt'.format(os.path.join(output_dir, j)[:-4]), 'w') as f:
    #         f.write(label_str)
    #         f.close()
        # with open('mask.txt', "w") as f:

    mask_path = './tmp/val_masks/10000.png'
    label_str, mask_generated = mask_to_polygons(mask_path)
    print(label_str, mask_generated)
main()
