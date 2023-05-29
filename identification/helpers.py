import cv2
from sklearn.cluster import KMeans
from webcolors import hex_to_rgb
from scipy.spatial import KDTree
import numpy as np

def get_color(image, k=3):
    # converte a imagem para RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # redimensiona a imagem para 2 dimensoes
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)

    # aplica o algoritmo kmens aos pixeis da imagem
    kmeans.fit(pixels)

    # calcula a media dos pixeis ao logo de X (RGB)
    rgb_arithmetic_mean = np.mean(pixels, axis=0)
    # captura o cluster da cor media
    predicted_cluster_mean = kmeans.predict([rgb_arithmetic_mean])[0]
    # captura as coordenadas (RGB) do cluster da cor media (a que predomina)
    rgb_dominant_coords = kmeans.cluster_centers_[predicted_cluster_mean]
    rgb_color = rgb_dominant_coords.astype(int)

    return rgb_color

def find_closest_rgb_name(rgb_tuple, colors_dic):
    colors_distance_dic = {}

    for color_name, color_in_hex in colors_dic.items():
        red, green, blue = hex_to_rgb(color_in_hex)
        
        # Calcula o quadrado da distancia de cada componente da cor do colors_dic para cor recebida
        rd = (red - rgb_tuple[0]) ** 2
        gd = (green - rgb_tuple[1]) ** 2
        bd = (blue - rgb_tuple[2]) ** 2
        
        # adiciona no dicionario colors_distance_dic qual a distancia de cada cor do dicionario colors_dic
        colors_distance_dic[(rd + gd + bd)] = color_name
    
    # captura a cor com a menor distancia
    closest_color = colors_distance_dic[min(colors_distance_dic.keys())]
    return closest_color.replace("xkcd:", "").replace("tab:", "")