from turtle import width
from matplotlib.pyplot import fill
import pandas as pd
import numpy as np
from random import random, choice, randint
from PIL import Image, ImageDraw
from tqdm import tqdm
import os
import argparse

def random_color(clara=True):
    if (clara):
        return (randint(150, 255), randint(150, 255), randint(150, 255))
    else:
        return (randint(0, 150), randint(0, 150), randint(0, 150))

def criar_imagem(img_path):
    imgw, imgh = randint(200, 300), randint(200, 300)
    img_pil = Image.new('RGB', (imgw, imgh), color='white')

    draw = ImageDraw.Draw(img_pil)
    for _ in range(25):
        x0, y0, x1, y1 = randint(0, imgw), randint(0, imgh), randint(0, imgw), randint(0, imgh)
        draw.line([x0, y0, x1, y1], fill=random_color(), width=1)
    
    anotacoes = []
    for n in range(2):

        w, h = randint(30, 60), randint(30, 60)
        x0, y0 = randint(0, imgw-w), randint(0, imgh-h)
        x1, y1 = x0+w, y0+h

        xc = x0 + w/2
        yc = y0 + h/2

        tipo = choice([0, 1]) # [vazio, circulo, triangulo]

        anotacao = [img_path, imgw, imgh, xc/imgw, yc/imgh, w/imgw, h/imgh, tipo]
        
        if (tipo == 0):
            draw.ellipse([x0, y0, x1, y1], fill=random_color(False), outline=random_color(False), width=2)
        elif (tipo == 1):
            p0, p1, p2 = (x0, y1), (x1, y1), (x0+w/2, y0)
            draw.polygon([p0, p1, p2], fill=random_color(False), outline=random_color(False))
        
        anotacoes.append(anotacao)
        
    return img_pil, anotacoes

def criar_dataset(n):

    df = []
    for k in tqdm(range(n)):
        nome = f'./imgs/img{k:05}.jpg'
        img_pil, anotacoes = criar_imagem(nome)
        img_pil.save(nome)
        df.append(np.array(anotacoes, dtype=np.object))
    
    df = np.array(df, dtype=np.object)
    df = df.reshape((-1, 8))

    df = pd.DataFrame(df, columns=['img_path', 'imgw', 'imgh', 'xc', 'yc', 'w', 'h', 'tipo'])
    df.to_csv('annotations.csv', index=False)

if (__name__ == '__main__'):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int,  help='numero de imagems do dataset')

    args = parser.parse_args()

    criar_dataset(args.n)

    print (f'fim')