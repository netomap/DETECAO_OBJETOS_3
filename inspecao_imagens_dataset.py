from PIL import Image, ImageDraw
import random
import pandas as pd

if __name__ == '__main__':

    df = pd.read_csv('annotations.csv')
    img_path = random.choice(df['img_path'].values)

    anotacoes = df[df['img_path'] == img_path].values

    img_pil = Image.open(img_path)
    draw = ImageDraw.Draw(img_pil)

    for img_path, imgw, imgh, xc, yc, w, h, tipo in anotacoes:
        xc, yc, w, h = xc*imgw, yc*imgh, w*imgw, h*imgh
        x0, y0, x1, y1 = xc-w/2, yc-h/2, xc+w/2, yc+h/2
        draw.rectangle([x0, y0, x1, y1], fill=None, outline='red', width=2)
        draw.text((x0, y0-15), str(tipo), fill='red')
    
    img_pil.show()