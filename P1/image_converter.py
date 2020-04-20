from PIL import Image
import numpy as np 

def img_to_numpy(path):
    image = np.asarray(Image.open(path))
    finArr = []
    for row in image:
        for pixel in row:
            finArr.append(pixel[0]/254)

    return np.array(finArr)


if __name__ == "__main__":
    a = img_to_numpy(path = "P1/pict1.jpg")
    print(a)
    pass    
    