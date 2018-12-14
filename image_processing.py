import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import math
import random

def normal():
    img = Image.open("static/img/normal_img.jpg")
    img.save(("static/img/temp_img.jpg"))

def grayscale():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")

    img_arr = np.asarray(img)
    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]
    arr_gray=r.astype(int)+g.astype(int)+b.astype(int)
    arr_gray=(arr_gray/3).astype("uint8")

    img_new = Image.fromarray(arr_gray)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def zoomin():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")

    img_arr = np.asarray(img)
    new_size = ((img_arr.shape[0] * 2),(img_arr.shape[1] * 2), img_arr.shape[2])
    new_arr = np.full(new_size, 255)
    new_arr.setflags(write=1)

    r = img_arr[:,:,0]
    g = img_arr[:,:,1]
    b = img_arr[:,:,2]
    
    new_r = []
    new_g = []
    
    new_b = []

    for row in range(len(r)):
        temp_r=[]
        temp_g=[]
        temp_b=[]
        for i in r[row]:
            temp_r.extend([i,i])
        for j in g[row]:
            temp_g.extend([j,j])
        for k in b[row]:
            temp_b.extend([k,k])
        for _ in (0,1):
            new_r.append(temp_r)
            new_g.append(temp_g)
            new_b.append(temp_b)
    
    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            new_arr[i,j,0]=new_r[i][j]
            new_arr[i,j,1]=new_g[i][j]
            new_arr[i,j,2]=new_b[i][j]

    new_arr = np.uint8(new_arr)
    img_new = Image.fromarray(new_arr)
    img_new.save("static/img/temp_img.jpg")


def zoomout():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")
    h_img,w_img = img.size
    img_arr = np.asarray(img)
    new_arr = np.zeros((int(h_img/ 2), int(w_img/ 2),3))
    for i in range(int(h_img/2)):
        for j in range(int(w_img/2)):
            for channel in range(3):
                temp=[]
                temp.append(img_arr[2 * i, 2 * j,channel])
                temp.append(img_arr[2 * i + 1, 2 * j,channel])
                temp.append(img_arr[2 * i, 2 * j + 1,channel])
                temp.append(img_arr[2 * i + 1, 2 * j + 1,channel])
                new_arr[i,j,channel]=int(np.sum(temp)/4)
    new_arr = np.uint8(new_arr)
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def moveleft():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")

    img_arr = np.asarray(img)
    r,g,b = img_arr[:,:,0],img_arr[:,:,1],img_arr[:,:,2]

    r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
    g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
    b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
    new_arr = np.dstack((r, g, b))
    
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")
    # img_new.save("static/img/temp_img.jpg")

def moveright():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")

    img_arr = np.asarray(img)
    r,g,b = img_arr[:,:,0],img_arr[:,:,1],img_arr[:,:,2]
    r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
    g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
    b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
    new_arr = np.dstack((r, g, b))
    
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def moveup():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")

    img_arr = np.asarray(img)
    new_arr = np.full((img_arr.shape),0)

    img_arr = np.asarray(img)
    r,g,b = img_arr[:,:,0],img_arr[:,:,1],img_arr[:,:,2]

    r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
    g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
    b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
    new_arr = np.dstack((r, g, b))
    
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def movedown():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")

    img_arr = np.asarray(img)
    r,g,b = img_arr[:,:,0],img_arr[:,:,1],img_arr[:,:,2]

    r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    b = np.pad(b, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    new_arr = np.dstack((r, g, b))
    
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def brightplus():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")

    img_arr = np.asarray(img,dtype=np.int)
    new_arr = img_arr+60
    new_arr = np.clip (new_arr,0,255)
    new_arr = np.uint8(new_arr)
    
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def brightsubs():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")

    img_arr = np.asarray(img,dtype=np.int)
    new_arr = img_arr-60
    new_arr = np.clip (new_arr,0,255)
    new_arr = np.uint8(new_arr)
    
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def brightmulti():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")

    img_arr = np.asarray(img,dtype=np.int)
    new_arr = img_arr*1.5
    new_arr = np.clip (new_arr,0,255)
    new_arr = np.uint8(new_arr)
    
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def brightdiv():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")

    img_arr = np.asarray(img,dtype=np.int)
    new_arr = img_arr//1.5
    new_arr = np.clip (new_arr,0,255)
    new_arr = np.uint8(new_arr)
    
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def view_histogram():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")

    img_arr = np.asarray(img)
    warna = ['red', 'green', 'blue']
    for i in range(3):
        unique, count = np.unique(img_arr[:,:,i], return_counts=True)
        plt.bar(unique, count, color=warna[i])
        plt.savefig(f'static/img/temp_img_{warna[i]}_histogram.jpg', dpi=300)
        plt.clf()
        
def hist_eq():
    grayscale()
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img)
    img_arr.setflags(write=1)
    def hist_eq(arr):
        histo = [0]*256
        for i in range(len(arr)):
            for j in arr[i]:
                histo[j] += 1
        cdf = [0] * len(histo)   
        cdf[0] = histo[0]
        for i in range(1, len(histo)):
            cdf[i]= cdf[i-1]+histo[i]
        cdf = [ele*255/cdf[-1] for ele in cdf]
        cdf_fix=[math.floor(cd) for cd in cdf]
        new={}
        for i in range(256):
            new[i]=cdf[i]
        for i in range(len(arr)):
            for idx,val in enumerate(arr[i]):
                arr[i][idx] = new[val]
        return arr
    new_arr = hist_eq(img_arr[:,:,0])
    img_arr = np.dstack((new_arr,new_arr,new_arr))
    img_new = Image.fromarray(img_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def convolute(img, kernel):
    h_img, w_img, _= img.shape
    out = np.zeros((h_img-2,w_img-2),dtype=np.float)
    new_img = np.zeros((h_img-2,w_img-2,3))
    if np.array_equal((img[:,:,1],img[:,:,0]),img[:,:,2])==True:
        array = img[:,:,0]
        for h in range(h_img-2):
            for w in range(w_img-2):
                S = np.multiply(array[h:h+3,w:w+3],kernel)
                out[h,w] = np.sum(S)
        out_ = np.clip(out,0,255)
        new_img=np.dstack((out_,out_,out_))
    else:
        for channel in range(3):
            array = img[:,:,channel]
            for h in range(h_img-2):
                for w in range(w_img-2):
                    S = np.multiply(array[h:h+3,w:w+3],kernel)
                    out[h,w] = np.sum(S)
            out_ = np.clip(out,0,255)
            new_img[:,:,channel]=out_
    new_img=np.uint8(new_img) 
    return new_img

def blur():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img,dtype=np.int)
    kernel = np.array([[0.0625,0.125,0.0625],[0.125, 0.25, 0.125],[0.0625,0.125,0.0625]])
    new_arr = convolute(img_arr,kernel) 
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def sharp():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img,dtype=np.int)
    kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
    new_arr = convolute(img_arr,kernel)  
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def edge():
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img,dtype=np.int)
    kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
    new_arr = convolute(img_arr,kernel)    
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def threshold_seg(img_arr, red, green, blue, segmen_color):
    new_arr = np.zeros((img_arr.shape))
    h_arr, w_arr, _ = img_arr.shape
    for h in range(h_arr):
        for w in range(w_arr):
            if red[0]<=img_arr[h,w,0]<=red[1] and green[0]<=img_arr[h,w,1]<=green[1] and blue[0]<=img_arr[h,w,2]<=blue[1]:
                new_arr[h,w,0]=segmen_color[0]
                new_arr[h,w,1]=segmen_color[1]
                new_arr[h,w,2]=segmen_color[2]
            else:
                new_arr[h,w,:]=img_arr[h,w,:]
    return new_arr

def threshold(red, green, blue, segmen_color):
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img,dtype=np.int)
    new_arr = threshold_seg(img_arr, red, green, blue, segmen_color)
    new_arr = np.uint8(new_arr)    
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")

def region_growth(seed, threshold_seed):
    grayscale()
    img = Image.open("static/img/temp_img.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img,dtype=np.int)
    h, w, _ = img_arr.shape

    def safe_index(x, y):
        # global w,h
        if(x < w and x >= 0 and y < h and y >= 0):
            return True
        else:
            return False

    def cek_tetangga(pixels_env, x, y):
        # global minimum, maximum
        neighbors = []
        up = (x, y-1)
        right = (x+1, y)
        left = (x-1, y)
        down = (x, y+1)
        if not safe_index(x, y-1):
            status[up]=True
        if not safe_index(x+1, y):
            status[right]=True
        if not safe_index(x-1, y):
            status[left]=True
        if not safe_index(x, y+1):
            status[down]=True
            
        if (status[up]==False and minimum<pixels_env[up]<maximum):
            neighbors.append(up)
        if (status[right]==False and minimum<pixels_env[right]<maximum):
            neighbors.append(right)
        if (status[left]==False and minimum<pixels_env[left]<maximum):
            neighbors.append(left)
        if (status[down]==False and minimum<pixels_env[down]<maximum):
            neighbors.append(down)
        
        if(len(neighbors) > 0):
            r = random.randint(0, len(neighbors)-1)
            return neighbors[r]
        else:
            return None

    def proses(pixels_env, seedX, seedY):
        still_searching = True
        stack = []
        selected_color= 255
        current_pixel=(seedX,seedY)
        while still_searching:
            status[current_pixel]=True
            next_pixel = cek_tetangga(pixels_env, current_pixel[0], current_pixel[1])
            if(next_pixel is not None):
                pixels_env[next_pixel]=selected_color
                stack.append(current_pixel)
                current_pixel=next_pixel
            elif (len(stack) > 0):
                current_pixel = stack[-1]
                stack.pop(-1)
            else:
                still_searching = False
        return pixels_env

    seedX = seed[0]
    seedY = seed[1]

    pixels=[]
    for i in range(h):
        for j in range(w):
            pixels.append((i,j))

    pixels_env={}
    status={}
    for pixel in pixels:
        pixels_env[pixel] = img_arr[pixel[0],pixel[1],0]
        status[pixel] = False

    current_color = pixels_env[seedX,seedY]
    minimum = current_color - threshold_seed
    maximum = current_color + threshold_seed

    new_env = proses(pixels_env, seedX, seedY)
    new_arr = np.zeros((img_arr.shape))
    for pixel in pixels:
        new_arr[pixel[0],pixel[1],0]=new_env[pixel]
        new_arr[pixel[0],pixel[1],1]=new_env[pixel]
        new_arr[pixel[0],pixel[1],2]=new_env[pixel]
    new_arr = np.uint8(new_arr)    
    img_new = Image.fromarray(new_arr)
    img_new = img_new.convert("RGB")
    img_new.save("static/img/temp_img.jpg")




    