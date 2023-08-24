import streamlit as st
from PIL import Image
import pandas as pd
import os
import imageio
import scipy.ndimage as ndi
from scipy.ndimage import uniform_filter
import skimage
import numpy as np
from math import log10, sqrt
import math
from skimage import img_as_ubyte

def load_image(image):
    img = Image.open(image)
    return img

def save_file(file):
    with open(os.path.join("tempDir", file.name), "wb") as save_file:
                save_file.write(file.getbuffer())
    return st.success("File Save")

def info_image(image):
    st.text({"dtype":image.dtype, "shape":image.shape, 'min':image.min(), 'max':image.max()})

def mean(image):
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            pixel = image[y,x]
            total += np.sum(pixel)
    m = total/(height*width)
    return m

def enl(image):
    nilai_enl = (mean(image) ** 2) / np.var(image)
    st.write("ENL:", nilai_enl)
    image_psnr = (image_ori - image_filtered) ** 2
    mse = hitung_mean(image_psnr)
    nilai_psnr = 10 * log10((255**2) / mse)
    st.write("PSNR:", nilai_psnr)
    nilai_nm = hitung_mean(image_filtered) / hitung_mean(image_ori)
    st.write("NM", nilai_nm)

def psnr(image_ori, image_filt):
    image_psnr = (image_ori - image_filtered) ** 2
    mse = hitung_mean(image_psnr)
    nilai_psnr = 10 * log10((255**2) / mse)
    st.write("PSNR:", nilai_psnr)

def nm(image_ori, image_filt):
    nilai_nm = hitung_mean(image_filtered) / hitung_mean(image_ori)
    st.write("NM", nilai_nm)

def hist_eq(image):
    histo = ndi.histogram(image, min=0, max=255, bins=256)
    cdf = histo.cumsum() / histo.sum()
    im_eq = cdf[image]
    return img_as_ubyte(im_eq)

from skimage import exposure
def hist_adapteq(image, cl):
    #im_adapteq = exposure.equalize_adapthist(image, clip_limit=cl)
    return img_as_ubyte(exposure.equalize_adapthist(image, clip_limit=cl))

def conts(image, c, d):
    #c = 8#st.number_input("input a low"f"{image.mean}" , 1, 10)#8
    #d = 150
    a =  image.min()
    b = image.max()
    im_conts = ( (image-c) * ((b-a)/(c-d)) ) + a
    im_cs = im_conts/np.amax(im_conts)
    return img_as_ubyte(np.clip(im_cs, 0, 1))

def mean_filter(image, size):
    im = uniform_filter(image, size)
    return im

def median_filter(image, size):
    im = ndi.median_filter(image, size)
    return im

def gaussian_filter(image, sigma):
    im = ndi.gaussian_filter(image, sigma)
    return im

def max_filter(image, size):
    im = ndi.maximum_filter(image, size)
    return im

def min_filter(image, size):
    im = ndi.minimum_filter(image, size)
    return im

def edge_detect(array_vertical, array_horizontal):
    gradient_mag = np.sqrt(array_vertical**2 + array_horizontal**2)
    im_edge = gradient_mag/np.amax(gradient_mag)  
    im_edge = np.clip(im_edge, 0, 255)
    gradient_mag2 = img_as_ubyte(im_edge)
    return gradient_mag2
def mask(image, thd):
    im = image<thd
    im = im * 255
    return im*255

def main():
    menu = ["Home", "Image Enhancement and Edge Detection", "Masking and Morphological Filter", "Dataset", "About"]
    choice = st.sidebar.radio("Menu", menu)
    #st.image("C:/Users\Tazkia\Downloads\pcm streamlit\mem.jpg")
    if choice == "Home":
        st.header("Home")
        st.header("FP Medical Image Processing")
        st.write("Hy, Welcome!")
        #if st.button("hehe"):
        #    st.image(Image.open("C:/Users\Tazkia\Downloads\pcm streamlit\mem.jpg"))
    if choice == "Image Enhancement and Edge Detection":
        st.header("Image Enhancement and Edge Detection")
        uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_files is not None:
            name_list=[]
            image_list=[]
            for uploaded_file in uploaded_files:
                uploaded_image = imageio.imread(uploaded_file) @ [0.2126, 0.7152, 0.0722]
                uploaded_image = img_as_ubyte (uploaded_image / 255)
                image_list.append(uploaded_image)
                name_list.append(uploaded_file.name)
            name_file = name_list
            image_file = image_list
        for i, image in enumerate (image_file):
            st.image(image, caption=name_file[i])
            st.text(info_image(image)) 
        if st.button("Save File"):
            save_file(image_file) 
        box = st.selectbox("Process", ["Enhance and Filter", "Edge Detection"])
        if box == "Enhance and Filter":
            col1, col2 = st.columns(2)
            with col1: #enhance
                st.subheader('Enhancement')
                tab1, tab2, tab3 = st.tabs(["HE", "AHE", "CS"])
                with tab1:
                    name_list_HE = []
                    image_list_HE = []
                    for image in image_file:
                        im_eq = hist_eq(image)
                        st.image(im_eq)
                        st.text(info_image(im_eq))
                        image_list_HE.append(im_eq)
                    image_file_HE = image_list_HE
                with tab2:
                    name_list_AHE = []
                    image_list_AHE = []
                    cliplimit = st.slider('Clip Limit', 0.0, 0.01, 1.0)
                    for image in image_file:
                        im_adapteq = hist_adapteq(image, cliplimit)
                        st.image(im_adapteq)
                        st.text(info_image(im_adapteq))
                        image_list_AHE.append(im_adapteq)
                    image_file_AHE = image_list_AHE
                with tab3:
                    #if st.button("Contrast Stretching"):
                    image_list_cs = []
                    c = st.slider('c conts', 1, 255)
                    d = st.slider('d conts', 1, 255)
                    for image in image_file:
                        #for i in range(len(image_file)):
                        #    c = st.number_input("input a low"f"{i}" , 1, 10)
                        #    d = st.number_input("input a high"f"{i}" , 1, 10)
                        im_cs = conts(image, c, d)
                        st.image(im_cs)
                        #st.image(ndi.histogram(im_cs, min=0, max=255, bins=256)*255)
                        st.text(info_image(im_cs))
                        image_list_cs.append(im_cs)
                    image_file_cs = image_list_cs
            with col2: #filtering
                st.subheader('Filtering')
                inp = st.selectbox('Input Image', ['citra raw', 'citra HE', 'citra AHE', 'citra CS'])
                tab1, tab2, tab3, tab4 = st.tabs(["Mean", "MinMax", "Median", "Gaussian"])
                if inp == "citra raw":
                    image_file_filter = image_file
                elif inp == "citra HE":
                    image_file_filter = image_file_HE
                elif inp == "citra AHE":
                    image_file_filter = image_file_AHE
                elif inp == "citra CS":
                    image_file_filter = image_file_cs

                with tab1:
                    mean_size = st.slider('Mean Filter Size', 1, 10)
                    image_list_mean = []
                    for image in image_file_filter:
                        im_mean = mean_filter(image, mean_size) 
                        st.image(im_mean)
                        st.text(info_image(im_mean))
                        image_list_mean.append(im_mean)
                    
                with tab2:
                    min_filtersize = st.slider('Min Filter Size', 1, 10)
                    max_filtersize = st.slider('Max Filter Size', 1, 10)
                    image_list_min = []
                    image_list_max = []
                    for image in image_file_filter:
                        im_max = max_filter(image, max_filtersize)
                        im_min = min_filter(im_max, min_filtersize)
                        st.image(im_min)
                        st.text(info_image(im_min))
                        image_list_max.append(im_max)
                        image_list_min.append(im_min)

                with tab3:
                    filtersize_med = st.slider('Median Filter Size', 1, 10)
                    image_list_median = []
                    for image in image_file_filter:
                        im_med = median_filter(image, filtersize_med)
                        st.image(im_med)
                        st.text(info_image(im_med))
                        image_list_median.append(im_med)

                with tab4:
                    filtersize_gauss = st.slider('Gaussian Filter Size', 1, 20)
                    image_list_gauss = []
                    for image in image_file_filter:
                        im_gauss = gaussian_filter(image, filtersize_gauss)
                        st.image(im_gauss)
                        st.text(info_image(image))
                        image_list_gauss.append(im_gauss)
        elif box == "Edge Detection":
            col1, col2 = st.columns(2)
            with col1:
                e = st.number_input("Berapa kali Enhancement :", 1, 5)
                cho_im = [e]
                i = 1
                imagem = uploaded_image
                list_image_edge = []
                for imagem in image_file:
                    for i in range(e):
                        cho_im = st.selectbox("Enhancement Methods "f"{i+1}.{len(imagem)}" , ['HE', 'AHE', 'CS', 'Mean', 'Median', 'Gaussian', 'Min-Max'])
                        #for imagem in image_file:
                            #if cho_im == "raw":
                            #    proc = image_file
                        if cho_im == "HE":
                            proc = hist_eq(imagem)
                        elif cho_im == "AHE":
                            cl = st.slider("Clip Limit"f"{i+1}.{len(imagem)}", 0.0, 0.001, 1.0)
                            proc = hist_adapteq(imagem, cl)
                        elif cho_im == "CS":
                            proc = conts(imagem)
                        elif cho_im == "Mean":
                            filter_size = st.slider("Mean Filter Size Edge Detect"f"{i+1}.{len(imagem)}", 1, 10)
                            proc = mean_filter(imagem, filter_size)
                        elif cho_im == "Median":
                            filter_size = st.slider("Median Filter Size"f"{i+1}.{len(imagem)}", 1, 10)
                            proc = median_filter(imagem, filter_size)
                        elif cho_im == "Gaussian":
                            sigma = st.slider("Gaussian Filter Size"f"{i+1}.{len(imagem)}", 1, 10)
                            proc = gaussian_filter(imagem, sigma)
                        elif cho_im == "Min-Max":
                            pro = min_filter(imagem, st.slider("Min Size"f"{i+1}.{len(imagem)}", 1, 20))
                            proc = max_filter(pro, st.slider("Max Size"f"{i+1}.{len(imagem)}", 1, 20))
                        imagem = proc
                        #imagem = img_as_ubyte(imagem)
                        #st.write(info_image(imagem))
                        st.image(imagem)
                        if i == e-1:
                            list_image_edge.append(imagem)
                
            with col2:
                methods = st.selectbox('Metode Edge Detection', ['Roberts', 'Prewitt', 'Sobel', 'LoG'])
                if methods == "Roberts":
                    arr_v = np.array([[-1, 0], [0, 1]])
                elif methods == "Prewitt":
                    arr_v = np.array([[-1, 0, 1],[-1, 0, 1], [-1, 0, 1]])
                elif methods == "Sobel":
                    arr_v = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
                elif methods == "LoG":
                    arr_v = np.array([[0, 1, 0],[1, -4, 1], [0, 1, 0]])
                #elif methods == "Canny":
                #    edges = ndi.generic_gradient_magnitude(imagem, ndi.filters.gaussian_filter, 1.5, mode='nearest')
                #    edges = ndi.morphology.binary_closing(edges)
                #    edges = ndi.morphology.binary_opening(edges)
                #    st.image(edges)
                    #arr_v = np.array(1)
                arr_h = arr_v.T
                #st.image(list_image_edge)
                for image in list_image_edge:
                    im_detect = edge_detect(ndi.convolve(image, arr_v), ndi.convolve(image, arr_h))
                    st.image(im_detect)
                    st.write(info_image(im_detect))
    elif choice == "Masking and Morphological Filter":
        st.header("Masking and Morphological Filter")
        image_masking = st.file_uploader("Upload Images for Assignment 3 (Masking)", type=["png", "jpg", "jpeg"])
        if image_masking is not None:
            imagem = imageio.imread(image_masking) @ [0.2126, 0.7152, 0.0722]
            imagem = imagem/255
            imagem = img_as_ubyte (imagem)
            st.text(info_image(imagem))
            #st.text(info_image(image_masking))
            #st.image(imagem)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Enhancement")
            m = st.number_input('Berapa kali enhancement:', 1, 5)
            cho_im = [m]
            i = 1
            for i in range (m):
                cho_im = st.selectbox("Enhancement Method " f"{i+1}" , ['raw', 'HE', 'AHE', 'CS', 'Mean', 'Median', 'Gaussian', 'Min-Max'])
                if cho_im == "raw":
                    proc = imagem
                elif cho_im == "HE":
                    proc = hist_eq(imagem)
                elif cho_im == "AHE":
                    cl = st.slider('Clip Limit:', 0.0, 0.0005, 1.0)
                    proc = hist_adapteq(imagem, cl)
                elif cho_im == "CS":
                    proc = conts(imagem)
                elif cho_im == "Mean":
                    filter_size = st.slider('Mean Filter Size', 1, 10)
                    proc = mean_filter(imagem, filter_size)
                elif cho_im == "Median":
                    filter_size = st.slider('Median Filter Size', 1, 10)
                    proc = median_filter(imagem, filter_size)
                elif cho_im == "Gaussian":
                    sigma = st.slider("Gaussian Filter Size", 1, 10)
                    proc = gaussian_filter(imagem, sigma)
                elif cho_im == "Min-Max":
                    pro = min_filter(imagem, st.slider("Min Size", 1, 20))
                    proc = max_filter(pro, st.slider("Max Size", 1, 20))
                imagem = proc
                imagem = img_as_ubyte(imagem)
                st.write(info_image(imagem))
                st.image(proc)
        with col2:
            st.subheader("Masking")
            thd = st.slider("Threshold", 0, 255)
            st.write(info_image(imagem))
            st.image(imagem)
            masking = imagem<thd
            masking = masking * 255

            #masking = mask(imagem, thd)
            #mask_bone = np.uint8(imagem < 148.0)
            #mask_skin = mask_bone*1
            #masking = np.where(imagem<148, 1, 0)
            st.image(masking)
            #st.image(np.where(imagem>140, 1, 0))
            st.write(info_image(masking))
            st.write(imagem.shape)
            #st.write(type(masking))
        with col3:
            st.subheader("Morphological Filter")
            n = st.number_input('Jumlah pengaplikasian Morph Filter:', 1, 5)
            morph = [n]
            i = 1
            for i in range (n):
                morph = st.selectbox("Morphologi Filter" f"{i+1}", ["Dilate", "Erose", "Open", "Close"])
                if morph == "Dilate":
                    it = st.slider("iteration Dilate", 1, 7)
                    imm = ndi.binary_dilation(masking, iterations=it)
                elif morph == "Erose":
                    it = st.slider("iteration Erose", 1, 7)
                    imm = ndi.binary_erosion(masking, iterations=it)
                elif morph == "Open":
                    it = st.slider("iteration Open", 1, 7)
                    imm = ndi.binary_opening(masking, iterations=it)
                elif morph == "Close":
                    it = st.slider("iteration Close", 1, 7)
                    imm = ndi.binary_closing(masking, iterations=it)
                masking = imm
                st.image(imm*255)
                if i == n-1:
                    im_mask = imm
            
            #im_edge_detect = np.where(im_mask, imagem, 0)
            #st.image(im_edge_detect)
    if choice == "Dataset":
        st.subheader("Dataset")
        data_file = st.file_uploader("Upload CSV", type=("csv"))
        if data_file is not None:
            st.write(type(data_file))
            file_details = {"filename":data_file.name, "filetype:":data_file.type, "filesize:" :data_file.size}
            st.write(file_details)
            df = pd.read_csv(data_file)
            st.dataframe(df)
    #else:
        #st.subheader("About")


if __name__=='__main__':
    main()

