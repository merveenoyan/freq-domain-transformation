import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fp
import cv2
from PIL import Image


# ideal filter
def ideal_filter(rows, cols, D0, filtr):
    H = np.zeros(shape = (rows, cols))
    for i in range(rows):
        for j in range(cols):
            # euclidean distance from u,v to origin of frequency

            Duv = np.sqrt(np.power(i - rows/2, 2) + np.power(j - cols/2, 2)) 
            if Duv < D0:
                H[i,j] = 1.0
    if filtr == "High Pass":
        H = 1-H
    #H = H*255
    cv2.imwrite('filter.jpg',np.abs(H*255))
    return H

def butterworth_filter(rows, cols, n_order, D0, filtr):
    H = np.zeros(shape = (rows, cols))
    for i in range(rows):
        for j in range(cols):
        
            Duv = np.sqrt(np.power(i - rows/2, 2) + np.power(j - cols/2, 2))
            H[i,j] = 1/(1+((Duv/D0)**(2*n_order)))
    if filtr == "High Pass":
        H = 1-H
    
    cv2.imwrite('filter.jpg',np.abs(H*255))
    
    return H

def gaussian_filter(rows, cols, filtr):

    H = np.zeros(shape = (rows, cols))
    for i in range(rows):
        for j in range(cols):
            Duv = np.sqrt(np.power(i - rows/2, 2) + np.power(j - cols/2, 2))
            H[i,j] = np.exp(-((Duv**2)/(2*(D0**2))))
    if filtr == "High Pass":
        H = 1-H
    #H = H*255
    cv2.imwrite('filter.jpg',np.abs(H*255))
    return H

def calculate_distance(rows, cols):
    
    dist =np.zeros((rows,cols))
    u=np.arange(0, rows, 1)
    v=np.arange(0, cols, 1)
    
    for i in range(rows):
        for j in range(cols):
            dist[i,j]=np.sqrt(((u[i]-rows/2)**2)+((v[j]-cols/2)**2))
    dist = np.float32(dist)
    return dist


if __name__ == "__main__":
    #image = Image.open(file).convert("L")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    uploaded_file = st.sidebar.file_uploader("Upload image", type = ["jpeg", "jpg", "png"])
    filtr = st.sidebar.radio("Filters", ("Low Pass", "High Pass"))
    kernel = st.sidebar.radio("Kernels", ("Ideal", "Butterworth", "Gaussian"))
    D0 = st.sidebar.slider("Cutoff Frequency", min_value = 0, max_value = 120)
    n_order = st.sidebar.number_input(label = "Order", min_value = 0, max_value = 5)

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img.save("read_image.jpg")
        st.subheader("Source Image")
        st.image("read_image.jpg", width = 300)
        img = cv2.imread("read_image.jpg", 0)
        rows, cols = img.shape
        if kernel == "Ideal":
            k = ideal_filter(rows, cols, D0, filtr)
        elif kernel == "Gaussian":
            k = gaussian_filter(rows, cols, filtr)
        elif kernel == "Butterworth":
            k = butterworth_filter(rows, cols, n_order, D0, filtr)
        
        H = fp.fft2(fp.ifftshift(k)) # fast fourier transform
        f_img = fp.fft2(img) # fast fourier transform
        conv_img = np.multiply(H, f_img)

        inv_img = fp.ifft2(conv_img).real

        output_img = ((inv_img - np.min(inv_img))/np.max(inv_img))*255

        #output_img = fp.ifft2(conv_img)

        cv2.imwrite('output_image.jpg',output_img)
        st.subheader(f"Target Image with {filtr} {kernel} Filter, and Filter itself")
        st.image(["filter.jpg", "output_image.jpg"], width = 320)
        dist = calculate_distance(rows, cols)
        st.subheader("Graph of Distance against Function")
        plt.plot(dist.ravel(), k.ravel())
        plt.xlabel('Distance from the Center')
        plt.ylabel('Filter Function')
        st.pyplot()








    


