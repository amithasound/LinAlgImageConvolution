#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import scipy as sp
import itertools as it
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation


# In[43]:


#plt.style.use('dark_background')  # comment out for "light" theme
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (16,8)


# In[44]:


def RGB_convolve(im1, kern):
    im2 = np.empty_like(im1)
    for dim in range(im1.shape[-1]):  # loop over rgb channels
        im2[:, :, dim] = sp.signal.convolve2d(im_data[:, :, dim],
                                              kern,
                                              mode="same",
                                              boundary="symm")
    return im2


# In[45]:


FNAME = "villanova.jpg"

KERNELS = {"Edge Detection 3x3": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
           "Identity 3x3": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
           "Sharpen 3x3": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
           "XtraSharpen 3x3": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
           "Blur 3x3": np.array([[0.11, 0.11, 0.11], [0.11, 0.11, 0.11], [0.11, 0.11, 0.11]]),
           "Gaussian Blur 3x3": np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]),}

kernel_name = "XtraSharpen 3x3"
kernel = KERNELS[kernel_name]


im_data = plt.imread(FNAME).astype(np.float)/255
im_filtered = RGB_convolve(im_data, kernel)


fig, (axL, axR) = plt.subplots(ncols =2, tight_layout = True)
fig.suptitle(kernel_name)

axL.imshow(im_data)
axR.imshow(im_filtered)


# In[46]:


im_filtered2 = RGB_convolve(im_filtered, kernel)
im_filtered3 = RGB_convolve(im_filtered2, kernel)
im_filtered4 = RGB_convolve(im_filtered3, kernel)
im_filtered5 = RGB_convolve(im_filtered4, kernel)
fig, (axL, axR) = plt.subplots(ncols =2, tight_layout = True)
fig.suptitle(kernel_name)

axL.imshow(im_data)
axR.imshow(im_filtered3)


# In[ ]:





# In[ ]:





# In[ ]:




