import h5py
import numpy as np

p = 'E:\\code\\shenduxuexi\\brats-unet-main\\data\\dataset\\BraTS2021_00002_mri_norm2.h5'
h5f = h5py.File(p, 'r')
image = h5f['image'][:]
label = h5f['label'][:]

print('image shape:',image.shape,'\t','label shape',label.shape)
print('label set:',np.unique(label))
