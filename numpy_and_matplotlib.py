
import numpy as np
import matplotlib.pyplot as plt
import h5py


#Get an empty int array of size 5:
np.empty(5, dtype=int)

#Draw dots:
plt.xlabel('First year A\'s')
plot(x_array, y_array, 'ro')


#Reading hdf5 file:
with h5py.File('the_filename', 'r') as f:
    my_array = f['array_name'][()]



#Save numpy array as image and show:

from PIL import Image
import numpy as np

w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[256, 256] = [255, 0, 0]
img = Image.fromarray(data, 'RGB')
img.save('my.png')
img.show()




#Load GBR and channel first image data:

data = d[b'data'].reshape(-1,32*32*3)
backup = data.copy()
data = data.reshape(-1,32,32,3)
for i in range(backup.shape[0]):
    single_img = backup[i,:]
    single_img_reshaped = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))
    data[i] = single_img_reshaped



#Classification report:

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))


