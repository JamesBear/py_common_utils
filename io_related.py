
def get_file_content(path, encoding='utf-8'):
    f = open(path, encoding=encoding)
    content = f.read()
    f.close()
    return content

def write_to_file(path, content, encoding='utf-8'):
    f = open(path, 'w', encoding=encoding)
    f.write(content)
    f.close()

#Split file name
def split_file_name(full_file_path):
    directory = ''
    name = full_file_path
    ext = ''
    full_file_path = full_file_path.strip(' "\' \t\r\n')
    slash_index = full_file_path.rfind('/')
    if slash_index < 0:
        slash_index = full_file_path.rfind('\\')
    if slash_index >= 0:
        directory = full_file_path[:slash_index+1]
        name = full_file_path[slash_index+1:]
    dot_index = name.rfind('.')
    if dot_index >= 0:
        ext = name[dot_index:]
        name = name[:dot_index]

    return directory, name, ext

#Right alignment:
DEFAULT_FORMAT = "{:>12}  {:>12}  {:>12}"
print(DEFAULT_FORMAT.format(item[0], item[1][0], item[1][1]))

#Read image pixel matrix:
from PIL import Image
im = Image.open(path) #Can be many different formats.
pix = im.load()
#print im.size #Get the width and hight of the image for iterating over
#print pix[x,y] #Get the RGBA Value of the a pixel of an image
w,h = im.size
bit_matrix = np.zeros(shape=(h,w))
print(bit_matrix.shape)
for y in range(h):
    for x in range(w):
        bit_matrix[y,x] = int(sum(pix[x,y])<255*1.5)


#Calculating md5:
def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


#split flie(system solution):
try:
    path = os.path.split(os.path.realpath(__file__))[0]
except NameError:
    path = os.getcwd() or os.getenv('PWD')


#convert file to utf-8 with BOM:
import codecs
import os
def is_utf8(file_name):
    try:
        with codecs.open(file_name, "r", encoding="utf-8") as f:
            data = f.read()
    except BaseException:
        return False
    else:
        return True
        
def convert_to_utf8(file_name):
    try:
        content = ''
        with codecs.open(file_name, 'r', 'ansi') as f:
            content = f.read()
        with codecs.open(file_name, 'w', 'utf_8_sig') as f:
            f.write(content)
    except BaseException:
        return False
    else:
        return True
    
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.csv'):
            result = is_utf8(f)
            if result:
                print('"{}" already is utf-8. Skipped.'.format(f))
            else:
                written = convert_to_utf8(f)
                if written:
                    print('"{}" is converted to utf-8.'.format(f))
                else:
                    print('Fail to convert "{}" to utf-8.'.format(f))
    break
    
input('Press enter to continue..')

#morse_talk:
#
#In [1]: import morse_talk as mtalk
#In [2]: code = '-... --- -- -... -..- .--. --'
#In [3]: mtalk.decode(code)
#Out[3]: 'BOMBXPM'



#nCr, combination number
#Method 1:

import scipy.special
scipy.special.comb(10, 3, exact=True)

#Method 2:

import operator as op
from functools import reduce
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, xrange(n, n-r, -1), 1)
    denom = reduce(op.mul, xrange(1, r+1), 1)
    return numer//denom



#count values for each possible value:
unique, counts = np.unique(result, return_counts=True)
d = dict(zip(unique, counts))
print(d)


#plot a distribution:
import seaborn as sns

result = run_n_tests(random_test_bernoulli_standard, 10, 100)
sns.distplot(result, bins=np.arange(result.min(), result.max() + 1))


#Save or compress PNG as JPG, loseless
from PIL import Image

im = Image.open("Ba_b_do8mag_c6_big.png")
rgb_im = im.convert('RGB')
rgb_im.save('colors.jpg', quality=100)


#Build/convert python script to exe:

#pip install PyInstaller
#pyinstaller --onefile <script_name>


#Solve equations:
from sympy.solvers import solve
from sympy import Symbol
x = Symbol('x')
solve(x**2 - 1, x)

# >>> from sympy import Matrix
# >>> from sympy.abc import x, y, z
# >>> from sympy.solvers.solvers import solve_linear_system_LU
# >>> solve_linear_system_LU(Matrix([ ... [1, 2, 0, 1], ... [3, 2, 2, 1], ... [2, 0, 0, 1]]), [x, y, z])
# {x: 1/2, y: 1/4, z: -1/2}



# >>> import numpy as np
# >>> from scipy.linalg import solve
# >>>
# >>> A = np.random.random((3, 3))>>> b = np.random.random(3)>>>
# >>> x = solve(A, b)>>> x
# array([ 0.98323512,  0.0205734 ,  0.06424613])>>>

# >>> np.dot(A, x) - b
# array([ 0.,  0.,  0.])

import random
random.sample(range(1, 100), 3)
[77, 52, 45]


# load and save object using pickle

import pickle
import numpy as np
def load_object(filename):
    if not os.path.exists(filename):
        return None
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
