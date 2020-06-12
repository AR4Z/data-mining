import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

## Leemos la imagen desde la url
url = 'https://farm2.staticflickr.com/1573/26146921423_29f9a86f2b_z_d.jpg'
kk = urllib.request.urlopen(url).read()

## Guardamos la imagen en el directorio donde nos encontremos
## con el nombre 'king.jpg'
imagen = open('king.jpg', 'wb')
imagen.write(kk)
imagen.close()
## Leemos la imagen como un numpy array
kk = plt.imread('king.jpg')
## Si hacemos kk.shape vemos que existen
## tres canales en la imagen (r, g, b)
## Pero como es una imagen en escala de grises
## Los tres canales tienen la misma informacion
## por lo que nos podemos quedar con un solo canal
plt.subplot(221)
plt.title('canal 1')
plt.imshow(kk[:,:,0])
plt.subplot(222)
plt.title('canal 2')
plt.imshow(kk[:,:,1])
plt.subplot(223)
plt.title('canal 3')
plt.imshow(kk[:,:,2])
## Vemos que la imagen esta rotada, hacemos uso de np.flipud
## http://docs.scipy.org/doc/numpy/reference/generated/numpy.flipud.html
plt.subplot(224)
plt.title('canal 1 rotado en BN')
plt.imshow(np.flipud(kk[:,:,0]), cmap=plt.gray())
plt.show()
## Finalmente, nos quedamos con una unica dimension
## Los tres canales rgb son iguales (escala de grises)
matriz = np.flipud(kk[:,:,0])
print(matriz.shape)
## Leemos la imagen desde la url
#for i in range(0, 425, 50):
#    ## Nos quedamos con i componentes principales
#    pca = PCA(n_components = i)
#    ## Ajustamos para reducir las dimensiones
#    kk = pca.fit_transform(matriz)
#    ## 'Deshacemos' y dibujamos
#    plt.imshow(pca.inverse_transform(kk), cmap=plt.gray())
#    plt.title(u'Nro.  de Componentes principales = %s' % str(i))
#    plt.show()
pca = PCA(n_components=350)
pca.fit(matriz)
varianza = pca.explained_variance_ratio_
print(varianza)
var_acum= np.cumsum(varianza)
print(var_acum)
plt.bar(range(len(varianza)), varianza)
plt.plot(range(len(varianza)), var_acum)
plt.show()

#*background:   [97]#282a36