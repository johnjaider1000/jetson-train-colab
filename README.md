# Entrenamiento de modelo con jetson-inference en colab

En esta ocación dejaré los pasos necesarios para entrenar tu modelo de inteligencia artificial con jetson-inference para Jetson Nano en Google Colab.

## Empecemos

A continuación, entrenaremos nuestro propio modelo de detección de objetos SSD-Mobilenet mediante PyTorch y el conjunto de datos [Open Images](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F06l9r). SSD-Mobilenet es una arquitectura de red popular para la detección de objetos en tiempo real en dispositivos móviles e integrados que combina el detector [SSD-300 Single-Shot Multibox con una red troncal](https://arxiv.org/abs/1512.02325) [Mobilnet](https://arxiv.org/abs/1704.04861).

![image](https://github.com/johnjaider1000/jetson-train-colab/assets/8765273/d0dc279e-147e-4c4f-8e9c-07b09aa543d1)

En el siguiente ejemplo, entrenaremos un modelo de detección, personalizado que localiza 8 variedades de diferentes frutas, aunque puede elegir cualquiera de las [600 clases](https://github.com/dusty-nv/pytorch-ssd/blob/master/open_images_classes.txt) en el conjunto de datos de imágenes abiertas para entrenar su modelo. Puede examinar virtualmente el conjunto de datos [aquí](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0xfy).

![image](https://github.com/johnjaider1000/jetson-train-colab/assets/8765273/b43807b0-0ee3-4b05-a33f-05e4250c0ccd)

## Vamos a Colab
Haga click [aquí](https://colab.research.google.com/drive/1b1zD2sO5kdBC1G7yKwMKgdrzuT4ZzS6h?usp=drive_link) o en el siguiente enlace para acceder al archivo colab que le guiará con la configuración del ambiente en colab y todo lo necesario, por favor lea atentamente los comentarios del documento.
https://colab.research.google.com/drive/1b1zD2sO5kdBC1G7yKwMKgdrzuT4ZzS6h?usp=drive_link


## Configuración Jetson Nano:
Para comenzar, primero asegúrese de tener [instalado](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-transfer-learning.md#installing-pytorch) [JetPack 4.4](https://developer.nvidia.com/embedded/jetpack) (o posterior) y PyTorch para Python 3 en su Jetson. JetPack 4.4 incluye TensorRT.7.1, que es la versión mínima de TensorRT que adminte la carga de SSD-Mobilenet a través de OONX. Las versiones más nuevas de TensorRT también funcionarán bien.
