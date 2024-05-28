# Perceptive Visual Urban Analytics is Not (Yet) Suitable for municipalities
This repository contains code and instructions to run the Python implementation of the discriminative clustering approach used in the paper Perceptive Visual Urban Analytics is Not (Yet) Suitable for Municipalities. The code is a reimplementation of the MATLAB code from What Makes Paris Look Like Paris by Doersch et al. 

## Dataset


## Compiling
The code uses a single C++ that is wrapped for use in python with pybind11. If necessary you can clone pybind [here](https://github.com/pybind/pybind11). The compiled module (hog.cpython-37m-x86_64-linux-gnu.so) works for Linux 64-bit x86 running a GNU C Library on Python 3.7. If you need to recompile the file follow these steps:

1. mkdir compiling; cd compiling
2. cp ../CMakeLists.txt ./ 
3. cp ../features.cpp ./
4. mkdir build
5. cd build
6. #Make sure you have cmake version >= 3.4 
7. cmake
8. make ..
9. You should now have a hog.so file in compiling/build/, move this file to root (mv hog.so ../../) and you're done!

## Dataset - IN PROGRESS
Structure your dataset as follows:
The path to your images should be basepath + topdir + cutouts. Inside their can be any number of folders (labelled by socio-economic bin for example) containing images such that the path is:

   ```\$basepath\$/\$topdir\$/cutouts/0/\*.jpg```
   
   ```\$basepath\$/\$topdir\$/cutouts/1/\*.jpg```
   
where 0 or 1 can refer to the bin the images belong to.
--basepath and --topdir arguments are passed through the command line.

## Running the code
Say we want to run the code for our directories of images in ./data/images/cutouts/, where this directory containts 8 folders labelled 0-3 and 8-11 containing our positive and negative set respectively we run the following command:

```python Run_Paris.py --basepath ./data/ --topdir images --positives 0 1 2 3 --negatives 8 9 10 11```

the --sampling and --clustering flags are to run those parts of the code. When completed they save a checkpoint to the output folder so you can stop the script and restart from this checkpoint by setting --sampling 0 for example.

The entire script takes about 60hours to run using gpu acceleration.





