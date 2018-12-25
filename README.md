# SimpleTF-CPP
A C++ example of running TensorFlow model in live mode. Inspired by many hours trying to build tensorflow library on Windows.

Inferring a model on Windows:
![Result on Windows 10](https://i.imgur.com/sANqDnM.gif "Result on Windows 10")

### Features:
- A C++ template for a quick testing any computer vision tensorflow model (.pb file) on Windows.
- Inferring TF model is designed to run on different thread.
- The performance may faster than inferring with python

### Requirements:
- MSVC 2015 (untested on MSVC 2017)
- OpenCV Library
- TensorFlow library
- Cuda 9.2
- CUDNN 7.2



### Compilation:
You can just download my prebuilt [OpenCV and TensorFlow library here](https://drive.google.com/file/d/1ITJetuyXGeoNstVoT6cXRyCf2qn7t_Cx/view?usp=sharing "OpenCV and TensorFlow library here"), then extract it in the same folder as source code and jump to step 5.

**or if you want to have adventures, follows these instructions may save your time:**

1/ Create dependencies folder

2/ Download and install prebuilt OpenCV lib at their official website, and put the library into dependencies folder. Should be looked like this:

![Img0](https://i.imgur.com/l0HprkB.png "Img0")

3/ [Follow this instruction to build tensorflow on Windows as a shared library](https://medium.com/@shiweili/building-tensorflow-c-shared-library-on-windows-e79c90e23e6e)

[or Download prebuilt Tensorflow library here:](https://github.com/fo40225/tensorflow-windows-wheel "or Download prebuilt Tensorflow library here:")

4/ Put Library and include file to tensorflow folder inside dependencies:

![](https://i.imgur.com/H0msWI6.png)

5/ Create folder build. Run CMake to generate the solution to folder build, this is what we expect to see:

![](https://i.imgur.com/wFzifDX.png)

6/ Open the solution then compile as usual.

7/ Have fun

**NOTE: the first time you run the application should be slow since it need to compile some gpu code.**

Follow this template and structure, you can play with any tensorflow model on C++.


#### References:
Model from: https://github.com/yeephycho/tensorflow-face-detection

Some util code from: https://github.com/lysukhin/tensorflow-object-detection-cpp
