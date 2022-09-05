 English | [简体中文](README.md)
# SlicerEISegMed3D

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/34859558/188449455-cd4e4099-6e70-44ca-b8de-57bab04c187c.png" align="middle" width = 500" />
</p>

**A user-friendly, efficient, AI-assisted 3D medical image annotation platform**  <img src="https://user-images.githubusercontent.com/34859558/188409382-467c4c45-df5f-4390-ac40-fa24149d4e16.png" width="30"/>

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

</div>

A 3D slicer extension for performing **E**fficient **I**nteractive **Seg**mentation on **Med**ical image in **3D**. Users will guide a deep learning model to perform segmentation by providing positive and negative points.

## Install

If you are just trying out this plugin, there is no prior installation steps requred. You will be prompted to install required packages when the plugin loads.

To achieve the best performance, it's advised to install the GPU version of paddlepaddle before running this plugin. Steps are as follows:

1. Check your CUDA driver version. Choose the first version that's lower than yours from the followings: 11.6, 11.2, 11.1, 10.2, 10.1

2. Remove the dot in the chosen version, eg: 11.6 -> 116, and replace the [CHOSEN VERSION] part in the following code with it.

```python
import sys
import os
slicer.app.processEvents()
os.system(f"'{sys.executable}' -m pip install paddlepaddle-gpu==2.3.1.post[CHOSEN VERSION] -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html"))
```

For CUDA 11.6, the code you run should be:

```python
import sys
import os
slicer.app.processEvents()
os.system(f"'{sys.executable}' -m pip install paddlepaddle-gpu==2.3.1.post116 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html"))
```

The final line will only show up after installation finshes. If you started 3D Slicer from a terminal, you should be able to see pip install progress there.

The final line should output 0. Anything else indicates the installation has failed.

# Usage

Model parameters:
