# SlicerEISegMed3D

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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
