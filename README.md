# SlicerEISegMed3D

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A 3D slicer extension for performing **E**fficient **I**nteractive **Seg**mentation for **Med**ical images in **3D**. Users guide a deep learning model to perform segmentation by providing positive and negative points.

## Install

EISegMed3D depends on pypi packages paddlepaddle and paddleseg. You would need to install them before using this plugin.

Open up the python terminal inside 3D Slicer and run the following code.

```python
import sys
import os
os.system(f"'{sys.executable}' -m pip install paddlepaddle paddleseg")
```

You should see the terminal print out 0. Anything else indicates the installation failed.
