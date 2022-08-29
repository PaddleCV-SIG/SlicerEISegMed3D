# SlicerEISeg

3D slicer extension for interactive medical image segmentation

## Install

EIMedSeg3D depends on pypi packages paddlepaddle and paddleseg. You would need to install them before using this plugin.

Open up the python terminal inside 3D Slicer and run the following code.

```python
import sys
import os
os.system(f"'{sys.executable}' -m pip install paddlepaddle paddleseg")
```

You should see the terminal print out 0. Anything else indicates the installation failed.
