cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(EIMedSeg3D)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/PaddleCV-SIG/SlicerEISegMed3D")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Shiyu Tang (Baidu Co.), Lin Han (NYU Tandon School of Engineering)")
set(EXTENSION_DESCRIPTION "A 3D slicer extension for performing Efficient Interactive Medical image Segmentation in 3D. Users guide a deep learning model to perform segmentation by providing positive and negative points.
")
set(EXTENSION_ICONURL "https://github.com/PaddleCV-SIG/SlicerEISegMed3D/releases/download/v0.1.0/EIMedSeg3D.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/PaddleCV-SIG/SlicerEISegMed3D/releases/download/v0.1.0/screenshot.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
