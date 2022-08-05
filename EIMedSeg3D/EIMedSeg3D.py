import logging
import os
import os.path as osp
import time
from functools import partial

import SimpleITK as sitk
import sitkUtils
import vtk
import numpy as np
import nibabel as nib
import qt

import ctk
import slicer
import SimpleITK as sitk

# from paddle.inference import create_predictor,Config
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

#
# EIMedSeg3D
#


class EIMedSeg3D(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "EIMedSeg3D"  # TODO: make this more human readable by adding spaces
        self.parent.categories = [
            "Interactive Segmentation"
        ]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Lin Han, Daisy (Baidu Corp.)"]
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#EIMedSeg3D">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", self.initializeAfterStartup)

    def initializeAfterStartup(self):
        print("initializeAfterStartup", slicer.app.commandOptions().noMainWindow)
        if not slicer.app.commandOptions().noMainWindow:
            print("in")
            # print("here")
            # self.settingsPanel = MONAILabelSettingsPanel()
            # slicer.app.settingsDialog().addPanel("MONAI Label", self.settingsPanel)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # EIMedSeg3D1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="placePoint",
        sampleName="placePoint1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "placePoint1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="placePoint1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="placePoint1",
    )

    # EIMedSeg3D2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="placePoint",
        sampleName="placePoint2",
        thumbnailFileName=os.path.join(iconsPath, "placePoint2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="placePoint2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="placePoint2",
    )


#
# EIMedSeg3DWidget
#


class EIMedSeg3DWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        self.logic = None
        self._parameterNode = None
        self._currVolumeNode = None
        self._allVolumeNodes = []  # TODO: remove
        self._scanPaths = []
        self._dataFolder = None
        self._segmentNode = None
        self._currScanIdx = None
        self._thresh = 0.9  # output threshold

        self._updatingGUIFromParameterNode = False
        self._endImportProcessing = False

        self.dgPositivePointListNode = None
        self.dgPositivePointListNodeObservers = []
        self.dgNegativePointListNode = None
        self.dgNegativePointListNodeObservers = []
        self.ignorePointListNodeAddEvent = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/EIMedSeg3D.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = EIMedSeg3DLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self.onSceneEndImport)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        # self.ui.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.threshSlider.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
        # TODO: sync select two paths to scene?

        # Buttons
        self.ui.loadModelButton.connect("clicked(bool)", self.loadModelClicked)
        self.ui.loadScanButton.connect("clicked(bool)", self.loadScans)
        self.ui.nextScanButton.connect("clicked(bool)", self.nextScan)
        self.ui.prevScanButton.connect("clicked(bool)", self.prevScan)

        # Positive/Negative Point
        self.ui.dgPositiveControlPointPlacementWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.dgPositiveControlPointPlacementWidget.placeButton().toolTip = "Select positive points"
        self.ui.dgPositiveControlPointPlacementWidget.buttonsVisible = False
        self.ui.dgPositiveControlPointPlacementWidget.placeButton().show()

        self.ui.dgNegativeControlPointPlacementWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.dgNegativeControlPointPlacementWidget.placeButton().toolTip = "Select negative points"
        self.ui.dgNegativeControlPointPlacementWidget.buttonsVisible = False
        self.ui.dgNegativeControlPointPlacementWidget.placeButton().show()

        # Segment editor
        self.ui.embeddedSegmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        # self.ui.embeddedSegmentEditorWidget.setSegmentationNodeSelectorVisible(False)
        # self.ui.embeddedSegmentEditorWidget.setMasterVolumeNodeSelectorVisible(False)
        self.ui.embeddedSegmentEditorWidget.setMRMLSegmentEditorNode(self.logic.get_segment_editor_node())

        # self.ui.embeddedSegmentEditorWidget.setSegmentationNode(segmentationNode)
        # self.ui.embeddedSegmentEditorWidget.setMasterVolumeNode(self._volumeNode)
        # self.ui.embeddedSegmentEditorWidget.setCurrentSegmentID(existing_label_ids[label])

        self.initializeParameterNode()

        # Set place point widget colors
        # TODO: move to initializeParameterNode
        self.ui.dgPositiveControlPointPlacementWidget.setNodeColor(qt.QColor(0, 255, 0))
        self.ui.dgNegativeControlPointPlacementWidget.setNodeColor(qt.QColor(255, 0, 0))
    
    def nextScan(self):
        if self._currScanIdx is None:
            print("scanIdx is none")
        self.turnTo(self._currScanIdx + 1)
    
    def prevScan(self):
        if self._currScanIdx is None:
            print("scanIdx is none")
        self.turnTo(self._currScanIdx - 1)
        
    def turnTo(self, scanIdx):
        if scanIdx < 0:
            print(f"{scanIdx} < 0, no prev scan")
            return
        if scanIdx >= len(self._scanPaths):
            print(f"{scanIdx} >= len(self._scanPaths), no next scan ")
            return

        # 1. unload previous scan, load new scan
        try:
            currScanNode = slicer.util.getNode("EIMedSeg3DScan")
            slicer.mrmlScene.RemoveNode(currScanNode)
        except slicer.util.MRMLNodeNotFoundException as e:
            pass

        self._currVolumeNode = slicer.util.loadVolume(self._scanPaths[scanIdx])
        self._currVolumeNode.SetName("EIMedSeg3DScan")

        # 2. load or create segmentation
        segmentNodeName = "EIMedSeg3DSegmentation"
        if osp.exists(self._scanPaths[scanIdx]):
            self._segmentNode = slicer.modules.segmentations.logic().LoadSegmentationFromFile(
                self._labelPaths[scanIdx], False
            )
        else:
            try:
                self._segmentNode = slicer.util.getNode(segmentNodeName)
            except slicer.util.MRMLNodeNotFoundException as e:
                self._segmentNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

        self._segmentNode.SetName(segmentNodeName)
        self._segmentNode.SetReferenceImageGeometryParameterFromVolumeNode(self._currVolumeNode)

        # 3. create catgs
        segmentation = self._segmentNode.GetSegmentation()
        self.catgTxt2Segmentation(segmentation)

        self.ui.embeddedSegmentEditorWidget.setSegmentationNode(self._segmentNode)
        self.ui.embeddedSegmentEditorWidget.setMasterVolumeNode(self._currVolumeNode)

        self._currScanIdx = scanIdx

    def getCatgFromSegmentation(self, segmentation):
        """Get category info from a segmentation

        Args:
            segmentation (_type_): _description_

        Returns:
            list: [label value, name, color_r, color_g, color_b] color is 0~255
        """
        catgs = []
        for segId in segmentation.GetSegmentIDs():
            segment = segmentation.GetSegment(segId)
            catgs.append(
                [
                    segment.GetLabelValue(),
                    segment.GetName(),
                    *[int(v * 255) for v in segment.GetColor()],
                ]
            )
        return {c[0]: {"name": c[1], "color": c[2:5]} for c in catgs}

    def getCatgFromTxt(self):
        txt_path = osp.join(self._dataFolder, "labels.txt")
        catgs = open(txt_path, "r").readlines()
        catgs = [info.split(" ") for info in catgs]
        for info in catgs:
            info[0] = int(info[0])
            info[2:5] = map(int, info[2:5])
        return {c[0]: {"name": c[1], "color": c[2:5]} for c in catgs}

    def catgTxt2Segmentation(self, segmentation):

        # 1. get catg info from txt and segmentation
        txt_catgs = self.getCatgFromTxt()
        logging.info(f"txt_catgs: {txt_catgs}")

        curr_catgs = self.getCatgFromSegmentation(segmentation)
        logging.info(f"curr_catgs: {curr_catgs}")

        # 2. create and modify info in segmentation
        for labelValue in set(txt_catgs.keys()) - set(curr_catgs.keys()):
            segmentation.AddEmptySegment("", txt_catgs[labelValue]["name"], txt_catgs[labelValue]["color"])

        # 3. sync settings from txt
        for segIdx in segmentation.GetSegmentIDs():
            segment = segmentation.GetSegment(segIdx)
            labelValue = segment.GetLabelValue()
            if labelValue in txt_catgs.keys():
                segment.SetColor(txt_catgs[labelValue]["color"])
                segment.SetName(txt_catgs[labelValue]["name"])
                segment.SetLabelValue(labelValue)

    def loadScans(self):
        currPath = self.ui.dataFolderLineEdit.currentPath
        if currPath is None or len(currPath) == 0:
            print("select path first")
            return

        self._dataFolder = osp.dirname(self.ui.dataFolderLineEdit.currentPath)
        paths = os.listdir(self._dataFolder)
        paths = [s for s in paths if s.split(".")[0][-len("_label") :] != "_label"]
        paths.sort()
        paths = [osp.join(self._dataFolder, s) for s in paths]
        self._scanPaths = paths
        self._labelPaths = []
        for scanPath in self._scanPaths:
            dotPos = scanPath.find(".")
            labelPath = scanPath[:dotPos] + "_label" + scanPath[dotPos:]
            self._labelPaths.append(labelPath)

        self.turnTo(0)

        # self.loadCategories(osp.join(self._dataFolder, "labels.txt"))

    # def nextScan(self):

    def getThresh(self):
        return self.ui.threshSlider.value

    def loadModelClicked(self):
        # self._segmentNode = slicer.util.getNode("EIMedSeg3DSegmentation")

        # self._volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")

        # self.ui.dgPositiveControlPointPlacementWidget.setPlaceModeEnabled(True)
        # self.ui.dgNegativeControlPointPlacementWidget.setPlaceModeEnabled(True)

        # volumeCollection = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
        # self._allVolumeNodes.clear()
        # for idx in range(volumeCollection.GetNumberOfItems()):
        #     self._allVolumeNodes.append(volumeCollection.GetItemAsObject(idx))

        # print(dir(slicer.util))

        # print(len(self._allVolumeNodes))
        # print(self._allVolumeNodes[0])
        # print(dir(self._allVolumeNodes[0]))
        # print(self._allVolumeNodes[0].GetImageData())
        # print((self._allVolumeNodes[0].GetImageData()))

        # self._segmentNode = None
        # self.onSceneEndImport(None, None)
        pass

    def onSceneEndImport(self, caller, event):
        if self._endImportProcessing:
            return

        self._endImportProcessing = True

        # get all volumes
        volumeCollection = slicer.mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
        self._allVolumeNodes.clear()
        for idx in range(volumeCollection.GetNumberOfItems()):
            self._allVolumeNodes.append(volumeCollection.GetItemAsObject(idx))

        # unset curr volume if not exist
        names = [v.GetName() for v in self._allVolumeNodes]
        if self._currVolumeNode is not None and self._currVolumeNode.GetName() not in names:
            self._currVolumeNode = None

        # set curr voulme if not set
        if len(self._allVolumeNodes) != 0 and self._currVolumeNode is None:
            self._currVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")

        # update volume selector
        # self.ui.volumeSelector.clear()
        # for v in self._allVolumeNodes:
        #     self.ui.volumeSelector.addItem(v.GetName())
            # self.ui.volumeSelector.setToolTip(self.current_sample.get("name", "") if self.current_sample else "")

        # # set current node in selector and init segment editor
        # if self._currVolumeNode is not None:
        #     self.ui.volumeSelector.setCurrentIndex(self.ui.volumeSelector.findText(self._currVolumeNode.GetName()))

        #     if self._segmentNode is None:
        #         name = "segmentation_" + self._currVolumeNode.GetName()
        #         self._segmentNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        #         self._segmentNode.SetReferenceImageGeometryParameterFromVolumeNode(self._currVolumeNode)
        #         self._segmentNode.SetName(name)

        #         segmentation = self._segmentNode.GetSegmentation()
        #         segmentation.AddEmptySegment("Tissue", "Tissue", [0, 0, 1.0])

        #     self.ui.embeddedSegmentEditorWidget.setSegmentationNode(self._segmentNode)
        #     self.ui.embeddedSegmentEditorWidget.setMasterVolumeNode(self._currVolumeNode)

        #     # self.ui.embeddedSegmentEditorWidget.setCurrentSegmentID

        self._endImportProcessing = False
        # if not self._allVolumeNodes:
        #     self.updateGUIFromParameterNode()

    def onControlPointAdded(self, observer, eventid):
        print("=====")
        posPoints = self.getControlPointsXYZ(self.dgPositivePointListNode, "positive")
        negPoints = self.getControlPointsXYZ(self.dgNegativePointListNode, "negative")

        newPointIndex = observer.GetDisplayNode().GetActiveControlPoint()
        newPointPos = self.getControlPointXYZ(observer, newPointIndex)
        isPositivePoint = False if len(posPoints) == 0 else newPointPos == posPoints[-1]
        logging.info(f"New point: {newPointPos}, is positive: {isPositivePoint}")

        self.ignorePointListNodeAddEvent = True

        # maybe run inference here
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            logging.info(f"Threshold: {self.getThresh()}")
            # self.ui.progressBar.setValue(33)
            # time.sleep(1)
            # self.ui.progressBar.setValue(66)
            # time.sleep(1)
            self.ui.progressBar.setValue(100)

            # shape = self._currVolumeNode.GetImageData().GetDimensions()

            segmentation = self._segmentNode.GetSegmentation()
            segmentId = segmentation.GetSegmentIdBySegmentName("Tissue")

            # get current seg mask as numpy
            res = slicer.util.arrayFromSegmentBinaryLabelmap(self._segmentNode, segmentId, self._currVolumeNode)

            # add new
            p = newPointPos
            p = [p[2], p[1], p[0]]
            res[p[0] - 10 : p[0] + 10, p[1] - 10 : p[1] + 10, p[2] - 10 : p[2] + 10] = 1

            # set new numpy mask to segmentation
            slicer.util.updateSegmentBinaryLabelmapFromArray(res, self._segmentNode, segmentId, self._currVolumeNode)

            # segmentId = segmentation.GetSegmentIdBySegmentName("Tissue")
            # self.ui.embeddedSegmentEditorWidget.setCurrentSegmentID(segmentId)
            # effect = self.ui.embeddedSegmentEditorWidget.effectByName("Paint")
            # effect.setParameter("BrushSphere", True)
            # selectedSegmentLabelmap = effect.selectedSegmentLabelmap()

            # img =
            # nib_img = nib.Nifti1Image(data, np.eye(4))
            # nib.save(nib_img, "/home/lin/Desktop/test.nii.gz")

            # labelImage = sitk.ReadImage(in_file)
            # labelmapVolumeNode = sitkUtils.PushVolumeToSlicer(labelImage, None, className="vtkMRMLLabelMapVolumeNode")

            # newLabelmap = slicer.vtkOrientedImageData()
            # self._segmentNode.GetBinaryLabelmapRepresentation(segmentId, newLabelmap)

            # effect.modifySelectedSegmentByLabelmap(
            #     newLabelmap, slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeAdd
            # )

        self.ignorePointListNodeAddEvent = False

        # self.onEditControlPoints(self.dgPositivePointListNode, "positive")
        # self.onEditControlPoints(self.dgNegativePointListNode, "MONAILabel.BackgroundPoints")
        # self.ignorePointListNodeAddEvent = False

    def getControlPointXYZ(self, pointListNode, index):
        v = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        # v = self._volumeNode
        RasToIjkMatrix = vtk.vtkMatrix4x4()
        v.GetRASToIJKMatrix(RasToIjkMatrix)

        coord = pointListNode.GetNthControlPointPosition(index)

        world = [0, 0, 0]
        pointListNode.GetNthControlPointPositionWorld(index, world)

        p_Ras = [coord[0], coord[1], coord[2], 1.0]
        p_Ijk = RasToIjkMatrix.MultiplyDoublePoint(p_Ras)
        p_Ijk = [round(i) for i in p_Ijk]

        logging.debug(f"RAS: {coord}; WORLD: {world}; IJK: {p_Ijk}")
        return p_Ijk[0:3]

    def getControlPointsXYZ(self, pointListNode, name):
        v = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        # v = self._volumeNode
        RasToIjkMatrix = vtk.vtkMatrix4x4()
        v.GetRASToIJKMatrix(RasToIjkMatrix)

        point_set = []
        n = pointListNode.GetNumberOfControlPoints()
        for i in range(n):
            coord = pointListNode.GetNthControlPointPosition(i)

            world = [0, 0, 0]
            pointListNode.GetNthControlPointPositionWorld(i, world)

            p_Ras = [coord[0], coord[1], coord[2], 1.0]
            p_Ijk = RasToIjkMatrix.MultiplyDoublePoint(p_Ras)
            p_Ijk = [round(i) for i in p_Ijk]

            logging.info(f"RAS: {coord}; WORLD: {world}; IJK: {p_Ijk}")
            point_set.append(p_Ijk[0:3])

        logging.info(f"{name} => Current control points: {point_set}")
        return point_set

    def getImageData(self, save=False):
        volumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        print(volumeNode)
        if volumeNode is None:
            return None
        image_id = volumeNode.GetName()
        print(image_id)
        s = img_data.GetDimensions()
        data_np = np.zeros(s)
        tic = time.time()
        for x in range(s[0]):
            for y in range(s[1]):
                for z in range(s[2]):
                    data_np[x, y, z] = img_data.GetScalarComponentAsFloat(x, y, z, 0)
        if save:
            np.save("/home/lin/Desktop/data.npy", data_np)

        """
        import nibabel as nib
        data = np.load('/home/lin/Desktop/data.npy')
        nib_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(nib_img, "/home/lin/Desktop/test.nii.gz")
        """

        return data_np

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        print("cleanup")
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        print("enter")
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(
            self._parameterNode,
            vtk.vtkCommand.ModifiedEvent,
            self.updateGUIFromParameterNode,
        )

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        print("onSceneStartClose")
        # Parameter node will be reset, do not use it anymore
        self._currVolumeNode = None
        self._allVolumeNodes.clear()
        self._segmentNode = None

        self.setParameterNode(None)
        self.resetPointList(
            self.ui.dgPositiveControlPointPlacementWidget,
            self.dgPositivePointListNode,
            self.dgPositivePointListNodeObservers,
        )
        self.dgPositivePointListNode = None
        self.resetPointList(
            self.ui.dgNegativeControlPointPlacementWidget,
            self.dgNegativePointListNode,
            self.dgNegativePointListNodeObservers,
        )
        self.dgNegativePointListNode = None

    def resetPointList(self, markupsPlaceWidget, pointListNode, pointListNodeObservers):
        if markupsPlaceWidget.placeModeEnabled:
            markupsPlaceWidget.setPlaceModeEnabled(False)

        if pointListNode:
            slicer.mrmlScene.RemoveNode(pointListNode)
            self.removePointListNodeObservers(pointListNode, pointListNodeObservers)

    def removePointListNodeObservers(self, pointListNode, pointListNodeObservers):
        if pointListNode and pointListNodeObservers:
            for observer in pointListNodeObservers:
                pointListNode.RemoveObserver(observer)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        print("initializeParameterNode")
        self.setParameterNode(self.logic.getParameterNode())
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        # if not self._parameterNode.GetNodeReference("InputVolume"):
        #     firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #     if firstVolumeNode:
        #         self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())
        pass

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        # if inputParameterNode:
        #     self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # for v in self._allVolumeNodes:
        #     self.ui.volumeSelector.addItem(v.GetName())
            # self.ui.volumeSelector.setToolTip(self.current_sample.get("name", "") if self.current_sample else "")

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True
        # print("_+_+", self.dgPositivePointListNode)
        if not self.dgPositivePointListNode:
            (
                self.dgPositivePointListNode,
                self.dgPositivePointListNodeObservers,
            ) = self.createPointListNode("P", self.onControlPointAdded, [0.5, 1, 0.5])
            self.ui.dgPositiveControlPointPlacementWidget.setCurrentNode(self.dgPositivePointListNode)
            self.ui.dgPositiveControlPointPlacementWidget.setPlaceModeEnabled(False)

        if not self.dgNegativePointListNode:
            (
                self.dgNegativePointListNode,
                self.dgNegativePointListNodeObservers,
            ) = self.createPointListNode("P", self.onControlPointAdded, [0.5, 1, 0.5])

            self.ui.dgNegativeControlPointPlacementWidget.setCurrentNode(self.dgNegativePointListNode)
            self.ui.dgNegativeControlPointPlacementWidget.setPlaceModeEnabled(False)

        # self.ui.dgPositiveControlPointPlacementWidget.setEnabled(self.ui.deepgrowModelSelector.currentText)
        # self.ui.dgPositiveControlPointPlacementWidget.setEnabled("deepedit")

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def createPointListNode(self, name, onMarkupNodeModified, color):
        displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsDisplayNode")
        displayNode.SetTextScale(0)
        displayNode.SetSelectedColor(color)

        pointListNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        pointListNode.SetName(name)
        pointListNode.SetAndObserveDisplayNodeID(displayNode.GetID())

        pointListNodeObservers = []
        self.addPointListNodeObserver(pointListNode, onMarkupNodeModified)
        return pointListNode, pointListNodeObservers

    def removePointListNodeObservers(self, pointListNode, pointListNodeObservers):
        if pointListNode and pointListNodeObservers:
            for observer in pointListNodeObservers:
                pointListNode.RemoveObserver(observer)

    def addPointListNodeObserver(self, pointListNode, onMarkupNodeModified):
        pointListNodeObservers = []
        if pointListNode:
            eventIds = [slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent]
            for eventId in eventIds:
                pointListNodeObservers.append(pointListNode.AddObserver(eventId, onMarkupNodeModified))
        return pointListNodeObservers

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        # self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        # self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
        # self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
        # self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
        # self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        pass

    def onClickDeepgrow(self, current_point, skip_infer=False):
        print("onClickDeepgrow")
        # model = self.ui.deepgrowModelSelector.currentText
        # if not model:
        #     slicer.util.warningDisplay("Please select a deepgrow model")
        #     return

        # _, segment = self.currentSegment()
        # if not segment:
        #     slicer.util.warningDisplay("Please add the required label to run deepgrow")
        #     return

        # foreground_all = self.getControlPointsXYZ(self.dgPositivePointListNode, "foreground")
        # background_all = self.getControlPointsXYZ(self.dgNegativePointListNode, "background")

        # segment.SetTag("MONAILabel.ForegroundPoints", json.dumps(foreground_all))
        # segment.SetTag("MONAILabel.BackgroundPoints", json.dumps(background_all))
        # if skip_infer:
        #     return

        # # use model info "deepgrow" to determine
        # deepgrow_3d = False if self.models[model].get("dimension", 3) == 2 else True
        # start = time.time()

        # label = segment.GetName()
        # operationDescription = f"Run Deepgrow for segment: {label}; model: {model}; 3d {deepgrow_3d}"
        # logging.debug(operationDescription)

        # if not current_point:
        #     if not foreground_all and not deepgrow_3d:
        #         slicer.util.warningDisplay(operationDescription + " - points not added")
        #         return
        #     current_point = foreground_all[-1] if foreground_all else background_all[-1] if background_all else None

        # try:
        #     qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

        #     sliceIndex = None
        #     if self.deepedit_multi_label:
        #         params = {}
        #         segmentation = self._segmentNode.GetSegmentation()
        #         for name in self.info.get("labels", []):
        #             points = []
        #             segmentId = segmentation.GetSegmentIdBySegmentName(name)
        #             segment = segmentation.GetSegment(segmentId) if segmentId else None
        #             if segment:
        #                 fPosStr = vtk.mutable("")
        #                 segment.GetTag("MONAILabel.ForegroundPoints", fPosStr)
        #                 pointset = str(fPosStr)
        #                 print(f"{segmentId} => {name} Control points are: {pointset}")
        #                 if fPosStr is not None and len(pointset) > 0:
        #                     points = json.loads(pointset)

        #             params[name] = points
        #         params["label"] = label
        #         labels = None
        #     else:
        #         sliceIndex = current_point[2] if current_point else None
        #         logging.debug(f"Slice Index: {sliceIndex}")

        #         if deepgrow_3d or not sliceIndex:
        #             foreground = foreground_all
        #             background = background_all
        #         else:
        #             foreground = [x for x in foreground_all if x[2] == sliceIndex]
        #             background = [x for x in background_all if x[2] == sliceIndex]

        #         logging.debug(f"Foreground: {foreground}")
        #         logging.debug(f"Background: {background}")
        #         logging.debug(f"Current point: {current_point}")

        #         params = {
        #             "label": label,
        #             "foreground": foreground,
        #             "background": background,
        #         }
        #         labels = [label]

        #     params["label"] = label
        #     params.update(self.getParamsFromConfig("infer", model))
        #     print(f"Request Params for Deepgrow/Deepedit: {params}")

        #     image_file = self.current_sample["id"]
        #     result_file, params = self.logic.infer(model, image_file, params, session_id=self.getSessionId())
        #     print(f"Result Params for Deepgrow/Deepedit: {params}")
        #     if labels is None:
        #         labels = (
        #             params.get("label_names")
        #             if params and params.get("label_names")
        #             else self.models[model].get("labels")
        #         )
        #         if labels and isinstance(labels, dict):
        #             labels = [k for k, _ in sorted(labels.items(), key=lambda item: item[1])]

        #     freeze = label if self.ui.freezeUpdateCheckBox.checked else None
        #     self.updateSegmentationMask(result_file, labels, None if deepgrow_3d else sliceIndex, freeze=freeze)
        # except:
        #     logging.exception("Unknown Exception")
        #     slicer.util.errorDisplay(operationDescription + " - unexpected error.", detailedText=traceback.format_exc())
        # finally:
        #     qt.QApplication.restoreOverrideCursor()

        # self.updateGUIFromParameterNode()
        # logging.info(f"Time consumed by Deepgrow: {time.time() - start:3.1f}")

    def onEditControlPoints(self, pointListNode, tagName):
        if pointListNode is None:
            return

        pointListNode.RemoveAllControlPoints()
        segmentId, segment = self.currentSegment()
        if segment and segmentId:
            # v = self._volumeNode
            v = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            IjkToRasMatrix = vtk.vtkMatrix4x4()
            v.GetIJKToRASMatrix(IjkToRasMatrix)

            fPosStr = vtk.mutable("")
            segment.GetTag(tagName, fPosStr)
            pointset = str(fPosStr)
            logging.debug(f"{segmentId} => {segment.GetName()} Control points are: {pointset}")

            if fPosStr is not None and len(pointset) > 0:
                points = json.loads(pointset)
                for p in points:
                    p_Ijk = [p[0], p[1], p[2], 1.0]
                    p_Ras = IjkToRasMatrix.MultiplyDoublePoint(p_Ijk)
                    logging.debug(f"Add Control Point: {p_Ijk} => {p_Ras}")
                    pointListNode.AddControlPoint(p_Ras[0:3])

    def onSelectLabel(self, caller=None, event=None):
        self.updateParameterNodeFromGUI(caller, event)

        self.ignorePointListNodeAddEvent = True
        self.onEditControlPoints(self.dgPositivePointListNode, "MONAILabel.ForegroundPoints")
        self.onEditControlPoints(self.dgNegativePointListNode, "MONAILabel.BackgroundPoints")
        self.ignorePointListNodeAddEvent = False

    def getControlPointsXYZ(self, pointListNode, name):
        RasToIjkMatrix = vtk.vtkMatrix4x4()

        point_set = []
        n = pointListNode.GetNumberOfControlPoints()
        for i in range(n):
            coord = pointListNode.GetNthControlPointPosition(i)

            world = [0, 0, 0]
            pointListNode.GetNthControlPointPositionWorld(i, world)

            p_Ras = [coord[0], coord[1], coord[2], 1.0]
            p_Ijk = RasToIjkMatrix.MultiplyDoublePoint(p_Ras)
            p_Ijk = [round(i) for i in p_Ijk]

            logging.debug(f"RAS: {coord}; WORLD: {world}; IJK: {p_Ijk}")
            point_set.append(p_Ijk[0:3])

        logging.info(f"{name} => Current control points: {point_set}")
        return point_set


#
# EIMedSeg3DLogic
#


class EIMedSeg3DLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")
        if not parameterNode.GetParameter("Invert"):
            parameterNode.SetParameter("Invert", "false")

    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(
            slicer.modules.thresholdscalarvolume,
            None,
            cliParams,
            wait_for_completion=True,
            update_display=showResult,
        )
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")

    def infer_image():
        # GPU下进行预测
        class Predictor:
            """
            用于预测的类
            """

            def __init__(self, model_path, param_path):
                self.pred_cfg = Config(model_path, param_path)
                self.pred_cfg.disable_glog_info()
                self.pred_cfg.enable_memory_optim()
                self.pred_cfg.switch_ir_optim(True)
                self.pred_cfg.enable_use_gpu(100, 0)
                # self.pred_cfg.disable_gpu()
                self.predictor = create_predictor(self.pred_cfg)

            def predict(self, data):
                input_names = self.predictor.get_input_names()
                input_handle = self.predictor.get_input_handle(input_names[0])
                output_names = self.predictor.get_output_names()
                output_handle = self.predictor.get_output_handle(output_names[0])
                input_handle.reshape(data.shape)
                input_handle.copy_from_cpu(data)
                self.predictor.run()
                result = output_handle.copy_to_cpu()
                return result

        def resampleImage(sitkimg, new_shape, new_spacing):
            # 对SimpleITK 的数据进行重新采样。重新设置spacing和shape
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(sitkimg)
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize(new_shape)
            resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
            resampler.SetInterpolator(sitk.sitkLinear)
            return resampler.Execute(sitkimg)

        def crop_wwwc(sitkimg, max_v, min_v):
            # 对SimpleITK的数据进行窗宽窗位的裁剪，应与训练前对数据预处理时一致
            intensityWindow = sitk.IntensityWindowingImageFilter()
            intensityWindow.SetWindowMaximum(max_v)
            intensityWindow.SetWindowMinimum(min_v)
            return intensityWindow.Execute(sitkimg)

        def GetLargestConnectedCompont(binarysitk_image):
            # 最大连通域提取,binarysitk_image 是掩膜
            cc = sitk.ConnectedComponent(binarysitk_image)
            stats = sitk.LabelIntensityStatisticsImageFilter()
            stats.SetGlobalDefaultNumberOfThreads(8)
            stats.Execute(cc, binarysitk_image)  # 根据掩膜计算统计量
            # stats.
            maxlabel = 0
            maxsize = 0
            for l in stats.GetLabels():  # 掩膜中存在的标签类别
                size = stats.GetPhysicalSize(l)
                if maxsize < size:  # 只保留最大的标签类别
                    maxlabel = l
                    maxsize = size
            labelmaskimage = sitk.GetArrayFromImage(cc)
            outmask = labelmaskimage.copy()
            if len(stats.GetLabels()):
                outmask[labelmaskimage == maxlabel] = 255
                outmask[labelmaskimage != maxlabel] = 0
            return outmask

        origin_path = "/home/aistudio/test.nii.gz"
        origin = sitk.ReadImage(origin_path)

        new_shape = (256, 128, 128)  # xyz #这个形状与训练的对数据预处理的形状要一致
        image_shape = origin.GetSize()
        spacing = origin.GetSpacing()
        new_spacing = tuple((image_shape / np.array(new_shape)) * spacing)

        itk_img_res = resampleImage(origin, new_shape, new_spacing)  # 得到重新采样后的图像
        itk_img_res = crop_wwwc(itk_img_res, max_v=611, min_v=-338)  # 和预处理文件一致
        npy_img = sitk.GetArrayFromImage(itk_img_res).astype("float32")
        input_data = np.expand_dims(npy_img, axis=0)
        if input_data.max() > 0:  # 归一化
            input_data = input_data / input_data.max()
        input_data = np.expand_dims(input_data, axis=0)
        print(f"输入网络前数据的形状:{input_data.shape}")  # shape(1, 1, 128, 128, 256)

        # 创建预测器，加载模型进行预测
        predictor = Predictor(
            "/home/aistudio/export_model/model.pdmodel", "/home/aistudio/export_model/model.pdiparams"
        )
        output_data = predictor.predict(input_data)
        print(f"预测结果的形状：{output_data.shape}")  # shape (1, 128, 128, 256)

        # 加载3d模型预测的mask，由numpy 转换成SimpleITK格式
        data = np.squeeze(output_data)
        mask_itk_new = sitk.GetImageFromArray(data)
        mask_itk_new.SetSpacing(new_spacing)
        mask_itk_new.SetOrigin(origin.GetOrigin())
        mask_itk_new.SetDirection(origin.GetDirection())
        mask_itk_new = sitk.Cast(mask_itk_new, sitk.sitkUInt8)

        x, y, z = mask_itk_new.GetSize()
        mask_array = np.zeros((z, y, x), np.uint8)
        max_value = np.max(sitk.GetArrayViewFromImage(mask_itk_new))
        # 对转换成SimpleITK的预测mask进行处理，只保留最大连通域，去除小目标
        for index in range(1, max_value + 1):
            sitk_seg = sitk.BinaryThreshold(
                mask_itk_new, lowerThreshold=index, upperThreshold=index, insideValue=255, outsideValue=0
            )
            # step2.形态学开运算
            BMO = sitk.BinaryMorphologicalOpeningImageFilter()
            BMO.SetKernelType(sitk.sitkNearestNeighbor)
            BMO.SetKernelRadius(2)
            BMO.SetForegroundValue(1)
            sitk_open = BMO.Execute(sitk_seg != 0)
            # 提取每个椎体的最大连通域提取，为了去掉小目标
            sitk_open_array = GetLargestConnectedCompont(sitk_open)
            mask_array[sitk_open_array == 255] = int(index)

        # 对处理好的预测mask，重采样原始的size 和spacing
        sitkMask = sitk.GetImageFromArray(mask_array)
        sitkMask.CopyInformation(mask_itk_new)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitkMask)  # 需要重新采样的目标图像
        resampler.SetSize(origin.GetSize())
        resampler.SetOutputSpacing(origin.GetSpacing())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        Mask = resampler.Execute(sitkMask)  # 得到重新采样后的图像
        Mask.CopyInformation(origin)
        sitk.WriteImage(Mask, origin_path.replace(".nii.gz", "_predict.nii.gz"))
        print("预测成功！")

    def get_segment_editor_node(self):
        # Use the Segment Editor module's parameter node for the embedded segment editor widget.
        # This ensures that if the user switches to the Segment Editor then the selected
        # segmentation node, volume node, etc. are the same.
        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
        if segmentEditorNode is None:
            segmentEditorNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorNode.UnRegister(None)
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        return segmentEditorNode


#
# EIMedSeg3DTest
#


class EIMedSeg3DTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_EIMedSeg3D1()

    def test_EIMedSeg3D1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("EIMedSeg3D1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = EIMedSeg3DLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
