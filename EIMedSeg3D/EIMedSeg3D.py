import logging
import os
import os.path as osp
import time
from functools import partial

import qt
import ctk
import vtk
import numpy as np
import SimpleITK as sitk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

# when test, wont use any paddle related funcion
TEST = True
if not TEST:
    import paddle
    import inference
    import inference.predictor as predictor


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
        # print("initializeAfterStartup", slicer.app.commandOptions().noMainWindow)
        pass


class LabelNode:
    def __init__(self, segmentId=None, labelValue=None, name=None, color=None):
        self.segmentId = segmentId
        self.labelValue = labelValue
        self.name = name
        self.color = color


class Clicker(object):
    def __init__(self):
        self.reset_clicks()

    def get_clicks(self, clicks_limit=None):  # [click1, click2, ...]
        return self.clicks_list[:clicks_limit]

    def add_click(self, click):
        coords = click.coords

        click.index = self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)

    def reset_clicks(self):
        self.num_pos_clicks = 0
        self.num_neg_clicks = 0
        self.clicks_list = []

    def __len__(self):
        return len(self.clicks_list)


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
        self._scanPaths = []
        self._dataFolder = None
        self._currScanIdx = None
        # self._segmentEditor = {}
        self._currVolumeNode_scanPath = {}
        self._prev_segId = None
        self._syncing_catg = False
        self._using_interactive = False

        self._updatingGUIFromParameterNode = False
        self._endImportProcessing = False

        self.dgPositivePointListNode = None
        self.dgPositivePointListNodeObservers = []
        self.dgNegativePointListNode = None
        self.dgNegativePointListNodeObservers = []
        self.ignorePointListNodeAddEvent = False
        self.init_params()

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/EIMedSeg3D.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        # TODO: we may not need logic. user have to interact
        self.logic = EIMedSeg3DLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self.onSceneEndImport)

        # TODO: is syncing settings between node and gui on show/scenestart/... necessary

        # button, slider
        self.ui.loadModelButton.connect("clicked(bool)", self.loadModelClicked)
        self.ui.loadScanButton.connect("clicked(bool)", self.loadScans)
        self.ui.nextScanButton.connect("clicked(bool)", self.nextScan)
        self.ui.prevScanButton.connect("clicked(bool)", self.prevScan)
        self.ui.finishScanButton.connect("clicked(bool)", self.finishScan)
        self.ui.finishSegmentButton.connect("clicked(bool)", self.exitInteractiveMode)
        self.ui.opacitySlider.connect("valueChanged(double)", self.setSegmentationOpacity)

        # positive/negative control point
        self.ui.dgPositiveControlPointPlacementWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.dgPositiveControlPointPlacementWidget.placeButton().toolTip = "Add positive points"
        self.ui.dgPositiveControlPointPlacementWidget.placeButton().show()
        self.ui.dgPositiveControlPointPlacementWidget.deleteButton().setFixedHeight(0)  # diable delete point button
        self.ui.dgPositiveControlPointPlacementWidget.deleteButton().setFixedWidth(0)
        self.ui.dgPositiveControlPointPlacementWidget.placeButton().connect("clicked(bool)", self.enterInteractiveMode)

        self.ui.dgNegativeControlPointPlacementWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.dgNegativeControlPointPlacementWidget.placeButton().toolTip = "Add negative points"
        self.ui.dgNegativeControlPointPlacementWidget.placeButton().show()
        self.ui.dgNegativeControlPointPlacementWidget.deleteButton().setFixedHeight(0)
        self.ui.dgNegativeControlPointPlacementWidget.deleteButton().setFixedWidth(0)
        self.ui.dgNegativeControlPointPlacementWidget.placeButton().connect("clicked(bool)", self.enterInteractiveMode)

        # segment editor
        self.ui.embeddedSegmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.embeddedSegmentEditorWidget.setMRMLSegmentEditorNode(self.logic.get_segment_editor_node())

        # Set place point widget colors
        self.ui.dgPositiveControlPointPlacementWidget.setNodeColor(qt.QColor(0, 255, 0))
        self.ui.dgNegativeControlPointPlacementWidget.setNodeColor(qt.QColor(255, 0, 0))

        self.initializeFromNode()

    def init_params(self):
        "init changble parameters here"
        self.predictor_params_ = {"norm_radius": 2, "spatial_scale": 1.0}
        self.ratio = (512 / 880, 512 / 880, 12 / 12)  # xyz 这个形状与训练的对数据预处理的形状要一致，怎么切换不同模型？ todo： 在模块上设置预处理形状。和模型一致
        self.train_shape = (512, 512, 12)
        self.image_ww = (0, 2650)  # low, high range for image crop
        self.test_iou = False  # the label file need to be set correctly
        self.file_suffix = [".nii", ".nii.gz"]  # files with these suffix will be loaded
        self.device, self.enable_mkldnn = "cpu", True

    def loadModelClicked(self):
        model_path, param_path = self.ui.modelPathInput.currentPath, self.ui.paramPathInput.currentPath
        if not model_path or not param_path:
            slicer.util.errorDisplay("Please set the model_path and parameter path before load model.")
            return

        self.inference_predictor = predictor.BasePredictor(
            model_path, param_path, device=self.device, enable_mkldnn=self.enable_mkldnn, **self.predictor_params_
        )

        slicer.util.delayDisplay("Sucessfully loaded model to {}!".format(self.device), autoCloseMsec=1500)

    def clearScene(self):
        if self._currVolumeNode is not None:
            slicer.mrmlScene.RemoveNode(self._currVolumeNode)

        segmentationNode = self.segmentationNode
        if segmentationNode is not None:
            slicer.mrmlScene.RemoveNode(segmentationNode)

    def saveOrReadCurrIdx(self, saveFlag=False):
        path = os.path.join(self._dataFolder, "currScanIdx.txt")
        if saveFlag:
            with open(osp.join(self._dataFolder, path), "w") as f:
                f.write(str(self._currScanIdx + 1))
        else:
            with open(osp.join(self._dataFolder, path), "r") as f:
                self._currScanIdx = int(f.read())

    """ load/change scan related """

    def loadScans(self):
        """Get all the scans under a folder and turn to the first one"""

        # 1. ensure valid input
        dataFolder = self.ui.dataFolderLineEdit.currentPath
        if dataFolder is None or len(dataFolder) == 0:
            slicer.util.delayDisplay("Please select a Data Folder first!", autoCloseMsec=5000)
            return

        if not osp.exists(dataFolder):
            slicer.util.delayDisplay(f"The Data Folder( {dataFolder} ) doesn't exist!", autoCloseMsec=2000)
            return

        self.ui.dataFolderLineEdit.addCurrentPathToHistory()
        self.clearScene()

        # 2. list files in assigned directory
        self._dataFolder = dataFolder
        paths = [p for p in os.listdir(self._dataFolder) if p[p.find(".") :] in self.file_suffix]
        paths = [p for p in paths if p.split(".")[0][-len("_label") :] != "_label"]
        paths.sort()
        self._scanPaths = [osp.join(self._dataFolder, p) for p in paths]

        slicer.util.delayDisplay(
            f"Found {len(self._scanPaths)} scans in folder {self._dataFolder}",
            autoCloseMsec=1200,
        )

        if osp.exists(osp.join(self._dataFolder, "currScanIdx.txt")):
            self.saveOrReadCurrIdx(saveFlag=False)
        else:
            self._currScanIdx = 0
        self.turnTo()

        logging.info(
            f"All scans found under {self._dataFolder} are {','.join([' '+osp.basename(p) for p in self._scanPaths])}"
        )

    def nextScan(self):
        if len(self._scanPaths) == 0:
            slicer.util.errorDisplay(
                "You have marked all the data, and there is no next scan. Please reselect the file path and click the Load Scans button."
            )
            return

        if self._currScanIdx == len(self._scanPaths) - 1:
            slicer.util.errorDisplay("This is the last scan. No next scan")
            return

        self._currScanIdx += 1
        self.turnTo()

    def prevScan(self):
        if len(self._scanPaths) == 0:
            slicer.util.errorDisplay(
                "You have marked all the data, and there is no next scan. Please reselect the file path and click the Load Scans button."
            )
            return
        if self._currScanIdx == 0:
            slicer.util.errorDisplay("This is the first scan. No previous scan")
            return

        self._currScanIdx -= 1
        self.turnTo()

    def turnTo(self):
        """
        Turn to the self._currScanIdx th scan, load scan and label
        """
        # 0. ensure valid status and clear scene
        if len(self._scanPaths) == 0:
            slicer.util.delayDisplay("No scan found, please load scans first.", autoCloseMsec=2000)
            return

        self.ui.dgPositiveControlPointPlacementWidget.setEnabled(False)
        self.ui.dgNegativeControlPointPlacementWidget.setEnabled(False)

        self.clearScene()  # remove the volume and segmentation node

        # 1. load new scan & preprocess
        image_path = self._scanPaths[self._currScanIdx]
        self._currVolumeNode = slicer.util.loadVolume(image_path)
        self._currVolumeNode_scanPath[self._currVolumeNode] = image_path  # TODO: remove

        # 2. load segmentation or create an empty one
        dot_pos = image_path.find(".")
        self._currLabelPath = image_path[:dot_pos] + "_label" + image_path[dot_pos:]
        if osp.exists(self._currLabelPath):
            segmentNode = slicer.modules.segmentations.logic().LoadSegmentationFromFile(self._currLabelPath, False)
        else:
            segmentNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

        segmentNode.SetName("EIMedSeg3DSegmentation")
        segmentNode.SetReferenceImageGeometryParameterFromVolumeNode(self._currVolumeNode)

        # 3. create category label from txt and segmentation
        self.catgTxt2Segmentation()
        self.catgSegmentation2Txt()

        def sync(*args):
            if self._syncing_catg:
                return
            self.catgSegmentation2Txt()

        segmentNode.AddObserver(segmentNode.GetContentModifiedEvents().GetValue(5), sync)

        # 4. set the editor as current result.
        self.ui.embeddedSegmentEditorWidget.setSegmentationNode(segmentNode)
        self.ui.embeddedSegmentEditorWidget.setMasterVolumeNode(self._currVolumeNode)

        self.ui.dgPositiveControlPointPlacementWidget.setEnabled(True)
        self.ui.dgNegativeControlPointPlacementWidget.setEnabled(True)

        # 5. set image
        if not TEST:
            self.prepImage()

    """ category and segmentation management """

    @property
    def segmentationNode(self):
        try:
            return slicer.util.getNode("EIMedSeg3DSegmentation")
        except slicer.util.MRMLNodeNotFoundException:
            return None

    @property
    def segmentation(self):
        segmentationNode = self.segmentationNode
        if segmentationNode is None:
            return None
        return segmentationNode.GetSegmentation()

    @property
    def segments(self):
        segmentation = self.segmentation
        for segId in segmentation.GetSegmentIDs():
            yield segmentation.GetSegment(segId)

    def getSegmentId(self, segment):
        segmentation = self.segmentation
        for segId in segmentation.GetSegmentIDs():
            if segmentation.GetSegment(segId) == segment:
                return segId

    def recordCatg(self):
        """Record current category info from segmentationNode

        Format: {segmentId: name} # note: segmentId is not labelValue
        """
        self._prev_catg = {}
        for segment in self.segments:
            self._prev_catg[self.getSegmentId(segment)] = segment.GetName()

    def writeCatgToTxt(self, catgs):
        infos = []
        for name, labelValue in catgs.items():
            infos.append([labelValue, name])
        infos.sort(key=lambda info: info[0])
        with open(osp.join(self._dataFolder, "labels.txt"), "w") as f:
            for info in infos:
                print(f"{info[0]} {info[1]}", file=f)

    def getCatgFromTxt(self):
        """Parse category info from labels.txt

        Returns:
            dict: {name: labelValue, ... }
        """
        txt_path = osp.join(self._dataFolder, "labels.txt")
        if not osp.exists(txt_path):
            return {}

        catgs = {}
        with open(txt_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if len(l.strip()) != 0]
            for info in lines:
                info = info.split(" ")
                catgs[info[1]] = int(info[0])
        return catgs

    def catgTxt2Segmentation(self):
        """
        Sync category name from labels.txt to segmentation, match by labelValue
        - create if missing
        - correct name if segmentation differes from labels.txt

        Note: Changing labelValue will break the link between segment editor and segmentation visualization. Thus labelValue is not changed in this func.
        """
        if self._syncing_catg:
            return
        self._syncing_catg = True

        # 1. get catg info from labels.txt
        txt_catgs = self.getCatgFromTxt()
        labelValue2name = {v: k for k, v in txt_catgs.items()}

        # 2. modify segment based on labels.txt
        segmentation_names = set()

        for segment in self.segments:
            labelValue = segment.GetLabelValue()
            if labelValue in labelValue2name.keys():
                txt_catg = txt_catgs[labelValue2name[labelValue]]
                segment.SetName(labelValue2name[labelValue])
            segmentation_names.add(segment.GetName())

        # 3. create segments in txt but not in segmentation
        for name in set(txt_catgs.keys()) - set(segmentation_names):
            txt_catg = txt_catgs[name]
            self.segmentation.AddEmptySegment("", name)
        self.recordCatg()
        self._syncing_catg = False

    def catgSegmentation2Txt(self):
        if self._syncing_catg:
            return
        self._syncing_catg = True

        catgs = self.getCatgFromTxt()
        if len(catgs) == 0:
            maxLabelValue = 0
        else:
            maxLabelValue = max([lv for lv in catgs.values()])

        for segment in self.segments:
            name = segment.GetName()
            segmentId = self.getSegmentId(segment)
            if segmentId in self._prev_catg.keys() and self._prev_catg[segmentId] != name:
                catgs[name] = catgs[self._prev_catg[segmentId]]
                del catgs[self._prev_catg[segmentId]]

            if name not in catgs.keys():
                catgs[name] = maxLabelValue + 1
                maxLabelValue += 1
        self.writeCatgToTxt(catgs)
        self.recordCatg()
        self._syncing_catg = False

    """ control point related """

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

    def getControlPointsXYZ(self, pointListNode, name):
        v = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
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
            point_set.append(p_Ijk[0:3])

        # logging.info(f"{name} => Current control points: {point_set}")
        return point_set

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

        return p_Ijk[0:3]

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

    def enterInteractiveMode(self):
        if self._using_interactive:
            return

        segmentation = self.segmentation
        segmentId = self.ui.embeddedSegmentEditorWidget.currentSegmentID()
        segment = segmentation.GetSegment(segmentId)
        current_mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            self.segmentationNode, segmentId, self._currVolumeNode
        )
        if current_mask.sum() != 0:
            # TODO: prompt and let user choose whether to create new segment
            segmentId = segmentation.AddEmptySegment("", segment.GetName(), segment.GetColor())
            self.ui.embeddedSegmentEditorWidget.setCurrentSegmentID(segmentId)

        if not TEST:
            self.setImage()
        self.clicker = Clicker()
        self.ui.embeddedSegmentEditorWidget.setDisabled(True)
        self._using_interactive = True

    def exitInteractiveMode(self):
        self.ui.dgPositiveControlPointPlacementWidget.deleteAllPoints()
        self.ui.dgNegativeControlPointPlacementWidget.deleteAllPoints()

        self.ui.embeddedSegmentEditorWidget.setDisabled(False)
        self._using_interactive = False

    def onControlPointAdded(self, observer, eventid):
        if self.ignorePointListNodeAddEvent:
            return
        self.ignorePointListNodeAddEvent = True

        if not self._using_interactive:
            self.enterInteractiveMode()

        # 1. get new point pos and type
        posPoints = self.getControlPointsXYZ(self.dgPositivePointListNode, "positive")
        negPoints = self.getControlPointsXYZ(self.dgNegativePointListNode, "negative")
        newPointIndex = observer.GetDisplayNode().GetActiveControlPoint()
        newPointPos = self.getControlPointXYZ(observer, newPointIndex)
        isPositivePoint = False if len(posPoints) == 0 else newPointPos == posPoints[-1]
        logging.info(f"{['Negative', 'Positive'][int(isPositivePoint)]} point added at {newPointPos}")

        # 2. ensure current segment empty, create if not
        segmentation = self.segmentation
        segmentId = self.ui.embeddedSegmentEditorWidget.currentSegmentID()
        segment = segmentation.GetSegment(segmentId)

        print("Current segment: ", self.getSegmentId(segment), segment.GetName(), segment.GetLabelValue())

        with slicer.util.tryWithErrorDisplay("Failed to run inference.", waitCursor=True):
            # get current seg mask as numpy
            self.ui.progressBar.setValue(10)

            # # predict image for test
            if TEST:
                p = newPointPos
                p = [p[2], p[1], p[0]]
                res = slicer.util.arrayFromSegmentBinaryLabelmap(self.segmentationNode, segmentId, self._currVolumeNode)
                mask = np.zeros_like(res)
                mask[p[0] - 10 : p[0] + 10, p[1] - 10 : p[1] + 10, p[2] - 10 : p[2] + 10] = 1
            else:
                paddle.device.cuda.empty_cache()
                mask = self.infer_image(newPointPos, isPositivePoint)  # (880, 880, 12) same as res

            # set new numpy mask to segmentation
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                mask, self.segmentationNode, segmentId, self._currVolumeNode
            )

            self.ui.progressBar.setValue(100)

            if self.test_iou:
                label = sitk.ReadImage(self._currLabelPath)
                label = sitk.GetArrayFromImage(label).astype("int32")
                iou = self.get_iou(label, mask, newPointPos)
                print("Current IOU is {}".format(iou))

        self.ignorePointListNodeAddEvent = False

    def get_iou(self, gt_mask, pred_mask, newPointPos, ignore_label=-1):
        ignore_gt_mask_inv = gt_mask != ignore_label
        pred_mask = pred_mask == 1
        obj_gt_mask = gt_mask == gt_mask[newPointPos[2], newPointPos[1], newPointPos[0]]

        intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
        union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

        return intersection / union

    def prepImage(self):
        self.origin = sitk.ReadImage(self._scanPaths[self._currScanIdx])
        itk_img_res = inference.crop_wwwc(
            self.origin, max_v=self.image_ww[1], min_v=self.image_ww[0]
        )  # 和预处理文件一致 (512, 512, 12) WHD
        itk_img_res, self.new_spacing = inference.resampleImage(
            itk_img_res, out_size=self.train_shape
        )  # 得到重新采样后的图像 origin: (880, 880, 12)

        npy_img = sitk.GetArrayFromImage(itk_img_res).astype("float32")  # 12, 512, 512 DHW

        input_data = np.expand_dims(np.transpose(npy_img, [2, 1, 0]), axis=0)
        if input_data.max() > 0:  # 归一化
            input_data = input_data / input_data.max()
        self.input_data = input_data

    def setImage(self):
        print(f"输入网络前数据的形状:{self.input_data.shape}")  # shape (1, 512, 512, 12)
        try:
            self.inference_predictor.set_input_image(self.input_data)
        except AttributeError:
            slicer.util.delayDisplay("Please load model first", autoCloseMsec=1200)

    def infer_image(self, click_position=None, positive_click=True, pred_thr=0.49):
        """
        click_position: one or serveral clicks represent by list like: [[234, 284, 7]]
        positive_click: whether this click is positive or negative
        """
        try:
            paddle.device.set_device(self.device)
        except AttributeError:
            slicer.util.errorDisplay("Model is not loaded. Please load model first")
            return

        tic = time.time()
        self.prepare_click(click_position, positive_click)
        with paddle.no_grad():
            pred_probs = self.inference_predictor.get_prediction_noclicker(self.clicker)

        output_data = (pred_probs > pred_thr) * pred_probs  # (12, 512, 512) DHW
        output_data[output_data > 0] = 1

        self.ui.progressBar.setValue(90)

        # 加载3d模型预测的mask，由 numpy 转换成SimpleITK格式
        output_data = np.transpose(output_data, [2, 1, 0])
        mask_itk_new = sitk.GetImageFromArray(output_data)  # (512, 512, 12) WHD
        mask_itk_new.SetSpacing(self.new_spacing)
        mask_itk_new.SetOrigin(self.origin.GetOrigin())
        mask_itk_new.SetDirection(self.origin.GetDirection())
        mask_itk_new = sitk.Cast(mask_itk_new, sitk.sitkUInt8)

        # 暂时没有杂散目标，不需要最大联通域提取
        Mask, _ = inference.resampleImage(
            mask_itk_new, self.origin.GetSize(), self.origin.GetSpacing(), sitk.sitkNearestNeighbor
        )
        Mask.CopyInformation(self.origin)

        npy_img = sitk.GetArrayFromImage(Mask).astype("float32")  # 12, 512, 512 DHW

        print(f"预测结果的形状：{output_data.shape}, 预测时间为 {(time.time() - tic) * 1000} ms")  # shape (12, 512, 512) DHW test

        return npy_img

    def prepare_click(self, click_position, positive_click):
        click_position_new = []
        for i, v in enumerate(click_position):
            click_position_new.append(int(self.ratio[i] * click_position[i]))

        if positive_click:
            click_position_new.append(100)
        else:
            click_position_new.append(-100)

        print(
            "The {} click is click on {} (resampled)".format(
                ["negative", "positive"][positive_click], click_position_new
            )
        )  # result is correct

        click = inference.Click(is_positive=positive_click, coords=click_position_new)
        self.clicker.add_click(click)
        print("####################### clicker length", len(self.clicker.clicks_list))

    def finishScan(self):
        if self._using_interactive:
            self.exitInteractiveMode()
        self.saveSegmentation()

    def saveSegmentation(self):
        """
        save the file to the current path
        """
        tic = time.time()
        # 1. generate final segmentation mask
        catgs = self.getCatgFromTxt()
        segmentationNode = self.segmentationNode
        segmentation = segmentationNode.GetSegmentation()
        size = self._currVolumeNode.GetImageData().GetDimensions()
        size = [size[2], size[1], size[0]]
        resFinal = np.zeros(size)
        print("resFinal.shape", resFinal.shape)

        # TODO: mp speed up
        # TODO: background save?
        for segment in self.segments:
            name = segment.GetName()
            res = slicer.util.arrayFromSegmentBinaryLabelmap(
                segmentationNode, self.getSegmentId(segment), self._currVolumeNode
            ).astype("bool")
            resFinal[res] = catgs[name]

        print("Final result ids", np.unique(resFinal))

        scanPath = self._currVolumeNode_scanPath.get(self._currVolumeNode)

        origin = sitk.ReadImage(scanPath)
        mask = sitk.GetImageFromArray(resFinal)
        mask.SetSpacing(origin.GetSpacing())
        mask.SetOrigin(origin.GetOrigin())
        mask.SetDirection(origin.GetDirection())
        mask = sitk.Cast(mask, sitk.sitkUInt8)

        dotPos = scanPath.find(".")
        labelPath = scanPath[:dotPos] + "_label" + scanPath[dotPos:]
        sitk.WriteImage(mask, labelPath)

        # TODO:
        # self._scanPaths.remove(scanPath)
        # self._currScanIdx -= 1
        self.nextScan()

        slicer.util.delayDisplay(f"{labelPath.split('/')[-1]} save successfully", autoCloseMsec=1200)
        print(f"saving took {time.time() - tic}s")

    def setSegmentationOpacity(self):
        if self.segmentationNode is None:
            return
        threshold = self.ui.opacitySlider.value
        displayNode = slicer.util.getNode("EIMedSeg3DSegmentation").GetDisplayNode()
        displayNode.SetOpacity3D(threshold)  # Set opacity for 3d render
        displayNode.SetOpacity(threshold)  # Set opacity for 2d

    """ life cycle related """

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        # print("cleanup")
        self.clearScene()
        self.removeObservers()
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

    def enter(self):
        """
        Called each time the user opens this module. Not when reload/switch back.
        """
        pass

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

    def onSceneEndImport(self, caller, event):
        if self._endImportProcessing:
            return

        self._endImportProcessing = True
        self._endImportProcessing = False

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self._currVolumeNode = None
        self.saveOrReadCurrIdx(saveFlag=True)

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
        self.clearScene()

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeFromNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.setParameterNode(self.logic.getParameterNode())
        segNode = self.segmentationNode
        if segNode is not None:
            self.ui.opacitySlider.sketValue(segNode.GetDisplayNode().GetOpacity())

        # print("initializeParameterNode")
        # self.setParameterNode(self.logic.getParameterNode())
        # if not self._parameterNode.GetNodeReference("InputVolume"):
        #     firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #     if firstVolumeNode:
        #         self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        # if not self._parameterNode.GetNodeReference("InputVolume"):
        #     firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #     if firstVolumeNode:
        #         self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

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
        logging.info(f"Processing completed in {stopTime - startTime:.2f} seconds")

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
