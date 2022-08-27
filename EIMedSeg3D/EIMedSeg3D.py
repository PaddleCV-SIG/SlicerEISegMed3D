import logging
import os
import os.path as osp
import time
from functools import partial

# import sitkUtils
import qt
import ctk
import vtk
import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import slicer

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

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
        print("initializeAfterStartup", slicer.app.commandOptions().noMainWindow)


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
        self._segmentNode = None
        self._currScanIdx = None
        self._segmentEditor = {}
        self._currVolumeNode_scanPath = {}
        self._thresh = 0.9  # output threshold
        self._prev_segId = None

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
        self.ui.threshSlider.connect("valueChanged(double)", self.set_segmentation_opacity)
        # TODO: sync select two paths to scene?

        # Buttons
        self.ui.loadModelButton.connect("clicked(bool)", self.loadModelClicked)
        self.ui.loadScanButton.connect("clicked(bool)", self.loadScans)
        self.ui.nextScanButton.connect("clicked(bool)", self.nextScan)
        self.ui.prevScanButton.connect("clicked(bool)", self.prevScan)
        self.ui.submitLabelButton.connect("clicked(bool)", self.submitLabel)
        self.ui.clearPointButton.connect("clicked(bool)", self.clearAllPoints)

        # Positive/Negative Point
        self.ui.dgPositiveControlPointPlacementWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.dgPositiveControlPointPlacementWidget.placeButton().toolTip = "Select positive points"
        self.ui.dgPositiveControlPointPlacementWidget.placeButton().show()

        self.ui.dgNegativeControlPointPlacementWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.dgNegativeControlPointPlacementWidget.placeButton().toolTip = "Select negative points"
        self.ui.dgNegativeControlPointPlacementWidget.placeButton().show()

        # Segment editor
        self.ui.embeddedSegmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.embeddedSegmentEditorWidget.setMRMLSegmentEditorNode(self.logic.get_segment_editor_node())
        # segmentEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
        # segmentEditorWidget.setEffectNameOrder(["Paint", "Erase"])
        # segmentEditorWidget.unorderedEffectsVisible = False

        self.initializeParameterNode()

        # Set place point widget colors
        # TODO: move to initializeParameterNode
        self.ui.dgPositiveControlPointPlacementWidget.setNodeColor(qt.QColor(0, 255, 0))
        self.ui.dgNegativeControlPointPlacementWidget.setNodeColor(qt.QColor(255, 0, 0))
        self.hideDeleteButtons()

    def clearAllPoints(self):
        self.ui.dgPositiveControlPointPlacementWidget.deleteAllPoints()
        self.ui.dgNegativeControlPointPlacementWidget.deleteAllPoints()

    def hideDeleteButtons(self):
        self.ui.dgPositiveControlPointPlacementWidget.deleteButton().hide()
        self.ui.dgNegativeControlPointPlacementWidget.deleteButton().hide()

    def init_params(self):
        "init changble parameters here"
        self.predictor_params_ = {"norm_radius": 2, "spatial_scale": 1.0}
        self.ratio = (512 / 880, 512 / 880, 12 / 12)  # xyz 这个形状与训练的对数据预处理的形状要一致，怎么切换不同模型？ todo： 在模块上设置预处理形状。和模型一致
        self.train_shape = (512, 512, 12)
        self.image_ww = (0, 2650)  # low, high range for image crop
        self.test_iou = False  # the label file need to be set correctly
        self.file_suffix = [".nii", ".nii.gz"]  # files with these suffix will be loaded
        self.device, self.enable_mkldnn = "gpu", True

    def loadModelClicked(self):
        model_path, param_path = self.ui.modelPathInput.currentPath, self.ui.paramPathInput.currentPath
        if not model_path or not param_path:
            slicer.util.errorDisplay("Please set the model_path and parameter path before load model.")
            return

        self.inference_predictor = predictor.BasePredictor(
            model_path, param_path, device=self.device, enable_mkldnn=self.enable_mkldnn, **self.predictor_params_
        )

        slicer.util.delayDisplay("Sucessfully loaded model to {}!".format(self.device), autoCloseMsec=1500)

    def loadScans(self):
        """Get all the scans under a folder and turn to the first one"""
        dataFolder = self.ui.dataFolderLineEdit.currentPath
        if dataFolder is None or len(dataFolder) == 0:
            # test remove
            dataFolder = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.join("test_MR", "Case2.nii"))
            slicer.util.delayDisplay("Please select a Data Folder first!", autoCloseMsec=5000)
            return

        self.clearScene()

        # list files in assigned directory
        self._dataFolder = dataFolder
        paths = os.listdir(self._dataFolder)
        paths = sorted([s for s in paths if s.split(".")[0][-len("_label") :] != "_label"])
        paths = [osp.join(self._dataFolder, s) for s in paths]

        self._scanPaths = [p for p in paths if p[p.find(".") :] in self.file_suffix]

        slicer.util.delayDisplay(
            "Successfully loaded {} scans! \nPlease press on next scan to show them!".format(len(self._scanPaths)),
            autoCloseMsec=3000,
        )

        if osp.exists(osp.join(self._dataFolder, "currScanIdx.txt")):
            self.saveOrReadCurrIdx(saveFlag=False)
        else:
            self._currScanIdx = None

        self.nextScan()
        # test
        print("All scan paths", self._scanPaths)

    def clearScene(self):
        self.hideDeleteButtons()
        if self._currVolumeNode is not None:
            slicer.mrmlScene.RemoveNode(self._currVolumeNode)
        if self._segmentNode is not None:
            slicer.mrmlScene.RemoveNode(self._segmentNode)

    def saveOrReadCurrIdx(self, saveFlag=False):
        path = os.path.join(self._dataFolder, "currScanIdx.txt")
        if saveFlag:
            with open(osp.join(self._dataFolder, path), "w") as f:
                f.write(str(self._currScanIdx + 1))
        else:
            with open(osp.join(self._dataFolder, path), "r") as f:
                self._currScanIdx = int(f.read())

    def nextScan(self):
        if len(self._scanPaths) == 0:
            slicer.util.errorDisplay(
                "You have marked all the data, and there is no next scan. Please reselect the file path and click the Load Scans button."
            )
            return
        if self._currScanIdx is None:
            self._currScanIdx = 0
        else:
            if self._currScanIdx + 1 >= len(self._scanPaths):
                self._currScanIdx -= len(self._scanPaths) - 1
            self._currScanIdx += 1

        self.turnTo()

    def prevScan(self):
        if len(self._scanPaths) == 0:
            slicer.util.errorDisplay(
                "You have marked all the data, and there is no next scan. Please reselect the file path and click the Load Scans button."
            )
            return
        if self._currScanIdx is None:
            self._currScanIdx = 0
        else:
            if self._currScanIdx - 1 < 0:
                self._currScanIdx += len(self._scanPaths) - 1
            else:
                self._currScanIdx -= 1
        self.turnTo()

    def turnTo(self):
        """
        Turn to the self._currScanIdx th scan, load scan and label
        """
        # sync the current increase/removed label
        if self._segmentNode is not None:
            self.segmentation2Labelnode(self._segmentNode.GetSegmentation())

        self.clearScene()  # 切图时就clear所有当前volume node 和 segmentation node

        if len(self._scanPaths) == 0:
            slicer.util.delayDisplay("No scan found, please load scans first.", autoCloseMsec=2000)
            return

        # 1. load new scan & preprocess
        image_path = self._scanPaths[self._currScanIdx]
        self._currVolumeNode = slicer.util.loadVolume(image_path)
        self.hideDeleteButtons()
        self._currVolumeNode_scanPath[self._currVolumeNode] = image_path

        # BUG: load image before loading model
        # self.inference_predictor.original_image = None

        # 2. load or create segmentation
        # todo if osp.exists(self._scanPaths[scanIdx]):
        dot_pos = image_path.find(".")
        self._currLabelPath = image_path[:dot_pos] + "_label" + image_path[dot_pos:]
        if osp.exists(self._currLabelPath):
            self._segmentNode = slicer.modules.segmentations.logic().LoadSegmentationFromFile(
                self._currLabelPath, False
            )
        else:
            self._segmentNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode"
            )  # Add segment node and segmentation

        self._segmentNode.SetName("EIMedSeg3DSegmentation")
        self._segmentNode.SetReferenceImageGeometryParameterFromVolumeNode(self._currVolumeNode)

        # 3. create category label from txt and segmentation
        self.catgTxt2Segmentation(self._segmentNode.GetSegmentation())

        # 4. set the editor as current result.
        self.ui.embeddedSegmentEditorWidget.setSegmentationNode(self._segmentNode)
        self.ui.embeddedSegmentEditorWidget.setMasterVolumeNode(self._currVolumeNode)

        self.ui.dgPositiveControlPointPlacementWidget.setEnabled(True)
        self.ui.dgNegativeControlPointPlacementWidget.setEnabled(True)

    def segmentation2Labelnode(self, segmentation):
        for segId in segmentation.GetSegmentIDs():
            print("sync from the segmentation", segId)
            segment = segmentation.GetSegment(segId)
            self._segmentEditor[segId] = LabelNode(
                segmentId=segId, labelValue=segment.GetLabelValue(), name=segment.GetName(), color=segment.GetColor()
            )

    def catgTxt2Segmentation(self, segmentation):
        """
        Sync category information from labels.txt to segmentation, and make sure the labelValue is correct
        Record current segments in labelNodes.
        Args:
            segmentation (_type_): _description_
        """
        if self._segmentEditor == {}:
            print("initialize the editor from txt.")
            self._labelValues = []
            # 1. get catg info from txt and segmentation
            txt_catgs = self.getCatgFromTxt()
            logging.info(f"txt_catgs: {txt_catgs}")  # revise data structure

            curr_catgs = self.getCatgFromSegmentation(segmentation)
            logging.info(f"curr_catgs: {curr_catgs}")

            # 2. create and modify info in segmentation sync segment from txt infor
            for Name in set(txt_catgs.keys()) - set(curr_catgs.keys()):
                segmentation.AddEmptySegment("", Name, txt_catgs[Name]["color"])

            # 3. sync label value from txt
            for segIdx in segmentation.GetSegmentIDs():
                segment = segmentation.GetSegment(segIdx)
                name = segment.GetName()
                if name in txt_catgs.keys():
                    segment.SetLabelValue(txt_catgs[name]["labelValue"])
                    segment.SetColor(tuple([round(item, 6) for item in np.divide(txt_catgs[name]["color"], 255)]))
        else:
            print("sync from segmenteditor")
            for segId in self._segmentEditor.keys():
                labelNode = self._segmentEditor[segId]
                segmentation.AddEmptySegment(segId, labelNode.name, labelNode.color)
                segment = segmentation.GetSegment(segId)
                segment.SetLabelValue(labelNode.labelValue)
                segment.SetColor(labelNode.color)

            self._segmentEditor.clear()

    def getCatgFromSegmentation(self, segmentation):
        """Get category info from a segmentation

        Args:
            segmentation (_type_): _description_

        Returns:
            dict: {"name": {labelValue: int, segmentName: str, "color": [color_r, color_g, color_b] }, ... } (color is 0~255)
        """
        catgs = []

        for segId in segmentation.GetSegmentIDs():
            segment = segmentation.GetSegment(segId)
            labelValue = max(self._labelValues) + 1
            catgs.append(
                [
                    labelValue,  # BUG: use GetLabelValue
                    segment.GetName(),
                    *[int(v * 255) for v in segment.GetColor()],
                ]
            )
            self._labelValues.append(labelValue)

        return {c[1]: {"labelValue": c[0], "color": c[2:5]} for c in catgs}

    def getCatgFromTxt(self):
        """Get category info from labelx.txt

        Returns:
            dict: same as getCatgFromSegmentation
        """
        txt_path = osp.join(self._dataFolder, "labels.txt")
        if not osp.exists(txt_path):
            return {}

        with open(txt_path, "r") as f:
            lines = f.readlines()
            infor_dict = {}
            for info in lines:
                infor = info.split(" ")
                if int(infor[0]) in self._labelValues:
                    slicer.util.delayDisplay(
                        "Label value {} of category {} has appeared before, please check your labels.txt do not have repeat label values.".format(
                            infor[0], infor[1]
                        )
                    )
                else:
                    self._labelValues.append(int(infor[0]))
                infor[2:5] = map(int, infor[2:5])
                # print("infor1", infor[2:5])
                infor_dict[infor[1]] = {"labelValue": int(infor[0]), "color": infor[2:5]}

        return infor_dict

    def onControlPointAdded(self, observer, eventid):
        self.hideDeleteButtons()
        posPoints = self.getControlPointsXYZ(self.dgPositivePointListNode, "positive")
        negPoints = self.getControlPointsXYZ(self.dgNegativePointListNode, "negative")

        newPointIndex = observer.GetDisplayNode().GetActiveControlPoint()
        newPointPos = self.getControlPointXYZ(observer, newPointIndex)
        isPositivePoint = False if len(posPoints) == 0 else newPointPos == posPoints[-1]

        slicer.util.delayDisplay(
            "A {} point have been added on {}".format(["negative", "positive"][isPositivePoint], newPointPos),
            autoCloseMsec=1200,
        )
        paddle.device.cuda.empty_cache()
        self.ignorePointListNodeAddEvent = True

        # maybe run inference here
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            segmentation = self._segmentNode.GetSegmentation()
            segmentId = self.ui.embeddedSegmentEditorWidget.currentSegmentID()
            segment = segmentation.GetSegment(segmentId)
            print("Current labelvalue is ", segment.GetLabelValue())

            if self._prev_segId is None:
                self._prev_segId = segmentId
                self.set_image()
            elif self._prev_segId != segmentId:
                self.set_image()
                self._prev_segId = segmentId
            else:
                if self.inference_predictor.original_image is None:
                    self.set_image()

            # get current seg mask as numpy
            res = slicer.util.arrayFromSegmentBinaryLabelmap(self._segmentNode, segmentId, self._currVolumeNode)
            self.ui.progressBar.setValue(10)

            # test
            # p = newPointPos
            # p = [p[2], p[1], p[0]]
            # res[p[0] - 10 : p[0] + 10, p[1] - 10 : p[1] + 10, p[2] - 10 : p[2] + 10] = 1
            # mask = res
            # !! predict image for test
            mask = self.infer_image(newPointPos, isPositivePoint)  # (880, 880, 12) same as res
            self.ui.progressBar.setValue(100)

            if self.test_iou:
                label = sitk.ReadImage(self._currLabelPath)
                label = sitk.GetArrayFromImage(label).astype("int32")
                iou = self.get_iou(label, mask, newPointPos)
                print("Current IOU is {}".format(iou))

            # set new numpy mask to segmentation
            slicer.util.updateSegmentBinaryLabelmapFromArray(mask, self._segmentNode, segmentId, self._currVolumeNode)

        self.ignorePointListNodeAddEvent = False

    def get_iou(self, gt_mask, pred_mask, newPointPos, ignore_label=-1):
        ignore_gt_mask_inv = gt_mask != ignore_label
        pred_mask = pred_mask == 1
        obj_gt_mask = gt_mask == gt_mask[newPointPos[2], newPointPos[1], newPointPos[0]]

        intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
        union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

        return intersection / union

    def set_image(self):
        # print("current image is ", self._scanPaths[self._currScanIdx])
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

        print(f"输入网络前数据的形状:{input_data.shape}")  # shape (1, 512, 512, 12)

        try:
            self.inference_predictor.set_input_image(input_data)
        except AttributeError:
            slicer.util.delayDisplay("Please load model first", autoCloseMsec=1200)

        self.clicker = Clicker()

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

            # logging.info(f"RAS: {coord}; WORLD: {world}; IJK: {p_Ijk}")
            point_set.append(p_Ijk[0:3])

        logging.info(f"{name} => Current control points: {point_set}")
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

    def infer_image(self, click_position=None, positive_click=True, pred_thr=0.49):
        """
        click_position: one or serveral clicks represent by list like: [[234, 284, 7]]
        positive_click: whether this click is positive or negative
        """
        try:
            paddle.device.set_device(self.device)
        except AttributeError:
            slicer.util.errorDisplay("The model is not loaded, Please press load model first")
            return

        a = time.time()
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
        b = time.time()
        print(f"预测结果的形状：{output_data.shape}, 预测时间为 {(b - a) * 1000} ms")  # shape (12, 512, 512) DHW test

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

    def submitLabel(self):
        """
        save the file to the current path
        """
        # 1. find all segmentid in current segmentation and save them to nii
        segmentation = self._segmentNode.GetSegmentation()
        resFinal = None
        for segId in segmentation.GetSegmentIDs():
            segment = segmentation.GetSegment(segId)
            labelValue = segment.GetLabelValue()
            print("labelValue", labelValue)

            # get current seg mask as numpy
            res = slicer.util.arrayFromSegmentBinaryLabelmap(self._segmentNode, segId, self._currVolumeNode)
            # res *= labelValue
            if resFinal is None:
                resFinal = np.zeros(shape=res.shape)
            resFinal[res == 1] = labelValue
            print("segid", labelValue, np.unique(resFinal), np.unique(res))

        print("Max of final result", np.unique(resFinal), resFinal.shape)

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
        self._scanPaths.remove(scanPath)
        self._currScanIdx -= 1
        self.nextScan()

        slicer.util.delayDisplay(f"{labelPath.split('/')[-1]} save successfully", autoCloseMsec=1200)

    def onSceneEndImport(self, caller, event):
        if self._endImportProcessing:
            return

        self._endImportProcessing = True
        self._endImportProcessing = False

    def set_segmentation_opacity(self):
        segmentation = slicer.util.getNode("EIMedSeg3DSegmentation")
        threshold = self.ui.threshSlider.value
        displayNode = segmentation.GetDisplayNode()
        displayNode.SetOpacity3D(threshold)  # Set opacity for 3d render
        displayNode.SetOpacity(threshold)  # Set opacity for 2d

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
        Called each time the user opens this module.
        """
        # print("enter")
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
        # ("onSceneStartClose")
        # Parameter node will be reset, do not use it anymore
        self._currVolumeNode = None
        self._segmentNode = None
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
        # print("initializeParameterNode")
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
        self.hideDeleteButtons()

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

    # def onEditControlPoints(self, pointListNode, tagName):
    #     if pointListNode is None:
    #         return

    #     pointListNode.RemoveAllControlPoints()
    #     segmentId, segment = self.currentSegment()
    #     if segment and segmentId:
    #         v = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
    #         IjkToRasMatrix = vtk.vtkMatrix4x4()
    #         v.GetIJKToRASMatrix(IjkToRasMatrix)

    #         fPosStr = vtk.mutable("")
    #         segment.GetTag(tagName, fPosStr)
    #         pointset = str(fPosStr)
    #         logging.debug(f"{segmentId} => {segment.GetName()} Control points are: {pointset}")

    #         if fPosStr is not None and len(pointset) > 0:
    #             points = json.loads(pointset)
    #             for p in points:
    #                 p_Ijk = [p[0], p[1], p[2], 1.0]
    #                 p_Ras = IjkToRasMatrix.MultiplyDoublePoint(p_Ijk)
    #                 logging.debug(f"Add Control Point: {p_Ijk} => {p_Ras}")
    #                 pointListNode.AddControlPoint(p_Ras[0:3])

    # def onSelectLabel(self, caller=None, event=None):
    #     self.updateParameterNodeFromGUI(caller, event)
    #     self.ignorePointListNodeAddEvent = True
    #     self.onEditControlPoints(self.dgPositivePointListNode, "MONAILabel.ForegroundPoints")
    #     self.onEditControlPoints(self.dgNegativePointListNode, "MONAILabel.BackgroundPoints")
    #     self.ignorePointListNodeAddEvent = False


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
