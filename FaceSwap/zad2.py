import os
import dlib
import cv2
import numpy as np

import models
import NonLinearLeastSquares
import ImageProcessing

from drawing import *

import FaceRendering
import utils

print "Press T to draw the keypoints and the 3D model"
print "Press R to start recording to a video file"

#you need to download shape_predictor_68_face_landmarks.dat from the link below and unpack it where the solution file is
#http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

#loading the keypoint detection model, the image and the 3D model
project_dir = os.path.dirname(os.path.realpath(__file__))
predictor_path = os.path.join(project_dir, "..", "shape_predictor_68_face_landmarks.dat")
candide_path = os.path.join(project_dir, "..", "candide.npz")
texture_image_path = os.path.join(project_dir, "..", "data", "jolie.jpg")
target_image_path = os.path.join(project_dir, "..", "data", "target.jpg")
#the smaller this value gets the faster the detection will work
#if it is too small, the user's face might not be detected
maxImageSizeForDetection = 320

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel(candide_path)

projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

modelParams = None
lockedTranslation = False
drawOverlay = False
targetImg = cv2.imread(target_image_path)
textureImg = cv2.imread(texture_image_path)
textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
renderer = FaceRendering.FaceRenderer(targetImg, textureImg, textureCoords, mesh)

shapes2D = utils.getFaceKeypoints(targetImg, detector, predictor, maxImageSizeForDetection)

if shapes2D is not None:
    for shape2D in shapes2D:
        #3D model parameter initialization
        modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

        #3D model parameter optimization
        modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

        #rendering the model to an image
        shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
        renderedImg = renderer.render(shape3D)

        #blending of the rendered face with the image
        mask = np.copy(renderedImg[:, :, 0])
        renderedImg = ImageProcessing.colorTransfer(targetImg, renderedImg, mask)
        targetImg = ImageProcessing.blendImages(renderedImg, targetImg, mask)


        #drawing of the mesh and keypoints
        if drawOverlay:
            drawPoints(targetImg, shape2D.T)
            drawProjectedShape(targetImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)

