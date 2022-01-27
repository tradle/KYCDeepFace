# KYCDeepFace
KYC face matching project.
## Face Detection
### Description
An ultra light face detector refering to [Link](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

**TODOs:**
- [X] Detection pipeline
- [X] Model conversion to ONNX for inference
- [ ] Add model download link & script to model zoos
- [ ] Evaluation
- [ ] Documentation
## Face Alignment
### Description
Face alignment for cropped faces with landmarks.

**TODOs:**
- [X] Alignment pipeline
- [ ] Documentation
- [ ] Upgrade with SDUNets
## Face Recognition
### Description
Face recognition pipeline with model trained on large dataset.

**TODOs:**
- [X] Pretrained model
- [X] Recognition pipeline
- [X] Model conversion to ONNX for inference
- [ ] NIST ranking
- [ ] Evaluation
- [ ] Documentation
- [ ] CMD tool
- [ ] Dockerization
- [ ] Search at large scale

## Image Processing Pipeline Explained

1. [Input: Image (RGB)](./kycdeepface.py#L114)
2. [Detect faces](./kycdeepface.py#L115) in the input

    1. Transform the input
        2. Use [cv2](./transforms/transforms.py#L142) to resize the input to  [320](./config.py#L23)x[240](./ssd/config/fd_config.py#L51)
        3. [Substract the means](./transforms/transforms.py#L99-L102)
        4. [Normalize using std=1.0](./ssd/data_preprocessing.py#L55)
    3. [Inference](./ssd/predictor.py#L39) through [Single Shot MultiBox Object Detector](https://arxiv.org/abs/1512.02325) to detect faces in images. With original code [from amdegroot](https://github.com/amdegroot/ssd.pytorch) which is a port of [Caffe version](https://github.com/weiliu89/caffe/tree/ssd).
        - Q: Where does the model come from?

    5. [Filter out](./ssd/predictor.py#L54) boxes with a output confidence of [`< 0.01`](./ssd/predictor.py#L8).
    6. [use](./ssd/predictor.py#L58) "[hard](./utils/box_utils.py#L141-L171)" nms (non-maximum suppression) with an [`intersection over union threshold=0.3`](./ssd/config/fd_config.py#L38) to remove overlapping face-boxes likely containing the same face.

3. [Detect the face landmarks](./kycdeepface.py#L102) inside the face boxes

    1. [Expand](./landmark_detector.py#L29-L30) the boxes by factor `x1.5` to have some surrounding pixels around the face.
    2. [Use numpy](./landmark_detector.py#L37) to crop the pixels out of the faces image.
    3. Use [cv2 to resize](./landmark_detector.py#L39) the face pixels to [160x160 pixels](./landmark_detector.py#L16)
    4. Normalize the colors from `0~255` [to `-1.0 ~ +1.0`](./landmark_detector.py#L44)
    5. [Run inference](./landmark_detector.py#L49) against the "[Slim](./core/slim.py#L34-L55)" model.
        - Q: Where does this model come from?
        - Q: Who trained?
        - Q: Paper?

3. [Align the face pixels in the detected face boxes](./kycdeepface.py#L101-L103) using the landmarks.

    1. Use CV2 and [the center](./core/face_landmarks.py#L60) of the landmarks [points for the eyes](./core/face_landmarks.py#L89) to [rotate/scale/move the face](./face_aligner.py#L20-L29) using [cv2.wrapAffine](https://theailearner.com/tag/cv2-warpaffine/).
        - The face will be placed in a [`96 x 112`](./config.py#L65) sized image.
        - The left eye will be at [30% from left and 30% from top](./config.py#L67) - `28.8px x 33.6px`.
        - The right eye will be at the same top offset, [30% from the right](./face_aligner.py#L12) - `67.2px x 33.6px`.

    2. We calculate the [faces angle in 3d space](./core/face_landmarks.py#L101) - _important to identify if the person faces the camera_
        - We use [cv2.solvePnP](https://docs.opencv.org/3.4/d5/d1f/calib3d_solvePnP.html)'s function to [find the reprojection distance](./core/face_landmarks.py#L128-L129).
        - With [cv2.Rodriques](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#void%20Rodrigues(InputArray%20src,%20OutputArray%20dst,%20OutputArray%20jacobian)) we get [the rotation matrix](./core/face_landmarks.py#L130).
        - Using [cv2.decomposeProjectionMatrix](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#void%20decomposeProjectionMatrix(InputArray%20projMatrix,%20OutputArray%20cameraMatrix,%20OutputArray%20rotMatrix,%20OutputArray%20transVect,%20OutputArray%20rotMatrixX,%20OutputArray%20rotMatrixY,%20OutputArray%20rotMatrixZ,%20OutputArray%20eulerAngles)) we get the [`euler_angle`](./core/face_landmarks.py#L132-L135) that is returned in the output result.

        - Q: How did we come up with this?
        - Q: Paper?

4. We [run](./kycdeepface.py#L135) the aligned faces [once normally and once horizontally flipped](./core/utils.py#L63-L64) through the MobileFacenet [implementation](./core/model.py#L81-L101)

    - Q: Paper?
