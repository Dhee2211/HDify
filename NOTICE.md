# Third-party notices

HDify downloads and runs the following at runtime. None of it is bundled in this repository.

| Component | License | Source |
| --- | --- | --- |
| Real-ESRGAN (model weights `RealESRGAN_x2plus`, `RealESRGAN_x4plus`) | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN |
| ONNX exports of those weights | BSD-3-Clause | https://huggingface.co/SceneWorks/real-esrgan-onnx |
| RestoreFormer++ (face restoration, ONNX export `restoreformer_pp_int8w`) | Apache-2.0 | https://github.com/wzhouxiff/RestoreFormerPlusPlus, https://huggingface.co/Saimon8420/restoreformer-pp-web |
| YuNet face detector (`face_detection_yunet_2023mar`) | MIT | https://github.com/opencv/opencv_zoo, https://github.com/ShiqiYu/libfacedetection |
| ONNX Runtime Web | MIT | https://github.com/microsoft/onnxruntime |

Real-ESRGAN: Copyright (c) 2021, Xintao Wang. All rights reserved.

RestoreFormer++ was trained on the FFHQ face dataset. The model weights are redistributed by the publisher under Apache-2.0; if you plan commercial use, review the upstream project and dataset terms yourself.

Example photos in `examples/` are public domain (CC0); see `examples/CREDITS.md`.

HDify does **not** use CodeFormer or its weights, which are licensed for non-commercial use only.
