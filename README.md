# EdgeTpuTesting
Scripts for testing Edge TPU and TFLite models.

```test_edgetpu.ipynb``` is a script that attempst to build a very simple model, convert it to tflite, and compile it for use on the Edge TPU.

```demo_conv_batchnorm_fusing.ipynb``` demonstrates how a batchnorm layer can be fused into a preceding or following convolutional layer. It also highlights that there will be errors at the edges of the output tensors if the convolutional layer follows the batchnorm layer and zero-padding is used.
