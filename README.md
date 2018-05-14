# Object-Oriented Super-Resolution Imaging

### Authors

* [**Xinsheng(Sean) Zhang**](https://github.com/nyuxz)
* [**Binqian(Eric) Zeng**](http://github.com/bz866)

### Pipeline Overview
<img src="Pipeline_Overview_Horizontal.png" width="700">

### Usage
- The `ssd` and the `srgan` directory should be mannully removed under the `Object-orientedImageDeblurringPipelineWorkSpace` directory, as 
   * `Object-orientedImageDeblurringPipelineWorkSpace\ssd`
   * `\Object-orientedImageDeblurringPipelineWorkSpace\srgan`. 
- In following steps, we assume that users have mannually moved directories as mentioned. 

#### Module-1 
The SSD code is based on (https://github.com/amdegroot/ssd.pytorch).  
#### Download a pre-trained SSD network
- Pre-trained model for SSD are provided in:
    * SSD300 trained on VOC0712 (newest PyTorch weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
    * SSD300 trained on VOC0712 (original Caffe weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth
- Models should be downloaded in the `ssd/weights` dir:
```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```


#### Requirements
VOC Dataset
PASCAL VOC: Visual Object Classes

Download VOC2007 trainval & test
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
Download VOC2012 trainval
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>



SSD code based on \cite{SSDGithub}, and trained model on PascalVOC2012 benchmark dataset. 

## References
- Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C
Berg. Ssd: Single shot multibox detector. In European conference on computer vision, pp. 21–37. Springer,
2016.((http://arxiv.org/abs/1512.02325))
- Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and superresolution.
In European Conference on Computer Vision, pp. 694–711. Springer, 2016.((http://arxiv.org/abs/1603.08155))
- Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel Rueckert, and
Zehan Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional
neural network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp.
1874–1883, 2016.((https://arxiv.org/abs/1609.05158)). 
- Jiwon Kim, Jung Kwon Lee, and Kyoung Mu Lee. Deeply-recursive convolutional network for image superresolution.
In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1637–1645,
2016.((https://arxiv.org/abs/1511.04491))
- Christian Ledig, Lucas Theis, Ferenc Huszár, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew
Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, et al. Photo-realistic single image super-resolution
using a generative adversarial network. arXiv preprint, 2016.((https://arxiv.org/abs/1609.04802))

