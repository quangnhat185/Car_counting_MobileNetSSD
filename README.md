# Car counting with OpenCV and Deep Learning

<p align = "center"><img src="./processe_video.gif" width = 1024></p>

## Set up environment
```
cd Car_counting_MobileNetSSD
.\env\Scripts\activate.bat
pip install -r requirements.txt
```

## Run the script

Without saving processed footage:
```
python car_counting.py -v test_video.mp4
```

Saving processed footage:
```
python car_counting.py -v test_video.mp4 -s True
```
## Usages

```
optional arguments:
  -h, --help            show this help message and exit
  -v VIDEO, --video VIDEO
                        Path to input video
  -c CONFIG, --config CONFIG
                        Path to the input configuration file
  -s SAVE, --save SAVE  Save processed video (True/False)
```

## Credit
[Adrian Rosebrok - Pyimagesearch: OpenCV Vehicle Detection, Tracking, and Speed Estimation](https://www.pyimagesearch.com/2019/12/02/opencv-vehicle-detection-tracking-and-speed-estimation/)

## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

[MIT License](./LICENSE)
