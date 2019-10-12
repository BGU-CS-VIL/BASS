![image alt >](https://github.com/uzielroy/BASS/blob/master/gif/vis_2.gif?raw=True)

# Bayesian Adaptive Superpixel Segmentation

This is a placeholder for the code from our ICCV 2019 paper, "Bayesian Adaptive Superpixel Segmentation", co-authored by Roy Uziel, Meitar Ronen, and Oren Freifeld.

# Installation

The code uses Python 3.6 and it was tested on Pytorch 1.3.0

Install pip and virtualenv
```
sudo apt-get install python-pip python-virtualenv
```

Clone the git project:
```
$ git clone https://github.com/BGU-CS-VIL/BASS.git
```

Set up virtual environment:
```
$ mkdir <your_home_dir>/.virtualenvs
$ virtualenv <your_home_dir>/.virtualenvs/BASS
```

Activate virtual environment:
```
$ cd BASS
$ source <your_home_dir>/BASS/bin/activate
```

The requirements can be installed using:
```
pip install -r requirements.txt
```

# Usage

Saving csv file as output
```
python BASS --img_folder /path/to/image/folder --csv
```
Saving mean colors and contours images
```
python BASS --img_folder /path/to/image/folder --vis
```

# License

This software is released under the MIT License (included with the software). Note, however, that if you are using this code (and/or the results of running it) to support any form of publication (e.g., a book, a journal paper, a conference paper, a patent application, etc.) then we request you will cite our paper:

```
@inproceedings{Uziel:ICCV:2019:BASS,
  title = {Bayesian Adaptive Superpixel Segmentation},
  author = {Roy Uziel and Meitar Ronen and Oren Freifeld},
  booktitle = {ICCV},
  year={2019}
 } 
