# opmap

光学マッピング解析ツール

<img src="opmap.png">

## Install

* Anaconda 2.4.0をインストール
  - Linuxならpyenvが簡単
  - Windowsならインストーラが簡単

* pythonのバージョン確認

```
> python -V
Python 2.7.13 :: Anaconda 2.4.0 (x86_64)
```

* opencvのインストール

condaコマンドを使ってインストール
(Windowsの場合、Anaconda Promptを使う)

```
conda install -c https://conda.binstar.org/jjhelmus opencv
```

* opmapモジュールのインストール

```
> git clone https://github.com/chenaoki/opmap.git
> cd opmap
> python setup.py install
```

* import test

```python
import opmap
from opmap.VmemMap import VmemMap

import cv2
```

## How to start

```
> cd apps/opapp
> python qt_main.py
```
