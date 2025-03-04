# r3kit
Real Robot Research Kit

## Setup
```bash
conda create -n rrr python=3.10
conda activate rrr

git clone git@github.com:ForceMimic/r3kit.git

cd r3kit
pip install -e .
```

## Usage
```python
import r3kit
from r3kit.devices.camera.realsense.l515 import L515

camera = L515()
image = camera.get()
```

```python
import r3kit
from r3kit.devices.ftsensor.ati.pyati import PyATI

ftsensor = PyATI()
data = ftsensor.get()
```

```python
import r3kit
from r3kit.algos.calib.chessboard import ChessboardExtCalibor

calibor = ChessboardExtCalibor()
calibor.add_image(img)
w2c = calibor.run()
```

```python
import r3kit
from r3kit.algos.tare.linear import LinearMFTarer

tarer = LinearMFTarer()
tarer.add_data(f, pose)
tare = tarer.run()
```
