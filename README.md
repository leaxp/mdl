# My Deep learning (mdl)

My Deep Learning (mdl) is a repository where I'll keep all my learnt example regarding machine learning codes. They will be mostly examples found on web that run and adapt to better works into my brain. Also all examples found here should run in macos using the updated version of Anaconda and the required packages.

For tidness reasons I'll try to mantain all examples organized in a structure that remainds this:

```javascript
    |--- output
    |--- core
    |   |--- __init__.py
    |   |--- classes.py
    |--- train.py
    |--- test.py
```

> Here the **\__init\__.py** defines the core folder as a module. Usually it imports the create classes like this:

>```javascript
# import the necessary packages
from lenet import LeNet
```