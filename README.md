# Robot Vision Project

This is a robot vision project.
It uses camera images and depth images to find objects and recognize them.
The robot can send object information to another program by RPC.

## Python version

This project is written for **Python 2.7**.


## Install libraries

Install the libraries from the requirements file:

```bash
pip install -r requirment.txt
```

The project uses NumPy, OpenCV, RealSense, msgpack-rpc, scikit-learn,
and Pyro4.

## Run the program

Use Python 2.7 to start the main file:

```bash
python2.7 main.py
```

The camera and the required robot services must be available before running
the program. The program uses camera color, depth, and CAD images for object
detection.

## Main work

The program does these steps:

1. Start the camera.
2. Read color and depth images.
3. Find object shapes in the images.
4. Create image features.
5. Recognize objects with a trained model.
6. Send object position and direction information.

