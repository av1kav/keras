# imaging.py - Fast custom imaging via webcam
This fully custom-built module uses an available camera or imaging device (such as the webcam of a laptop) to produce images. It takes approx 4 images a second and is useful for generating custom datasets. It contains:

1. A loop for imaging
2. A cleanup() function which has:
  * A loop to delete all files having a '0' in them (this *needs* changing)
  * A mechanism to neatly sort groups of images into a folder structure of label names

Note that the imaging module is built for Windows and uses a \\ character.
