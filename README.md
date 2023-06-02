## Heterogeneous parallel computing based real-time chaotic video encryption and its application to secure drone communication

### Development Enviroment

* Operating System     : Ubuntu 20.04
* OpenCV               : 4.5.2
* CUDA                 : 11.2
* cmake                : 3.16.3
* Programming Language : C/C++

### File Description

The original video and the source files are stored in video and source directories, respectively. The number of byte generator threads, rounds of confusion and diffusion operations, etc., are declared in source/kernel.cuh. Directly execute source/make.sh script, which will automatically compile the source files, and run the demo.

Recommended setting:

* the width of the frame == the height of the frame
* CONFUSION_BLOCK_ROWS_COLS = the width of the frame / CONFUSION_THREAD_ROWS_COLS

