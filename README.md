## Heterogeneous parallel computing based real-time chaotic video encryption and its application to secure drone communication

### Development Enviroment

* Operating System     : Ubuntu 20.04
* OpenCV               : 4.5.2
* CUDA                 : 11.2
* cmake                : 3.16.3
* Programming Language : C/C++

### File Description

The original video is located in the video directory. The number of byte generator threads, rounds of confusion and diffusion oprations, etc., are declared in the kernel.cuh file. To complile the source files and run the program, simply execute the make.sh script.

####  VideoEncryptionDecryption

The program retrieves frames from the original video, encrypts them using a randomly selected key, decrypts the encrypted frames, and displays the original, encrypted, and decrypted frames.

####  videoSecureCommunication

The client retrieves frames from the original video, encrypts them with a key, displays both the original and encrypted frames, encodes the encrypted frames into PNG format, and transmits them to the server over the Internet. The server decodes the obtained PNG files, decrypts the resulting frames using the same key, and displays the obtained and decrypted frames.

### Recommended setting:

* the width of the frame == the height of the frame
* CONFUSION_BLOCK_ROWS_COLS = the width of the frame / CONFUSION_THREAD_ROWS_COLS

