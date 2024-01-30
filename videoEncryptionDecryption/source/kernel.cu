#include "kernel.cuh"

extern int redChannelConfusionSeed, greenChannelConfusionSeed, blueChannelConfusionSeed;
extern sem_t startToGenerateBytesMutex[BYTE_GENERATOR_THREADS], bytesAreGeneratedMutex[BYTE_GENERATOR_THREADS];

//perform confusion operation
__global__ void confusion(uchar3 * d_encryptedFrame, uchar3 * d_tempFrame, int frameWidth, 
                          int redChannelConfusionSeed,  int greenChannelConfusionSeed, 
                          int blueChannelConfusionSeed){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y; 

    double PI = 3.1415926;

    int xn = ( x + y) % frameWidth;
    int temp = round(redChannelConfusionSeed * sin(2 * PI * xn / frameWidth)); 
    int yn = ((y + temp) % frameWidth + frameWidth) % frameWidth;
    d_encryptedFrame[yn * frameWidth + xn].x = d_tempFrame[y * frameWidth + x].x;

    xn = ( x + y) % frameWidth;
    temp = round(greenChannelConfusionSeed * sin(2 * PI * xn / frameWidth)); 
    yn = ((y + temp) % frameWidth + frameWidth) % frameWidth;
    d_encryptedFrame[yn * frameWidth + xn].y = d_tempFrame[y * frameWidth + x].y;

    xn = ( x + y) % frameWidth;
    temp = round(blueChannelConfusionSeed * sin(2 * PI * xn / frameWidth)); 
    yn = ((y + temp) % frameWidth + frameWidth) % frameWidth;
    d_encryptedFrame[yn * frameWidth + xn].z = d_tempFrame[y * frameWidth + x].z;
}

//perform inverse transformation of confusion
__global__ void inverseConfusion(uchar3 * d_decryptedFrame, uchar3 * d_tempFrame, int frameWidth, 
                          int redChannelConfusionSeed, int greenChannelConfusionSeed, 
                          int blueChannelConfusionSeed){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y; 

    double PI = 3.1415926;

    int temp = round(redChannelConfusionSeed * sin(2 * PI * x / frameWidth)); 
    int xn = ((x - y + temp) % frameWidth + frameWidth) % frameWidth;  
    int yn = ((y - temp) % frameWidth + frameWidth) % frameWidth;
    d_decryptedFrame[yn * frameWidth + xn].x = d_tempFrame[y * frameWidth + x].x;

    temp = round(greenChannelConfusionSeed * sin(2 * PI * x / frameWidth)); 
    xn = ((x - y + temp) % frameWidth + frameWidth) % frameWidth;  
    yn = ((y - temp) % frameWidth + frameWidth) % frameWidth;
    d_decryptedFrame[yn * frameWidth + xn].y = d_tempFrame[y * frameWidth + x].y;

    temp = round(blueChannelConfusionSeed * sin(2 * PI * x / frameWidth)); 
    xn = ((x - y + temp) % frameWidth + frameWidth) % frameWidth;  
    yn = ((y - temp) % frameWidth + frameWidth) % frameWidth;
    d_decryptedFrame[yn * frameWidth + xn].z = d_tempFrame[y * frameWidth + x].z;                     
}

//perform horizontal diffusion operation
__global__ void l2rDiffusion(uchar3 * d_encryptedFrame, uchar3 * d_tempFrame, unsigned char * d_byteSequence, int frameWidth){
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int y = 0;

    unsigned char redChannelDiffusionSeed, greenChannelDiffusionSeed, blueChannelDiffusionSeed;
    
    for(y = 0; y < frameWidth; y++){
        //obtain the diffusion seeds
        if(y == 0){
            redChannelDiffusionSeed   = d_tempFrame[(x + 1) * frameWidth - 1].x;
            greenChannelDiffusionSeed = d_tempFrame[(x + 1) * frameWidth - 1].y;
            blueChannelDiffusionSeed  = d_tempFrame[(x + 1) * frameWidth - 1].z;
        }
        else{
            redChannelDiffusionSeed   = d_encryptedFrame[x * frameWidth + y - 1].x;
            greenChannelDiffusionSeed = d_encryptedFrame[x * frameWidth + y - 1].y;
            blueChannelDiffusionSeed  = d_encryptedFrame[x * frameWidth + y - 1].z;
        }

        int pixelIdx = x * frameWidth + y;

        //perform diffusion operations
        d_encryptedFrame[pixelIdx].x = d_byteSequence[x * frameWidth * 3 + y]       ^ ((d_tempFrame[pixelIdx].x + d_byteSequence[x * frameWidth * 3 + y])      % 256)  ^ redChannelDiffusionSeed;
        d_encryptedFrame[pixelIdx].y = d_byteSequence[(3 * x + 1) * frameWidth + y] ^ ((d_tempFrame[pixelIdx].y + d_byteSequence[(3 * x + 1) * frameWidth + y]) % 256) ^ greenChannelDiffusionSeed;
        d_encryptedFrame[pixelIdx].z = d_byteSequence[(3 * x + 2) * frameWidth + y] ^ ((d_tempFrame[pixelIdx].z + d_byteSequence[(3 * x + 2) * frameWidth + y]) % 256) ^ blueChannelDiffusionSeed;
    }
}

//perform inverse transformation of horizontal diffusion operation
__global__ void l2rInverseDiffusion(uchar3 * d_decryptedFrame, uchar3 * d_tempFrame, unsigned char * d_byteSequence, int frameWidth){
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int y = 0;

    unsigned char redChannelDiffusionSeed, greenChannelDiffusionSeed, blueChannelDiffusionSeed;
    
    for(y = frameWidth - 1; y >= 0; y--){
        //obtain the diffusion seeds
        if(y == 0){
            redChannelDiffusionSeed   = d_decryptedFrame[(x + 1) * frameWidth - 1].x;
            greenChannelDiffusionSeed = d_decryptedFrame[(x + 1) * frameWidth - 1].y;
            blueChannelDiffusionSeed  = d_decryptedFrame[(x + 1) * frameWidth - 1].z;
        }
        else{
            redChannelDiffusionSeed   = d_tempFrame[x * frameWidth + y - 1].x;
            greenChannelDiffusionSeed = d_tempFrame[x * frameWidth + y - 1].y;
            blueChannelDiffusionSeed  = d_tempFrame[x * frameWidth + y - 1].z;
        }

        int pixelIdx = x * frameWidth + y;

        //perform inverse diffusion operations
        d_decryptedFrame[pixelIdx].x = (d_byteSequence[x * frameWidth * 3 + y]      ^ d_tempFrame[pixelIdx].x ^ redChannelDiffusionSeed)   + 256 - d_byteSequence[x * frameWidth * 3 + y];
        d_decryptedFrame[pixelIdx].y = (d_byteSequence[(3 * x + 1) * frameWidth +y] ^ d_tempFrame[pixelIdx].y ^ greenChannelDiffusionSeed) + 256 - d_byteSequence[(3 * x + 1) * frameWidth +y];
        d_decryptedFrame[pixelIdx].z = (d_byteSequence[(3 * x + 2) * frameWidth +y] ^ d_tempFrame[pixelIdx].z ^ blueChannelDiffusionSeed)  + 256 - d_byteSequence[(3 * x + 2) * frameWidth +y];
    }
}

//perform vertical diffusion operation
__global__ void t2bDiffusion(uchar3 * d_encryptedFrame, uchar3 * d_tempFrame, unsigned char * d_byteSequence, int frameWidth){
    int x = 0;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned char redChannelDiffusionSeed, greenChannelDiffusionSeed, blueChannelDiffusionSeed;
    
    for(x = 0; x < frameWidth; x++){
        //obtain the diffusion seeds
        if(x == 0){
            redChannelDiffusionSeed   = d_tempFrame[(frameWidth - 1) * frameWidth + y].x;
            greenChannelDiffusionSeed = d_tempFrame[(frameWidth - 1) * frameWidth + y].y;
            blueChannelDiffusionSeed  = d_tempFrame[(frameWidth - 1) * frameWidth + y].z;
        }
        else{
            redChannelDiffusionSeed   = d_encryptedFrame[(x - 1) * frameWidth + y].x;
            greenChannelDiffusionSeed = d_encryptedFrame[(x - 1) * frameWidth + y].y;
            blueChannelDiffusionSeed  = d_encryptedFrame[(x - 1) * frameWidth + y].z;
        }

        int pixelIdx = x * frameWidth + y;

        //perform diffusion operations
        d_encryptedFrame[pixelIdx].x = d_byteSequence[y * frameWidth * 3 + x]       ^ ((d_tempFrame[pixelIdx].x + d_byteSequence[y * frameWidth * 3 + x])       % 256)  ^ redChannelDiffusionSeed;
        d_encryptedFrame[pixelIdx].y = d_byteSequence[(3 * x + 1) * frameWidth + y] ^ ((d_tempFrame[pixelIdx].y + d_byteSequence[(3 * x + 1) * frameWidth + y]) % 256) ^ greenChannelDiffusionSeed;
        d_encryptedFrame[pixelIdx].z = d_byteSequence[(3 * x + 2) * frameWidth + y] ^ ((d_tempFrame[pixelIdx].z + d_byteSequence[(3 * x + 2) * frameWidth + y]) % 256) ^ blueChannelDiffusionSeed;
    }
}

//perform inverse transformation of diffusion operation
__global__ void t2bInverseDiffusion(uchar3 * d_decryptedFrame, uchar3 * d_tempFrame, unsigned char * d_byteSequence, int frameWidth){
    int x = 0;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned char redChannelDiffusionSeed, greenChannelDiffusionSeed, blueChannelDiffusionSeed;
    
    for(x = frameWidth - 1; x >= 0; x--){
        //obtain the diffusion seeds
        if(x == 0){
            redChannelDiffusionSeed   = d_decryptedFrame[(frameWidth - 1) * frameWidth + y].x;
            greenChannelDiffusionSeed = d_decryptedFrame[(frameWidth - 1) * frameWidth + y].y;
            blueChannelDiffusionSeed  = d_decryptedFrame[(frameWidth - 1) * frameWidth + y].z;
        }
        else{
            redChannelDiffusionSeed   = d_tempFrame[(x - 1) * frameWidth + y].x;
            greenChannelDiffusionSeed = d_tempFrame[(x - 1) * frameWidth + y].y;
            blueChannelDiffusionSeed  = d_tempFrame[(x - 1) * frameWidth + y].z;
        }

        int pixelIdx = x * frameWidth + y;

        //perform inverse diffusion operations
        d_decryptedFrame[pixelIdx].x = (d_byteSequence[y * frameWidth * 3 + x]      ^ d_tempFrame[pixelIdx].x ^ redChannelDiffusionSeed)   + 256 - d_byteSequence[y * frameWidth * 3 + x];
        d_decryptedFrame[pixelIdx].y = (d_byteSequence[(3 * x + 1) * frameWidth +y] ^ d_tempFrame[pixelIdx].y ^ greenChannelDiffusionSeed) + 256 - d_byteSequence[(3 * x + 1) * frameWidth +y];
        d_decryptedFrame[pixelIdx].z = (d_byteSequence[(3 * x + 2) * frameWidth +y] ^ d_tempFrame[pixelIdx].z ^ blueChannelDiffusionSeed)  + 256 - d_byteSequence[(3 * x + 2) * frameWidth +y];
    }
}

extern "C"
void encryptionKernelCaller(uchar3 * d_originalFrame, uchar3 * d_encryptedFrame, uchar3 * d_decryptedFrame, 
                                      unsigned char * h_byteSequence, unsigned char * d_byteSequence, int frameWidth, int memSize, int iterations){
    dim3 confusionBlock(CONFUSION_THREAD_ROWS_COLS, CONFUSION_THREAD_ROWS_COLS);
    dim3 confusionGrid(CONFUSION_BLOCK_ROWS_COLS, CONFUSION_BLOCK_ROWS_COLS);
    dim3 diffusionBlock(1, frameWidth);
    dim3 diffusionGrid(1, 1);

    uchar3 * d_tempFrame = NULL;
    cudaMalloc((void **) & d_tempFrame, memSize);
    cudaMemcpy(d_tempFrame, d_originalFrame, memSize, cudaMemcpyDeviceToDevice);

    //encrypt the original frame
    //perform confusion operation on the original frame
    for(int i = 0; i < CONFUSION_ROUNDS; i++){     
        confusion<<<confusionGrid, confusionBlock>>>(d_encryptedFrame, d_tempFrame, frameWidth,
                                  redChannelConfusionSeed, greenChannelConfusionSeed, blueChannelConfusionSeed);
        cudaDeviceSynchronize();
        cudaMemcpy(d_tempFrame, d_encryptedFrame, memSize, cudaMemcpyDeviceToDevice);
    }  

    //wait for all byte generator threads to complete byte generation
    for(int i = 0; i < BYTE_GENERATOR_THREADS; i++)
        sem_wait(&bytesAreGeneratedMutex[i]);

    //upload the byte sequence to the device
    cudaMemcpy(d_byteSequence, h_byteSequence, iterations * 4 * BYTES_RESERVED * BYTE_GENERATOR_THREADS * sizeof(unsigned char), cudaMemcpyHostToDevice);

    //wake up all byte generator threads to generate byte sequence for encrypting next frame
    for(int i = 0; i < BYTE_GENERATOR_THREADS; i++)
        sem_post(&startToGenerateBytesMutex[i]);

    for(int i = 0; i < DIFUSION_CONFUSION_ROUNDS; i++){
        //perform diffusion operations from left to right
        l2rDiffusion<<<diffusionGrid, diffusionBlock>>>(d_encryptedFrame, d_tempFrame, &d_byteSequence[i * frameWidth * frameWidth * 3 * 2], frameWidth);
        cudaDeviceSynchronize();
        cudaMemcpy(d_tempFrame, d_encryptedFrame, memSize, cudaMemcpyDeviceToDevice);

        //perform diffusion operations from top to bottom
        t2bDiffusion<<<diffusionGrid, diffusionBlock>>>(d_encryptedFrame, d_tempFrame, &d_byteSequence[(2 * i + 1) * frameWidth * frameWidth * 3], frameWidth);
        cudaDeviceSynchronize();
        cudaMemcpy(d_tempFrame, d_encryptedFrame, memSize, cudaMemcpyDeviceToDevice);

        //perform confusion operations
        confusion<<<confusionGrid, confusionBlock>>>(d_encryptedFrame, d_tempFrame, frameWidth,
                                  redChannelConfusionSeed, greenChannelConfusionSeed, 
                                  blueChannelConfusionSeed);
        cudaDeviceSynchronize();
        cudaMemcpy(d_tempFrame, d_encryptedFrame, memSize, cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_tempFrame);
}

extern "C"
void decryptionKernelCaller(uchar3 * d_originalFrame, uchar3 * d_encryptedFrame, uchar3 * d_decryptedFrame, 
                                      unsigned char * h_byteSequence, unsigned char * d_byteSequence, int frameWidth, int memSize, int iterations){
    dim3 confusionBlock(CONFUSION_THREAD_ROWS_COLS, CONFUSION_THREAD_ROWS_COLS);
    dim3 confusionGrid(CONFUSION_BLOCK_ROWS_COLS, CONFUSION_BLOCK_ROWS_COLS);
    dim3 diffusionBlock(1, frameWidth);
    dim3 diffusionGrid(1, 1);

    uchar3 * d_tempFrame = NULL;
    cudaMalloc((void **) & d_tempFrame, memSize);
    cudaMemcpy(d_tempFrame, d_encryptedFrame, memSize, cudaMemcpyDeviceToDevice);

    //decrypt the encrypted frame
    for(int i = 0; i < DIFUSION_CONFUSION_ROUNDS; i++){
        //perform inverse confusion operations
        inverseConfusion<<<confusionGrid, confusionBlock>>>(d_decryptedFrame, d_tempFrame, frameWidth, 
                                  redChannelConfusionSeed, greenChannelConfusionSeed, 
                                  blueChannelConfusionSeed);
        cudaDeviceSynchronize();
        cudaMemcpy(d_tempFrame, d_decryptedFrame, memSize, cudaMemcpyDeviceToDevice);

        //perform inverse diffusion operations
        t2bInverseDiffusion<<<diffusionGrid, diffusionBlock>>>(d_decryptedFrame, d_tempFrame, &d_byteSequence[(DIFUSION_CONFUSION_ROUNDS - i - 1) * frameWidth * frameWidth * 3 * 2 + frameWidth * frameWidth * 3], frameWidth);
        cudaDeviceSynchronize();
        cudaMemcpy(d_tempFrame, d_decryptedFrame, memSize, cudaMemcpyDeviceToDevice);

        //perform inverse diffusion operations
        l2rInverseDiffusion<<<diffusionGrid, diffusionBlock>>>(d_decryptedFrame, d_tempFrame, &d_byteSequence[(DIFUSION_CONFUSION_ROUNDS - i - 1) * frameWidth * frameWidth * 3 * 2], frameWidth);
        cudaDeviceSynchronize();
        cudaMemcpy(d_tempFrame, d_decryptedFrame, memSize, cudaMemcpyDeviceToDevice);
    }

    //perform inverse transform of confusion operation on the frame
    for(int i = 0; i < CONFUSION_ROUNDS; i ++){
        inverseConfusion<<<confusionGrid, confusionBlock>>>(d_decryptedFrame, d_tempFrame, frameWidth, 
                                  redChannelConfusionSeed, greenChannelConfusionSeed, 
                                  blueChannelConfusionSeed);
        cudaMemcpy(d_tempFrame, d_decryptedFrame, memSize, cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(d_tempFrame);
}


