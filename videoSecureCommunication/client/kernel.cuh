#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define COMMUNICATION_PORT              55318
#define SERVER_IP_ADDRESS               "192.168.1.102"

#define VIDEO_NAME                      "../../video/foreman_cif.yuv.mp4"
#define CONFUSION_BLOCK_ROWS_COLS       16
#define CONFUSION_THREAD_ROWS_COLS      32

#define CONFUSION_ROUNDS                4
#define DIFUSION_CONFUSION_ROUNDS       3

#define CONFUSION_SEED_UPPER_BOUND      30000
#define CONFUSION_SEED_LOWWER_BOUND     1000

#define BYTES_RESERVED                  6
#define PRE_ITERATIONS                  500
#define BYTE_GENERATOR_THREADS          32

struct byteGeneratorThreadParameter{
    int threadIdx;
    int iterations;
    unsigned char * byteSequence;
    double * initParameterArray;
};

//encrypt the original frame
extern "C"
void encryptionKernelCaller(uchar3 * d_originalFrame, uchar3 * d_encryptedFrame, uchar3 * d_decryptedFrame, 
                                      unsigned char * h_byteSequence, unsigned char * d_byteSequence, int frameWidth, int memSize, int iterations);

//byte generator threads execute this function
static void * byteGeneratorThread(void * arg);

//iterate Lorenz hyper-chaotic map
void iterateLorenzHyperChaoticMap(double * x, double * y, double * z, double * w);

//generate parameters for initializing byte generator threads
void generateParametersForByteGeneration(double * x1, double * y1, double * z1, double * w1, double * x2, double * y2, double * z2, double * w2, double * initParameterArray);

//generate confusion seeds
void generateConfusionSeeds(double * x1, double * y1, double * z1, double * w1, double * x2, double * y2, double * z2, double * w2);

//iterate chaotic system and generate iteration results
void generateIterationResults(double * x, double * y, double * z, double * w, int iterations, double * iterationResultsArray);

//convert iteration results to byte sequence
void convertResultsToBytes(double * resultArray, unsigned char * uCharResultArray, int resultArrayElems);

double getCPUSecond(void);

#endif