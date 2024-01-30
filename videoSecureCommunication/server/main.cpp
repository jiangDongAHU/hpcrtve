#include "kernel.cuh"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//parameters for Lorenz hyper-chaotic systems
#define SIGMA                           10
#define RHO                             28
#define BETA                            8 / 3
#define GAMMA                           -1
#define h                               0.002

//seeds for performing confusion operations on red, green, and blue channels
int redChannelConfusionSeed, greenChannelConfusionSeed, blueChannelConfusionSeed;

//mutex for synchronizing the byte generator therads
sem_t startToGenerateBytesMutex[BYTE_GENERATOR_THREADS], bytesAreGeneratedMutex[BYTE_GENERATOR_THREADS];

int main(int argc, const char ** argv){
    system("clear");
    printf("Server:\n");
    
    //create and initialize a socket
    printf("\n1. Create a socket.\n");

    int socketfp = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if(socketfp == -1){
        perror("Failed to create the socket");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in serverAddress, clientAddress;
    bzero(&serverAddress, sizeof(serverAddress));
    serverAddress.sin_family =AF_INET;
    serverAddress.sin_addr.s_addr = INADDR_ANY;
    serverAddress.sin_port = htons(COMMUNICATION_PORT);

    if(bind(socketfp, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) == -1){
        perror("Failed to bind the socket");
        exit(EXIT_FAILURE);
    }

    if(listen(socketfp, 5) < 0){
        perror("Failed to enter listening state");
        exit(EXIT_FAILURE);
    }

    //wait for connection
    printf("\n2. Wait for the client to connect to the server\n");
    socklen_t clientAddrLen;
    int connfd = accept(socketfp, (struct sockaddr *)&clientAddress, &clientAddrLen);
    if(connfd < 0){
        perror("The client failed to connect to the server");
        exit(EXIT_FAILURE);
    }
    
    //receive the parameters of the video
    int frameWidth, frameHeight, videoFPS, totalFrames;
    recv(connfd, &frameWidth,  sizeof(int), 0);
    recv(connfd, &frameHeight, sizeof(int), 0);
    recv(connfd, &videoFPS,    sizeof(int), 0);
    recv(connfd, &totalFrames, sizeof(int), 0);

    printf("   frame width: %d frame height: %d FPS: %d frames: %d\n", frameWidth, frameHeight, videoFPS, totalFrames);

    //malloc GPU memory
    size_t frameSize = frameWidth * frameHeight * sizeof(uchar3);
    int memSize      = frameWidth * frameHeight * sizeof(uchar3);

    uchar3 * d_originalFrame  = NULL;
    uchar3 * d_encryptedFrame = NULL;
    uchar3 * d_decryptedFrame = NULL; 

    cudaMalloc((void **) & d_originalFrame, memSize);
    cudaMalloc((void **) & d_encryptedFrame, memSize);
    cudaMalloc((void **) & d_decryptedFrame, memSize);

    double x1 = 27.235975; 
    double y1 = 12.459348;
    double z1 = 61.239579;
    double w1 = 126.54395;

    double x2 = 18.238953; 
    double y2 = 23.525843;
    double z2 = 56.534957;
    double w2 = 231.95543;

    //byte sequence for diffusion operations
    int iterations                 = int(frameWidth * frameHeight * 3 * DIFUSION_CONFUSION_ROUNDS * 2 / (4 * BYTES_RESERVED * BYTE_GENERATOR_THREADS)) + 1;
    unsigned char * h_byteSequence = (unsigned char *)malloc(iterations * 4 * BYTES_RESERVED* BYTE_GENERATOR_THREADS * sizeof(unsigned char));

    //generate parameters for initializing chaotic systems of byte generator threads
    double * initParameterArray    = (double *)malloc(8 * BYTE_GENERATOR_THREADS * sizeof(double));
    generateParametersForByteGeneration(&x1, &y1, &z1, &w1, &x2, &y2, &z2, &w2, initParameterArray);

    unsigned char * d_byteSequence = NULL;
    cudaMalloc((void **) & d_byteSequence, iterations * 4 * BYTES_RESERVED * BYTE_GENERATOR_THREADS * sizeof(unsigned char));

    //initialize the mutex semaphores
    for(int i = 0; i < BYTE_GENERATOR_THREADS; i++){
        sem_init(&startToGenerateBytesMutex[i], 0, 0);
        sem_init(&bytesAreGeneratedMutex[i], 0, 0);
    }

    //create byte generator threads
    struct byteGeneratorThreadParameter tp[BYTE_GENERATOR_THREADS];
    for(int i = 0; i < BYTE_GENERATOR_THREADS; i++){
        tp[i].threadIdx          = i;
        tp[i].iterations         = iterations;
        tp[i].byteSequence       = h_byteSequence;
        tp[i].initParameterArray = initParameterArray;
    }

    pthread_t th[BYTE_GENERATOR_THREADS];
    for(int i = 0; i < BYTE_GENERATOR_THREADS; i++)
        pthread_create(&th[i], NULL, byteGeneratorThread, (void *)&tp[i]);

    //wake up all byte generator threads to generate byte sequence
    for(int i = 0; i < BYTE_GENERATOR_THREADS; i++)
        sem_post(&startToGenerateBytesMutex[i]);

    //printf("\n3. Receive the encrypted frame and decrypte the obtained frame");
    Mat h_encryptedFrame, h_decryptedFrame;
    Size resolution(frameWidth, frameHeight);
    vector<uchar> buffer;
    double totalTime    = 0;
    int delayFrameCount = 0;
    int frameCount      = 0;
    for(int i = 0; i < totalFrames; i++){
        //receive the encrypted frame
        int bufferSize;
        recv(connfd, &bufferSize, sizeof(int), 0);
        buffer.resize(bufferSize);

        int bytesReceived = 0;
        while(bytesReceived < bufferSize){
            int ret        = recv(connfd, buffer.data() + bytesReceived, bufferSize - bytesReceived, 0);
            bytesReceived += ret;
        }

        h_encryptedFrame = imdecode(buffer, IMREAD_COLOR);
        h_decryptedFrame = h_encryptedFrame.clone();

        generateConfusionSeeds(&x1, &y1, &z1, &w1, &x2, &y2, &z2, &w2);

        //upload the encrypted frame to the device
        cudaMemcpy(d_encryptedFrame, h_encryptedFrame.data, memSize, cudaMemcpyHostToDevice);

        //decrypt the encrypted frame
        decryptionKernelCaller(d_originalFrame, d_encryptedFrame, d_decryptedFrame, h_byteSequence, d_byteSequence, frameWidth, memSize, iterations);

        //download the decryption result from the device
        cudaMemcpy(h_decryptedFrame.data, d_decryptedFrame, memSize, cudaMemcpyDeviceToHost);

        imshow("server: decrypted frame", h_decryptedFrame);
        imshow("server: obtained frame", h_encryptedFrame);
        waitKey(1);
    }

    for(int i = 0; i < BYTE_GENERATOR_THREADS; i++){
        sem_destroy(&startToGenerateBytesMutex[i]);
        sem_destroy(&startToGenerateBytesMutex[i]);
        pthread_cancel(th[i]);
    }
    cudaFree(d_decryptedFrame);
    cudaFree(d_encryptedFrame);
    cudaFree(d_byteSequence);
    free(h_byteSequence);
    free(initParameterArray);

    close(socketfp);
    close(connfd);

    system("clear");

    return 0;
}

static void * byteGeneratorThread(void * arg){
    struct byteGeneratorThreadParameter *p = (struct byteGeneratorThreadParameter *)arg;
    int threadIdx                          = p->threadIdx;
    int iterations                         = p->iterations;
    unsigned char * byteSequence           = &p->byteSequence[threadIdx * iterations * 4 * BYTES_RESERVED];
    double * initParameterArray            = &p->initParameterArray[threadIdx * 8];

    double * iterationResultsArray1     = (double *)malloc(iterations * 4 * sizeof(double));
    double * iterationResultsArray2     = (double *)malloc(iterations * 4 * sizeof(double));
    unsigned char * uCharResultsArray1  = (unsigned char *)malloc(iterations * 4 * BYTES_RESERVED * sizeof(unsigned char));
    unsigned char * uCharResultsArray2  = (unsigned char *)malloc(iterations * 4 * BYTES_RESERVED * sizeof(unsigned char));

    //initialize chaotic systems
    int idx = 0;
    double x1 = initParameterArray[idx++];
    double y1 = initParameterArray[idx++];
    double z1 = initParameterArray[idx++];
    double w1 = initParameterArray[idx++];

    double x2 = initParameterArray[idx++];
    double y2 = initParameterArray[idx++];
    double z2 = initParameterArray[idx++];
    double w2 = initParameterArray[idx++];

    //pre-iterate the hyper chaotic map
    for(int i = 0; i < PRE_ITERATIONS; i++){
        iterateLorenzHyperChaoticMap(&x1, &y1, &z1, &w1);
        iterateLorenzHyperChaoticMap(&x2, &y2, &z2, &w2);
    }

    while(1){
         sem_wait(&startToGenerateBytesMutex[threadIdx]);
    
        //generate byte sequence
        generateIterationResults(&x1, &y1, &z1, &w1, iterations, iterationResultsArray1);
        generateIterationResults(&x2, &y2, &z2, &w2, iterations, iterationResultsArray2);

        convertResultsToBytes(iterationResultsArray1, uCharResultsArray1, iterations * 4);
        convertResultsToBytes(iterationResultsArray2, uCharResultsArray2, iterations * 4);

        int idx = 0;
        for(int i = 0; i < iterations * 4 * BYTES_RESERVED; i++)
            byteSequence[idx++] = uCharResultsArray1[i] ^ uCharResultsArray2[i];

        sem_post(&bytesAreGeneratedMutex[threadIdx]);
    }

    free(iterationResultsArray1);
    free(iterationResultsArray2);
    free(uCharResultsArray1);
    free(uCharResultsArray2);
    return NULL;
}

//iterate Lorenz map and return the results
void iterateLorenzHyperChaoticMap(double * x, double * y, double * z, double * w){
    
    double K11 = SIGMA * ((* y) - (* x)) + (* w);
    double K12 = SIGMA * ((* y) - ((* x) + (h / 2) * K11)) + (* w);
    double K13 = SIGMA * ((* y) - ((* x) + (h / 2) * K12)) + (* w);
    double K14 = SIGMA * ((* y) - ((* x) + h * K13)) + (* w);
    * x = (* x) + (h / 6) * (K11 + 2 * K12 + 2 * K13 + K14);

    double K21 = RHO * (* x) - (* y) - (* x) * (* z);
    double K22 = RHO * (* x) - ((* y) + (h / 2) * K21) - (* x) * (* z); 
    double K23 = RHO * (* x) - ((* y) + (h / 2) * K22) - (* x) * (* z); 
    double K24 = RHO * (* x) - ((* y) + h * K23) - (* x) * (* z);
    * y = (* y) + (h / 6) * (K21 + 2 * K22 + 2 * K23 + K24);

    double K31 = (* x) * (* y) - BETA * (* z);
    double K32 = (* x) * (* y) - BETA * ((* z) + (h / 2) * K31);
    double K33 = (* x) * (* y) - BETA * ((* z) + (h / 2) * K32);
    double K34 = (* x) * (* y) - BETA * ((* z) + h * K33);
    * z = (* z) + (h / 6) * (K31 + 2 * K32 + 2 * K33 + K34);

    double K41 = GAMMA * (* w) - ((* y) * (* z));
    double K42 = GAMMA * (* w + (h / 2) * K41) - ((* y) * (* z));
    double K43 = GAMMA * (* w + (h / 2) * K42) - ((* y) * (* z));
    double K44 = GAMMA * (* w + h * K43) - ((* y) * (* z));
    * w = (* w) + (h / 6) * (K41 + 2 * K42 + 2 * K43 + K44);
}

void generateParametersForByteGeneration(double * x1, double * y1, double * z1, double * w1, double * x2, double * y2, double * z2, double * w2, double * initParameterArray){
    //pre-iterate the hyper chaotic map
    for(int i = 0; i < PRE_ITERATIONS;i ++){
        iterateLorenzHyperChaoticMap(x1, y1, z1, w1);
        iterateLorenzHyperChaoticMap(x2, y2, z2, w2);
    }

    int idx = 0;
    for(int i = 0; i < BYTE_GENERATOR_THREADS; i++){
        iterateLorenzHyperChaoticMap(x1, y1, z1, w1);
        initParameterArray[idx++] = * x1;
        initParameterArray[idx++] = * y1;
        initParameterArray[idx++] = * z1;
        initParameterArray[idx++] = * w1;

        iterateLorenzHyperChaoticMap(x2, y2, z2, w2);
        initParameterArray[idx++] = * x2;
        initParameterArray[idx++] = * y2;
        initParameterArray[idx++] = * z2;
        initParameterArray[idx++] = * w2;
    }
}

void generateConfusionSeeds(double * x1, double * y1, double * z1, double * w1, double * x2, double * y2, double * z2, double * w2){

    iterateLorenzHyperChaoticMap(x1, y1, z1, w1);

    double iterationResult = * x1;
    int confusionSeed      = 0;
    memcpy(&confusionSeed, (unsigned char *)&iterationResult, BYTES_RESERVED);
    redChannelConfusionSeed = abs(confusionSeed) % (CONFUSION_SEED_UPPER_BOUND - CONFUSION_SEED_LOWWER_BOUND) + CONFUSION_SEED_LOWWER_BOUND;

    iterationResult = * y1;
    confusionSeed   = 0;
    memcpy(&confusionSeed, (unsigned char *)&iterationResult, BYTES_RESERVED);
    redChannelConfusionSeed = abs(confusionSeed) % (CONFUSION_SEED_UPPER_BOUND - CONFUSION_SEED_LOWWER_BOUND) + CONFUSION_SEED_LOWWER_BOUND;

    iterateLorenzHyperChaoticMap(x2, y2, z2, w2);

    iterationResult = * x2;
    confusionSeed   = 0;
    memcpy(&confusionSeed, (unsigned char *)&iterationResult, BYTES_RESERVED);
    blueChannelConfusionSeed = abs(confusionSeed) % (CONFUSION_SEED_UPPER_BOUND - CONFUSION_SEED_LOWWER_BOUND) + CONFUSION_SEED_LOWWER_BOUND;
}

//iterate Lorenz map and store the results in array
void generateIterationResults(double * x, double * y, double * z, double * w, int iterations, double * iterationResultsArray){
    //iterate the hyper chaotic map, generate iteration results, and store the results
    int idx = 0;
    for(int i = 0; i < iterations; i++){
        iterateLorenzHyperChaoticMap(x, y, z, w);

        iterationResultsArray[idx++] = * x;
        iterationResultsArray[idx++] = * y;
        iterationResultsArray[idx++] = * z;
        iterationResultsArray[idx++] = * w;
    }
}

//convert iteration results into bytes
void convertResultsToBytes(double * resultArray, unsigned char * uCharResultArray, int resultArrayElems){    
    for(int i = 0; i < resultArrayElems; i++){
        unsigned char * p = &uCharResultArray[i * BYTES_RESERVED];
        memcpy(p, & resultArray[i], BYTES_RESERVED);
    }
}

//get cpu time
double getCPUSecond(void){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

