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
    printf("Client:\n");

    //connect to the server
    printf("\n1. Connect to the server\n");
    int socketfp = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    struct sockaddr_in serverAddress;
    serverAddress.sin_family      = AF_INET;
    serverAddress.sin_port        = htons(COMMUNICATION_PORT);
    serverAddress.sin_addr.s_addr = inet_addr(SERVER_IP_ADDRESS);

    if(connect(socketfp, (struct sockaddr *)&serverAddress, sizeof(serverAddress)) < 0){
        perror("Failed to connect to the server");
        exit(EXIT_FAILURE);
    }
    printf("   Server IP: %s Port: %d\n", SERVER_IP_ADDRESS, COMMUNICATION_PORT);

    //open the original video
    VideoCapture capture;
    capture.open(VIDEO_NAME);
    if(!capture.isOpened()){
        printf("failed to open the original video file!\n");
        return -1;
    }

    //obtain the width, height, FPS, frames of the original frames
    int frameWidth  = (int)capture.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = (int)capture.get(CAP_PROP_FRAME_HEIGHT);
    int videoFPS    = (int)capture.get(CAP_PROP_FPS);
    int totalFrames = (int)capture.get(CAP_PROP_FRAME_COUNT);

    //send the parameters of the video
    printf("\n2.  Send the parameters of the video\n");
    send(socketfp, &frameWidth,  sizeof(int), 0);
    send(socketfp, &frameHeight, sizeof(int), 0);
    send(socketfp, &videoFPS,    sizeof(int), 0);
    send(socketfp, &totalFrames, sizeof(int), 0);

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

    printf("\n3. Encrypte the original frame and send the encrypted frame\n");

    Mat h_originalFrame, h_encryptedFrame;
    Size resolution(frameWidth, frameHeight);
    vector<uchar> buffer;
    double totalTime    = 0;
    int delayFrameCount = 0;
    int frameCount      = 0;
    while(capture.read(h_originalFrame)){
        double startTime = getCPUSecond();
        generateConfusionSeeds(&x1, &y1, &z1, &w1, &x2, &y2, &z2, &w2);

        h_encryptedFrame = h_originalFrame.clone();

        //upload the original frame to the device
        cudaMemcpy(d_originalFrame, h_originalFrame.data, memSize, cudaMemcpyHostToDevice);

        //encrypt the original frame
        encryptionKernelCaller(d_originalFrame, d_encryptedFrame, d_decryptedFrame, h_byteSequence, d_byteSequence, frameWidth, memSize, iterations);

        //download the scrambled frame from the device
        cudaMemcpy(h_encryptedFrame.data, d_encryptedFrame, memSize, cudaMemcpyDeviceToHost);

        imshow("client: original frame", h_originalFrame);
        imshow("client: encrypted frame", h_encryptedFrame);

        //encode the encrypted frame and send it to the server
        imencode(".png", h_encryptedFrame, buffer);
        int bufferSize = buffer.size();

        send(socketfp, &bufferSize, sizeof(int), 0);
        int bytesSent = 0;
        while(bytesSent < bufferSize){
            int ret    = send(socketfp, buffer.data() + bytesSent, bufferSize - bytesSent, 0);
            bytesSent += ret;
        }
        
        double endTime = getCPUSecond();

        totalTime = totalTime + (endTime - startTime);
        int waitKeyTime = 1000 / videoFPS - (int)((endTime - startTime) * 1000);    
        if(waitKeyTime <= 0){
            waitKeyTime = 1;
        }
        waitKey(waitKeyTime); 

    }

    for(int i = 0; i < BYTE_GENERATOR_THREADS; i++){
        sem_destroy(&startToGenerateBytesMutex[i]);
        sem_destroy(&startToGenerateBytesMutex[i]);
        pthread_cancel(th[i]);
    }
    cudaFree(d_originalFrame);
    cudaFree(d_encryptedFrame);
    cudaFree(d_byteSequence);
    free(h_byteSequence);
    free(initParameterArray);
    close(socketfp);

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