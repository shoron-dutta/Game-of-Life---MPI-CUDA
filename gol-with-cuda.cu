#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>


// Result from last compute of world.
extern "C" unsigned char *g_resultData;

// Current state of world.
extern "C" unsigned char *g_data;

// Current width of world.
extern "C" size_t g_worldWidth;

// Current height of world.
extern "C" size_t g_worldHeight;

// Current data length (product of width and height)
size_t g_dataLength=0;  // g_worldWidth * g_worldHeight


//allocate ghost row here
extern "C" unsigned char *my_first_row; // first row of the current rank
extern "C" unsigned char *my_last_row; // last row of the current rank

extern "C" unsigned char *previous_last_row; //  last row of the previous rank
extern "C" unsigned char *next_first_row; // first row of the next rank

extern "C" void init_Ghost_rows()
{
    // ghost row size is the same as world width because its just one row
    cudaMallocManaged(&my_first_row, (g_worldWidth * sizeof(unsigned char)));
    cudaMallocManaged(&my_last_row, (g_worldWidth * sizeof(unsigned char)));

    // Initialize values by assigning values from g_data
    int i;
    for(i=0; i< g_worldWidth; i++)
        my_first_row[i] = g_data[i];

    size_t x = g_worldWidth*(g_worldHeight - 1);
    for(i = 0; i<g_worldWidth; i++){
        my_last_row[i] = g_data[x];
        x++;
    }

    // ghost row size is the same as world width because its just one row
    cudaMallocManaged(&previous_last_row, (g_worldWidth * sizeof(unsigned char))); // no initial value, will be updated via Irecv
    cudaMallocManaged(&next_first_row, (g_worldWidth * sizeof(unsigned char))); // no initial value, will be updated via Irecv

}

//pattern 0
static inline void gol_initAllZeros( size_t worldWidth, size_t worldHeight, int myRank, int numRank )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    const size_t sz = size_t(g_dataLength) * sizeof(unsigned char);

    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_data, 0, sz);

    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, sz);
}

//pattern 1
static inline void gol_initAllOnes( size_t worldWidth, size_t worldHeight, int myRank, int numRank )
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    const size_t sz = size_t(g_dataLength) * sizeof(unsigned char);

    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));

    // set all rows of world to true
    for( i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 1;
    }

   cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char)));
   cudaMemset(g_resultData, 0, sz);
}

//pattern 2
static inline void gol_initOnesInMiddle( size_t worldWidth, size_t worldHeight, int myRank, int numRank )
{
    int i;

    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    const size_t sz = size_t(g_dataLength) * sizeof(unsigned char);

    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_data, 0, sz);

    int row_offset = (worldHeight - 1)*worldHeight;
    int col_offset = row_offset + 127;

    for( i = col_offset; i < col_offset+10; i++)
    {
        g_data[i] = 1;  
    }

    //allocate memory for resultData
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char)));
    // fill in resultData with zeroes
    cudaMemset(g_resultData, 0, sz);
}

//pattern 3
static inline void gol_initOnesAtCorners( size_t worldWidth, size_t worldHeight, int myRank, int numRank )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    const size_t sz = size_t(g_dataLength) * sizeof(unsigned char);

    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_data, 0, sz);
    
    if(myRank == 0){
        g_data[0] = 1; // upper left
        g_data[worldWidth-1]=1; // upper right
    }
    
    else if(myRank == numRank - 1){
        g_data[(worldHeight * (worldWidth-1))]=1; // lower left
        g_data[(worldHeight * (worldWidth-1)) + worldWidth-1]=1; // lower right
    }
    
    
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, sz);
}

//pattern 4
static inline void gol_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight, int myRank, int numRank )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    const size_t sz = size_t(g_dataLength) * sizeof(unsigned char);

    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_data, 0, sz);
    
    if(myRank == 0){
        g_data[0] = 1; // upper left
        g_data[1] = 1; // upper left +1
        g_data[worldWidth-1]=1; // upper right

    }

    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char)));
    cudaMemset(g_resultData, 0, sz);

}

// function that swaps the result words and original world back to back
extern "C" static inline void gol_swap( unsigned char **pA, unsigned char **pB)
{
    // The type of the temporary variable is the same as g_data and g_resultData
    // Because the **pA and **pB are pointers to g_data and g_resultData
    unsigned char *temp = NULL;

    //swap the variables

    temp = *pA;
    *pA = *pB;
    *pB = temp;

}

// the kernel function that updates the cells
__global__ void gol_kernel(const unsigned char* d_data,
unsigned int worldWidth,
unsigned int worldHeight,
unsigned char* d_resultData,
unsigned char* previous_last_row,
unsigned char* next_first_row){

    int index;
    // calculate index and iterate over
    for(index = blockIdx.x *blockDim.x + threadIdx.x; index < worldWidth*worldHeight; index += blockDim.x * gridDim.x)
    {
        size_t x, y;

        // calculate a cell position in 2D using index
        y = (int) index/worldWidth;
        x = index - y*worldWidth;

        
        // calculating values for x0, x1 and x2 in 1-D
        size_t x0, x1, x2;
        x1 = x;
        x0 = (x + worldWidth - 1) % worldWidth;
        x2 = (x + 1) % worldWidth;


        // calculating values for y0, y1 and y2 in 1-D
        size_t y0, y1, y2;

        y0=(y-1)*worldWidth;
        y1=y*worldWidth;
        y2=(y+1)*worldWidth;

        
        int alive_neighbors = 0;
        printf("BEFORE AND");
        if(y1 < worldWidth) // if the current row is the first row then take y0 values from the ghost row (previous rank's last row)
            alive_neighbors = (unsigned int)previous_last_row[x0] + (unsigned int)d_data[x0+y1] + (unsigned int)d_data[x0+y2] + 
                            (unsigned int)previous_last_row[x1] + (unsigned int)d_data[x1+y2] + 
                            (unsigned int)previous_last_row[x2] + (unsigned int)d_data[x2+y1] + (unsigned int)d_data[x2+y2];
        else if(y1 >= worldWidth*(worldHeight - 1)) // if the current row is the last row then take y0 values from the ghost row (next rank's first row)
            alive_neighbors = (unsigned int)d_data[x0+y0] + (unsigned int)d_data[x0+y1] + (unsigned int)next_first_row[x0] + 
                        (unsigned int)d_data[x1+y0] + (unsigned int)next_first_row[x1] + 
                        (unsigned int)d_data[x2+y0] + (unsigned int)d_data[x2+y1] + (unsigned int)next_first_row[x2];        
        else // calculate as usual
            alive_neighbors = (unsigned int)d_data[x0+y0] + (unsigned int)d_data[x0+y1] + 
                            (unsigned int)d_data[x0+y2] + (unsigned int)d_data[x1+y0] + 
                            (unsigned int)d_data[x1+y2] + (unsigned int)d_data[x2+y0] + (unsigned int)d_data[x2+y1] + (unsigned int)d_data[x2+y2];

        // If a cell is alive, there are three possible scenarios
        // Either it dies due to overpopulation or underppopulation or it remains alive
        if(d_data[y1 + x] == 1){
            if(alive_neighbors<2)
                d_resultData[y1 + x] = 0;
            else if (alive_neighbors ==2 || alive_neighbors == 3)
                d_resultData[y1 + x] = 1;
            else if (alive_neighbors>3){
                d_resultData[y1+x] = 0;

            }
        }
        // If a cell is dead, there are two scenarios
        // either it is alive again or remains dead
        else if(d_data[y1+x] == 0)
        {
            if(alive_neighbors == 3)
                d_resultData[y1+x] = 1;
            else
                d_resultData[y1+x] = 0;
    }


    }

}
// this function calls the kernel
extern "C" bool gol_kernelLaunch(unsigned char** d_data,
unsigned char** d_resultData,
size_t worldWidth,
size_t worldHeight,
size_t iterationsCount,
ushort threadsCount){

    // number of block is calculated
    size_t blocksCount = (size_t)((worldWidth*worldHeight)/threadsCount);
    
    // calling kernel
    gol_kernel<<<blocksCount, threadsCount>>>(*d_data, worldWidth, worldHeight, *d_resultData, previous_last_row, next_first_row);
    // synchronization ensures the end of execution for all kernels before swapping 
    cudaDeviceSynchronize();
    // swap the worlds
    gol_swap(d_data, d_resultData);
    

    return true;

}
// initialize the world using the pattern
extern "C" void gol_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight, int myRank, int numRank )
{
    int cudaDeviceCount;
    cudaError_t cE;
    if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n",
        cE, cudaDeviceCount );
        exit(-1);
    }
    if( (cE = cudaSetDevice( myRank % cudaDeviceCount )) != cudaSuccess )
    {
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n",
        myRank, (myRank % cudaDeviceCount), cE);
        exit(-1);
    }
    switch(pattern)
    {
    case 0:
        gol_initAllZeros( worldWidth, worldHeight, myRank, numRank );
        break;

    case 1:
        gol_initAllOnes( worldWidth, worldHeight, myRank, numRank );
        break;

    case 2:
        gol_initOnesInMiddle( worldWidth, worldHeight, myRank, numRank );
        break;

    case 3:
        gol_initOnesAtCorners( worldWidth, worldHeight, myRank, numRank );
        break;

    case 4:
        gol_initSpinnerAtCorner( worldWidth, worldHeight, myRank, numRank );
        break;

    default:
        printf("Pattern %u has not been implemented \n", pattern);
        exit(-1);
    }
}





extern "C" void cuda_finalize()
{
    //free up memory
    cudaFree(g_data);
    cudaFree(g_resultData);
}


