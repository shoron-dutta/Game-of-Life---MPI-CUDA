#include<stdio.h>
#include<stdlib.h>
#include <mpi.h>
#include<stdbool.h>

void cuda_finalize();
bool gol_kernelLaunch(unsigned char** d_data, unsigned char** d_resultData, size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount);
void gol_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight, int myRank, int numRank );
void init_Ghost_rows();

unsigned char *previous_last_row = NULL, *next_first_row = NULL, *my_first_row = NULL, *my_last_row = NULL;
unsigned char *g_data  = NULL, *g_resultData = NULL;
size_t g_worldWidth = 0,  g_worldHeight = 0;


// this function writes output to text files
static inline void gol_printWorld(FILE* fp, int myRank)
{
    int i, j;

    for( i = 0; i < g_worldHeight; i++){
    	fprintf(fp,"Row %2d: ", (int)g_worldHeight*myRank+i);
	    for( j = 0; j < g_worldWidth; j++){
	        fprintf(fp,"%u ", (unsigned int)g_data[(i*g_worldWidth) + j]);
	    }
	    fprintf(fp,"\n");
    }
}

int main(int argc, char *argv[])
{

    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int itterations = 0;
    ushort threadsCount; // number of threads to be run
    unsigned int on_off; // determines if the program shows any output or not


    FILE *fp = NULL; // pointer to the output file
    char fname[32]; // variable for storing the dynamic filenames for each rank

    if( argc != 6 )
    {
        printf("GOL requires 5 arguments: pattern number, sq size of the world and the number of itterations, threads per block and output-on-off e.g. ./gol 0 32 2 512 0 \n");
        exit(-1);
    }
 
    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    itterations = atoi(argv[3]);
    threadsCount = atoi(argv[4]); 
    on_off = atoi(argv[5]); 

    int myRank, numRank, previous_rank, next_rank; // rank related variables
    
    //initialize MPI
    MPI_Init(&argc, &argv);
    // receive my own rank id
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    // receive total number of ranks
    MPI_Comm_size(MPI_COMM_WORLD, &numRank);

    // if output flag is on create the files
    if (on_off == 1 ){
        sprintf(fname, "Rank_%d_of_%d.txt", myRank, numRank);
        fp=fopen(fname,"w");
        if (fp==NULL){
            printf("ERROR IN RANK %d", myRank);
            exit(-1);
        }
        
    }

    // initialize the variables
    gol_initMaster(pattern, worldSize, worldSize, myRank, numRank);
    // initialize the ghost rows
    init_Ghost_rows();

    double startTime, endTime, duration; // needed for calculating runtime
    if(myRank == 0)
        startTime = MPI_Wtime();

    // calculate the previous and next rank id
    // use mod to avoid out of range values
    previous_rank = (myRank - 1 + numRank) % numRank;
    next_rank = (myRank + 1) % numRank;
    
    MPI_Request s1, s2; // the information current rank sends
    MPI_Request r1, r2; // the information current rank receives
    MPI_Status status;

    int i = 0;
    for( i = 0; i < itterations; i++){
        
        //receive the last row from the previous rank
        MPI_Irecv(previous_last_row, g_worldWidth, MPI_UNSIGNED_CHAR, previous_rank, 0, MPI_COMM_WORLD, &r1);

        //receive the first row from the next rank
        MPI_Irecv(next_first_row, g_worldWidth, MPI_UNSIGNED_CHAR, next_rank, 0, MPI_COMM_WORLD, &r2);

        
        //send my first row to the previous rank
        MPI_Isend(my_first_row, g_worldWidth, MPI_UNSIGNED_CHAR, previous_rank, 0, MPI_COMM_WORLD, &s1);
        
        //send my last row to the next rank
        MPI_Isend(my_last_row, g_worldWidth, MPI_UNSIGNED_CHAR, next_rank, 0, MPI_COMM_WORLD, &s2);

        // wait for receiving the ghost rows from adjacent ranks
        MPI_Wait(&r1, &status);
        MPI_Wait(&r2, &status);
        
        // calling gol kernel launch function
        gol_kernelLaunch(&g_data, &g_resultData, g_worldWidth, g_worldHeight, itterations, threadsCount);
        
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // end timer and calculate duration
    if (myRank == 0){
        endTime = MPI_Wtime();
        duration = endTime - startTime; // calculate duration
        printf("TOTAL DURATION : %.5lf, number of cell updates = %ld\n", duration, (long) numRank    *(long)g_worldHeight*
                                                        (long)g_worldWidth  *(long)itterations);
      
        
    }


    if(myRank == 0){
            printf("This is the Game of Life running in parallel on a GPU on multiple ranks.\n");
    }
    // print results if the on_off flag is set to 1
    if ( on_off == 1 ){
        fprintf(fp,"######################### FINAL WORLD IN RANK %d IS ###############################\n",myRank);
        gol_printWorld(fp, myRank); // print to file
        fclose(fp); // close file
    }
   
    // calling the MPI finalize function
    MPI_Finalize();
    // free up memory
    cuda_finalize();
    return 0;
}

