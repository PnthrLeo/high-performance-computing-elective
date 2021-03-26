#include <mpi.h>
#include <cstdio>


int main(int argc, char** argv) {
	int proc_max, proc_num;
	int name_len;
	char node_name[MPI_MAX_PROCESSOR_NAME];
	
    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_max);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);
	MPI_Get_processor_name(node_name, &name_len);
	printf("Hello world from node %s, rank [%d/%d]\n", node_name, proc_num, proc_max);
	MPI_Finalize();

    return 0;
}
