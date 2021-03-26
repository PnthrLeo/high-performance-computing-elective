#include <mpi.h>
#include <cstdio>


int main(int argc, char** argv) {
	int proc_max, proc_num;
    int recv_num;
    MPI_Status msg_status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_max);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);

    if (proc_num == 0) {
        printf("Hello from process %d\n", proc_num);
        MPI_Send(&proc_num, 1, MPI_INT, proc_num + 1, proc_num, MPI_COMM_WORLD);
        MPI_Recv(&recv_num, 1, MPI_INT, proc_num - 1, proc_num - 1, MPI_COMM_WORLD, &msg_status);
        printf("Hello again, I'm process %d get msg from process %d\n", proc_num, recv_num);
    }
    else if (proc_num == proc_max - 1) {
        MPI_Recv(&recv_num, 1, MPI_INT, proc_num - 1, proc_num - 1, MPI_COMM_WORLD, &msg_status);
        printf("Hello, I'm process %d get msg from process %d\n", proc_num, recv_num);
        MPI_Send(&proc_num, 1, MPI_INT, 0, proc_num, MPI_COMM_WORLD);
    }
    else {
        MPI_Recv(&recv_num, 1, MPI_INT, proc_num - 1, proc_num - 1, MPI_COMM_WORLD, &msg_status);
        printf("Hello, I'm process %d get msg from process %d\n", proc_num, recv_num);
        MPI_Send(&proc_num, 1, MPI_INT, proc_num + 1, proc_num, MPI_COMM_WORLD);
    }
    MPI_Finalize();

    return 0;
}
