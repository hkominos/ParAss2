/* Solution to Compulsory Assignment 2
 * Parallel and Distributed Programming
 * Spring 2017
 * Authors: Joel Backsell and Charalampos Kominos
 * Based on supplied reference
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

//#define WRITE_TO_FILE
#define VERIFY

double timer();
double initialize(double x, double y, double t);
void save_solution(double *u, int Ny, int Nx, int n);

int main(int argc, char *argv[]) {

  // Variables of world communication
  int rank, nproc;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int N;

  // Set size of grid to command line argument
  if (argc != 2) {
    if (rank == 0)
      printf("Usage: %s size-of-grid\n", argv[0]);
    MPI_Finalize();
    return 0;
  } else {
    N = atoi(argv[1]);
  }

  // Time variable
  double time_measure;
  double mpi_start_time;

  // Variables for Cartesian topology processor grid
  int ndims;
  int dims[2], coords[2], cyclic[2], reorder;

  // 2D-grid
  MPI_Comm proc_grid;

  // MPI status variable
  MPI_Status status;

  // Tiles local to each processor
  double *u, *u_old, *u_new;

  // MPI datatype vector
  MPI_Datatype vec_type;

  ndims = 2;
  dims[0] = 0;
  dims[1] = 0;
  cyclic[0] = 0;
  cyclic[1] = 0;
  reorder = 0;

  MPI_Dims_create(nproc, ndims, dims);

  // Create Cartesian grid
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, cyclic, reorder, &proc_grid);

  // Set my coordinates in grid
  MPI_Cart_coords(proc_grid, rank, ndims, coords);

  // Set my rank in grid
  int my_rank;
  MPI_Cart_rank(proc_grid, coords, &my_rank);

  //printf("Grid: %d times %d\n", dims[0], dims[1]);

  //printf("my_rank: %d. Coords: (%d, %d)\n", my_rank, coords[0], coords[1]);

  // Static partitioning of points
  int mx_static = N % dims[1];
  int my_static = N % dims[0];
  int nx_static = floor(N / dims[1]);
  int ny_static = floor(N / dims[0]);

  // Set sizes of tiles
  int Nx = nx_static;
  int Ny = ny_static;

  if (coords[1] < mx_static) {
  	Nx = nx_static + 1;
  }

  if (coords[0] < my_static) {
  	Ny = ny_static + 1;
  }

  int Nt;
  double dt, dx, lambda_sq;

  Nt = N;
  dx = 1.0 / (N - 1);
  dt = 0.50 * dx;
  lambda_sq = (dt/dx) * (dt/dx);

  // Set sizes of extended tiles
  int Nx_ext = Nx;
  int Ny_ext = Ny;

  if (coords[0] != 0)
  	Ny_ext++;

  if (coords[0] != dims[0] - 1)
  	Ny_ext++;

  if (coords[1] != 0)
  	Nx_ext++;

  if (coords[1] != dims[1] - 1)
  	Nx_ext++;

  // Create new vector type to contain vertical halo points
  MPI_Type_vector(Ny_ext, 1, Nx_ext, MPI_DOUBLE, &vec_type);
  MPI_Type_commit(&vec_type);

  // Allocate for extended tiles
  u = malloc(Nx_ext * Ny_ext * sizeof(double));
  u_old = malloc(Nx_ext * Ny_ext * sizeof(double));
  u_new = malloc(Nx_ext * Ny_ext * sizeof(double));

  // Setup intitial conditions
  memset(u, 0, Nx_ext * Ny_ext * sizeof(double));
  memset(u_old, 0, Nx_ext * Ny_ext * sizeof(double));
  memset(u_new, 0, Nx_ext * Ny_ext * sizeof(double));

  // Set my initial x and y values
  double x, y;
  if (coords[1] < mx_static) {
    x = coords[1] * (nx_static + 1) * dx;
  } else {
    x = (mx_static + coords[1] * nx_static) * dx;
  }

  if (coords[0] < my_static) {
    y = coords[0] * (ny_static + 1) * dx;
  } else {
    y = (my_static + coords[0] * ny_static) * dx;
  }

  if (coords[1] == 0)
  	x += dx;

  if (coords[0] == 0)
  	y += dx;

  //printf("my_rank: %d. initial x: %f, initial y: %f\n", my_rank, x, y);

	for (int i = 1; i < Ny_ext - 1; i++) {
    for (int j = 1; j < Nx_ext - 1; j++) {
    	double xj = x + (j - 1) * dx;
    	double yi = y + (i - 1) * dx;

      // Set u0
      u[i * Nx_ext + j] = initialize(xj, yi, 0);

      // Set u1
      u_new[i * Nx_ext + j] = initialize(xj, yi, dt);
    }
  }

#ifdef WRITE_TO_FILE
  save_solution(u_new, Ny, Nx, 1);
#endif
#ifdef VERIFY
  double max_error = 0.0;
#endif

  // for (int i = 0; i < Ny_ext; i++) {
  // 	printf("my_rank: %d. ", my_rank);
  // 	for (int j = 0; j < Nx_ext; j++) {
  // 		printf("%f, ", u[i*Nx_ext+j]);
  // 	}
  // 	printf("\n");
  // }

  // Start time measurement
  if (rank == 0) {
  	time_measure = timer();
		mpi_start_time = MPI_Wtime();
	}

	// Start integration
  for (int n = 2; n < Nt; n++) {
    // Swap pointers
    double *tmp = u_old;
    u_old = u;
    u = u_new;
    u_new = tmp;

    int source;
  	int dest;

  	// Call for shift source and destination
  	MPI_Cart_shift(proc_grid, 0, 1, &source, &dest);

  	// Send up and receive from below and send down and receive from above, respectively
  	if (coords[0] == 0) {
  		MPI_Recv(&u[Nx_ext*Ny_ext-Nx_ext], Nx_ext, MPI_DOUBLE, dest, 111, proc_grid, &status);
  		MPI_Send(&u[Nx_ext*Ny_ext-2*Nx_ext], Nx_ext, MPI_DOUBLE, dest, 222, proc_grid);
  	} else if (coords[0] == dims[0] - 1) {
  		MPI_Send(&u[Nx_ext], Nx_ext, MPI_DOUBLE, source, 111, proc_grid);
  		MPI_Recv(&u[0], Nx_ext, MPI_DOUBLE, source, 222, proc_grid, &status);
  	} else {
  		MPI_Sendrecv(&u[Nx_ext], Nx_ext, MPI_DOUBLE, source, 111, &u[Nx_ext*Ny_ext-Nx_ext], Nx_ext, MPI_DOUBLE, dest, 111, proc_grid, &status);
  		MPI_Sendrecv(&u[Nx_ext*Ny_ext-2*Nx_ext], Nx_ext, MPI_DOUBLE, dest, 222, &u[0], Nx_ext, MPI_DOUBLE, source, 222, proc_grid, &status);
  	}

  	// Send right and receive from left and send left and receive from right, respectively
  	MPI_Cart_shift(proc_grid, 1, 1, &source, &dest);
  	if (coords[1] == 0) {
  		MPI_Recv(&u[Nx_ext-1], 1, vec_type, dest, 333, proc_grid, &status);
  		MPI_Send(&u[Nx_ext-2], 1, vec_type, dest, 444, proc_grid);
  	} else if (coords[1] == dims[1] - 1) {
  		MPI_Send(&u[1], 1, vec_type, source, 333, proc_grid);
  		MPI_Recv(&u[0], 1, vec_type, source, 444, proc_grid, &status);
  	} else {
  		MPI_Sendrecv(&u[1], 1, vec_type, source, 333, &u[Nx_ext-1], 1, vec_type, dest, 333, proc_grid, &status);
  		MPI_Sendrecv(&u[Nx_ext-2], 1, vec_type, dest, 444, &u[0], 1, vec_type, source, 444, proc_grid, &status);
  	}

    // Apply stencil
    for (int i = 1; i < Ny_ext - 1; i++) {
      for (int j = 1; j < Nx_ext - 1; j++) {
        u_new[i*Nx_ext+j] = 2 * u[i*Nx_ext+j] - u_old[i*Nx_ext+j] + lambda_sq *
          (u[(i-1)*Nx_ext+j] + u[(i+1)*Nx_ext+j] + u[i*Nx_ext+j+1] + u[i*Nx_ext+j-1] - 4 * u[i*Nx_ext+j]);
      }
    }

#ifdef VERIFY
    double error = 0.0;
    for (int i = 1; i < Ny_ext - 1; i++) {
      for (int j = 1; j < Nx_ext - 1; j++) {
      	
    		double xj = x + (j - 1) * dx;
    		double yi = y + (i - 1) * dx;

        double e = fabs(u_new[i*Nx_ext+j] - initialize(xj, yi, n * dt));

        if(e > error)
          error = e;
      }
    }
    if(error > max_error)
      max_error = error;
#endif

#ifdef WRITE_TO_FILE
    save_solution(u_new, Ny, Nx, n);
#endif

  }

  if (rank == 0) {
		time_measure = timer() - time_measure;
		double mpi_elapsed_time = MPI_Wtime() - mpi_start_time;
		printf("Time: %f s\n", time_measure);
		printf("MPI Wall time: %f s\n", mpi_elapsed_time);
	}

#ifdef VERIFY
  double glob_error;
  MPI_Reduce(&max_error, &glob_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0)
  	printf("Global maximum error: %g\n", glob_error);
#endif

  free(u);
  free(u_old);
  free(u_new);

  // Close MPI
  MPI_Finalize();

  return 0;
}

double timer()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
  return seconds;
}

double initialize(double x, double y, double t)
{
  double value = 0;
#ifdef VERIFY
  /* standing wave */
  value = sin(3*M_PI*x) * sin(4*M_PI*y) * cos(5*M_PI*t);
#else
  /* squared-cosine hump */
  const double width = 0.1;

  double centerx = 0.25;
  double centery = 0.5;

  double dist = sqrt((x-centerx)*(x-centerx) +
                     (y-centery)*(y-centery));
  if(dist < width) {
    double cs = cos(M_PI_2*dist/width);
    value = cs*cs;
  }
#endif
  return value;
}

void save_solution(double *u, int Ny, int Nx, int n)
{
  char fname[50];
  sprintf(fname,"solution-%d.dat",n);
  FILE *fp = fopen(fname,"w");

  fprintf(fp,"%d %d\n",Nx,Ny);

  for(int j = 0; j < Ny; ++j) {
    for(int k = 0; k < Nx; ++k) {
      fprintf(fp,"%e\n",u[j*Nx+k]);
    }
  }

  fclose(fp);
}
