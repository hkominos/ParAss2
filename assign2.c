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

/* #define WRITE_TO_FILE */
/* #define VERIFY */

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
  double mpi_start_time;

  // Variables for Cartesian topology processor grid
  int ndims;
  int dims[2], coords[2], cyclic[2], reorder;

  // 2D-grid
  MPI_Comm proc_grid;

  // MPI request variables
  MPI_Request send_request, recv_request;

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
  nx_static = floor(N / dims[1]);
  ny_static = floor(N / dims[0]);
  mx_static = N % dims[1];
  my_static = N % dims[0];

  // Set sizes of tiles
  int Nx = nx_static;
  int Ny = ny_static;

  if (coords[1] < mx_static) {
  	Nx += 1;
  }

  if (coords[0] < my_static) {
  	Ny += 1;
  }

  int Nt;
  double dt, dx, lambda_sq;
  double begin, end;

  Nt = N;
  dx = 1.0 / (N - 1);
  dt = 0.50 * dx;
  lambda_sq = (dt/dx) * (dt/dx);

  // // Create new vector type to contain vertical halo points
  // MPI_Type_vector(r1, 1, N, MPI_DOUBLE, &vec_type);
  // MPI_Type_commit(&vec_type);

  // Allocate for extended tiles
  u = malloc((Nx + 2) * (Ny + 2) * sizeof(double));
  u_old = malloc((Nx + 2) * (Ny + 2) * sizeof(double));
  u_new = malloc((Nx + 2) * (Ny + 2) * sizeof(double));

  /* Setup IC */

  memset(u, 0, (Nx + 2) * (Ny + 2) * sizeof(double));
  memset(u_old, 0, (Nx + 2) * (Ny + 2) * sizeof(double));
  memset(u_new, 0, (Nx + 2) * (Ny + 2) * sizeof(double));

  for (int i = 1; i < Ny - 1; ++i) {
    for (int j = 1; j < Nx - 1; ++j) {
      double x = j * dx;
      double y = i * dx;

      /* u0 */
      u[i * Nx + j] = initialize(x, y, 0);

      /* u1 */
      u_new[i * Nx + j] = initialize(x, y, dt);
    }
  }

#ifdef WRITE_TO_FILE
  save_solution(u_new, Ny, Nx, 1);
#endif
#ifdef VERIFY
  double max_error = 0.0;
#endif

  /* Integrate */

  begin = timer();
  for(int n = 2; n < Nt; ++n) {
    /* Swap ptrs */
    double *tmp = u_old;
    u_old = u;
    u = u_new;
    u_new = tmp;

    /* Apply stencil */
    for(int i = 1; i < (Ny-1); ++i) {
      for(int j = 1; j < (Nx-1); ++j) {

        u_new[i*Nx+j] = 2*u[i*Nx+j]-u_old[i*Nx+j]+lambda_sq*
          (u[(i+1)*Nx+j] + u[(i-1)*Nx+j] + u[i*Nx+j+1] + u[i*Nx+j-1] -4*u[i*Nx+j]);
      }
    }

#ifdef VERIFY
    double error=0.0;
    for(int i = 0; i < Ny; ++i) {
      for(int j = 0; j < Nx; ++j) {
        double e = fabs(u_new[i*Nx+j]-initialize(j*dx,i*dx,n*dt));
        if(e>error)
          error = e;
      }
    }
    if(error > max_error)
      max_error=error;
#endif

#ifdef WRITE_TO_FILE
    save_solution(u_new,Ny,Nx,n);
#endif

  }
  end=timer();

  printf("Time elapsed: %g s\n",(end-begin));

#ifdef VERIFY
  printf("Maximum error: %g\n",max_error);
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
  value=sin(3*M_PI*x)*sin(4*M_PI*y)*cos(5*M_PI*t);
#else
  /* squared-cosine hump */
  const double width=0.1;

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
