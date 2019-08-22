#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>

#include "neuralnetwork_ocl.h"
#include "helpers.h"

#define MAX_SOURCE_SIZE	16384
#define WORKGROUP_SIZE	16

void sigmoid(float* vector, int len) {
	/*
	Na elementih vektorja izracuna logisticno funkcijo in rezultate zapise v isti vektor
	*/

	for (int i = 0; i < len; i++) {
		vector[i] = 1.0f / (1.0f + (float)exp(vector[i] * (-1)));
	}
}

float * predict(float ** X, int Xrows, int Xcols, float ** Theta1, int T1rows, int T1cols, float ** Theta2, int T2rows, int T2cols) {
	/*
	Predikcija oznak vhodnih primerov z podanimi parametri
	*/

	float * result = (float*)calloc(Xrows, sizeof(float));

	for (int i = 0; i < Xrows; i++) { // Gremo v zanki cez vsak testni primer
		float * a2 = (float*)calloc(T1rows, sizeof(float));
		float * a3 = (float*)calloc(T2rows, sizeof(float));

		// aktivacija drugega nivoja a2
		for (int theta_i = 0; theta_i < T1rows; theta_i++) {
			a2[theta_i] = Theta1[theta_i][0];
			for (int theta_j = 1; theta_j < T1cols; theta_j++) {
				a2[theta_i] += (Theta1[theta_i][theta_j] * X[i][theta_j - 1]);
			}
		}
		sigmoid(a2, T1rows);

		// aktivacija tretjega nivoja a3
		for (int theta_i = 0; theta_i < T2rows; theta_i++) {
			a3[theta_i] = Theta2[theta_i][0];
			for (int theta_j = 1; theta_j < T2cols; theta_j++) {
				a3[theta_i] += (Theta2[theta_i][theta_j] * a2[theta_j - 1]);
			}
		}
		sigmoid(a3, T2rows);

		// Index najvecjega izhodnega nevrona predstavlja ustrezno crko
		float max = a3[0];
		int indeks = 0;
		for (int count = 1; count < T2rows; count++) {
			if (a3[count] > max) {
				max = a3[count];
				indeks = count;
			}
		}
		result[i] = (float)indeks; // shranim rezultat klasifikacije

		free(a2);
		free(a3);
	}

	return result;
}

void randInitializeWeights(float * param, int paramSize) {
	/*
	Nakljucna inicializacija parametrov
	*/

	float epsilon_init = 0.12f;
	for (int i = 0; i < paramSize; i++) {
		float r = (float)rand() / (float)RAND_MAX;
		param[i] = (r * 2.0f * epsilon_init) - epsilon_init;
	}
}

float gradientDescent(float * param, int paramSize, int iterations, float ** X, int Xrows, int Xcols, int hiddenLayerSize, float * y, int yLabels, float lambda) {
	/*
	Optimizacija parametrov z gradientnim spustom
	*/

	int numExamples = Xrows;

	// pretvori parametre v dva vektorja
	int iparam = 0;
	int T1rows = hiddenLayerSize;
	int T1cols = Xcols + 1;
	float * T1 = (float*)calloc(T1rows * T1cols, sizeof(float));
	for (int j = 0; j < T1cols; j++) {
		for (int i = 0; i < T1rows; i++) {
			T1[i * T1cols + j] = param[iparam];
			iparam++;
		}
	}

	int T2rows = yLabels;
	int T2cols = hiddenLayerSize + 1;
	float * T2 = (float*)calloc(T2rows * T2cols, sizeof(float));
	for (int j = 0; j < T2cols; j++) {
		for (int i = 0; i < T2rows; i++) {
			T2[i * T2cols + j] = param[iparam];
			iparam++;
		}
	}

	// pretvori oznake y
	float * Y = (float*)calloc(yLabels * numExamples, sizeof(float));
	for (int i = 0; i < numExamples; i++) {
		int label = (int)y[i];
		Y[label * numExamples + i] = 1.0f;
	}

	// vhodna plast a1
	float * a1 = (float*)calloc((Xcols + 1) * numExamples, sizeof(float));
	for (int j = 0; j < numExamples; j++) {
		a1[j] = 1.0f;
		for (int i = 1; i < (Xcols + 1); i++) {
			a1[i * numExamples + j] = X[j][i - 1];
		}
	}

	// OpenCL inicializacija
	cl_int ret;
	// Podatki o platformi
	cl_platform_id	platform_id[10];
	cl_uint			ret_num_platforms;
	ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);

	// Podatki o napravi
	cl_device_id	device_id[10];
	cl_uint			ret_num_devices;
	ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10, device_id, &ret_num_devices);
	
	// Kontekst
	cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);

	// Ukazna vrsta
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);

	// Beri kernele
	char * programBuffer[1];

	FILE * f;
	size_t fsize;

	f = fopen("kernels.cl", "r");
	if (f == NULL) errexit("Napaka pri branju kernela.");
	programBuffer[0] = (char*)malloc(MAX_SOURCE_SIZE);
	fsize = fread(programBuffer[0], 1, MAX_SOURCE_SIZE, f);
	programBuffer[0][fsize] = '\0';
	fclose(f);

	// Priprava programa
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)programBuffer, NULL, &ret);

	// Prevajanje
	ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);

	// Build log
	size_t build_log_len;
	char *build_log;
	ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
	build_log = (char *)malloc(sizeof(char)*(build_log_len + 1));
	ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);
	printf("%s", build_log);
	free(build_log);

	// Alokacija pomnilnika na napravi
	// vhodni podatki
	cl_mem a1_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		T1cols * numExamples * sizeof(float), a1, &ret);
	cl_mem T1_dev = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		T1rows * T1cols * sizeof(float), T1, &ret);
	cl_mem T2_dev = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		T2rows * T2cols * sizeof(float), T2, &ret);
	cl_mem y_dev = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		yLabels * numExamples * sizeof(float), Y, &ret);
	// feedforward
	cl_mem z2_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
		T1rows * numExamples * sizeof(float), NULL, &ret);
	cl_mem a2_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
		T2cols * numExamples * sizeof(float), NULL, &ret);
	cl_mem a3_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
		T2rows * numExamples * sizeof(float), NULL, &ret);
	// backpropagation
	cl_mem d3_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
		T2rows * numExamples * sizeof(float), NULL, &ret);
	cl_mem T2t_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
		T2cols * T2rows * sizeof(float), NULL, &ret);
	cl_mem d2_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
		T1rows * numExamples * sizeof(float), NULL, &ret);
	cl_mem a1t_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
		numExamples * T1cols * sizeof(float), NULL, &ret);
	cl_mem a2t_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
		numExamples * T2cols * sizeof(float), NULL, &ret);
	cl_mem T1grad_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
		T1rows * T1cols * sizeof(float), NULL, &ret);
	cl_mem T2grad_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
		T2rows * T2cols * sizeof(float), NULL, &ret);

	// sub buffers
	cl_buffer_region region;
	region.size = T1rows * numExamples * sizeof(float);
	region.origin = numExamples * sizeof(float);
	cl_mem a2_dev_sub = clCreateSubBuffer(a2_dev, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &ret);

	region.size = (T2cols - 1) * T2rows * sizeof(float);
	region.origin = T2rows * sizeof(float);
	cl_mem T2t_dev_sub = clCreateSubBuffer(T2t_dev, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &ret);
	
	// postavi enice v prvo vrstico a2 na napravi
	float * ones = (float*)calloc(numExamples, sizeof(float));
	for (int i = 0; i < numExamples; i++) {
		ones[i] = 1.0f;
	}
	ret = clEnqueueWriteBuffer(command_queue, a2_dev, CL_TRUE, 0, numExamples * sizeof(float), ones, 0, NULL, NULL);
	
	// Pripravi kernele
	int vecLen;
	// z2: matricno mnozenje
	cl_kernel kernel_z2 = clCreateKernel(program, "multiplyMatrix", &ret);
	clSetKernelArg(kernel_z2, 0, sizeof(cl_mem), (void *)&T1_dev);
	clSetKernelArg(kernel_z2, 1, sizeof(cl_mem), (void *)&a1_dev);
	clSetKernelArg(kernel_z2, 2, sizeof(cl_mem), (void *)&z2_dev);
	clSetKernelArg(kernel_z2, 3, sizeof(cl_int), (void *)&T1rows);
	clSetKernelArg(kernel_z2, 4, sizeof(cl_int), (void *)&T1cols);
	clSetKernelArg(kernel_z2, 5, sizeof(cl_int), (void *)&T1cols);
	clSetKernelArg(kernel_z2, 6, sizeof(cl_int), (void *)&numExamples);

	size_t localItemSize_z2[2] = { WORKGROUP_SIZE, WORKGROUP_SIZE };
	int globalWorkRows_z2 = (int)(ceil((double)T1rows / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	int globalWorkCols_z2 = (int)(ceil((double)numExamples / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	size_t globalItemSize_z2[2] = { globalWorkRows_z2, globalWorkCols_z2 };

	// a2: sigmoid funkcija
	vecLen = T1rows * numExamples;
	cl_kernel kernel_a2 = clCreateKernel(program, "sigmoid", &ret);
	clSetKernelArg(kernel_a2, 0, sizeof(cl_mem), (void *)&z2_dev);
	clSetKernelArg(kernel_a2, 1, sizeof(cl_mem), (void *)&a2_dev_sub);
	clSetKernelArg(kernel_a2, 2, sizeof(cl_int), (void *)&vecLen);

	size_t localItemSize_a2 = WORKGROUP_SIZE * WORKGROUP_SIZE;
	size_t globalItemSize_a2 = (int)(ceil((double)vecLen / localItemSize_a2) * localItemSize_a2);

	// z3: matricno mnozenje
	cl_kernel kernel_z3 = clCreateKernel(program, "multiplyMatrix", &ret);
	clSetKernelArg(kernel_z3, 0, sizeof(cl_mem), (void *)&T2_dev);
	clSetKernelArg(kernel_z3, 1, sizeof(cl_mem), (void *)&a2_dev);
	clSetKernelArg(kernel_z3, 2, sizeof(cl_mem), (void *)&a3_dev);
	clSetKernelArg(kernel_z3, 3, sizeof(cl_int), (void *)&T2rows);
	clSetKernelArg(kernel_z3, 4, sizeof(cl_int), (void *)&T2cols);
	clSetKernelArg(kernel_z3, 5, sizeof(cl_int), (void *)&T2cols);
	clSetKernelArg(kernel_z3, 6, sizeof(cl_int), (void *)&numExamples);

	size_t localItemSize_z3[2] = { WORKGROUP_SIZE, WORKGROUP_SIZE };
	int globalWorkRows_z3 = (int)(ceil((double)T2rows / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	int globalWorkCols_z3 = (int)(ceil((double)numExamples / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	size_t globalItemSize_z3[2] = { globalWorkRows_z3, globalWorkCols_z3 };

	// a3: sigmoid funkcija
	vecLen = T2rows * numExamples;
	cl_kernel kernel_a3 = clCreateKernel(program, "sigmoid", &ret);
	clSetKernelArg(kernel_a3, 0, sizeof(cl_mem), (void *)&a3_dev);
	clSetKernelArg(kernel_a3, 1, sizeof(cl_mem), (void *)&a3_dev);
	clSetKernelArg(kernel_a3, 2, sizeof(cl_int), (void *)&vecLen);

	size_t localItemSize_a3 = WORKGROUP_SIZE * WORKGROUP_SIZE;
	size_t globalItemSize_a3 = (int)(ceil((double)vecLen / localItemSize_a3) * localItemSize_a3);

	// d3: odstevanje po elementih
	vecLen = T2rows * numExamples;
	cl_kernel kernel_d3 = clCreateKernel(program, "subtractElements", &ret);
	clSetKernelArg(kernel_d3, 0, sizeof(cl_mem), (void *)&a3_dev);
	clSetKernelArg(kernel_d3, 1, sizeof(cl_mem), (void *)&y_dev);
	clSetKernelArg(kernel_d3, 2, sizeof(cl_mem), (void *)&d3_dev);
	clSetKernelArg(kernel_d3, 3, sizeof(cl_int), (void *)&vecLen);

	size_t localItemSize_d3 = WORKGROUP_SIZE * WORKGROUP_SIZE;
	size_t globalItemSize_d3 = (int)(ceil((double)vecLen / localItemSize_d3) * localItemSize_d3);

	// d2: mnozenje z transponirano matriko, sigmoid gradient
	cl_kernel kernel_d2_t = clCreateKernel(program, "transposeMatrix", &ret);
	clSetKernelArg(kernel_d2_t, 0, sizeof(cl_mem), (void *)&T2_dev);
	clSetKernelArg(kernel_d2_t, 1, sizeof(cl_mem), (void *)&T2t_dev);
	clSetKernelArg(kernel_d2_t, 2, sizeof(cl_int), (void *)&T2rows);
	clSetKernelArg(kernel_d2_t, 3, sizeof(cl_int), (void *)&T2cols);

	size_t localItemSize_d2_t[2] = { WORKGROUP_SIZE, WORKGROUP_SIZE };
	int globalWorkRows_d2_t = (int)(ceil((double)T2rows / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	int globalWorkCols_d2_t = (int)(ceil((double)T2cols / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	size_t globalItemSize_d2_t[2] = { globalWorkRows_d2_t, globalWorkCols_d2_t };

	cl_kernel kernel_d2_m = clCreateKernel(program, "multiplyMatrix", &ret);
	clSetKernelArg(kernel_d2_m, 0, sizeof(cl_mem), (void *)&T2t_dev_sub);
	clSetKernelArg(kernel_d2_m, 1, sizeof(cl_mem), (void *)&d3_dev);
	clSetKernelArg(kernel_d2_m, 2, sizeof(cl_mem), (void *)&d2_dev);
	clSetKernelArg(kernel_d2_m, 3, sizeof(cl_int), (void *)&T1rows);
	clSetKernelArg(kernel_d2_m, 4, sizeof(cl_int), (void *)&T2rows);
	clSetKernelArg(kernel_d2_m, 5, sizeof(cl_int), (void *)&T2rows);
	clSetKernelArg(kernel_d2_m, 6, sizeof(cl_int), (void *)&numExamples);

	size_t localItemSize_d2_m[2] = { WORKGROUP_SIZE, WORKGROUP_SIZE };
	int globalWorkRows_d2_m = (int)(ceil((double)T1rows / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	int globalWorkCols_d2_m = (int)(ceil((double)numExamples / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	size_t globalItemSize_d2_m[2] = { globalWorkRows_d2_m, globalWorkCols_d2_m };

	vecLen = T1rows * numExamples;
	cl_kernel kernel_d2_sg = clCreateKernel(program, "sigmoidGradient", &ret);
	clSetKernelArg(kernel_d2_sg, 0, sizeof(cl_mem), (void *)&z2_dev);
	clSetKernelArg(kernel_d2_sg, 1, sizeof(cl_mem), (void *)&z2_dev);
	clSetKernelArg(kernel_d2_sg, 2, sizeof(cl_int), (void *)&vecLen);

	size_t localItemSize_d2_sg = WORKGROUP_SIZE * WORKGROUP_SIZE;
	size_t globalItemSize_d2_sg = (int)(ceil((double)vecLen / localItemSize_d2_sg) * localItemSize_d2_sg);

	vecLen = T1rows * numExamples;
	cl_kernel kernel_d2 = clCreateKernel(program, "multiplyElements", &ret);
	clSetKernelArg(kernel_d2, 0, sizeof(cl_mem), (void *)&d2_dev);
	clSetKernelArg(kernel_d2, 1, sizeof(cl_mem), (void *)&z2_dev);
	clSetKernelArg(kernel_d2, 2, sizeof(cl_mem), (void *)&d2_dev);
	clSetKernelArg(kernel_d2, 3, sizeof(cl_int), (void *)&vecLen);

	size_t localItemSize_d2 = WORKGROUP_SIZE * WORKGROUP_SIZE;
	size_t globalItemSize_d2 = (int)(ceil((double)vecLen / localItemSize_d2) * localItemSize_d2);

	// T1grad: mnozenje s transponirano matriko
	cl_kernel kernel_a1t = clCreateKernel(program, "transposeMatrix", &ret);
	clSetKernelArg(kernel_a1t, 0, sizeof(cl_mem), (void *)&a1_dev);
	clSetKernelArg(kernel_a1t, 1, sizeof(cl_mem), (void *)&a1t_dev);
	clSetKernelArg(kernel_a1t, 2, sizeof(cl_int), (void *)&T1cols);
	clSetKernelArg(kernel_a1t, 3, sizeof(cl_int), (void *)&numExamples);

	size_t localItemSize_a1t[2] = { WORKGROUP_SIZE, WORKGROUP_SIZE };
	int globalWorkRows_a1t = (int)(ceil((double)T1cols / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	int globalWorkCols_a1t = (int)(ceil((double)numExamples / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	size_t globalItemSize_a1t[2] = { globalWorkRows_a1t, globalWorkCols_a1t };

	cl_kernel kernel_T1grad = clCreateKernel(program, "multiplyMatrix", &ret);
	clSetKernelArg(kernel_T1grad, 0, sizeof(cl_mem), (void *)&d2_dev);
	clSetKernelArg(kernel_T1grad, 1, sizeof(cl_mem), (void *)&a1t_dev);
	clSetKernelArg(kernel_T1grad, 2, sizeof(cl_mem), (void *)&T1grad_dev);
	clSetKernelArg(kernel_T1grad, 3, sizeof(cl_int), (void *)&T1rows);
	clSetKernelArg(kernel_T1grad, 4, sizeof(cl_int), (void *)&numExamples);
	clSetKernelArg(kernel_T1grad, 5, sizeof(cl_int), (void *)&numExamples);
	clSetKernelArg(kernel_T1grad, 6, sizeof(cl_int), (void *)&T1cols);

	size_t localItemSize_T1grad[2] = { WORKGROUP_SIZE, WORKGROUP_SIZE };
	int globalWorkRows_T1grad = (int)(ceil((double)T1rows / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	int globalWorkCols_T1grad = (int)(ceil((double)T1cols / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	size_t globalItemSize_T1grad[2] = { globalWorkRows_T1grad, globalWorkCols_T1grad };

	// T2grad: mnozenje s transponirano matriko
	cl_kernel kernel_a2t = clCreateKernel(program, "transposeMatrix", &ret);
	clSetKernelArg(kernel_a2t, 0, sizeof(cl_mem), (void *)&a2_dev);
	clSetKernelArg(kernel_a2t, 1, sizeof(cl_mem), (void *)&a2t_dev);
	clSetKernelArg(kernel_a2t, 2, sizeof(cl_int), (void *)&T2cols);
	clSetKernelArg(kernel_a2t, 3, sizeof(cl_int), (void *)&numExamples);

	size_t localItemSize_a2t[2] = { WORKGROUP_SIZE, WORKGROUP_SIZE };
	int globalWorkRows_a2t = (int)(ceil((double)T2cols / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	int globalWorkCols_a2t = (int)(ceil((double)numExamples / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	size_t globalItemSize_a2t[2] = { globalWorkRows_a2t, globalWorkCols_a2t };

	cl_kernel kernel_T2grad = clCreateKernel(program, "multiplyMatrix", &ret);
	clSetKernelArg(kernel_T2grad, 0, sizeof(cl_mem), (void *)&d3_dev);
	clSetKernelArg(kernel_T2grad, 1, sizeof(cl_mem), (void *)&a2t_dev);
	clSetKernelArg(kernel_T2grad, 2, sizeof(cl_mem), (void *)&T2grad_dev);
	clSetKernelArg(kernel_T2grad, 3, sizeof(cl_int), (void *)&T2rows);
	clSetKernelArg(kernel_T2grad, 4, sizeof(cl_int), (void *)&numExamples);
	clSetKernelArg(kernel_T2grad, 5, sizeof(cl_int), (void *)&numExamples);
	clSetKernelArg(kernel_T2grad, 6, sizeof(cl_int), (void *)&T2cols);

	size_t localItemSize_T2grad[2] = { WORKGROUP_SIZE, WORKGROUP_SIZE };
	int globalWorkRows_T2grad = (int)(ceil((double)T2rows / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	int globalWorkCols_T2grad = (int)(ceil((double)T2cols / WORKGROUP_SIZE) * WORKGROUP_SIZE);
	size_t globalItemSize_T2grad[2] = { globalWorkRows_T2grad, globalWorkCols_T2grad };
	
	// T1grad: normalizacija in regularizacija
	cl_kernel kernel_T1grad_reg = clCreateKernel(program, "normRegGrad", &ret);
	clSetKernelArg(kernel_T1grad_reg, 0, sizeof(cl_mem), (void *)&T1grad_dev);
	clSetKernelArg(kernel_T1grad_reg, 1, sizeof(cl_mem), (void *)&T1_dev);
	clSetKernelArg(kernel_T1grad_reg, 2, sizeof(cl_int), (void *)&T1rows);
	clSetKernelArg(kernel_T1grad_reg, 3, sizeof(cl_int), (void *)&T1cols);
	clSetKernelArg(kernel_T1grad_reg, 4, sizeof(cl_int), (void *)&numExamples);
	clSetKernelArg(kernel_T1grad_reg, 5, sizeof(cl_float), (void *)&lambda);

	// T2grad: normalizacija in regularizacija
	cl_kernel kernel_T2grad_reg = clCreateKernel(program, "normRegGrad", &ret);
	clSetKernelArg(kernel_T2grad_reg, 0, sizeof(cl_mem), (void *)&T2grad_dev);
	clSetKernelArg(kernel_T2grad_reg, 1, sizeof(cl_mem), (void *)&T2_dev);
	clSetKernelArg(kernel_T2grad_reg, 2, sizeof(cl_int), (void *)&T2rows);
	clSetKernelArg(kernel_T2grad_reg, 3, sizeof(cl_int), (void *)&T2cols);
	clSetKernelArg(kernel_T2grad_reg, 4, sizeof(cl_int), (void *)&numExamples);
	clSetKernelArg(kernel_T2grad_reg, 5, sizeof(cl_float), (void *)&lambda);

	float alpha = 1.0f;
	// T1: posodobi parametre
	vecLen = T1rows * T1cols;
	cl_kernel kernel_T1_update = clCreateKernel(program, "updateParameters", &ret);
	clSetKernelArg(kernel_T1_update, 0, sizeof(cl_mem), (void *)&T1_dev);
	clSetKernelArg(kernel_T1_update, 1, sizeof(cl_mem), (void *)&T1grad_dev);
	clSetKernelArg(kernel_T1_update, 2, sizeof(cl_int), (void *)&vecLen);
	clSetKernelArg(kernel_T1_update, 3, sizeof(cl_float), (void *)&alpha);

	size_t localItemSize_T1_update = WORKGROUP_SIZE * WORKGROUP_SIZE;
	size_t globalItemSize_T1_update = (int)(ceil((double)vecLen / localItemSize_T1_update) * localItemSize_T1_update);

	// T2: posodobi parametre
	vecLen = T2rows * T2cols;
	cl_kernel kernel_T2_update = clCreateKernel(program, "updateParameters", &ret);
	clSetKernelArg(kernel_T2_update, 0, sizeof(cl_mem), (void *)&T2_dev);
	clSetKernelArg(kernel_T2_update, 1, sizeof(cl_mem), (void *)&T2grad_dev);
	clSetKernelArg(kernel_T2_update, 2, sizeof(cl_int), (void *)&vecLen);
	clSetKernelArg(kernel_T2_update, 3, sizeof(cl_float), (void *)&alpha);

	size_t localItemSize_T2_update = WORKGROUP_SIZE * WORKGROUP_SIZE;
	size_t globalItemSize_T2_update = (int)(ceil((double)vecLen / localItemSize_T2_update) * localItemSize_T2_update);

	// Gradient descent

	// a1 lahko transponiramo izven zanke
	ret = clEnqueueNDRangeKernel(command_queue, kernel_a1t, 2, NULL,
		globalItemSize_a1t, localItemSize_a1t, 0, NULL, NULL);

	for (int iter = 0; iter < iterations; iter++) {
		// Feedforward
		ret = clEnqueueNDRangeKernel(command_queue, kernel_z2, 2, NULL,
			globalItemSize_z2, localItemSize_z2, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(command_queue, kernel_a2, 1, NULL,
			&globalItemSize_a2, &localItemSize_a2, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(command_queue, kernel_z3, 2, NULL,
			globalItemSize_z3, localItemSize_z3, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(command_queue, kernel_a3, 1, NULL,
			&globalItemSize_a3, &localItemSize_a3, 0, NULL, NULL);

		// backpropagation
		ret = clEnqueueNDRangeKernel(command_queue, kernel_d3, 1, NULL,
			&globalItemSize_d3, &localItemSize_d3, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(command_queue, kernel_d2_t, 2, NULL,
			globalItemSize_d2_t, localItemSize_d2_t, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(command_queue, kernel_d2_m, 2, NULL,
			globalItemSize_d2_m, localItemSize_d2_m, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(command_queue, kernel_d2_sg, 1, NULL,
			&globalItemSize_d2_sg, &localItemSize_d2_sg, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(command_queue, kernel_d2, 1, NULL,
			&globalItemSize_d2, &localItemSize_d2, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(command_queue, kernel_T1grad, 2, NULL,
			globalItemSize_T1grad, localItemSize_T1grad, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(command_queue, kernel_a2t, 2, NULL,
			globalItemSize_a2t, localItemSize_a2t, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(command_queue, kernel_T2grad, 2, NULL,
			globalItemSize_T2grad, localItemSize_T2grad, 0, NULL, NULL);

		// gradient normalization and regularization
		ret = clEnqueueNDRangeKernel(command_queue, kernel_T1grad_reg, 2, NULL,
			globalItemSize_T1grad, localItemSize_T1grad, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(command_queue, kernel_T2grad_reg, 2, NULL,
			globalItemSize_T2grad, localItemSize_T2grad, 0, NULL, NULL);

		// posodobi parametre
		ret = clEnqueueNDRangeKernel(command_queue, kernel_T1_update, 1, NULL,
			&globalItemSize_T1_update, &localItemSize_T1_update, 0, NULL, NULL);

		ret = clEnqueueNDRangeKernel(command_queue, kernel_T2_update, 1, NULL,
			&globalItemSize_T2_update, &localItemSize_T2_update, 0, NULL, NULL);
	}

	// Koncni cost function
	float * a3 = (float*)calloc(T2rows * numExamples, sizeof(float));
	ret = clEnqueueReadBuffer(command_queue, a3_dev, CL_TRUE, 0,
		T2rows * numExamples * sizeof(float), a3, 0, NULL, NULL);
	
	ret = clEnqueueReadBuffer(command_queue, T1_dev, CL_TRUE, 0,
		T1rows * T1cols * sizeof(float), T1, 0, NULL, NULL);
	
	ret = clEnqueueReadBuffer(command_queue, T2_dev, CL_TRUE, 0,
		T2rows * T2cols * sizeof(float), T2, 0, NULL, NULL);
	
	float cost = 0.0f;
	for (int i = 0; i < (T2rows * numExamples); i++) {
		cost += (-Y[i] * (float)log(a3[i])) - ((1.0f - Y[i]) * (float)log(1.0f - a3[i]));
	}
	cost = (1.0f / numExamples) * cost;
	free(a3);

	float reg = 0.0f;
	for (int i = 0; i < T1rows; i++) {
		for (int j = 1; j < T1cols; j++) {
			float tmp = T1[i * T1cols + j];
			reg += tmp * tmp;
		}
	}
	for (int i = 0; i < T2rows; i++) {
		for (int j = 1; j < T2cols; j++) {
			float tmp = T2[i * T2cols + j];
			reg += tmp * tmp;
		}
	}
	cost += (lambda / (2.0f * numExamples)) * reg;
	printf("Iteration: %4d | Cost: %4.6f\n", iterations, cost);

	// Pretvori parametre nazaj v ustrezno obliko
	iparam = 0;
	for (int j = 0; j < T1cols; j++) {
		for (int i = 0; i < T1rows; i++) {
			param[iparam] = T1[i * T1cols + j];
			iparam++;
		}
	}
	for (int j = 0; j < T2cols; j++) {
		for (int i = 0; i < T2rows; i++) {
			param[iparam] = T2[i * T2cols + j];
			iparam++;
		}
	}

	// Ciscenje
	ret = clReleaseKernel(kernel_z2);
	ret = clReleaseKernel(kernel_a2);
	ret = clReleaseKernel(kernel_z3);
	ret = clReleaseKernel(kernel_a3);
	ret = clReleaseKernel(kernel_d3);
	ret = clReleaseKernel(kernel_d2_t);
	ret = clReleaseKernel(kernel_d2_m);
	ret = clReleaseKernel(kernel_d2_sg);
	ret = clReleaseKernel(kernel_d2);
	ret = clReleaseKernel(kernel_a1t);
	ret = clReleaseKernel(kernel_T1grad);
	ret = clReleaseKernel(kernel_a2t);
	ret = clReleaseKernel(kernel_T2grad);
	ret = clReleaseKernel(kernel_T1grad_reg);
	ret = clReleaseKernel(kernel_T2grad_reg);
	ret = clReleaseKernel(kernel_T1_update);
	ret = clReleaseKernel(kernel_T2_update);

	ret = clReleaseMemObject(a1_dev);
	ret = clReleaseMemObject(T1_dev);
	ret = clReleaseMemObject(T2_dev);
	ret = clReleaseMemObject(y_dev);

	ret = clReleaseMemObject(z2_dev);
	ret = clReleaseMemObject(a2_dev);
	ret = clReleaseMemObject(a3_dev);

	ret = clReleaseMemObject(d3_dev);
	ret = clReleaseMemObject(T2t_dev);
	ret = clReleaseMemObject(d2_dev);
	ret = clReleaseMemObject(a1t_dev);
	ret = clReleaseMemObject(a2t_dev);
	ret = clReleaseMemObject(T1grad_dev);
	ret = clReleaseMemObject(T2grad_dev);

	ret = clReleaseMemObject(a2_dev_sub);
	ret = clReleaseMemObject(T2t_dev_sub);

	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseProgram(program);
	ret = clReleaseContext(context);
	
	free(T1);
	free(T2);
	free(Y);
	free(a1);

	return cost;
}
