
#define WORKGROUP_SIZE	16

__kernel void multiplyMatrix(
	__global float * A,
	__global float * B,
	__global float * C,
	int rowsA,
	int colsA,
	int rowsB,
	int colsB)
{
	// lokalni pomnilnik
	__local float Atile[WORKGROUP_SIZE][WORKGROUP_SIZE+1];
	__local float Btile[WORKGROUP_SIZE][WORKGROUP_SIZE+1];

	// globalni indeks
	int row = get_global_id(0);
	int col = get_global_id(1);

	// indeks znotraj bloka
	int row_t = get_local_id(0);
	int col_t = get_local_id(1);
	
	float acc = 0.0f;
	for (int tile = 0; tile < (colsA - 1) / WORKGROUP_SIZE + 1; tile++) {
		// kopiranje podatkov v lokalni pomnilnik
		int col_tile = tile * WORKGROUP_SIZE + col_t;
		if (row < rowsA && col_tile < colsA) {
			Atile[row_t][col_t] = A[row * colsA + col_tile];
		} else {
			Atile[row_t][col_t] = 0.0f;
		}
		int row_tile = tile * WORKGROUP_SIZE + row_t;
		if (row_tile < rowsB && col < colsB) {
			Btile[row_t][col_t] = B[row_tile * colsB + col];
		} else {
			Btile[row_t][col_t] = 0.0f;
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		// nit poracuna svojo vsoto
		for (int i = 0; i < WORKGROUP_SIZE; i++) {
			acc += Atile[row_t][i] * Btile[i][col_t];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// zapisi vsoto v pomnilnik
	if (row < rowsA && col < colsB) {
		C[row * colsB + col] = acc;
	}
}

__kernel void transposeMatrix(
	__global float * A,
	__global float * At,
	int rowsA,
	int colsA)
{
	// lokalni pomnilnik
	__local float tile[WORKGROUP_SIZE][WORKGROUP_SIZE + 1];

	// globalni indeks
	int row = get_global_id(0);
	int col = get_global_id(1);

	// indeks znotraj bloka
	int row_t = get_local_id(0);
	int col_t = get_local_id(1);

	// kopiranje v lokalni pomnilnik
	if (row < rowsA && col < colsA) {
		tile[row_t][col_t] = A[row * colsA + col];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// transponirano pisanje v glavni pomnilnik
	row = get_group_id(1) * WORKGROUP_SIZE + get_local_id(0);
	col = get_group_id(0) * WORKGROUP_SIZE + get_local_id(1);

	if (row < colsA && col < rowsA) {
		At[row * rowsA + col] = tile[col_t][row_t];
	}
}

__kernel void sigmoid(
	__global float * op,
	__global float * result,
	int len)
{
	int i = get_global_id(0);

	if (i < len) {
		result[i] = 1.0f / (1.0f + exp((-1) * op[i]));
	}
}

__kernel void sigmoidGradient(
	__global float * op,
	__global float * result,
	int len)
{
	int i = get_global_id(0);

	if (i < len) {
		float sig = 1.0f / (1.0f + exp((-1) * op[i]));
		result[i] = sig * (1.0f - sig);
	}
}

__kernel void subtractElements(
	__global float * op1,
	__global float * op2,
	__global float * result,
	int len)
{
	int i = get_global_id(0);

	if (i < len) {
		result[i] = op1[i] - op2[i];
	}
}

__kernel void multiplyElements(
	__global float * op1,
	__global float * op2,
	__global float * result,
	int len)
{
	int i = get_global_id(0);

	if (i < len) {
		result[i] = op1[i] * op2[i];
	}
}

__kernel void normRegGrad(
	__global float * Tgrad,
	__global float * T,
	int Trows,
	int Tcols,
	int numExamples,
	float lambda)
{
	int row = get_global_id(0);
	int col = get_global_id(1);

	if (row < Trows && col < Tcols) {
		float tmp = Tgrad[row*Tcols + col] / numExamples;
		if (col > 0) {
			tmp = tmp + (lambda / numExamples) * T[row*Tcols + col];
		}
		Tgrad[row*Tcols + col] = tmp;
	}
}

__kernel void updateParameters(
	__global float * T,
	__global float * Tgrad,
	int len,
	float alpha)
{
	int i = get_global_id(0);

	if (i < len) {
		float tmp = T[i] - alpha * Tgrad[i];
		T[i] = tmp;
	}
}