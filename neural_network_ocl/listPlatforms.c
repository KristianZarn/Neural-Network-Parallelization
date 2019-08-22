#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

int main(void) 
{
	cl_int ret;

    // Podatki o platformi
    cl_platform_id	platform_id[10];
    cl_uint			ret_num_platforms;
	char			*buf;
	size_t			buf_len;

	// Podatki o napravi
	cl_device_id	device_id[10];
	cl_uint			ret_num_devices;
	char			buffer[10240];
	cl_uint			buf_uint;
	cl_ulong		buf_ulong;
	size_t			buf_size_t;

	ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
			// max. "stevilo platform, kazalec na platforme, dejansko "stevilo platform
	
	for(int i=0; i<ret_num_platforms; i++)
	{
		printf("platforma[%d]:\n", i);
		ret = clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, 0, NULL, &buf_len);	
				// dejanska dol"zina niza: 0, NULL, kazalec na dol!zinos
		buf = (char *)malloc(sizeof(char*)*(buf_len+1));
		ret = clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, buf_len, buf, NULL);	
				// vsebina: buf_len, buf, NULL
		printf("  %s\n", buf);
		free(buf);

		ret = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_ALL, 10,	
							 device_id, &ret_num_devices);
				// izbrana platforma, naprava, koliko naprav nas zanima, 
				// kazalec na naprave, dejansko "stevilo naprav
		for(int j=0; j<ret_num_devices; j++)
		{
			printf("  naprava[%d]:\n", j);
			clGetDeviceInfo(device_id[j], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
			printf("    DEVICE_NAME = %s\n", buffer);
			clGetDeviceInfo(device_id[j], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
			printf("    DEVICE_VENDOR = %s\n", buffer);
			clGetDeviceInfo(device_id[j], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
			printf("    DEVICE_VERSION = %s\n", buffer);
			clGetDeviceInfo(device_id[j], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
			printf("    DRIVER_VERSION = %s\n", buffer);
			clGetDeviceInfo(device_id[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
			printf("    DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
			clGetDeviceInfo(device_id[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
			printf("    DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
			clGetDeviceInfo(device_id[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
			printf("    DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
			clGetDeviceInfo(device_id[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf_size_t), &buf_size_t, NULL);
			printf("    DEVICE_MAX_WORK_GROUP_SIZE = %u\n", (size_t)buf_size_t);
			clGetDeviceInfo(device_id[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(buf_uint), &buf_uint, NULL);
			printf("    DEVICE_MAX_WORK_ITEM_DIMENSIONS = %u\n", (unsigned int)buf_uint);
			clGetDeviceInfo(device_id[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, sizeof(buf_uint), &buf_uint, NULL);
			printf("    DEVICE_NATIVE_VECTOR_WIDTH_FLOAT = %u\n", (unsigned int)buf_uint);
		}
	}

    return 0;
}
