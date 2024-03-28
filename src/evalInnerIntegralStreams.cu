//
// Created by gkluhana on 26/03/24.
//

#include "../include/evalInnerIntegralStreams.h"
namespace cuslater {
//__constant__ double c1[3];
//__constant__ double c2[3];
//__constant__ double c3[3];
//__constant__ double c4[3];
#define THREADS_PER_BLOCK 128


    double evaluateInnerWithStreams(double* c1234_input,
                                    double r,
                                    double* w_input, double* w_weights, int Nl,
                                    double* xrange, double* yrange, double* zrange,
                                    unsigned int x_axis_points, unsigned int y_axis_points, unsigned int z_axis_points,
                                    int gpu_num)
    {
            cudaSetDevice(gpu_num);
            int device;
            HANDLE_CUDA_ERROR(cudaGetDevice(&device));

            //std::cout << "Current GPU device index: " << device << std::endl;
            //std::cout << "c1234_input" << std::endl;
            //for (int i = 0; i<12; ++i){
            //	std::cout << c1234_input[i] << std::endl;
            //}
            //std::cout << "r " << r << std::endl;
            //
            //std::cout << "w_input" << std::endl;
            //for (int i = 0; i< (Nl*3); ++i){
            //	std::cout << w_input[i] << std::endl;
            //}

            //std::cout << "w_weights" << std::endl;
            //for (int i = 0; i<Nl; ++i){
            //	std::cout << w_weights[i] << std::endl;
            //}
            //std::cout << "Nl" << Nl << std::endl;
//        std::cout << "x_axis_points" << x_axis_points << std::endl;
//        std::cout << "y_axis_points" << y_axis_points << std::endl;
//        std::cout << "z_axis_points" << z_axis_points << std::endl;
//	std::cout << "gpu num " << gpu_num << std::endl;
            int num_streams = 1;
            if (Nl < num_streams) num_streams = 1;
            cudaStream_t streams[num_streams];
            bool allocate_memory[num_streams] = {false};
            std::fill_n(allocate_memory, num_streams, true);
            double ** d_results = new double*[Nl];


            double *d_x_grid = nullptr;
//	double *d_y_grid = nullptr;
//	double *d_z_grid = nullptr;
            double *d_x_weights = nullptr;
//	double *d_y_weights = nullptr;
//	double *d_z_weights = nullptr;
            double *d_c1234 = nullptr;

            std::vector<double> x_grid;
//   	std::vector<double> y_grid;
            // 	std::vector<double> z_grid;

            std::vector<double> x_weights;
            //	std::vector<double> y_weights;
            //	std::vector<double> z_weights;

            make_1d_grid_simpson(xrange[0],xrange[1],x_axis_points, &x_grid, &x_weights);
//	make_1d_grid_simpson(yrange[0],yrange[1],y_axis_points, &y_grid, &y_weights);
//	make_1d_grid_simpson(zrange[0],zrange[1],z_axis_points, &z_grid, &z_weights);

            unsigned int PX = x_grid.size();
            unsigned int PY = x_grid.size();
            unsigned int PZ = x_grid.size();

            // Allocate memory on GPU
            // TODO: write a preprocess function that generates and saves grid_files and weights on GPU
            HANDLE_CUDA_ERROR( cudaMalloc(&d_x_grid, PX*sizeof(double)) );
            HANDLE_CUDA_ERROR( cudaMalloc(&d_x_weights, PX*sizeof(double)) );
            HANDLE_CUDA_ERROR( cudaMalloc(&d_c1234, 12*sizeof(double) ));

            // Evaluate Funciton on GPU
            assert(PX== (x_axis_points+1));
            assert(PY== (y_axis_points+1));
            assert(PZ== (z_axis_points+1));


            HANDLE_CUDA_ERROR(cudaMemcpy(d_x_grid, x_grid.data(), PX*sizeof(double), cudaMemcpyHostToDevice));
            HANDLE_CUDA_ERROR(cudaMemcpy(d_x_weights, x_weights.data(), PX*sizeof(double), cudaMemcpyHostToDevice));
            HANDLE_CUDA_ERROR(cudaMemcpy(d_c1234, c1234_input, 12*sizeof(double), cudaMemcpyHostToDevice));


            //    double *result = new double[(PX)*(PY)*(PZ)]();
            int threads_eval = 96; // Max threads per block
            int blocks = (PX*PY*PZ + threads_eval -1)/threads_eval; // Max blocks, better if multiple of SM = 80
            int threads_sum = 96;

            //      std::cout << "Number of Streams= " << num_streams<< std::endl;
            double** d_ws = new double*[Nl];
            double** results = new double*[Nl];
            double* d_w_weights;
            //      double sum_array_streams[Nl];
            double* d_sum_array_ws = new double[Nl];
            for (int i=0; i<num_streams; ++i)
            {
                    //	std::cout<< "creating stream for i= "<< i << std::endl;
                    cudaStreamCreate(&streams[i]);
                    if (i==0) {
                            HANDLE_CUDA_ERROR(cudaMallocAsync(&d_sum_array_ws, Nl*sizeof(double), streams[0]));
                            HANDLE_CUDA_ERROR(cudaMallocAsync(&d_w_weights, Nl*sizeof(double), streams[0]));
                            HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_w_weights, w_weights, Nl*sizeof(double), cudaMemcpyHostToDevice, streams[0]));
                    }
            }
            for (int i =0; i< num_streams; ++i) {
                    // Divide the l's among the streams... right now it's 2 l per stream... probably not efficient
                    int num_elements = Nl/num_streams;
                    for (int l_i=0; l_i< num_elements; ++l_i){
                            if (allocate_memory[i])
                            {
                                    //std::cout<< "Allocating memory manually"	<< std::endl;
                                    //std::cout<< "Stream"	<< i << std::endl;
                                    HANDLE_CUDA_ERROR(cudaMallocAsync(&d_results[i], PX*PY*PZ* sizeof(double),streams[i]));
                                    allocate_memory[i] = false;
                            }
                            auto w_i = i*num_elements + l_i;
                            //		std::cout << "l_i,w_i, i: " << l_i << " , " <<  w_i <<	" , " << i << std::endl;
                            //allocate memory for w coordinates on device
                            HANDLE_CUDA_ERROR(cudaMallocAsync(&d_ws[w_i], 3*sizeof(double),streams[i]));
                            //copy w coordinates to device
                            HANDLE_CUDA_ERROR(cudaMemcpyAsync(d_ws[w_i], &w_input[w_i*3], 3*sizeof(double), cudaMemcpyHostToDevice,streams[i]));
                            //set all values to 0 for next grid evaluation (assumes multiple l's per SM)
                            //need to make this asynchronous
                            //cudaMemsetAsync(d_results[i],0.0, blocks*sizeof(double), streams[i]);
                            //evaluate grid for current value of l
                            evaluateInnerIntegrand<<<blocks,threads_eval,0, streams[i]>>>(d_c1234,
                                                                                          d_x_grid, d_x_grid, d_x_grid,
                                                                                          d_x_weights, d_x_weights, d_x_weights,
                                                                                          x_grid.size(),r, d_ws[w_i], d_results[i]);

                            // made this into a kernel launch for asynchronous execution over streams
                            // work in streams is queued in order
                            reduceSumWrapper<<<1,1,0,streams[i]>>>(d_results[i], PX*PY*PZ, threads_sum);

                            // Copy value to array
//			cudaMemcpyAsync(&sum_array_streams[w_i], d_results[i], sizeof(double), cudaMemcpyDeviceToHost,streams[i]);
                            HANDLE_CUDA_ERROR(cudaMemcpyAsync(&d_sum_array_ws[w_i], d_results[i], sizeof(double), cudaMemcpyDeviceToDevice, streams[i]));
                    }
            }
            HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

            reduceSumWithWeights<<<1, Nl, Nl* sizeof(double)>>>(d_sum_array_ws, d_sum_array_ws, d_w_weights, Nl);
//	}
            HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
            double sumGPU=0.0;
            HANDLE_CUDA_ERROR(cudaMemcpy(&sumGPU, d_sum_array_ws, sizeof(double), cudaMemcpyDeviceToHost));

            //for(int i=0; i<Nl; ++i){
            //	sumGPU += sum_array_streams[i]*w_weights[i];
            //		std::cout << "i: " << i <<  std::endl;
            //		std::cout << "sum_array_streams[i]: " << sum_array_streams[i] <<  std::endl;
            //		std::cout << "w_weights[i]: " << w_weights[i] <<  std::endl;
            //	}
            //std::cout << "Number of Streams: " << num_streams << std::endl;
//  	std::cout << "Sum on GPU: " << sumGPU << std::endl;
            // Destroy CUDA streams
            for (int i = 0; i < num_streams; ++i) {
                    HANDLE_CUDA_ERROR(cudaStreamDestroy(streams[i]));
                    HANDLE_CUDA_ERROR(cudaFree(d_results[i]));
            }
            for (int i=0; i< Nl; ++i){
                    HANDLE_CUDA_ERROR(cudaFree(d_ws[i]));
            }
            HANDLE_CUDA_ERROR(cudaFree(d_sum_array_ws));
            HANDLE_CUDA_ERROR(cudaFree(d_w_weights));
            HANDLE_CUDA_ERROR(cudaFree(d_x_grid));
            HANDLE_CUDA_ERROR(cudaFree(d_x_weights));
            HANDLE_CUDA_ERROR(cudaFree(d_c1234));
            HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

            HANDLE_CUDA_ERROR(cudaGetLastError());

            return sumGPU;
    }//evaluateInnerStreams




}
