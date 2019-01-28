/*Copyright (c) Sebastian Hahn

Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/


#include <torch/torch.h>
#include <random>
#include <vector>
#include <limits>


#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 32

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<
    typename T,
    int nbLabels,
    bool use_debug
>
void
__global__
forward
(
    T const * const unaries,
    T const * const binaries,
    int * sample,
    int * next_sample,
    int * result,
    int const nbCols,
    int const nbRows,
    int const * const neighborhood,
    int const nbNeighbors,
    T const * const randValues,
    T * min_value_gpu,
    T * energies_gpu,
    T * probs_gpu
)
{
/**
    This function implements the forward step for sampling based interferance on GPU.
    @tparam T: Type of the values of the calculation. Should be float or double
    @tparam nbLabels: how many labels are used.
    @tparam use_debug: True to store additional debug informations. False otherwise
    @param unaries. Values for the unaries. This is a Tensor of shape [nb_Dims, nbCols, nbRows]
    @param binaries. Values for the binaries. This is a Tensor of shape [nbNeighbors, nbLabels, nbLabels]
    @param sample: Values for the sample. This is an input and output variable. The Tensor has the shape [nbCols,
    nbRows]. The values of this tensor are in the range from 0 to nb_dims - 1
    @param nbCols: number of cols.
    @param nbRows: number of rows
    @param neighborhood: Defines the neighborhood. Each neighbor is a tupel [distance_rows, distance_cols].
    @param nbNeighbors: number of neighbors.
    @param randValues: this is an array of shape [nbCols, nbRows]. The array defines random values in the range 0 to 1.
    @param min_value_gpu: Stores the minimum value for each thread on gpu. This is only enabled if use_debug is set to
    true
    @param energies_gpu: stores the energies of calculation
    @param probs_gpu: stores the probabilities of the calculation.
*/
    // postion
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = threadIdx.y + blockIdx.y * blockDim.y;

    // border
    if(col >= nbCols)
        return;
    if(row >= nbRows)
        return;

    // random value
    T const rand_value = randValues[row + col * nbRows];

#define MAX(X, Y) ((X) > (Y)) * X + ((X) <= (Y)) * Y
#define MIN(X, Y) ((X) > (Y)) * Y + ((X) <= (Y)) * X

// unarie has shape [nbLabels, Rows, Cols]
#define UNARIES_POS(L, C, R) ( (R) +  (C) * nbRows + (L) * nbRows * nbCols)

// binary has shape [nbNeighbors, nbLabels, nbLabels]
// sample has shape [Rows, Cols]

#define BINARY_POS(N, L, C, R) ((N) * nbLabels * nbLabels /* the neighbor */ \
    + ((L) * nbLabels)  /* the label at the current position */ \
    + (sample[((C) + neighborhood[2 *(N)]) * nbRows + (R) + neighborhood[2 *(N) + 1]  ])) /* the label of the neighbor*/

    // the values
    T values[nbLabels];

    T min_value= static_cast<T>(1000000000);
    // sum the energies
    for(auto cur_label=0; cur_label<nbLabels; ++cur_label)
    {
        T value = unaries[UNARIES_POS(cur_label, col, row)];

        for(auto i=0; i<nbNeighbors; ++i)
        {
            if (col + neighborhood[2*i] < nbCols and row + neighborhood[2*i+1] < nbRows)
                if(col + neighborhood[2*i] >= 0 and row + neighborhood[2*i+1] >= 0)
                {
                    value += binaries[BINARY_POS(i, cur_label, col, row)];
                }
        }


        min_value = MIN(
            value,
            min_value
        );

        values[cur_label] = value;
        if(use_debug)
        {
            energies_gpu[UNARIES_POS(cur_label, col, row)] = value;
        }
    }


    if(use_debug)
    {
        min_value_gpu[(row) +  (col) * nbRows] = min_value;
    }

    // norm probabilities
    T sum{0};
    for(auto cur_label=0; cur_label<nbLabels; ++cur_label)
    {
        values[cur_label] = exp(-1 *(values[cur_label] - min_value));
        sum += values[cur_label];
    }


    for(auto cur_label=0; cur_label<nbLabels; ++cur_label)
    {
        values[cur_label] /= sum;
        if(use_debug)
        {
            probs_gpu[UNARIES_POS(cur_label, col, row)] = values[cur_label];
        }
    }

    T cur_integral{static_cast<T>(0)};

    for(auto cur_label=0; cur_label<nbLabels; ++cur_label)
    {
        if(cur_integral + values[cur_label] > rand_value)
        {
            result[UNARIES_POS(cur_label, col, row)] += 1;
            next_sample[row + col * nbRows] = cur_label;
            // tha label was set. We set the integral to -100 such that the sum is never again > 0
            cur_integral = static_cast<T>(-100);
        }
        cur_integral += values[cur_label];
    }

#undef UNARIES_POS
#undef BINARY_POS
#undef MIN
#undef MAX

}

template<
    bool use_debug
>
std::vector<at::Tensor>
sampling_forward
(
    at::Tensor const & unaries,
    at::Tensor const & binaries,
    at::Tensor & sample,
    std::vector<int> neighborhood,
    int const sample_steps
)
{
    CHECK_INPUT(unaries);
    CHECK_INPUT(binaries);
    CHECK_INPUT(sample);
    using namespace at; // assumed in the following
    auto result = zeros(torch::CUDA(kInt), unaries.sizes());
    auto next_sample = zeros_like(sample);


    auto batch_size = static_cast<int>(unaries.size(0));
    auto nbLabels = static_cast<int>(unaries.size(1));
    auto cols = static_cast<int>(unaries.size(2));
    auto rows = static_cast<int>(unaries.size(3));

    int* neighborhood_gpu;

    // do some debug stuff
    float * min_value_gpu = nullptr;
    float * energies_gpu = nullptr;
    float * probs_gpu = nullptr;

    if(use_debug)
    {
        gpuErrchk(
            cudaMalloc(
                (void**)&min_value_gpu,
                rows * cols * sizeof(float)
            )
        );
        gpuErrchk(
            cudaMalloc(
                (void**)&energies_gpu,
                rows * cols * nbLabels * sizeof(float)
            )
        );
        gpuErrchk(
            cudaMalloc(
                (void**)&probs_gpu,
                rows * cols * nbLabels * sizeof(float)
            )
        );
    }

    // create random numbers
    float *random_number_gpu, *random_number;
    random_number = new float[rows * cols];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    int nbNeighbors = neighborhood.size() / 2;

    // copy it to device
    gpuErrchk(
        cudaMalloc(
            (void**)&random_number_gpu,
            rows * cols * sizeof(float)
        )
    );
    //


    // copy neighbor to device
    gpuErrchk(
        cudaMalloc(
            (void**)&neighborhood_gpu,
            2 * nbNeighbors * sizeof(int)
        )
    );

    gpuErrchk(
        cudaMemcpy(
            neighborhood_gpu,
            neighborhood.data(),
            2 * nbNeighbors * sizeof(int),
            cudaMemcpyHostToDevice
        )
    );

    // block size
    dim3 blocks(
        (cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
        1
    );
    dim3 blockSize(
        BLOCK_SIZE,
        BLOCK_SIZE,
        1
    );
    for(int i=0; i<sample_steps; ++i)
    {
        // create new random numbers
        for(int i=0; i<rows*cols; ++i)
        {
            random_number[i] = static_cast<float>(dis(gen));
        }
        gpuErrchk(
            cudaMemcpy(
                random_number_gpu,
                random_number,
                rows * cols  * sizeof(float),
                cudaMemcpyHostToDevice
            )
        );
        if( nbLabels == 34)
        {

            forward<float, 34, use_debug><<<blocks, blockSize>>>
            (
                unaries.data<float>(),
                binaries.data<float>(),
                sample.data<int>(),
                next_sample.data<int>(),
                result.data<int>(),
                cols,
                rows,
                neighborhood_gpu,
                nbNeighbors,
                random_number_gpu,
                min_value_gpu,
                energies_gpu,
                probs_gpu
            );
        }
        else if ( nbLabels == 5 )
        {

            forward<float, 5, use_debug><<<blocks, blockSize>>>
            (
                unaries.data<float>(),
                binaries.data<float>(),
                sample.data<int>(),
                next_sample.data<int>(),
                result.data<int>(),
                cols,
                rows,
                neighborhood_gpu,
                nbNeighbors,
                random_number_gpu,
                min_value_gpu,
                energies_gpu,
                probs_gpu
            );

        }
        else if ( nbLabels == 8 )
        {

            forward<float, 8, use_debug><<<blocks, blockSize>>>
            (
                unaries.data<float>(),
                binaries.data<float>(),
                sample.data<int>(),
                next_sample.data<int>(),
                result.data<int>(),
                cols,
                rows,
                neighborhood_gpu,
                nbNeighbors,
                random_number_gpu,
                min_value_gpu,
                energies_gpu,
                probs_gpu
            );

        }
        else
        {
            std::cerr << "Error this number of labels (" << nbLabels << ") is not support. Please add it and recompile" << std::endl;
            exit(1);
        }
        sample = next_sample;

    }

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(
        cudaFree(
            neighborhood_gpu
        )
    );



    if(use_debug)
    {

        auto min_value_tensor = torch::CUDA(kFloat).tensorFromBlob(
            min_value_gpu,
            {cols, rows}
        );
        auto energies_tensor = torch::CUDA(kFloat).tensorFromBlob(
            energies_gpu,
            {nbLabels, cols, rows}
        );
        auto probs_tensor = torch::CUDA(kFloat).tensorFromBlob(
            probs_gpu,
            {nbLabels, cols, rows}
        );
        auto random_number_tensor = torch::CUDA(kFloat).tensorFromBlob(
            random_number_gpu,
            {cols, rows}
        );

        return {
            result,
            sample,
            random_number_tensor,
            min_value_tensor,
            energies_tensor,
            probs_tensor
        };
    }
    else
    {
        gpuErrchk(
            cudaFree(
                random_number_gpu
            )
        );
        // this is the default return. The allocated memory is deleted and the result returned.
        delete random_number;
        return {result, sample};
    }

}


template<
    typename T
>
void
__global__
backward
(
    T * grad_unaries,
    T * grad_binaries,
    int * sample,
    int * label,
    int const nbCols,
    int const nbRows,
    int const * const neighborhood,
    int const nbNeighbors,
    int const nbLabels
)
{
    const int col = threadIdx.x + blockIdx.x * blockDim.x;
    const int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(col >= nbCols)
        return;
    if(row >= nbRows)
        return;

    // calculate the error for the unaries
    /**
    The gradient for the unaries is calculated for each position (row, col) independent. The gradient has the shape
    [nb_classes, nbCols, nbRows]. The sample and the labels have the shapes [nbCols, nbRows].
    So we set
    grad_unaries[ sample[C, R], C, R] += 1
    grad_unaries[ label[C, R], C, R] -= 1
    UNARIES_POS(L, C, R) ( (R) +  (C) * nbRows + (L) * nbRows * nbCols)
    */

    #define POS(SAMPLE, C, R) ( (SAMPLE[(C) * nbRows + (R)] * nbRows * nbCols) \
                    + ((C) * nbRows) \
                    + (R))
    atomicAdd(&(grad_unaries[POS(sample, col, row)]), static_cast<T>(-1));
    atomicAdd(&(grad_unaries[POS(label, col, row)]), static_cast<T>(1));

    #undef POS

    // calculate the error of the binaries
    /**
    The binary gradient is calculated similar to the unary gradient.
    binary shape [nbNeighbors, nbLabels, nbLabels]
    SAMPLE shape [nbCols, nbRows]
    Label shape [nbCols, nbRows]
(sample[((C) + neighborhood[2 *(N)]) * nbRows + (R) + neighborhood[2 *(N) + 1]  ]))
    */

    #define POS(SAMPLE, N, C, R) ( ((N) * nbLabels * nbLabels)  /* the neighbor */ \
        + (SAMPLE[(C) * nbRows + (R)] * nbLabels)  /* the label at the current position */ \
        + (SAMPLE[((C) + neighborhood[2 *(N)]) * nbRows + (R) + neighborhood[2 *(N) + 1] ])) // label at the neighbor position

    for(int n=0; n<nbNeighbors; ++n)
    {
        if (col + neighborhood[2*n] < nbCols and row + neighborhood[2*n+1] < nbRows)
            if(col + neighborhood[2*n] >= 0 and row + neighborhood[2*n+1] >= 0)
            {
                atomicAdd(&(grad_binaries[POS(sample, n, col, row)]), static_cast<T>(-1));
                atomicAdd(&(grad_binaries[POS(label, n, col, row)]), static_cast<T>(1));
            }
    }
    #undef POS
    #undef SAMPLE_POS
}

std::vector<at::Tensor>
sampling_backward
(
    at::Tensor const & unaries,
    at::Tensor const & binaries,
    at::Tensor & sample,
    at::Tensor & labels,
    std::vector<int> neighborhood
)
{

    CHECK_INPUT(unaries);
    CHECK_INPUT(binaries);
    CHECK_INPUT(sample);
    CHECK_INPUT(labels);


    // check that there is only one example
    AT_ASSERT(unaries.size(0) == 1);
    AT_ASSERT(sample.size(0) == 1);
    AT_ASSERT(labels.size(0) == 1);

    // check the number of rows
    AT_ASSERT(unaries.size(2) == sample.size(1));
    AT_ASSERT(unaries.size(2) == labels.size(1));


    // check the number of cols
    AT_ASSERT(unaries.size(3) == sample.size(2));
    AT_ASSERT(unaries.size(3) == labels.size(2));

    // check the number of neighbors
    AT_ASSERT(neighborhood.size()  % 2 == 0);
    AT_ASSERT(binaries.size(0) == neighborhood.size() / 2);


    // get the variables
    auto batch_size = static_cast<int>(unaries.size(0));
    auto nbLabels = static_cast<int>(unaries.size(1));
    auto cols = static_cast<int>(unaries.size(2));
    auto rows = static_cast<int>(unaries.size(3));

    auto unaries_grad = at::zeros_like(unaries);
    auto binaries_grad = at::zeros_like(binaries);
    int* neighborhood_gpu = nullptr;
    int nbNeighbors = neighborhood.size() / 2;
    gpuErrchk(
        cudaMalloc(
            (void**)&neighborhood_gpu,
            2 * nbNeighbors*sizeof(int)
        )
    );

    gpuErrchk(
        cudaMemcpy(
            neighborhood_gpu,
            neighborhood.data(),
            2 * nbNeighbors * sizeof(int),
            cudaMemcpyHostToDevice
        )
    );

    // block size
    dim3 blocks(
        (cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
        1
    );
    dim3 blockSize(
        BLOCK_SIZE,
        BLOCK_SIZE,
        1
    );

    backward<float><<<blocks, blockSize>>>
    (
        unaries_grad.data<float>(),
        binaries_grad.data<float>(),
        sample.data<int>(),
        labels.data<int>(),
        cols,
        rows,
        neighborhood_gpu,
        nbNeighbors,
        nbLabels
    );

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk(
        cudaFree(
            neighborhood_gpu
        )
    );
    // binaries_grad /= binaries_grad.max();
    return {unaries_grad, binaries_grad};

}

// bind it to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sampling_forward<false>, "Sampling forward");
  m.def("forward_debug", &sampling_forward<true>, "Sampling forward");
  m.def("backward", &sampling_backward, "Sampling backward");
}