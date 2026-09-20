/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Test the correctness of shuffle_on implementation. This test has the following steps:
 *
 * 1. Each GPU independently generate a table with a single key column, filled with random integers.
 * 2. The generated table is shuffled across GPUs, using identity hash function.
 * 3. Each GPU verifies the shuffled keys has the same remainders modulo the number of MPI ranks.
 */

#include "../src/communicator.hpp"
#include "../src/error.hpp"
#include "../src/setup.hpp"
#include "../src/shuffle_on.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda_runtime.h>

#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

using cudf::column;
using cudf::table;

std::unique_ptr<table> generate_table(cudf::size_type size)
{
  std::vector<std::unique_ptr<column>> columns;

  auto key_column = cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT32), size);
  std::vector<int> keys(size);
  for (int &key : keys) { key = rand() % (size * 10); }
  CUDA_RT_CALL(cudaMemcpy(key_column->mutable_view().head<int>(),
                          keys.data(),
                          size * sizeof(int),
                          cudaMemcpyDefault));

  columns.push_back(std::move(key_column));

  return std::make_unique<table>(std::move(columns));
}

void run_test(int nrows_per_gpu, bool compression, Communicator *communicator)
{
  auto input_table = generate_table(nrows_per_gpu);

  auto compression_options =
    generate_compression_options_distributed(input_table->view(), compression);

  CUDA_RT_CALL(cudaDeviceSynchronize());
  MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
  auto start = std::chrono::high_resolution_clock::now();

  std::unique_ptr<cudf::table> output_table = shuffle_on(
    input_table->view(), {0}, communicator, compression_options, cudf::hash_id::HASH_IDENTITY);

  CUDA_RT_CALL(cudaDeviceSynchronize());
  MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
  auto stop = std::chrono::high_resolution_clock::now();

  if (output_table->view().num_columns() != 1) {
    std::cerr << "FAIL: expected 1 column\n";
    exit(1);
  }
  cudf::size_type num_rows_shuffled = output_table->view().column(0).size();
  std::vector<int> keys(num_rows_shuffled);
  if (num_rows_shuffled != 0) {
    CUDA_RT_CALL(cudaMemcpy(keys.data(),
                            output_table->view().column(0).head<int>(),
                            num_rows_shuffled * sizeof(int),
                            cudaMemcpyDefault));
    int const mod_result = keys[0] % communicator->mpi_size;
    for (int key : keys) {
      if (key % communicator->mpi_size != mod_result) {
        std::cerr << "FAIL: rank " << communicator->mpi_rank << " received key " << key
                  << " with hash " << key % communicator->mpi_size << ", expected " << mod_result
                  << "\n";
        exit(1);
      }
    }
  }

  if (communicator->mpi_rank == 0) {
    std::cerr << std::boolalpha;
    std::cerr << "Test case (" << nrows_per_gpu << "," << compression << ") passes successfully.\n";
    std::cerr << "Shuffle time (s) " << std::chrono::duration<double>(stop - start).count()
              << " compression=" << compression << "\n";
  }
}

int main(int argc, char *argv[])
{
  MPI_CALL(MPI_Init(&argc, &argv));
  set_cuda_device();

  std::string communicator_name = "UCX";
  int nrows_per_gpu             = 1'000'000;
  for (int iarg = 0; iarg + 1 < argc; iarg++) {
    if (!strcmp(argv[iarg], "--communicator")) { communicator_name = argv[iarg + 1]; }
    if (!strcmp(argv[iarg], "--nrows")) { nrows_per_gpu = atoi(argv[iarg + 1]); }
  }

  Communicator *communicator{nullptr};
  registered_memory_resource *registered_mr{nullptr};
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> *pool_mr{nullptr};
  setup_memory_pool_and_communicator(
    communicator, registered_mr, pool_mr, communicator_name, "preregistered", 0);
  if (communicator->mpi_rank == 0) { std::cerr << "Communicator: " << communicator_name << "\n"; }

  run_test(nrows_per_gpu, false, communicator);  // warmup: first-touch pool + connections
  run_test(nrows_per_gpu, false, communicator);
  run_test(nrows_per_gpu, true, communicator);

  destroy_memory_pool_and_communicator(
    communicator, registered_mr, pool_mr, communicator_name, "preregistered");
  MPI_CALL(MPI_Finalize());

  return 0;
}
