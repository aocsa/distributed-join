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

#include "compression.hpp"

#include "error.hpp"

#include <nvcomp.hpp>
#include <nvcomp/cascaded.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <mpi.h>

#include <cstdint>
#include <vector>

std::vector<ColumnCompressionOptions> generate_auto_select_compression_options(
  cudf::table_view input_table)
{
  std::vector<ColumnCompressionOptions> compression_options;

  for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
    cudf::column_view input_column = input_table.column(icol);
    cudf::data_type dtype          = input_column.type();
    if (dtype.id() == cudf::type_id::STRING) {
      std::vector<ColumnCompressionOptions> children_options;

      // offset subcolumn
      cudf::data_type offset_dtype = input_column.child(0).type();
      CascadedFormatOpts offset_cascaded_opts =
        cudf::type_dispatcher(offset_dtype,
                              cascaded_selector_functor{},
                              input_column.child(0).head(),
                              input_column.child(0).size() * cudf::size_of(offset_dtype));
      children_options.emplace_back(CompressionMethod::cascaded, offset_cascaded_opts);

      // do not compress char subcolumn
      children_options.emplace_back(CompressionMethod::none);

      compression_options.emplace_back(
        CompressionMethod::none, CascadedFormatOpts(), children_options);
    } else {
      CascadedFormatOpts column_cascaded_opts =
        cudf::type_dispatcher(dtype,
                              cascaded_selector_functor{},
                              input_column.head(),
                              input_column.size() * cudf::size_of(dtype));

      compression_options.emplace_back(CompressionMethod::cascaded, column_cascaded_opts);
    }
  }

  return compression_options;
}

std::vector<ColumnCompressionOptions> generate_none_compression_options(
  cudf::table_view input_table)
{
  std::vector<ColumnCompressionOptions> compression_options;

  for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
    if (input_table.column(icol).type().id() == cudf::type_id::STRING) {
      std::vector<ColumnCompressionOptions> children_options;
      // offset subcolumn
      children_options.emplace_back(CompressionMethod::none);
      // char subcolumn
      children_options.emplace_back(CompressionMethod::none);
      compression_options.emplace_back(
        CompressionMethod::none, CascadedFormatOpts(), children_options);
    } else {
      compression_options.emplace_back(CompressionMethod::none);
    }
  }

  return compression_options;
}

ColumnCompressionOptions broadcast_compression_options(cudf::column_view input_column,
                                                       ColumnCompressionOptions input_options)
{
  int mpi_rank;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

  cudf::data_type dtype = input_column.type();

  CompressionMethod compression_method     = input_options.compression_method;
  CascadedFormatOpts cascaded_format = input_options.cascaded_format;
  std::vector<ColumnCompressionOptions> children_compression_options;

  MPI_CALL(MPI_Bcast(&compression_method, sizeof(compression_method), MPI_CHAR, 0, MPI_COMM_WORLD));
  MPI_CALL(MPI_Bcast(&cascaded_format, sizeof(cascaded_format), MPI_CHAR, 0, MPI_COMM_WORLD));

  if (dtype.id() == cudf::type_id::STRING) {
    if (mpi_rank == 0) {
      // Offsets still live in child(0); chars live in the parent data buffer.
      assert(input_options.children_compression_options.size() == 2);
    }

    ColumnCompressionOptions offsets_options;
    if (mpi_rank == 0) { offsets_options = input_options.children_compression_options[0]; }
    children_compression_options.push_back(
      broadcast_compression_options(input_column.child(0), offsets_options));

    CompressionMethod chars_method{};
    CascadedFormatOpts chars_format{};
    if (mpi_rank == 0) {
      chars_method = input_options.children_compression_options[1].compression_method;
      chars_format = input_options.children_compression_options[1].cascaded_format;
    }
    MPI_CALL(MPI_Bcast(&chars_method, sizeof(chars_method), MPI_CHAR, 0, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&chars_format, sizeof(chars_format), MPI_CHAR, 0, MPI_COMM_WORLD));
    children_compression_options.emplace_back(chars_method, chars_format);
  }

  return ColumnCompressionOptions(
    compression_method, cascaded_format, children_compression_options);
}

std::vector<ColumnCompressionOptions> broadcast_compression_options(
  cudf::table_view input_table, std::vector<ColumnCompressionOptions> input_options)
{
  int mpi_rank;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

  std::vector<ColumnCompressionOptions> output_options;
  output_options.reserve(input_table.num_columns());

  for (cudf::size_type icol = 0; icol < input_table.num_columns(); icol++) {
    ColumnCompressionOptions input_options_icol;
    if (mpi_rank == 0) { input_options_icol = input_options[icol]; }

    output_options.push_back(
      broadcast_compression_options(input_table.column(icol), input_options_icol));
  }

  return output_options;
}

std::vector<ColumnCompressionOptions> generate_compression_options_distributed(
  cudf::table_view input_table, bool compression)
{
  if (!compression) { return generate_none_compression_options(input_table); }

  int mpi_rank;
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

  std::vector<ColumnCompressionOptions> compression_options;
  if (mpi_rank == 0) {
    compression_options = generate_auto_select_compression_options(input_table);
  }

  compression_options = broadcast_compression_options(input_table, compression_options);

  return compression_options;
}

void warmup_nvcomp()
{
  using T = int;

  constexpr size_t warmup_size = 1000;
  rmm::device_buffer input_data(warmup_size * sizeof(T), rmm::cuda_stream_default);

  std::vector<rmm::device_buffer> compressed_data(1);
  size_t compressed_size;

  CascadedFormatOpts cascaded_format = {.num_RLEs = 1, .num_deltas = 1, .use_bp = 1};

  compression_functor{}.operator()<T>({input_data.data()},
                                      {warmup_size},
                                      compressed_data,
                                      &compressed_size,
                                      {rmm::cuda_stream_default},
                                      cascaded_format);

  rmm::device_buffer decompressed_data(warmup_size * sizeof(T), rmm::cuda_stream_default);

  decompression_functor{}.operator()<T>({compressed_data[0].data()},
                                        {static_cast<int64_t>(compressed_size)},
                                        {decompressed_data.data()},
                                        {warmup_size},
                                        {rmm::cuda_stream_default});
}
