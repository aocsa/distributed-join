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

#pragma once

#include "error.hpp"

#include <nvcomp.hpp>
#include <nvcomp/cascaded.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>
#include <cuda/std/type_traits>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

enum class CompressionMethod { none, cascaded, lz4 };

/* A structure outlining how to compress a column */
struct ColumnCompressionOptions {
  CompressionMethod compression_method;
  nvcompCascadedFormatOpts cascaded_format;
  std::vector<ColumnCompressionOptions> children_compression_options;

  ColumnCompressionOptions(CompressionMethod compression_method     = CompressionMethod::none,
                           nvcompCascadedFormatOpts cascaded_format = {},
                           std::vector<ColumnCompressionOptions> children_compression_options = {})
    : compression_method(compression_method),
      cascaded_format(cascaded_format),
      children_compression_options(children_compression_options)
  {
  }
};

template <typename T>
using is_cascaded_supported = cuda::std::disjunction<std::is_same<int8_t, T>,
                                                     std::is_same<uint8_t, T>,
                                                     std::is_same<int16_t, T>,
                                                     std::is_same<uint16_t, T>,
                                                     std::is_same<int32_t, T>,
                                                     std::is_same<uint32_t, T>,
                                                     std::is_same<int64_t, T>,
                                                     std::is_same<uint64_t, T>>;

template <typename T>
using is_time_t = cuda::std::disjunction<cudf::is_timestamp_t<T>, cudf::is_duration_t<T>>;

struct compression_functor {
  /**
   * Compress a vector of buffers using cascaded compression.
   *
   * @param[in] uncompressed_data Input buffers to be compressed.
   * @param[in] uncompressed_counts Number of elements to be compressed for each buffer in
   * *uncompressed_data*. Note that in general this is different from the size of the buffer.
   * @param[out] compressed_data Compressed buffers after cascaded compression. This argument does
   * not need to be preallocated.
   * @param[out] compressed_sizes Number of bytes for each buffer in *compressed_data*.
   * @param[in] streams CUDA streams used for the compression kernels.
   */
  template <typename T, std::enable_if_t<is_cascaded_supported<T>::value> * = nullptr>
  void operator()(std::vector<const void *> const &uncompressed_data,
                  std::vector<cudf::size_type> const &uncompressed_counts,
                  std::vector<rmm::device_buffer> &compressed_data,
                  size_t *compressed_sizes,
                  std::vector<rmm::cuda_stream_view> const &streams,
                  nvcompCascadedFormatOpts cascaded_format)
  {
    size_t npartitions = uncompressed_counts.size();
    compressed_data.resize(npartitions);

    nvcompBatchedCascadedOpts_t opts = nvcompBatchedCascadedDefaultOpts;
    opts.type                        = nvcomp::TypeOf<T>();
    opts.num_RLEs                    = cascaded_format.num_RLEs;
    opts.num_deltas                  = cascaded_format.num_deltas;
    opts.use_bp                      = cascaded_format.use_bp;

    // Managers are kept alive across both passes so compression on different streams can overlap.
    std::vector<std::unique_ptr<nvcomp::CascadedManager>> managers(npartitions);

    for (size_t ipartition = 0; ipartition < npartitions; ipartition++) {
      if (uncompressed_counts[ipartition] == 0) {
        compressed_sizes[ipartition] = 0;
        continue;
      }

      managers[ipartition] =
        std::make_unique<nvcomp::CascadedManager>(opts, streams[ipartition].value());

      nvcomp::CompressionConfig config = managers[ipartition]->configure_compression(
        uncompressed_counts[ipartition] * sizeof(T));

      compressed_data[ipartition] =
        rmm::device_buffer(config.max_compressed_buffer_size, streams[ipartition]);

      managers[ipartition]->compress(
        static_cast<const uint8_t *>(uncompressed_data[ipartition]),
        static_cast<uint8_t *>(compressed_data[ipartition].data()),
        config);
    }

    for (size_t ipartition = 0; ipartition < npartitions; ipartition++) {
      if (uncompressed_counts[ipartition] == 0) continue;

      // Synchronizes the corresponding stream.
      compressed_sizes[ipartition] = managers[ipartition]->get_compressed_output_size(
        static_cast<uint8_t *>(compressed_data[ipartition].data()));
    }
  }

  template <typename T, std::enable_if_t<is_time_t<T>::value> * = nullptr>
  void operator()(std::vector<const void *> const &uncompressed_data,
                  std::vector<cudf::size_type> const &uncompressed_counts,
                  std::vector<rmm::device_buffer> &compressed_data,
                  size_t *compressed_sizes,
                  std::vector<rmm::cuda_stream_view> const &streams,
                  nvcompCascadedFormatOpts cascaded_format)
  {
    // If the data type is duration or time, use the corresponding arithmetic type
    operator()<typename T::rep>(uncompressed_data,
                                uncompressed_counts,
                                compressed_data,
                                compressed_sizes,
                                streams,
                                cascaded_format);
  }

  // Checking whether the type is supported if necessary because T might be an incomplete type.
  template <typename T,
            std::enable_if_t<!is_cascaded_supported<T>::value && !is_time_t<T>::value> * = nullptr>
  void operator()(std::vector<const void *> const &uncompressed_data,
                  std::vector<cudf::size_type> const &uncompressed_counts,
                  std::vector<rmm::device_buffer> &compressed_data,
                  size_t *compressed_sizes,
                  std::vector<rmm::cuda_stream_view> const &streams,
                  nvcompCascadedFormatOpts cascaded_format)
  {
    throw std::runtime_error("Unsupported type for cascaded compressor");
  }
};

struct decompression_functor {
  /**
   * Decompress a vector of buffers previously compressed by `compression_functor{}.operator()`.
   *
   * @param[in] compressed_data Vector of input data to be decompressed.
   * @param[in] compressed_sizes Sizes of *compressed_data* in bytes.
   * @param[out] outputs Decompressed outputs. This argument needs to be preallocated.
   * @param[in] expected_output_counts Expected number of elements in the decompressed buffers.
   */
  template <typename T, std::enable_if_t<is_cascaded_supported<T>::value> * = nullptr>
  void operator()(std::vector<const void *> const &compressed_data,
                  std::vector<int64_t> const &compressed_sizes,
                  std::vector<void *> const &outputs,
                  std::vector<int64_t> const &expected_output_counts,
                  std::vector<rmm::cuda_stream_view> const &streams)
  {
    size_t npartitions = compressed_sizes.size();

    // Managers are kept alive until all partitions are issued so decompression on different
    // streams can overlap.
    std::vector<std::shared_ptr<nvcomp::nvcompManagerBase>> managers(npartitions);

    for (size_t ipartition = 0; ipartition < npartitions; ipartition++) {
      if (expected_output_counts[ipartition] == 0) continue;

      const uint8_t *compressed_buffer = static_cast<const uint8_t *>(compressed_data[ipartition]);

      managers[ipartition] = nvcomp::create_manager(compressed_buffer, streams[ipartition].value());

      nvcomp::DecompressionConfig config =
        managers[ipartition]->configure_decompression(compressed_buffer);

      assert(config.decomp_data_size == expected_output_counts[ipartition] * sizeof(T));

      managers[ipartition]->decompress(
        static_cast<uint8_t *>(outputs[ipartition]), compressed_buffer, config);
    }
  }

  template <typename T, std::enable_if_t<is_time_t<T>::value> * = nullptr>
  void operator()(std::vector<const void *> const &compressed_data,
                  std::vector<int64_t> const &compressed_sizes,
                  std::vector<void *> const &outputs,
                  std::vector<int64_t> const &expected_output_counts,
                  std::vector<rmm::cuda_stream_view> const &streams)
  {
    // If the data type is duration or time, use the corresponding arithmetic type
    operator()<typename T::rep>(
      compressed_data, compressed_sizes, outputs, expected_output_counts, streams);
  }

  // Checking whether the type is supported if necessary because T might be an incomplete type.
  template <typename T,
            std::enable_if_t<!is_cascaded_supported<T>::value && !is_time_t<T>::value> * = nullptr>
  void operator()(std::vector<const void *> const &compressed_data,
                  std::vector<int64_t> const &compressed_sizes,
                  std::vector<void *> const &outputs,
                  std::vector<int64_t> const &expected_output_counts,
                  std::vector<rmm::cuda_stream_view> const &streams)
  {
    throw std::runtime_error("Unsupported type for cascaded decompressor");
  }
};

struct cascaded_selector_functor {
  /**
   * Generate cascaded compression configuration options using auto-selector.
   *
   * @param[in] uncompressed_data Data used by auto-selector.
   * @param[in] byte_len Number of bytes in *uncompressed_data*.
   *
   * @returns Cascaded compression configuration options for *uncompressed_data*.
   */
  template <typename T, std::enable_if_t<is_cascaded_supported<T>::value> * = nullptr>
  nvcompCascadedFormatOpts operator()(const void *uncompressed_data, size_t byte_len)
  {
    // nvcomp >= 2.3 removed CascadedSelector, so the format is fixed instead of sampled.
    return nvcompCascadedFormatOpts{.num_RLEs = 1, .num_deltas = 1, .use_bp = 1};
  }

  template <typename T, std::enable_if_t<is_time_t<T>::value> * = nullptr>
  nvcompCascadedFormatOpts operator()(const void *uncompressed_data, size_t byte_len)
  {
    // If the data type is duration or time, use the corresponding arithmetic type
    return operator()<typename T::rep>(uncompressed_data, byte_len);
  }

  template <typename T,
            std::enable_if_t<!is_cascaded_supported<T>::value && !is_time_t<T>::value> * = nullptr>
  nvcompCascadedFormatOpts operator()(const void *uncompressed_data, size_t byte_len)
  {
    throw std::runtime_error("Unsupported type for CascadedSelector");
    return nvcompCascadedFormatOpts();
  }
};

/**
 * Generate compression options using auto selector.
 *
 * @param[in] input_table Table for which to generate compression options.
 *
 * @returns Vector of length equal to number of columns in *input_table*, where each element
 * representing the compression options for each column.
 */
std::vector<ColumnCompressionOptions> generate_auto_select_compression_options(
  cudf::table_view input_table);

/**
 * Generate compression options that no compression should be performed.
 *
 * @param[in] input_table Table for which to generate compression options.
 *
 * @returns Vector of length equal to number of columns in *input_table*, where each element
 * representing the compression options for each column.
 */
std::vector<ColumnCompressionOptions> generate_none_compression_options(
  cudf::table_view input_table);

/**
 * Broadcast the compression options of a column from the root rank to all ranks.
 *
 * Note: This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] input_column Column which *input_options* is associated with. This argument is
 * significant on all ranks.
 * @param[in] input_options Compression options associated with *input_column* that needs to be
 * broadcasted. This argument is only significant on the root rank.
 *
 * @returns Broadcasted compression options on all ranks.
 */
ColumnCompressionOptions broadcast_compression_options(cudf::column_view input_column,
                                                       ColumnCompressionOptions input_options);

/**
 * Broadcast the compression options of a table from the root rank to all ranks.
 *
 * Note: This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] input_table Table which *input_options* is associated with. This argument is
 * significant on all ranks.
 * @param[in] input_options Vector of lenght equal to the number of columns in *input_table*,
 * representing compression options associated with *input_table* that needs to be
 * broadcasted. Each element represents the compression option of one column in *input_table*. This
 * argument is only significant on the root rank.
 *
 * @returns Broadcasted compression options on all ranks.
 */
std::vector<ColumnCompressionOptions> broadcast_compression_options(
  cudf::table_view input_table, std::vector<ColumnCompressionOptions> input_options);

/**
 * Generate the same compression option on all ranks.
 *
 * Note: This function needs to be called collectively by all ranks in MPI_COMM_WORLD.
 *
 * @param[in] input_table Table to generate compression options on. This argument is significant on
 * all ranks.
 * @param[in] compression Whether to use compression. If *true*, the compression options will be
 * generated by auto selector on the root rank. If *false*, compression options indicating no
 * compression will be generated.
 *
 * @returns Compression options for *input_table* on all ranks.
 */
std::vector<ColumnCompressionOptions> generate_compression_options_distributed(
  cudf::table_view input_table, bool compression);

/**
 * This helper function runs compression and decompression on a small buffer to avoid nvcomp's
 * setup time during the actual run.
 */
void warmup_nvcomp();
