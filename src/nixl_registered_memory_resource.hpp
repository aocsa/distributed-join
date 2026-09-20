/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
 * See the License for the specific language governing permissions
 * and limitations under the License.
 */

#pragma once

#include "nixl_communicator.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime.h>

/**
 * cudaMalloc/Free resource that registers each block with NIXL at allocate time.
 * Used as the upstream of the RMM pool so the pool backing store is registered once.
 */
class nixl_registered_memory_resource final : public rmm::mr::device_memory_resource {
 public:
  explicit nixl_registered_memory_resource(NIXLCommunicator *communicator)
    : communicator(communicator)
  {
  }

  ~nixl_registered_memory_resource() override                             = default;
  nixl_registered_memory_resource(nixl_registered_memory_resource const &) = default;
  nixl_registered_memory_resource(nixl_registered_memory_resource &&)      = default;
  nixl_registered_memory_resource &operator=(nixl_registered_memory_resource const &) = default;
  nixl_registered_memory_resource &operator=(nixl_registered_memory_resource &&) = default;

 private:
  void *do_allocate(std::size_t bytes, rmm::cuda_stream_view) override
  {
    void *p{nullptr};
    if (communicator->host_pool) {
      RMM_CUDA_TRY_ALLOC(cudaMallocHost(&p, bytes), bytes);
    } else {
      RMM_CUDA_TRY_ALLOC(cudaMalloc(&p, bytes), bytes);
    }
    communicator->register_range(p, bytes);
    return p;
  }

  void do_deallocate(void *p, std::size_t bytes, rmm::cuda_stream_view) noexcept override
  {
    communicator->deregister_range(p, bytes);
    RMM_ASSERT_CUDA_SUCCESS(communicator->host_pool ? cudaFreeHost(p) : cudaFree(p));
  }

  bool do_is_equal(device_memory_resource const &other) const noexcept override
  {
    return dynamic_cast<nixl_registered_memory_resource const *>(&other) != nullptr;
  }

  NIXLCommunicator *communicator;
};
