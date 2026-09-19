/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

/* This file is adapted from "rmm/mr/cuda_memory_resource.hpp" */

#pragma once

#include "communicator.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime.h>

#include <map>

/**
 * @brief `device_memory_resource` derived class that uses cudaMalloc/Free for
 * allocation/deallocation, and register through UCX at allocation time.
 */
class registered_memory_resource final : public rmm::mr::device_memory_resource {
 public:
  registered_memory_resource(UCXCommunicator* communicator) { this->communicator = communicator; }

  ~registered_memory_resource() override                        = default;
  registered_memory_resource(registered_memory_resource const&) = default;
  registered_memory_resource(registered_memory_resource&&)      = default;
  registered_memory_resource& operator=(registered_memory_resource const&) = default;
  registered_memory_resource& operator=(registered_memory_resource&&) = default;

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view) override
  {
    void* p{nullptr};
    RMM_CUDA_TRY_ALLOC(cudaMalloc(&p, bytes), bytes);
    ucp_mem_h memory_handle;
    communicator->register_buffer(p, bytes, &memory_handle);
    registered_handles[p] = memory_handle;
    return p;
  }

  void do_deallocate(void* p, std::size_t, rmm::cuda_stream_view) noexcept override
  {
    ucp_mem_h memory_handle = registered_handles.find(p)->second;
    communicator->deregister_buffer(memory_handle);
    RMM_ASSERT_CUDA_SUCCESS(cudaFree(p));
  }

  bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<registered_memory_resource const*>(&other) != nullptr;
  }

  UCXCommunicator* communicator;
  std::map<void*, ucp_mem_h> registered_handles;
};
