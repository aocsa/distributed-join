/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "communicator.hpp"
#include "registered_memory_resource.hpp"

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

void set_cuda_device();

/**
 * Size the RMM pool from currently free GPU memory.
 *
 * On discrete GPUs this is ~90% of free memory (or free minus *reserve_bytes*
 * when that argument is set). On unified-memory SoCs (GB10 / DGX Spark) the
 * pool is capped so the CPU and OS keep RAM.
 */
size_t recommended_rmm_pool_size(size_t reserve_bytes = 0);

/**
 * Setup RMM memory pool and communicator.
 *
 * This function will set the current device's memory pool. The memory pool and communicator
 * initialized in this function can be destroyed by *destroy_memory_pool_and_communicator*.
 *
 * @param[out]: communicator Communicator to be constructed.
 * @param[out]: registered_mr If the UCX memory pool needs to be preregistered, this argument holds
 * pointer to the registered memory resource. NIXL registers the pool internally; this is *nullptr*.
 * @param[out]: pool_mr RMM memory resource for memory pool.
 * @param[in]: communicator_name Can be either "NCCL", "NIXL", or "UCX".
 * @param[in]: registration_method If using UCX communicator, this argument can be either "none",
 * "buffer" or "preregistered".
 * @param[in]: communicator_buffer_size If the registration_method is set to "buffer", this argument
 * controls the size of the communication buffer used by the communicator.
 */
void setup_memory_pool_and_communicator(
  Communicator *&communicator,
  registered_memory_resource *&registered_mr,
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> *&pool_mr,
  std::string communicator_name,
  std::string registration_method,
  int64_t communicator_buffer_size);

void destroy_memory_pool_and_communicator(
  Communicator *communicator,
  registered_memory_resource *registered_mr,
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> *pool_mr,
  std::string communicator_name,
  std::string registration_method);
