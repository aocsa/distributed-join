/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "communicator.hpp"

#include <nixl.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rmm {
namespace mr {
class device_memory_resource;
}
}  // namespace rmm

class nixl_registered_memory_resource;

/**
 * GPU payload communicator that uses NIXL one-sided WRITE over the UCX backend.
 * MPI remains the process launcher and the metadata / dest-address side channel.
 *
 * The RMM pool backing store is registered once (same order as UCX preregistered).
 * send/recv WRITE the user pointers directly; stop() only exchanges dest VAs and posts.
 */
class NIXLCommunicator : public Communicator {
 public:
  NIXLCommunicator();
  NIXLCommunicator(NIXLCommunicator const &)            = delete;
  NIXLCommunicator &operator=(NIXLCommunicator const &) = delete;
  ~NIXLCommunicator() override;

  void initialize() override;
  void start() override;
  void stop() override;
  void send(const void *buf, int64_t count, int element_size, int dest) override;
  void recv(void *buf, int64_t count, int element_size, int source) override;
  void finalize() override;
  bool group_by_batch() override { return false; }

  void register_range(void *buf, std::size_t bytes);
  void deregister_range(void *buf, std::size_t bytes);
  void publish_registered_memory();
  rmm::mr::device_memory_resource &memory_resource();

  cudaStream_t comm_stream{};

 private:
  struct PendingSend {
    int dest;
    const void *ptr;
    std::size_t nbytes;
  };

  struct PendingRecv {
    int source;
    void *ptr;
    std::size_t nbytes;
  };

  struct RemoteDest {
    std::uint64_t addr;
    std::uint64_t len;
    std::uint64_t dev_id;
  };

  static std::string agent_name(int rank);
  void exchange_metadata();
  void make_connections();
  void fail_xfer(char const *what, nixl_status_t status);
  void wait_posted(std::vector<nixlXferReqH *> const &reqs);

  std::unique_ptr<nixlAgent> agent;
  std::unique_ptr<nixl_registered_memory_resource> registered_mr;
  nixlBackendH *backend{nullptr};
  nixl_opt_args_t extra_params{};
  std::vector<std::string> remote_names;
  std::vector<PendingSend> pending_sends;
  std::vector<PendingRecv> pending_recvs;
  std::map<void *, std::size_t> registered_ranges;
  std::map<std::string, std::size_t> notif_credits;
  // GB10 has no GPUDirect RDMA / dmabuf, so the NIC cannot register cudaMalloc memory and UCX
  // RMA degrades to 8 KiB bounce copies. NIXL_HOST_POOL=1 backs the pool with pinned host
  // memory (the GPU is integrated and reads it directly) registered as DRAM_SEG.
  nixl_mem_t mem_seg{VRAM_SEG};
  std::uint64_t seg_dev_id() const { return host_pool ? 0 : static_cast<std::uint64_t>(current_device); }

 public:
  bool host_pool{false};
};
