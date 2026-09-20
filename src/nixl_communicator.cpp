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

#include "nixl_communicator.hpp"

#include "error.hpp"
#include "nixl_registered_memory_resource.hpp"

#include <rmm/mr/device_memory_resource.hpp>

#include <mpi.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#define NIXL_CALL(call)                                                                          \
  {                                                                                              \
    nixl_status_t status = (call);                                                               \
    if (status < NIXL_SUCCESS) {                                                                 \
      fprintf(stderr,                                                                            \
              "ERROR: nixl call \"%s\" in line %d of file %s failed with %s (%d).\n",            \
              #call,                                                                             \
              __LINE__,                                                                          \
              __FILE__,                                                                          \
              nixlEnumStrings::statusStr(status).c_str(),                                        \
              static_cast<int>(status));                                                         \
      exit(1);                                                                                   \
    }                                                                                            \
  }

std::string NIXLCommunicator::agent_name(int rank) { return "rank" + std::to_string(rank); }

NIXLCommunicator::NIXLCommunicator()  = default;
NIXLCommunicator::~NIXLCommunicator() = default;

void NIXLCommunicator::exchange_metadata()
{
  std::string local_md;
  NIXL_CALL(agent->getLocalMD(local_md));

  int local_len = static_cast<int>(local_md.size());
  std::vector<int> lens(mpi_size);
  MPI_CALL(MPI_Allgather(&local_len, 1, MPI_INT, lens.data(), 1, MPI_INT, MPI_COMM_WORLD));

  std::vector<int> displs(mpi_size);
  int total = 0;
  for (int irank = 0; irank < mpi_size; irank++) {
    displs[irank] = total;
    total += lens[irank];
  }

  std::vector<char> gathered(static_cast<std::size_t>(total));
  MPI_CALL(MPI_Allgatherv(local_md.data(),
                          local_len,
                          MPI_BYTE,
                          gathered.data(),
                          lens.data(),
                          displs.data(),
                          MPI_BYTE,
                          MPI_COMM_WORLD));

  remote_names.assign(mpi_size, std::string{});
  remote_names[mpi_rank] = agent_name(mpi_rank);
  for (int irank = 0; irank < mpi_size; irank++) {
    if (irank == mpi_rank) continue;
    std::string blob(gathered.data() + displs[irank],
                     gathered.data() + displs[irank] + lens[irank]);
    std::string name;
    NIXL_CALL(agent->loadRemoteMD(blob, name));
    remote_names[irank] = std::move(name);
  }
}

void NIXLCommunicator::make_connections()
{
  for (int irank = 0; irank < mpi_size; irank++) {
    if (irank == mpi_rank) continue;
    NIXL_CALL(agent->makeConnection(remote_names[irank], &extra_params));
  }
}

void NIXLCommunicator::fail_xfer(char const *what, nixl_status_t status)
{
  fprintf(stderr,
          "ERROR: nixl %s in line %d of file %s failed with %s (%d).\n",
          what,
          __LINE__,
          __FILE__,
          nixlEnumStrings::statusStr(status).c_str(),
          static_cast<int>(status));
  exit(1);
}

void NIXLCommunicator::wait_posted(std::vector<nixlXferReqH *> const &reqs)
{
  std::vector<nixl_status_t> status(reqs.size(), NIXL_IN_PROG);
  for (std::size_t i = 0; i < reqs.size(); i++) {
    status[i] = agent->postXferReq(reqs[i]);
    if (status[i] < NIXL_SUCCESS) { fail_xfer("postXferReq", status[i]); }
  }

  bool in_progress = true;
  while (in_progress) {
    in_progress = false;
    for (std::size_t i = 0; i < reqs.size(); i++) {
      if (status[i] != NIXL_IN_PROG) continue;
      status[i] = agent->getXferStatus(reqs[i]);
      if (status[i] < NIXL_SUCCESS) { fail_xfer("getXferStatus", status[i]); }
      if (status[i] == NIXL_IN_PROG) { in_progress = true; }
    }
  }

  for (nixlXferReqH *req : reqs) { NIXL_CALL(agent->releaseXferReq(req)); }
}

void NIXLCommunicator::register_range(void *buf, std::size_t bytes)
{
  nixl_reg_dlist_t dlist(mem_seg);
  dlist.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(buf), bytes, seg_dev_id()));
  NIXL_CALL(agent->registerMem(dlist, &extra_params));
  registered_ranges[buf] = bytes;
}

void NIXLCommunicator::deregister_range(void *buf, std::size_t bytes)
{
  auto it = registered_ranges.find(buf);
  std::size_t len = bytes;
  if (it != registered_ranges.end()) {
    len = it->second;
    registered_ranges.erase(it);
  }
  nixl_reg_dlist_t dlist(mem_seg);
  dlist.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(buf), len, seg_dev_id()));
  NIXL_CALL(agent->deregisterMem(dlist, &extra_params));
}

void NIXLCommunicator::publish_registered_memory()
{
  exchange_metadata();
  make_connections();
}

rmm::mr::device_memory_resource &NIXLCommunicator::memory_resource() { return *registered_mr; }

void NIXLCommunicator::initialize()
{
  Communicator::initialize();

  if (char const *env = std::getenv("NIXL_HOST_POOL"); env && env[0] == '1') {
    host_pool = true;
    mem_seg   = DRAM_SEG;
  }
  if (mpi_rank == 0) { std::cout << "NIXL pool memory: " << (host_pool ? "host" : "vram") << std::endl; }

  nixlAgentConfig config(/*use_prog_thread=*/true);
  agent = std::make_unique<nixlAgent>(agent_name(mpi_rank), config);

  nixl_mem_list_t mems;
  nixl_b_params_t params;
  NIXL_CALL(agent->getPluginParams("UCX", mems, params));
  NIXL_CALL(agent->createBackend("UCX", params, backend));
  extra_params.backends = {backend};
  NIXL_CALL(agent->getBackendParams(backend, mems, params));
  bool has_vram = false;
  for (nixl_mem_t mem : mems) {
    if (mem == VRAM_SEG) has_vram = true;
  }
  if (!has_vram) { throw std::runtime_error("NIXL UCX plugin does not advertise VRAM_SEG"); }

  CUDA_RT_CALL(cudaStreamCreate(&comm_stream));
  registered_mr = std::make_unique<nixl_registered_memory_resource>(this);
}

void NIXLCommunicator::start()
{
  pending_sends.clear();
  pending_recvs.clear();
}

void NIXLCommunicator::send(const void *buf, int64_t count, int element_size, int dest)
{
  pending_sends.push_back(
    PendingSend{dest, buf, static_cast<std::size_t>(count) * static_cast<std::size_t>(element_size)});
}

void NIXLCommunicator::recv(void *buf, int64_t count, int element_size, int source)
{
  pending_recvs.push_back(PendingRecv{
    source, buf, static_cast<std::size_t>(count) * static_cast<std::size_t>(element_size)});
}

void NIXLCommunicator::stop()
{
  CUDA_RT_CALL(cudaStreamSynchronize(cudaStreamDefault));

  std::vector<int> sendcounts(static_cast<std::size_t>(mpi_size), 0);
  for (const PendingRecv &recv : pending_recvs) { sendcounts[recv.source] += 1; }

  std::vector<int> recvcounts(static_cast<std::size_t>(mpi_size));
  MPI_CALL(MPI_Alltoall(
    sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD));

  std::vector<int> sdispls(static_cast<std::size_t>(mpi_size), 0);
  std::vector<int> rdispls(static_cast<std::size_t>(mpi_size), 0);
  int send_total = 0;
  int recv_total = 0;
  for (int irank = 0; irank < mpi_size; irank++) {
    sdispls[irank] = send_total;
    rdispls[irank] = recv_total;
    send_total += sendcounts[irank];
    recv_total += recvcounts[irank];
  }

  std::vector<int> cursor = sdispls;
  std::vector<RemoteDest> outgoing(static_cast<std::size_t>(send_total));
  for (const PendingRecv &recv : pending_recvs) {
    int const slot = cursor[recv.source]++;
    outgoing[static_cast<std::size_t>(slot)] = RemoteDest{
      reinterpret_cast<std::uint64_t>(recv.ptr),
      recv.nbytes,
      seg_dev_id(),
    };
  }

  std::vector<RemoteDest> remotes(static_cast<std::size_t>(recv_total));
  std::vector<int> sendbytes(static_cast<std::size_t>(mpi_size));
  std::vector<int> recvbytes(static_cast<std::size_t>(mpi_size));
  std::vector<int> sdispl_bytes(static_cast<std::size_t>(mpi_size));
  std::vector<int> rdispl_bytes(static_cast<std::size_t>(mpi_size));
  int const dest_bytes = static_cast<int>(sizeof(RemoteDest));
  for (int irank = 0; irank < mpi_size; irank++) {
    sendbytes[irank]    = sendcounts[irank] * dest_bytes;
    recvbytes[irank]    = recvcounts[irank] * dest_bytes;
    sdispl_bytes[irank] = sdispls[irank] * dest_bytes;
    rdispl_bytes[irank] = rdispls[irank] * dest_bytes;
  }
  MPI_CALL(MPI_Alltoallv(outgoing.data(),
                         sendbytes.data(),
                         sdispl_bytes.data(),
                         MPI_BYTE,
                         remotes.data(),
                         recvbytes.data(),
                         rdispl_bytes.data(),
                         MPI_BYTE,
                         MPI_COMM_WORLD));

  std::vector<std::vector<PendingSend>> sends_for(static_cast<std::size_t>(mpi_size));
  for (const PendingSend &send : pending_sends) {
    sends_for[static_cast<std::size_t>(send.dest)].push_back(send);
  }

  // One-sided WRITE completes locally when the initiator's flush returns; the receiver has no
  // way to know its buffers were filled without a notification from every sender.
  nixl_opt_args_t xfer_params = extra_params;
  xfer_params.hasNotif        = true;
  xfer_params.notifMsg        = "stop";

  std::vector<nixlXferReqH *> reqs;
  for (int dest = 0; dest < mpi_size; dest++) {
    auto const &sends = sends_for[static_cast<std::size_t>(dest)];
    if (sends.empty()) continue;
    if (recvcounts[dest] != static_cast<int>(sends.size())) {
      fprintf(stderr,
              "ERROR: NIXL dest count mismatch: rank %d -> %d local_sends=%zu remote_recvs=%d\n",
              mpi_rank,
              dest,
              sends.size(),
              recvcounts[dest]);
      exit(1);
    }

    if (dest == mpi_rank) {
      for (std::size_t i = 0; i < sends.size(); i++) {
        RemoteDest const &remote = remotes[static_cast<std::size_t>(rdispls[dest]) + i];
        CUDA_RT_CALL(cudaMemcpyAsync(reinterpret_cast<void *>(remote.addr),
                                     sends[i].ptr,
                                     sends[i].nbytes,
                                     cudaMemcpyDefault,
                                     comm_stream));
      }
      continue;
    }

    nixl_xfer_dlist_t src(mem_seg);
    nixl_xfer_dlist_t dst(mem_seg);
    for (std::size_t i = 0; i < sends.size(); i++) {
      RemoteDest const &remote = remotes[static_cast<std::size_t>(rdispls[dest]) + i];
      if (remote.addr == 0 || remote.len != sends[i].nbytes) {
        fprintf(stderr,
                "ERROR: NIXL dest exchange mismatch: rank %d -> %d addr=%llu len=%llu expected=%zu\n",
                mpi_rank,
                dest,
                static_cast<unsigned long long>(remote.addr),
                static_cast<unsigned long long>(remote.len),
                sends[i].nbytes);
        exit(1);
      }
      src.addDesc(
        nixlBasicDesc(reinterpret_cast<uintptr_t>(sends[i].ptr), sends[i].nbytes, seg_dev_id()));
      dst.addDesc(nixlBasicDesc(static_cast<uintptr_t>(remote.addr), remote.len, remote.dev_id));
    }

    nixlXferReqH *req = nullptr;
    NIXL_CALL(
      agent->createXferReq(NIXL_WRITE, src, dst, remote_names[dest], req, &xfer_params));
    reqs.push_back(req);
  }

  wait_posted(reqs);

  std::vector<int> awaiting;
  for (int source = 0; source < mpi_size; source++) {
    if (source != mpi_rank && sendcounts[source] > 0) { awaiting.push_back(source); }
  }
  while (!awaiting.empty()) {
    nixl_notifs_t notifs;
    nixl_status_t const status = agent->getNotifs(notifs, &extra_params);
    if (status < NIXL_SUCCESS) { fail_xfer("getNotifs", status); }
    for (auto const &[remote, msgs] : notifs) { notif_credits[remote] += msgs.size(); }
    for (auto it = awaiting.begin(); it != awaiting.end();) {
      auto credit = notif_credits.find(remote_names[*it]);
      if (credit != notif_credits.end() && credit->second > 0) {
        credit->second -= 1;
        it = awaiting.erase(it);
      } else {
        ++it;
      }
    }
  }

  CUDA_RT_CALL(cudaStreamSynchronize(comm_stream));
  pending_sends.clear();
  pending_recvs.clear();
}

void NIXLCommunicator::finalize()
{
  registered_mr.reset();
  if (agent) {
    for (int irank = 0; irank < mpi_size; irank++) {
      if (irank == mpi_rank) continue;
      if (irank < static_cast<int>(remote_names.size()) && !remote_names[irank].empty()) {
        agent->invalidateRemoteMD(remote_names[irank]);
      }
    }
    agent.reset();
    backend = nullptr;
  }
  if (comm_stream) {
    CUDA_RT_CALL(cudaStreamDestroy(comm_stream));
    comm_stream = nullptr;
  }
}
