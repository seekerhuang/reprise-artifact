 

#include <ATen/cuda/CUDAContextLight.h>
#include <c10/core/Allocator.h>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {
thread_local std::string error;
using Key = std::tuple<void*, void*>;
using Owner = std::shared_ptr<c10::DataPtr>;
void drop_owner(void* value) { delete static_cast<Owner*>(value); }
c10::DataPtr alias(const Owner& owner) {
  return c10::DataPtr(owner->get(), new Owner(owner), drop_owner, owner->device());
}
int bind(at::cuda::WorkspaceMapWithMutex& table, Key source, Key target,
         bool source_required) {
  std::unique_lock<std::shared_mutex> lock(table.mutex);
  auto source_it = table.map.find(source);
  if (source_it == table.map.end()) {
    if (source_required) throw std::runtime_error("Original source workspace does not exist");
    return 0;
  }
  if (!source_it->second.get()) throw std::runtime_error("Source workspace is empty");
  auto target_it = table.map.find(target);
  if (target_it != table.map.end()) {
    if (target_it->second.get() == source_it->second.get()) return 1;
    throw std::runtime_error("Target already owns a different workspace: no captured address may be replaced");
  }
  auto owner = std::make_shared<c10::DataPtr>(std::move(source_it->second));
  try {
    auto source_alias = alias(owner);
    auto target_alias = alias(owner);
    table.map.emplace(target, std::move(target_alias));
    source_it->second = std::move(source_alias);
  } catch (...) {
    source_it->second = std::move(*owner);
    throw;
  }
  return 2;
}
int frees = 0;
void free_cpu(void* value) { ++frees; std::free(value); }
}  

extern "C" const char* yelp_blas_workspace_error_v1() { return error.c_str(); }
extern "C" int yelp_blas_workspace_abi_v1() { return 1; }

extern "C" int yelp_bind_serial_blas_workspace_v1(
    std::uintptr_t handle, std::uintptr_t source_stream,
    std::uintptr_t target_stream, int* classic_status, int* lt_status) {
  try {
    error.clear();
    if (!handle || source_stream == target_stream)
      throw std::runtime_error("Distinct original source/target streams and nonempty handle required");
    auto source = std::make_tuple(reinterpret_cast<void*>(handle), reinterpret_cast<void*>(source_stream));
    auto target = std::make_tuple(reinterpret_cast<void*>(handle), reinterpret_cast<void*>(target_stream));
    *classic_status = bind(at::cuda::cublas_handle_stream_to_workspace(), source, target, true);
     
    *lt_status = bind(at::cuda::cublaslt_handle_stream_to_workspace(), source, target, false);
    return 0;
  } catch (const std::exception& exception) { error = exception.what(); return -1; }
}

extern "C" int yelp_blas_workspace_snapshot_v1(
    int family, std::uint64_t* rows, int capacity) {
  try {
    auto& table = family == 0 ? at::cuda::cublas_handle_stream_to_workspace()
                             : at::cuda::cublaslt_handle_stream_to_workspace();
    std::shared_lock<std::shared_mutex> lock(table.mutex);
    int count = 0;
    for (const auto& entry : table.map) {
      if (count < capacity) {
        rows[3*count] = reinterpret_cast<std::uintptr_t>(std::get<0>(entry.first));
        rows[3*count+1] = reinterpret_cast<std::uintptr_t>(std::get<1>(entry.first));
        rows[3*count+2] = reinterpret_cast<std::uintptr_t>(entry.second.get());
      }
      ++count;
    }
    return count;
  } catch (const std::exception& exception) { error = exception.what(); return -1; }
}

extern "C" int yelp_blas_workspace_owner_cpu_test_v1() {

  try {
    frees = 0;
    {
      at::cuda::WorkspaceMapWithMutex table;
      auto source = std::make_tuple(reinterpret_cast<void*>(1), reinterpret_cast<void*>(2));
      auto target = std::make_tuple(reinterpret_cast<void*>(1), reinterpret_cast<void*>(3));
      void* data = std::malloc(256);
      if (!data) throw std::bad_alloc();
      static_cast<unsigned char*>(data)[11] = 73;
      table.map.emplace(source, c10::DataPtr(data, data, free_cpu, c10::Device(c10::DeviceType::CPU)));
      if (bind(table, source, target, true) != 2 || table.map.at(source).get() != data
          || table.map.at(target).get() != data || static_cast<unsigned char*>(table.map.at(target).get())[11] != 73)
        throw std::runtime_error("CPU alias identity failed");
      if (bind(table, source, target, true) != 1) throw std::runtime_error("Idempotence failed");
      auto other = std::make_tuple(reinterpret_cast<void*>(1), reinterpret_cast<void*>(4));
      void* separate = std::malloc(16);
      table.map.emplace(other, c10::DataPtr(separate, separate, free_cpu, c10::Device(c10::DeviceType::CPU)));
      bool collision = false;
      try { bind(table, source, other, true); } catch (const std::runtime_error&) { collision = true; }
      if (!collision) throw std::runtime_error("Existing captured-address collision accepted");
      if (bind(table, other, std::make_tuple(reinterpret_cast<void*>(9),reinterpret_cast<void*>(9)), false) != 2)
        throw std::runtime_error("Second owner failed");
      table.map.erase(source);
      if (frees != 0 || table.map.at(target).get() != data)
        throw std::runtime_error("Owner was released while alias remained");
      table.map.erase(target);
      if (frees != 1) throw std::runtime_error("Original allocation not released once");
      bool missing = false;
      try { bind(table, source, target, true); } catch (const std::runtime_error&) { missing = true; }
      if (!missing) throw std::runtime_error("Missing source accepted");
    }
    if (frees != 2) throw std::runtime_error("Aliases caused a double release or leak");
    return 0;
  } catch (const std::exception& exception) { error = exception.what(); return -1; }
}
