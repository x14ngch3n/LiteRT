// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_SHARED_CORE_H_
#define ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_SHARED_CORE_H_

#include <memory>
#include <mutex>  // NOLINT
#include <optional>
#include <string>
#include <vector>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

class OpenVINOSharedCore {
 public:
  OpenVINOSharedCore(const OpenVINOSharedCore&) = delete;
  OpenVINOSharedCore(OpenVINOSharedCore&&) = delete;
  OpenVINOSharedCore& operator=(const OpenVINOSharedCore&) = delete;
  OpenVINOSharedCore& operator=(OpenVINOSharedCore&&) = delete;

  static OpenVINOSharedCore* GetInstance();

  // Returns the shared OpenVINO core, creating it on first use.
  std::shared_ptr<ov::Core> GetCore() {
    absl::MutexLock lock(state_mutex_);
    EnsureCore();
    return core_;
  }

  // Alias for GetCore().
  std::shared_ptr<ov::Core> getCore() { return GetCore(); }

  // Increments the reference count tracking active invocation contexts.
  void Acquire() {
    absl::MutexLock lock(state_mutex_);
    ++ref_count_;
  }

  // Decrements the reference count. Once the last reference is released, the
  // remote context is cleared and the shared core is recreated so the next
  // acquisition starts from a clean state.
  void Release() {
    absl::MutexLock lock(state_mutex_);
    if (--ref_count_ <= 0) {
      ref_count_ = 0;
      remote_context_.reset();
      // Destroy the existing core before creating a new one so the underlying
      // hardware resources are fully released first.
      core_.reset();
      core_ = std::make_shared<ov::Core>();
    }
  }

  void SetDevice(const std::string& device) {
    absl::MutexLock lock(state_mutex_);
    device_ = device;
    remote_context_.reset();
  }
  std::string GetDevice() {
    absl::MutexLock lock(state_mutex_);
    return device_;
  }

  ov::RemoteContext GetRemoteContext() {
    absl::MutexLock lock(state_mutex_);
    EnsureCore();
    if (!remote_context_.has_value()) {
      remote_context_ = core_->get_default_context(device_);
    }
    return *remote_context_;
  }

  // Returns the list of OpenVINO devices reported by `core_->
  // get_available_devices()`.  Queried lazily on first call and cached for
  // the lifetime of the process (the set of installed devices does not
  // change at runtime).  Thread-safe.  Returns an empty vector if the
  // underlying query throws.
  const std::vector<std::string>& GetAvailableDevices();

 private:
  OpenVINOSharedCore();
  ~OpenVINOSharedCore();

  // Creates the shared core if it does not exist yet. Callers must hold
  // `state_mutex_`.
  void EnsureCore() ABSL_EXCLUSIVE_LOCKS_REQUIRED(state_mutex_) {
    if (core_ == nullptr) {
      core_ = std::make_shared<ov::Core>();
    }
  }

  // Guards core_, device_, remote_context_, and ref_count_.
  absl::Mutex state_mutex_;
  std::shared_ptr<ov::Core> core_ ABSL_GUARDED_BY(state_mutex_);
  std::string device_ ABSL_GUARDED_BY(state_mutex_) = "NPU";  // Default device
  std::optional<ov::RemoteContext> remote_context_
      ABSL_GUARDED_BY(state_mutex_);
  int ref_count_ ABSL_GUARDED_BY(state_mutex_) = 0;
  std::once_flag available_devices_once_;
  std::vector<std::string> available_devices_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_OPENVINO_DISPATCH_OPENVINO_SHARED_CORE_H_
