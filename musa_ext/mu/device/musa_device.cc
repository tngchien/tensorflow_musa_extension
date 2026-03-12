#include "musa_device.h"

#include <iostream>

#include "mu/device/musa_event.h"
#include "musa_allocator.h"
#include "musa_memcpy.h"
#include "musa_memset.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"

namespace tensorflow {
namespace musa {

MusaDeviceContext::MusaDeviceContext(
    musaStream_t stream, musaStream_t h2d_stream,
    ::stream_executor::StreamExecutor* executor)
    : stream_handle_(stream), h2d_stream_(h2d_stream) {
  implementation_ = new ::stream_executor::musa::MusaStream(stream);
  official_stream_ = new ::stream_executor::Stream(executor, implementation_);

  // 初始化 Stream
  official_stream_->Init();

  // --- 新增：启动 H2D 异步轮询后台线程 ---
  polling_thread_ = std::thread(&MusaDeviceContext::PollingLoop, this);
}

MusaDeviceContext::~MusaDeviceContext() {
  // --- 新增：优雅关闭轮询线程并安全排空队列 ---
  stop_polling_ = true;
  if (polling_thread_.joinable()) {
    polling_thread_.join();
  }
  {
    std::lock_guard<std::mutex> lock(cleanup_mu_);
    while (!cleanup_queue_.empty()) {
      auto* payload = cleanup_queue_.front();
      cleanup_queue_.pop();
      // 强行等待剩余任务完成，确保安全释放 TF 内存
      musaEventSynchronize(payload->sync_event);
      payload->done(Status::OK());
      musaEventDestroy(payload->sync_event);
      delete payload;
    }
  }

  if (official_stream_) {
    official_stream_->BlockHostUntilDone().IgnoreError();
    delete official_stream_;
  }
}

// --- 新增：宿主线程轮询循环，完美避开 Eager 死锁 ---
void MusaDeviceContext::PollingLoop() const {
  while (!stop_polling_) {
    AsyncCopyPayload* payload = nullptr;
    {
      std::lock_guard<std::mutex> lock(cleanup_mu_);
      if (!cleanup_queue_.empty()) {
        payload = cleanup_queue_.front();
      }
    }

    if (payload) {
      // 非阻塞查询 GPU 状态
      if (musaEventQuery(payload->sync_event) == musaSuccess) {
        // 关键：通知 TF 框架这块内存可以用啦，立刻放行下游算子
        payload->done(Status::OK());
        musaEventDestroy(payload->sync_event);
        delete payload;

        std::lock_guard<std::mutex> lock(cleanup_mu_);
        cleanup_queue_.pop();
        continue;
      }
    }
    // 让出 CPU 切片，防止空转拉高负载
    std::this_thread::sleep_for(std::chrono::microseconds(20));
  }
}

void MusaDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                              Device* device,
                                              Tensor* device_tensor,
                                              StatusCallback done,
                                              bool sync_dst_compute) const {
  auto* musa_dev = static_cast<MusaDevice*>(device);
  musaSetDevice(musa_dev->get_device_id());

  const void* src = cpu_tensor->tensor_data().data();
  void* dst = const_cast<char*>(device_tensor->tensor_data().data());
  size_t bytes = cpu_tensor->TotalBytes();

  if (bytes > 0) {
    // 1. 核心修复：回退到 stream_handle_ 下发拷贝
    // 因为 MusaBFCAllocator 缺乏跨流追踪，只有在计算流上下发，
    // 才能天然保证不会覆盖该流中前一个算子正在使用的复用内存。
    MusaMemcpyAsyncH2D(dst, src, bytes, stream_handle_);

    // 2. 创建并记录完成事件 (记录在 stream_handle_ 上)
    musaEvent_t copy_done_event;
    musaEventCreate(&copy_done_event);
    musaEventRecord(copy_done_event, stream_handle_);

    // 3. 将 done 闭包托管给异步轮询线程
    // 绝不在此处调用 musaStreamSynchronize，彻底释放 CPU 调度器！
    AsyncCopyPayload* payload =
        new AsyncCopyPayload{std::move(done), copy_done_event};

    {
      std::lock_guard<std::mutex> lock(cleanup_mu_);
      cleanup_queue_.push(payload);
    }
  } else {
    done(Status::OK());
  }
}

void MusaDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                              StringPiece tensor_name,
                                              Device* device,
                                              Tensor* cpu_tensor,
                                              StatusCallback done) {
  auto* musa_dev = static_cast<MusaDevice*>(device);
  musaSetDevice(musa_dev->get_device_id());

  const void* src = device_tensor->tensor_data().data();
  void* dst = const_cast<char*>(cpu_tensor->tensor_data().data());
  size_t bytes = device_tensor->TotalBytes();

  if (bytes > cpu_tensor->TotalBytes()) {
    bytes = cpu_tensor->TotalBytes();
  }

  if (bytes > 0) {
    mStatus m_stat = MusaMemcpyAsyncD2H(dst, src, bytes, stream_handle_);
    if (m_stat != mStatus::SUCCESS) {
      done(errors::Internal("MUSA D2H async copy failed."));
      return;
    }
    // --- 强制同步底线：防止 Shape 为 0 和 Eager 崩溃 ---
    // 这个同步是必须的，以确保小规模控制流数据的即时可用性
    musaError_t sync_err = musaStreamSynchronize(stream_handle_);
    if (sync_err != musaSuccess) {
      done(errors::Internal("MUSA D2H stream sync failed: ",
                            musaGetErrorString(sync_err)));
      return;
    }
  }
  done(Status::OK());
}

MusaDevice::MusaDevice(Env* env, const DeviceAttributes& attributes,
                       int device_id,
                       ::stream_executor::StreamExecutor* executor)
    : Device(env, attributes), device_id_(device_id) {
  musaSetDevice(device_id_);

  // 初始化计算流
  musaError_t stream_err = musaStreamCreate(&stream_);
  if (stream_err != musaSuccess) {
    LOG(ERROR) << ">>> [MUSA] ERROR: Device " << device_id_
               << " failed to create stream";
    return;
  }

  // --- 新增：初始化专门用于数据预取的 H2D 流 ---
  musaError_t h2d_err = musaStreamCreate(&h2d_stream_);
  if (h2d_err != musaSuccess) {
    LOG(ERROR) << ">>> [MUSA] ERROR: Device " << device_id_
               << " failed to create h2d_stream";
    return;
  }

  mudnn_handle_.reset(new ::musa::dnn::Handle());
  ::musa::dnn::Status s = mudnn_handle_->SetStream(stream_);
  if (s != ::musa::dnn::Status::SUCCESS) {
    mudnn_handle_.reset();
    musaStreamDestroy(stream_);
    musaStreamDestroy(h2d_stream_);
    return;
  }

  mublasStatus_t blas_err = mublasCreate(&mublas_handle_);
  if (blas_err != MUBLAS_STATUS_SUCCESS) {
    mublas_handle_ = nullptr;
    musaStreamDestroy(stream_);
    musaStreamDestroy(h2d_stream_);
    mudnn_handle_.reset();
    return;
  }

  blas_err = mublasSetStream(mublas_handle_, stream_);
  if (blas_err != MUBLAS_STATUS_SUCCESS) {
    mublasDestroy(mublas_handle_);
    mublas_handle_ = nullptr;
    musaStreamDestroy(stream_);
    musaStreamDestroy(h2d_stream_);
    mudnn_handle_.reset();
    return;
  }

  // --- 修改：将新增的 h2d_stream_ 传递给 Context ---
  device_context_ = new MusaDeviceContext(stream_, h2d_stream_, executor);
  musa_allocator_ = new MusaBFCAllocator(device_id_);

  gpu_device_info_.stream = device_context_->stream();
  gpu_device_info_.default_context = device_context_;
  gpu_device_info_.gpu_id = device_id_;

  set_tensorflow_gpu_device_info(&gpu_device_info_);
}

MusaDevice::~MusaDevice() {
  musaSetDevice(device_id_);
  if (device_context_) {
    device_context_->Unref();
  }
  if (mublas_handle_) {
    mublasDestroy(mublas_handle_);
  }
  if (musa_allocator_) {
    delete musa_allocator_;
  }
  // --- 新增：释放 H2D 流 ---
  if (h2d_stream_) {
    musaStreamDestroy(h2d_stream_);
  }
  if (stream_) {
    musaStreamDestroy(stream_);
  }
}

Allocator* MusaDevice::GetAllocator(AllocatorAttributes attr) {
  return attr.on_host() ? cpu_allocator() : musa_allocator_;
}

Status MusaDevice::Sync() {
  musaSetDevice(device_id_);
  // 在设备同步时，排空所有底层操作，确保生命周期安全
  musaError_t err = musaDeviceSynchronize();
  return (err == musaSuccess) ? Status::OK()
                              : errors::Internal("MUSA Device Sync Failed");
}

Status MusaDevice::TryGetDeviceContext(DeviceContext** out_context) {
  if (device_context_) {
    *out_context = device_context_;
    device_context_->Ref();
    return Status::OK();
  }
  return errors::Internal("MusaDeviceContext is null");
}

}  // namespace musa
}  // namespace tensorflow
