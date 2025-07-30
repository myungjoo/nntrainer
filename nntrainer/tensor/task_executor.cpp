// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   task_executor.h
 * @date   04 April 2025
 * @brief  This file contains a task executor with Windows x64 optimizations
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Task executor class
 *
 */

#include "task_executor.h"

#include <nntrainer_error.h>
#include <nntrainer_log.h>

// Windows x64 optimization: Enhanced thread pool with CPU affinity
#ifdef _WIN32
#include <windows.h>
#include <thread>

namespace x64_task_config {
    // Optimize for 6-thread configuration from windows-native.ini
    constexpr int OPTIMAL_THREADS = 6;
    constexpr int HIGH_PRIORITY_THREADS = 2;    // Reserve for critical tasks
    constexpr int NORMAL_PRIORITY_THREADS = 4;  // For regular LLM operations
    
    // Windows-specific CPU affinity optimization
    static void set_thread_affinity(size_t thread_id) {
        HANDLE thread_handle = GetCurrentThread();
        DWORD_PTR affinity_mask = 1ULL << (thread_id % std::thread::hardware_concurrency());
        SetThreadAffinityMask(thread_handle, affinity_mask);
        
        // Set thread priority for better responsiveness
        if (thread_id < HIGH_PRIORITY_THREADS) {
            SetThreadPriority(thread_handle, THREAD_PRIORITY_ABOVE_NORMAL);
        } else {
            SetThreadPriority(thread_handle, THREAD_PRIORITY_NORMAL);
        }
    }
    
    // Windows x64: Optimize thread count for transformer workloads  
    static int get_optimal_thread_count_for_workload(size_t task_count, size_t complexity) {
        // For attention heads (typically 8, 12, 16)
        if (task_count <= 16 && complexity < 10000) {
            return std::min(static_cast<int>(task_count), 4);
        }
        
        // For large matrix operations or feed-forward layers
        if (complexity >= 100000) {
            return OPTIMAL_THREADS;
        }
        
        // For medium-sized operations
        return std::min(static_cast<int>(task_count), NORMAL_PRIORITY_THREADS);
    }
}
#endif

namespace nntrainer {

TaskExecutor::TaskExecutor(std::string n, size_t thread_count) :
  name(n), stop(false) {
  
#ifdef _WIN32
  // Windows x64 optimization: Use optimal thread count
  thread_count = std::min(thread_count, static_cast<size_t>(x64_task_config::OPTIMAL_THREADS));
#endif
  
  for (size_t i = 0; i < thread_count; ++i) {
    workers.emplace_back([this, i] { this->worker_thread(i); });
  }
}

TaskExecutor::~TaskExecutor() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }

  cond_var.notify_all();
  for (std::thread &t : workers) {
    if (t.joinable())
      t.join();
  }
}

void TaskExecutor::worker_thread(size_t worker_id) {
#ifdef _WIN32
  // Windows x64 optimization: Set CPU affinity and thread priority
  x64_task_config::set_thread_affinity(worker_id);
#endif

  while (true) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      cond_var.wait(lock, [this]() { return stop || !task_queue.empty(); });

      if (stop && task_queue.empty()) {
        return;
      }

      task = std::move(task_queue.front());
      task_queue.pop();
      task_started[task.id] = true;
      task_started_cv.notify_all();

      // we are not going to remove the Done Tasks.
      // we exeplicitly call release tasks. until then, we keep the results and
      // not going to submit that task again
      // queued_ids.erase(task.id);
    }

    try {
      task.callback(task.data);
      task.promise.set_value();
    } catch (...) {
      ml_loge("[%s] : [Error ] Worker %zu Task ID %d threw an exception\n", 
              name.c_str(), worker_id, task.id);
    }
  }
}

int TaskExecutor::submit(TaskCallback cb, void *data) {

  auto canceled = std::make_shared<std::atomic_bool>(false);
  auto promise = std::make_shared<std::promise<void>>();
  std::shared_future<void> fut = promise->get_future().share();
  int id = getNextTaskId();

  {
    std::lock_guard<std::mutex> lock(queue_mutex);

    if (future_map.count(id)) {
      if (!future_map[id].valid()) {
        ml_loge("[%s] : [Error] Future is not valid : Task id - %d\n",
                name.c_str(), id);
      }
      auto status = future_map[id].wait_for(std::chrono::seconds(0));
      if (status != std::future_status::ready) {
        ml_logi("[%s] : Task id - %d is still active\n", name.c_str(), id);
        return id;
      }
    }

    Task task{id, std::move(*promise), cb, data};

    future_map[id] = fut;

    task_queue.push(std::move(task));
  }
  cond_var.notify_one();

  return id;
}

// Windows x64 optimization: Batch task submission for transformer layers
int TaskExecutor::submit_batch_transformer(const std::vector<TaskCallback>& callbacks,
                                          const std::vector<void*>& data,
                                          TransformerTaskType task_type) {
#ifdef _WIN32
    if (callbacks.size() != data.size() || callbacks.empty()) {
        ml_loge("[%s] : Invalid batch submission parameters\n", name.c_str());
        return -1;
    }
    
    // Determine optimal parallelization based on task type
    size_t complexity = 0;
    switch (task_type) {
        case TransformerTaskType::ATTENTION_HEADS:
            complexity = 5000;  // Small per-head operations
            break;
        case TransformerTaskType::FEEDFORWARD:
            complexity = 50000; // Large matrix operations
            break;
        case TransformerTaskType::LAYERNORM:
            complexity = 1000;  // Fast vector operations
            break;
        default:
            complexity = 10000;
    }
    
    int optimal_threads = x64_task_config::get_optimal_thread_count_for_workload(
        callbacks.size(), complexity);
    
    // If we have more tasks than optimal threads, batch them
    if (callbacks.size() > static_cast<size_t>(optimal_threads)) {
        return submit_batched_tasks(callbacks, data, optimal_threads);
    }
#endif
    
    // Submit all tasks individually if count is manageable
    std::vector<int> task_ids;
    for (size_t i = 0; i < callbacks.size(); i++) {
        int task_id = submit(callbacks[i], data[i]);
        task_ids.push_back(task_id);
    }
    
    return task_ids.empty() ? -1 : task_ids[0];
}

#ifdef _WIN32
int TaskExecutor::submit_batched_tasks(const std::vector<TaskCallback>& callbacks,
                                      const std::vector<void*>& data,
                                      int num_batches) {
    std::vector<TaskCallback> batch_callbacks;
    std::vector<void*> batch_data;
    
    struct BatchData {
        std::vector<TaskCallback> callbacks;
        std::vector<void*> data;
        size_t start_idx;
        size_t end_idx;
    };
    
    std::vector<std::unique_ptr<BatchData>> batch_data_storage;
    
    size_t tasks_per_batch = callbacks.size() / num_batches;
    size_t remaining_tasks = callbacks.size() % num_batches;
    
    size_t current_idx = 0;
    for (int batch = 0; batch < num_batches; batch++) {
        size_t batch_size = tasks_per_batch + (batch < remaining_tasks ? 1 : 0);
        
        auto batch_data_ptr = std::make_unique<BatchData>();
        batch_data_ptr->start_idx = current_idx;
        batch_data_ptr->end_idx = current_idx + batch_size;
        
        for (size_t i = current_idx; i < current_idx + batch_size; i++) {
            batch_data_ptr->callbacks.push_back(callbacks[i]);
            batch_data_ptr->data.push_back(data[i]);
        }
        
        // Create batch executor lambda
        auto batch_executor = [](void* data) {
            BatchData* batch = static_cast<BatchData*>(data);
            for (size_t i = 0; i < batch->callbacks.size(); i++) {
                batch->callbacks[i](batch->data[i]);
            }
        };
        
        batch_callbacks.push_back(batch_executor);
        batch_data.push_back(batch_data_ptr.get());
        batch_data_storage.push_back(std::move(batch_data_ptr));
        
        current_idx += batch_size;
    }
    
    // Submit batched tasks
    std::vector<int> batch_task_ids;
    for (size_t i = 0; i < batch_callbacks.size(); i++) {
        int task_id = submit(batch_callbacks[i], batch_data[i]);
        batch_task_ids.push_back(task_id);
    }
    
    // Wait for all batches to complete
    for (int task_id : batch_task_ids) {
        waitForTaskCompletion(task_id);
    }
    
    return batch_task_ids.empty() ? -1 : batch_task_ids[0];
}
#endif

void TaskExecutor::submitTasks(const std::vector<TaskDesc> &tasks) {
  for (const auto &task : tasks) {
    submit(task.callback, task.data);
  }
}

bool TaskExecutor::cancel(int id) {
  std::lock_guard<std::mutex> lock(queue_mutex);
  auto it = cancel_map.find(id);
  if (it != cancel_map.end()) {
    *(it->second) = true;
    return true;
  }
  return false;
}

void TaskExecutor::wait(int id) {
  std::shared_future<void> fut;
  {
    std::unique_lock<std::mutex> lock(queue_mutex);

    task_started_cv.wait(
      lock, [&] { return task_started.count(id) && task_started[id]; });

    auto it = future_map.find(id);
    if (it == future_map.end() || !it->second.valid()) {
      return;
    }
    fut = it->second;
  }
  try {
    fut.wait();
  } catch (const std::future_error &e) {
    ml_loge("[%s] : exception while waiting on future : %s\n", name.c_str(),
            e.what());
  }
}

void TaskExecutor::waitAll(const std::vector<int> &ids) {
  std::vector<std::shared_future<void>> futures;
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (int id : ids) {
      auto it = future_map.find(id);
      if (it != future_map.end()) {
        futures.push_back(it->second);
      } else {
        ml_logw("[%s] : Task ID is not found : %d\n", name.c_str(), id);
      }
    }
  }

  for (auto &fut : futures) {
    try {
      fut.wait();
    } catch (const std::exception &e) {
      ml_loge("[%s] : exception while waiting on future : %s\n", name.c_str(),
              e.what());
    }
  }
}

bool TaskExecutor::isDone(int id) {
  std::lock_guard<std::mutex> lock(queue_mutex);
  auto it = future_map.find(id);
  if (it == future_map.end())
    return false;
  return it->second.wait_for(std::chrono::seconds(0)) ==
         std::future_status::ready;
}

bool TaskExecutor::isAllDone(const std::vector<int> &ids) {
  std::lock_guard<std::mutex> lock(queue_mutex);
  for (int id : ids) {
    isDone(id);
  }
  return true;
}

void TaskExecutor::releaseTask(int id) {
  std::lock_guard<std::mutex> lock(queue_mutex);
  future_map.erase(id);
  cancel_map.erase(id);
  reusable_ids.push(id);
}

} // namespace nntrainer
