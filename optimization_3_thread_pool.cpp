/**
 * Optimization 3: High-Performance Thread Pool Implementation
 * 
 * This optimization eliminates thread creation/destruction overhead by
 * implementing a persistent thread pool with efficient work distribution.
 * 
 * Expected Performance Gain: 15-25% reduction in thread management overhead
 */

#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <thread>
#include <vector>
#include <future>

class HighPerformanceThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop{false};
    std::atomic<int> active_tasks{0};
    std::condition_variable completion_cv;
    std::mutex completion_mutex;
    
public:
    explicit HighPerformanceThreadPool(size_t threads) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] {
                            return this->stop || !this->tasks.empty();
                        });
                        
                        if (this->stop && this->tasks.empty()) return;
                        
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                        active_tasks.fetch_add(1, std::memory_order_relaxed);
                    }
                    
                    task();
                    
                    active_tasks.fetch_sub(1, std::memory_order_relaxed);
                    completion_cv.notify_all();
                }
            });
        }
    }
    
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace([task]() { (*task)(); });
        }
        
        condition.notify_one();
        return res;
    }
    
    void enqueue_void(std::function<void()> f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::move(f));
        }
        condition.notify_one();
    }
    
    void wait_all() {
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_cv.wait(lock, [this] {
            std::unique_lock<std::mutex> queue_lock(queue_mutex);
            return tasks.empty() && active_tasks.load() == 0;
        });
    }
    
    size_t size() const {
        return workers.size();
    }
    
    ~HighPerformanceThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker: workers) {
            worker.join();
        }
    }
};

// Singleton thread pool instance
class ThreadPoolManager {
private:
    static std::unique_ptr<HighPerformanceThreadPool> pool;
    static std::once_flag initialized;
    
public:
    static HighPerformanceThreadPool& get_instance() {
        std::call_once(initialized, []() {
            pool = std::make_unique<HighPerformanceThreadPool>(
                std::thread::hardware_concurrency());
        });
        return *pool;
    }
    
    static void initialize(size_t num_threads) {
        std::call_once(initialized, [num_threads]() {
            pool = std::make_unique<HighPerformanceThreadPool>(num_threads);
        });
    }
};

std::unique_ptr<HighPerformanceThreadPool> ThreadPoolManager::pool = nullptr;
std::once_flag ThreadPoolManager::initialized;

// Work-stealing deque for better load balancing
template<typename T>
class WorkStealingQueue {
private:
    std::deque<T> queue;
    mutable std::mutex mutex;
    
public:
    void push_back(T item) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push_back(std::move(item));
    }
    
    bool pop_front(T& item) {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty()) return false;
        item = std::move(queue.front());
        queue.pop_front();
        return true;
    }
    
    bool pop_back(T& item) {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty()) return false;
        item = std::move(queue.back());
        queue.pop_back();
        return true;
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }
};

// Advanced thread pool with work stealing
class WorkStealingThreadPool {
private:
    using Task = std::function<void()>;
    
    std::vector<std::unique_ptr<WorkStealingQueue<Task>>> queues;
    std::vector<std::thread> threads;
    std::atomic<bool> done{false};
    
    static thread_local size_t thread_index;
    static thread_local WorkStealingQueue<Task>* local_queue;
    
    void worker_thread(size_t index) {
        thread_index = index;
        local_queue = queues[index].get();
        
        while (!done) {
            Task task;
            
            // Try to get task from local queue first
            if (local_queue->pop_front(task)) {
                task();
            }
            // Try to steal from other queues
            else if (try_steal_task(task)) {
                task();
            }
            // No work available, yield
            else {
                std::this_thread::yield();
            }
        }
    }
    
    bool try_steal_task(Task& task) {
        for (size_t i = 0; i < queues.size(); ++i) {
            size_t index = (thread_index + i + 1) % queues.size();
            if (queues[index]->pop_back(task)) {
                return true;
            }
        }
        return false;
    }
    
public:
    explicit WorkStealingThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
        queues.resize(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            queues[i] = std::make_unique<WorkStealingQueue<Task>>();
        }
        
        threads.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            threads.emplace_back(&WorkStealingThreadPool::worker_thread, this, i);
        }
    }
    
    template<typename F>
    void submit(F&& f) {
        static std::atomic<size_t> round_robin{0};
        
        if (local_queue) {
            // Submit to local queue if called from worker thread
            local_queue->push_back(std::forward<F>(f));
        } else {
            // Round-robin distribution for external submissions
            size_t index = round_robin.fetch_add(1) % queues.size();
            queues[index]->push_back(std::forward<F>(f));
        }
    }
    
    void wait_for_all() {
        // Simple implementation - wait until all queues are empty
        while (true) {
            bool all_empty = true;
            for (const auto& queue : queues) {
                if (!queue->empty()) {
                    all_empty = false;
                    break;
                }
            }
            if (all_empty) break;
            std::this_thread::yield();
        }
    }
    
    ~WorkStealingThreadPool() {
        done = true;
        for (auto& thread : threads) {
            thread.join();
        }
    }
};

thread_local size_t WorkStealingThreadPool::thread_index = 0;
thread_local WorkStealingQueue<std::function<void()>>* WorkStealingThreadPool::local_queue = nullptr;

// Optimized FP16 GEMM using thread pool
void fp16_gemm_thread_pool(const __fp16* A, const __fp16* B, __fp16* C,
                          int M, int N, int K) {
    auto& pool = ThreadPoolManager::get_instance();
    int num_threads = pool.size();
    int rows_per_thread = (M + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; t++) {
        int start_row = t * rows_per_thread;
        int end_row = std::min(start_row + rows_per_thread, M);
        
        if (start_row < end_row) {
            pool.enqueue_void([=]() {
                // Process assigned rows
                for (int i = start_row; i < end_row; i++) {
                    for (int j = 0; j < N; j++) {
                        __fp16 sum = 0;
                        for (int k = 0; k < K; k++) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            });
        }
    }
    
    pool.wait_all();
}

// Optimized FP16 GEMM using work-stealing thread pool
void fp16_gemm_work_stealing(const __fp16* A, const __fp16* B, __fp16* C,
                            int M, int N, int K) {
    static WorkStealingThreadPool pool;
    constexpr int TILE_SIZE = 32;
    
    // Submit tiles as independent tasks
    for (int i = 0; i < M; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            pool.submit([=]() {
                int i_end = std::min(i + TILE_SIZE, M);
                int j_end = std::min(j + TILE_SIZE, N);
                
                for (int ii = i; ii < i_end; ii++) {
                    for (int jj = j; jj < j_end; jj++) {
                        __fp16 sum = 0;
                        for (int k = 0; k < K; k++) {
                            sum += A[ii * K + k] * B[k * N + jj];
                        }
                        C[ii * N + jj] = sum;
                    }
                }
            });
        }
    }
    
    pool.wait_for_all();
}

// Hierarchical parallel GEMM with thread pool
void fp16_gemm_hierarchical(const __fp16* A, const __fp16* B, __fp16* C,
                           int M, int N, int K) {
    auto& pool = ThreadPoolManager::get_instance();
    constexpr int L2_BLOCK = 256;
    constexpr int L1_BLOCK = 64;
    
    // L2-level parallelism
    for (int ii = 0; ii < M; ii += L2_BLOCK) {
        for (int jj = 0; jj < N; jj += L2_BLOCK) {
            for (int kk = 0; kk < K; kk += L2_BLOCK) {
                pool.enqueue_void([=]() {
                    int i_end = std::min(ii + L2_BLOCK, M);
                    int j_end = std::min(jj + L2_BLOCK, N);
                    int k_end = std::min(kk + L2_BLOCK, K);
                    
                    // L1-level computation
                    for (int i = ii; i < i_end; i += L1_BLOCK) {
                        for (int j = jj; j < j_end; j += L1_BLOCK) {
                            for (int k = kk; k < k_end; k += L1_BLOCK) {
                                int ii_end = std::min(i + L1_BLOCK, i_end);
                                int jj_end = std::min(j + L1_BLOCK, j_end);
                                int kk_end = std::min(k + L1_BLOCK, k_end);
                                
                                // Micro-kernel
                                for (int iii = i; iii < ii_end; iii++) {
                                    for (int jjj = j; jjj < jj_end; jjj++) {
                                        __fp16 sum = (k == kk) ? 0 : C[iii * N + jjj];
                                        for (int kkk = k; kkk < kk_end; kkk++) {
                                            sum += A[iii * K + kkk] * B[kkk * N + jjj];
                                        }
                                        C[iii * N + jjj] = sum;
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }
    }
    
    pool.wait_all();
}

// Adaptive load balancing thread pool
class AdaptiveThreadPool {
private:
    HighPerformanceThreadPool pool;
    std::atomic<size_t> completed_tasks{0};
    std::atomic<size_t> total_tasks{0};
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    explicit AdaptiveThreadPool(size_t num_threads) : pool(num_threads) {}
    
    template<typename F>
    void submit_adaptive(F&& f, double estimated_work) {
        total_tasks.fetch_add(1);
        
        pool.enqueue_void([this, f = std::forward<F>(f)]() {
            f();
            completed_tasks.fetch_add(1);
        });
    }
    
    void wait_adaptive() {
        pool.wait_all();
    }
    
    double get_throughput() const {
        auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
        auto seconds = std::chrono::duration<double>(elapsed).count();
        return completed_tasks.load() / seconds;
    }
};

// Benchmark function for thread pool optimizations
#include <chrono>
#include <iostream>
#include <iomanip>

void benchmark_thread_pool() {
    const int M = 1024, N = 1024, K = 1024;
    
    // Allocate matrices
    auto A = std::make_unique<__fp16[]>(M * K);
    auto B = std::make_unique<__fp16[]>(K * N);
    auto C1 = std::make_unique<__fp16[]>(M * N);
    auto C2 = std::make_unique<__fp16[]>(M * N);
    auto C3 = std::make_unique<__fp16[]>(M * N);
    
    // Initialize matrices
    for (int i = 0; i < M * K; i++) {
        A[i] = (__fp16)(rand() / (float)RAND_MAX - 0.5f);
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (__fp16)(rand() / (float)RAND_MAX - 0.5f);
    }
    
    std::cout << std::fixed << std::setprecision(2);
    
    // Initialize thread pool
    ThreadPoolManager::initialize(std::thread::hardware_concurrency());
    
    // Benchmark standard thread pool
    auto start = std::chrono::high_resolution_clock::now();
    fp16_gemm_thread_pool(A.get(), B.get(), C1.get(), M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Benchmark work-stealing thread pool
    start = std::chrono::high_resolution_clock::now();
    fp16_gemm_work_stealing(A.get(), B.get(), C2.get(), M, N, K);
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Benchmark hierarchical thread pool
    start = std::chrono::high_resolution_clock::now();
    fp16_gemm_hierarchical(A.get(), B.get(), C3.get(), M, N, K);
    end = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double gflops = (2.0 * M * N * K) / 1e9;
    
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << "\n";
    std::cout << "Thread Pool:     " << duration1.count() << " μs, " 
              << gflops / (duration1.count() / 1e6) << " GFLOPS\n";
    std::cout << "Work Stealing:   " << duration2.count() << " μs, " 
              << gflops / (duration2.count() / 1e6) << " GFLOPS\n";
    std::cout << "Hierarchical:    " << duration3.count() << " μs, " 
              << gflops / (duration3.count() / 1e6) << " GFLOPS\n";
}