// Tiny persistent thread pool: ParallelFor(n, fn(begin, end)) splits [0, n) into contiguous chunks.
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
namespace ppocr::indra {
class Pool {
 public:
  explicit Pool(int n) : n_(n < 1 ? 1 : n) {
    for (int i = 1; i < n_; ++i) th_.emplace_back([this, i] { Loop(i); });
  }
  ~Pool() {
    { std::lock_guard<std::mutex> g(m_); stop_ = true; ++gen_; }
    cv_.notify_all();
    for (auto& t : th_) t.join();
  }
  int size() const { return n_; }
  // fn(begin, end); chunks of at least `grain` items
  void For(int n, int grain, const std::function<void(int, int)>& fn) {
    int parts = std::min(n_, (n + grain - 1) / std::max(1, grain));
    if (parts <= 1) { if (n > 0) fn(0, n); return; }
    std::unique_lock<std::mutex> lk(run_);
    {
      std::lock_guard<std::mutex> g(m_);
      fn_ = &fn; total_ = n; parts_ = parts; pending_ = parts - 1; ++gen_;
    }
    cv_.notify_all();
    Chunk(0);
    std::unique_lock<std::mutex> g(m_);
    done_.wait(g, [this] { return pending_ == 0; });
    fn_ = nullptr;
  }
 private:
  void Chunk(int i) {
    if (i >= parts_) return;
    int b = int(int64_t(total_) * i / parts_), e = int(int64_t(total_) * (i + 1) / parts_);
    (*fn_)(b, e);
  }
  void Loop(int i) {
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> g(m_);
      cv_.wait(g, [&] { return gen_ != seen; });
      seen = gen_;
      if (stop_) return;
      bool mine = i < parts_;
      g.unlock();
      if (mine) {
        Chunk(i);
        std::lock_guard<std::mutex> g2(m_);
        if (--pending_ == 0) done_.notify_one();
      }
    }
  }
  int n_;
  std::vector<std::thread> th_;
  std::mutex m_, run_;
  std::condition_variable cv_, done_;
  const std::function<void(int, int)>* fn_ = nullptr;
  int total_ = 0, parts_ = 0, pending_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
};
}  // namespace ppocr::indra
