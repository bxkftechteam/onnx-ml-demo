#ifndef STOPWATCH_H_
#define STOPWATCH_H_

#include <chrono>

/**
 * A Stopwatch can be triggerd multiple times by calling Lap() after object
 * construction. Each Lap() call returns milliseconds elapsed since last Lap()
 * call. You can call Reset() to start next recording and get elapsed time
 * since object construction.
 */
class Stopwatch {
 public:
  Stopwatch() : begin_(Now()), latest_(begin_) {}

  std::chrono::milliseconds Lap() {
    auto now = Now();
    auto elapsed = now - latest_;
    latest_ = now;
    return elapsed;
  }

  std::chrono::milliseconds Reset() {
    auto now = Now();
    auto total_elapsed = now - begin_;
    begin_ = now;
    latest_ = now;
    return total_elapsed;
  }

 protected:
  static std::chrono::milliseconds Now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch());
  }

 private:
  std::chrono::milliseconds begin_;
  std::chrono::milliseconds latest_;
};

#endif  // STOPWATCH_H_
