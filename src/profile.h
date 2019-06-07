#ifdef OUMU_PROF
#ifndef OUMU_PROF_H
#define OUMU_PROF_H
#include <string>
#include <atomic>

class profiler {
  static std::atomic<bool> running_profiler;

public:
  profiler(const std::string&);

  ~profiler();
};

#endif
#endif
