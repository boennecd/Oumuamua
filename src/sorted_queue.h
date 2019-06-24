#ifndef SORTED_QUEUE_H
#define SORTED_QUEUE_H
#include <deque>
#include <algorithm>

template<typename T, double keyFunc(const T&)>
class sorted_queue : public std::deque<T> {
  using iterator = typename std::deque<T>::iterator;
  using base = std::deque<T>;
public:
  using base::base;
  using base::begin;
  using base::end;
  using base::cbegin;
  using base::cend;
  using base::push_front;
  using base::emplace_front;

  void sort(){
    std::sort(
      begin(), end(), [](const T& i1, const T& i2){
        return keyFunc(i1) < keyFunc(i2);
      });
  }
};

#endif
