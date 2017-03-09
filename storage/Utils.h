#ifndef OBAMADB_UTILS_H
#define OBAMADB_UTILS_H

#include <chrono>
#include <cmath>

#include "glog/logging.h"

#define PRINT_TIMING(block) { auto time_start = std::chrono::steady_clock::now();\
                              block\
                              auto time_end = std::chrono::steady_clock::now();\
                              std::chrono::duration<double, std::milli> time_ms = time_end - time_start;\
                              printf("[TIMING][%s:%d] elapsed time %.2f\n",__FILE__, __LINE__, time_ms.count()); \
                            }

#define PRINT_TIMING_MSG(msg, block) { \
                              auto time_start = std::chrono::steady_clock::now();\
                              block\
                              auto time_end = std::chrono::steady_clock::now();\
                              std::chrono::duration<double, std::milli> time_ms = time_end - time_start;\
                              printf("[TIMING][%s:%d][%s] elapsed time %.2f\n",__FILE__, __LINE__, msg, time_ms.count()); \
                            }

#define DISABLE_COPY_AND_ASSIGN(CLASS) \
            CLASS & operator=(const CLASS&) = delete;\
            CLASS(const CLASS&) = delete

namespace obamadb {


  /**
   * Simple vector for floating type numbers.
   */
  struct fvector {
    fvector(unsigned dimension)
      : dimension_(dimension) {
      values_ = new int_t[dimension_];
    }

    fvector(const fvector &other) {
      dimension_ = other.dimension_;
      values_ = new int_t[dimension_];
      memcpy(values_, other.values_, sizeof(int_t) * dimension_);
    }

    /**
     * An fvector filled with values [-1,1]
     * @param dim Number of elements in the new fvector.
     * @return An fvector filled with random floats.
     */
    static fvector GetRandomFVector(int const dim);

    ~fvector() {
      delete[] values_;
    }

    int_t &operator[](int idx) const {
      DCHECK_GT(dimension_, idx);

      return values_[idx];
    }

    void clear() {
      memset(values_, 0, sizeof(int_t) * dimension_);
    }

    unsigned dimension_;
    int_t *values_;

  };

  class QuickRandom {
  public:
    QuickRandom() : x(15486719), y(19654991), z(16313527), char_index(0) {
      LOG_IF(FATAL, sizeof(std::uint64_t) != 8) << "Expect 64_t to be 8 bytes long.";
      // warm up 5 cycles.
      for (int i = 0; i < 5; i++) {
        nextInt64();
      }
    }

    inline std::uint64_t nextInt64() {
      // TODO: there are many other ways to generate random numbers,
      // http://stackoverflow.com/questions/1640258/need-a-fast-random-generator-for-c
      // has several more techniques to choose from and includes this one.
      std::uint64_t t;
      x ^= x << 16;
      x ^= x >> 5;
      x ^= x << 1;

      t = x;
      x = y;
      y = z;
      z = t ^ x ^ y;

      return z;
    }

    inline std::uint32_t nextInt32() {
      if (char_index == 0) {
        char_index = 4;
        return *reinterpret_cast<uint32_t *>(&z);
      } else if (char_index == 4) {
        char_index = 8;
        return reinterpret_cast<uint32_t *>(&z)[1];
      } else {
        nextInt64();
        char_index = 4;
        return *reinterpret_cast<uint32_t *>(&z);
      }
    }

    inline float nextFloat() {
      std::uint32_t randi = nextInt32();
      randi &= 0x807FFFFF; // mask out the exponent.
      float randf = *reinterpret_cast<float*>(&randi);
      return randf;
    }

    inline unsigned char nextChar() {
      if (char_index >= 8) {
        nextInt64();
        char_index = 0;
      }
      return reinterpret_cast<unsigned char *>(&z)[char_index++];
    }

    std::uint64_t x, y, z;
    int char_index;
  };

  namespace stats {
    template<class T>
    double mean(std::vector<T> values) {
      double sum = 0;
      for (T& t : values) {
        sum += t;
      }
      return sum / values.size();
    }

    template<class T>
    double variance(std::vector<T> values) {
      double m = mean<T>(values);
      double sq_res = 0;
      for (T& t : values) {
        double diff = m - t;
        sq_res += diff * diff;
      }
      return sq_res/values.size();
    }

    template<class T>
    double stddev(std::vector<T> values) {
      double var = variance<T>(values);
      return sqrt(var);
    }

    template<class T>
    double stderr(std::vector<T> values) {
      double stdv = stddev<T>(values);
      return stdv/sqrt(values.size()); // stderr grows smaller as the number of samples increases
    }
  }

  /**
   * Takes a list like "1,2,3,42" and converts it into a vector of integers.
   * @param list - Integers to convert.
   * @return A vector of the represented ints.
   */
  std::vector<int> GetIntList(std::string const & list);

}  // namespace obamadb

#endif //OBAMADB_UTILS_H
