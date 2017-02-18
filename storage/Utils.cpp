#include "storage/StorageConstants.h"

#include <cmath>
#include <climits>

#include "storage/Utils.h"

namespace obamadb {

  fvector fvector::GetRandomFVector(int const dim) {
    fvector shared_theta(dim);
    // initialize to values [-1,1]
    float const half_max = ((float)INT_MAX)/2.0;
    for (unsigned i = 0; i < dim; ++i) {
      float const randf = (float)rand();
      shared_theta[i] = static_cast<num_t>(randf/half_max - 1.0);
    }
    return shared_theta;
  }

  aligned_fvector aligned_fvector::GetRandomFVector(int const dim) {
    aligned_fvector shared_theta(dim);
    // initialize to values [-1,1]
    float const half_max = ((float)INT_MAX)/2.0;
    for (unsigned i = 0; i < dim; ++i) {
      float const randf = (float)rand();
      shared_theta[i] = static_cast<num_t>(randf/half_max - 1.0);
    }
    return shared_theta;
  }

  std::vector<int> GetIntList(std::string const & list) {
    std::vector<int> vec;
    char const * raw = list.c_str();
    int cur_int = 0;
    bool negate = false;
    for (int i = 0; i < list.size(); i++) {
      if (raw[i] == ',') {
        vec.push_back(cur_int * (negate ? -1 : 1));
        cur_int = 0;
        negate = false;
      } else {
        if (raw[i] == '-' && (i == 0 || raw[i-1] == ',')) {
          negate = true;
        } else {
          CHECK_LT(raw[i], 48 + 10) << "invalid int";
          CHECK_GE(raw[i], 48) << "invalid int";
          cur_int *= 10;
          cur_int += raw[i] - 48;
        }
      }
    }
    vec.push_back(cur_int);
    return vec;
  }

}
