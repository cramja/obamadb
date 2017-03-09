#include <cmath>

#include "storage/StorageConstants.h"

#include "storage/Utils.h"

namespace obamadb {

  fvector fvector::GetRandomFVector(int const dim) {
    fvector shared_theta(dim);
    // initialize to values [-1000000,1000000]
    for (unsigned i = 0; i < dim; ++i) {
      //shared_theta[i] = (rand() % kScaleTheta * 2) - (kScaleTheta);
      shared_theta[i] = 0;
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