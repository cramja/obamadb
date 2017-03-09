#ifndef OBAMADB_STORAGECONSTANTS_H_
#define OBAMADB_STORAGECONSTANTS_H_

#include <cinttypes>

namespace obamadb {

  // The basic machine learning number type. Since it can be either floating point or
  // integer type, we call it num(ber) type.
  typedef float num_t;

  const std::uint64_t kStorageBlockSize = (2000000);// (std::uint64_t)(256000 * 0.7);  // 2 megabytes.
  // mac l2 - 256kb
  //     l3 - 8mb

}  // namespace obamadb

#endif //OBAMADB_STORAGECONSTANTS_H_
