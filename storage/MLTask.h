#ifndef OBAMADB_MLTASK_H
#define OBAMADB_MLTASK_H

#include <cmath>
#include <mutex>

#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/exvector.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

namespace obamadb {

  namespace ml {
    /**
     * Dot product
     */
    num_t dot(const dvector <num_t> &v1, num_t const *d2);

    /**
     * Sparse dot product
     */
    num_t dot(const svector <num_t> &v1, num_t *d2);

    void scale(dvector <num_t> &v1, num_t e);

    /**
     * Dense scale and add
     */
    void scale_and_add(dvector <num_t> &v1, dvector <num_t> &v2, num_t e);

    /**
     * Sparse scale and add. Only updates indices present in delta.
     */
    void scale_and_add(num_t *theta, const svector <num_t> &delta, const num_t e);

  }  // namespace ml

  enum class MLAlgorithm {
    kSVM,
    kMC
  };

  class MLTask {
  public:
    /*
     * Takes ownership of the DataView.
     */
    MLTask(CacheDataView *dataView)
      : data_view_(dataView) {}

    ~MLTask() {
      if (data_view_ != nullptr) {
        delete data_view_;
      }
    };

    virtual MLAlgorithm getType() = 0;

    /**
     * This method will be executed by a threadpool.
     *
     * @param thread_id The ID of the thread which is executing the task.
     * @param ml_state  Some associated state relevent to the computation.
     */
    virtual void execute(int thread_id, void* ml_state) = 0;

  protected:
    CacheDataView *data_view_;

    DISABLE_COPY_AND_ASSIGN(MLTask);
  };
}



#endif //OBAMADB_MLTASK_H
