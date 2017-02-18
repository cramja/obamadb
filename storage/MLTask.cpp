#include "storage/MLTask.h"

namespace obamadb {
  namespace ml {
    /**
     * Dot product
     */
    num_t dot(const dvector <num_t> &v1, num_t const *d2) {
      num_t sum = 0;
      num_t const *__restrict__ pv1 = v1.values_;
      num_t const *__restrict__ pv2 = d2;
      for (int i = 0; i < v1.size(); ++i) {
        sum += pv1[i] * pv2[i];
      }
      return sum;
    }

    /**
     * Aligned Dot product
     */
    num_t dot(const svector <num_t> &v1, aligned_fvector::aligned_num_t const *d2) {
      num_t sum = 0;
      num_t const *const __restrict__ pv1 = v1.values_;
      int const *const __restrict__ pvi1 = v1.index_;
      aligned_fvector::aligned_num_t const *const __restrict__ pv2 = d2;
      for (int i = 0; i < v1.numElements(); ++i) {
        sum += pv1[i] * pv2[pvi1[i]].val;
      }
      return sum;
    }

    /**
     * Applies delta to theta with a constant scaling parameter applied to delta.
     */
    void scale_and_add(aligned_fvector::aligned_num_t *theta, const svector<num_t> &delta, const num_t e) {
      aligned_fvector::aligned_num_t *const __restrict__ tptr = theta;
      num_t const *__restrict__ const vptr = delta.values_;
      int const *__restrict__ const iptr = delta.index_;
      for (int i = 0; i < delta.num_elements_; i++) {
        const int idx = iptr[i];
        tptr[idx].val = tptr[idx].val + (vptr[i] * e);
      }
    }

    void scale(dvector <num_t> &v1, num_t e) {
      int size = v1.size();
      for (int i = 0; i < size; i++) {
        v1.values_[i] *= e;
      }
    }

    void scale_and_add(dvector <num_t> &v1, dvector <num_t> &v2, num_t e) {
      int size = v1.size();
      for (int i = 0; i < size; i++) {
        v1.values_[i] += v2.values_[i] * e;
      }
    }

    /**
     * Dot product
     */
    num_t dot(const svector <num_t> &v1, num_t *d2) {
      num_t sum = 0;
      num_t const *const __restrict__ pv1 = v1.values_;
      int const *const __restrict__ pvi1 = v1.index_;
      num_t const *const __restrict__ pv2 = d2;
      for (int i = 0; i < v1.numElements(); ++i) {
        sum += pv1[i] * pv2[pvi1[i]];
      }
      return sum;
    }

    /**
     * Applies delta to theta with a constant scaling parameter applied to delta.
     * @param theta Sparse vector of weights.
     * @param delta Sparse vector of changes.
     * @param e Scaling constant
     */
    void scale_and_add(num_t *theta, const svector<num_t> &delta, const num_t e) {
      num_t *const __restrict__ tptr = theta;
      num_t const *__restrict__ const vptr = delta.values_;
      int const *__restrict__ const iptr = delta.index_;
      for (int i = 0; i < delta.num_elements_; i++) {
        const int idx = iptr[i];
        tptr[idx] = tptr[idx] + (vptr[i] * e);
      }
    }
  }  // namespace ml

} // namespace obamadb