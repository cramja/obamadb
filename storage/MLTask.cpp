#include "storage/MLTask.h"

#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/exvector.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

#define USE_ATOMIC_ADD 0

namespace obamadb {

  namespace ml {

    /**
     * Returns the sign of the number.
     * Returns zero if input is zero.
     */
    int sign(int i) {
      return i > 0 ? 1 : i < 0 ? -1 : 0;
    }

    /**
     * Dot product
     */
    int_t dot(const dvector<int_t> &v1, int_t *d2) {
      int_t dotsum = 0;
      int_t *__restrict__ pv1 = v1.values_;
      int_t *__restrict__ pv2 = d2;
      for (int i = 0; i < v1.size(); ++i) {
        dotsum += pv1[i] * pv2[i];
      }
      return dotsum;
    }

    /**
     * Dot product
     */
    int_t dot(const svector<int_t> &v1, int_t *d2) {
      int_t dotsum = 0;
      int_t const *const __restrict__ pv1 = v1.values_;
      int const *const __restrict__ pvi1 = v1.index_;
      int_t const *const __restrict__ pv2 = d2;
      int const numElements = v1.numElements();
      for (int i = 0; i < numElements; ++i) {
        dotsum += pv1[i] * pv2[pvi1[i]];

        // uncomment to do overflow checks:
        // overflow checks will not work in multithreaded scenerios.
        //
//        int oldsum = dotsum;
//
//        int signPv1 = sign(pv1[i]);
//        int signPv2 = sign(pv2[pvi1[i]]);
//
//        int prod = pv1[i] * pv2[pvi1[i]];
//
//        CHECK(sign(prod) == signPv1 * signPv2) << "overflow @ multiplication";
//
//        dotsum += prod;
//
//        // same sign or the product was a different sign than the old sum
//        CHECK(sign(oldsum) * sign(dotsum) >= 0 || sign(prod) != sign(oldsum)) << "overflow @ addition";

      }
      return dotsum;
    }

    int numMisclassified(const fvector &theta, const SparseDataBlock<int_t> &block) {
      svector<int_t> row(0, nullptr);
      int misclassified = 0;
      for (int i = 0; i < block.getNumRows(); i++) {
        block.getRowVectorFast(i, &row);
        const int_t dot_prod = dot(row, theta.values_);
        const int_t classification = *row.getClassification();
        DCHECK(classification == 1 || classification == -1) << "Expected binary classification.";

        misclassified += (classification == 1 && dot_prod < 0) || (classification == -1 && dot_prod >= 0);
      }
      return misclassified;
    }

    double fractionMisclassified(const fvector &theta, std::vector<SparseDataBlock<int_t> *> const &blocks) {
      double total_misclassified = 0;
      double total_examples = 0;
      for (int i = 0; i < blocks.size(); i++) {
        SparseDataBlock<int_t> const *block = blocks[i];
        total_misclassified += numMisclassified(theta, *block);
        total_examples += block->getNumRows();
      }
      return total_misclassified / total_examples;
    }

    double rmsError(const fvector &theta, std::vector<SparseDataBlock<int_t> *> const &blocks) {
      return std::sqrt(fractionMisclassified(theta, blocks));
    }

    double rmsErrorLoss(const fvector &theta, std::vector<SparseDataBlock<int_t> *> const &blocks) {
      double total_examples = 0;
      double loss = 0;
      svector<int_t> row(0, nullptr);
      for (int i = 0; i < blocks.size(); i++) {
        SparseDataBlock<int_t> const &block = *blocks[i];

        for (int i = 0; i < block.getNumRows(); i++) {
          block.getRowVector(i, &row);
          const double dot_prod = dot(row, theta.values_) / kScaleFloats;
          const int_t classification = *row.getClassification();
          DCHECK(classification == 1 || classification == -1);
          loss += std::max(1.0 - dot_prod* ((double)classification), 0.0);
        }
        total_examples += block.getNumRows();
      }
      return std::sqrt(loss) / std::sqrt(total_examples);
    }

    /**
     * Applies delta to theta with a constant scaling parameter applied to delta.
     * @param theta Sparse vector of weights.
     * @param delta Sparse vector of changes.
     * @param e Scaling constant
     */
    inline void scaleAndAdd(int_t *theta, const svector<int_t> &delta, const float e) {
      int_t *const __restrict__ tptr = theta;
      int_t const *__restrict__ const vptr = delta.values_;
      int const *__restrict__ const iptr = delta.index_;
      int const numElements = delta.num_elements_;
      for (int i = 0; i < numElements; i++) {
        const int idx = iptr[i];
#ifdef USE_ATOMIC_ADD
        __sync_fetch_and_add(tptr + idx, (vptr[i] * e));
#else
        tptr[idx] = tptr[idx] + (vptr[i] * e);
#endif
      }
    }
  } // namespace ml

  void SVMTask::execute(int threadId, void *svm_state) {
    (void) svm_state; // silence compiler warning.

    data_view_->reset();
    svector<int_t> row(0, nullptr);
    int_t *theta = shared_theta_->values_;
    const float mu = shared_params_->mu;
    const float step_size = shared_params_->step_size;

    // perform update with all the data in its view,
    while (data_view_->getNext(&row)) {
      int_t const y = *row.getClassification();
      int_t wxy = ml::dot(row, theta);
      wxy = wxy * y; // {-1, 1}
      // hinge active
      if (wxy < kScaleFloats * kScaleFloats) {
        float const e = step_size * y;
        // scale weights
        ml::scaleAndAdd(theta, row, e);
      }

      // Is this normalization? It doesn't seem to have an effect on the convergence...
      // With the code the RCV1 per epoch time on my mac is:
      // 0.417699 s/epoch
      // Without:
      // 0.209497 s/epoch
      // Both versions converge to a model with the same accuracy.


      // There's also potential for write conflict here
      // we do not use an atomic update because there is no such atomic instruction
      float const scalar = step_size * mu;
      // scale only the values which were updated.
      for (int i = row.num_elements_; i-- > 0;) {
        const int idx_j = row.index_[i];
        int const deg = shared_params_->degrees[idx_j];
        theta[idx_j] *= 1 - scalar / deg;
      }

    }

    if (threadId == 0) {
      // TODO: this is a small bug where the step size is updated before all the threads finish.
      shared_params_->step_size = step_size * shared_params_->step_decay;
    }
  }

} // namespace obamadb