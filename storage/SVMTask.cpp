#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/exvector.h"
#include "storage/MLTask.h"
#include "storage/SparseDataBlock.h"
#include "storage/Utils.h"

#include "storage/SVMTask.h"

// comment this out depending on the test you are doing:
// #define USE_HINGE 0
// #define USE_SCALING 0

namespace obamadb {

  void SVMTask::execute(int threadId, void *svm_state) {
    (void) svm_state; // silence compiler warning.

    data_view_->reset();
    svector<num_t> row(0, nullptr);
    num_t *theta = shared_theta_->values_;
    const num_t mu = shared_params_->mu;
    const num_t step_size = shared_params_->step_size;

    // perform update with all the data in its view,
    while (data_view_->getNext(&row)) {
      num_t const y = *row.getClassification();
      num_t wxy = ml::dot(row, theta);
      wxy = wxy * y; // {-1, 1}

#ifdef USE_HINGE
      // apply the hinge function like in a normal SVM
      if (wxy < 1) {
        num_t const e = step_size * y;
        // scale weights
        ml::scaleAndAdd(theta, row, e);
      }
#else
      // always apply the hinge loss, for memory-access
      if (wxy < 1) {
        num_t const e = step_size * y;
        // scale weights
        ml::scale_and_add(theta, row, e);
      } else {
        num_t const e = step_size * y * -1 * 1e-3;
        // scale weights
        ml::scale_and_add(theta, row, e);
      }
#endif

#ifdef USE_SCALING
      num_t const scalar = step_size * mu;
      // scale only the values which were updated.
      for (int i = row.numElements(); i-- > 0;) {
        const int idx_j = row.index_[i];
        num_t const deg = shared_params_->degrees[idx_j];
        theta[idx_j] *= 1 - scalar / deg;
      }
#endif
    }

    if (threadId == 0) {
      shared_params_->step_size = step_size * shared_params_->step_decay;
    }
  }

  int SVMTask::numMisclassified(const fvector &theta, const SparseDataBlock<num_t> &block) {
    svector<num_t> row(0, nullptr);
    int misclassified = 0;
    for (int i = 0; i < block.getNumRows(); i++) {
      block.getRowVectorFast(i, &row);
      const num_t dot_prod = ml::dot(row, theta.values_);
      const num_t classification = *row.getClassification();
      DCHECK(classification == 1 || classification == -1) << "Expected binary classification.";

      misclassified += (classification == 1 && dot_prod < 0) || (classification == -1 && dot_prod >= 0);
    }
    return misclassified;
  }

  double SVMTask::fractionMisclassified(const fvector &theta, std::vector<SparseDataBlock<num_t> *> const &blocks) {
    long total_misclassified = 0;
    long total_examples = 0;
    for (int i = 0; i < blocks.size(); i++) {
      SparseDataBlock<num_t> const *block = blocks[i];
      total_misclassified += SVMTask::numMisclassified(theta, *block);
      total_examples += block->getNumRows();
    }
    return (double) total_misclassified / (double) total_examples;
  }

  double SVMTask::rmsError(const fvector &theta, std::vector<SparseDataBlock<num_t> *> const &blocks) {
    return std::sqrt(SVMTask::fractionMisclassified(theta, blocks));
  }

  double SVMTask::rmsErrorLoss(const fvector &theta, std::vector<SparseDataBlock<num_t> *> const &blocks) {
    double total_examples = 0;
    double loss = 0;
    svector<num_t> row(0, nullptr);
    for (int i = 0; i < blocks.size(); i++) {
      SparseDataBlock<num_t> const &block = *blocks[i];

      for (int i = 0; i < block.getNumRows(); i++) {
        block.getRowVector(i, &row);
        const num_t dot_prod = ml::dot(row, theta.values_);
        const num_t classification = *row.getClassification();
        DCHECK(classification == 1 || classification == -1);
        loss += std::max(1 - dot_prod * classification, static_cast<num_t >(0.0));
      }
      total_examples += block.getNumRows();
    }
    return std::sqrt(loss) / std::sqrt(total_examples);
  }

  // Aligned //

  void AlignedSVMTask::execute(int threadId, void *svm_state) {
    (void) svm_state; // silence compiler warning.

    data_view_->reset();
    svector<num_t> row(0, nullptr);
    aligned_fvector::aligned_num_t *theta = shared_theta_->values_;
    const num_t mu = shared_params_->mu;
    const num_t step_size = shared_params_->step_size;

    // perform update with all the data in its view,
    while (data_view_->getNext(&row)) {
      num_t const y = *row.getClassification();
      num_t wxy = ml::dot(row, theta);
      wxy = wxy * y; // {-1, 1}

#ifdef USE_HINGE
      // apply the hinge function like in a normal SVM
      if (wxy < 1) {
        num_t const e = step_size * y;
        // scale weights
        ml::scaleAndAdd(theta, row, e);
      }
#else
      // always apply the hinge loss, for memory-access
      if (wxy < 1) {
        num_t const e = step_size * y;
        // scale weights
        ml::scale_and_add(theta, row, e);
      } else {
        num_t const e = step_size * y * -1 * 1e-3;
        // scale weights
        ml::scale_and_add(theta, row, e);
      }
#endif

#ifdef USE_SCALING
      num_t const scalar = step_size * mu;
      // scale only the values which were updated.
      for (int i = row.numElements(); i-- > 0;) {
        const int idx_j = row.index_[i];
        num_t const deg = shared_params_->degrees[idx_j];
        theta[idx_j] *= 1 - scalar / deg;
      }
#endif
    }

    if (threadId == 0) {
      shared_params_->step_size = step_size * shared_params_->step_decay;
    }
  }

  int AlignedSVMTask::numMisclassified(const aligned_fvector &theta, const SparseDataBlock<num_t> &block) {
    svector<num_t> row(0, nullptr);
    int misclassified = 0;
    for (int i = 0; i < block.getNumRows(); i++) {
      block.getRowVectorFast(i, &row);
      const num_t dot_prod = ml::dot(row, theta.values_);
      const num_t classification = *row.getClassification();
      DCHECK(classification == 1 || classification == -1) << "Expected binary classification.";

      misclassified += (classification == 1 && dot_prod < 0) || (classification == -1 && dot_prod >= 0);
    }
    return misclassified;
  }

  double AlignedSVMTask::fractionMisclassified(const aligned_fvector &theta, std::vector<SparseDataBlock<num_t> *> const &blocks) {
    long total_misclassified = 0;
    long total_examples = 0;
    for (int i = 0; i < blocks.size(); i++) {
      SparseDataBlock<num_t> const *block = blocks[i];
      total_misclassified += AlignedSVMTask::numMisclassified(theta, *block);
      total_examples += block->getNumRows();
    }
    return (double) total_misclassified / (double) total_examples;
  }

  double AlignedSVMTask::rmsError(const aligned_fvector &theta, std::vector<SparseDataBlock<num_t> *> const &blocks) {
    return std::sqrt(AlignedSVMTask::fractionMisclassified(theta, blocks));
  }

  double AlignedSVMTask::rmsErrorLoss(const aligned_fvector &theta, std::vector<SparseDataBlock<num_t> *> const &blocks) {
    double total_examples = 0;
    double loss = 0;
    svector<num_t> row(0, nullptr);
    for (int i = 0; i < blocks.size(); i++) {
      SparseDataBlock<num_t> const &block = *blocks[i];

      for (int i = 0; i < block.getNumRows(); i++) {
        block.getRowVector(i, &row);
        const num_t dot_prod = ml::dot(row, theta.values_);
        const num_t classification = *row.getClassification();
        DCHECK(classification == 1 || classification == -1);
        loss += std::max(1 - dot_prod * classification, static_cast<num_t >(0.0));
      }
      total_examples += block.getNumRows();
    }
    return std::sqrt(loss) / std::sqrt(total_examples);
  }

} // namespace obamadb

