#include "storage/LinearMath.h"

#include "storage/DataBlock.h"

#include "glog/logging.h"
#include "LinearMath.h"
#include "Task.h"

namespace obamadb {

  double error(DenseDataBlock *const A, DenseDataBlock *const y, double const *theta) {
    DCHECK_EQ(A->getNumRows(), y->getNumRows());

    const unsigned N = y->getNumRows();
    const unsigned n_theta = A->getNumColumns();
    double *a_cursor = A->getStore();
    double *y_cursor = y->getStore();
    long double e_sum = 0;
    for (unsigned i = 0; i < N; ++i) {
      long double a_theta = 0;
      for (unsigned j = 0; j < n_theta; ++j) {
        a_theta += theta[j] * *a_cursor;
        a_cursor += 1;
      }
      // Round
      if (a_theta < 0.0) {
        a_theta = -1.0;
      } else {
        a_theta = 1.0;
      }
      long double r = *y_cursor - a_theta;
      e_sum += r * r;
      y_cursor++;
    }
    return e_sum / N;
  }

  double rowDot(double const *row_a, double const *row_b, unsigned row_dimension) {

    double sum = 0.0;
    for (unsigned col = 0; col < row_dimension; ++col) {
      sum += row_a[col] * row_b[col];
    }
    return sum;
  }

  void rowGradient(double const *training_example, double y, double *theta, unsigned width, double num_training_examples) {

    double residual = y - rowDot(training_example, theta, width);
    double train_factor = (alpha * 2.0) / num_training_examples;
    for (unsigned col = 0; col < width; ++col) {
      theta[col] += train_factor * residual * training_example[col];
    }
  }

  void gradientItr(
    DenseDataBlock const *A,
    DenseDataBlock const *y,
    double *theta) {
    for (unsigned row = 0; row < A->getNumRows(); ++row) {
      rowGradient(A->getRow(row), y->get(row, 0), theta, A->getNumColumns(), A->getNumRows());
    }
  }

  double distance(double const *p1, double const *p2, unsigned dimension) {
    double sq_diff_sum = 0;
    for (unsigned i = 0; i < dimension; ++i) {
      sq_diff_sum += std::pow(p1[i] - p2[i], 2);
    }
    return std::sqrt(sq_diff_sum);
  }
}