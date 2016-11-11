#ifndef OBAMADB_MATRIX_H
#define OBAMADB_MATRIX_H

#include "storage/SparseDataBlock.h"
#include "storage/StorageConstants.h"

#include <vector>

namespace obamadb {

  namespace {

    inline float_t sparseDot(const se_vector<float_t> & a, const se_vector<signed char> & b) {
      int ai = 0, bi =0;
      float_t sum_prod = 0;
      while(ai < a.num_elements_ && bi < b.num_elements_) {
        if(a.index_[ai] == b.index_[bi]) {
          sum_prod += a.values_[ai] * b.values_[bi];
          ai++; bi++;
        } else if (a.index_[ai] < b.index_[bi]) {
          ai++;
        } else {
          bi++;
        }
      }
      return sum_prod;
    }
  }

  class Matrix {
  public:
    /**
     * Takes ownership of the passed DataBlocks.
     */
    Matrix(const std::vector<SparseDataBlock<float_t> *> &blocks)
      : numColumns_(0),
        numRows_(0),
        blocks_() {
      for (int i = 0; i < blocks.size(); i++) {
        addBlock(blocks[i]);
      }
    }

    /**
     * Takes ownership of the passed DataBlocks.
     */
    Matrix()
      : numColumns_(0),
        numRows_(0),
        blocks_() {}

    Matrix(const Matrix& other) = delete;

    Matrix& operator=(const Matrix& other) = delete;

    ~Matrix() {
      for(auto block : blocks_)
        delete block;
    }

    /**
     * Takes ownership of an entire block and adds it to the matrix and increases the size if necessary.
     * @param block The sparse datablock to add.
     */
    void addBlock(SparseDataBlock<float_t> *block) {
      if (block->getNumColumns() > numColumns_) {
        numColumns_ = block->getNumColumns();
        // each block should be the same dimension as the matrix.
        for (int i = 0; i < blocks_.size(); i++) {
          blocks_[i]->num_columns_ = numColumns_;
        }
      }
      numRows_ += block->getNumRows();
      blocks_.push_back(block);
    }

    /**
     * Appends the row to the last block in the matrix's list of datablocks. If it does not fit, a new data
     * block will be created.
     * @param row Row to append
     */
    void addRow(const se_vector<float_t> &row) {
      if(blocks_.size() == 0 || !blocks_.back()->appendRow(row)) {
        blocks_.push_back(new SparseDataBlock<float_t>());
        bool appended = blocks_.back()->appendRow(row);
        DCHECK(appended);
      }

      if (row.size() > numColumns_) {
        numColumns_ = row.size();
        for (int i = 0; i < blocks_.size(); i++) {
          blocks_[i]->num_columns_ = numColumns_;
        }
      }

      numRows_++;
    }

    /**
     * Performs a random projection multiplication on the matrix and returns a new compressed
     * version of the matrix.
     */
    Matrix* randomProjectionsCompress() const {
      const int kCompressionConstant = 0.5 * numColumns_; // TODO: how to choose this number.
      std::unique_ptr<SparseDataBlock<signed char>> projection(
        GetRandomProjectionMatrix(numColumns_, kCompressionConstant));
      Matrix *result = new Matrix();
      se_vector<float_t> row_a;
      row_a.setMemory(0, nullptr);
      se_vector<signed char> row_b;
      row_b.setMemory(0, nullptr);
      int current_block = 0;
      for (int i = 0; i < blocks_.size(); i++) {
        const SparseDataBlock<float_t> *block = blocks_[i];
        for (int j = 0; j < block->getNumRows(); j++) {
          se_vector<float_t> row_c;
          block->getRowVectorFast(j, &row_a);
          for (int k = 0; k < projection->getNumRows(); k++) {
            projection->getRowVectorFast(k, &row_b);
            float_t f = sparseDot(row_a, row_b);
            if (f != 0) {
              row_c.push_back(k, f);
            }
          }
          result->addRow(row_c);
        }
      }
      return result;
    }

    /**
     * @return Fraction of elements which are zero.
     */
    double getSparsity() const {
      std::uint64_t nnz = 0;
      std::uint64_t numElements = static_cast<std::uint64_t >(numColumns_) * static_cast<std::uint64_t >(numRows_);
      for (auto block : blocks_) {
        nnz += block->numNonZeroElements();
      }
      return (double ) (numElements - nnz) / (double) numElements;
    }

    int numColumns_;
    int numRows_;
    std::vector<SparseDataBlock<float_t>*> blocks_;
  };

}

#endif //OBAMADB_MATRIX_H