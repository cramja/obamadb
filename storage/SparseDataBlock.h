#ifndef OBAMADB_SPARSEDATABLOCK_H
#define OBAMADB_SPARSEDATABLOCK_H

#include "storage/DataBlock.h"
#include "storage/StorageConstants.h"
#include "storage/Utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>

#include <glog/logging.h>
#include <iomanip>

namespace obamadb {

  template<class T> class SparseDataBlock;

  template <typename T>
  std::ostream &operator<<(std::ostream &os, const SparseDataBlock<T> &block);

  std::ostream &operator<<(std::ostream &os, const SparseDataBlock<float_t> &block);


  /**
   * Optimized for storing rows of data where the majority of elements are null.
   */
  template<class T>
  class SparseDataBlock : public DataBlock<T> {
    struct SDBEntry {
      std::uint32_t offset_;
      std::uint32_t size_;
    };

  public:
    SparseDataBlock();

    /**
     * Use this function while initializing to pack the block.
     * @param row Row to append.
     * @return True if the append succeeded, false if the block is full.
     */
    bool appendRow(const se_vector<T> &row);

    /**
     * Blocks which have been finalized no longer can have rows appended to them.
     */
    void finalize() {
      initializing_ = false;
    }

    DataBlockType getDataBlockType() const override {
      return DataBlockType::kSparse;
    }

    void getRowVector(int row, e_vector<T> *vec) const override;

    inline void getRowVectorFast(const int row, se_vector<T> *vec) const {
//      DCHECK_LT(row, this->num_rows_) << "Row index out of range.";
//      DCHECK(dynamic_cast<se_vector<float_t> *>(vec) != nullptr);

      SDBEntry const &entry = entries_[row];
      vec->num_elements_ = entry.size_;
      vec->index_ = reinterpret_cast<int*>(end_of_block_ - entry.offset_);
      vec->values_ = reinterpret_cast<T*>(vec->index_ + entry.size_);
      vec->class_ = vec->values_ + entry.size_;
    }

    /**
     * @return  The value stored at a particular index.
     */
    T* get(unsigned row, unsigned col) const override;

    T* operator()(unsigned row, unsigned col) override;

  private:
    /**
     * @return The number of bytes remaining in the heap.
     */
    inline unsigned remainingSpaceBytes() const;


    bool initializing_;
    SDBEntry *entries_;
    unsigned heap_offset_; // the heap grows backwards from the end of the block.
    // The end of last entry offset_ bytes from the end of the structure.
    char *end_of_block_;

    template<class A>
    friend std::ostream &operator<<(std::ostream &os, const SparseDataBlock<A> &block);

    friend std::ostream &operator<<(std::ostream &os, const SparseDataBlock<float_t> &block);
  };

  template<class T>
  SparseDataBlock<T>::SparseDataBlock()
    : DataBlock<T>(0),
      initializing_(true),
      entries_(reinterpret_cast<SDBEntry *>(this->store_)),
      heap_offset_(0),
      end_of_block_(reinterpret_cast<char *>(this->store_) + kStorageBlockSize) {}


  template<class T>
  std::ostream &operator<<(std::ostream &os, const SparseDataBlock<T> &block) {
    os << "SparseDataBlock[" << block.getNumRows() << ", " << block.getNumColumns() << "]" << std::endl;
  }

  template<class T>
  T* SparseDataBlock<T>::operator()(unsigned row, unsigned col) {
    return get(row, col);
  }

  template<class T>
  bool SparseDataBlock<T>::appendRow(const se_vector<T> &row) {
    DCHECK(initializing_);

    if (remainingSpaceBytes() < (sizeof(SDBEntry) + row.sizeBytes())) {
      return false;
    }

    heap_offset_ += row.sizeBytes();
    SDBEntry *entry = entries_ + this->num_rows_;
    entry->offset_ = heap_offset_;
    entry->size_ = row.numElements();
    row.copyTo(end_of_block_ - heap_offset_);

    this->num_rows_++;
    this->num_columns_ = this->num_columns_ >= row.size() ? this->num_columns_ : row.size();
    return true;
  }

  template<class T>
  void SparseDataBlock<T>::getRowVector(const int row, e_vector<T> *vec) const {
    DCHECK_LT(row, this->num_rows_) << "Row index out of range.";
    DCHECK(dynamic_cast<se_vector<float_t> *>(vec) != nullptr);

    SDBEntry const &entry = entries_[row];
    vec->setMemory(entry.size_, end_of_block_ - entry.offset_);
  }



  template<class T>
  T* SparseDataBlock<T>::get(unsigned row, unsigned col) const {
    DCHECK_LT(row, this->num_rows_) << "Row index out of range.";
    DCHECK_LT(col, this->num_columns_) << "Column index out of range.";

    SDBEntry const &entry = entries_[row];
    se_vector<T> vec(entry.size_, end_of_block_ - entry.offset_);
    T* value = vec.get(col);
    if (value == nullptr) {
      return 0;
    } else {
      return value;
    }
  }

  template<class T>
  unsigned SparseDataBlock<T>::remainingSpaceBytes() const {
    return kStorageBlockSize - (heap_offset_ + sizeof(SDBEntry) * this->num_rows_);
  }

  /**
   * Get the maximum number of columns in a set of blocks.
   * @param blocks
   * @return
   */
  template<class T>
  int maxColumns(std::vector<SparseDataBlock<T>*> blocks) {
    int max_columns = 0;
    for (int i = 0; i < blocks.size(); ++i) {
      if (max_columns < blocks[i]->getNumColumns()) {
        max_columns = blocks[i]->getNumColumns();
      }
    }
    return max_columns;
  }
}

#endif //OBAMADB_SPARSEDATABLOCK_H