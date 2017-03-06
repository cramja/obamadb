#ifndef OBAMADB_DATAVIEW_H_
#define OBAMADB_DATAVIEW_H_

#include "storage/DataBlock.h"
#include "storage/exvector.h"
#include "storage/SparseDataBlock.h"
#include "storage/StorageConstants.h"

#include <algorithm>
#include <vector>
#include <random>

namespace obamadb {

  static int minibatch_decay = 3;
  static int minibatch_round = 0;
  static int minibatch_rounds = 4;

  class DataView {
  public:
    DataView(std::vector<SparseDataBlock<num_t> const *> blocks)
      : blocks_(blocks), current_block_(0),current_idx_(0) {}

    DataView() : blocks_(), current_block_(0), current_idx_(0) {}

    virtual bool getNext(svector<num_t> * row) {
      if (current_idx_ < blocks_[current_block_]->num_rows_) {
        blocks_[current_block_]->getRowVectorFast(current_idx_++, row);
        return true;
      } else if (current_block_ < blocks_.size() - 1) {
        current_block_++;
        current_idx_ = 0;
        return getNext(row);
      }

      return false;
    }

    virtual void appendBlock(SparseDataBlock<num_t> const * block) {
      blocks_.push_back(block);
    }

    void clear() {
      blocks_.clear();
    }

    virtual void reset() {
      current_block_ = 0;
      current_idx_ = 0;
    }

  protected:

    std::vector<SparseDataBlock<num_t> const *> blocks_;
    int current_block_;
    int current_idx_;
  };

  class CacheDataView : public DataView {
  public:
    CacheDataView(std::vector<SparseDataBlock<num_t> const *> blocks)
      : DataView(blocks), idx_order(), repeat(minibatch_rounds), round(0) {
      init_idx_shuffle();
    }

    CacheDataView() : DataView(), idx_order(), repeat(minibatch_rounds), round(0) { }

    bool getNext(svector<num_t> * row) override {
      std::vector<int> const & idx_perm = idx_order[current_block_];

      if (current_idx_ < idx_perm.size()) {
        blocks_[current_block_]->getRowVectorFast(idx_perm[current_idx_++], row);
        return true;
      } else if (round < minibatch_rounds) {
        round++;
        reshuffle(current_block_);
        current_idx_ = 0;
        return getNext(row);
      } else if (current_block_ < blocks_.size() - 1) {
        round = 0;
        current_block_++;
        current_idx_ = 0;
        return getNext(row);
      }

      return false;
    }

    void reset() override {
      current_block_ = 0;
      current_idx_ = 0;
      round = 0;
      if (minibatch_decay > 0) {
        minibatch_round ++;
        if (minibatch_round % minibatch_decay == 0) {
          minibatch_rounds -= 1;
        }
      }
    }

    virtual void appendBlock(SparseDataBlock<num_t> const * block) {
      blocks_.push_back(block);
      std::vector<int> idx;
      for (int j = 0; j < block->num_rows_; j++) {
        idx.push_back(j);
      }
      idx_order.push_back(idx);
    }

    void init_idx_shuffle() {
      for (int i = 0; i < blocks_.size(); i ++) {
        std::vector<int> idx;
        for (int j = 0; j < blocks_[i]->num_rows_; j++) {
          idx.push_back(j);
        }
        idx_order.push_back(idx);
      }
    }

    void reshuffle(int block_idx) {
      std::vector<int> & block = idx_order[block_idx];
      std::shuffle(block.begin(), block.end(), std::default_random_engine());
    }

  protected:

    std::vector<std::vector<int>> idx_order;
    int repeat;
    int round;

  };
}

#endif //OBAMADB_DATAVIEW_H_
