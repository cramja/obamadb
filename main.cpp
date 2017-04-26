#include "storage/DataBlock.h"
#include "storage/DataView.h"
#include "storage/IO.h"
#include "storage/Matrix.h"
#include "storage/MCTask.h"
#include "storage/MLTask.h"
#include "storage/SVMTask.h"

#include <algorithm>
#include <gflags/gflags.h>
#include <string>
#include <unistd.h>
#include <vector>

static bool ValidateThreads(const char* flagname, std::int64_t value) {
  int const max = 256;
  if (value > 0 && value <= max) {
    return true;
  }
  printf("The number of threads should be between 0 and %d\n", max);
  return false;
}
DEFINE_int64(threads, 1, "The number of threads the system will use to run the machine learning algorithm");
DEFINE_validator(threads, &ValidateThreads);

DEFINE_bool(measure_convergence, false, "If true, an observer thread will collect copies of the model"
  " as the algorithm does its first iteration. Useful for the SVM.");

static bool ValidateAlgorithm(const char* flagname, std::string const & value) {
  std::vector<std::string> valid_algorithms = {"svm", "mc"};
  if (std::find(valid_algorithms.begin(), valid_algorithms.end(), value) != valid_algorithms.end()) {
    return true;
  } else {
    printf("Invalid algorithm choice. Choices are:\n");
    for (std::string& alg : valid_algorithms) {
      printf("\t%s\n", alg.c_str());
    }
    return false;
  }
}
DEFINE_string(algorithm, "svm", "The machine learning algorithm to use. Select one of [svm, mc].");
DEFINE_validator(algorithm, &ValidateAlgorithm);

DEFINE_string(train_file, "", "The TSV format file to train the algorithm over.");
DEFINE_string(test_file, "", "The TSV format file to test the algorithm over.");

DEFINE_bool(verbose, false, "Print out extra diagnostic information.");

DEFINE_int64(num_epochs, 10, "The number of passes over the training data while training the model.");

DEFINE_int64(num_trials, 1, "The number of trials to perform."
  "This means the number of times we will train a model."
  "This is useful for computing the variance/stddev between trials.");
DEFINE_string(core_affinities, "-1", "A comma separated list of cores to have threads bind to."
  " The program will greedily use core, so over specify if you like. Ex: "
  " -core_affinities 0,1,2,3 -threads 2 is valid");

DEFINE_int64(rank, 10, "The rank of the LR factoring matrices used in Matrix Completion");


#define VPRINT(str) { if(FLAGS_verbose) { printf(str); } }
#define VPRINTF(str, ...) { if(FLAGS_verbose) { printf(str, __VA_ARGS__); } }
#define VSTREAM(obj) {if(FLAGS_verbose){ std::cout << obj <<std::endl; }}

namespace obamadb {

  /**
   * Used to snoop on the inter-epoch values of the model.
   */
  class ConvergenceObserver {
  public:
    ConvergenceObserver(fvector const * const sharedTheta) :
      model_ref_(sharedTheta),
      observedTimes_(),
      observedModels_(),
      cyclesObserved_(0),
      threadPool_(nullptr) {}

    void record() {
      auto current_time = std::chrono::steady_clock::now();
      observedTimes_.push_back(current_time.time_since_epoch().count());
      observedModels_.push_back(*model_ref_);
    }

    int size() {
      CHECK_EQ(observedModels_.size(), observedTimes_.size());
      return observedModels_.size();
    }

    static int kObserverWaitTimeUS; // Time between observation captures.

    fvector const * const model_ref_;
    std::vector<std::uint64_t> observedTimes_;
    std::vector<fvector> observedModels_;
    int cyclesObserved_;
    ThreadPool * threadPool_;
  };

  int ConvergenceObserver::kObserverWaitTimeUS = 1000;

  /**
   * The function which a thread will call once per epoch to record the
   * model's state change over time.
   * @param tid Thread id. Not used.
   * @param observerState Pair<remaining epochs to observe, observer>.
   */
  void observerThreadFn(int tid, void* observerState) {
    (void) tid;
    ConvergenceObserver* observer =
      reinterpret_cast<ConvergenceObserver*>(observerState);

    if (observer->cyclesObserved_ > 0) {
      return;
    }

    // Loop until the number of waiters is the number of worker threads.
    int const total_workers = observer->threadPool_->getNumWorkers();
    while (total_workers > observer->threadPool_->getWaiterCount()) {
      observer->record();
      usleep(ConvergenceObserver::kObserverWaitTimeUS);
    }
    observer->cyclesObserved_++;
  }

  /**
   * Allocates Datablocks to DataViews. Dataviews will then be given to threads in the form of tasks.
   * @param num_threads How many dataviews to allocate and distribute amongst.
   * @param data_blocks The set of training data.
   * @return Vector of Dataviews.
   */
  template<class T>
  void allocateBlocks(const int num_threads,
                      const std::vector<SparseDataBlock<T> *> &data_blocks,
                      std::vector<std::unique_ptr<DataView>>& views) {
    CHECK(views.size() == 0) << "Only accepts empty view vectors";
    CHECK_GE(data_blocks.size(), views.size())
      << "Partitioned data would not distribute to all threads."
      << " Use fewer threads.";

    for (int i = 0; i < data_blocks.size(); i++) {
      if (i < num_threads) {
        views.push_back(std::unique_ptr<DataView>(new DataView()));
      }
      SparseDataBlock<T> const *dbptr = data_blocks[i];
      views[i % num_threads]->appendBlock(dbptr);
    }
  }

  void printSVMEpochStats(Matrix const * matTrain,
                        Matrix const * matTest,
                        fvector const & theta,
                        int iteration,
                        float timeTrain) {

    if (!FLAGS_verbose) {
      return;
    }

    double const trainRmsLoss = SVMTask::rmsErrorLoss(theta, matTrain->blocks_);
    double const testRmsLoss = SVMTask::rmsErrorLoss(theta, matTest->blocks_);
    double const trainFractionMisclassified = SVMTask::fractionMisclassified(theta,matTrain->blocks_);
    double const testFractionMisclassified = SVMTask::fractionMisclassified(theta,matTest->blocks_);

    printf("%-3d, %.3f, %.4f, %.2f, %.4f, %.2f\n",
           iteration,
           timeTrain,
           trainFractionMisclassified,
           trainRmsLoss,
           testFractionMisclassified,
           testRmsLoss);
  }

  /**
   * @return A vector of the epoch times.
   */
  std::vector<double> trainSVM(Matrix *mat_train,
                               Matrix *mat_test) {
    SVMParams* svm_params = DefaultSVMParams<num_t>(mat_train->blocks_);
    DCHECK_EQ(svm_params->degrees.size(), maxColumns(mat_train->blocks_));
    fvector sharedTheta = fvector::GetRandomFVector(mat_train->numColumns_);

    // Arguments to the thread pool.
    std::vector<void*> threadStates;
    std::vector<std::function<void(int, void*)>> threadFns;

    // Create an observer thread, if specified
    std::unique_ptr<ConvergenceObserver> observer;
    if (FLAGS_measure_convergence) {
      observer.reset(new ConvergenceObserver(&sharedTheta));
      auto observer_fn = observerThreadFn;
      threadFns.push_back(observer_fn);
      threadStates.push_back(observer.get());
    }

    // Create the tasks for the thread pool.
    // Roughly allocates work.
    std::vector<std::unique_ptr<DataView>> data_views;

    allocateBlocks(FLAGS_threads, mat_train->blocks_, data_views);
    // Create tasks
    auto update_fn = [](int tid, void* state) {
      SVMTask* task = reinterpret_cast<SVMTask*>(state);
      task->execute(tid, nullptr);
    };
    std::vector<std::unique_ptr<SVMTask>> tasks(FLAGS_threads);
    for (int i = 0; i < tasks.size(); i++) {
      tasks[i].reset(new SVMTask(data_views[i].release(), &sharedTheta, svm_params));
      threadStates.push_back(tasks[i].get());
      threadFns.push_back(update_fn);
    }

    ThreadPool tp(threadFns, threadStates);

    // If we are observing convergence, the thread pool must be referenced.
    if (observer) {
      observer->threadPool_ = &tp;
    }

    tp.begin();

    VPRINT("epoch, train_time, train_fraction_misclassified, train_RMS_loss, test_fraction_misclassified, test_RMS_loss\n");
    printSVMEpochStats(mat_train, mat_test, sharedTheta, -1, -1);
    double totalTrainTime = 0.0;
    std::vector<double> epoch_times;
    for (int cycle = 0; cycle < FLAGS_num_epochs; cycle++) {
      auto time_start = std::chrono::steady_clock::now();
      tp.cycle();
      auto time_end = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::milli> time_ms = time_end - time_start;
      double elapsedTimeSec = (time_ms.count())/ 1e3;
      totalTrainTime += elapsedTimeSec;

      printSVMEpochStats(mat_train, mat_test, sharedTheta, cycle, elapsedTimeSec);
      epoch_times.push_back(elapsedTimeSec);
    }
    tp.stop();

    printf("num_threads,avg_train_time,frac_mispredicted_test\n");
    printf(">>>\n%d,%f,%f\n",
           (int)FLAGS_threads,
           totalTrainTime / FLAGS_num_epochs,
           SVMTask::fractionMisclassified(sharedTheta, mat_test->blocks_));

    if (FLAGS_measure_convergence) {
      printf("Convergence Info (%d measures)\n", (int)observer->observedModels_.size());

      for (int obs_idx = 0; obs_idx < observer->observedModels_.size(); obs_idx++) {
        std::uint64_t timeObs = observer->observedTimes_[obs_idx];
        fvector const & thetaObs = observer->observedModels_[obs_idx];
        printf("%d,%llu,%.4f,%.4f\n",
               (int)FLAGS_threads,
               timeObs,
               SVMTask::rmsErrorLoss(thetaObs, mat_test->blocks_),
               SVMTask::fractionMisclassified(thetaObs, mat_test->blocks_));
      }
    }
    return epoch_times;
  }

  void getSvmData(std::unique_ptr<Matrix> &train, std::unique_ptr<Matrix> &test) {
    VPRINT("Reading input files...\n");
    if (FLAGS_test_file.size() == 0) {
      VPRINTF("Loading: %s\n", FLAGS_train_file.c_str());
      std::unique_ptr<Matrix> source_data;
      PRINT_TIMING({source_data.reset(IO::load(FLAGS_train_file));});
      VSTREAM(*source_data);
      VPRINT("Test file not specified, using an 80-20 train-test split of train file\n");
      std::pair<Matrix*, Matrix*> split = source_data->split(0.8);
      train.reset(split.first);
      test.reset(split.second);
    } else {
      VPRINTF("Loading: %s\n", FLAGS_train_file.c_str());
      PRINT_TIMING({train.reset(IO::load(FLAGS_train_file));});
      VPRINTF("Loading: %s\n", FLAGS_test_file.c_str());
      PRINT_TIMING({test.reset(IO::load(FLAGS_test_file));});
    }

    VSTREAM(*train);
    VSTREAM(*test);
  }

  void runSvmExperiment() {
    std::unique_ptr<Matrix> mat_train;
    std::unique_ptr<Matrix> mat_test;
    getSvmData(mat_train, mat_test);

    CHECK_EQ(mat_test->numColumns_, mat_train->numColumns_)
      << "Train and Test matrices had differing number of features.";

    std::vector<double> all_epoch_times;
    for (int i = 0; i < FLAGS_num_trials; i++) {
      std::vector<double> times = trainSVM(mat_train.get(), mat_test.get());
      all_epoch_times.insert(all_epoch_times.end(), times.begin(), times.end());

      if (FLAGS_num_trials != i -1) {
        usleep(1e7);
      }
    }

    // calculate variance, etc.
    if (FLAGS_verbose) {
      printf("epoch runtimes:\n");
      for (double time : all_epoch_times) {
        printf("%f\n", time);
      }
    }

    printf("epoch runtime summary:\nmean,variance,stddev,stderr\n%f,%f,%f,%f\n",
           stats::mean<double>(all_epoch_times),
           stats::variance<double>(all_epoch_times),
           stats::stddev<double>(all_epoch_times),
           stats::stderr<double>(all_epoch_times));
  }

  void printMCEpochStats(int epoch, double time, MCState const * state, UnorderedMatrix const * probe_mat) {
    if (FLAGS_verbose) {
      double rmse = MCTask::rmse(state, probe_mat);
      printf("%d,%.6f,%.4f\n",epoch, time, rmse);
    }
  }

  std::vector<double> trainMC(const UnorderedMatrix* train_matrix,
                              const UnorderedMatrix* probe_matrix) {
    int const rank = FLAGS_rank;
    MCState* mcstate = new MCState(train_matrix, rank); // TODO: this is a memory leak which
		// fixes a mysterious memory corruption bug on ubuntu systems. It should be a smart pointer
		// and also, there should be no memory corruption
    if (FLAGS_verbose) {
      printf("Model matrix properties (L,R):\n");
      std::cout << *mcstate->mat_l << std::endl;
      std::cout << *mcstate->mat_r << std::endl;
    }

    // Arguments to the thread pool.
    std::vector<std::function<void(int, void*)>> threadFns;
    auto update_fn = [](int tid, void* state) {
      MCTask* task = reinterpret_cast<MCTask*>(state);
      task->execute(tid, nullptr);
    };
    std::vector<std::unique_ptr<MCTask>> tasks(FLAGS_threads);
    std::vector<void*> tp_states;
    for (int i = 0; i < tasks.size(); i++) {
      tasks[i].reset(new MCTask(FLAGS_threads, train_matrix, mcstate));
      tp_states.push_back(tasks[i].get());
      threadFns.push_back(update_fn);
    }

    ThreadPool tp(threadFns, tp_states);
    tp.begin();

    VPRINT("epoch, train_time, probe_RMS_loss\n");
    printMCEpochStats(-1, -1, mcstate, probe_matrix);
    double totalTrainTime = 0.0;
    std::vector<double> epoch_times;
    for (int cycle = 0; cycle < FLAGS_num_epochs; cycle++) {
      auto time_start = std::chrono::steady_clock::now();
      tp.cycle();
      auto time_end = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::milli> time_ms = time_end - time_start;
      double elapsedTimeSec = (time_ms.count())/ 1e3;
      totalTrainTime += elapsedTimeSec;

      printMCEpochStats(cycle, elapsedTimeSec, mcstate, probe_matrix);
      epoch_times.push_back(elapsedTimeSec);
    }
    tp.stop();
    return epoch_times;
  }

  void trainMC() {
    std::unique_ptr<UnorderedMatrix> train_matrix;
    std::unique_ptr<UnorderedMatrix> probe_matrix;

    VPRINT("Reading input files...\n");
    VPRINTF("Loading: %s\n", FLAGS_train_file.c_str());
    PRINT_TIMING({train_matrix.reset(IO::loadUnorderedMatrix(FLAGS_train_file));});
    VSTREAM(*train_matrix);

    VPRINTF("Loading: %s\n", FLAGS_test_file.c_str());
    PRINT_TIMING({probe_matrix.reset(IO::loadUnorderedMatrix(FLAGS_test_file));});
    VSTREAM(*probe_matrix);

    CHECK_LE(probe_matrix->numColumns(), train_matrix->numColumns());
    CHECK_LE(probe_matrix->numRows(), train_matrix->numRows());

    std::vector<double> all_epoch_times;
    for (int i = 0; i < FLAGS_num_trials; i++) {
      std::vector<double> times = trainMC(train_matrix.get(), probe_matrix.get());
      all_epoch_times.insert(all_epoch_times.end(), times.begin(), times.end());

      if (FLAGS_num_trials != i - 1) {
        usleep(1e7);
      }
    }

    // calculate variance, etc.
    if (FLAGS_verbose) {
      printf("epoch runtimes:\n");
      for (double time : all_epoch_times) {
        printf("%f\n", time);
      }
    }

    printf("epoch runtime summary:\nmean,variance,stddev,stderr\n%f,%f,%f,%f\n",
           stats::mean<double>(all_epoch_times),
           stats::variance<double>(all_epoch_times),
           stats::stddev<double>(all_epoch_times),
           stats::stderr<double>(all_epoch_times));
  }

  int main(int argc, char** argv) {
    ::google::InitGoogleLogging(argv[0]);
    ::gflags::SetUsageMessage(std::string(argv[0]) + " -help");
    ::gflags::SetVersionString("0.0");
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::vector<int> affinities = GetIntList(FLAGS_core_affinities);
    if (affinities[0] != -1) {
      threading::setCoreAffinity(affinities[0]);
    } else {
      LOG(INFO) << "Main thread affinitized to core 0";
      threading::setCoreAffinity(0);
    }

    if (FLAGS_algorithm.compare("svm") == 0) {
      runSvmExperiment();
    } else if (FLAGS_algorithm.compare("mc") == 0) {
      LOG_IF(WARNING, FLAGS_measure_convergence)
        << "Measure convergence not implemented for MC";
      trainMC();
    } else {
      LOG(FATAL) << "unknown training algorithm";
    }

    return 0;
  }
} // namespace obamadb

int main(int argc, char **argv) {
  return obamadb::main(argc, argv);
}
