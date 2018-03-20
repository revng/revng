#ifndef _STATISTICS_H
#define _STATISTICS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <csignal>
#include <cstdlib>
extern "C" {
#include <strings.h>
}
#include <vector>

// LLVM includes
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ManagedStatic.h"

// Local includes
#include "debug.h"

class OnQuitInteraface {
public:
  virtual void onQuit() = 0;
  virtual ~OnQuitInteraface();
};

/// \brief Collect mean and variance about a certain event.
///
/// To use this class, simply create a global variable, call push with the value
/// you want to record, when you're done use the various methods to obtain mean
/// and variance.
///
/// If a name is provided, the results will be registered for printing at
/// program termination.
class RunningStatistics : public OnQuitInteraface {
public:

  RunningStatistics() : RunningStatistics(llvm::Twine(), false) { }

  RunningStatistics(llvm::Twine Name) : RunningStatistics(Name, true) { }

  /// \arg Name the name to use when printing the statistics.
  /// \arg Register whether this object should be registered for being printed
  ///      upon program termination or not.
  RunningStatistics(llvm::Twine Name, bool Register) : Name(Name), N(0) {
    if (Register)
      init();
  }

  virtual ~RunningStatistics() { }

  void clear() { N = 0; }

  /// \brief Record a new value
  void push(double X) {
    N++;

    // See Knuth TAOCP vol 2, 3rd edition, page 232
    if (N == 1) {
      OldM = NewM = X;
      OldS = 0.0;
    } else {
      NewM = OldM + (X - OldM) / N;
      NewS = OldS + (X - OldM) * (X - NewM);

      // Set up for next iteration
      OldM = NewM;
      OldS = NewS;
    }
  }

  /// \return the total number of recorded values.
  int size() const { return N; }

  double mean() const {
    return (N > 0) ? NewM : 0.0;
  }

  double variance() const {
    return ((N > 1) ? NewS / (N - 1) : 0.0);
  }

  double standardDeviation() const {
    return sqrt(variance());
  }

  template<typename T>
  void dump(T &Output) {
    if (!Name.isTriviallyEmpty())
      Output << Name.str() << ": ";
    Output << "{ n: " << size()
           << " u: " << mean()
           << " o: " << variance()
           << " }";
  }

  void dump() { dump(dbg); }

  virtual void onQuit();

private:
  void init();

private:
  llvm::Twine Name;
  int N;
  double OldM, NewM, OldS, NewS;
};

// TODO: this is duplicated
template<typename T, typename... Args>
inline std::array<T, sizeof...(Args)>
make_array(Args&&... args) {
  return { { std::forward<Args>(args)... } };
}

class OnQuitRegistry {
public:
  void install();

  /// \brief Registers an object for having its onQuit method called upon
  ///        program termination
  void add(OnQuitInteraface *S) { Register.push_back(S); }

  void dump() {
    for (OnQuitInteraface *S : Register)
      S->onQuit();
  }

private:
  std::vector<OnQuitInteraface *> Register;
};

extern llvm::ManagedStatic<OnQuitRegistry> OnQuitStatistics;

inline void RunningStatistics::init() {
  OnQuitStatistics->add(this);
}

#endif // _STATISTICS_H
