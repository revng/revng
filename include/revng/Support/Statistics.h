#ifndef STATISTICS_H
#define STATISTICS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <csignal>
#include <cstdlib>
#include <map>
#include <string>
extern "C" {
#include <strings.h>
}
#include <vector>

// LLVM includes
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ManagedStatic.h"

// Local libraries includes
#include "revng/Support/Debug.h"

const size_t MaxCounterMapDump = 32;

class OnQuitInteraface {
public:
  virtual void onQuit() = 0;
  virtual ~OnQuitInteraface();
};

template<typename T>
inline size_t digitsCount(T Value) {
  size_t Digits = 0;

  while (Value != 0) {
    Value /= 10;
    Digits++;
  }

  return Digits;
}

template<typename K, typename T = uint64_t>
class CounterMap : public OnQuitInteraface {
private:
  using Container = std::map<K, T>;
  Container Map;
  std::string Name;

public:
  CounterMap(const llvm::Twine &Name) : Name(Name.str()) { init(); }
  virtual ~CounterMap() {}

  void push(K Key) { Map[Key]++; }
  void push(K Key, T Value) { Map[Key] += Value; }
  void clear(K Key) { Map.erase(Key); }
  void clear() { Map.clear(); }

  virtual void onQuit() { dump(); }

  template<typename O>
  void dump(size_t Max, O &Output) {
    if (not Name.empty())
      Output << Name << ":\n";

    using Pair = std::pair<K, T>;
    std::vector<Pair> Sorted;
    Sorted.reserve(Map.size());
    std::copy(Map.begin(), Map.end(), std::back_inserter(Sorted));

    auto Compare = [](const Pair &A, const Pair &B) {
      return A.second < B.second;
    };
    std::sort(Sorted.begin(), Sorted.end(), Compare);

    size_t MaxLength = 0;
    size_t MaxDigits = 0;
    for (unsigned I = 0; I < std::min(Max, Sorted.size()); I++) {
      Pair &P = Sorted[Sorted.size() - 1 - I];
      MaxLength = std::max(MaxLength, P.first.size());
      MaxDigits = std::max(MaxDigits, digitsCount(P.second));
    }

    for (unsigned I = 0; I < std::min(Max, Sorted.size()); I++) {
      Pair &P = Sorted[Sorted.size() - 1 - I];
      Output << "  " << P.first << ": ";
      Output << std::string(MaxLength - P.first.size(), ' ');
      Output << std::string(MaxDigits - digitsCount(P.second), ' ');
      Output << P.second << "\n";
    }
  }

  void dump(size_t Max) { dump(Max, dbg); }
  void dump() { dump(MaxCounterMapDump, dbg); }

private:
  void init();
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
  RunningStatistics() : RunningStatistics(llvm::Twine(), false) {}

  RunningStatistics(const llvm::Twine &Name) : RunningStatistics(Name, true) {}

  /// \arg Name the name to use when printing the statistics.
  /// \arg Register whether this object should be registered for being printed
  ///      upon program termination or not.
  RunningStatistics(const llvm::Twine &Name, bool Register) :
    Name(Name.str()),
    N(0) {

    if (Register)
      init();
  }

  virtual ~RunningStatistics() {}

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

  double mean() const { return (N > 0) ? NewM : 0.0; }

  double variance() const { return ((N > 1) ? NewS / (N - 1) : 0.0); }

  double standardDeviation() const { return sqrt(variance()); }

  template<typename T>
  void dump(T &Output) {
    if (not Name.empty())
      Output << Name << ": ";
    Output << "{ n: " << size() << " u: " << mean() << " o: " << variance()
           << " }";
  }

  void dump() { dump(dbg); }

  virtual void onQuit();

private:
  void init();

private:
  std::string Name;
  int N;
  double OldM, NewM, OldS, NewS;
};

// TODO: this is duplicated
template<typename T, typename... Args>
inline std::array<T, sizeof...(Args)> make_array(Args &&... args) {
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

template<typename K, typename T>
inline void CounterMap<K, T>::init() {
  OnQuitStatistics->add(this);
}

extern void installStatistics();

#endif // STATISTICS_H
