#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <string>
#include <vector>

#include "revng/Support/Debug.h"
#include "revng/Support/OnQuit.h"

const size_t MaxCounterMapDump = 32;

extern llvm::cl::opt<bool> Statistics;

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
class CounterMap {
private:
  using Container = std::map<K, T>;
  Container Map;
  std::string Name;

public:
  CounterMap(const llvm::StringRef Name) : Name(Name.str()) {
    OnQuit->add([this] {
      if (Statistics)
        dump();
    });
  }

  void push(K Key) { Map[Key]++; }
  void push(K Key, T Value) { Map[Key] += Value; }
  void clear(K Key) { Map.erase(Key); }
  void clear() { Map.clear(); }

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
};

/// Collect mean and variance about a certain event.
///
/// To use this class, simply create a global variable, call push with the value
/// you want to record, when you're done use the various methods to obtain mean
/// and variance.
///
/// If a name is provided, the results will be registered for printing at
/// program termination.
class RunningStatistics {
private:
  std::string Name;
  int N = 0;
  double OldM = 0.0;
  double NewM = 0.0;
  double OldS = 0.0;
  double NewS = 0.0;
  double Sum = 0.0;

public:
  RunningStatistics() = default;

  RunningStatistics(const llvm::StringRef Name) : Name(Name.str()) {
    OnQuit->add([this] {
      if (Statistics)
        dump();
    });
  }

  void clear() { N = 0; }

  // TODO: make a template
  /// Record a new value
  void push(double X) {
    N++;
    Sum += X;

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

  double sum() const { return Sum; }

  template<typename T>
  void dump(T &Output) {
    Output << Name << ": "
           << "{ s: " << sum() << " "
           << "n: " << size() << " "
           << "u: " << mean() << " "
           << "o: " << variance() << " }\n";
  }

  void dump() { dump(dbg); }
};
