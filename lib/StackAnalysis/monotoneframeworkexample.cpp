/// \file monotoneframeworkexample.cpp
/// \brief Example of minimal data-flow analysis using the MonotoneFramework
///        class

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Note: this compilation unit should result in no code and no data

namespace llvm {
class Module;
}

// Local includes
#include "monotoneframework.h"

namespace StackAnalysis {

namespace ExampleAnalysis {

class Label {};

class LatticeElement {
public:
  static LatticeElement bottom() { return LatticeElement(); }
  LatticeElement copy() { abort(); }
  void combine(const LatticeElement &) { abort(); }
  bool greaterThan(const LatticeElement &) { abort(); }
  void dump() { abort(); }
};

class Interrupt {
public:
  bool requiresInterproceduralHandling() { abort(); }
  LatticeElement &&extractResult() { abort(); }
  bool isReturn() const { abort(); }
};

class Analysis : public MonotoneFramework<Label *,
                                          LatticeElement,
                                          Interrupt,
                                          Analysis,
                                          llvm::iterator_range<Label **>> {
public:
  void assertLowerThanOrEqual(const LatticeElement &A,
                              const LatticeElement &B) const {
    abort();
  }

  Analysis(Label *Entry) : MonotoneFramework(Entry) {}

  void dumpFinalState() const { abort(); }

  llvm::iterator_range<Label **> successors(Label *, Interrupt &) const {
    abort();
  }

  size_t successor_size(Label *, Interrupt &) const { abort(); }
  Interrupt createSummaryInterrupt() { abort(); }
  Interrupt createNoReturnInterrupt() const { abort(); }
  LatticeElement extremalValue(Label *) const { abort(); }
  LabelRange extremalLabels() const { abort(); }
  Interrupt transfer(Label *) { abort(); }
};

inline void testFunction() {
  Label Entry;
  Analysis Example(&Entry);
  Example.initialize();
  Interrupt Result = Example.run();
  (void) Result;
}

} // namespace ExampleAnalysis

} // namespace StackAnalysis
