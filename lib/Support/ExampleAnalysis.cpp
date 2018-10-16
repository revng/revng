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

// Local libraries includes
#include "revng/Support/MonotoneFramework.h"

namespace ExampleAnalysis {

class Label {};

class LatticeElement {
public:
  static LatticeElement bottom() { return LatticeElement(); }
  LatticeElement copy() { revng_abort(); }
  void combine(const LatticeElement &) { revng_abort(); }
  bool greaterThan(const LatticeElement &) { revng_abort(); }
  void dump() { revng_abort(); }
};

class Interrupt {
public:
  bool requiresInterproceduralHandling() { revng_abort(); }
  LatticeElement &&extractResult() { revng_abort(); }
  bool isReturn() const { revng_abort(); }
};

class Analysis : public MonotoneFramework<Label *,
                                          LatticeElement,
                                          Interrupt,
                                          Analysis,
                                          llvm::iterator_range<Label **>> {
public:
  void assertLowerThanOrEqual(const LatticeElement &A,
                              const LatticeElement &B) const {
    revng_abort();
  }

  Analysis(Label *Entry) : MonotoneFramework(Entry) {}

  void dumpFinalState() const { revng_abort(); }

  llvm::iterator_range<Label **> successors(Label *, Interrupt &) const {
    revng_abort();
  }

  llvm::Optional<LatticeElement> handleEdge(const LatticeElement &Original,
                                            Label *Source,
                                            Label *Destination) const {
    revng_abort();
  }
  size_t successor_size(Label *, Interrupt &) const { revng_abort(); }
  Interrupt createSummaryInterrupt() { revng_abort(); }
  Interrupt createNoReturnInterrupt() const { revng_abort(); }
  LatticeElement extremalValue(Label *) const { revng_abort(); }
  LabelRange extremalLabels() const { revng_abort(); }
  Interrupt transfer(Label *) { revng_abort(); }
};

inline void testFunction() {
  Label Entry;
  Analysis Example(&Entry);
  Example.initialize();
  Example.run();
}

} // namespace ExampleAnalysis
