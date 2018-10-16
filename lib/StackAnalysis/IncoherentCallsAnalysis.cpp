/// \file incoherentcallsanalysis.cpp
/// \brief Implementation of a simple analysis to identify incoherence among the
///        ABI analysis of a call site and of a callee

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local libraries includes
#include "revng/Support/MonotoneFramework.h"

// Local includes
#include "ABIIR.h"

using llvm::Module;

namespace StackAnalysis {

// Specialize debug_cmp for MonotoneFrameworkSet
template<typename T>
struct debug_cmp<MonotoneFrameworkSet<T>> {
  static unsigned cmp(const MonotoneFrameworkSet<T> &This,
                      const MonotoneFrameworkSet<T> &Other,
                      const llvm::Module *M) {
    return This.lowerThanOrEqual(Other) ? 0 : 1;
  }
};

namespace IncoherentCallsAnalysis {

using Element = MonotoneFrameworkSet<int32_t>;

class Interrupt {
private:
  enum Reason { Regular, Return, SpecialStart, NoReturn, Summary };

private:
  /// Interrupt reason
  Reason TheReason;

  /// Final result: the set of stack slots read from the caller
  Element Result;

private:
  explicit Interrupt(Reason TheReason, Element Result) :
    TheReason(TheReason),
    Result(std::move(Result)) {}

  explicit Interrupt(Reason TheReason) : TheReason(TheReason), Result() {}

public:
  static Interrupt createRegular(Element Result) {
    return Interrupt(Regular, std::move(Result));
  }

  static Interrupt createReturn(Element Result) {
    return Interrupt(Return, std::move(Result));
  }

  static Interrupt createNoReturn() { return Interrupt(NoReturn); }

  static Interrupt createSummary(Element Result) {
    return Interrupt(Summary, std::move(Result));
  }

  bool requiresInterproceduralHandling() {
    switch (TheReason) {
    case Regular:
    case SpecialStart:
    case Return:
      return false;

    case NoReturn:
    case Summary:
      return true;
    }

    revng_abort();
  }

  Element &&extractResult() { return std::move(Result); }

  bool isReturn() const { return TheReason == Return; }
};

/// \brief Analysis that computes the set of stack slots used incoherently
///
/// This (backward) analysis identifies stack slots that are used as stack
/// arguments in a function call, but are read (before a store) by the
/// caller. We consider these incoherent.
class Analysis : public MonotoneFramework<ABIIRBasicBlock *,
                                          Element,
                                          Interrupt,
                                          Analysis,
                                          ABIIRBasicBlock::links_const_range,
                                          PostOrder> {

private:
  using DirectedLabelRange = ABIIRBasicBlock::reverse_range;

public:
  using Base = MonotoneFramework<ABIIRBasicBlock *,
                                 Element,
                                 Interrupt,
                                 Analysis,
                                 ABIIRBasicBlock::links_const_range,
                                 PostOrder>;
  using LabelRange = typename Base::LabelRange;

private:
  ABIIRBasicBlock *FunctionEntry;
  std::set<ABIIRBasicBlock *> RegularExtremals;
  std::set<FunctionCall> Incoherent;

public:
  Analysis(ABIIRBasicBlock *FunctionEntry) :
    Base(FunctionEntry),
    FunctionEntry(FunctionEntry) {}

public:
  void assertLowerThanOrEqual(const Element &A, const Element &B) const {
    const Module *M = getModule(FunctionEntry->basicBlock());
    ::StackAnalysis::assertLowerThanOrEqual(A, B, M);
  }

public:
  const std::set<FunctionCall> &incoherentCalls() { return Incoherent; }

  void dumpFinalState() const {}

  llvm::Optional<Element> handleEdge(const Element &Original,
                                     ABIIRBasicBlock *Source,
                                     ABIIRBasicBlock *Destination) const {
    return llvm::Optional<Element>();
  }

  ABIIRBasicBlock::links_const_range
  successors(ABIIRBasicBlock *BB, Interrupt &) const {
    return BB->next<false>();
  }

  size_t successor_size(ABIIRBasicBlock *BB, Interrupt &) const {
    return BB->next_size<false>();
  }

  Interrupt createSummaryInterrupt() {
    return Interrupt::createSummary(std::move(this->FinalResult));
  }

  Interrupt createNoReturnInterrupt() const {
    return Interrupt::createNoReturn();
  }

  Element extremalValue(ABIIRBasicBlock *) const { return Element(); }

  Interrupt transfer(ABIIRBasicBlock *BB) {
    revng_log(SaABI, "Analyzing " << BB->basicBlock());
    Element Result = this->State[BB];
    auto SP0 = ASID::stackID();

    for (ABIIRInstruction &I : range(BB)) {

      switch (I.opcode()) {
      case ABIIRInstruction::Load:
        // The last thing we know about this stack slot is that it has been read
        if (I.target().addressSpace() == SP0)
          Result.insert(I.target().offset());
        break;

      case ABIIRInstruction::Store:
        // The last thing we know about this stack slot is that it has been
        // written to
        if (I.target().addressSpace() == SP0)
          Result.drop(I.target().offset());
        break;

      case ABIIRInstruction::DirectCall:
        // If a stack argument is read by the caller after a call but before a
        // store, it's incoherent
        if (Result.contains(I.stackArguments()))
          Incoherent.insert(I.call());
        break;

      default:
        break;
      }
    }

    // We don't really care about the final result in this case
    if (BB->predecessor_size() == 0)
      return Interrupt::createReturn(std::move(Result));
    else
      return Interrupt::createRegular(std::move(Result));
  }

private:
  DirectedLabelRange range(ABIIRBasicBlock *BB) {
    return instructionRange<DirectedLabelRange, false>(BB);
  }
};

} // namespace IncoherentCallsAnalysis

std::set<FunctionCall>
computeIncoherentCalls(ABIIRBasicBlock *Entry,
                       std::vector<ABIIRBasicBlock *> &Extremals) {
  using namespace IncoherentCallsAnalysis;

  revng_log(SaABI, "Checking coherency for stack arguments");
  Analysis BackwardFunctionAnalyses(Entry);

  for (ABIIRBasicBlock *Extremal : Extremals)
    BackwardFunctionAnalyses.registerExtremal(Extremal);

  BackwardFunctionAnalyses.initialize();
  Interrupt Result = BackwardFunctionAnalyses.run();

  return BackwardFunctionAnalyses.incoherentCalls();
}

} // namespace StackAnalysis
