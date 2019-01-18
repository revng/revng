/// \file stackanalysis.cpp
/// \brief Implementation of the stack analysis, which provides information
///        about function boundaries, basic block types, arguments and return
///        values.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

// LLVM includes
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

// Local libraries includes
#include "revng/StackAnalysis/StackAnalysis.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/IRHelpers.h"

// Local includes
#include "Cache.h"
#include "InterproceduralAnalysis.h"
#include "Intraprocedural.h"

using llvm::BasicBlock;
using llvm::Function;
using llvm::Module;
using llvm::RegisterPass;

static Logger<> ClobberedLog("clobbered");
static Logger<> StackAnalysisLog("stackanalysis");

using namespace llvm::cl;

namespace StackAnalysis {

const std::set<llvm::GlobalVariable *> EmptyCSVSet;

template<>
char StackAnalysis<true>::ID = 0;

namespace {
const char *Name = "Stack Analysis Pass";
static RegisterPass<StackAnalysis<false>> X("stack-analysis", Name, true, true);

static opt<std::string> StackAnalysisOutputPath("stack-analysis-output",
                                                desc("Destination path for the "
                                                     "Static Analysis Pass"),
                                                value_desc("path"),
                                                cat(MainCategory));

} // namespace

template<>
char StackAnalysis<false>::ID = 0;

using RegisterABI = RegisterPass<StackAnalysis<true>>;
static RegisterABI Y("abi-analysis", "ABI Analysis Pass", true, true);

static opt<std::string> ABIAnalysisOutputPath("abi-analysis-output",
                                              desc("Destination path for the "
                                                   "ABI Analysis Pass"),
                                              value_desc("path"),
                                              cat(MainCategory));

template<bool AnalyzeABI>
bool StackAnalysis<AnalyzeABI>::runOnModule(Module &M) {
  Function &F = *M.getFunction("root");

  revng_log(PassesLog, "Starting StackAnalysis");

  auto &GCBI = getAnalysis<GeneratedCodeBasicInfo>();

  // The stack analysis works function-wise. We consider two sets of functions:
  // first (Force == true) those that are highly likely to be real functions
  // (i.e., they have a direct call) and then (Force == false) all the remaining
  // candidates whose entry point is not included in any function of the first
  // set.

  struct CFEP {
    CFEP(BasicBlock *Entry, bool Force) : Entry(Entry), Force(Force) {}

    BasicBlock *Entry;
    bool Force;
  };
  std::vector<CFEP> Functions;

  // Register all the Candidate Function Entry Points
  for (BasicBlock &BB : F) {
    if (GCBI.getType(&BB) != JumpTargetBlock)
      continue;

    uint32_t Reasons = GCBI.getJTReasons(&BB);
    bool IsCallee = hasReason(Reasons, JTReason::Callee);
    bool IsUnusedGlobalData = hasReason(Reasons, JTReason::UnusedGlobalData);
    bool IsSETNotToPC = hasReason(Reasons, JTReason::SETNotToPC);
    bool IsSETToPC = hasReason(Reasons, JTReason::SETToPC);
    bool IsReturnAddress = hasReason(Reasons, JTReason::ReturnAddress);
    bool IsLoadAddress = hasReason(Reasons, JTReason::LoadAddress);

    if (IsCallee) {
      // Called addresses are a strong hint
      Functions.emplace_back(&BB, true);
    } else if (not IsLoadAddress
               and (IsUnusedGlobalData
                    || (IsSETNotToPC and not IsSETToPC
                        and not IsReturnAddress))) {
      // TODO: keep IsReturnAddress?
      // Consider addresses found in global data that have not been used in SET
      // or addresses coming from SET that are not return addresses and do not
      // end up in the PC directly.
      Functions.emplace_back(&BB, false);
    }
  }

  // Initialize the cache where all the results will be accumulated
  Cache TheCache(&F);

  // Pool where the final results will be collected
  ResultsPool Results;

  // First analyze all the `Force`d functions (i.e., with an explicit direct
  // call)
  for (CFEP &Function : Functions) {
    if (Function.Force) {
      auto &GCBI = getAnalysis<GeneratedCodeBasicInfo>();
      InterproceduralAnalysis SA(TheCache, GCBI, AnalyzeABI);
      SA.run(Function.Entry, Results);
    }
  }

  // Now analyze all the remaining candidates which are not already part of
  // another function
  std::set<BasicBlock *> Visited = Results.visitedBlocks();
  for (CFEP &Function : Functions) {
    if (not Function.Force and Visited.count(Function.Entry) == 0) {
      auto &GCBI = getAnalysis<GeneratedCodeBasicInfo>();
      InterproceduralAnalysis SA(TheCache, GCBI, AnalyzeABI);
      SA.run(Function.Entry, Results);
    }
  }

  std::stringstream Output;
  GrandResult = Results.finalize(&M);
  GrandResult.dump(&M, Output);
  TextRepresentation = Output.str();

  if (ClobberedLog.isEnabled()) {
    for (auto &P : GrandResult.Functions) {
      ClobberedLog << getName(P.first) << ":";
      for (const llvm::GlobalVariable *CSV : P.second.ClobberedRegisters)
        ClobberedLog << " " << CSV->getName().data();
      ClobberedLog << DoLog;
    }
  }

  revng_log(StackAnalysisLog, TextRepresentation);

  revng_log(PassesLog, "Ending StackAnalysis");

  if (AnalyzeABI and ABIAnalysisOutputPath.getNumOccurrences() == 1) {
    std::ofstream Output;
    serialize(pathToStream(ABIAnalysisOutputPath, Output));
  } else if (not AnalyzeABI
             and StackAnalysisOutputPath.getNumOccurrences() == 1) {
    std::ofstream Output;
    serialize(pathToStream(StackAnalysisOutputPath, Output));
  }

  return false;
}

template<bool AnalyzeABI>
void StackAnalysis<AnalyzeABI>::serializeMetadata(Function &F) {
  using namespace llvm;

  const FunctionsSummary &Summary = GrandResult;

  LLVMContext &Context = getContext(&F);
  QuickMetadata QMD(Context);

  // Temporary data structure so we can set all the `func.member.of` in a single
  // shot at the end
  std::map<TerminatorInst *, std::vector<Metadata *>> MemberOf;

  // Loop over all the detected functions
  for (const auto &P : Summary.Functions) {
    BasicBlock *Entry = P.first;
    const FunctionsSummary::FunctionDescription &Function = P.second;

    if (Entry == nullptr or Function.BasicBlocks.size() == 0)
      continue;

    //
    // Add `func.entry`:
    // { name, type, { clobbered csv, ... }, { { csv, argument, return value },
    // ... } }
    //
    auto TypeMD = QMD.get(FunctionType::getName(Function.Type));

    // Clobbered registers metadata
    std::vector<Metadata *> ClobberedMDs;
    for (GlobalVariable *ClobberedCSV : Function.ClobberedRegisters) {
      ClobberedMDs.push_back(QMD.get(ClobberedCSV));
    }

    // Register slots metadata
    std::vector<Metadata *> SlotMDs;
    if (AnalyzeABI) {
      for (auto &P : Function.RegisterSlots) {
        auto *CSV = QMD.get(P.first);
        auto *Argument = QMD.get(P.second.Argument.valueName());
        auto *ReturnValue = QMD.get(P.second.ReturnValue.valueName());
        SlotMDs.push_back(QMD.tuple({ CSV, Argument, ReturnValue }));
      }
    }

    // Create func.entry metadata
    MDTuple *FunctionMD = QMD.tuple({ QMD.get(getName(Entry)),
                                      TypeMD,
                                      QMD.tuple(ClobberedMDs),
                                      QMD.tuple(SlotMDs) });
    Entry->getTerminator()->setMetadata("func.entry", FunctionMD);

    if (AnalyzeABI) {
      //
      // Create func.call
      //
      for (const FunctionsSummary::CallSiteDescription &CallSite :
           Function.CallSites) {
        Instruction *Call = CallSite.Call;

        // Register slots metadata
        std::vector<Metadata *> SlotMDs;
        for (auto &P : CallSite.RegisterSlots) {
          auto *CSV = QMD.get(P.first);
          auto *Argument = QMD.get(P.second.Argument.valueName());
          auto *ReturnValue = QMD.get(P.second.ReturnValue.valueName());
          SlotMDs.push_back(QMD.tuple({ CSV, Argument, ReturnValue }));
        }

        Call->setMetadata("func.call", QMD.tuple(QMD.tuple(SlotMDs)));
      }
    }

    //
    // Create func.member.of
    //

    // Loop over all the basic blocks composing the function
    for (const auto &P : Function.BasicBlocks) {
      BasicBlock *BB = P.first;
      BranchType::Values Type = P.second;

      auto *Pair = QMD.tuple({ FunctionMD, QMD.get(getName(Type)) });

      // Register that this block is associated to this function
      MemberOf[BB->getTerminator()].push_back(Pair);
    }
  }

  // Apply `func.member.of`
  for (auto &P : MemberOf)
    P.first->setMetadata("func.member.of", QMD.tuple(P.second));
}

template void StackAnalysis<true>::serializeMetadata(Function &F);
template void StackAnalysis<false>::serializeMetadata(Function &F);

} // namespace StackAnalysis
