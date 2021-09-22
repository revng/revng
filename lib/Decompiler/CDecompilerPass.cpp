//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <memory>
#include <system_error>
#include <utility>

#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/DataLayoutAnalysis/DLALayouts.h"
#include "revng-c/DataLayoutAnalysis/DLAPass.h"
#include "revng-c/Decompiler/CDecompilerPass.h"
#include "revng-c/MarkForSerialization/MarkForSerializationPass.h"
#include "revng-c/PHIASAPAssignmentInfo/PHIASAPAssignmentInfo.h"
#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/RestructureCFG.h"
#include "revng-c/TargetFunctionOption/TargetFunctionOption.h"
#include "revng-c/ThreadSafeClangTooling/ThreadSafeClangTooling.h"
#include "revng-c/Utils/Utils.h"

#include "CDecompilerAction.h"
#include "DecompilationHelpers.h"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

using PHIIncomingMap = SmallMap<llvm::PHINode *, unsigned, 4>;
using BBPHIMap = SmallMap<llvm::BasicBlock *, PHIIncomingMap, 4>;
using DuplicationMap = std::map<const llvm::BasicBlock *, size_t>;

using llvm::cl::NumOccurrencesFlag;

// Prefix for the decompiled output filename.
static cl::opt<std::string> DecompiledDir("decompiled-dir",
                                          cl::desc("decompiled code dir"),
                                          cl::value_desc("decompiled-dir"),
                                          cl::cat(MainCategory),
                                          NumOccurrencesFlag::Optional);

// Prefix for the short circuit metrics dir.
static cl::opt<std::string> OutputPath("short-circuit-metrics-output-dir",
                                       cl::desc("Short circuit metrics dir"),
                                       cl::value_desc("short-circuit-dir"),
                                       cl::cat(MainCategory),
                                       NumOccurrencesFlag::Optional);

char CDecompilerPass::ID = 0;

using Register = RegisterPass<CDecompilerPass>;
static Register X("decompilation", "Decompilation Pass", false, false);

CDecompilerPass::CDecompilerPass(std::unique_ptr<llvm::raw_ostream> Out) :
  llvm::FunctionPass{ ID }, Out{ std::move(Out) } {
}

CDecompilerPass::CDecompilerPass() : CDecompilerPass(nullptr) {
}

bool CDecompilerPass::doInitialization(llvm::Module &) {

  // This is a hack to prevent clashes between LLVM's `opt` arguments and
  // clangTooling's arguments.
  // At this point opt's arguments have already been parsed, so there should
  // be no problem in clearing the map (held by cl::opt's GlobalParser to know
  // which are the registered valid options) and let clangTooling reinitialize
  // it with its own stuff.
  std::scoped_lock ClangToolingGuard{ ClangToolingMutex };
  cl::getRegisteredOptions().clear();

  return false;
}

void CDecompilerPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<RestructureCFG>();
  AU.addRequired<llvm::ScalarEvolutionWrapperPass>();
  AU.addRequired<MarkForSerializationPass>();
  AU.addRequired<PHIASAPAssignmentInfo>();
  AU.addUsedIfAvailable<DLAPass>();
  AU.setPreservesAll();
}

bool CDecompilerPass::runOnFunction(llvm::Function &F) {

  ShortCircuitCounter = 0;
  TrivialShortCircuitCounter = 0;

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Lifted))
    return false;

  // If the `-single-decompilation` option was passed from command line, skip
  // decompilation for all the functions that are not the selected one.
  if (not TargetFunction.empty())
    if (not F.getName().equals(TargetFunction.c_str()))
      return false;

  // If the -decompiled-dir flag was passed, the decompiled function needs to be
  // written to file, in the specified directory.
  // We initialize Out with a proper file descriptor to make it happen.
  if (DecompiledDir.getNumOccurrences())
    Out = openFunctionFile(DecompiledDir, F.getName(), ".c");

  // If the --short-circuit-metrics-output-dir=dir argument was passed from
  // command line, we need to print the statistics for the short circuit metrics
  // into a file with the function name, inside the directory 'dir'.
  std::unique_ptr<llvm::raw_fd_ostream> StatsFileStream;
  if (OutputPath.getNumOccurrences())
    StatsFileStream = openFunctionFile(OutputPath, F.getName(), ".csv");

  // Get the Abstract Syntax Tree of the restructured code.
  auto &RestructureCFGAnalysis = getAnalysis<RestructureCFG>();
  ASTTree &GHAST = RestructureCFGAnalysis.getAST();

  // Get information about which instructions are marked to be serialized.
  const auto &Mark = getAnalysis<MarkForSerializationPass>().getMap();

  // Get information about serialization of PHI nodes.
  auto &PHIASAPAssignments = getAnalysis<PHIASAPAssignmentInfo>();
  BBPHIMap PHIMap = PHIASAPAssignments.extractBBToPHIIncomingMap();

  // Get the results of the Data Layout Analysis, if present.
  // If DLA has been executed, we also need to get the ScalarEvolution, because
  // we will use it in decompilation to transform pointer arithmetics into
  // access to structs' fields.
  // Eventually, when we will integrate revng's Model, the DLA results will not
  // be necessary anymore. We'll only need to check if the Model contains type
  // information.
  auto *DLA = getAnalysisIfAvailable<DLAPass>();
  const dla::ValueLayoutMap *LayoutMap = nullptr;
  llvm::ScalarEvolution *SE = nullptr;
  if (DLA) {
    LayoutMap = DLA->getLayoutMap();
    SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  }

  auto Action = std::make_unique<CDecompilerAction>(F,
                                                    GHAST,
                                                    PHIMap,
                                                    LayoutMap,
                                                    SE,
                                                    Mark,
                                                    std::move(Out));

  const std::string CCode{ "#include <stdint.h>" };
  runThreadSafeClangTool(std::move(Action), CCode);

  // Serialize the collected metrics in the statistics file if necessary
  if (StatsFileStream) {
    *StatsFileStream << "function,short-circuit,trivial-short-circuit\n"
                     << F.getName().data() << "," << ShortCircuitCounter << ","
                     << TrivialShortCircuitCounter << "\n";
  }

  return true;
}
