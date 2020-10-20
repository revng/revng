//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/ADT/SmallString.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Scalar.h>

// clang includes
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

// revng includes
#include <revng/Support/IRHelpers.h>

// local libraries includes
#include "revng-c/Decompiler/CDecompilerPass.h"
#include "revng-c/DecompilerResourceFinder/ResourceFinder.h"
#include "revng-c/PHIASAPAssignmentInfo/PHIASAPAssignmentInfo.h"
#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/RestructureCFG.h"
#include "revng-c/TargetFunctionOption/TargetFunctionOption.h"

// local includes
#include "CDecompilerAction.h"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

using PHIIncomingMap = SmallMap<llvm::PHINode *, unsigned, 4>;
using BBPHIMap = SmallMap<llvm::BasicBlock *, PHIIncomingMap, 4>;
using DuplicationMap = std::map<llvm::BasicBlock *, size_t>;

static cl::OptionCategory RevNgCategory("revng options");

// Prefix for the decompiled output filename.
static cl::opt<string> DecompiledDir("decompiled-dir",
                                     cl::desc("decompiled code dir"),
                                     cl::value_desc("decompiled-dir"),
                                     cl::cat(RevNgCategory));

// Prefix for the short circuit metrics dir.
static cl::opt<string> OutputPath("short-circuit-metrics-output-dir",
                                  cl::desc("Short circuit metrics dir"),
                                  cl::value_desc("short-circuit-dir"),
                                  cl::cat(RevNgCategory));

char CDecompilerPass::ID = 0;

using Register = RegisterPass<CDecompilerPass>;
static Register X("decompilation", "Decompilation Pass", false, false);

CDecompilerPass::CDecompilerPass(std::unique_ptr<llvm::raw_ostream> Out) :
  llvm::FunctionPass(ID), Out(std::move(Out)) {
}

CDecompilerPass::CDecompilerPass() : CDecompilerPass(nullptr) {
}

bool CDecompilerPass::runOnFunction(llvm::Function &F) {

  ShortCircuitCounter = 0;
  TrivialShortCircuitCounter = 0;

  if (not F.getMetadata("revng.func.entry"))
    return false;
  // HACK!!!
  if (F.getName().startswith("bb.quotearg_buffer_restyled")
      or F.getName().startswith("bb.printf_parse")
      or F.getName().startswith("bb.printf_core")
      or F.getName().startswith("bb._Unwind_VRS_Pop")
      or F.getName().startswith("bb.vasnprintf")) {
    return false;
  }

  // If we passed the `-single-decompilation` option to the command line, skip
  // decompilation for all the functions that are not the selected one.
  if (TargetFunction.size() != 0) {
    if (!F.getName().equals(TargetFunction.c_str())) {
      return false;
    }
  }

  // If we passed the `-decompiled-prefix` option to the command line, we take
  // care of serializing the decompiled source code on file.
  if (DecompiledDir.size() != 0) {
    SourceCode.clear();
    Out = std::make_unique<llvm::raw_string_ostream>(SourceCode);
  }

  // This is a hack to prevent clashes between LLVM's `opt` arguments and
  // clangTooling's CommonOptionParser arguments.
  // At this point opt's arguments have already been parsed, so there should
  // be no problem in clearing the map and let clangTooling reinitialize it
  // with its own stuff.
  cl::getRegisteredOptions().clear();

  // Remove calls to newpc
  for (Function &ParentF : *F.getParent()) {
    for (BasicBlock &BB : ParentF) {
      if (not ParentF.getMetadata("revng.func.entry"))
        continue;

      std::vector<Instruction *> ToErase;
      for (Instruction &I : BB)
        if (auto *C = dyn_cast<CallInst>(&I))
          if (getCallee(C)->getName() == "newpc")
            ToErase.push_back(C);

      for (Instruction *I : ToErase)
        I->eraseFromParent();
    }
  }

  auto &RestructureCFGAnalysis = getAnalysis<RestructureCFG>();
  ASTTree &GHAST = RestructureCFGAnalysis.getAST();
  DuplicationMap &NDuplicates = RestructureCFGAnalysis.getNDuplicates();
  auto &PHIASAPAssignments = getAnalysis<PHIASAPAssignmentInfo>();
  BBPHIMap PHIMap = PHIASAPAssignments.extractBBToPHIIncomingMap();

  // Construct the path of the include (hack copied from revng-lift). Even if
  // the include path is unique for now, we have anyway set up the search in
  // multiple paths.
  static std::string RevNgCIncludeFile;
  auto &FileFinder = revng::c::ResourceFinder;
  auto OptionalRevNgIncludeFile = FileFinder.findFile("share/revngc/"
                                                      "revng-c-include.c");
  revng_assert(OptionalRevNgIncludeFile.has_value());
  RevNgCIncludeFile = OptionalRevNgIncludeFile.value();

  // Here we build the artificial command line for clang tooling
  static std::array<const char *, 5> ArgV = {
    "revng-c",  RevNgCIncludeFile.data(),
    "--", // separator between tool arguments and clang arguments
    "-xc", // tell clang to compile C language
    "-std=c11", // tell clang to compile C11
  };
  static int ArgC = ArgV.size();
  static CommonOptionsParser OptionParser(ArgC, ArgV.data(), RevNgCategory);
  ClangTool RevNg = ClangTool(OptionParser.getCompilations(),
                              OptionParser.getSourcePathList());

  CDecompilerAction Decompilation(F,
                                  GHAST,
                                  PHIMap,
                                  std::move(Out),
                                  NDuplicates);

  using FactoryUniquePtr = std::unique_ptr<FrontendActionFactory>;
  FactoryUniquePtr Factory = newFrontendActionFactory(&Decompilation);
  RevNg.run(Factory.get());

  // Decompiled code serialization on file.
  if (DecompiledDir.size() != 0) {
    if (auto Err = llvm::sys::fs::create_directory(DecompiledDir)) {
      revng_abort("Could not create revng-c-decompiled-source directory");
    }
    std::ofstream CFile;
    std::string FileName = F.getName().str();
    CFile.open(DecompiledDir + "/" + FileName + ".c");
    if (not CFile.is_open()) {
      revng_abort("Could not open file for dumping C source file.");
    }
    std::ifstream IncludeFile;
    IncludeFile.open(RevNgCIncludeFile);
    if (not IncludeFile.is_open()) {
      revng_abort("Could not open revng-c include file.");
    }
    CFile << IncludeFile.rdbuf();
    IncludeFile.close();
    CFile << SourceCode;
    CFile.close();
  }

  // Serialize the collected metrics in the outputfile.
  if (OutputPath.getNumOccurrences() == 1) {
    std::ofstream Output;
    std::ostream &OutputStream = pathToStream(OutputPath + "/"
                                                + F.getName().data(),
                                              Output);
    OutputStream << "function,short-circuit,trivial-short-circuit\n";
    OutputStream << F.getName().data() << "," << ShortCircuitCounter << ","
                 << TrivialShortCircuitCounter << "\n";
  }

  return true;
}
