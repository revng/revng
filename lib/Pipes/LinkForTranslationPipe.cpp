/// \file LinkForTranslation.cpp
/// \brief the link for translation pipe is used to link object files into a
/// executable

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/Registry.h"
#include "revng/Pipes/LiftPipe.h"
#include "revng/Pipes/LinkForTranslationPipe.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/Assert.h"
#include "revng/Support/ProgramRunner.h"

using namespace llvm;
using namespace llvm::sys;
using namespace pipeline;
using namespace revng::pipes;

static std::string linkFunctionArgument(llvm::StringRef Lib) {
  auto LastSlash = Lib.rfind('/');
  if (LastSlash != llvm::StringRef::npos)
    Lib = Lib.drop_front(LastSlash + 1);

  llvm::Regex Reg("^lib(.*).so(\\.[0-9]+)*$");
  if (not Reg.match(Lib))
    return "-l:" + Lib.str();

  Lib = Lib.drop_front(3);
  auto LastDot = Lib.find('.');
  revng_assert(LastDot != llvm::StringRef::npos);
  Lib = Lib.substr(0, LastDot);

  return "-l" + Lib.str();
}

void LinkForTranslationPipe::run(const Context &Ctx,
                                 FileContainer &InputBinary,
                                 FileContainer &ObjectFile,
                                 FileContainer &OutputBinary) {
  const auto &Model = getModelFromContext(Ctx);

  const size_t PageSize = 4096;

  FileContainer LinkerOutput(Binary, InputBinary.name());

  llvm::SmallVector<std::string, 0> Args = {
    ObjectFile.path()->str(),
    "-lz",
    "-lm",
    "-lrt",
    "-lpthread",
    "-L./",
    "-no-pie",
    "-o",
    LinkerOutput.getOrCreatePath().str(),
    ("-Wl,-z,max-page-size=" + Twine(PageSize)).str(),
    "-fuse-ld=bfd"
  };

  revng_assert(Model.Segments.size() > 0);
  uint64_t Min = Model.Segments.begin()->StartAddress.address();
  uint64_t Max = Model.Segments.begin()->EndAddress.address();

  for (const auto &Segment : Model.Segments) {
    Min = std::min(Min, Segment.StartAddress.address());
    Max = std::max(Max, Segment.EndAddress.address());

    std::stringstream NameStream;
    NameStream << "segment-" << Segment.StartAddress.toString() << "-"
               << Segment.EndAddress.toString();

    // Force section address
    const auto &StartAddr = Segment.StartAddress.address();
    Args.push_back((Twine("-Wl,--section-start=.") + NameStream.str()
                    + Twine("=") + Twine::utohexstr(StartAddr))
                     .str());
  }

  // Force text to start on the page after all the original program segments
  auto PageAddress = PageSize * ((Max + PageSize - 1) / PageSize);
  Args.push_back(("-Wl,-Ttext-segment=" + Twine::utohexstr(PageAddress).str()));

  // Force a page before the lowest original address for the ELF header
  auto Str = "-Wl,--section-start=.elfheaderhelper=";
  Args.push_back((Str + Twine::utohexstr(Min - 1)).str());

  // Link required dynamic libraries
  Args.push_back("-Wl,--no-as-needed");
  for (const std::string &ImportedLibrary : Model.ImportedLibraries)
    Args.push_back(linkFunctionArgument(ImportedLibrary));
  Args.push_back("-Wl,--as-needed");

  // Prepare actual arguments
  int ExitCode = ::Runner.run("c++", Args);
  revng_assert(ExitCode == 0);

  // Invoke revng-merge-dynamic
  Args = { "merge-dynamic",
           LinkerOutput.path()->str(),
           InputBinary.path()->str(),
           OutputBinary.getOrCreatePath().str() };

  // Add base
  if (BaseAddress.getNumOccurrences() > 0) {
    Args.push_back("--base");
    Args.push_back(Twine::utohexstr(BaseAddress).str());
  }

  ExitCode = ::Runner.run("revng", Args);

  revng_assert(ExitCode == 0);
}

static RegisterPipe<LinkForTranslationPipe> E5;
