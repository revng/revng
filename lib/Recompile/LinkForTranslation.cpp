/// \file LinkForTranslation.cpp
/// \brief implementation of link for translation

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/Importer/Binary/BinaryImporterOptions.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Recompile/LinkForTranslation.h"
#include "revng/Support/Assert.h"
#include "revng/Support/ProgramRunner.h"
#include "revng/Support/TemporaryFile.h"

using namespace llvm;
using namespace llvm::sys;

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

static std::vector<std::string>
defineProgramHeadersSymbols(RawBinaryView &BinaryView,
                            const MemoryBuffer &Data) {
  using namespace llvm::object;

  auto Process = [&BinaryView](auto &ELF) -> std::vector<std::string> {
    auto Header = ELF->getELFFile().getHeader();

    auto Offset = Header.e_phoff;
    MetaAddress ProgramHeadersAddress = BinaryView.offsetToAddress(Offset);
    revng_assert(ProgramHeadersAddress.isValid());

    auto Address = Twine::utohexstr(ProgramHeadersAddress.address()).str();
    return { ("-Wl,--defsym=e_phnum=0x" + Twine(Header.e_phnum)).str(),
             ("-Wl,--defsym=e_phentsize=0x" + Twine(Header.e_phentsize)).str(),
             "-Wl,--defsym=phdr_address=0x" + Address };
  };

  auto MaybeObject = ObjectFile::createELFObjectFile(Data);
  revng_check(MaybeObject);
  if (auto *ELF = dyn_cast<ELFObjectFile<ELF32LE>>(&**MaybeObject))
    return Process(ELF);
  else if (auto *ELF = dyn_cast<ELFObjectFile<ELF64LE>>(&**MaybeObject))
    return Process(ELF);
  else if (auto *ELF = dyn_cast<ELFObjectFile<ELF32BE>>(&**MaybeObject))
    return Process(ELF);
  else if (auto *ELF = dyn_cast<ELFObjectFile<ELF64BE>>(&**MaybeObject))
    return Process(ELF);
  else
    revng_abort();
}

struct LinkingArgsOutput {
  using CommandArguments = std::vector<std::string>;

  std::vector<CommandArguments> SegmentCopyArgs;
  CommandArguments LinkerArgs;
  CommandArguments MergeDynamicArgs;
  std::vector<TemporaryFile> RawSegments;
  std::vector<TemporaryFile> SegmentELFs;
};

static void linkingArgs(const model::Binary &Model,
                        llvm::StringRef InputBinary,
                        llvm::StringRef ObjectFile,
                        llvm::StringRef OutputBinary,
                        LinkingArgsOutput &Output) {
  using CommandArguments = LinkingArgsOutput::CommandArguments;

  std::vector<TemporaryFile> &RawSegments = Output.RawSegments;
  std::vector<TemporaryFile> &SegmentELFs = Output.SegmentELFs;

  CommandArguments &Args = Output.LinkerArgs;
  std::vector<CommandArguments> &CopySegmentArgs = Output.SegmentCopyArgs;
  CommandArguments &ArgsMergeDynamic = Output.MergeDynamicArgs;

  auto UToHexStr = Twine::utohexstr;

  const size_t PageSize = 4096;

  TemporaryFile LinkerOutput("", "");

  auto MaybeBuffer = llvm::MemoryBuffer::getFileOrSTDIN(InputBinary);
  revng_assert(MaybeBuffer);
  llvm::MemoryBuffer &Buffer = **MaybeBuffer;
  RawBinaryView BinaryView(Model, Buffer.getBuffer());

  Args = { ObjectFile.str(),
           "-lz",
           "-lm",
           "-lrt",
           "-lpthread",
           "-L./",
           "-no-pie",
           "-o",
           LinkerOutput.path().str(),
           ("-Wl,-z,max-page-size=" + Twine(PageSize)).str(),
           "-fuse-ld=bfd" };

  revng_assert(Model.Segments.size() > 0);
  uint64_t Min = Model.Segments.begin()->StartAddress.address();
  uint64_t Max = Model.Segments.begin()->endAddress().address();

  for (auto &[Segment, Data] : BinaryView.segments()) {
    // Compute section name
    std::string SectionName;
    {
      llvm::raw_string_ostream NameStream(SectionName);
      NameStream << "segment-" << Segment.StartAddress.toString() << "-"
                 << Segment.endAddress().toString();
    }

    RawSegments.emplace_back("", "");
    TemporaryFile &RawSegment = RawSegments.back();

    // Create a file containing the raw segment (including .bss)
    {
      std::error_code EC;
      llvm::raw_fd_ostream Stream(RawSegment.path(), EC);
      revng_assert(!EC);
      Stream.write(reinterpret_cast<const char *>(Data.data()), Data.size());
      Stream.write_zeros(Segment.VirtualSize - Segment.FileSize);
    }

    // Create an object file we can later link
    SegmentELFs.push_back(TemporaryFile("", "o"));
    TemporaryFile &SegmentELF = SegmentELFs.back();

    std::string SectionFlags = "alloc";
    if (not Segment.IsWriteable)
      SectionFlags += ",readonly";

    std::vector<std::string> Command = {
      "-Ibinary",
      "-Oelf64-x86-64",
      "--rename-section=.data=." + SectionName,
      "--set-section-flags=.data=" + SectionFlags,
      RawSegment.path().str(),
      SegmentELF.path().str()
    };
    CopySegmentArgs.emplace_back(std::move(Command));

    Min = std::min(Min, Segment.StartAddress.address());
    Max = std::max(Max, Segment.endAddress().address());

    // Add to linker command line
    Args.push_back(SegmentELF.path().str());

    // Force section address at link-time
    const auto &StartAddr = Segment.StartAddress.address();
    Args.push_back((Twine("-Wl,--section-start=.") + SectionName + Twine("=0x")
                    + UToHexStr(StartAddr))
                     .str());
  }

  // Force text to start on the page after all the original program segments
  auto PageAddress = PageSize * ((Max + PageSize - 1) / PageSize);
  Args.push_back(("-Wl,-Ttext-segment=0x" + UToHexStr(PageAddress).str()));

  // Force a page before the lowest original address for the ELF header
  auto Str = "-Wl,--section-start=.elfheaderhelper=0x";
  Args.push_back((Str + UToHexStr(Min - 1)).str());

  // Link required dynamic libraries
  Args.push_back("-Wl,--no-as-needed");
  for (const std::string &ImportedLibrary : Model.ImportedLibraries)
    Args.push_back(linkFunctionArgument(ImportedLibrary));
  Args.push_back("-Wl,--as-needed");

  // Define program headers-related symbols
  llvm::copy(defineProgramHeadersSymbols(BinaryView, Buffer),
             std::back_inserter(Args));

  // Invoke revng-merge-dynamic
  ArgsMergeDynamic = { "merge-dynamic",
                       LinkerOutput.path().str(),
                       InputBinary.str(),
                       OutputBinary.str() };

  // Add base
  if (BaseAddress.getNumOccurrences() > 0) {
    ArgsMergeDynamic.push_back("--base");
    ArgsMergeDynamic.push_back("0x" + UToHexStr(BaseAddress).str());
  }
}

void linkForTranslation(const model::Binary &Model,
                        llvm::StringRef InputBinary,
                        llvm::StringRef ObjectFile,
                        llvm::StringRef OutputBinary) {
  LinkingArgsOutput Output;

  linkingArgs(Model, InputBinary, ObjectFile, OutputBinary, Output);
  for (const auto &Invocation : Output.SegmentCopyArgs) {
    auto ExitCode = ::Runner.run("objcopy", Invocation);
    revng_assert(ExitCode == 0);
  }

  auto ExitCode = ::Runner.run("c++", Output.LinkerArgs);
  revng_assert(ExitCode == 0);

  ExitCode = ::Runner.run("revng", Output.MergeDynamicArgs);
  revng_assert(ExitCode == 0);
}

static void printCommand(llvm::raw_ostream &OS,
                         llvm::StringRef CommandName,
                         ArrayRef<std::string> Command) {

  OS << CommandName << " ";
  for (const auto &S : Command) {
    OS << S;
    OS << " ";
  }

  OS << "\n";
}

void printLinkForTranslationCommands(llvm::raw_ostream &OS,
                                     const model::Binary &Model,
                                     llvm::StringRef InputBinary,
                                     llvm::StringRef ObjectFile,
                                     llvm::StringRef OutputBinary) {
  LinkingArgsOutput Output;

  linkingArgs(Model, InputBinary, ObjectFile, OutputBinary, Output);
  for (const auto &Invocation : Output.SegmentCopyArgs)
    printCommand(OS, "objcopy", Invocation);

  printCommand(OS, "c++", Output.LinkerArgs);
  printCommand(OS, "revng", Output.MergeDynamicArgs);
}
