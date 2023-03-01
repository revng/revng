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
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Recompile/LinkForTranslation.h"
#include "revng/Support/Assert.h"
#include "revng/Support/PathList.h"
#include "revng/Support/ProgramRunner.h"
#include "revng/Support/ResourceFinder.h"
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
    return { ("--defsym=e_phnum=0x" + Twine(Header.e_phnum)).str(),
             ("--defsym=e_phentsize=0x" + Twine(Header.e_phentsize)).str(),
             "--defsym=phdr_address=0x" + Address };
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

class Command {
public:
  std::string CommandName;
  std::vector<std::string> Arguments;

public:
  Command(std::string CommandName) : CommandName(CommandName) {}
};

class CommandList {
private:
  std::vector<Command> Commands;
  std::vector<std::unique_ptr<TemporaryFile>> Temporaries;

public:
  void print(llvm::raw_ostream &OS) const {
    for (const Command &C : Commands) {
      OS << shellEscape(C.CommandName);
      for (const std::string &Argument : C.Arguments) {
        OS << " " << shellEscape(Argument);
      }
      OS << "\n";
    }
  }

  void run() const {
    for (const Command &C : Commands) {
      auto ExitCode = ::Runner.run(C.CommandName, C.Arguments);
      revng_check(ExitCode == 0);
    }
  }

public:
  void enqueueCommand(Command C) { Commands.push_back(std::move(C)); }

  TemporaryFile &createTemporary(std::string Prefix, std::string Suffix) {
    Temporaries.emplace_back(std::make_unique<TemporaryFile>(Prefix, Suffix));
    return *Temporaries.back();
  }

public:
  static std::string shellEscape(StringRef String) {
    std::string Result = "\"";

    for (char C : String) {
      if (C == '"')
        Result += "'\"'\"'";
      else
        Result += C;
    }

    Result += "\"";

    return Result;
  }
};

template<typename T = std::string,
         typename Container = std::initializer_list<T>>
void appendTo(Container &&Range, auto &Destination) {
  for (auto &&Element : Range)
    Destination.push_back(std::forward<decltype(Element)>(Element));
}

static CommandList linkingArgs(const model::Binary &Model,
                               llvm::StringRef InputBinary,
                               llvm::StringRef ObjectFile,
                               llvm::StringRef OutputBinary) {
  CommandList Result;

  auto UToHexStr = Twine::utohexstr;

  const size_t PageSize = 4096;

  TemporaryFile &LinkerOutput = Result.createTemporary("revng-link-for-"
                                                       "translation",
                                                       "");

  auto MaybeBuffer = llvm::MemoryBuffer::getFileOrSTDIN(InputBinary);
  revng_assert(MaybeBuffer);
  llvm::MemoryBuffer &Buffer = **MaybeBuffer;
  RawBinaryView BinaryView(Model, Buffer.getBuffer());

  Command Linker("ld.bfd");

  // Parse options from environment variable.
  if (auto EnvValue = sys::Process::GetEnv("REVNG_TRANSLATE_LDFLAGS")) {
    SmallVector<const char *, 20> ExtraArguments;
    BumpPtrAllocator A;
    StringSaver Saver(A);
    cl::TokenizeGNUCommandLine(*EnvValue, Saver, ExtraArguments);

    appendTo(ExtraArguments, Linker.Arguments);
  }

  // Disable PIE
  appendTo({ "-no-pie" }, Linker.Arguments);

  // Disable huge pages
  appendTo({ "-z", ("max-page-size=" + Twine(PageSize)).str() },
           Linker.Arguments);

  // Output file
  appendTo({ "-o", LinkerOutput.path().str() }, Linker.Arguments);

  revng_assert(Model.Segments().size() > 0);
  uint64_t Min = Model.Segments().begin()->StartAddress().address();
  uint64_t Max = Model.Segments().begin()->endAddress().address();

  // Link opening crt files
  appendTo({ "-l:crt1.o", "-l:crti.o", "-l:crtbegin.o" }, Linker.Arguments);

  Linker.Arguments.push_back(ObjectFile.str());

  for (auto &[Segment, Data] : BinaryView.segments()) {
    // Compute section name
    std::string SectionName;
    {
      llvm::raw_string_ostream NameStream(SectionName);
      NameStream << "segment-" << Segment.StartAddress().toString() << "-"
                 << Segment.endAddress().toString();
    }

    TemporaryFile &RawSegment = Result.createTemporary("revng-link-for-"
                                                       "translation",
                                                       "raw");

    Command DD("dd");
    DD.Arguments = { { "status=none",
                       "bs=1",
                       ("skip=" + Twine(Segment.StartOffset())).str(),
                       ("if=" + InputBinary).str(),
                       ("count=" + Twine(Segment.FileSize())).str(),
                       ("of=" + RawSegment.path()).str() } };
    Result.enqueueCommand(std::move(DD));

    Command Truncate("truncate");
    Truncate.Arguments = { { ("--size=" + Twine(Segment.VirtualSize())).str(),
                             RawSegment.path().str() } };
    Result.enqueueCommand(std::move(Truncate));

    Command ObjCopy("objcopy");

    // Create an object file we can later link
    TemporaryFile &SegmentELF = Result.createTemporary("revng-link-for-"
                                                       "translation",
                                                       "o");

    std::string SectionFlags = "alloc";
    if (not Segment.IsWriteable())
      SectionFlags += ",readonly";

    ObjCopy.Arguments = { "-Ibinary",
                          "-Oelf64-x86-64",
                          "--add-section=.note.GNU-stack=/dev/null",
                          "--rename-section=.data=." + SectionName,
                          "--set-section-flags=.data=" + SectionFlags,
                          RawSegment.path().str(),
                          SegmentELF.path().str() };

    // Register the objcopy invocation
    Result.enqueueCommand(ObjCopy);

    Min = std::min(Min, Segment.StartAddress().address());
    Max = std::max(Max, Segment.endAddress().address());

    // Add to linker command line
    Linker.Arguments.push_back(SegmentELF.path().str());

    // Force section address at link-time
    const auto &StartAddr = Segment.StartAddress().address();
    Linker.Arguments.push_back((Twine("--section-start=.") + SectionName
                                + Twine("=0x") + UToHexStr(StartAddr))
                                 .str());
  }

  // Link C standard library
  appendTo({ "-l:ld-linux-x86-64.so.2", "-lc", "-lm", "-lrt", "-lpthread" },
           Linker.Arguments);

  // Link additional libraries needed by helpers
  appendTo({ "-lgcc", "-lz", "-lunwind" }, Linker.Arguments);

  // Link C++ standard library, we use its personality function
  appendTo({ "-lstdc++" }, Linker.Arguments);

  appendTo({ "-dynamic-linker", "/lib64/ld-linux-x86-64.so.2" },
           Linker.Arguments);

  // Link closing crt files
  appendTo({ "-l:crtend.o", "-l:crtn.o" }, Linker.Arguments);

  // Force text to start on the page after all the original program segments
  auto PageAddress = PageSize * ((Max + PageSize - 1) / PageSize);
  auto HexPageAddress = UToHexStr(PageAddress).str();
  Linker.Arguments.push_back("-Ttext-segment=0x" + HexPageAddress);

  // Force a page before the lowest original address for the ELF header
  auto Str = "--section-start=.elfheaderhelper=0x";
  Linker.Arguments.push_back((Str + UToHexStr(Min - 1)).str());

  // Link required dynamic libraries
  Linker.Arguments.push_back("--no-as-needed");
  for (const std::string &ImportedLibrary : Model.ImportedLibraries())
    Linker.Arguments.push_back(linkFunctionArgument(ImportedLibrary));
  Linker.Arguments.push_back("--as-needed");

  // Define program headers-related symbols
  llvm::copy(defineProgramHeadersSymbols(BinaryView, Buffer),
             std::back_inserter(Linker.Arguments));

  Result.enqueueCommand(std::move(Linker));

  std::string MainExecutablePath = *revng::ResourceFinder.findFile("bin/revng");
  Command MergeDynamic(MainExecutablePath);

  // Invoke revng-merge-dynamic
  MergeDynamic.Arguments = { "merge-dynamic",
                             LinkerOutput.path().str(),
                             InputBinary.str(),
                             OutputBinary.str() };

  // Add base
  if (BaseAddress.getNumOccurrences() > 0) {
    MergeDynamic.Arguments.push_back("--base");
    MergeDynamic.Arguments.push_back("0x" + UToHexStr(BaseAddress).str());
  }

  Result.enqueueCommand(std::move(MergeDynamic));

  return Result;
}

void linkForTranslation(const model::Binary &Model,
                        llvm::StringRef InputBinary,
                        llvm::StringRef ObjectFile,
                        llvm::StringRef OutputBinary) {
  CommandList Commands = linkingArgs(Model,
                                     InputBinary,
                                     ObjectFile,
                                     OutputBinary);
  Commands.run();
}

void printLinkForTranslationCommands(llvm::raw_ostream &OS,
                                     const model::Binary &Model,
                                     llvm::StringRef InputBinary,
                                     llvm::StringRef ObjectFile,
                                     llvm::StringRef OutputBinary) {
  CommandList Commands = linkingArgs(Model,
                                     InputBinary,
                                     ObjectFile,
                                     OutputBinary);
  Commands.print(OS);
}
