//
// This file is distributed under the MIT License. See LICENSE.md for details.
// \file main.cpp
// \brief Dumps call information present in PDB files.

// standard imports
#include "revng/Support/CommandLine.h"

#include "DwarfExtractor.h"
#include "InputFile.h"
#include "StreamManager.h"

// llvm imports
#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/RawError.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/COM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::msf;
using namespace llvm::pdb;

static cl::OptionCategory ExtractorCategory("Extractor options");
static cl::opt<std::string> LibName("n",
                                    cl::init("None"),
                                    cl::desc("library name the data"),
                                    cl::value_desc("database"),
                                    cl::Required,
                                    cl::cat(ExtractorCategory));
static cl::opt<std::string> DatabasePath("d",
                                         cl::init("./parameters.db"),
                                         cl::desc("Database to be used to save "
                                                  "the data"),
                                         cl::value_desc("database"),
                                         cl::cat(ExtractorCategory));
static cl::list<std::string> InputFilenames(cl::Positional,
                                            cl::desc("<input object files>"),
                                            cl::ZeroOrMore,
                                            cl::cat(ExtractorCategory));

static ExitOnError ExitOnErr;

static void dump(InputFile &IF, ParameterSaver &Db) {
  std::map<std::string, FunctionDecl> FunctionMap;

  auto O = std::make_unique<StreamManager>(IF, FunctionMap, LibName);
  ExitOnErr(O->dump());
  std::for_each(FunctionMap.begin(),
                FunctionMap.end(),
                [&Db](const auto &Pair) { Db.save(Pair.second); });
}

int main(int Argc, const char **Argv) {
  InitLLVM X(Argc, Argv);
  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  ExitOnErr.setBanner(": ");

  cl::HideUnrelatedOptions(ExtractorCategory);
  cl::ParseCommandLineOptions(Argc,
                              Argv,
                              "LLVM Dwarf and PDB call extractor\n");
  ParameterSaver Db(DatabasePath);

  llvm::sys::InitializeCOMRAII COM(llvm::sys::COMThreadingMode::MultiThreaded);

  llvm::for_each(InputFilenames, [&Db](llvm::StringRef Ref) {
    Expected<InputFile> IF = InputFile::open(Ref);
    if (!IF) {
      handleAllErrors(IF.takeError(), [](const StringError &error) {});
      extractDwarf(Ref, outs(), Db, LibName);
      return;
    }
    dump(*IF, Db);
  });

  outs().flush();
  return 0;
}
