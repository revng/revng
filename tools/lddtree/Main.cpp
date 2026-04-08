//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/Support/CommandLine.h"
#include "revng/Support/Configuration.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/LDDTree.h"

using namespace llvm;
using namespace llvm::object;

class StringPositionalArgument : public cl::opt<std::string> {
public:
  StringPositionalArgument(const char *Description) :
    cl::opt<std::string>(cl::Positional,
                         cl::Required,
                         cl::desc(Description),
                         cl::cat(MainCategory)) {}
};

static StringPositionalArgument Input("BINARY");

static cl::opt<std::string> CustomCanonicalPath("canonical-path",
                                                cl::desc("Canonical path for "
                                                         "the lookup of "
                                                         "dependencies "
                                                         "and debug symbols"),
                                                cl::cat(MainCategory));

static cl::list<std::string>
  Symbols(cl::desc("[SYMBOL [SYMBOL ...]"), cl::Positional, cl::ZeroOrMore);

static llvm::ExitOnError AbortOnError;

int main(int argc, char *argv[]) {
  revng::InitRevng X(argc, argv, "", { &MainCategory });

  // Collect requested symbols
  std::optional<LDDTree::SymbolSet> RequestedSymbols;
  RequestedSymbols.emplace();
  for (std::string &Symbol : Symbols)
    RequestedSymbols->insert(Symbol);

  // Load the binary
  auto MaybeBuffer = llvm::MemoryBuffer::getFileOrSTDIN(Input, false, false);
  AbortOnError(llvm::errorCodeToError(MaybeBuffer.getError()));

  auto BinaryOrError = llvm::object::createBinary(**MaybeBuffer);
  AbortOnError(BinaryOrError.takeError());

  StringRef CanonicalPath = Input;

  if (CustomCanonicalPath.getNumOccurrences() > 0) {
    CanonicalPath = CustomCanonicalPath;
  }

  // Collect dependencies
  auto Handler = [&](auto &ObjectFile) {
    return LDDTree::fromPath(CanonicalPath,
                             ObjectFile,
                             RequestedSymbols->size() > 0 ? RequestedSymbols :
                                                            std::nullopt);
  };

  std::optional<LDDTree> MaybeDependencies;
  llvm::object::Binary *Binary = BinaryOrError->get();
  if (auto *ObjectFile = dyn_cast<llvm::object::ELF32BEObjectFile>(Binary)) {
    MaybeDependencies = Handler(*ObjectFile);
  } else if (auto *ObjectFile = dyn_cast<ELF32LEObjectFile>(Binary)) {
    MaybeDependencies = Handler(*ObjectFile);
  } else if (auto *ObjectFile = dyn_cast<ELF32BEObjectFile>(Binary)) {
    MaybeDependencies = Handler(*ObjectFile);
  } else if (auto *ObjectFile = dyn_cast<ELF64LEObjectFile>(Binary)) {
    MaybeDependencies = Handler(*ObjectFile);
  } else if (auto *ObjectFile = dyn_cast<ELF64BEObjectFile>(Binary)) {
    MaybeDependencies = Handler(*ObjectFile);
  } else if (auto *ObjectFile = dyn_cast<COFFObjectFile>(Binary)) {
    MaybeDependencies = Handler(*ObjectFile);
  } else {
    dbg << "Unknown image format";
    return EXIT_FAILURE;
  }

  // Dump
  if (MaybeDependencies.has_value()) {
    MaybeDependencies->dump(llvm::outs());
  } else {
    dbg << "Couldn't compute dependencies\n";
  }

  return EXIT_SUCCESS;
}
