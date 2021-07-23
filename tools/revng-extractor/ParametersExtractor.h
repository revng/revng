//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef LLVM_TOOLS_LLVMPDBUTIL_MINIMAL_SYMBOL_DUMPER_H
#define LLVM_TOOLS_LLVMPDBUTIL_MINIMAL_SYMBOL_DUMPER_H

// revng includes
#include "revng/DeclarationsDb/DeclarationsDb.h"

// llvm includes
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbacks.h"

// standard includes
#include <map>

namespace llvm {
namespace codeview {
class LazyRandomTypeCollection;
}

namespace pdb {
class SymbolGroup;

class ParametersExtractor : public codeview::SymbolVisitorCallbacks {
public:
  ParametersExtractor(codeview::LazyRandomTypeCollection &Ids,
                      codeview::LazyRandomTypeCollection &Types,
                      std::map<std::string, FunctionDecl> &FunctionMap,
                      const std::string &LibName) :
    Ids(Ids),
    Types(Types),
    FunctionMap(FunctionMap),
    LibName(LibName),
    CompilationCPU(codeview::CPUType::X64) {}

  Error visitSymbolBegin(codeview::CVSymbol &Record) override;
  Error visitSymbolBegin(codeview::CVSymbol &Record, uint32_t Offset) override;
  Error visitSymbolEnd(codeview::CVSymbol &Record) override;

  Error
  visitKnownRecord(codeview::CVSymbol &CVR, codeview::ProcSym &Proc) override;
  Error
  visitKnownRecord(codeview::CVSymbol &CVR, codeview::LocalSym &Local) override;
  Error visitKnownRecord(codeview::CVSymbol &CVR,
                         codeview::DefRangeRegisterSym &Symb) override;
  Error visitKnownRecord(codeview::CVSymbol &CVR,
                         codeview::DefRangeRegisterRelSym &Symb) override;
  Error visitKnownRecord(codeview::CVSymbol &CVR,
                         codeview::Compile3Sym &Symb) override;

private:
  std::string typeOrIdIndex(codeview::TypeIndex TI, bool IsType) const;

  std::string typeIndex(codeview::TypeIndex TI) const;
  std::string idIndex(codeview::TypeIndex TI) const;

  codeview::LazyRandomTypeCollection &Ids;
  codeview::LazyRandomTypeCollection &Types;
  std::string LastFunctionName;
  std::string LastParamName;
  std::map<std::string, FunctionDecl> &FunctionMap;
  const std::string &LibName;
  codeview::CPUType CompilationCPU;
};
} // namespace pdb
} // namespace llvm

#endif
