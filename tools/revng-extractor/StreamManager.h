//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
//
#ifndef LLVM_TOOLS_LLVMPDBDUMP_DUMPOUTPUTSTYLE_H
#define LLVM_TOOLS_LLVMPDBDUMP_DUMPOUTPUTSTYLE_H

// revng includes
#include "revng/DeclarationsDb/DeclarationsDb.h"

// llvm includes
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/Support/Error.h"

// standard include
#include <map>
#include <string>

namespace llvm {
namespace pdb {
class PDBFile;
}

namespace object {
class COFFObjectFile;
}

namespace pdb {
class GSIHashTable;
class InputFile;

class StreamManager {

public:
  StreamManager(InputFile &File,
                std::map<std::string, FunctionDecl> &functionMap,
                const std::string &LibName);

  Error dump();

private:
  PDBFile &getPdb();
  object::COFFObjectFile &getObj();

  Error dumpTpiStream(uint32_t StreamIdx);
  Error dumpTypesFromObjectFile();
  Error dumpModuleSymsForPdb();
  Error dumpModuleSymsForObj();
  const std::string &LibName;
  std::map<std::string, FunctionDecl> &FunctionMap;

  InputFile &File;
};
} // namespace pdb
} // namespace llvm

#endif
