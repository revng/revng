#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Binary.h"

#include "revng/Model/Binary.h"

namespace llvm {
namespace object {
class ObjectFile;
}
} // namespace llvm

struct ImporterOptions;
llvm::Error importBinary(TupleTree<model::Binary> &Model,
                         llvm::object::ObjectFile &BinaryHandle,
                         const ImporterOptions &Options);
llvm::Error importBinary(TupleTree<model::Binary> &Model,
                         llvm::StringRef Path,
                         const ImporterOptions &Options);
