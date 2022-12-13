#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "llvm/Support/Error.h"

#include "revng/Model/Binary.h"

namespace llvm {
namespace object {
class ELFObjectFileBase;
class COFFObjectFile;
class MachOObjectFile;
} // namespace object
} // namespace llvm

llvm::Error importELF(TupleTree<model::Binary> &Model,
                      const llvm::object::ELFObjectFileBase &TheBinary,
                      uint64_t PreferredBaseAddress,
                      unsigned FetchDebugInfoWithLevel);
llvm::Error importPECOFF(TupleTree<model::Binary> &Model,
                         const llvm::object::COFFObjectFile &TheBinary,
                         uint64_t PreferredBaseAddress,
                         unsigned FetchDebugInfoWithLevel);
llvm::Error importMachO(TupleTree<model::Binary> &Model,
                        llvm::object::MachOObjectFile &TheBinary,
                        uint64_t PreferredBaseAddress);

template<typename... Ts>
llvm::Error createError(char const *Fmt, const Ts &...Vals) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), Fmt, Vals...);
}
