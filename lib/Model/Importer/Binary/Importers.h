#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "llvm/Support/Error.h"

#include "revng/Model/Binary.h"
#include "revng/Support/CommandLine.h"

namespace llvm {
namespace object {
class ELFObjectFileBase;
class COFFObjectFile;
class MachOObjectFile;
} // namespace object
} // namespace llvm

struct ImporterOptions;

template<typename T>
struct BinaryDescriptor {
public:
  T &ObjectFile;
  llvm::MemoryBufferRef Buffer;
  llvm::StringRef Path;
  model::BinaryReference &Reference;

  llvm::StringRef getFilename() const {
    // The debug info discovery relies on knowing the filename of the input
    // binary, this is because, among the standard candidate files there are
    // options that depend on it such as `${FILENAME}.debug`. If the reference
    // to the `Binaries` entry is present, use the name from there as it's more
    // reliable than the filename of the file on disk (this is still used as a
    // fallback in order to be compatible with the old pipeline).
    if (Reference.isValid())
      return Reference.get()->Path();
    else if (not InputPath.empty())
      return InputPath;
    else if (not ObjectFile.getFileName().empty())
      return ObjectFile.getFileName();
    else
      return Path;
  }
};

using ELFBinary = BinaryDescriptor<llvm::object::ELFObjectFileBase>;
llvm::Error importELF(TupleTree<model::Binary> &Model,
                      const ELFBinary &TheBinary,
                      const ImporterOptions &Options);

using COFFBinary = BinaryDescriptor<llvm::object::COFFObjectFile>;
llvm::Error importPECOFF(TupleTree<model::Binary> &Model,
                         const COFFBinary &TheBinary,
                         const ImporterOptions &Options);

using MachOBinary = BinaryDescriptor<llvm::object::MachOObjectFile>;
llvm::Error importMachO(TupleTree<model::Binary> &Model,
                        MachOBinary &TheBinary,
                        const ImporterOptions &Options);
