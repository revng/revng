#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "llvm/Support/Error.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Binary/BinaryDescriptor.h"
#include "revng/Support/CommandLine.h"

struct ImporterOptions;

llvm::Error importELF(TupleTree<model::Binary> &Model,
                      const ELFBinary &TheBinary,
                      const ImporterOptions &Options);

llvm::Error importPECOFF(TupleTree<model::Binary> &Model,
                         const COFFBinary &TheBinary,
                         const ImporterOptions &Options);

llvm::Error importMachO(TupleTree<model::Binary> &Model,
                        MachOBinary &TheBinary,
                        const ImporterOptions &Options);
