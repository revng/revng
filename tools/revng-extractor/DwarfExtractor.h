#ifndef DWARF_EXTRACTOR_H
#define DWARF_EXTRACTOR_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

// Local includes
#include "revng/DeclarationsDb/DeclarationsDb.h"

bool extractDwarf(llvm::StringRef Filename,
                  llvm::raw_ostream &OS,
                  ParameterSaver &Db,
                  llvm::StringRef libName);

#endif
