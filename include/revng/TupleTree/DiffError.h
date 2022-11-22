#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "revng/TupleTree/DocumentError.h"
#include "revng/TupleTree/TupleTreePath.h"

namespace revng {

class DiffError
  : public DocumentError<DiffError, TupleTreeLocation<model::Binary>> {
public:
  using DocumentError<DiffError,
                      TupleTreeLocation<model::Binary>>::DocumentError;
  inline static char ID = '0';

  std::string getTypeName() const override { return "DiffError"; }
};

} // namespace revng
