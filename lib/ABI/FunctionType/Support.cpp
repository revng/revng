/// \file Support.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/ABI/FunctionType/Support.h"
#include "revng/TupleTree/TupleTree.h"

namespace abi::FunctionType {

const model::TypePath &replaceAllUsesWith(const model::Type::Key &OldKey,
                                          const model::TypePath &NewTypePath,
                                          TupleTree<model::Binary> &Model) {
  auto CheckTheKey = [&OldKey](const model::TypePath &Reference) -> bool {
    if (Reference.empty())
      return false;

    return OldKey == Reference.getConst()->key();
  };
  Model.replaceReferencesIf(NewTypePath, CheckTheKey);

  return NewTypePath;
}

} // namespace abi::FunctionType
