/// \file Support.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/FunctionType/Support.h"
#include "revng/TupleTree/TupleTree.h"

namespace abi::FunctionType {

const model::TypeDefinitionPath &
replaceAllUsesWith(const model::TypeDefinition::Key &OldKey,
                   const model::TypeDefinitionPath &NewTypePath,
                   TupleTree<model::Binary> &Model) {
  auto CheckTheKey = [&OldKey](const model::TypeDefinitionPath &Path) -> bool {
    if (Path.empty())
      return false;

    return OldKey == Path.getConst()->key();
  };
  Model.replaceReferencesIf(NewTypePath, CheckTheKey);

  return NewTypePath;
}

} // namespace abi::FunctionType
