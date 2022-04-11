//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Pass/AddPrimitiveTypes.h"
#include "revng/Model/Pass/RegisterModelPass.h"

static RegisterModelPass R(model::AddPrimitiveTypesFlag,
                           "Adds all the required model::PrimitiveTypes to the "
                           "Model, if necessary",
                           model::addPrimitiveTypes);

void model::addPrimitiveTypes(TupleTree<model::Binary> &Model) {
  // For each of these types, we want to have in the model a corresponding
  // PrimitiveType for each possible dimension.
  // TODO: for now we don't support floats.
  using namespace model::PrimitiveTypeKind;
  static constexpr const Values PrimitiveTypes[] = {
    Generic, PointerOrNumber, Number, Unsigned, Signed
  };
  static constexpr const uint8_t Sizes[] = { 1, 2, 4, 8, 16 };

  // getPrimitiveType() creates the type if it does not exist
  for (auto &Type : PrimitiveTypes)
    for (auto &Size : Sizes)
      Model->getPrimitiveType(Type, Size);

  // Finally, add void.
  Model->getPrimitiveType(Void, 0);
}
