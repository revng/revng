//
// This file is distributed under the MIT License. See LICENSE.mit for details.
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
  using namespace model::PrimitiveTypeKind;
  static constexpr const Values PointerOrNumberPrimitiveTypes[] = {
    PointerOrNumber, Number, Unsigned, Signed
  };
  static constexpr const uint8_t PointerOrNumberSizes[] = { 1, 2, 4, 8, 16 };
  static constexpr const uint8_t FloatSizes[] = { 2, 4, 8, 10, 12, 16 };
  // Generic sizes must be the union of all other non-zero primitive type sizes
  static constexpr const uint8_t GenericSizes[] = { 1, 2, 4, 8, 10, 12, 16 };

  // getPrimitiveType() creates the type if it does not exist
  // Create pointer or number types
  for (auto &Type : PointerOrNumberPrimitiveTypes)
    for (auto &Size : PointerOrNumberSizes)
      Model->getPrimitiveType(Type, Size);
  // Create float types
  for (auto &Size : FloatSizes)
    Model->getPrimitiveType(Float, Size);
  // Create generic types
  for (auto &Size : GenericSizes)
    Model->getPrimitiveType(Generic, Size);

  // Finally, add void.
  Model->getPrimitiveType(Void, 0);
}
