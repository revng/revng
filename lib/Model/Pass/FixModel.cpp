/// \file FixModel.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Pass/FixModel.h"
#include "revng/Model/Pass/RegisterModelPass.h"
#include "revng/Model/Processing.h"

using namespace llvm;

static RegisterModelPass
  R("fix", "Remove all invalid types from the model.", model::fixModel);

static Logger<> ModelFixLogger("model-fix");

void model::fixModel(TupleTree<model::Binary> &Model) {
  std::set<const model::Type *> ToDrop;

  for (UpcastablePointer<model::Type> &T : Model->Types) {
    // Filter out empty structs and unions.
    if (!T->size()) {
      if (isa<StructType>(T.get()) or isa<UnionType>(T.get())) {
        ToDrop.insert(T.get());
        continue;
      }
    }

    // Filter out invalid PrimitiveTypes.
    auto *ThePrimitiveType = dyn_cast<PrimitiveType>(T.get());
    if (ThePrimitiveType) {
      if (ThePrimitiveType->PrimitiveKind == PrimitiveTypeKind::Invalid)
        ToDrop.insert(T.get());
    }

    // Filter out invalid functions.
    auto *FunctionType = dyn_cast<CABIFunctionType>(T.get());
    if (FunctionType) {
      // Remove functions with more than one `void` argument.
      for (auto &Group : llvm::enumerate(FunctionType->Arguments)) {
        auto &Argument = Group.value();
        if (*Argument.Type.size() == 0 and FunctionType->Arguments.size() > 1)
          ToDrop.insert(T.get());
      }
    }
  }

  unsigned DroppedTypes = dropTypesDependingOnTypes(Model, ToDrop);
  revng_log(ModelFixLogger, "Purging " << DroppedTypes << " types.");
}
