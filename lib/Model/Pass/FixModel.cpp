/// \file FixModel.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Pass/FixModel.h"
#include "revng/Model/Pass/RegisterModelPass.h"
#include "revng/Model/Processing.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyTypeHelper.h"

using namespace llvm;

static RegisterModelPass
  R("fix", "Remove all invalid types from the model.", model::fixModel);

static Logger<> ModelFixLogger("model-fix");
using namespace model;

void model::fixModel(TupleTree<model::Binary> &Model) {
  std::set<const model::Type *> ToDrop;

  for (UpcastablePointer<model::Type> &T : Model->Types()) {
    // Filter out empty structs and unions.
    if (!T->size()) {
      if (isa<StructType>(T.get()) or isa<UnionType>(T.get())) {
        ToDrop.insert(T.get());
        continue;
      }
    }

    // Filter out empty arrays.
    bool FoundEmptyArray = false;
    for (const model::QualifiedType &QT : T->edges()) {
      auto IsEmptyArray = [](const model::Qualifier &Q) {
        return Q.Kind() == model::QualifierKind::Array && Q.Size() == 0;
      };
      auto Iterator = llvm::find_if(QT.Qualifiers(), IsEmptyArray);
      if (Iterator != QT.Qualifiers().end()) {
        ToDrop.insert(T.get());
        FoundEmptyArray = true;
      }
    }
    if (FoundEmptyArray)
      continue;

    // Filter out invalid PrimitiveTypes.
    auto *ThePrimitiveType = dyn_cast<PrimitiveType>(T.get());
    if (ThePrimitiveType) {
      if (ThePrimitiveType->PrimitiveKind() == PrimitiveTypeKind::Invalid)
        ToDrop.insert(T.get());
    }

    // Filter out invalid functions.
    auto *FunctionType = dyn_cast<CABIFunctionType>(T.get());
    if (FunctionType) {
      // Remove functions with more than one `void` argument.
      for (auto &Group : llvm::enumerate(FunctionType->Arguments())) {
        auto &Argument = Group.value();
        VoidConstResult VoidConst = isVoidConst(&Argument.Type());
        if (VoidConst.IsVoid) {
          if (FunctionType->Arguments().size() > 1) {
            // More than 1 void argument.
            ToDrop.insert(T.get());
            break;
          }

          if (VoidConst.IsConst) {
            // Cannot have const void argument.
            ToDrop.insert(T.get());
            break;
          }
        }
      }
    }
  }

  unsigned DroppedTypes = dropTypesDependingOnTypes(Model, ToDrop);
  revng_log(ModelFixLogger, "Purging " << DroppedTypes << " types.");
}
