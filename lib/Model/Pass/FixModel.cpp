/// \file FixModel.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Pass/FixModel.h"
#include "revng/Model/Pass/RegisterModelPass.h"
#include "revng/Model/Processing.h"
#include "revng/Model/Type.h"

using namespace llvm;

static RegisterModelPass
  R("fix", "Remove all invalid types from the model.", model::fixModel);

static Logger<> ModelFixLogger("model-fix");
using namespace model;

template<typename T>
bool filterZeroSizedElements(T *AggregateType) {
  // Filter out aggregates with zero-sized members.
  auto FieldIt = AggregateType->Fields().begin();
  auto FieldEnd = AggregateType->Fields().end();
  for (; FieldIt != FieldEnd; ++FieldIt) {
    auto &Field = *FieldIt;
    auto MaybeSize = Field.Type().trySize();
    if (!MaybeSize || *MaybeSize == 0) {
      return true;
    }
  }
  return false;
}

static bool shouldDrop(UpcastablePointer<model::Type> &T) {
  // Filter out empty structs and unions.
  if (isa<model::StructType>(T.get()) or isa<model::UnionType>(T.get())) {
    if (!T->size())
      return true;

    if (auto *Struct = dyn_cast<model::StructType>(T.get()))
      if (filterZeroSizedElements(Struct))
        return true;
    if (auto *Union = dyn_cast<UnionType>(T.get()))
      if (filterZeroSizedElements(Union))
        return true;
  }

  // Filter out empty arrays.
  for (const model::QualifiedType &QT : T->edges()) {
    auto IsEmptyArray = [](const model::Qualifier &Q) {
      return Q.Kind() == model::QualifierKind::Array && Q.Size() == 0;
    };
    auto Iterator = llvm::find_if(QT.Qualifiers(), IsEmptyArray);
    if (Iterator != QT.Qualifiers().end())
      return true;
  }

  // Filter out invalid PrimitiveTypes.
  auto *ThePrimitiveType = dyn_cast<PrimitiveType>(T.get());
  if (ThePrimitiveType) {
    if (ThePrimitiveType->PrimitiveKind() == PrimitiveTypeKind::Invalid)
      return true;
  }

  // Filter out invalid functions.
  auto *FunctionType = dyn_cast<CABIFunctionType>(T.get());
  if (FunctionType) {
    // Remove functions with 0-sized arguments
    for (auto &Group : llvm::enumerate(FunctionType->Arguments())) {
      auto &Argument = Group.value();
      if (not Argument.Type().size())
        return true;
    }
  }

  return false;
}

void model::fixModel(TupleTree<model::Binary> &Model) {
  std::set<const model::Type *> ToDrop;

  for (UpcastablePointer<model::Type> &T : Model->Types()) {
    if (shouldDrop(T))
      ToDrop.insert(T.get());
  }

  unsigned DroppedTypes = dropTypesDependingOnTypes(Model, ToDrop);
  revng_log(ModelFixLogger, "Purging " << DroppedTypes << " types.");
}
