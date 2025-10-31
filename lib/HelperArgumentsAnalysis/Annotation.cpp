/// \file Annotation.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"

#include "revng/HelperArgumentsAnalysis/Annotation.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

namespace aua {

llvm::MDNode &
Annotation::serializeToMetadata(llvm::LLVMContext &Context) const {
  QuickMetadata QMD(Context);

  auto ToTuple = [&QMD](const OffsetAndSizeSet &Set) {
    SmallVector<llvm::Metadata *, 2> Entries;
    for (auto &[Offset, Size] : Set)
      Entries.push_back(QMD.tuple({ QMD.get(Offset), QMD.get(Size) }));
    return QMD.tuple(Entries);
  };

  return *QMD.tuple({ QMD.get(Escapes), ToTuple(Reads), ToTuple(Writes) });
}

void Annotation::serialize(llvm::User &ToAnnotate) {
  auto &MDAnnotation = serializeToMetadata(getContext(&ToAnnotate));
  if (auto *F = dyn_cast<llvm::Function>(&ToAnnotate)) {
    F->setMetadata(MetadataKind, &MDAnnotation);
  } else if (auto *I = dyn_cast<llvm::Instruction>(&ToAnnotate)) {
    I->setMetadata(MetadataKind, &MDAnnotation);
  }
}

Annotation Annotation::deserializeFromMetadata(llvm::LLVMContext &Context,
                                               llvm::MDNode &MD) {
  using llvm::MDTuple;
  Annotation Result;
  QuickMetadata QMD(Context);

  auto FromTuple = [&QMD](const MDTuple &Tuple) -> OffsetAndSizeSet {
    OffsetAndSizeSet Result;
    for (llvm::Metadata *MD : Tuple.operands()) {
      Result.insert({ QMD.extract<uint64_t>(MD, 0),
                      QMD.extract<uint64_t>(MD, 1) });
    }
    return Result;
  };

  Result.Escapes = QMD.extract<bool>(&MD, 0);
  Result.Reads = FromTuple(*QMD.extract<MDTuple *>(&MD, 1));
  Result.Writes = FromTuple(*QMD.extract<MDTuple *>(&MD, 2));

  return Result;
}

bool Annotation::isAnnotated(llvm::Instruction &I) {
  return I.getMetadata(MetadataKind) != nullptr;
}

std::optional<Annotation> Annotation::deserialize(llvm::User &ToAnnotate) {

  MDNode *MD = nullptr;
  if (auto *F = dyn_cast<llvm::Function>(&ToAnnotate)) {
    MD = F->getMetadata(MetadataKind);
  } else if (auto *I = dyn_cast<llvm::Instruction>(&ToAnnotate)) {
    MD = I->getMetadata(MetadataKind);
  } else {
    revng_abort();
  }

  if (MD == nullptr)
    return std::nullopt;

  return deserializeFromMetadata(getContext(&ToAnnotate), *MD);
}

} // namespace aua
