#include <functional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"

#include "revng/ADT/RecursiveCoroutine-coroutine.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Generated/Early/QualifierKind.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Qualifier.h"

#include "revng-c/Support/ModelHelpers.h"

using llvm::dyn_cast;
using QualKind = model::QualifierKind::Values;

model::QualifiedType
peelConstAndTypedefs(const model::QualifiedType &QT, model::VerifyHelper &VH) {
  model::QualifiedType Result = QT;

  while (true) {

    const auto &NonConst = std::not_fn(model::Qualifier::isConst);
    auto QIt = llvm::find_if(Result.Qualifiers, NonConst);
    auto QEnd = Result.Qualifiers.end();

    if (QIt != QEnd)
      return model::QualifiedType(QT.UnqualifiedType, { QIt, QEnd });

    // If we reach this point, Result has no pointer nor array qualifiers.
    // Let's look at the Unqualified type and unwrap it if it's a typedef.
    using llvm::dyn_cast;
    if (auto *TD = dyn_cast<model::TypedefType>(Result.UnqualifiedType.get())) {
      Result = model::QualifiedType(Result.UnqualifiedType, {});
    } else {
      // If Result is not a typedef we can bail out
      break;
    }
  }
  revng_assert(Result.verify(VH));
  return Result;
}

model::TypePath createEmptyStruct(model::Binary &Binary, uint64_t Size) {
  using namespace model;
  revng_assert(Size > 0 and Size < std::numeric_limits<int64_t>::max());
  TypePath Path = Binary.recordNewType(makeType<model::StructType>());
  model::StructType *NewStruct = llvm::cast<model::StructType>(Path.get());
  NewStruct->Size = Size;
  return Path;
}

const model::QualifiedType
llvmIntToModelType(const llvm::Type *LLVMType, const model::Binary &Model) {
  model::QualifiedType ModelType;
  using namespace model::PrimitiveTypeKind;
  const llvm::Type *TypeToConvert = LLVMType;
  bool IsPointer = false;

  if (auto *PtrType = dyn_cast<llvm::PointerType>(LLVMType)) {
    TypeToConvert = PtrType->getElementType();
    IsPointer = true;
  }

  if (auto *IntType = dyn_cast<llvm::IntegerType>(TypeToConvert)) {
    switch (IntType->getIntegerBitWidth()) {
    case 1:
    case 8:
      ModelType.UnqualifiedType = Model.getPrimitiveType(Number, 1);

    case 16:
      ModelType.UnqualifiedType = Model.getPrimitiveType(Number, 2);
      break;

    case 32:
      ModelType.UnqualifiedType = Model.getPrimitiveType(Number, 4);
      break;

    case 64:
      ModelType.UnqualifiedType = Model.getPrimitiveType(Number, 8);
      break;

    case 128:
      ModelType.UnqualifiedType = Model.getPrimitiveType(Number, 16);
      break;

    default:
      revng_abort("Found an LLVM integer with a size that is not a power of "
                  "two");
    }

    if (IsPointer) {
      addPointerQualifier(ModelType, Model);
    }
  } else if (IsPointer) {
    // If it's a pointer to a non-integer type, return an integer
    auto PtrSize = getPointerSize(Model.Architecture);
    ModelType.UnqualifiedType = Model.getPrimitiveType(Number, PtrSize);
  } else {
    revng_abort("Only integer and pointer types can be directly converted "
                "from LLVM types to C types.");
  }

  return ModelType;
}

model::QualifiedType
parseQualifiedType(const llvm::StringRef QTString, const model::Binary &Model) {
  model::QualifiedType ParsedType;
  {
    llvm::yaml::Input YAMLInput(QTString);
    YAMLInput >> ParsedType;
    std::error_code EC = YAMLInput.error();
    if (EC)
      revng_abort("Could not deserialize the ModelGEP base type");
  }
  ParsedType.UnqualifiedType.Root = &Model;
  revng_assert(ParsedType.UnqualifiedType.isValid());

  return ParsedType;
}

model::QualifiedType dropPointer(const model::QualifiedType &QT) {
  auto QIt = QT.Qualifiers.end();
  auto QBegin = QT.Qualifiers.begin();
  while (QIt != QBegin) {
    --QIt;

    if (model::Qualifier::isConst(*QIt))
      continue;

    if (model::Qualifier::isPointer(*QIt)) {
      auto PrevIt = QIt == QBegin ? QBegin : std::prev(QIt);
      return model::QualifiedType(QT.UnqualifiedType, { QBegin, PrevIt });
    }

    return QT;
  }

  return QT;
}

RecursiveCoroutine<model::QualifiedType>
dropPointerRecursively(const model::QualifiedType &QT) {
  auto QIt = QT.Qualifiers.end();
  auto QBegin = QT.Qualifiers.begin();
  while (QIt != QBegin) {
    --QIt;

    if (model::Qualifier::isConst(*QIt))
      continue;

    if (model::Qualifier::isPointer(*QIt)) {
      auto PrevIt = QIt == QBegin ? QBegin : std::prev(QIt);
      rc_return model::QualifiedType(QT.UnqualifiedType, { QBegin, PrevIt });
    }

    rc_return QT;
  }

  if (auto *TD = dyn_cast<model::TypedefType>(QT.UnqualifiedType.get()))
    rc_return rc_recur dropPointerRecursively(TD->UnderlyingType);

  rc_return QT;
}
