//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"

#include "revng/ADT/RecursiveCoroutine-coroutine.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Qualifier.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/Support/ModelHelpers.h"

using llvm::dyn_cast;
using QualKind = model::QualifierKind::Values;
using model::TypedefType;

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
  using namespace model::PrimitiveTypeKind;

  const llvm::Type *TypeToConvert = LLVMType;
  size_t NPtrQualifiers = 0;

  // If it's a pointer, find the pointed type
  while (auto *PtrType = dyn_cast<llvm::PointerType>(TypeToConvert)) {
    TypeToConvert = PtrType->getElementType();
    ++NPtrQualifiers;
  }

  model::QualifiedType ModelType;

  if (auto *IntType = dyn_cast<llvm::IntegerType>(TypeToConvert)) {
    // Convert the integer type
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
    // Add qualifiers
    for (size_t I = 0; I < NPtrQualifiers; ++I)
      addPointerQualifier(ModelType, Model);

  } else if (NPtrQualifiers > 0) {
    // If it's a pointer to a non-integer type, return an integer type of the
    // length of a pointer
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

llvm::Constant *
serializeToLLVMString(model::QualifiedType &QT, llvm::Module &M) {
  // Create a string containing a serialization of the model type
  std::string SerializedQT;
  {
    llvm::raw_string_ostream StringStream(SerializedQT);
    llvm::yaml::Output YAMLOutput(StringStream);
    YAMLOutput << QT;
  }

  // Build a constant global string containing the serialized type
  return buildStringPtr(&M, SerializedQT, "");
}

RecursiveCoroutine<model::QualifiedType>
dropPointer(const model::QualifiedType &QT) {
  model::QualifiedType NewQT = QT;

  auto It = std::find_if(NewQT.Qualifiers.rbegin(),
                         NewQT.Qualifiers.rend(),
                         model::Qualifier::isPointer);

  if (It != NewQT.Qualifiers.rend()) {
    std::erase(NewQT.Qualifiers, *It);
    rc_return NewQT;
  }

  if (auto *TD = dyn_cast<TypedefType>(NewQT.UnqualifiedType.getConst()))
    rc_return rc_recur dropPointer(TD->UnderlyingType);

  revng_abort("Cannot dropPointer, QT does not have pointer qualifiers");

  rc_return{};
}
