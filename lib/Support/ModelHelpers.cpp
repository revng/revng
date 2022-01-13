#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include "revng-c/Support/ModelHelpers.h"

model::QualifiedType
peelConstAndTypedefs(const model::QualifiedType &QT, model::VerifyHelper &VH) {
  model::QualifiedType Result = QT;

  while (true) {

    auto QIt = llvm::find_if(Result.Qualifiers, [](const model::Qualifier &Q) {
      return Q.isArrayQualifier() or Q.isPointerQualifier();
    });
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
