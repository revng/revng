//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringExtras.h"

#include "revng/Model/PrimitiveType.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/SegmentReferences/EmitStringConstants.h"
#include "revng/SegmentReferences/SegmentUsesEnumerator.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/Unicode.h"

using namespace llvm;

static Logger Log("emit-string-constants");

// TODO: consider to build a cache
static RecursiveCoroutine<void>
processType(const MetaAddress &Target,
            const MetaAddress &TypeStartAddress,
            const model::Type *CurrentType,
            SmallVector<const model::Type *> &Result) {
  revng_log(Log,
            "Processing the following type starting at "
              << TypeStartAddress.toString() << "\n"
              << CurrentType->toString());
  LoggerIndent Indent(Log);

  if (Target == TypeStartAddress) {
    revng_log(Log, "Recording in results");
    Result.push_back(CurrentType);
  }

  revng_assert(CurrentType != nullptr);
  if (auto *Struct = CurrentType->skipConstAndTypedefs()->getStruct()) {
    for (const model::StructField &Field : Struct->Fields()) {
      MetaAddress FieldStart = TypeStartAddress + Field.Offset();
      auto Size = Field.Type()->size().value();
      MetaAddress FieldEnd = FieldStart + Size;
      revng_assert(FieldStart.isValid() and FieldEnd.isValid());
      MetaAddressRange FieldRange(FieldStart, FieldEnd);

      revng_log(Log,
                "Considering field starting at " << FieldStart.toString()
                                                 << " of size " << Size);

      if (FieldRange.contains(Target)) {
        // This field contains Target, recur
        rc_recur processType(Target, FieldStart, Field.Type().get(), Result);
      }
    }
  } else if (auto *Union = CurrentType->skipConstAndTypedefs()->getUnion()) {
    for (auto &[Index, Field] : llvm::enumerate(Union->Fields())) {
      auto Size = Field.Type()->size().value();
      MetaAddress FieldEnd = TypeStartAddress + Size;
      revng_assert(FieldEnd.isValid());
      MetaAddressRange FieldRange(TypeStartAddress, FieldEnd);

      revng_log(Log,
                "Considering union entry #" << Index << " of size " << Size);

      if (FieldRange.contains(Target)) {
        // This field contains Target, recur
        rc_recur processType(Target,
                             TypeStartAddress,
                             Field.Type().get(),
                             Result);
      }
    }
  }
}

static SmallVector<const model::Type *> typesAt(const model::Binary &Model,
                                                const MetaAddress &Target) {
  SmallVector<const model::Type *> Result;
  auto [Segment, _] = Model.getSegmentFor(Target);
  if (Segment == nullptr or Segment->Type().isEmpty())
    return Result;

  const model::Type &SegmentType = *Segment->Type();
  processType(Target, Segment->StartAddress(), &SegmentType, Result);
  return Result;
}

/// \return 1 for uint8_t const[], 2 for uint16_t const [], 0 otherwise.
static unsigned getConstCharArrayElementSize(const model::Type *Type) {
  Type = Type->skipConstAndTypedefs();

  const model::ArrayType *Array = Type->getArray();
  if (Array == nullptr)
    return 0;

  const model::Type &ElementType = Array->getArrayElement();
  const model::PrimitiveType *PrimitiveType = ElementType.getPrimitive();
  if (not ElementType.IsConst() or PrimitiveType == nullptr
      or PrimitiveType->PrimitiveKind() != model::PrimitiveKind::Unsigned) {
    return 0;
  }

  if (PrimitiveType->Size() != 1 and PrimitiveType->Size() != 2)
    return 0;

  return PrimitiveType->Size();
}

class EmitStringConstants {
private:
  const model::Binary &Binary;
  RawBinaryView &BinaryView;
  SegmentUsesEnumerator SegmentUses;

public:
  EmitStringConstants(const model::Binary &Binary, RawBinaryView &BinaryView) :
    Binary(Binary),
    BinaryView(BinaryView),
    SegmentUses(Binary, SegmentUsesEnumerator::SegmentAccess::ReadOnly) {}

  void run(llvm::Module &M, llvm::Function *LimitTo = nullptr);

private:
  llvm::StringRef getStringOfTypeAt(const MetaAddress &Address,
                                    const model::Type &Type);
};

void EmitStringConstants::run(llvm::Module &M, llvm::Function *LimitTo) {
  revng::IRBuilder B(M.getContext());

  for (auto &&SegmentUse : SegmentUses.getUses(M, LimitTo)) {
    revng_log(Log,
              "Considering segment use "
                << getName(SegmentUse.TheUse->getUser()) << ". Address is "
                << SegmentUse.Address.toString() << ".");
    LoggerIndent Indent(Log);

    const MetaAddress &Address = SegmentUse.Address;

    // Check if we have a uint{8,16}_t there
    // TODO: we should do this in bulk so we visit the model once only
    for (const model::Type *Type : typesAt(Binary, Address)) {
      revng_log(Log, "Considering " << Type->toDebugString());
      LoggerIndent Indent(Log);
      if (auto String = getStringOfTypeAt(Address, *Type); not String.empty()) {
        Constant *Global = getUniqueString(&M, String, false);

        Use *Use = SegmentUse.TheUse;
        llvm::Type *UseType = Use->get()->getType();

        if (UseType->isPointerTy()) {
          SegmentUse.TheUse->set(Global);
        } else {
          revng_assert(UseType->isIntegerTy());
          SegmentUse.TheUse->set(ConstantExpr::getPtrToInt(Global, UseType));
        }

        break;
      }
    }
  }
}

llvm::StringRef
EmitStringConstants::getStringOfTypeAt(const MetaAddress &Address,
                                       const model::Type &Type) {
  unsigned CharSize = getConstCharArrayElementSize(&Type);
  if (CharSize == 0) {
    revng_log(Log, "Ignoring unsuitable type: " << Type.toDebugString());
    return {};
  }

  // This is a char array! Let's now extract the data.
  auto ByteCount = Type.size().value();
  auto MaybeData = BinaryView.getByAddress(Address, ByteCount);
  if (not MaybeData.has_value()) {
    revng_log(Log, "Couldn't get the data");
    return {};
  }

  auto String = UnicodeCStringView::getPrintable(*MaybeData);

  if (not String.isValid()) {
    revng_log(Log, "No printable string found");
    return {};
  }

  if (String.charSize() != CharSize) {
    revng_log(Log, "Unexpected char size for the string");
    return {};
  }

  if (String.data().size() != MaybeData->size()) {
    revng_log(Log, "String length does not match");
    return {};
  }

  return String.data();
}

namespace revng::pypeline::piperuns {

void EmitStringConstants::runOnLLVMFunction(const model::Function &Function,
                                            llvm::Function &LLVMFunction) {
  ::EmitStringConstants Replacer(Binary, BinaryView);
  Replacer.run(*LLVMFunction.getParent(), &LLVMFunction);
}

} // namespace revng::pypeline::piperuns
