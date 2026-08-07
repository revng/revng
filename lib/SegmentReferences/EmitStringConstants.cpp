//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringExtras.h"

#include "revng/Model/PrimitiveType.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/SegmentReferences/EmitStringConstants.h"
#include "revng/SegmentReferences/SegmentUsesEnumerator.h"
#include "revng/SegmentReferences/StringConstants.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/Unicode.h"

using namespace llvm;

static Logger Log("emit-string-constants");

/// A type covering an address, and how far into it that address falls.
struct TypeAtOffset {
  const model::Type *Type = nullptr;
  uint64_t Offset = 0;
};

// TODO: consider to build a cache
static RecursiveCoroutine<void> processType(const MetaAddress &Target,
                                            const MetaAddress &TypeStartAddress,
                                            const model::Type *CurrentType,
                                            SmallVector<TypeAtOffset> &Result) {
  revng_log(Log,
            "Processing the following type starting at "
              << TypeStartAddress.toString() << "\n"
              << CurrentType->toString());
  LoggerIndent Indent(Log);

  revng_assert(CurrentType != nullptr);
  revng_assert(Target >= TypeStartAddress);
  Result.push_back({ CurrentType, (Target - TypeStartAddress).value() });

  if (auto *Array = CurrentType->skipConstAndTypedefs()->getArray()) {
    const model::Type &ElementType = Array->getArrayElement();
    uint64_t ElementSize = ElementType.size().value();
    revng_assert(ElementSize != 0);

    // The division floors, so this is the element the address falls in, at
    // whatever offset within it, and the recursion carries that offset down.
    uint64_t Index = (Target - TypeStartAddress).value() / ElementSize;

    // Nothing says the type of a segment covers all of it, so an address can
    // fall past the end of the array. This is the same bound the struct case
    // gets from asking each field whether it contains the address.
    if (Index < Array->ElementCount()) {
      MetaAddress ElementStart = TypeStartAddress + Index * ElementSize;
      revng_assert(ElementStart.isValid());

      revng_log(Log, "Descending into element #" << Index);
      rc_recur processType(Target, ElementStart, &ElementType, Result);
    }
  } else if (auto *Struct = CurrentType->skipConstAndTypedefs()->getStruct()) {
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

/// The types covering \p Target, outermost first, each with how far into it
/// \p Target falls.
static SmallVector<TypeAtOffset> typesAt(const model::Binary &Model,
                                         const MetaAddress &Target) {
  SmallVector<TypeAtOffset> Result;
  auto [Segment, _] = Model.getSegmentFor(Target);
  if (Segment == nullptr or Segment->Type().isEmpty())
    return Result;

  const model::Type &SegmentType = *Segment->Type();
  processType(Target, Segment->StartAddress(), &SegmentType, Result);
  return Result;
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
                                    const TypeAtOffset &Type);
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
    for (const TypeAtOffset &Type : typesAt(Binary, Address)) {
      revng_log(Log, "Considering " << Type.Type->toDebugString());
      LoggerIndent Indent(Log);
      if (auto String = getStringOfTypeAt(Address, Type); not String.empty()) {
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
                                       const TypeAtOffset &Type) {
  unsigned CharSize = getConstCharArrayElementSize(*Type.Type);
  if (CharSize == 0) {
    revng_log(Log, "Ignoring unsuitable type: " << Type.Type->toDebugString());
    return {};
  }

  // A reference into the middle of a string is a reference to the rest of it,
  // which is what suffix sharing looks like: `"hello world" + 6` is `"world"`.
  // A reference landing mid-character names no string at all.
  if (Type.Offset % CharSize != 0) {
    revng_log(Log, "Ignoring a reference to the middle of a character");
    return {};
  }

  // This is a char array! Let's now extract the data.
  uint64_t ByteCount = Type.Type->size().value() - Type.Offset;
  UnicodeCStringView String = readString(BinaryView,
                                         Address,
                                         ByteCount,
                                         CharSize);
  if (not String.isValid())
    return {};

  return String.data();
}

namespace revng::pypeline::piperuns {

void EmitStringConstants::runOnLLVMFunction(const model::Function &Function,
                                            llvm::Function &LLVMFunction) {
  ::EmitStringConstants Replacer(Binary, BinaryView);
  Replacer.run(*LLVMFunction.getParent(), &LLVMFunction);
}

} // namespace revng::pypeline::piperuns
