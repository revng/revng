//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine-coroutine.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Binary.h"
#include "revng/Model/GlobalVariableBuilder.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/Debug.h"

using namespace llvm;

static Logger Log("model-global-variable-builder");

model::GlobalVariableBuilder::GlobalVariableBuilder(model::Binary &Binary) :
  Binary(Binary) {
  for (auto &Type : Binary.TypeDefinitions())
    for (const model::Type *EdgeType : Type->edges())
      if (auto *Definition = EdgeType->tryGetAsDefinition())
        ++Instances[Definition];
}

// TODO: consider to build a cache like map<MetaAddress, StructDefinition>
static RecursiveCoroutine<std::pair<model::StructDefinition *, uint64_t>>
processType(const std::map<const model::TypeDefinition *, uint64_t> &Instances,
            const MetaAddressRange &TargetRange,
            const MetaAddress &TypeStartAddress,
            model::Type *CurrentType) {
  static const std::pair<model::StructDefinition *, uint64_t> Fail(nullptr, 0);

  revng_log(Log,
            "Processing the following type starting at "
              << TypeStartAddress.toString() << "\n"
              << CurrentType->toString());
  LoggerIndent Indent(Log);

  revng_assert(CurrentType != nullptr);
  auto *Struct = CurrentType->skipConstAndTypedefs()->getStruct();
  if (Struct == nullptr) {
    revng_log(Log,
              "The struct we found has an instance in multiple data "
              "structures, ignoring");
    rc_return Fail;
  }

  // Check if this struct is instantiated in more than one place
  // TODO: replace this logic with an "singleton" attribute for structs
  auto It = Instances.find(Struct);
  if (It != Instances.end() and It->second > 1)
    rc_return Fail;

  for (model::StructField &Field : Struct->Fields()) {
    MetaAddress FieldStart = TypeStartAddress + Field.Offset();
    auto Size = Field.Type()->size().value();
    MetaAddress FieldEnd = FieldStart + Size;
    revng_assert(FieldStart.isValid() and FieldEnd.isValid());
    MetaAddressRange FieldRange(FieldStart, FieldEnd);

    revng_log(Log,
              "Considering field starting at " << FieldStart.toString()
                                               << " of size " << Size);

    if (FieldRange.contains(TargetRange)) {
      // This field contains TargetAddress, recur
      rc_return rc_recur processType(Instances,
                                     TargetRange,
                                     FieldStart,
                                     Field.Type().get());
    } else if (FieldRange.overlaps(TargetRange)) {
      // Partial overlap, bail out
      revng_log(Log, "Partial overlap, bailing out");
      rc_return Fail;
    }
  }

  // If we get here, no field even partially overlaps with the TargetRange.
  // We can therefore inject the field.
  auto Offset = (TargetRange.start() - TypeStartAddress).value();
  revng_log(Log, "Match!");
  rc_return{ Struct, Offset };
}

bool model::GlobalVariableBuilder::insert(const MetaAddress &Address,
                                          model::UpcastableType &&Type) {
  revng_assert(Address.isValid());
  revng_log(Log,
            "Inserting " << Type->toDebugString() << " at "
                         << Address.toString());
  LoggerIndent Indent(Log);

  auto [Segment, Offset] = Binary.getSegmentFor(Address);
  if (Segment == nullptr) {
    revng_log(Log, "Segment not found");
    return false;
  } else {
    revng_log(Log,
              "Found segment starting at "
                << Segment->StartAddress().toString());
  }

  if (Segment->Type().isEmpty()) {
    revng_log(Log, "The segment has not type, ignoring");
    return false;
  }

  MetaAddress EndAddress = Address + Type->size().value();
  revng_assert(EndAddress.isValid());

  MetaAddressRange NewFieldRange = { Address, EndAddress };
  auto [Struct, FieldOffset] = rc_eval(processType(Instances,
                                                   NewFieldRange,
                                                   Segment->StartAddress(),
                                                   Segment->Type().get()));
  if (Struct == nullptr) {
    revng_log(Log, "Can't find a place to insert the field");
    return false;
  }

  auto StructReference = Binary.getTypeDefinitionReference(Struct->key());
  auto StructType = model::DefinedType::make(StructReference);

  revng_log(Log,
            "We can insert the field at offset "
              << FieldOffset << " in " << StructType->toDebugString());

  model::StructField NewField;
  Struct->addField(FieldOffset, std::move(Type));

  revng_log(Log, "Added");

  return true;
}
