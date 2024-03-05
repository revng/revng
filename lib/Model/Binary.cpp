/// \file Binary.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <queue>
#include <system_error>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Qualifier.h"
#include "revng/Model/TypeSystemPrinter.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/OverflowSafeInt.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/Tracking.h"

using namespace llvm;

namespace {

// TODO: all this logic should be moved to lib/TupleTree
Logger<> FieldAccessedLogger("field-accessed");

constexpr const char *StructNameHelpText = "regex that will make the program "
                                           "assert when a model struct which "
                                           "name matches this option is "
                                           "accessed. NOTE: enable "
                                           "field-accessed logger, optionally "
                                           "break on onFieldAccess from gdb.";
cl::opt<std::string> StructNameRegex("tracking-debug-struct-name",
                                     cl::desc(StructNameHelpText),
                                     cl::init(""),
                                     cl::cat(MainCategory));
constexpr const char *FieldNameHelpText = "regex that will "
                                          "make the "
                                          "program assert when "
                                          "a field "
                                          "of a model struct "
                                          "which name "
                                          "matches this "
                                          "option accessed. NOTE: enable "
                                          "field-accessed logger, optionally "
                                          "break on onFieldAccess from gdb.";

cl::opt<std::string> FieldNameRegex("tracking-debug-field-name",
                                    cl::desc(FieldNameHelpText),
                                    cl::init(""),
                                    cl::cat(MainCategory));

} // namespace

void onFieldAccess(StringRef FieldName, StringRef StructName) {
  if (FieldAccessedLogger.isEnabled()) {
    FieldAccessedLogger << (StructName + "::" + FieldName + " accessed").str();
    {
      auto LLVMStream = FieldAccessedLogger.getAsLLVMStream();
      llvm::sys::PrintStackTrace(*LLVMStream);
    }

    FieldAccessedLogger << DoLog;
  }
}

void fieldAccessed(StringRef FieldName, StringRef StructName) {
  if (StructNameRegex == "" and FieldNameRegex == "")
    return;

  Regex Reg(StructNameRegex);
  if (StructNameRegex != "" and not Reg.match(StructName))
    return;

  Regex Reg2(FieldNameRegex);
  if (FieldNameRegex != "" and not Reg2.match(FieldName))
    return;

  onFieldAccess(FieldName, StructName);
}

static std::string toIdentifier(const MetaAddress &Address) {
  return model::Identifier::sanitize(Address.toString()).str().str();
}

namespace model {

MetaAddressRangeSet Binary::executableRanges() const {
  MetaAddressRangeSet ExecutableRanges;

  struct Entry {
    Entry(MetaAddress Start, const model::TypeDefinition *Type) :
      Start(Start), Type(Type) {}
    MetaAddress Start;
    const model::TypeDefinition *Type = nullptr;
  };
  std::queue<Entry> Queue;

  for (const model::Segment &Segment : Segments()) {
    if (Segment.IsExecutable()) {
      if (not Segment.Type().empty()) {
        Queue.emplace(Segment.StartAddress(), Segment.Type().get());
      } else {
        ExecutableRanges.add(Segment.StartAddress(), Segment.endAddress());
      }
    }
  }

  while (not Queue.empty()) {
    auto QueueEntry = Queue.front();
    Queue.pop();

    auto *Struct = dyn_cast<model::StructDefinition>(QueueEntry.Type);

    if (not Struct or not Struct->CanContainCode())
      continue;

    MetaAddress PaddingStart = QueueEntry.Start;
    MetaAddress PaddingEnd;
    model::VerifyHelper Helper;

    for (const model::StructField &Field : Struct->Fields()) {
      // Record the start address of field
      MetaAddress FieldStart = QueueEntry.Start + Field.Offset();

      // Update the end of padding
      PaddingEnd = FieldStart;

      // Register the padding as an executable range
      if (PaddingStart != PaddingEnd)
        ExecutableRanges.add(PaddingStart, PaddingEnd);

      // Enqueue the field type for processing
      if (Field.Type().Qualifiers().size() == 0)
        Queue.emplace(FieldStart, Field.Type().UnqualifiedType().get());

      // Set the next padding start
      auto FieldSize = *Field.Type().size(Helper);
      PaddingStart = FieldStart + FieldSize;
    }

    // Record the trailing padding, if any
    PaddingEnd = QueueEntry.Start + Struct->Size();
    if (PaddingStart != PaddingEnd)
      ExecutableRanges.add(PaddingStart, PaddingEnd);
  }

  return ExecutableRanges;
}

model::DefinitionReference Binary::getPrimitiveType(PrimitiveKind::Values V,
                                                    uint8_t ByteSize) {
  PrimitiveDefinition Temporary(V, ByteSize);
  TypeDefinition::Key PrimitiveKey{ Temporary.ID(),
                                    TypeDefinitionKind::PrimitiveDefinition };
  auto It = TypeDefinitions().find(PrimitiveKey);

  // If we couldn't find it, create it
  if (It == TypeDefinitions().end())
    It = TypeDefinitions().emplace(new PrimitiveDefinition(V, ByteSize)).first;

  return getDefinitionReference(It->get());
}

model::DefinitionReference Binary::getPrimitiveType(PrimitiveKind::Values V,
                                                    uint8_t ByteSize) const {
  PrimitiveDefinition Temporary(V, ByteSize);
  TypeDefinition::Key PrimitiveKey{ Temporary.ID(),
                                    TypeDefinitionKind::PrimitiveDefinition };
  return getDefinitionReference(TypeDefinitions().at(PrimitiveKey).get());
}

uint64_t Binary::getAvailableTypeID() const {
  uint64_t Result = 0;

  if (not TypeDefinitions().empty())
    Result = TypeDefinitions().rbegin()->get()->ID() + 1;

  Result = std::max(model::PrimitiveDefinition::FirstNonPrimitiveID, Result);
  return Result;
}

DefinitionReference Binary::recordNewType(model::UpcastableTypeDefinition &&T) {
  if (not isa<PrimitiveDefinition>(T.get())) {
    // Assign progressive ID
    revng_assert(T->ID() == 0);
    T->ID() = getAvailableTypeID();
  }

  auto [It, Success] = TypeDefinitions().insert(T);
  revng_assert(Success);
  return getDefinitionReference(It->get());
}

void Binary::dumpTypeGraph(const char *Path) const {
  DisableTracking Guard(*this);

  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC);
  if (EC)
    revng_abort(EC.message().c_str());

  TypeSystemPrinter TSPrinter(Out);
  TSPrinter.print(*this);
}

std::string Binary::toString() const {
  DisableTracking Guard(*this);
  std::string S;
  llvm::raw_string_ostream OS(S);
  serialize(OS, *this);
  return S;
}

Identifier Function::name() const {
  using llvm::Twine;
  if (not CustomName().empty()) {
    return CustomName();
  } else {
    auto AutomaticName = (Twine("_function_") + toIdentifier(Entry())).str();
    return Identifier(AutomaticName);
  }
}

static const model::DefinitionReference &
prototypeOr(const model::DefinitionReference &Prototype,
            const model::DefinitionReference &Default) {
  if (not Prototype.empty()) {
    revng_assert(Prototype.isValid());
    return Prototype;
  }

  return Default;
}

// \note This function triggers a lot of materializations of
// std::vector::insert, which is heavy on build time
model::QualifiedType
model::QualifiedType::getPointerTo(model::Architecture::Values Arch) const {
  QualifiedType Result = *this;
  Result.Qualifiers().insert(Result.Qualifiers().begin(),
                             model::Qualifier::createPointer(Arch));
  return Result;
}

model::DefinitionReference Function::prototype(const model::Binary &B) const {
  model::DefinitionReference Result;
  auto ThePrototype = prototypeOr(Prototype(), B.DefaultPrototype());
  if (not ThePrototype.empty())
    return model::QualifiedType::getFunctionType(ThePrototype).value();
  else
    return Result;
}

Identifier DynamicFunction::name() const {
  using llvm::Twine;
  if (not CustomName().empty()) {
    return CustomName();
  } else {
    auto AutomaticName = (Twine("_dynamic_") + OriginalName()).str();
    return Identifier(AutomaticName);
  }
}

model::DefinitionReference
DynamicFunction::prototype(const model::Binary &Root) const {
  auto ThePrototype = prototypeOr(Prototype(), Root.DefaultPrototype());
  return model::QualifiedType::getFunctionType(ThePrototype).value();
}

Identifier Segment::name() const {
  using llvm::Twine;
  if (not CustomName().empty()) {
    return CustomName();
  } else {
    auto AutomaticName = (Twine("_segment_") + toIdentifier(StartAddress())
                          + "_" + Twine(VirtualSize()))
                           .str();
    return Identifier(AutomaticName);
  }
}

void Function::dumpTypeGraph(const char *Path) const {
  DisableTracking Guard(*this);
  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC);
  if (EC)
    revng_abort(EC.message().c_str());

  TypeSystemPrinter TSPrinter(Out);
  TSPrinter.print(*this);
}

namespace RelocationType {

Values fromELFRelocation(model::Architecture::Values Architecture,
                         unsigned char ELFRelocation) {
  using namespace llvm::ELF;
  switch (Architecture) {
  case model::Architecture::x86:
    switch (ELFRelocation) {
    case R_386_RELATIVE:
    case R_386_32:
      return AddAbsoluteAddress32;

    case R_386_JUMP_SLOT:
    case R_386_GLOB_DAT:
      return WriteAbsoluteAddress32;

    case R_386_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  case model::Architecture::x86_64:
    switch (ELFRelocation) {
    case R_X86_64_RELATIVE:
      return AddAbsoluteAddress64;

    case R_X86_64_JUMP_SLOT:
    case R_X86_64_GLOB_DAT:
    case R_X86_64_64:
      return WriteAbsoluteAddress64;

    case R_X86_64_32:
      return WriteAbsoluteAddress32;

    case R_X86_64_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  case model::Architecture::arm:
    switch (ELFRelocation) {
    case R_ARM_RELATIVE:
      return AddAbsoluteAddress32;

    case R_ARM_JUMP_SLOT:
    case R_ARM_GLOB_DAT:
      return WriteAbsoluteAddress32;

    case R_ARM_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  case model::Architecture::aarch64:
    return Invalid;

  case model::Architecture::mips:
  case model::Architecture::mipsel:
    switch (ELFRelocation) {
    case R_MIPS_IMPLICIT_RELATIVE:
      return AddAbsoluteAddress32;

    case R_MIPS_JUMP_SLOT:
    case R_MIPS_GLOB_DAT:
      return WriteAbsoluteAddress32;

    case R_MIPS_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  case model::Architecture::systemz:
    switch (ELFRelocation) {
    case R_390_GLOB_DAT:
      return WriteAbsoluteAddress64;

    case R_390_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  default:
    revng_abort();
  }
}

bool isELFRelocationBaseRelative(model::Architecture::Values Architecture,
                                 unsigned char ELFRelocation) {
  using namespace llvm::ELF;
  switch (Architecture) {
  case model::Architecture::x86:
    switch (ELFRelocation) {
    case R_386_RELATIVE:
      return true;

    case R_386_32:
    case R_386_JUMP_SLOT:
    case R_386_GLOB_DAT:
      return false;

    case R_386_COPY:
      // TODO: use

    default:
      return Invalid;
    }

  case model::Architecture::x86_64:
    switch (ELFRelocation) {
    case R_X86_64_RELATIVE:
      return true;

    case R_X86_64_JUMP_SLOT:
    case R_X86_64_GLOB_DAT:
    case R_X86_64_64:
    case R_X86_64_32:
      return false;

    case R_X86_64_COPY:
      // TODO: use

    default:
      return Invalid;
    }

  case model::Architecture::arm:
    switch (ELFRelocation) {
    case R_ARM_RELATIVE:
      return true;

    case R_ARM_JUMP_SLOT:
    case R_ARM_GLOB_DAT:
      return false;

    case R_ARM_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  case model::Architecture::aarch64:
    return Invalid;

  case model::Architecture::mips:
  case model::Architecture::mipsel:
    switch (ELFRelocation) {
    case R_MIPS_IMPLICIT_RELATIVE:
      return true;

    case R_MIPS_JUMP_SLOT:
    case R_MIPS_GLOB_DAT:
      return false;

    case R_MIPS_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  case model::Architecture::systemz:
    switch (ELFRelocation) {
    case R_390_GLOB_DAT:
      return false;

    case R_390_COPY:
      // TODO: use
    default:
      return Invalid;
    }

  default:
    revng_abort();
  }
}

Values formCOFFRelocation(model::Architecture::Values Architecture) {
  switch (Architecture) {
  case model::Architecture::x86:
  case model::Architecture::arm:
  case model::Architecture::mips:
  case model::Architecture::mipsel:
    return WriteAbsoluteAddress32;

  case model::Architecture::x86_64:
  case model::Architecture::aarch64:
  case model::Architecture::systemz:
    return WriteAbsoluteAddress64;
  default:
    revng_abort();
  }
}

} // namespace RelocationType

} // namespace model

void dumpModel(const model::Binary &Model, const char *Path) debug_function;

void dumpModel(const model::Binary &Model, const char *Path) {
  std::error_code EC;
  raw_fd_stream Stream(Path, EC);
  revng_assert(not EC);
  serialize(Stream, Model);
}

void dumpModel(const TupleTree<model::Binary> &Model,
               const char *Path) debug_function;

void dumpModel(const TupleTree<model::Binary> &Model, const char *Path) {
  dumpModel(*Model, Path);
}
