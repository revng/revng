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

bool Binary::verifyTypeDefinitions() const {
  return verifyTypeDefinitions(false);
}

bool Binary::verifyTypeDefinitions(bool Assert) const {
  VerifyHelper VH(Assert);
  return verifyTypeDefinitions(VH);
}

bool Binary::verifyTypeDefinitions(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  // All types on their own should verify
  std::set<Identifier> Names;
  for (const model::UpcastableTypeDefinition &Definition : TypeDefinitions()) {
    // Verify the type
    if (not Definition.get()->verify(VH))
      return VH.fail();

    // Ensure the names are unique
    auto Name = Definition->name();
    if (not Names.insert(Name).second)
      return VH.fail(Twine("Multiple types with the following name: ") + Name);

    using CFT = model::CABIFunctionDefinition;
    using RFT = model::RawFunctionDefinition;
    if (const auto *T = llvm::dyn_cast<CFT>(Definition.get())) {
      if (getArchitecture(T->ABI()) != Architecture())
        return VH.fail("Function type architecture differs from the binary "
                       "architecture");
    } else if (const auto *T = llvm::dyn_cast<RFT>(Definition.get())) {
      if (T->Architecture() != Architecture())
        return VH.fail("Function type architecture differs from the binary "
                       "architecture");
    }
  }

  return true;
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

bool Binary::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Binary::verify() const {
  VerifyHelper VH(false);
  return verify(VH);
}

bool VerifyHelper::isGlobalSymbol(const model::Identifier &Name) const {
  return GlobalSymbols.count(Name) > 0;
}

bool VerifyHelper::registerGlobalSymbol(const model::Identifier &Name,
                                        const std::string &Path) {
  if (Name.empty())
    return true;

  auto It = GlobalSymbols.find(Name);
  if (It == GlobalSymbols.end()) {
    GlobalSymbols.insert({ Name, Path });
    return true;
  } else {
    std::string Message;
    Message += "Duplicate global symbol \"";
    Message += Name.str().str();
    Message += "\":\n\n";

    Message += "  " + It->second + "\n";
    Message += "  " + Path + "\n";
    return fail(Message);
  }
}

static bool verifyGlobalNamespace(VerifyHelper &VH,
                                  const model::Binary &Model) {

  // Namespacing rules:
  //
  // 1. each struct/union induces a namespace for its field names;
  // 2. each prototype induces a namespace for its arguments (and local
  //    variables, but those are not part of the model yet);
  // 3. the global namespace includes segment names, function names, dynamic
  //    function names, type names and entries of `enum`s;
  //
  // Verify needs to verify that each namespace has no internal clashes.
  // Also, the global namespace clashes with everything.
  for (const Function &F : Model.Functions()) {
    if (not VH.registerGlobalSymbol(F.CustomName(), Model.path(F)))
      return VH.fail("Duplicate name", F);
  }

  // Verify DynamicFunctions
  for (const DynamicFunction &DF : Model.ImportedDynamicFunctions()) {
    if (not VH.registerGlobalSymbol(DF.CustomName(), Model.path(DF)))
      return VH.fail();
  }

  // Verify types and enum entries
  for (const model::UpcastableTypeDefinition &Def : Model.TypeDefinitions()) {
    if (not VH.registerGlobalSymbol(Def->CustomName(), Model.path(*Def)))
      return VH.fail();

    if (auto *Enum = dyn_cast<model::EnumDefinition>(Def.get()))
      for (auto &Entry : Enum->Entries())
        if (not VH.registerGlobalSymbol(Entry.CustomName(),
                                        Model.path(*Enum, Entry)))
          return VH.fail();
  }

  // Verify Segments
  for (const Segment &S : Model.Segments()) {
    if (not VH.registerGlobalSymbol(S.CustomName(), Model.path(S)))
      return VH.fail();
  }

  return true;
}

bool Binary::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  // First of all, verify the global namespace: we need to fully populate it
  // before we can verify namespaces with smaller scopes
  if (not verifyGlobalNamespace(VH, *this))
    return VH.fail();

  // Verify individual functions
  for (const Function &F : Functions())
    if (not F.verify(VH))
      return VH.fail();

  // Verify DynamicFunctions
  for (const DynamicFunction &DF : ImportedDynamicFunctions())
    if (not DF.verify(VH))
      return VH.fail();

  // Verify Segments
  for (const Segment &S : Segments())
    if (not S.verify(VH))
      return VH.fail();

  // Make sure no segments overlap
  for (const auto &[LHS, RHS] : zip_pairs(Segments())) {
    revng_assert(LHS.StartAddress() <= RHS.StartAddress());
    if (LHS.endAddress() > RHS.StartAddress()) {
      std::string Error = "Overlapping segments:\n" + serializeToString(LHS)
                          + "and\n" + serializeToString(RHS);
      return VH.fail(Error);
    }
  }

  //
  // Verify the type system
  //
  return verifyTypeDefinitions(VH);
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

bool Relocation::verify() const {
  return verify(false);
}

bool Relocation::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Relocation::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Type() == model::RelocationType::Invalid)
    return VH.fail("Invalid relocation", *this);

  return true;
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

bool Segment::verify() const {
  return verify(false);
}

bool Segment::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Segment::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  using OverflowSafeInt = OverflowSafeInt<uint64_t>;

  if (FileSize() > VirtualSize())
    return VH.fail("FileSize cannot be larger than VirtualSize", *this);

  auto EndOffset = OverflowSafeInt(StartOffset()) + FileSize();
  if (not EndOffset)
    return VH.fail("Computing the segment end offset leads to overflow", *this);

  auto EndAddress = StartAddress() + VirtualSize();
  if (not EndAddress.isValid())
    return VH.fail("Computing the end address leads to overflow", *this);

  for (const model::Relocation &Relocation : Relocations()) {
    if (not Relocation.verify(VH))
      return VH.fail("Invalid relocation", Relocation);
  }

  if (not Type().empty()) {

    if (not Type().isValid())
      return VH.fail("Invalid segment type", *this);

    // The segment has a type

    auto *Struct = dyn_cast<model::StructDefinition>(Type().get());
    if (not Struct)
      return VH.fail("The segment type is not a StructDefinition", *this);

    if (VirtualSize() != Struct->Size()) {
      return VH.fail(Twine("The segment's size (VirtualSize) is not equal to "
                           "the size of the segment's type. VirtualSize: ")
                       + Twine(VirtualSize())
                       + Twine(" != Segment->Type()->Size(): ")
                       + Twine(Struct->Size()),
                     *this);
    }

    if (Struct->CanContainCode() != IsExecutable()) {
      if (IsExecutable()) {
        return VH.fail("The StructType representing the type of a executable "
                       "segment has CanContainedCode disabled",
                       *this);
      } else {
        return VH.fail("The StructType representing the type of a "
                       "non-executable segment has CanContainedCode enabled",
                       *this);
      }
    }

    if (not Type().get()->verify(VH))
      return VH.fail("Segment type does not verify", *this);
  }

  return true;
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

bool Function::verify() const {
  return verify(false);
}

bool Function::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Function::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (not Entry().isValid())
    return VH.fail("Invalid Entry", *this);

  if (not Prototype().empty()) {

    if (not Prototype().isValid())
      return VH.fail("Invalid prototype", *this);

    // The function has a prototype

    if (not model::QualifiedType::getFunctionType(Prototype()).has_value()) {
      return VH.fail("The prototype is neither a RawFunctionDefinition nor a "
                     "CABIFunctionDefinition",
                     *this);
    }

    if (not Prototype().get()->verify(VH))
      return VH.fail("Function prototype does not verify", *this);
  }

  if (not StackFrameType().empty()) {

    if (not StackFrameType().isValid())
      return VH.fail("Invalid stack frame type", *this);

    // The stack frame has a type

    if (not isa<model::StructDefinition>(StackFrameType().get()))
      return VH.fail("The stack frame type is not a StructDefinition", *this);

    if (not StackFrameType().get()->verify(VH))
      return VH.fail("Stack frame type does not verify", *this);
  }

  for (auto &CallSitePrototype : CallSitePrototypes())
    if (not CallSitePrototype.verify(VH))
      return VH.fail();

  return true;
}

bool DynamicFunction::verify() const {
  return verify(false);
}

bool DynamicFunction::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool DynamicFunction::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  // Ensure we have a name
  if (OriginalName().size() == 0)
    return VH.fail("Dynamic functions must have a OriginalName", *this);

  if (not Prototype().empty() and not Prototype().isValid())
    return VH.fail("Invalid prototype", *this);

  // Prototype is valid
  if (not Prototype().empty()) {
    if (not Prototype().get()->verify(VH))
      return VH.fail();

    if (not model::QualifiedType::getFunctionType(Prototype()).has_value()) {
      return VH.fail("The prototype is neither a RawFunctionDefinition nor a "
                     "CABIFunctionDefinition",
                     *this);
    }
  }

  for (auto &Attribute : Attributes()) {
    if (Attribute == model::FunctionAttribute::Inline) {
      return VH.fail("Dynamic function cannot have Inline attribute", *this);
    }
  }

  return true;
}

bool CallSitePrototype::verify() const {
  return verify(false);
}

bool CallSitePrototype::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool CallSitePrototype::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);

  if (Prototype().empty() or not Prototype().isValid())
    return VH.fail("Invalid prototype");

  // Prototype is valid
  if (not Prototype().get()->verify(VH))
    return VH.fail();

  if (not model::QualifiedType::getFunctionType(Prototype()).has_value()) {
    return VH.fail("The prototype is neither a RawFunctionDefinition nor a "
                   "CABIFunctionDefinition",
                   *this);
  }

  return true;
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
