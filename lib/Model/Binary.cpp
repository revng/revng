/// \file Binary.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/TypeSystemPrinter.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/OverflowSafeInt.h"
#include "revng/TupleTree/Tracking.h"

using namespace llvm;

namespace {

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

void onFieldAccess(StringRef FieldName, StringRef StructName) debug_function;

void onFieldAccess(StringRef FieldName, StringRef StructName) {
  FieldAccessedLogger << ((StringRef("Field ") + FieldName + " of struct "
                           + StructName + " accessed")
                            .str()
                            .c_str());
  FieldAccessedLogger.flush();
}
} // namespace

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

model::TypePath Binary::getPrimitiveType(PrimitiveTypeKind::Values V,
                                         uint8_t ByteSize) {
  PrimitiveType Temporary(V, ByteSize);
  Type::Key PrimitiveKey{ Temporary.ID(), TypeKind::PrimitiveType };
  auto It = Types().find(PrimitiveKey);

  // If we couldn't find it, create it
  if (It == Types().end()) {
    auto *NewPrimitiveType = new PrimitiveType(V, ByteSize);
    It = Types().insert(UpcastablePointer<model::Type>(NewPrimitiveType)).first;
  }

  return getTypePath(It->get());
}

model::TypePath Binary::getPrimitiveType(PrimitiveTypeKind::Values V,
                                         uint8_t ByteSize) const {
  PrimitiveType Temporary(V, ByteSize);
  Type::Key PrimitiveKey{ Temporary.ID(), TypeKind::PrimitiveType };
  return getTypePath(Types().at(PrimitiveKey).get());
}

uint64_t Binary::getAvailableTypeID() const {
  uint64_t Result = 0;

  if (not Types().empty())
    Result = Types().rbegin()->get()->ID() + 1;

  Result = std::max(model::PrimitiveType::FirstNonPrimitiveID, Result);
  return Result;
}

TypePath Binary::recordNewType(UpcastablePointer<Type> &&T) {
  if (not isa<PrimitiveType>(T.get())) {
    // Assign progressive ID
    revng_assert(T->ID() == 0);
    T->ID() = getAvailableTypeID();
  }

  auto [It, Success] = Types().insert(T);
  revng_assert(Success);
  return getTypePath(It->get());
}

bool Binary::verifyTypes() const {
  return verifyTypes(false);
}

bool Binary::verifyTypes(bool Assert) const {
  VerifyHelper VH(Assert);
  return verifyTypes(VH);
}

bool Binary::verifyTypes(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);
  // All types on their own should verify
  std::set<Identifier> Names;
  for (auto &Type : Types()) {
    // Verify the type
    if (not Type.get()->verify(VH))
      return VH.fail();

    // Ensure the names are unique
    auto Name = Type->name();
    if (not Names.insert(Name).second)
      return VH.fail(Twine("Multiple types with the following name: ") + Name);
  }

  return true;
}

void Binary::dump() const {
  TrackGuard Guard(*this);
  serialize(dbg, *this);
}

void Binary::dumpTypeGraph(const char *Path) const {
  TrackGuard Guard(*this);

  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC);
  if (EC)
    revng_abort(EC.message().c_str());

  TypeSystemPrinter TSPrinter(Out);
  TSPrinter.print(*this);
}

std::string Binary::toString() const {
  TrackGuard Guard(*this);
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

  auto Guard = VH.suspendTracking(Model);
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
  for (auto &Type : Model.Types()) {
    if (not VH.registerGlobalSymbol(Type->CustomName(), Model.path(*Type)))
      return VH.fail();

    if (auto *Enum = dyn_cast<EnumType>(Type.get()))
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
  // First of all, verify the global namespace: we need to fully populate it
  // before we can verify namespaces with smaller scopes

  auto Guard = VH.suspendTracking(*this);
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
  return verifyTypes(VH);
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

static const model::TypePath &prototypeOr(const model::TypePath &Prototype,
                                          const model::TypePath &Default) {
  if (not Prototype.empty()) {
    revng_assert(Prototype.isValid());
    return Prototype;
  }

  return Default;
}

model::TypePath Function::prototype(const model::Binary &Root) const {
  model::TypePath Result;
  auto ThePrototype = prototypeOr(Prototype(), Root.DefaultPrototype());
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

model::TypePath DynamicFunction::prototype(const model::Binary &Root) const {
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

bool Section::verify() const {
  return verify(false);
}

bool Section::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Section::verify(VerifyHelper &VH) const {
  auto Guard = VH.suspendTracking(*this);
  auto EndAddress = StartAddress() + Size();
  if (not EndAddress.isValid())
    return VH.fail("Computing the end address leads to overflow");

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

void Segment::dump() const {
  TrackGuard Guard(*this);
  serialize(dbg, *this);
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

  for (const model::Section &Section : Sections()) {
    if (not Section.verify(VH))
      return VH.fail("Invalid section", Section);

    if (not contains(Section.StartAddress())
        or (VirtualSize() > 0 and not contains(Section.endAddress() - 1))) {
      return VH.fail("The segment contains a section out of its boundaries",
                     Section);
    }

    if (Section.ContainsCode() and not IsExecutable()) {
      return VH.fail("A Section is marked as containing code but the "
                     "containing segment is not executable",
                     *this);
    }
  }

  for (const model::Relocation &Relocation : Relocations()) {
    if (not Relocation.verify(VH))
      return VH.fail("Invalid relocation", Relocation);
  }

  if (not Type().empty()) {

    if (not Type().isValid())
      return VH.fail("Invalid segment type", *this);

    // The segment has a type

    auto *Struct = dyn_cast<model::StructType>(Type().get());
    if (not Struct)
      return VH.fail("The segment type is not a StructType", *this);

    if (VirtualSize() != Struct->Size()) {
      return VH.fail(Twine("The segment's size (VirtualSize) is not equal to "
                           "the size of the segment's type. VirtualSize: ")
                       + Twine(VirtualSize())
                       + Twine(" != Segment->Type()->Size(): ")
                       + Twine(Struct->Size()),
                     *this);
    }

    if (not Type().get()->verify(VH))
      return VH.fail("Segment type does not verify", *this);
  }

  return true;
}

void Function::dump() const {
  TrackGuard Guard(*this);
  serialize(dbg, *this);
}

void Function::dumpTypeGraph(const char *Path) const {
  TrackGuard Guard(*this);
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
      return VH.fail("The prototype is neither a RawFunctionType nor a "
                     "CABIFunctionType",
                     *this);
    }

    if (not Prototype().get()->verify(VH))
      return VH.fail("Function prototype does not verify", *this);
  }

  if (not StackFrameType().empty()) {

    if (not StackFrameType().isValid())
      return VH.fail("Invalid stack frame type", *this);

    // The stack frame has a type

    if (not isa<model::StructType>(StackFrameType().get()))
      return VH.fail("The stack frame type is not a StructType", *this);

    if (not StackFrameType().get()->verify(VH))
      return VH.fail("Stack frame type does not verify", *this);
  }

  for (auto &CallSitePrototype : CallSitePrototypes())
    if (not CallSitePrototype.verify(VH))
      return VH.fail();

  return true;
}

void DynamicFunction::dump() const {
  TrackGuard Guard(*this);
  serialize(dbg, *this);
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
      return VH.fail("The prototype is neither a RawFunctionType nor a "
                     "CABIFunctionType",
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

void CallSitePrototype::dump() const {
  TrackGuard Guard(*this);
  serialize(dbg, *this);
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
    return VH.fail("The prototype is neither a RawFunctionType nor a "
                   "CABIFunctionType",
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
