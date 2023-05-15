/// \file Binary.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/TypeSystemPrinter.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/OverflowSafeInt.h"

using namespace llvm;

namespace model {

model::TypePath Binary::getPrimitiveType(PrimitiveTypeKind::Values V,
                                         uint8_t ByteSize) {
  PrimitiveType Temporary(V, ByteSize);
  Type::Key PrimitiveKey{ TypeKind::PrimitiveType, Temporary.ID() };
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
  Type::Key PrimitiveKey{ TypeKind::PrimitiveType, Temporary.ID() };
  return getTypePath(Types().at(PrimitiveKey).get());
}

TypePath Binary::recordNewType(UpcastablePointer<Type> &&T) {
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
  serialize(dbg, *this);
}

void Binary::dumpTypeGraph(const char *Path) const {
  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC);
  if (EC)
    revng_abort(EC.message().c_str());

  TypeSystemPrinter TSPrinter(Out);
  TSPrinter.print(*this);
}

std::string Binary::toString() const {
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

bool Binary::verify(VerifyHelper &VH) const {
  // Prepare for checking symbol names. We will populate and check this against
  // functions, dynamic functions, segments, types and enum entries
  std::set<Identifier> Symbols;
  auto CheckCustomName = [&VH, &Symbols, this](const Identifier &CustomName) {
    if (CustomName.empty())
      return true;

    return VH.maybeFail(Symbols.insert(CustomName).second,
                        "Duplicate name: " + CustomName.str().str(),
                        *this);
  };

  for (const Function &F : Functions()) {
    // Verify individual functions
    if (not F.verify(VH))
      return VH.fail();

    if (not CheckCustomName(F.CustomName()))
      return VH.fail("Duplicate name", F);
  }

  // Verify DynamicFunctions
  for (const DynamicFunction &DF : ImportedDynamicFunctions()) {
    if (not DF.verify(VH))
      return VH.fail();

    if (not CheckCustomName(DF.CustomName()))
      return VH.fail();
  }

  for (auto &Type : Types()) {
    if (not CheckCustomName(Type->CustomName()))
      return VH.fail();

    if (auto *Enum = dyn_cast<EnumType>(Type.get()))
      for (auto &Entry : Enum->Entries())
        if (not CheckCustomName(Entry.CustomName()))
          return VH.fail();
  }

  // Verify Segments
  for (const Segment &S : Segments()) {
    if (not S.verify(VH))
      return VH.fail();

    if (not CheckCustomName(S.CustomName()))
      return VH.fail();
  }

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
    auto AutomaticName = (Twine("function_") + Entry().toString()).str();
    return Identifier::fromString(AutomaticName);
  }
}

static const model::TypePath &prototypeOr(const model::TypePath &Prototype,
                                          const model::TypePath &Default) {
  if (Prototype.isValid())
    return Prototype;

  revng_assert(Default.isValid());
  return Default;
}

const model::TypePath &Function::prototype(const model::Binary &Root) const {
  return prototypeOr(Prototype(), Root.DefaultPrototype());
}

Identifier DynamicFunction::name() const {
  using llvm::Twine;
  if (not CustomName().empty()) {
    return CustomName();
  } else {
    auto AutomaticName = (Twine("dynamic_function_") + OriginalName()).str();
    return Identifier::fromString(AutomaticName);
  }
}

const model::TypePath &
DynamicFunction::prototype(const model::Binary &Root) const {
  return prototypeOr(Prototype(), Root.DefaultPrototype());
}

bool Relocation::verify() const {
  return verify(false);
}

bool Relocation::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Relocation::verify(VerifyHelper &VH) const {
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
    auto AutomaticName = (Twine("segment_") + StartAddress().toString() + "_"
                          + Twine(VirtualSize()))
                           .str();
    return Identifier::fromString(AutomaticName);
  }
}

void Segment::dump() const {
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

  return true;
}

void Function::dump() const {
  serialize(dbg, *this);
}

void Function::dumpTypeGraph(const char *Path) const {
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
  if (Prototype().isValid()) {
    // The function has a prototype
    if (not Prototype().get()->verify(VH))
      return VH.fail("Function prototype does not verify", *this);

    const model::Type *FunctionType = Prototype().get();
    if (not(isa<RawFunctionType>(FunctionType)
            or isa<CABIFunctionType>(FunctionType))) {
      return VH.fail("Function prototype is not a RawFunctionType or "
                     "CABIFunctionType",
                     *this);
    }
  }

  for (auto &CallSitePrototype : CallSitePrototypes())
    if (not CallSitePrototype.verify(VH))
      return VH.fail();

  return true;
}

void DynamicFunction::dump() const {
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
  // Ensure we have a name
  if (OriginalName().size() == 0)
    return VH.fail("Dynamic functions must have a OriginalName", *this);

  // Prototype is valid
  if (Prototype().isValid()) {
    if (not Prototype().get()->verify(VH))
      return VH.fail();

    const model::Type *FunctionType = Prototype().get();
    if (not(isa<RawFunctionType>(FunctionType)
            or isa<CABIFunctionType>(FunctionType))) {
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
  // Prototype is present
  if (not Prototype().isValid())
    return VH.fail("Invalid prototype", *this);

  // Prototype is valid
  if (not Prototype().get()->verify(VH))
    return VH.fail();

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
