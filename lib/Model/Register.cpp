//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Model/Register.h"

namespace {

struct InternalRegister {
  model::Register::Values Register = model::Register::Invalid;
  uint64_t BaseOffset = 0;

  uint64_t CSVCount = 1;
  uint64_t CSVSize = 0;

  explicit InternalRegister(model::Register::Values Register,
                            uint64_t BaseOffset = 0,
                            uint64_t CSVCount = 1) :
    Register(Register),
    BaseOffset(BaseOffset),
    CSVCount(CSVCount),
    CSVSize(model::Register::getSize(Register) / CSVCount) {

    revng_assert(CSVCount != 0);
    revng_assert(model::Register::getSize(Register) % CSVCount == 0);

    if (CSVCount > 1)
      revng_assert(BaseOffset != 0);
  }
};

// TODO: the only reason this map is not constexpr is because of the asserts
//       in the constructor, but IMO, those are worth it.
static std::array<InternalRegister, 9> SpecialRegisters = {
  // x86-64
  InternalRegister(model::Register::zmm0_x86_64,
                   /* BaseOffset = */ 0x2b10,
                   /* CSVCount = */ 8),
  InternalRegister(model::Register::zmm1_x86_64,
                   /* BaseOffset = */ 0x2b50,
                   /* CSVCount = */ 8),
  InternalRegister(model::Register::zmm2_x86_64,
                   /* BaseOffset = */ 0x2b90,
                   /* CSVCount = */ 8),
  InternalRegister(model::Register::zmm3_x86_64,
                   /* BaseOffset = */ 0x2bd0,
                   /* CSVCount = */ 8),
  InternalRegister(model::Register::zmm4_x86_64,
                   /* BaseOffset = */ 0x2c10,
                   /* CSVCount = */ 8),
  InternalRegister(model::Register::zmm5_x86_64,
                   /* BaseOffset = */ 0x2c50,
                   /* CSVCount = */ 8),
  InternalRegister(model::Register::zmm6_x86_64,
                   /* BaseOffset = */ 0x2c90,
                   /* CSVCount = */ 8),
  InternalRegister(model::Register::zmm7_x86_64,
                   /* BaseOffset = */ 0x2cd0,
                   /* CSVCount = */ 8),

  // x86
  InternalRegister(model::Register::st0_x86, /* BaseOffset = */ 0x2960),

  // TODO: extend!
};
InternalRegister lookupRegister(model::Register::Values V) {
  revng_assert(V != model::Register::Invalid);

  for (const InternalRegister &Entry : SpecialRegisters)
    if (Entry.Register == V)
      return Entry;

  return InternalRegister(V);
}
std::optional<InternalRegister>
lookupRegister(uint64_t Offset, model::Architecture::Values Arch) {
  revng_assert(Offset != 0);

  for (InternalRegister &Entry : SpecialRegisters) {
    if (Arch == model::Register::getReferenceArchitecture(Entry.Register)
        and Entry.BaseOffset <= Offset
        and Offset < Entry.BaseOffset + Entry.CSVCount * Entry.CSVSize) {
      return Entry;
    }
  }

  return std::nullopt;
}

static constexpr llvm::StringRef UnknownCSVPrefix = "state_0x";

std::string getCSVName(model::Register::Values V, uint64_t Offset = 0) {
  InternalRegister Internal = lookupRegister(V);
  if (Internal.BaseOffset == 0) {
    revng_assert(Internal.CSVCount == 1,
                 "Registers without an offset must have a single CSV.");
    return "_" + model::Register::getRegisterName(V).str();
  }

  return "_" + UnknownCSVPrefix.str()
         + llvm::utohexstr(Internal.BaseOffset + Offset,
                           /* LowerCase = */ true);
}

std::pair<std::optional<InternalRegister>, uint64_t>
fromCSVName(llvm::StringRef Name, model::Architecture::Values Architecture) {
  if (not Name.consume_front("_"))
    return { std::nullopt, 0 };

  if (not Name.consume_front(UnknownCSVPrefix)) {
    auto Deserialized = model::Register::fromRegisterName(Name, Architecture);
    if (Deserialized != model::Register::Invalid)
      return { lookupRegister(Deserialized), 0 };
    else
      return { std::nullopt, 0 };
  }

  uint64_t FullOffset = 0;
  if (Name.getAsInteger(/* Radix = */ 16, FullOffset))
    return { std::nullopt, 0 };

  std::optional<InternalRegister> Result = lookupRegister(FullOffset,
                                                          Architecture);
  revng_assert(not Result.has_value() or FullOffset >= Result->BaseOffset);
  uint64_t Offset = Result.has_value() ? FullOffset - Result->BaseOffset : 0;
  return { std::move(Result), Offset };
}

} // namespace

uint64_t model::Register::getCSVCount(Values V) {
  return lookupRegister(V).CSVCount;
}

std::string model::Register::singleCSVName(Values V) {
  revng_assert(getCSVCount(V) == 1);

  return ::getCSVName(V);
}

model::Register::Values
model::Register::fromCSVName(llvm::StringRef Name,
                             model::Architecture::Values Architecture) {
  const auto &[Result, _] = ::fromCSVName(Name, Architecture);
  if (Result)
    return Result->Register;
  else
    return model::Register::Invalid;
}

cppcoro::generator<model::Register::CSV> model::Register::getCSVs(Values V) {
  revng_assert(V != model::Register::Invalid);

  InternalRegister Internal = lookupRegister(V);
  for (uint64_t Index = 0; Index < Internal.CSVCount; ++Index)
    co_yield CSV{ getCSVName(V, Index * Internal.CSVSize),
                  Index * Internal.CSVSize,
                  Internal.CSVSize };
}

model::Register::Portion::Portion(llvm::StringRef Name,
                                  model::Architecture::Values Architecture) {
  const auto &[Internal, Offset] = ::fromCSVName(Name, Architecture);
  if (not Internal.has_value())
    return;

  this->Register = Internal->Register;
  this->StartOffset = Offset;
  this->Size = Internal->CSVSize;

  revng_assert(this->Size > 0);
}
