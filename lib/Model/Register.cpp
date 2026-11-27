//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Model/Register.h"

namespace {

struct RegisterCSVComposition {
  model::Register::Values Register = model::Register::Invalid;
  uint64_t BaseOffset = 0;

  uint64_t CSVCount = 1;
  uint64_t CSVSize = 0;

  explicit RegisterCSVComposition(model::Register::Values Register,
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
static const std::array<RegisterCSVComposition, 41> SpecialRegisters = {
  // x86-64
  RegisterCSVComposition(model::Register::zmm0_x86_64,
                         /* BaseOffset = */ 0x2b10,
                         /* CSVCount = */ 8),
  RegisterCSVComposition(model::Register::zmm1_x86_64,
                         /* BaseOffset = */ 0x2b50,
                         /* CSVCount = */ 8),
  RegisterCSVComposition(model::Register::zmm2_x86_64,
                         /* BaseOffset = */ 0x2b90,
                         /* CSVCount = */ 8),
  RegisterCSVComposition(model::Register::zmm3_x86_64,
                         /* BaseOffset = */ 0x2bd0,
                         /* CSVCount = */ 8),
  RegisterCSVComposition(model::Register::zmm4_x86_64,
                         /* BaseOffset = */ 0x2c10,
                         /* CSVCount = */ 8),
  RegisterCSVComposition(model::Register::zmm5_x86_64,
                         /* BaseOffset = */ 0x2c50,
                         /* CSVCount = */ 8),
  RegisterCSVComposition(model::Register::zmm6_x86_64,
                         /* BaseOffset = */ 0x2c90,
                         /* CSVCount = */ 8),
  RegisterCSVComposition(model::Register::zmm7_x86_64,
                         /* BaseOffset = */ 0x2cd0,
                         /* CSVCount = */ 8),

  // x86
  RegisterCSVComposition(model::Register::st0_x86, /* BaseOffset = */ 0x2960),

  // AArch64
  //
  // TODO: consider generating this table instead of listing them all.
  RegisterCSVComposition(model::Register::v0_aarch64,
                         /* BaseOffset = */ 0x3480,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v1_aarch64,
                         /* BaseOffset = */ 0x3580,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v2_aarch64,
                         /* BaseOffset = */ 0x3680,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v3_aarch64,
                         /* BaseOffset = */ 0x3780,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v4_aarch64,
                         /* BaseOffset = */ 0x3880,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v5_aarch64,
                         /* BaseOffset = */ 0x3980,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v6_aarch64,
                         /* BaseOffset = */ 0x3a80,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v7_aarch64,
                         /* BaseOffset = */ 0x3b80,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v8_aarch64,
                         /* BaseOffset = */ 0x3c80,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v9_aarch64,
                         /* BaseOffset = */ 0x3d80,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v10_aarch64,
                         /* BaseOffset = */ 0x3e80,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v11_aarch64,
                         /* BaseOffset = */ 0x3f80,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v12_aarch64,
                         /* BaseOffset = */ 0x4080,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v13_aarch64,
                         /* BaseOffset = */ 0x4180,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v14_aarch64,
                         /* BaseOffset = */ 0x4280,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v15_aarch64,
                         /* BaseOffset = */ 0x4380,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v16_aarch64,
                         /* BaseOffset = */ 0x4480,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v17_aarch64,
                         /* BaseOffset = */ 0x4580,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v18_aarch64,
                         /* BaseOffset = */ 0x4680,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v19_aarch64,
                         /* BaseOffset = */ 0x4780,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v20_aarch64,
                         /* BaseOffset = */ 0x4880,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v21_aarch64,
                         /* BaseOffset = */ 0x4980,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v22_aarch64,
                         /* BaseOffset = */ 0x4a80,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v23_aarch64,
                         /* BaseOffset = */ 0x4b80,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v24_aarch64,
                         /* BaseOffset = */ 0x4c80,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v25_aarch64,
                         /* BaseOffset = */ 0x4d80,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v26_aarch64,
                         /* BaseOffset = */ 0x4e80,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v27_aarch64,
                         /* BaseOffset = */ 0x4f80,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v28_aarch64,
                         /* BaseOffset = */ 0x5080,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v29_aarch64,
                         /* BaseOffset = */ 0x5180,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v30_aarch64,
                         /* BaseOffset = */ 0x5280,
                         /* CSVCount = */ 2),
  RegisterCSVComposition(model::Register::v31_aarch64,
                         /* BaseOffset = */ 0x5380,
                         /* CSVCount = */ 2),
};

RegisterCSVComposition lookupRegister(model::Register::Values V) {
  revng_assert(V != model::Register::Invalid);

  for (const RegisterCSVComposition &Entry : SpecialRegisters)
    if (Entry.Register == V)
      return Entry;

  return RegisterCSVComposition(V);
}

std::optional<RegisterCSVComposition>
lookupRegister(model::Architecture::Values Arch, uint64_t Offset) {
  revng_assert(Offset != 0);

  for (const RegisterCSVComposition &Entry : SpecialRegisters) {
    if (Arch == model::Register::getReferenceArchitecture(Entry.Register)
        and Entry.BaseOffset <= Offset
        and Offset < Entry.BaseOffset + Entry.CSVCount * Entry.CSVSize) {
      return Entry;
    }
  }

  return std::nullopt;
}

static constexpr llvm::StringRef UnknownCSVPrefix = "state_0x";

std::string getCSVName(model::Register::Values V, uint64_t Offset) {
  RegisterCSVComposition Composition = lookupRegister(V);
  if (Composition.BaseOffset == 0) {
    revng_assert(Composition.CSVCount == 1,
                 "Registers without an offset must have a single CSV.");
    return "_" + model::Register::getRegisterName(V).str();
  }

  return "_" + UnknownCSVPrefix.str()
         + llvm::utohexstr(Composition.BaseOffset + Offset,
                           /* LowerCase = */ true);
}

std::pair<std::optional<RegisterCSVComposition>, uint64_t>
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

  std::optional<RegisterCSVComposition> Result = lookupRegister(Architecture,
                                                                FullOffset);
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

  return ::getCSVName(V, /* Offset = */ 0);
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

  RegisterCSVComposition Composition = lookupRegister(V);
  for (uint64_t Index = 0; Index < Composition.CSVCount; ++Index)
    co_yield CSV{ getCSVName(V, Index * Composition.CSVSize),
                  Index * Composition.CSVSize,
                  Composition.CSVSize };
}

model::Register::Portion::Portion(llvm::StringRef Name,
                                  model::Architecture::Values Architecture) {
  const auto &[Composition, Offset] = ::fromCSVName(Name, Architecture);
  if (not Composition.has_value())
    return;

  Register = Composition->Register;
  StartOffset = Offset;
  Size = Composition->CSVSize;

  revng_assert(Size > 0);
  revng_assert(StartOffset + Size <= model::Register::getSize(Register));
}
