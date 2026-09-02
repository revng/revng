//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Register.h"

#define UnknownCSVPrefix "state_"

std::string model::Register::singleCSVName(Values V) {
  revng_assert(getCSVCount(V) == 1);

  // TODO: handle xmm0_x86

  switch (V) {
  case st0_x86:
    return "_" UnknownCSVPrefix "0x2960";
  case xmm0_x86_64:
    return "_" UnknownCSVPrefix "0x2b10";
  case xmm1_x86_64:
    return "_" UnknownCSVPrefix "0x2b50";
  case xmm2_x86_64:
    return "_" UnknownCSVPrefix "0x2b90";
  case xmm3_x86_64:
    return "_" UnknownCSVPrefix "0x2bd0";
  case xmm4_x86_64:
    return "_" UnknownCSVPrefix "0x2c10";
  case xmm5_x86_64:
    return "_" UnknownCSVPrefix "0x2c50";
  case xmm6_x86_64:
    return "_" UnknownCSVPrefix "0x2c90";
  case xmm7_x86_64:
    return "_" UnknownCSVPrefix "0x2cd0";
  default:
    return "_" + model::Register::getRegisterName(V).str();
  }
}

model::Register::Values
model::Register::fromCSVName(llvm::StringRef Name,
                             model::Architecture::Values Architecture) {
  if (not Name.starts_with("_"))
    return model::Register::Invalid;

  Name = Name.substr(1);

  if (Architecture == model::Architecture::x86) {
    if (Name == UnknownCSVPrefix "0x2960") {
      return st0_x86;
    }
  } else if (Architecture == model::Architecture::x86_64) {
    // TODO: handle xmm0_x86
    if (Name == UnknownCSVPrefix "0x2b10") {
      return xmm0_x86_64;
    } else if (Name == UnknownCSVPrefix "0x2b50") {
      return xmm1_x86_64;
    } else if (Name == UnknownCSVPrefix "0x2b90") {
      return xmm2_x86_64;
    } else if (Name == UnknownCSVPrefix "0x2bd0") {
      return xmm3_x86_64;
    } else if (Name == UnknownCSVPrefix "0x2c10") {
      return xmm4_x86_64;
    } else if (Name == UnknownCSVPrefix "0x2c50") {
      return xmm5_x86_64;
    } else if (Name == UnknownCSVPrefix "0x2c90") {
      return xmm6_x86_64;
    } else if (Name == UnknownCSVPrefix "0x2cd0") {
      return xmm7_x86_64;
    }
  }

  return model::Register::fromRegisterName(Name, Architecture);
}

#undef UnknownCSVPrefix

uint64_t model::Register::getCSVCount(Values V) {
  return 1;
}

cppcoro::generator<model::Register::CSV> model::Register::getCSVs(Values V) {
  // Every register currently maps to exactly one CSV.
  co_yield CSV{ singleCSVName(V), 0, model::Register::getSize(V) };
}

// Each CSV currently covers a whole register, so the portion always starts
// at offset 0 and spans the full register.
model::Register::Portion::Portion(llvm::StringRef Name,
                                  model::Architecture::Values A) :
  Register(fromCSVName(Name, A)),
  StartOffset(0),
  Size(Register != model::Register::Invalid ?
         model::Register::getSize(Register) :
         0) {
}
