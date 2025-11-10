#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Register.h"
#include "revng/Model/Type.h"

#include "revng/Model/Generated/Early/PrimitiveType.h"

class model::PrimitiveType : public model::generated::PrimitiveType {
public:
  static constexpr const auto AssociatedKind = TypeKind::PrimitiveType;

public:
  using generated::PrimitiveType::PrimitiveType;

  ///
  /// \name General construction
  /// @{

  static UpcastableType make(PrimitiveKind::Values Kind, uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(false, Kind, Size);
  }

  static UpcastableType makeConst(PrimitiveKind::Values Kind, uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(true, Kind, Size);
  }

  /// @}

private:
  using PK = PrimitiveKind::Values;

public:
  ///
  /// \name Kind specific construction
  /// @{

  static UpcastableType makeVoid() { return make(PK::Void, 0); }
  static UpcastableType makeConstVoid() { return makeConst(PK::Void, 0); }
  static UpcastableType makeGeneric(uint64_t Size) {
    return make(PK::Generic, Size);
  }
  static UpcastableType makeConstGeneric(uint64_t Size) {
    return makeConst(PK::Generic, Size);
  }
  static UpcastableType makePointerOrNumber(uint64_t Size) {
    return make(PK::PointerOrNumber, Size);
  }
  static UpcastableType makeConstPointerOrNumber(uint64_t Size) {
    return makeConst(PK::PointerOrNumber, Size);
  }
  static UpcastableType makeNumber(uint64_t Size) {
    return make(PK::Number, Size);
  }
  static UpcastableType makeConstNumber(uint64_t Size) {
    return makeConst(PK::Number, Size);
  }
  static UpcastableType makeUnsigned(uint64_t Size) {
    return make(PK::Unsigned, Size);
  }
  static UpcastableType makeConstUnsigned(uint64_t Size) {
    return makeConst(PK::Unsigned, Size);
  }
  static UpcastableType makeSigned(uint64_t Size) {
    return make(PK::Signed, Size);
  }
  static UpcastableType makeConstSigned(uint64_t Size) {
    return makeConst(PK::Signed, Size);
  }
  static UpcastableType makeFloat(uint64_t Size) {
    return make(PK::Float, Size);
  }
  static UpcastableType makeConstFloat(uint64_t Size) {
    return makeConst(PK::Float, Size);
  }

  /// @}

public:
  ///
  /// \name Register-based construction
  /// @{

  static UpcastableType make(Register::Values Register) {
    return make(model::Register::primitiveKind(Register),
                model::Register::getSize(Register));
  }

  static UpcastableType makeGeneric(Register::Values Register) {
    return makeGeneric(model::Register::getSize(Register));
  }

  /// @}

public:
  ///
  /// \name Naming helpers
  /// @{

  std::string getCName() const {
    revng_assert(isSizeValid());
    if (PrimitiveKind() == model::PrimitiveKind::Void) {
      revng_assert(Size() == 0);
      return "void";
    } else {
      revng_assert(Size() != 0);
      return model::PrimitiveKind::getCPrefix(PrimitiveKind()).str()
             + std::to_string(Size() * 8) + "_t";
    }
  }

  static std::string getCName(PrimitiveKind::Values Kind, uint64_t Size) {
    return PrimitiveType(false, Kind, Size).getCName();
  }

  static UpcastableType fromCName(llvm::StringRef Name) {
    // Figure the primitive kind out.
    auto &&[Kind, RemainingName] = PrimitiveKind::tryConsumeCPrefix(Name);
    if (Kind == PrimitiveKind::Invalid || RemainingName == Name)
      return UpcastableType::empty();

    if (Kind == PrimitiveKind::Void)
      return RemainingName.empty() ? makeVoid() : UpcastableType::empty();

    // Ensure the name ends with _t
    if (not RemainingName.consume_back("_t"))
      return UpcastableType::empty();

    // Consume bit size
    unsigned Bits = 0;
    if (RemainingName.consumeInteger(10, Bits))
      return UpcastableType::empty();

    // Ensure we consumed everything
    if (RemainingName.size() != 0)
      return UpcastableType::empty();

    // Ensure the bit size is a multiple of 8
    if (Bits % 8 != 0)
      return UpcastableType::empty();

    model::UpcastableType Result = make(Kind, Bits / 8);
    if (not Result->verify())
      return UpcastableType::empty();

    return Result;
  }

  static bool isCName(llvm::StringRef Name) {
    return not fromCName(Name).isEmpty();
  }

  /// @}

public:
  ///
  /// \name size-related helpers (including verification)
  /// @{

  static constexpr uint64_t ValidVoidSize = 0;
  static constexpr std::array<uint64_t, 5> ValidIntegerSizes{ 1, 2, 4, 8, 16 };
  static constexpr std::array<uint64_t, 6> ValidFloatSizes{
    2, 4, 8, 10, 12, 16
  };

  static constexpr bool isValidPrimitiveSize(PrimitiveKind::Values Kind,
                                             uint8_t Size) {
    // NOTE: We are supporting floats that are 10 bytes long, since we found
    // such
    //       cases in some PDB files by using VS on Windows platforms. The
    //       source code of those cases could be written in some language other
    //       than C/C++ (probably Swift). We faced some struct fields by using
    //       this (10b long float) type, so by ignoring it we would not have
    //       accurate layout for the structs.

    switch (Kind) {
    case PrimitiveKind::Invalid:
      return false;

    case PrimitiveKind::Void:
      return Size == ValidVoidSize;

    case PrimitiveKind::PointerOrNumber:
    case PrimitiveKind::Number:
    case PrimitiveKind::Unsigned:
    case PrimitiveKind::Signed:
      return std::ranges::binary_search(ValidIntegerSizes, Size);

    case PrimitiveKind::Float:
      return std::ranges::binary_search(ValidFloatSizes, Size);

    case PrimitiveKind::Generic:
      return std::ranges::binary_search(ValidIntegerSizes, Size)
             || std::ranges::binary_search(ValidFloatSizes, Size);

    default:
      revng_abort("Unsupported primitive kind");
    }
  }

  bool isSizeValid() const {
    return isValidPrimitiveSize(PrimitiveKind(), Size());
  }

  static constexpr std::span<const uint64_t>
  staticValidSizes(PrimitiveKind::Values Kind) {
    revng_assert(Kind != PrimitiveKind::Generic,
                 "Unable to give static size list for generics, please use the "
                 "dynamic version");
    if (Kind == PrimitiveKind::Values::Void)
      return std::span<const uint64_t>{ &ValidVoidSize, 1 };
    if (Kind == PrimitiveKind::Values::Float)
      return ValidFloatSizes;
    else
      return ValidIntegerSizes;
  }

  static std::vector<uint64_t> validSizes(PrimitiveKind::Values Kind) {
    if (Kind == PrimitiveKind::Values::Generic) {
      std::vector<uint64_t> Result;

      std::ranges::set_union(ValidIntegerSizes,
                             ValidFloatSizes,
                             std::back_inserter(Result));
      return Result;
    }

    std::span Result = staticValidSizes(Kind);
    return std::vector(Result.begin(), Result.end());
  }

  /// @}
};

#include "revng/Model/Generated/Late/PrimitiveType.h"
