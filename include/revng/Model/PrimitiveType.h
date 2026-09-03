#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/PrimitiveKind.h"
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
  /// \name Verification
  /// @{

  static constexpr std::array ValidNumericSizes{ 1, 2, 4, 8, 16, 32, 64 };
  static constexpr std::array ValidFloatSizes{ 2, 4, 8, 10, 12, 16 };
  // NOTE: We are supporting floats that are 10 bytes long, since we found
  //       such cases in some PDB files by using VS on Windows platforms.
  //       The source code of those cases could be written in some language
  //       other than C/C++ (probably Swift). We faced some struct fields by
  //       using this (10b long float) type, so by ignoring it we would not
  //       have accurate layout for the structs.

  static constexpr bool isSizeValid(model::PrimitiveKind::Values Kind,
                                    uint8_t Size) {
    switch (Kind) {
    case PrimitiveKind::Invalid:
      return false;

    case PrimitiveKind::Void:
      return Size == 0;

    case PrimitiveKind::PointerOrNumber:
    case PrimitiveKind::Number:
    case PrimitiveKind::Unsigned:
    case PrimitiveKind::Signed:
      return std::ranges::binary_search(ValidNumericSizes, Size);

    case PrimitiveKind::Float:
      return std::ranges::binary_search(ValidFloatSizes, Size);

    case PrimitiveKind::Generic:
      return std::ranges::binary_search(ValidNumericSizes, Size)
             || std::ranges::binary_search(ValidFloatSizes, Size);

    default:
      revng_abort("Unsupported primitive kind");
    }
  }

  bool isValid() const { return isSizeValid(PrimitiveKind(), Size()); }

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

  ///
  /// \name Dynamic size construction
  /// @{

private:
  static std::optional<size_t> closestNumericSize(uint64_t Size) {
    auto Iterator = std::ranges::lower_bound(ValidNumericSizes, Size);
    if (Iterator == ValidNumericSizes.end())
      return std::nullopt;
    return *Iterator;
  }
  static std::optional<size_t> closestFloatSize(uint64_t Size) {
    auto Iterator = std::ranges::lower_bound(ValidFloatSizes, Size);
    if (Iterator == ValidFloatSizes.end())
      return std::nullopt;
    return *Iterator;
  }

public:
  static UpcastableType makeBigEnoughFor(PrimitiveKind::Values Kind,
                                         uint64_t Size) {
    uint64_t SelectedSize = -1;
    switch (Kind) {
    case PrimitiveKind::PointerOrNumber:
    case PrimitiveKind::Number:
    case PrimitiveKind::Unsigned:
    case PrimitiveKind::Signed:
      SelectedSize = closestNumericSize(Size).value_or(-1u);
      break;

    case PrimitiveKind::Float:
      SelectedSize = closestFloatSize(Size).value_or(-1u);
      break;

    case PrimitiveKind::Generic:
      SelectedSize = std::min(closestNumericSize(Size).value_or(-1u),
                              closestFloatSize(Size).value_or(-1u));
      break;

    default:
      revng_abort("Unsupported primitive kind");
      break;
    };

    if (SelectedSize == -1u) {
      std::string Error = "Requested size (" + std::to_string(Size)
                          + ") is bigger than the largest allowed `"
                          + ::toString(Kind) + "` primitive.\n";
      revng_abort(Error.c_str());
    }

    return make(Kind, SelectedSize);
  }

  static UpcastableType makeGenericBigEnoughFor(uint64_t Size) {
    return makeBigEnoughFor(model::PrimitiveKind::Generic, Size);
  }

  /// @}

public:
  ///
  /// \name Naming helpers
  /// @{

  std::string getCName() const {
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
};

#include "revng/Model/Generated/Late/PrimitiveType.h"
