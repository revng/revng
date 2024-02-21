#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Register.h"
#include "revng/Model/Type.h"

/* TUPLE-TREE-YAML
name: PrimitiveType
type: struct
inherits: Type
fields:
  - name: PrimitiveKind
    type: PrimitiveKind
  - name: Size
    type: uint64_t
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/PrimitiveType.h"

class model::PrimitiveType : public model::generated::PrimitiveType {
public:
  static constexpr const auto AssociatedKind = TypeKind::PrimitiveType;

public:
  using generated::PrimitiveType::PrimitiveType;

  //
  // General construction
  //

  static UpcastableType make(PrimitiveKind::Values Kind, uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(false, Kind, Size);
  }

  static UpcastableType makeConst(PrimitiveKind::Values Kind, uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(true, Kind, Size);
  }

private:
  using PK = PrimitiveKind::Values;

public:
  //
  // Kind specific construction
  //

  static UpcastableType makeVoid() {
    return UpcastableType::make<PrimitiveType>(false, PK::Void, 0);
  }
  static UpcastableType makeConstVoid() {
    return UpcastableType::make<PrimitiveType>(true, PK::Void, 0);
  }
  static UpcastableType makeGeneric(uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(false, PK::Generic, Size);
  }
  static UpcastableType makeConstGeneric(uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(true, PK::Generic, Size);
  }
  static UpcastableType makePointerOrNumber(uint64_t S) {
    return UpcastableType::make<PrimitiveType>(false, PK::PointerOrNumber, S);
  }
  static UpcastableType makeConstPointerOrNumber(uint64_t S) {
    return UpcastableType::make<PrimitiveType>(true, PK::PointerOrNumber, S);
  }
  static UpcastableType makeNumber(uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(false, PK::Number, Size);
  }
  static UpcastableType makeConstNumber(uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(true, PK::Number, Size);
  }
  static UpcastableType makeUnsigned(uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(false, PK::Unsigned, Size);
  }
  static UpcastableType makeConstUnsigned(uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(true, PK::Unsigned, Size);
  }
  static UpcastableType makeSigned(uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(false, PK::Signed, Size);
  }
  static UpcastableType makeConstSigned(uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(true, PK::Signed, Size);
  }
  static UpcastableType makeFloat(uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(false, PK::Float, Size);
  }
  static UpcastableType makeConstFloat(uint64_t Size) {
    return UpcastableType::make<PrimitiveType>(true, PK::Float, Size);
  }

public:
  //
  // Register-based construction
  //

  static UpcastableType make(Register::Values Register) {
    return make(model::Register::primitiveKind(Register),
                model::Register::getSize(Register));
  }

  static UpcastableType makeGeneric(Register::Values Register) {
    return makeGeneric(model::Register::getSize(Register));
  }
};

#include "revng/Model/Generated/Late/PrimitiveType.h"
