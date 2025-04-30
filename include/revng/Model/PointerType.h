#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Architecture.h"
#include "revng/Model/Type.h"

#include "revng/Model/Generated/Early/PointerType.h"

class model::PointerType : public model::generated::PointerType {
public:
  static constexpr const auto AssociatedKind = TypeKind::PointerType;

public:
  using generated::PointerType::PointerType;

  static UpcastableType make(UpcastableType &&PointeeType,
                             uint64_t PointerSize) {
    return UpcastableType::make<PointerType>(false,
                                             PointerSize,
                                             std::move(PointeeType));
  }

  static UpcastableType makeConst(UpcastableType &&PointeeType,
                                  uint64_t PointerSize) {
    return UpcastableType::make<PointerType>(true,
                                             PointerSize,
                                             std::move(PointeeType));
  }

  static UpcastableType make(UpcastableType &&PointeeType,
                             model::Architecture::Values Architecture) {
    return make(std::move(PointeeType),
                model::Architecture::getPointerSize(Architecture));
  }

  static UpcastableType makeConst(UpcastableType &&PointeeType,
                                  model::Architecture::Values Architecture) {
    return makeConst(std::move(PointeeType),
                     model::Architecture::getPointerSize(Architecture));
  }
};

#include "revng/Model/Generated/Late/PointerType.h"
