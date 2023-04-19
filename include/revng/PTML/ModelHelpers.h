#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/DynamicFunction.h"
#include "revng/Model/Function.h"
#include "revng/Model/Segment.h"
#include "revng/PTML/Tag.h"

namespace modelEditPath {
using ptml::str;

inline std::string getCustomNamePath(const model::Segment &Segment) {
  return "/Segments/" + str(Segment.key()) + "/CustomName";
}

inline std::string getCustomNamePath(const model::DynamicFunction &DF) {
  return "/ImportedDynamicFunctions/" + str(DF.key()) + "/CustomName";
}

inline std::string getCustomNamePath(const model::Function &Function) {
  return "/Functions/" + str(Function.key()) + "/CustomName";
}

inline std::string getCustomNamePath(const model::Type &Type) {
  return "/Types/" + str(Type.key()) + "/CustomName";
}

template<typename T>
  requires std::same_as<model::UnionType, T>
           or std::same_as<model::StructType, T>
inline std::string getCustomNamePath(const T &Obj, uint64_t FieldIdx) {
  return "/Types/" + str(Obj.key()) + "/Fields/" + std::to_string(FieldIdx)
         + "/CustomName";
}

inline std::string getCustomNamePath(const model::StructType &Struct,
                                     const model::StructField Field) {
  return "/Types/" + str(Struct.key()) + "/Fields/" + str(Field.key())
         + "/CustomName";
}

inline std::string getCustomNamePath(const model::UnionType &Union,
                                     const model::UnionField Field) {
  return "/Types/" + str(Union.key()) + "/Fields/" + str(Field.key())
         + "/CustomName";
}

inline std::string
getCustomNamePath(const model::EnumType &Type, const model::EnumEntry &Entry) {
  return "/Types/" + str(Type.key()) + "/" + str(Entry.key()) + "/CustomName";
}

} // namespace modelEditPath
