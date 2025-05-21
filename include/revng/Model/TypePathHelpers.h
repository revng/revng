#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

namespace model::detail {

template<typename T>
std::string key(const T &Object) {
  return getNameFromYAMLScalar(KeyedObjectTraits<T>::key(Object));
}

inline std::string path(const model::Function &F) {
  return "/Functions/" + key(F);
}

inline std::string path(const model::DynamicFunction &F) {
  return "/ImportedDynamicFunctions/" + key(F);
}

inline std::string path(const model::TypeDefinition &T) {
  return "/TypeDefinitions/" + key(T);
}

inline std::string path(const model::EnumDefinition &Enum,
                        const model::EnumEntry &Entry) {
  return path(Enum) + "/EnumDefinition::Entries/" + key(Entry);
}

inline std::string path(const model::StructDefinition &Struct,
                        const model::StructField &Field) {
  return path(Struct) + "/StructDefinition::Fields/" + key(Field);
}

inline std::string path(const model::UnionDefinition &Union,
                        const model::UnionField &Field) {
  return path(Union) + "/UnionDefinition::Fields/" + key(Field);
}

inline std::string argumentPath(const model::CABIFunctionDefinition &CFT,
                                const model::Argument &Argument) {
  return path(CFT) + "/CABIFunctionDefinition::Arguments/" + key(Argument);
}

inline std::string argumentPath(const model::RawFunctionDefinition &RFT,
                                const model::NamedTypedRegister &Argument) {
  return path(RFT) + "/RawFunctionDefinition::Arguments/" + key(Argument);
}

inline std::string returnValuePath(const model::RawFunctionDefinition &RFT,
                                   const model::NamedTypedRegister &Argument) {
  return path(RFT) + "/RawFunctionDefinition::ReturnValues/" + key(Argument);
}

inline std::string path(const model::Segment &Segment) {
  return "/Segments/" + key(Segment);
}

} // namespace model::detail
