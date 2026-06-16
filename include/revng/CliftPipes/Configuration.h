#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/YAMLTraits.h"

namespace revng::pypeline::piperuns {

enum class EmissionMode {
  Recompilable,
  Editable
};

struct CEmissionPipeConfiguration {
  bool DisableMarkup = false;
  EmissionMode Mode = EmissionMode::Recompilable;
};

} // namespace revng::pypeline::piperuns

namespace detail {

using EmissionMode = revng::pypeline::piperuns::EmissionMode;
using PipeConfiguration = revng::pypeline::piperuns::CEmissionPipeConfiguration;

} // namespace detail

template<>
struct llvm::yaml::ScalarEnumerationTraits<detail::EmissionMode> {
  static void enumeration(IO &IO, ::detail::EmissionMode &Value) {
    IO.enumCase(Value, "recompilable", ::detail::EmissionMode::Recompilable);
    IO.enumCase(Value, "editable", ::detail::EmissionMode::Editable);
  }
};

template<>
struct llvm::yaml::MappingTraits<detail::PipeConfiguration> {
  static void mapping(IO &TheIO, ::detail::PipeConfiguration &Value) {
    TheIO.mapOptional("disable-markup", Value.DisableMarkup);
    TheIO.mapOptional("emission-mode", Value.Mode);
  }
};

namespace revng::pypeline::piperuns {

inline CEmissionPipeConfiguration
parseCEmissionPipeConfiguration(llvm::StringRef Config) {
  if (Config.empty())
    return CEmissionPipeConfiguration{};

  return llvm::cantFail(fromString<CEmissionPipeConfiguration>(Config));
}

} // namespace revng::pypeline::piperuns
