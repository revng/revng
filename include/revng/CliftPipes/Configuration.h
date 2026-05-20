#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/YAMLTraits.h"

namespace revng::pypeline::piperuns {

struct CEmissionPipeConfiguration {
  bool DisableMarkup = false;
};

} // namespace revng::pypeline::piperuns

namespace detail {

using PipeConfiguration = revng::pypeline::piperuns::CEmissionPipeConfiguration;

} // namespace detail

template<>
struct llvm::yaml::MappingTraits<detail::PipeConfiguration> {
  static void mapping(IO &TheIO, ::detail::PipeConfiguration &Object) {
    TheIO.mapRequired("disable-markup", Object.DisableMarkup);
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
