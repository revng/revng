//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/PassRegistry.h"

#include "revng/Pipebox/LLVMPipe.h"

namespace rpp = revng::pypeline::pipes;
using Configuration = rpp::detail::PureLLVMPassesPipeBase::Configuration;

template<>
struct llvm::yaml::MappingTraits<Configuration> {
  static void mapping(IO &IO, Configuration &Fields) {
    IO.mapRequired("Passes", Fields.Passes);
  }
};

Configuration Configuration::parse(llvm::StringRef Input) {
  return llvm::cantFail(fromString<Configuration>(Input));
}

namespace revng::pypeline::pipes::detail {

PureLLVMPassesPipeBase::PureLLVMPassesPipeBase(llvm::StringRef
                                                 StaticConfiguration) :
  StaticConfiguration(StaticConfiguration) {

  Configuration Configuration = Configuration::parse(StaticConfiguration);
  llvm::PassRegistry &Registry = *llvm::PassRegistry::getPassRegistry();
  for (llvm::StringRef PassName : Configuration.Passes) {
    const llvm::PassInfo *PassInfo = Registry.getPassInfo(PassName);
    revng_assert(PassInfo != nullptr,
                 ("llvm-pipe: Requested pass " + PassName
                  + " was not found in the PassRegistry")
                   .str()
                   .c_str());
    PassInfos.push_back(PassInfo);
  }
  TaskName = "PureLLVMPasses(" + llvm::join(Configuration.Passes, ",") + ")";
}

} // namespace revng::pypeline::pipes::detail
