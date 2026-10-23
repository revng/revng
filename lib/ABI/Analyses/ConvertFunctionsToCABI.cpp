//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/Analyses/ConvertFunctionsToCABI.h"
#include "revng/ABI/FunctionType/Conversion.h"
#include "revng/ABI/FunctionType/Support.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Model/Binary.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Model/Pass/PurgeUnnamedAndUnreachableTypes.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/IRHelpers.h"
#include "revng/TupleTree/TupleTree.h"

// TODO: Also take advantage of `::edges()` helpers for traversing
//       the file system.

static Logger Log("function-type-conversion-to-cabi-analysis");

static void convertFunctionsToCABI(TupleTree<model::Binary> &Model,
                                   model::ABI::Values ABI,
                                   llvm::StringRef Mode,
                                   llvm::StringRef ABIConfidence) {
  // Minimize the negative impact on binaries with ABI that is not fully
  // supported by disabling the conversion by default.
  //
  // Pass `Mode: unsafe` as a pipe configuration option to run on any ABI.
  if (Mode != "unsafe") {
    // TODO: extend this list.
    static constexpr std::array ABIsTheConversionIsEnabledFor = {
      model::ABI::SystemV_x86_64,
      model::ABI::Microsoft_x86_64,
      model::ABI::Microsoft_x86_64_vectorcall,
      model::ABI::SystemV_x86,
      model::ABI::SystemV_x86_regparm_3,
      model::ABI::SystemV_x86_regparm_2,
      model::ABI::SystemV_x86_regparm_1,
      model::ABI::Microsoft_x86_cdecl,
      model::ABI::Microsoft_x86_cdecl_gcc,
      model::ABI::Microsoft_x86_fastcall,
      model::ABI::Microsoft_x86_fastcall_gcc,
      model::ABI::Microsoft_x86_stdcall,
      model::ABI::Microsoft_x86_stdcall_gcc,
      model::ABI::Microsoft_x86_thiscall,
      model::ABI::Microsoft_x86_vectorcall,
      model::ABI::AAPCS_hardfloat,
      model::ABI::AAPCS_softfloat,
      model::ABI::AAPCS64,
      model::ABI::Apple_AAPCS64,
      model::ABI::Microsoft_AAPCS64

      // There are known issues
      // model::ABI::SystemV_MIPS_o32_hardfloat,
      // model::ABI::SystemV_MIPS_o32_softfloat,
      // model::ABI::SystemV_MIPSEL_o32_hardfloat,
      // model::ABI::SystemV_MIPSEL_o32_softfloat,

      // Unable to reliably test: QEMU aborts
      // model::ABI::SystemZ_s390x,
    };
    if (!llvm::is_contained(ABIsTheConversionIsEnabledFor, ABI)) {
      revng_log(Log,
                "Analysis was aborted because the `safe` (default) mode of "
                "the conversion was selected and the conversion for the "
                "current ABI (`"
                  << model::ABI::getName(ABI).str()
                  << "`) is not considered stable.");
      return;
    }
  }

  // Determines the strictness of register state deductions
  bool SoftDeductions = (ABIConfidence == "low");

  // Choose functions for the conversion, but leave DefaultPrototype alone
  // (in order to avoid invalidation),
  using abi::FunctionType::filterTypes;
  using RawFD = model::RawFunctionDefinition;
  // TODO: Switch to std::optional after C++26 support arrives.
  llvm::SmallVector<model::TypeDefinition *, 1> TypesToIgnore = {};
  if (auto &DefaultPrototype = Model->DefaultPrototype())
    if (auto *Definition = DefaultPrototype->tryGetAsDefinition())
      TypesToIgnore.push_back(Definition);
  auto ToConvert = filterTypes<RawFD>(Model->TypeDefinitions(), TypesToIgnore);

  using NB = model::CNameBuilder;
  auto NameBuilder = Log.isEnabled() ? std::make_optional<NB>(*Model) :
                                       std::nullopt;

  auto LogFunctionName = [&NameBuilder,
                          &Model](const model::TypeDefinition::Key &Key) {
    if (not Log.isEnabled())
      return;

    revng_log(Log,
              "Converting a function: "
                << toString(Model->getTypeDefinitionReference(Key)));

    std::string Names;
    constexpr llvm::StringRef Separator = ", ";
    for (model::Function &Function : Model->Functions())
      if (Function.prototype() && Function.prototype()->key() == Key)
        Names += '"' + NameBuilder->name(Function) + '"' + Separator.str();

    if (Names.empty()) {
      revng_log(Log, "There are no functions using it as a prototype.");
    } else {
      Names.resize(Names.size() - Separator.size());
      revng_log(Log, "It's a prototype of " << Names);
    }
  };

  // And convert them.
  for (model::RawFunctionDefinition *Old : ToConvert) {
    LogFunctionName(Old->key());

    namespace FT = abi::FunctionType;
    if (auto New = FT::tryConvertToCABI(*Old, Model, ABI, SoftDeductions)) {
      // If the conversion succeeds, make sure the returned type is valid,
      revng_assert(!New->isEmpty());

      // and verifies
      if (VerifyLog.isEnabled())
        New->get()->verify(true);

      revng_log(Log, "Function Conversion Successful: " << toString(*New));
    } else {
      // Do nothing if the conversion failed (the model is not modified).
      // `RawFunctionDefinition` is still used for those functions.
      // This might be an indication of an ABI misdetection.
      revng_log(Log, "Function Conversion Failed.");
    }
  }

  // Don't forget to clean up any possible remainders of removed types.
  purgeUnnamedAndUnreachableTypes(Model);
}

struct ConvertFunctionsToCABIConfiguration {
  std::string ABI;
  std::string Mode;
  std::string Confidence;
};

template<>
struct llvm::yaml::MappingTraits<ConvertFunctionsToCABIConfiguration> {
  static void mapping(IO &IO, ConvertFunctionsToCABIConfiguration &Fields) {
    IO.mapOptional("ABI", Fields.ABI);
    IO.mapOptional("Mode", Fields.Mode);
    IO.mapOptional("Confidence", Fields.Confidence);
  }
};

namespace revng::pypeline::analyses {

using Configuration = ::ConvertFunctionsToCABIConfiguration;

llvm::Error ConvertFunctionsToCABI::run(Model &Model,
                                        const Request &Incoming,
                                        llvm::StringRef StrConfiguration) {
  auto MaybeConfiguration = fromString<Configuration>(StrConfiguration);
  if (not MaybeConfiguration)
    return MaybeConfiguration.takeError();

  Configuration &Configuration = MaybeConfiguration.get();
  model::ABI::Values ABI;
  if (not Configuration.ABI.empty()) {
    ABI = model::ABI::fromName(Configuration.ABI);
  } else {
    revng_log(Log,
              "No ABI explicitly specified for the conversion, using the "
              "`Model->DefaultABI()`.");
    ABI = Model.get()->DefaultABI();
  }

  convertFunctionsToCABI(Model.get(),
                         ABI,
                         Configuration.Mode,
                         Configuration.Confidence);

  return llvm::Error::success();
}

} // namespace revng::pypeline::analyses
