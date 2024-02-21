/// \file ConvertFunctionsToCABI.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/FunctionType/Conversion.h"
#include "revng/ABI/FunctionType/Support.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Pass/PurgeUnnamedAndUnreachableTypes.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Pipeline/Analysis.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/Kinds.h"
#include "revng/TupleTree/TupleTree.h"

// TODO: dismiss using VerifyHelper for this verification. Introduce a
//       new class instead.

static RecursiveCoroutine<bool>
checkVectorRegisterSupport(model::VerifyHelper &VH,
                           const model::TypeDefinition &Type);

static RecursiveCoroutine<bool>
checkVectorRegisterSupport(model::VerifyHelper &VH,
                           const model::QualifiedType &Type) {
  if (revng::is_contained_if(Type.Qualifiers(), model::Qualifier::isPointer)) {
    // If it's a pointer, it's acceptable no matter what it points to.
    rc_return true;
  }

  // `Array` and `Const` do not impact the type, so we can just ignore them.
  const model::TypeDefinition *Unqualified = Type.UnqualifiedType().get();
  revng_assert(Unqualified != nullptr);
  rc_return rc_recur checkVectorRegisterSupport(VH, *Unqualified);
}

static RecursiveCoroutine<bool>
checkVectorRegisterSupport(model::VerifyHelper &VH,
                           const model::PrimitiveKind::Values &Kind) {
  rc_return VH.maybeFail(Kind != model::PrimitiveKind::Float,
                         "Floating Point primitive found.");
}

template<typename RealType>
inline RecursiveCoroutine<bool>
underlyingHelper(model::VerifyHelper &VH, const model::TypeDefinition &Value) {
  const RealType *Cast = llvm::cast<RealType>(&Value);
  rc_return rc_recur checkVectorRegisterSupport(VH, Cast->UnderlyingType());
}

static RecursiveCoroutine<bool>
checkVectorRegisterSupport(model::VerifyHelper &VH,
                           const model::DefinitionReference &Reference) {
  const model::TypeDefinition *Pointer = Reference.getConst();
  revng_assert(Pointer != nullptr);
  rc_return rc_recur checkVectorRegisterSupport(VH, *Pointer);
}

static RecursiveCoroutine<bool>
checkVectorRegisterSupport(model::VerifyHelper &VH,
                           const model::TypeDefinition &Type) {
  if (VH.isVerified(&Type))
    rc_return true;

  // Ensure we never recur indefinitely
  if (VH.isVerificationInProgress(&Type))
    rc_return VH.fail();

  VH.verificationInProgress(&Type);

  bool Result = false;

  switch (Type.Kind()) {
  case model::TypeDefinitionKind::PrimitiveDefinition: {
    using PrimitiveD = model::PrimitiveDefinition;
    const auto &Kind = llvm::cast<PrimitiveD>(&Type)->PrimitiveKind();
    Result = rc_recur checkVectorRegisterSupport(VH, Kind);
  } break;

  case model::TypeDefinitionKind::EnumDefinition:
    Result = rc_recur underlyingHelper<model::EnumDefinition>(VH, Type);
    break;

  case model::TypeDefinitionKind::TypedefDefinition:
    Result = rc_recur underlyingHelper<model::TypedefDefinition>(VH, Type);
    break;

  case model::TypeDefinitionKind::StructDefinition:
    Result = true;
    for (const auto &F : llvm::cast<model::StructDefinition>(&Type)->Fields())
      Result = Result && rc_recur checkVectorRegisterSupport(VH, F.Type());
    break;

  case model::TypeDefinitionKind::UnionDefinition:
    Result = true;
    for (const auto &F : llvm::cast<model::UnionDefinition>(&Type)->Fields())
      Result = Result && rc_recur checkVectorRegisterSupport(VH, F.Type());
    break;

  case model::TypeDefinitionKind::CABIFunctionDefinition: {
    Result = true;
    using CABIFT = model::CABIFunctionDefinition;
    for (const auto &A : llvm::cast<CABIFT>(&Type)->Arguments())
      Result = Result && rc_recur checkVectorRegisterSupport(VH, A.Type());
    const auto &ReturnType = llvm::cast<CABIFT>(&Type)->ReturnType();
    Result = Result && rc_recur checkVectorRegisterSupport(VH, ReturnType);
  } break;

  case model::TypeDefinitionKind::RawFunctionDefinition: {
    Result = true;
    using RawFT = model::RawFunctionDefinition;
    for (const auto &A : llvm::cast<RawFT>(&Type)->Arguments()) {
      auto Kind = model::Register::primitiveKind(A.Location());
      Result = Result && rc_recur checkVectorRegisterSupport(VH, Kind);
      Result = Result && rc_recur checkVectorRegisterSupport(VH, A.Type());
    }
    for (const auto &V : llvm::cast<RawFT>(&Type)->ReturnValues()) {
      auto Kind = model::Register::primitiveKind(V.Location());
      Result = Result && rc_recur checkVectorRegisterSupport(VH, Kind);
      Result = Result && rc_recur checkVectorRegisterSupport(VH, V.Type());
    }

    const auto &Stack = llvm::cast<RawFT>(&Type)->StackArgumentsType();
    if (not Stack.empty())
      Result = Result && rc_recur checkVectorRegisterSupport(VH, Stack);
  } break;

  default:
    revng_abort("Unknown type.");
  }

  if (Result) {
    VH.setVerified(&Type);
    VH.verificationCompleted(&Type);
  }

  rc_return VH.maybeFail(Result);
}

using namespace std::string_literals;

static Logger Log("function-type-conversion-to-cabi-analysis");

class ConvertFunctionsToCABI {
public:
  static constexpr auto Name = "convert-functions-to-cabi";
  inline static const std::tuple Options = {
    // Allows overriding the default ABI with a specific value when invoking
    // the analysis.
    pipeline::Option("abi", "Invalid"),

    // Allows specifying the mode of operation,
    // - safe: only convert the function if ABI belongs to the "tested" list.
    // - unsafe: always convert the function.
    pipeline::Option("mode", "safe"),

    // Allows specifying the confidence we have in the ABI, which then leads to
    // different levels of strictness when doing the argument deductions
    // (different behaviour in cases where the function does not seem to comply
    // to the abi):
    // - low: use safe deduction that will avoid changing function in cases of
    //        non-compliance.
    // - high: override/discard any information about the function that does not
    //         comply with an ABI (i.e. an argument in a register that is not
    //         dedicated for passing arguments, etc.).
    pipeline::Option("confidence", "low")
  };
  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {};

  void run(pipeline::ExecutionContext &Context,
           std::string TargetABI,
           std::string Mode,
           std::string ABIConfidence) {
    auto &Model = revng::getWritableModelFromContext(Context);
    revng_assert(!TargetABI.empty());

    model::ABI::Values ABI = model::ABI::fromName(TargetABI);
    if (ABI == model::ABI::Values::Invalid) {
      revng_log(Log,
                "No ABI explicitly specified for the conversion, using the "
                "`Model->DefaultABI()`.");
      ABI = Model->DefaultABI();
    }

    // Minimize the negative impact on binaries with ABI that is not fully
    // supported by disabling the conversion by default.
    //
    // Use `--convert-functions-to-cabi-mode=unsafe` to force conversion even
    // when ABI is not considered fully tested.
    if (Mode != "safe") {
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
        model::ABI::AAPCS

        // There are known issues
        // model::ABI::AAPCS64,

        // There are known issues
        // model::ABI::SystemV_MIPS_o32,
        // model::ABI::SystemV_MIPSEL_o32

        // Unable to reliably test: no easy access to a compiler
        // model::ABI::Pascal_x86,

        // Unable to reliably test: QEMU aborts
        // model::ABI::SystemZ_s390x,
      };
      if (!llvm::is_contained(ABIsTheConversionIsEnabledFor, ABI)) {
        revng_log(Log,
                  "Analysis was aborted because the `safe` (default) mode of "
                  "the conversion was selected and the conversion for the "
                  "current ABI ('"
                    << model::ABI::getName(ABI).str()
                    << "') is not considered stable.");
        return;
      }
    }

    // Determines the strictness of register state deductions
    bool SoftDeductions = (ABIConfidence == "low");

    // This reuses the verification map within the `model::VerifyHelper` in
    // an incompatible manner. DO NOT pass this object into a normal
    // verification routine or things are going to break down.
    model::VerifyHelper VectorVH;

    // Choose the applicable functions and run the conversion for them.
    using abi::FunctionType::filterTypes;
    using RawFD = model::RawFunctionDefinition;
    auto ToConvert = filterTypes<RawFD>(Model->TypeDefinitions());
    for (model::RawFunctionDefinition *Old : ToConvert) {
      auto DefinitionPath = Model->getDefinitionReference(Old->key());
      if (!checkVectorRegisterSupport(VectorVH, DefinitionPath)) {
        // TODO: remove this check after `abi::FunctionType` supports vectors.
        revng_log(Log,
                  "Skip a function conversion because it requires vector type "
                  "support: "
                    << serializeToString(DefinitionPath));
        continue;
      }

      revng_log(Log,
                "Converting a function: " << serializeToString(DefinitionPath));
      if (Log.isEnabled()) {
        revng_assert(!DefinitionPath.empty());

        std::string Message = "";
        for (model::Function &Function : Model->Functions())
          if (Function.Prototype() == DefinitionPath)
            Message += "'" + Function.name().str().str() + "', ";

        if (!Message.empty()) {
          Message.resize(Message.size() - 2);
          revng_log(Log, "It's a prototype of " << Message);
        }
      }

      namespace FT = abi::FunctionType;
      if (auto New = FT::tryConvertToCABI(*Old, Model, ABI, SoftDeductions)) {
        // If the conversion succeeds, make sure the returned type is valid,
        revng_assert(New->isValid());

        // and verifies
        if (VerifyLog.isEnabled())
          New->get()->verify(true);

        revng_log(Log,
                  "Function Conversion Successful: "
                    << serializeToString(*New));
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
};

pipeline::RegisterAnalysis<ConvertFunctionsToCABI> ToCABIAnalysis;
