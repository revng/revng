//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTQEMUHELPERTRANSLATION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

enum class QEMUTypeKind : uint8_t {
  None,
  Discard,
  Signed,
  Unsigned,
  Float,
};

// Represents the transformed type of a QEMU helper.
// Zero size specifies that only the kind is changed.
struct QEMUType {
  QEMUTypeKind Kind;
  uint8_t Size;
};

struct QEMUHelperDescriptor {
  std::string_view Identifier;
  QEMUType ReturnType;
  QEMUType ParameterTypes[5];
};

static constexpr QEMUType Discard = { QEMUTypeKind::Discard, 0 };

static constexpr QEMUType Ixx = { QEMUTypeKind::Signed, 0 };
static constexpr QEMUType I16 = { QEMUTypeKind::Signed, 2 };
static constexpr QEMUType I32 = { QEMUTypeKind::Signed, 4 };
static constexpr QEMUType I64 = { QEMUTypeKind::Signed, 8 };

static constexpr QEMUType Uxx = { QEMUTypeKind::Unsigned, 0 };
static constexpr QEMUType U16 = { QEMUTypeKind::Unsigned, 2 };
static constexpr QEMUType U32 = { QEMUTypeKind::Unsigned, 4 };
static constexpr QEMUType U64 = { QEMUTypeKind::Unsigned, 8 };

static constexpr QEMUType F32 = { QEMUTypeKind::Float, 4 };
static constexpr QEMUType F64 = { QEMUTypeKind::Float, 8 };

// Booleans are treated as unsigned. Using an explicit name is useful as a
// documentation device, however.
static constexpr QEMUType Bool = Uxx;

// List of translated QEMU helper functions and the typing applied to them.
// NOTE: The helpers must be listed in alphabetical order.
static constexpr QEMUHelperDescriptor QEMUHelpers[] = {
  { "float32_abs", F32, { F32 } },
  { "float32_add", F32, { F32, F32, Discard } },
  { "float32_chs", F32, { F32 } },
  { "float32_compare", Ixx, { F32, F32, Discard } },
  { "float32_compare_quiet", Ixx, { F32, F32, Discard } },
  { "float32_default_nan", F32, { Discard } },
  { "float32_div", F32, { F32, F32, Discard } },
  { "float32_eq", Bool, { F32, F32, Discard } },
  { "float32_eq_quiet", Bool, { F32, F32, Discard } },
  { "float32_exp2", F32, { F32, Discard } },
  { "float32_is_any_nan", Bool, { F32 } },
  { "float32_is_denormal", Bool, { F32 } },
  { "float32_is_infinity", Bool, { F32 } },
  { "float32_is_neg", Bool, { F32 } },
  { "float32_is_normal", Bool, { F32 } },
  { "float32_is_quiet_nan", Bool, { F32, Discard } },
  { "float32_is_signaling_nan", Bool, { F32, Discard } },
  { "float32_is_zero", Bool, { F32 } },
  { "float32_is_zero_or_denormal", Bool, { F32 } },
  { "float32_is_zero_or_normal", Bool, { F32 } },
  { "float32_le", Bool, { F32, F32, Discard } },
  { "float32_le_quiet", Bool, { F32, F32, Discard } },
  { "float32_log2", F32, { F32, Discard } },
  { "float32_lt", Bool, { F32, F32, Discard } },
  { "float32_lt_quiet", Bool, { F32, F32, Discard } },
  { "float32_max", F32, { F32, F32, Discard } },
  { "float32_maximum_number", F32, { F32, F32, Discard } },
  { "float32_maxnum", F32, { F32, F32, Discard } },
  { "float32_maxnummag", F32, { F32, F32, Discard } },
  { "float32_min", F32, { F32, F32, Discard } },
  { "float32_minimum_number", F32, { F32, F32, Discard } },
  { "float32_minnum", F32, { F32, F32, Discard } },
  { "float32_minnummag", F32, { F32, F32, Discard } },
  { "float32_mul", F32, { F32, F32, Discard } },
  { "float32_muladd", F32, { F32, F32, F32, Uxx, Discard } },
  { "float32_rem", F32, { F32, F32, Discard } },
  { "float32_round_to_int", F32, { F32, Discard } },
  { "float32_scalbn", F32, { F32, Ixx, Discard } },
  { "float32_set_sign", F32, { F32, Uxx } },
  { "float32_silence_nan", F32, { F32, Discard } },
  { "float32_sqrt", F32, { F32, Discard } },
  { "float32_squash_input_denormal", F32, { F32, Discard } },
  { "float32_sub", F32, { F32, F32, Discard } },
  { "float32_to_float64", F64, { F32, Discard } },
  { "float32_to_int16", I16, { F32, Discard } },
  { "float32_to_int16_round_to_zero", I16, { F32, Discard } },
  { "float32_to_int16_scalbn", I16, { F32, Discard, Ixx, Discard } },
  { "float32_to_int32", I32, { F32, Discard } },
  { "float32_to_int32_round_to_zero", I32, { F32, Discard } },
  { "float32_to_int32_scalbn", I32, { F32, Discard, Ixx, Discard } },
  { "float32_to_int64", I64, { F32, Discard } },
  { "float32_to_int64_round_to_zero", I64, { F32, Discard } },
  { "float32_to_int64_scalbn", I64, { F32, Discard, Ixx, Discard } },
  { "float32_to_uint16", U16, { F32, Discard } },
  { "float32_to_uint16_round_to_zero", U16, { F32, Discard } },
  { "float32_to_uint16_scalbn", U16, { F32, Discard, Ixx, Discard } },
  { "float32_to_uint32", U32, { F32, Discard } },
  { "float32_to_uint32_round_to_zero", U32, { F32, Discard } },
  { "float32_to_uint32_scalbn", U32, { F32, Discard, Ixx, Discard } },
  { "float32_to_uint64", U64, { F32, Discard } },
  { "float32_to_uint64_round_to_zero", U64, { F32, Discard } },
  { "float32_to_uint64_scalbn", U64, { F32, Discard, Ixx, Discard } },
  { "float32_unordered", Bool, { F32, F32, Discard } },
  { "float32_unordered_quiet", Bool, { F32, F32, Discard } },
  { "float64_abs", F64, { F64 } },
  { "float64_add", F64, { F64, F64, Discard } },
  { "float64_chs", F64, { F64 } },
  { "float64_compare", Ixx, { F64, F64, Discard } },
  { "float64_compare_quiet", Ixx, { F64, F64, Discard } },
  { "float64_default_nan", F64, { Discard } },
  { "float64_div", F64, { F64, F64, Discard } },
  { "float64_eq", Bool, { F64, F64, Discard } },
  { "float64_eq_quiet", Bool, { F64, F64, Discard } },
  { "float64_is_any_nan", Bool, { F64 } },
  { "float64_is_denormal", Bool, { F64 } },
  { "float64_is_infinity", Bool, { F64 } },
  { "float64_is_neg", Bool, { F64 } },
  { "float64_is_normal", Bool, { F64 } },
  { "float64_is_quiet_nan", Bool, { F64, Discard } },
  { "float64_is_signaling_nan", Bool, { F64, Discard } },
  { "float64_is_zero", Bool, { F64 } },
  { "float64_is_zero_or_denormal", Bool, { F64 } },
  { "float64_is_zero_or_normal", Bool, { F64 } },
  { "float64_le", Bool, { F64, F64, Discard } },
  { "float64_le_quiet", Bool, { F64, F64, Discard } },
  { "float64_log2", F64, { F64, Discard } },
  { "float64_lt", Bool, { F64, F64, Discard } },
  { "float64_lt_quiet", Bool, { F64, F64, Discard } },
  { "float64_max", F64, { F64, F64, Discard } },
  { "float64_maximum_number", F64, { F64, F64, Discard } },
  { "float64_maxnum", F64, { F64, F64, Discard } },
  { "float64_maxnummag", F64, { F64, F64, Discard } },
  { "float64_min", F64, { F64, F64, Discard } },
  { "float64_minimum_number", F64, { F64, F64, Discard } },
  { "float64_minnum", F64, { F64, F64, Discard } },
  { "float64_minnummag", F64, { F64, F64, Discard } },
  { "float64_mul", F64, { F64, F64, Discard } },
  { "float64_muladd", F64, { F64, F64, F64, Uxx, Discard } },
  { "float64_rem", F64, { F64, F64, Discard } },
  { "float64_round_to_int", F64, { F64, Discard } },
  { "float64_scalbn", F64, { F64, Ixx, Discard } },
  { "float64_set_sign", F64, { F64, Uxx } },
  { "float64_silence_nan", F64, { F64, Discard } },
  { "float64_sqrt", F64, { F64, Discard } },
  { "float64_squash_input_denormal", F64, { F64, Discard } },
  { "float64_sub", F64, { F64, F64, Discard } },
  { "float64_to_float32", F32, { F64, Discard } },
  { "float64_to_int16", I16, { F64, Discard } },
  { "float64_to_int16_round_to_zero", I16, { F64, Discard } },
  { "float64_to_int16_scalbn", I16, { F64, Discard, Ixx, Discard } },
  { "float64_to_int32", I32, { F64, Discard } },
  { "float64_to_int32_modulo", I32, { F64, Discard, Discard } },
  { "float64_to_int32_round_to_zero", I32, { F64, Discard } },
  { "float64_to_int32_scalbn", I32, { F64, Discard, Ixx, Discard } },
  { "float64_to_int64", I64, { F64, Discard } },
  { "float64_to_int64_modulo", I64, { F64, Discard, Discard } },
  { "float64_to_int64_round_to_zero", I64, { F64, Discard } },
  { "float64_to_int64_scalbn", I64, { F64, Discard, Ixx, Discard } },
  { "float64_to_uint16", U16, { F64, Discard } },
  { "float64_to_uint16_round_to_zero", U16, { F64, Discard } },
  { "float64_to_uint16_scalbn", U16, { F64, Discard, Ixx, Discard } },
  { "float64_to_uint32", U32, { F64, Discard } },
  { "float64_to_uint32_round_to_zero", U32, { F64, Discard } },
  { "float64_to_uint32_scalbn", U32, { F64, Discard, Ixx, Discard } },
  { "float64_to_uint64", U64, { F64, Discard } },
  { "float64_to_uint64_round_to_zero", U64, { F64, Discard } },
  { "float64_to_uint64_scalbn", U64, { F64, Discard, Ixx, Discard } },
  { "float64_unordered", Bool, { F64, F64, Discard } },
  { "float64_unordered_quiet", Bool, { F64, F64, Discard } },
  { "float64r32_add", F64, { F64, F64, Discard } },
  { "float64r32_div", F64, { F64, F64, Discard } },
  { "float64r32_mul", F64, { F64, F64, Discard } },
  { "float64r32_muladd", F64, { F64, F64, F64, Uxx, Discard } },
  { "float64r32_sqrt", F64, { F64, Discard } },
  { "float64r32_sub", F64, { F64, F64, Discard } },
  { "int16_to_float32", F32, { I16, Discard } },
  { "int16_to_float32_scalbn", F32, { I16, Ixx, Discard } },
  { "int16_to_float64", F64, { I16, Discard } },
  { "int16_to_float64_scalbn", F64, { I16, Ixx, Discard } },
  { "int32_to_float32", F32, { I32, Discard } },
  { "int32_to_float32_scalbn", F32, { I32, Ixx, Discard } },
  { "int32_to_float64", F64, { I32, Discard } },
  { "int32_to_float64_scalbn", F64, { I32, Ixx, Discard } },
  { "int64_to_float32", F32, { I64, Discard } },
  { "int64_to_float32_scalbn", F32, { I64, Ixx, Discard } },
  { "int64_to_float64", F64, { I64, Discard } },
  { "int64_to_float64_scalbn", F64, { I64, Ixx, Discard } },
  { "uint16_to_float32", F32, { U16, Discard } },
  { "uint16_to_float32_scalbn", F32, { U16, Ixx, Discard } },
  { "uint16_to_float64", F64, { U16, Discard } },
  { "uint16_to_float64_scalbn", F64, { U16, Ixx, Discard } },
  { "uint32_to_float32", F32, { U32, Discard } },
  { "uint32_to_float32_scalbn", F32, { U32, Ixx, Discard } },
  { "uint32_to_float64", F64, { U32, Discard } },
  { "uint32_to_float64_scalbn", F64, { U32, Ixx, Discard } },
  { "uint64_to_float32", F32, { U64, Discard } },
  { "uint64_to_float32_scalbn", F32, { U64, Ixx, Discard } },
  { "uint64_to_float64", F64, { U64, Discard } },
  { "uint64_to_float64_scalbn", F64, { U64, Ixx, Discard } },
};

static_assert(std::ranges::is_sorted(QEMUHelpers,
                                     std::less(),
                                     &QEMUHelperDescriptor::Identifier));

static mlir::Type materializeType(mlir::MLIRContext *Context,
                                  mlir::Type Type,
                                  const QEMUType &Descriptor) {
  switch (Descriptor.Kind) {
  case QEMUTypeKind::None:
    break;

  case QEMUTypeKind::Discard:
    return {};

  case QEMUTypeKind::Signed: {
    uint64_t Size = Descriptor.Size ? Descriptor.Size : getObjectSize(Type);
    return IntegerType::get(Context, IntegerKind::Signed, Size);
  }

  case QEMUTypeKind::Unsigned: {
    uint64_t Size = Descriptor.Size ? Descriptor.Size : getObjectSize(Type);
    return IntegerType::get(Context, IntegerKind::Unsigned, Size);
  }

  case QEMUTypeKind::Float:
    revng_assert(Descriptor.Size == 4 or Descriptor.Size == 8);
    return FloatType::get(Context, Descriptor.Size);
  }

  revng_abort("Unsupported parameter type");
}

static FunctionType
transformHelperType(FunctionType HelperType,
                    const QEMUHelperDescriptor &Descriptor) {

  llvm::SmallVector<mlir::Type> ParameterTypes;
  for (auto [I, T] : llvm::enumerate(HelperType.getArgumentTypes())) {
    revng_assert(I < std::size(Descriptor.ParameterTypes));

    if (auto NewType = materializeType(HelperType.getContext(),
                                       T,
                                       Descriptor.ParameterTypes[I]))
      ParameterTypes.push_back(NewType);
  }

  return FunctionType::get(HelperType.getContext(),
                           HelperType.getHandle(),
                           HelperType.getMutableName(),
                           HelperType.getMutableComment(),
                           materializeType(HelperType.getContext(),
                                           HelperType.getReturnType(),
                                           Descriptor.ReturnType),
                           ParameterTypes,
                           HelperType.getCAttributeList());
}

static bool retypeHelperFunctions(mlir::ModuleOp Module) {
  llvm::DenseMap<FunctionOp, const QEMUHelperDescriptor *> HelperFunctions;

  // Loop over module functions looking for supported QEMU helpers. The type of
  // each recognized function is changed and the function is recorded in the
  // above map for subsequent processing of callsites.
  for (mlir::Operation &Op : Module.getOps()) {
    auto Function = mlir::dyn_cast<FunctionOp>(&Op);
    if (not Function or not Function.isExternal())
      continue;

    auto I = std::ranges::lower_bound(QEMUHelpers,
                                      std::string_view(Function.getSymName()),
                                      std::less(),
                                      &QEMUHelperDescriptor::Identifier);

    if (I == std::ranges::end(QEMUHelpers)
        or I->Identifier != std::string_view(Function.getSymName()))
      continue;

    QEMUHelperDescriptor const &Descriptor = *I;

    auto [MI, Inserted] = HelperFunctions.try_emplace(Function, &Descriptor);
    revng_assert(Inserted);

    auto TransformedType = transformHelperType(Function.getFunctionType(),
                                               Descriptor);

    llvm::SmallVector<mlir::Attribute> ArgAttrs;
    for (auto [I, T] : llvm::enumerate(Descriptor.ParameterTypes)) {
      if (T.Kind != QEMUTypeKind::None and T.Kind != QEMUTypeKind::Discard)
        ArgAttrs.push_back(Function.getArgAttrs(I).getDictionaryAttr());
    }

    Function.setType(TransformedType);
    Function.setArgAttrsAttr(mlir::ArrayAttr::get(Function.getContext(),
                                                  ArgAttrs));
  }

  if (HelperFunctions.empty())
    return false;

  mlir::OpBuilder Builder(Module->getContext());

  // Find each use of an affected helper function and transform the call by
  // erasing any discarded arguments, and by inserting casts on the result and
  // arguments.
  Module->walk([&](UseOp Use) {
    auto Function = mlir::dyn_cast<FunctionOp>(Use.getUsedGlobal()
                                                 .getOperation());
    if (not Function)
      return;

    auto I = HelperFunctions.find(Function);
    if (I == std::ranges::end(HelperFunctions))
      return;

    auto Call = getOnlyUser<CallOp>(Use);
    if (not Call)
      return;

    const QEMUHelperDescriptor &Descriptor = *I->second;

    // Erase discarded arguments:
    {
      llvm::BitVector Erase(Call->getNumOperands());
      for (auto [I, T] : llvm::enumerate(Descriptor.ParameterTypes)) {
        if (T.Kind == QEMUTypeKind::Discard)
          Erase.set(I + 1);
      }

      Call->eraseOperands(Erase);
    }

    Builder.setInsertionPoint(Call);
    for (auto [I, T] : llvm::enumerate(Function.getArgumentTypes())) {
      mlir::Value Value = Call->getOperand(I + 1);

      if (not equivalent(Value.getType(), T))
        Call->setOperand(I + 1,
                         Builder.create<BitCastOp>(Call->getLoc(), T, Value));
    }

    if (auto T = Function.getReturnType(); Call.getType() != T) {
      Builder.setInsertionPointAfter(Call);
      getOnlyUse(Call)->set(Builder.create<BitCastOp>(Call->getLoc(),
                                                      Call.getType(),
                                                      Call));
      Call.getResult().setType(T);
    }

    Use.getResult().setType(Function.getType());
  });

  return true;
}

struct QEMUHelperTranslationPass
  : clift::impl::CliftQEMUHelperTranslationBase<QEMUHelperTranslationPass> {

  void runOnOperation() override {
    if (retypeHelperFunctions(getOperation())) {
      mlir::RewritePatternSet Patterns(&getContext());
      if (mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(Patterns))
            .failed())
        signalPassFailure();
    }
  }
};

} // namespace

PassPtr<mlir::ModuleOp> clift::createQEMUHelperTranslationPass() {
  return std::make_unique<QEMUHelperTranslationPass>();
}
