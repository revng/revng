//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "mlir/IR/Attributes.h"

#include "revng/CliftEmitC/CCommentEmitter.h"
#include "revng/PTML/CDoxygenEmitter.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

using CCE = mlir::clift::CCommentEmitter;
void CCE::emitComment(llvm::StringRef Content) {
  ptml::emitDoxygenLineComment(Tokens).emit(Content);
}

void CCE::emitFunctionComment(mlir::clift::FunctionOp Function) {
  auto Guard = Tokens.enterRegion(ptml::CTokenEmitter::RegionKind::Commentable,
                                  Function.getHandle());

  auto Emitter = ptml::emitDoxygenLineComment(Tokens);

  bool NeedsAnEmptyLine = false;

  // Function comment
  mlir::Attribute RawAttribute = Function->getAttr("clift.comment");
  if (RawAttribute != nullptr) {
    llvm::StringRef CBody = mlir::cast<mlir::StringAttr>(RawAttribute)
                              .getValue();
    if (not CBody.empty()) {
      Emitter.emit(CBody);
      Emitter.emitNewline();
      NeedsAnEmptyLine = true;
    }
  }

  // `\param` comments
  bool AnyArgumentCommentsEmitted = false;
  for (unsigned I = 0; I < Function.getArgCount(); ++I) {
    auto Attrs = Function.getArgAttrs(I);
    auto GetStringAttr = [&Attrs](llvm::StringRef Name) {
      if (auto Attribute = Attrs.get(Name))
        return mlir::cast<mlir::StringAttr>(Attribute).getValue();
      else
        return llvm::StringRef{};
    };

    llvm::StringRef CommentBody = GetStringAttr("clift.comment");
    if (CommentBody.empty())
      continue;

    if (NeedsAnEmptyLine) {
      Emitter.emitEmptyLine();
      Emitter.emitNewline();
      NeedsAnEmptyLine = false;
    }

    llvm::StringRef Handle = GetStringAttr("clift.handle");
    revng_assert(not Handle.empty());
    auto ArgG = Tokens.enterRegion(ptml::CTokenEmitter::RegionKind::Commentable,
                                   Handle);

    static constexpr llvm::StringRef Keyword = "param";
    Emitter.emitKeyword(Keyword);
    Emitter.emitSpace();

    llvm::StringRef EmittedName = GetStringAttr("clift.name");
    Tokens.emitIdentifier(EmittedName,
                          Handle,
                          ptml::CTokenEmitter::EntityKind::FunctionParameter,
                          ptml::CTokenEmitter::IdentifierKind::Reference);
    Emitter.emitSpace();

    size_t Indentation = Keyword.size() + 2 + EmittedName.size() + 1;

    Emitter.indent(Indentation);
    Emitter.emit(CommentBody);
    Emitter.indent(-Indentation);
    Emitter.emitNewline();

    AnyArgumentCommentsEmitted = true;
  }

  // `\returns` comment
  auto RVCommentBody = Function.getFunctionType().getReturnValueComment();
  if (not RVCommentBody.empty()) {
    if (NeedsAnEmptyLine or AnyArgumentCommentsEmitted) {
      Emitter.emitEmptyLine();
      Emitter.emitNewline();
    }

    auto FunctionTypeHandle = Function.getFunctionType().getHandle();
    auto FTLoc = pipeline::locationFromString(revng::ranks::TypeDefinition,
                                              FunctionTypeHandle);
    revng_assert(FTLoc.has_value());
    auto RVLoc = FTLoc->transmute(revng::ranks::ReturnValue).toString();

    using RegionKind = ptml::CTokenEmitter::RegionKind;
    auto Guard = Tokens.enterRegion(RegionKind::Commentable, RVLoc);

    static constexpr llvm::StringRef Keyword = "returns";
    Emitter.emitKeyword(Keyword);
    Emitter.emitSpace();

    size_t Indentation = Keyword.size() + 2;

    Emitter.indent(Indentation);
    Emitter.emit(RVCommentBody);
    Emitter.indent(-Indentation);
  }
}
