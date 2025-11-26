#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Model.h"

class ApplyDiff {
public:
  static constexpr llvm::StringRef Name = "apply-diff";

  llvm::Error run(class Model &Model,
                  const revng::pypeline::Request &Incoming,
                  llvm::StringRef Configuration) {
    auto MaybeDiff = TupleTreeDiff<model::Binary>::fromString(Configuration);
    if (not MaybeDiff)
      return MaybeDiff.takeError();

    if (llvm::Error Error = MaybeDiff->apply(Model.get()); Error)
      return Error;

    return llvm::Error::success();
  }
};

// NOTE: this is a temporary hack to allow remote clients (e.g. clients using
//       the daemon) to verify that a diff will apply without actually changing
//       the current model.
class VerifyDiff {
public:
  static constexpr llvm::StringRef Name = "verify-diff";

  llvm::Error run(class Model &Model,
                  const revng::pypeline::Request &Incoming,
                  llvm::StringRef Configuration) {
    auto MaybeDiff = TupleTreeDiff<model::Binary>::fromString(Configuration);
    if (not MaybeDiff)
      return MaybeDiff.takeError();
    else
      return llvm::Error::success();
  }
};

class SetModel {
public:
  static constexpr llvm::StringRef Name = "set-model";

  llvm::Error run(class Model &Model,
                  const revng::pypeline::Request &Incoming,
                  llvm::StringRef Configuration) {
    auto MaybeModel = TupleTree<model::Binary>::fromString(Configuration);
    if (not MaybeModel)
      return MaybeModel.takeError();

    if (not MaybeModel->verify())
      return revng::createError("SetModel failed: model failed to verify");

    Model.get() = *MaybeModel;
    return llvm::Error::success();
  }
};

// NOTE: this is a temporary hack to allow remote clients (e.g. clients using
//       the daemon) to run `model::verify` without actually changing the
//       current model.
class VerifyModel {
public:
  static constexpr llvm::StringRef Name = "verify-model";

  llvm::Error run(class Model &Model,
                  const revng::pypeline::Request &Incoming,
                  llvm::StringRef Configuration) {
    auto MaybeModel = TupleTree<model::Binary>::fromString(Configuration);
    if (not MaybeModel)
      return MaybeModel.takeError();

    if (not MaybeModel->verify())
      return revng::createError("model failed to verify");
    else
      return llvm::Error::success();
  }
};
