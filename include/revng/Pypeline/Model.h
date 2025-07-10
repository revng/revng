#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/Registry.h"
#include "revng/Support/MetaAddress.h"

class Model {
private:
  TupleTree<model::Binary> TheModel;

public:
  static constexpr llvm::StringRef Name = "Model";
  Model() : TheModel() {}

  std::set<detail::ModelPath> diff(const Model &Other) const {
    auto Diff = ::diff(*TheModel.get(), *Other.TheModel.get());
    std::set<detail::ModelPath> Result;
    for (auto &Entry : Diff.Changes) {
      std::optional<std::string>
        MaybePath = pathAsString<model::Binary>(Entry.Path);
      revng_assert(MaybePath.has_value());
      Result.insert(*MaybePath);
    }
    return Result;
  }

  Model clone() const {
    Model Clone = *this;
    return Clone;
  }

  std::set<ObjectID> children(const ObjectID &Obj, ObjectID::Kind Kind) const {
    std::set<ObjectID> Result;
    return Result;
  }

  detail::Buffer serialize() const {
    detail::Buffer Out;
    llvm::raw_svector_ostream OS(Out.get());
    TheModel.serialize(OS);
    return Out;
  }

  bool deserialize(llvm::StringRef Input) {
    auto MaybeModel = TupleTree<model::Binary>::fromString(Input);
    if (not MaybeModel) {
      llvm::consumeError(MaybeModel.takeError());
      return false;
    }
    TheModel = std::move(*MaybeModel);
    return true;
  }
};

struct RegisterModel {
  RegisterModel() {
    TheRegistry.registerPythonCallback([](nanobind::module_ &M) {
      nanobind::class_<Model>(M, Model::Name.data())
        .def(nanobind::init<>())
        .def("diff",
             [](Model *Handle, nanobind::handle_t<Model> Other) {
               return Handle->diff(*nanobind::cast<Model *>(Other));
             })
        .def("clone",
             [](Model *Handle) {
               Model Cloned = Handle->clone();
               return nanobind::cast<Model>(std::move(Cloned));
             })
        .def("serialize",
             [](Model *Handle) {
               detail::Buffer Buffer = Handle->serialize();
               llvm::ArrayRef<char> Ref = Buffer.release();
               return nanobind::bytes(Ref.data(), Ref.size());
             })
        .def("deserialize", [](Model *Handle, nanobind::str String) {
          return Handle->deserialize({ String.c_str() });
        });
    });
  }
};
