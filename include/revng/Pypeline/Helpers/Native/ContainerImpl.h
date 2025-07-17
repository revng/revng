#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Helpers/Native/Registry.h"

namespace revng::pypeline::helpers::native {

template<IsContainer T>
class ContainerImpl final : public Container {
private:
  T Instance;

public:
  ContainerImpl() : Instance() {}
  ~ContainerImpl() override = default;

public:
  virtual void
  deserialize(const std::map<const ObjectID *, llvm::ArrayRef<const char>>
                &Input) override {
    Instance.deserialize(Input);
  }

  virtual std::map<ObjectID, pypeline::Buffer>
  serialize(llvm::ArrayRef<const ObjectID> ToSave) const override {
    std::vector<const ObjectID *> Input;
    for (const ObjectID &Element : ToSave)
      Input.push_back(&Element);

    return Instance.serialize(Input);
  }

public:
  virtual void *get() override { return &Instance; }
};

} // namespace revng::pypeline::helpers::native
