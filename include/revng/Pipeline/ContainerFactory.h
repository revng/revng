#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/Container.h"
#include "revng/Support/Assert.h"

namespace pipeline {
class ContainerFactoryBase {
public:
  virtual std::unique_ptr<ContainerBase>
  operator()(llvm::StringRef Name) const = 0;

  virtual llvm::StringRef mimeType() const = 0;

  virtual std::vector<revng::FilePath>
  getWrittenFiles(const revng::FilePath &Path) const = 0;

  virtual ~ContainerFactoryBase() = default;

  virtual std::unique_ptr<ContainerFactoryBase> clone() const = 0;
};

template<typename ContainerT, typename... Args>
class ContainerFactoryWithArgs : public ContainerFactoryBase {
private:
  std::tuple<Args...> GlobalValue;

public:
  ContainerFactoryWithArgs(Args &&...GlobalValue) :
    GlobalValue(std::make_tuple(std::forward<Args>(GlobalValue)...)) {}

  std::unique_ptr<ContainerBase>
  operator()(llvm::StringRef Name) const override {
    auto Creator = [Name]<typename... T>(T &&...Values) {
      return std::make_unique<ContainerT>(Name, std::forward<T>(Values)...);
    };
    return std::apply(Creator, GlobalValue);
  }

  llvm::StringRef mimeType() const override { return ContainerT::MIMEType; }

  std::vector<revng::FilePath>
  getWrittenFiles(const revng::FilePath &Path) const override {
    return ContainerT::getWrittenFiles(Path);
  }

  std::unique_ptr<ContainerFactoryBase> clone() const override {
    return std::make_unique<ContainerFactoryWithArgs>(*this);
  }

  ContainerFactoryWithArgs(const ContainerFactoryWithArgs &) = default;
  ContainerFactoryWithArgs(ContainerFactoryWithArgs &&) = default;
  ContainerFactoryWithArgs &operator=(ContainerFactoryWithArgs &&) = default;
  ContainerFactoryWithArgs &
  operator=(const ContainerFactoryWithArgs &) = default;
  ~ContainerFactoryWithArgs() override = default;
};

class ContainerFactory {
private:
  std::unique_ptr<ContainerFactoryBase> Content;

  ContainerFactory(std::unique_ptr<ContainerFactoryBase> Ptr) :
    Content(std::move(Ptr)) {
    revng_assert(Content != nullptr);
  }

public:
  template<typename T>
  static ContainerFactory create() {
    return ContainerFactory(std::make_unique<ContainerFactoryWithArgs<T>>());
  }

  template<typename T, typename... G>
  static ContainerFactory fromGlobal(G &&...Vals) {
    using Factory = ContainerFactoryWithArgs<T, G...>;
    return ContainerFactory(make_unique<Factory>(std::forward<G>(Vals)...));
  }

  ContainerFactory(const ContainerFactory &Other) :
    Content(Other.Content->clone()) {
    revng_assert(Content != nullptr);
  }

  ContainerFactory(ContainerFactory &&Other) = default;
  ~ContainerFactory() = default;

  ContainerFactory &operator=(const ContainerFactory &Other) {
    if (this == &Other)
      return *this;

    Content = Other.Content->clone();
    return *this;
  }

  ContainerFactory &operator=(ContainerFactory &&Other) = default;

  std::unique_ptr<ContainerBase> operator()(llvm::StringRef Name) const {
    return Content->operator()(Name);
  }

  llvm::StringRef mimeType() const { return Content->mimeType(); }

  std::vector<revng::FilePath>
  getWrittenFiles(const revng::FilePath &Path) const {
    return Content->getWrittenFiles(Path);
  }
};
} // namespace pipeline
