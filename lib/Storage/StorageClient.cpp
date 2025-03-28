/// \file StorageClient.cpp
/// \brief Opaque interface to file-related operations

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Storage/StorageClient.h"

#include "LocalStorageClient.h"

llvm::Expected<std::unique_ptr<revng::StorageClient>>
revng::StorageClient::fromPathOrURL(llvm::StringRef URL) {
  revng_assert(not URL.empty());
  return std::make_unique<revng::LocalStorageClient>(URL);
}
