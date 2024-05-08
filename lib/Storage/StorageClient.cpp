/// \file StorageClient.cpp
/// \brief Opaque interface to file-related operations

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Storage/StorageClient.h"

#include "LocalStorageClient.h"
#include "S3StorageClient.h"

llvm::Expected<std::unique_ptr<revng::StorageClient>>
revng::StorageClient::fromPathOrURL(llvm::StringRef URL) {
  revng_assert(not URL.empty());
  if (S3StorageClient::isS3URL(URL)) {
    return S3StorageClient::fromURL(URL);
  } else {
    return std::make_unique<revng::LocalStorageClient>(URL);
  }
}
