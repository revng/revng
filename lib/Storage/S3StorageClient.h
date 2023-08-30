#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "aws/core/auth/AWSCredentials.h"
#include "aws/s3/S3Client.h"

#include "llvm/ADT/StringRef.h"

#include "revng/Storage/StorageClient.h"

namespace revng {

class S3WritableFile;

class S3StorageClient : public StorageClient {
private:
  Aws::Auth::AWSCredentials Credentials;
  Aws::S3::S3Client Client;
  std::string Bucket;
  std::string SubPath;
  std::string RedactedURL;
  llvm::StringMap<std::string> FilenameMap;
  static constexpr auto IndexName = "index.yml";

public:
  S3StorageClient(llvm::StringRef URL);
  ~S3StorageClient() override = default;

  static llvm::Expected<std::unique_ptr<S3StorageClient>>
  fromURL(llvm::StringRef URL);

  static bool isS3URL(llvm::StringRef Input) {
    return Input.starts_with("s3://") or Input.starts_with("s3s://");
  }

  llvm::Expected<PathType> type(llvm::StringRef Path) override;
  llvm::Error createDirectory(llvm::StringRef Path) override;
  llvm::Error remove(llvm::StringRef Path) override;
  llvm::sys::path::Style getStyle() const override;

  llvm::Error copy(llvm::StringRef Source,
                   llvm::StringRef Destination) override;

  llvm::Expected<std::unique_ptr<ReadableFile>>
  getReadableFile(llvm::StringRef Path) override;

  llvm::Expected<std::unique_ptr<WritableFile>>
  getWritableFile(llvm::StringRef Path, ContentEncoding Encoding) override;

  llvm::Error commit() override;

  // In S3StorageClient, the Credentials are in the format:
  // '<username>:<password>'
  llvm::Error setCredentials(llvm::StringRef Credentials) override;

private:
  std::string dumpString() const override;
  std::string resolvePath(llvm::StringRef Path);
  friend class S3WritableFile;
};

} // namespace revng
