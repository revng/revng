/// \file S3StorageClient.cpp
/// \brief Implementation of StorageClient operations in S3

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>

#include "aws/core/Aws.h"
#include "aws/core/auth/AWSCredentials.h"
#include "aws/core/auth/AWSCredentialsProvider.h"
#include "aws/core/utils/logging/FormattedLogSystem.h"
#include "aws/s3/S3Client.h"
#include "aws/s3/model/CopyObjectRequest.h"
#include "aws/s3/model/DeleteObjectRequest.h"
#include "aws/s3/model/GetObjectRequest.h"
#include "aws/s3/model/HeadObjectRequest.h"
#include "aws/s3/model/PutObjectRequest.h"

#include "llvm/Support/Process.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/Storage/Path.h"
#include "revng/Support/Assert.h"
#include "revng/Support/OnQuit.h"
#include "revng/Support/PathList.h"
#include "revng/Support/TemporaryFile.h"

#include "S3StorageClient.h"
#include "Utils.h"

using FilenameMapType = llvm::StringMap<std::string>;

template<>
struct llvm::yaml::CustomMappingTraits<FilenameMapType> {
  static void inputOne(IO &IO, llvm::StringRef Key, FilenameMapType &Value) {
    IO.mapRequired(Key.str().c_str(), Value[Key]);
  }

  static void output(IO &IO, FilenameMapType &Value) {
    for (auto &Pair : Value) {
      IO.mapRequired(Pair.first().str().c_str(), Pair.second);
    }
  }
};

namespace {

using Aws::Utils::Logging::FormattedLogSystem;
using Aws::Utils::Logging::LogLevel;

Logger<> Logger("s3-storage");

class LoggerSystem : public FormattedLogSystem {
private:
  std::mutex Mutex;

public:
  LoggerSystem(LogLevel LogLevel) : FormattedLogSystem(LogLevel) {}
  ~LoggerSystem() override = default;
  void Flush() override { Logger.flush(); }

  void ProcessFormattedStatement(Aws::String &&Statement) override {
    // This function needs to be thread-safe, hence the lock_guard
    // TODO: check if the mangled text is the result of aggressive flushed
    //       and/or if buffering might alleviate this problem
    std::lock_guard Guard(Mutex);
    revng_log(Logger, Statement);
  }
};

bool SDKIsInitialized = false;
std::optional<llvm::ThreadPool> ThreadPool;

void initializeSDK() {
  revng_assert(!SDKIsInitialized);

  LogLevel Level(LogLevel::Info);
  Aws::SDKOptions Options;

  Options.loggingOptions.logLevel = Level;
  Options.loggingOptions.logger_create_fn = [Level]() {
    return std::make_shared<LoggerSystem>(Level);
  };
  Aws::InitAPI(Options);
  OnQuit->add([Options = std::move(Options)] { Aws::ShutdownAPI(Options); });

  ThreadPool.emplace(llvm::hardware_concurrency(8));
}

} // namespace

namespace revng {

static llvm::StringRef consumeSplit(llvm::StringRef &Input, char SplitChar) {
  size_t Pos = Input.find(SplitChar);
  revng_assert(Pos != llvm::StringRef::npos);
  llvm::StringRef Result = Input.slice(0, Pos);
  Input = Input.drop_front(Pos + 1);
  return Result;
}

template<typename T>
static llvm::Error
toError(const Aws::Utils::Outcome<T, Aws::S3::S3Error> &Request) {
  revng_check(!Request.IsSuccess());
  return revng::createError(Request.GetError().GetMessage());
}

std::string S3StorageClient::resolvePath(llvm::StringRef Path) {
  if (Path.empty()) {
    return SubPath;
  } else {
    revng_assert(checkPath(Path, getStyle()));
    return joinPath(getStyle(), SubPath, Path);
  }
}

// Given the path `dir1/dir2/filename.ext` this function will return the path
// `dir1/dir2/<random UUID>-filename.ext`
static std::string generateNewFilename(llvm::StringRef Path) {
  revng_assert(not Path.ends_with("/"));

  using uuid_t = uint8_t[16];
  uuid_t UUID;
  for (int I = 0; I < 16; I++)
    UUID[I] = (llvm::sys::Process::GetRandomNumber() % UINT8_MAX);

  std::string Output;
  llvm::raw_string_ostream OS(Output);
  OS.write_uuid(UUID);
  OS.flush();

  if (Path.contains("/")) {
    auto Parts = Path.rsplit("/");
    return Parts.first.str() + "/" + Output + "-" + Parts.second.str();
  } else {
    return Output + "-" + Path.str();
  }
}

static Aws::Auth::AWSCredentials readCredentials(llvm::StringRef Credentials) {
  llvm::StringRef Username = consumeSplit(Credentials, ':');
  llvm::StringRef Password = Credentials;
  return Aws::Auth::AWSCredentials{ Username.str(), Password.str() };
}

class S3ReadableFile : public ReadableFile {
private:
  TemporaryFile TempFile;
  std::unique_ptr<llvm::MemoryBuffer> Buffer;

public:
  S3ReadableFile(TemporaryFile &&TempFile,
                 std::unique_ptr<llvm::MemoryBuffer> &&Buffer) :
    TempFile(std::move(TempFile)), Buffer(std::move(Buffer)) {}
  ~S3ReadableFile() override = default;
  llvm::MemoryBuffer &buffer() override { return *Buffer; };
};

class AsyncUploadTask {
private:
  Aws::S3::S3Client &Client;
  Aws::S3::Model::PutObjectRequest Request;
  std::string Path;
  std::string NewFilename;
  TemporaryFile TempFile;

public:
  AsyncUploadTask(Aws::S3::S3Client &Client,
                  Aws::S3::Model::PutObjectRequest &&Request,
                  llvm::StringRef Path,
                  llvm::StringRef NewFilename,
                  TemporaryFile &&TempFile) :
    Client(Client),
    Request(std::move(Request)),
    Path(Path),
    NewFilename(NewFilename),
    TempFile(std::move(TempFile)) {}

  S3StorageClient::UploadResult operator()() {
    Aws::S3::Model::PutObjectOutcome Result;
    auto File = std::make_shared<Aws::FStream>(TempFile.path().str(),
                                               std::ios_base::in
                                                 | std::ios_base::binary);
    if (File->fail()) {
      return { Result, Path, NewFilename, "Could not open temporary file" };
    }

    Request.SetBody(File);
    Result = Client.PutObject(Request);
    return { Result, Path, NewFilename, std::string{} };
  }
};

class S3WritableFile : public WritableFile {
private:
  TemporaryFile TempFile;
  std::unique_ptr<llvm::raw_fd_ostream> OS;
  std::string Path;
  ContentEncoding Encoding;
  S3StorageClient &Client;

public:
  S3WritableFile(TemporaryFile &&TempFile,
                 std::unique_ptr<llvm::raw_fd_ostream> &&OS,
                 llvm::StringRef Path,
                 ContentEncoding Encoding,
                 S3StorageClient &Client) :
    TempFile(std::move(TempFile)),
    OS(std::move(OS)),
    Path(Path.str()),
    Encoding(Encoding),
    Client(Client) {}

  llvm::raw_pwrite_stream &os() override { return *OS; }
  llvm::Error commit() override {
    OS->flush();

    Aws::S3::Model::PutObjectRequest Request;
    Request.SetBucket(Client.Bucket);

    if (Encoding == ContentEncoding::Gzip)
      Request.SetContentEncoding("gzip");

    std::string NewFilename = generateNewFilename(Path);
    Request.SetKey(Client.resolvePath(NewFilename));

    auto Task = std::make_shared<AsyncUploadTask>(Client.Client,
                                                  std::move(Request),
                                                  Path,
                                                  NewFilename,
                                                  std::move(TempFile));
    using UploadResult = S3StorageClient::UploadResult;
    std::shared_future<UploadResult>
      ResultFuture = Client.TaskGroup.async([Task]() { return (*Task)(); });
    Client.PendingUploads.push_back(std::move(ResultFuture));
    return llvm::Error::success();
  }
};

class S3CredentialsProvider : public Aws::Auth::AWSCredentialsProvider {
private:
  Aws::Auth::AWSCredentials &Credentials;

public:
  explicit S3CredentialsProvider(Aws::Auth::AWSCredentials &Credentials) :
    Credentials(Credentials) {}
  ~S3CredentialsProvider() override = default;
  Aws::Auth::AWSCredentials GetAWSCredentials() override { return Credentials; }
};

S3StorageClient::S3StorageClient(llvm::StringRef RawURL) :
  TaskGroup(*ThreadPool) {
  // Url format is:
  // s3://<username>:<password>@<region>+<host:port>/<bucket name>/<path>
  revng_assert(isS3URL(RawURL));

  Aws::Client::ClientConfiguration Config(false, "standard", true);
  Config.enableEndpointDiscovery = false;

  llvm::StringRef URL(RawURL);

  if (URL.starts_with("s3://")) {
    Config.scheme = Aws::Http::Scheme::HTTP;
    RedactedURL += "s3://";
    revng_assert(URL.consume_front("s3://"));
  } else {
    Config.scheme = Aws::Http::Scheme::HTTPS;
    RedactedURL += "s3s://";
    revng_assert(URL.consume_front("s3s://"));
  }

  llvm::StringRef CredentialsString = consumeSplit(URL, '@');
  Credentials = readCredentials(CredentialsString);
  RedactedURL += "<user>:<pwd>@";

  Config.region = consumeSplit(URL, '+').str();
  Config.endpointOverride = consumeSplit(URL, '/').str();
  RedactedURL += Config.region + '+' + Config.endpointOverride + '/';

  Client = { std::make_shared<S3CredentialsProvider>(Credentials),
             Config,
             Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Always,
             false };

  if (URL.contains('/')) {
    Bucket = consumeSplit(URL, '/').str();
    SubPath = URL.str();
  } else {
    Bucket = URL.str();
    SubPath = "";
  }

  RedactedURL += Bucket + '/' + SubPath;
}

llvm::Expected<std::unique_ptr<S3StorageClient>>
S3StorageClient::fromURL(llvm::StringRef URL) {
  if (not SDKIsInitialized)
    initializeSDK();

  auto Instance = std::make_unique<S3StorageClient>(URL);
  std::string IndexPath = Instance->resolvePath(IndexName);

  Aws::S3::Model::GetObjectRequest Request;
  Request.SetBucket(Instance->Bucket);
  Request.SetKey(IndexPath);

  Aws::S3::Model::GetObjectOutcome Result = Instance->Client.GetObject(Request);
  if (not Result.IsSuccess()) {
    using Aws::Http::HttpResponseCode::NOT_FOUND;
    if (Result.GetError().GetResponseCode() == NOT_FOUND)
      return Instance;
    else
      return toError(Result);
  }

  std::string SerializedIndex;
  {
    constexpr size_t BufSize = 4096;
    char Buffer[BufSize];
    auto &Body = Result.GetResult().GetBody();

    while (!Body.eof()) {
      Body.read(Buffer, BufSize);
      SerializedIndex.append(Buffer, Body.gcount());
    }
  }

  llvm::yaml::Input YAMLInput(SerializedIndex);
  YAMLInput >> Instance->FilenameMap;

  return Instance;
}

std::string S3StorageClient::dumpString() const {
  return RedactedURL;
}

llvm::Expected<PathType> S3StorageClient::type(llvm::StringRef Path) {
  if (FilenameMap.count(Path) > 0) {
    Aws::S3::Model::HeadObjectRequest Request;
    Request.SetBucket(Bucket);
    Request.SetKey(resolvePath(FilenameMap[Path]));

    Aws::S3::Model::HeadObjectOutcome Result = Client.HeadObject(Request);
    if (not Result.IsSuccess()) {
      using Aws::Http::HttpResponseCode::NOT_FOUND;
      if (Result.GetError().GetResponseCode() == NOT_FOUND)
        return PathType::Missing;
      else
        return toError(Result);
    }

    return PathType::File;
  } else {
    std::string Prefix = Path.endswith("/") ? Path.str() : (Path.str() + "/");
    for (auto &[MapPath, _] : FilenameMap) {
      if (MapPath.startswith(Prefix))
        return PathType::Directory;
    }
    return PathType::Missing;
  }
}

llvm::Error S3StorageClient::createDirectory(llvm::StringRef Path) {
  return llvm::Error::success();
}

llvm::Error S3StorageClient::remove(llvm::StringRef Path) {
  revng_assert(FilenameMap.count(Path) != 0);
  FilenameMap.erase(Path);
  return llvm::Error::success();
}

llvm::sys::path::Style S3StorageClient::getStyle() const {
  return llvm::sys::path::Style::posix;
}

llvm::Error S3StorageClient::copy(llvm::StringRef Source,
                                  llvm::StringRef Destination) {
  if (FilenameMap.count(Source) == 0) {
    return revng::createError("Source file %s does not exist",
                              Source.str().c_str());
  }

  FilenameMap[Destination] = FilenameMap[Source];
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<ReadableFile>>
S3StorageClient::getReadableFile(llvm::StringRef Path) {
  using llvm::MemoryBuffer;
  if (FilenameMap.count(Path) == 0) {
    return revng::createError("File %s does not exist", Path.str().c_str());
  }

  Aws::S3::Model::GetObjectRequest Request;
  Request.SetBucket(Bucket);
  Request.SetKey(resolvePath(FilenameMap[Path]));

  Aws::S3::Model::GetObjectOutcome Result = Client.GetObject(Request);
  if (not Result.IsSuccess())
    return toError(Result);

  auto MaybeTemporary = TemporaryFile::make("revng-s3-storage");
  if (!MaybeTemporary) {
    return llvm::createStringError(MaybeTemporary.getError(),
                                   "Could not create temporary file");
  }

  std::ofstream OS(MaybeTemporary->path().str(), std::ios::binary);
  if (OS.fail()) {
    return revng::createError("Could not open temporary file");
  }

  constexpr size_t BufSize = 4096;
  llvm::SmallVector<char> Buffer;
  Buffer.resize_for_overwrite(BufSize);
  auto &Body = Result.GetResult().GetBody();

  while (!Body.eof()) {
    Body.read(Buffer.data(), BufSize);
    OS.write(Buffer.data(), Body.gcount());
  }

  OS.flush();
  OS.close();

  auto MaybeReadableStream = MemoryBuffer::getFile(MaybeTemporary->path());
  if (not MaybeReadableStream) {
    return llvm::createStringError(MaybeReadableStream.getError(),
                                   "Failed to open the file for reading");
  }

  return std::make_unique<S3ReadableFile>(std::move(MaybeTemporary.get()),
                                          std::move(MaybeReadableStream.get()));
}

llvm::Expected<std::unique_ptr<WritableFile>>
S3StorageClient::getWritableFile(llvm::StringRef Path,
                                 ContentEncoding Encoding) {
  auto MaybeTemporary = TemporaryFile::make("revng-s3-storage");
  if (!MaybeTemporary) {
    return llvm::createStringError(MaybeTemporary.getError(),
                                   "Could not create temporary file");
  }

  std::error_code EC;
  auto OS = std::make_unique<llvm::raw_fd_ostream>(MaybeTemporary->path(), EC);
  if (EC)
    return llvm::createStringError(EC, "Could not open temporary file");

  return std::make_unique<S3WritableFile>(std::move(MaybeTemporary.get()),
                                          std::move(OS),
                                          Path,
                                          Encoding,
                                          *this);
}

llvm::Error S3StorageClient::commit() {
  TaskGroup.wait();

  std::vector<llvm::Error> Errors;
  for (const auto &Element : PendingUploads) {
    const UploadResult &Result = Element.get();
    if (not Result.Error.empty())
      Errors.push_back(revng::createError(Result.Error));
    else if (not Result.Outcome.IsSuccess())
      Errors.push_back(toError(Result.Outcome));
    else
      FilenameMap[Result.Path] = Result.NewPath;
  }
  if (Errors.size() > 0) {
    return joinErrors(Errors);
  }

  std::string SerializedIndex;

  {
    llvm::raw_string_ostream OS(SerializedIndex);
    llvm::yaml::Output YAMLOutput(OS);
    YAMLOutput << FilenameMap;
  }

  Aws::S3::Model::PutObjectRequest Request;
  Request.SetBucket(Bucket);
  Request.SetKey(resolvePath(IndexName));

  auto Stream = std::make_shared<std::stringstream>(SerializedIndex,
                                                    std::ios_base::in
                                                      | std::ios_base::binary);

  Request.SetBody(Stream);
  Aws::S3::Model::PutObjectOutcome Result = Client.PutObject(Request);
  if (not Result.IsSuccess())
    return toError(Result);

  return llvm::Error::success();
}

llvm::Error S3StorageClient::setCredentials(llvm::StringRef Credentials) {
  this->Credentials = readCredentials(Credentials);
  return llvm::Error::success();
}

} // namespace revng
