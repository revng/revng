#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Target.h"
#include "revng/Pipes/PipelineManager.h"

// NOLINTBEGIN

struct rp_error_reason {
public:
  std::string Message;
  std::string Location;

  rp_error_reason(std::string Message, std::string Location) :
    Message(std::move(Message)), Location(std::move(Location)) {}
};

struct rp_document_error {
public:
  std::vector<rp_error_reason> Reasons;
  std::string LocationType;
  std::string ErrorType;

  rp_document_error(std::string ErrorType, std::string LocationType) :
    LocationType(std::move(LocationType)), ErrorType(std::move(ErrorType)) {}
};

struct rp_simple_error {
public:
  std::string Message;
  std::string ErrorType;

  rp_simple_error(std::string Message, std::string Type) :
    Message(std::move(Message)), ErrorType(std::move(Type)) {}
};

using rp_error = std::variant<std::monostate /* allows "no value" state */,
                              rp_simple_error,
                              rp_document_error>;

typedef revng::pipes::PipelineManager rp_manager;
typedef const pipeline::Kind rp_kind;
typedef const pipeline::Rank rp_rank;
typedef pipeline::Step rp_step;
typedef pipeline::ContainerSet::value_type rp_container;
typedef const pipeline::ContainerFactorySet::value_type rp_container_identifier;
typedef const pipeline::Target rp_target;
typedef const pipeline::TargetsList rp_targets_list;
typedef const pipeline::Step::AnalysisValueType rp_analysis;
typedef const pipeline::DiffMap rp_diff_map;
typedef llvm::StringMap<std::string> rp_string_map;
typedef pipeline::InvalidationMap rp_invalidations;
typedef llvm::SmallVector<char, 0> rp_buffer;
typedef pipeline::ContainerToTargetsMap rp_container_targets_map;
typedef const pipeline::AnalysesList rp_analyses_list;

// NOLINTEND
