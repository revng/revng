#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Target.h"
#include "revng/Pipes/PipelineManager.h"

struct rp_error {
public:
  std::string Message;
  std::string Location;
  std::string LocationType;
  std::string ErrorType;

  rp_error(std::string Message,
           std::string Location,
           std::string ErrorType,
           std::string LocationType) :
    Message(std::move(Message)),
    Location(std::move(Location)),
    LocationType(std::move(LocationType)),
    ErrorType(std::move(ErrorType)) {}
};

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
typedef std::vector<rp_error> rp_error_list;
typedef llvm::StringMap<std::string> rp_string_map;
typedef pipeline::InvalidationMap rp_invalidations;
typedef llvm::SmallVector<char, 0> rp_buffer;
