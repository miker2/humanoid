#pragma once

#include <flexi_cfg/utils.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cassert>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

enum class Type : int8_t {
  kUNDEFINED = -1,
  kDOUBLE = 0,
  kFLOAT = 1,
  kINT = 2,
  kUINT8 = 3,
  kUINT64 = 4
};

static inline std::unordered_map<std::type_index, Type> type_map = {
    {std::type_index(typeid(double)), Type::kDOUBLE},
    {std::type_index(typeid(float)), Type::kFLOAT},
    {std::type_index(typeid(int)), Type::kINT},
    {std::type_index(typeid(uint8_t)), Type::kUINT8},
    {std::type_index(typeid(uint64_t)), Type::kUINT64}};

namespace reg_info {
constexpr size_t MAX_NAME_LEN{128};
using offset_t = uint32_t;
struct Column {
  char name[MAX_NAME_LEN];
  offset_t offset{0};
  std::underlying_type_t<Type> type{static_cast<std::underlying_type_t<Type>>(Type::kUNDEFINED)};
  uint8_t size{0};
};
}  // namespace reg_info

struct VarInfo {
  VarInfo(const std::string& name, void* ptr, Type type, size_t size, reg_info::offset_t offset)
      : name{name}, addr{ptr}, type{type} {
    assert(name.size() < 128);
    std::copy(std::begin(name), std::end(name), info.name);
    info.type = static_cast<std::underlying_type_t<Type>>(type);
    info.size = size;
    info.offset = offset;
  };
  virtual ~VarInfo() = default;

  std::string name;
  void* addr = nullptr;

  Type type{Type::kUNDEFINED};

  reg_info::Column info;
};

template <typename T>
struct VarInfoImpl : VarInfo {
  using value_type = T;

  VarInfoImpl(const std::string& name, T* ptr, reg_info::offset_t offset)
      : VarInfo(name, ptr, type_map.at(std::type_index(typeid(T))), sizeof(T), offset) {}
};

namespace log_helpers {
const std::array<std::string, 3> cartesian = {"x", "y", "z"};
const std::array<std::string, 4> quaternion = {"x", "y", "z", "w"};
}  // namespace log_helpers

class DataRegistrar {
 public:
  static auto getInstance() -> DataRegistrar&;

  template <typename T>
  void registerVar(const std::string& name, T* var);

  template <typename T, size_t N>
  void registerVar(const std::string& name, std::array<T, N>* var,
                   std::optional<std::array<std::string, N>> suffix);

  template <typename T, int N>
  void registerVar(const std::string& name, Eigen::Vector<T, N>* var,
                   std::optional<std::array<std::string, N>> suffix);

  template <typename T>
  void registerVar(const std::string& name, Eigen::Quaternion<T>* var);

  void init();

  void tick();

  void dumpColumns();

 protected:
  DataRegistrar();

  std::vector<std::unique_ptr<VarInfo>> vars_;

  std::set<std::string> registered_names_;
  std::set<void*> registered_addrs_;

  reg_info::offset_t offset_{0};
};

template <typename T>
void DataRegistrar::registerVar(const std::string& name, T* var) {
  if (registered_names_.contains(name)) {
    throw std::runtime_error(name + " already registered");
  }
  if (registered_addrs_.contains(var)) {
    auto dupe = std::find_if(std::begin(vars_), std::end(vars_),
                             [&](const auto& rvar) { return rvar->addr == var; });
    throw std::runtime_error("Duplicate address registered! " + (*dupe)->name + " vs " + name);
  }
  auto regvar = std::make_unique<VarInfoImpl<T>>(name, var, offset_);
  registered_names_.insert(name);
  registered_addrs_.insert(var);
  vars_.push_back(std::move(regvar));
  offset_ += vars_.back()->info.size;
}

template <typename T, size_t N>
void DataRegistrar::registerVar(const std::string& name, std::array<T, N>* var,
                                std::optional<std::array<std::string, N>> suffix) {
  for (size_t i = 0; i < N; ++i) {
    registerVar(utils::makeName(name, suffix.has_value() ? suffix.value()[i] : std::to_string(i)),
                var->data() + i);
  }
}

template <typename T, int N>
void DataRegistrar::registerVar(const std::string& name, Eigen::Vector<T, N>* var,
                                std::optional<std::array<std::string, N>> suffix) {
  for (size_t i = 0; i < N; ++i) {
    registerVar(utils::makeName(name, suffix.has_value() ? suffix.value()[i] : std::to_string(i)),
                var->data() + i);
  }
  // const auto foo = std::make_integer_sequence<int, N>;
}

template <typename T>
void DataRegistrar::registerVar(const std::string& name, Eigen::Quaternion<T>* var) {
  registerVar(utils::makeName(name, "x"), &(var->x()));
  registerVar(utils::makeName(name, "y"), &(var->y()));
  registerVar(utils::makeName(name, "z"), &(var->z()));
  registerVar(utils::makeName(name, "w"), &(var->w()));
}
