#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

enum class Type : int16_t {
  kUNDEFINED = -1,
  kDOUBLE = 0,
  kFLOAT = 1,
  kINT = 2,
  kUINT8 = 3,
};

static inline std::unordered_map<std::type_index, Type> type_map = {
    {std::type_index(typeid(double)), Type::kDOUBLE},
    {std::type_index(typeid(float)), Type::kFLOAT},
    {std::type_index(typeid(int)), Type::kINT},
    {std::type_index(typeid(uint8_t)), Type::kUINT8}};

struct VarInfo {
  VarInfo(const std::string& name, void* ptr, Type type, size_t size)
      : name{name}, addr{ptr}, type{type}, size{size} {};
  virtual ~VarInfo() = default;

  std::string name;
  void* addr = nullptr;

  Type type{Type::kUNDEFINED};
  size_t size{0};
};

template <typename T>
struct VarInfoImpl : VarInfo {
  using value_type = T;

  VarInfoImpl(const std::string& name, T* ptr)
      : VarInfo(name, ptr, type_map.at(std::type_index(typeid(T))), sizeof(T)) {}
};

class DataRegistrar {
 public:
  static auto getInstance() -> DataRegistrar&;

  template <typename T>
  void registerVar(const std::string& name, T* var);

 protected:
  DataRegistrar();

  std::vector<std::unique_ptr<VarInfo>> vars_;
};

template <typename T>
void DataRegistrar::registerVar(const std::string& name, T* var) {
  auto regvar = std::make_unique<VarInfoImpl<T>>(name, var);
  std::cout << "Registering '" << regvar->name << "' of type '" << typeid(T).name()
            << "' at address: " << regvar->addr << std::endl;
  std::cout << " - Type: " << static_cast<int>(regvar->type) << ", size: " << regvar->size
            << std::endl;
  vars_.push_back(std::move(regvar));
}
