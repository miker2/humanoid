#include "data_registrar.h"

DataRegistrar::DataRegistrar() {}

auto DataRegistrar::getInstance() -> DataRegistrar& {
  static DataRegistrar instance_s;

  return instance_s;
}

void DataRegistrar::init() {
  std::cout << "Column info size: " << sizeof(reg_info::Column) << std::endl;
  std::cout << "Header size: " << vars_.size() * sizeof(reg_info::Column) << std::endl;
  std::cout << "Frame size: " << offset_ << std::endl;
}

void DataRegistrar::tick() {}

void DataRegistrar::dumpColumns() {
  std::cout << "===== Registered vars =====" << std::endl;
  for (const auto& var : vars_) {
    std::cout << var->name << " - size: " << static_cast<uint16_t>(var->info.size)
              << ", offset: " << var->info.offset << ", type: " << static_cast<int>(var->info.type)
              << ", addr: " << var->addr << std::endl;
  }
  std::cout << "===========================" << std::endl;
}
