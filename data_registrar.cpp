#include "data_registrar.h"

DataRegistrar::DataRegistrar() {}

auto DataRegistrar::getInstance() -> DataRegistrar& {
  static DataRegistrar instance_s;

  return instance_s;
}

void DataRegistrar::init() {

}

void DataRegistrar::tick() {}

void DataRegistrar::dumpColumns() {
  for (const auto& var : vars_) {
    std::cout << var->name << " - size: " << var->info.size << ", offset: " << var->info.offset
              << ", type: " << static_cast<int>(var->info.type) << ", addr: " << var->addr
              << std::endl;
  }
}
