#include "data_registrar.h"

DataRegistrar::DataRegistrar() {}

auto DataRegistrar::getInstance() -> DataRegistrar& {
  static DataRegistrar instance_s;

  return instance_s;
}
