#include "Controller.hpp"
#include "../Ai/Model.hpp"

namespace MedNES {

u8 Controller::read(u16 address) {
  if (address == 0x4016) {
    if (strobe) {
      return 0x40 | (btnState & 1);
    }

    JOY1 = 0x80 | (btnStateLocked & 1);
    btnStateLocked >>= 1;
    return JOY1;
  } else {
    // TODO: Implement JOY2
    return JOY2;
  }
}

void Controller::write(u16 address, u8 data) {
  if (address == 0x4016) {
    if (strobe && !(data & 0x1)) {
      btnStateLocked = btnState;
    }

    strobe = data & 0x1;
  } else {
    // TODO: Implement JOY2
  }
}

void Controller::setButtonPressed(ButtonPresses keys) {
  btnState = 0;

  if (keys.enter) {
    btnState = btnState | (1 << 3);
  }
  if (keys.a) {
    btnState = btnState | (1 << 0);
  }

  if (keys.b) {
    btnState = btnState | (1 << 1);
  }

  if (keys.up) {
    btnState = btnState | (1 << 4);
  }

  if (keys.down) {
    btnState = btnState | (1 << 5);
  }

  if (keys.left) {
    btnState = btnState | (1 << 6);
  }

  if (keys.right) {
    btnState = btnState | (1 << 7);
  }
}
} // namespace MedNES
