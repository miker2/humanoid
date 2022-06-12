// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
  // backspace: reset simulation
  if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
    mj_resetData(m, d);
    mj_forward(m, d);
  }
}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods) {
  // update button state
  button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
  button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
  button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  // compute mouse displacement, save
  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  // get current window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

  // determine action based on mouse button
  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  // move camera
  mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
  // emulate vertical mouse motion = 5% of window height
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}


// main function
int main(int argc, const char** argv) {
  // check command-line arguments
  if (argc!=2) {
    std::printf(" USAGE:  basic modelfile\n");
    return 0;
  }

  // load and compile model
  char error[1000] = "Could not load binary model";
  if (std::strlen(argv[1])>4 && !std::strcmp(argv[1]+std::strlen(argv[1])-4, ".mjb")) {
    m = mj_loadModel(argv[1], 0);
  } else {
    m = mj_loadXML(argv[1], 0, error, 1000);
  }
  if (!m) {
    mju_error_s("Load model error: %s", error);
  }

  // make data
  d = mj_makeData(m);

  const auto torso_id = (m != nullptr) ? mj_name2id(m, mjOBJ_BODY, "torso") : -1;
  const size_t base_dofs = (m != nullptr && torso_id != -1) ? *(m->body_dofnum + torso_id) : 0;
  const size_t base_q = (m != nullptr && *(m->jnt_type) == mjJNT_FREE) ? 7 : base_dofs;
  const size_t base_v = (m != nullptr && *(m->jnt_type) == mjJNT_FREE) ? 6 : base_dofs;

  std::map<int, size_t> act2jnt;
  std::map<size_t, int> jnt2act;
  std::map<size_t, size_t> jnt2qvec;
  std::map<size_t, size_t> jnt2vvec;
  if (m != nullptr) {
    std::cout << "nq: " << m->nq << ", nv: " << m->nv << ", nu: " << m->nu << ", njnt: " << m->njnt << std::endl;
    if (m->nv != m->nu) {
      std::cout << "joint mismatch!" << std::endl;
    }
    std::cout << "Torso id: " << mj_name2id(m, mjOBJ_BODY, "torso") << std::endl;
    std::cout << "Base q: " << base_q << ", base v: " << base_v << std::endl;
    for (size_t i = 0; i < m->nbody; ++i) {
      std::cout << "body " << i << " '" << mj_id2name(m, mjOBJ_BODY, i) << "' - " << *(m->body_dofnum + i)
                << " dofs, " << *(m->body_jntnum + i) << " joints, parent_id: " << *(m->body_parentid + i) << std::endl;
    }
    for (size_t i = 0; i < m->njnt; ++i) {
      const auto body_id = *(m->jnt_bodyid + i);
      jnt2qvec[i] = *(m->jnt_qposadr + i);
      jnt2vvec[i] = *(m->jnt_dofadr + i);
      std::cout << "joint " << i << " '" << mj_id2name(m, mjOBJ_JOINT, i) << "' - type: " << *(m->jnt_type + i)
                << ", body id: " << body_id << ", qposadr: " << jnt2qvec[i]
                << ", qdofadr: " << jnt2vvec[i] << std::endl;
    }
    for (size_t i = 0; i < m->nv; ++i) {
      const auto body_id = *(m->dof_bodyid + i);
      const auto jnt_id = *(m->dof_jntid + i);
      std::cout << "dof " << i << "' - body: " << body_id << " - " << mj_id2name(m, mjOBJ_BODY, body_id)
                << ", jnt id: " << jnt_id << " - " << mj_id2name(m, mjOBJ_JOINT, jnt_id)
                << ", parent id: " << *(m->dof_parentid) << std::endl;
    }
    for (size_t i = 0; i < m->nu; ++i) {
      std::cout << "act " << i << " '" << mj_id2name(m, mjOBJ_ACTUATOR, i) << "'"
                << " - trntype: " << *(m->actuator_trntype + i)
                << ", dyntype: " << *(m->actuator_dyntype + i)
                << ", trnid: " << *(m->actuator_trnid + (i*2))  << " | " << *(m->actuator_trnid + (i*2) + 1)<< std::endl;
      act2jnt[i] = *(m->actuator_trnid + (i*2));
      jnt2act[*(m->actuator_trnid + (i*2))] = i;
    }
  }
  
  // init GLFW
  if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
  }

  // create window, make OpenGL context current, request v-sync
  GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // initialize visualization data structures
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);

  // create scene and context
  mjv_makeScene(m, &scn, 2000);
  mjr_makeContext(m, &con, mjFONTSCALE_150);

  // install GLFW mouse and keyboard callbacks
  glfwSetKeyCallback(window, keyboard);
  glfwSetCursorPosCallback(window, mouse_move);
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetScrollCallback(window, scroll);

  if (d && m) {
    const auto j = mj_name2id(m, mjOBJ_JOINT, "left_hip_y");
    *(d->qpos + j) = -0.4;
  }

  // Create an array of desired joint angles and initilize them to the current joint angles.
  std::vector<double> q_d(m->nq - base_q, 0);
  if (d != nullptr && m != nullptr) {
    for (size_t j = 0; j < q_d.size(); ++j) {
      q_d[j] = *(d->qpos + j + base_q);
    }
  }
  // Set up a vector of target angles
  std::vector<double> q_tgt = {0.0, 0.0, 0, 0, 0, -0.707, 1.414, 0, -0.707, 0, 0, -0.707, 1.414, 0, -0.707};

  std::cout << "q_d.size()=" << q_d.size() << ", q_tgt.size()=" << q_tgt.size() << ", nu=" << m->nu << std::endl;

  // run main loop, target real-time simulation and 60 fps rendering
  while (!glfwWindowShouldClose(window)) {
    // advance interactive simulation for 1/60 sec
    //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
    //  this loop will finish on time for the next frame to be rendered at 60 fps.
    //  Otherwise add a cpu timer and exit this loop when it is time to render.
    size_t count{ 0 };
    mjtNum simstart = d->time;
    auto t_last = d->time;
    while (d->time - simstart < 1.0/60.0) {
      //mj_step(m, d);
      mj_step1(m, d);

      const auto dt = d->time - t_last;
      t_last = d->time;

      if (count % 100 == 0) {
        std::cout << "time: " << d->time << ", dt: " << dt << ", timestep: " << m->opt.timestep << std::endl;
      }
      // Run controller here!
      if (m->nu == m->nv - base_v && m->nu == m->nq - base_q) {
        for (size_t a = 0; a < m->nu; ++a) {
          const auto j = act2jnt[a];
          const auto qi = jnt2qvec[j];
          const auto vi = jnt2vvec[j];
          double kp = 1.7;
          double kd = 0.15;
          /*
          if (a == 7 || a == 8 || a == 13 || a == 14) {
            kp *= 0.1;
            kd *= 0.1;
          }
          */

          double qd_d = 0;
          // Set the desired position & velocity based on the target angle:
          if (dt > 0) {
            constexpr double max_vel{ 0.1 };  // rad/s
            qd_d = std::clamp((q_tgt[a] - q_d[a]) / m->opt.timestep, -max_vel, max_vel);
            q_d[a] += qd_d * m->opt.timestep;
          }

          *(d->ctrl + a) = kp * (q_d[a] - *(d->qpos + qi)) + kd * (qd_d - *(d->qvel + vi));
          if (count % 100 == 0) {
            std::cout << "joint: " << mj_id2name(m, mjOBJ_JOINT, j)
                      << " # " << j << " - pos: " << *(d->qpos + qi)
                      << ", q_d: " << q_d[a] << ", q_tgt: " << q_tgt[a]
                      << ", vel: " << *(d->qvel + vi)
                      << ", qd_d: " << qd_d
                      << ", ctrl: " << *(d->ctrl + a) << std::endl;
          }
        }
      }
      
      mj_step2(m, d);

      ++count;
    }

    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // update scene and render
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
  }

  //free visualization storage
  mjv_freeScene(&scn);
  mjr_freeContext(&con);

  // free MuJoCo model and data
  mj_deleteData(d);
  mj_deleteModel(m);

  // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  return 1;
}
