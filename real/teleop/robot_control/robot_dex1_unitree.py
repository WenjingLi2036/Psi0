import os
import sys
import threading
import time
from enum import IntEnum
from multiprocessing import Array, Event, Lock, Process, shared_memory
import numpy as np

from multiprocessing import Value, Lock, Array
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

kTopicGripperLeftCommand = "rt/dex1/left/cmd"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightCommand = "rt/dex1/right/cmd"
kTopicGripperRightState = "rt/dex1/right/state"


class Dex1_hand_Controller:
    def __init__(
            self,
        hand_shm_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        hand_target_array=None,
        fps=100.0,
        Unit_Test=False,
    ):
        print("[Dex1_1_Gripper_Controller] Initializing...")
        self.hand_target_array = hand_target_array
        self.fps = fps
        self.Unit_Test = Unit_Test

        # Other initializations
        self.hand_shm_array = hand_shm_array
        self.dual_hand_data_lock = dual_hand_data_lock
        self.dual_hand_state_array = dual_hand_state_array
        self.dual_hand_action_array = dual_hand_action_array

        # Initialize gripper state values
        self.left_gripper_state_value = Value('d', 0.0)
        self.right_gripper_state_value = Value('d', 0.0)

        # Initialize hand state arrays
        self.left_hand_state_array = Array('d', 7, lock=False)
        self.right_hand_state_array = Array('d', 7, lock=False)

        # Initialize stop event
        self.stop_event = Event()

        dq = 0.0
        tau = 0.0
        kp = 5.00
        kd = 0.05
        # initialize gripper cmd msg
        self.left_gripper_msg  = MotorCmds_()
        self.left_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self.right_gripper_msg = MotorCmds_()
        self.right_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]

        self.left_gripper_msg.cmds[0].dq  = dq
        self.left_gripper_msg.cmds[0].tau = tau
        self.left_gripper_msg.cmds[0].kp  = kp
        self.left_gripper_msg.cmds[0].kd  = kd

        self.right_gripper_msg.cmds[0].dq  = dq
        self.right_gripper_msg.cmds[0].tau = tau
        self.right_gripper_msg.cmds[0].kp  = kp
        self.right_gripper_msg.cmds[0].kd  = kd

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if self.left_gripper_state_value.value != 0.0 and self.right_gripper_state_value.value != 0.0:
                break
            time.sleep(0.01)
            print("[Dex1_1_Controller] Waiting to subscribe dds...")
        print("[Dex1_1_Gripper_Controller] Subscribe dds ok.")

        # Start control process
        self.hand_control_process = Process(
            target=self.control_process,
            args=(
                hand_shm_array,
                self.left_hand_state_array,
                self.right_hand_state_array,
                self.dual_hand_data_lock,
                self.dual_hand_state_array,
                self.dual_hand_action_array,
            ),
        )
        self.hand_control_process.daemon = True
        self.hand_control_process.start()

        print("Initialize Dex1_1_Controller OK!\n")

    
    def _subscribe_gripper_state(self):

        # initialize handcmd publisher and handstate subscriber
        self.LeftGripperCmb_publisher = ChannelPublisher(kTopicGripperLeftCommand, MotorCmds_)
        self.LeftGripperCmb_publisher.Init()
        self.RightGripperCmb_publisher = ChannelPublisher(kTopicGripperRightCommand, MotorCmds_)
        self.RightGripperCmb_publisher.Init()

        self.LeftGripperState_subscriber = ChannelSubscriber(kTopicGripperLeftState, MotorStates_)
        self.LeftGripperState_subscriber.Init()
        self.RightGripperState_subscriber = ChannelSubscriber(kTopicGripperRightState, MotorStates_)
        self.RightGripperState_subscriber.Init()

        while True:
            left_gripper_msg  = self.LeftGripperState_subscriber.Read()
            right_gripper_msg  = self.RightGripperState_subscriber.Read()
            self.gripper_sub_ready = True
            if left_gripper_msg is not None and right_gripper_msg is not None:
                self.left_gripper_state_value.value = left_gripper_msg.states[0].q
                self.right_gripper_state_value.value = right_gripper_msg.states[0].q

                # write the current gripper state to shared memory for other processes to read
                if self.dual_hand_state_array is not None:
                    with self.dual_hand_data_lock:
                        self.left_hand_state_array[6]  = self.left_gripper_state_value.value
                        self.right_hand_state_array[6] = self.right_gripper_state_value.value
            time.sleep(0.002)

    def ctrl_dual_gripper(self, left_q_target, right_q_target):
        """set current left, right gripper motor cmd target q"""
        self.left_gripper_msg.cmds[0].q  = left_q_target[6]
        self.right_gripper_msg.cmds[0].q = right_q_target[6]

        self.LeftGripperCmb_publisher.Write(self.left_gripper_msg)
        self.RightGripperCmb_publisher.Write(self.right_gripper_msg)
    
    def ctrl_dual_hand(self, left_hand_angles, right_hand_angles):
        print("[Dex1_1_Controller] skip here")
        

    def control_process(
        self,
        hand_shm_array,
        left_hand_state_array,
        right_hand_state_array,
        dual_hand_data_lock,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        while not self.stop_event.is_set():
            start_time = time.time()

            # Compute target qpos values using the transformation function.
            left_q_target = hand_shm_array[0:7]
            right_q_target = hand_shm_array[7:14]

            # Only update if valid targets were computed.
            if left_q_target is not None and right_q_target is not None:
                # Read the current state data from the left and right hand state arrays.
                state_data = np.concatenate(
                    (
                        np.array(left_hand_state_array[:]),
                        np.array(right_hand_state_array[:]),
                    )
                )
                # Concatenate the qpos targets for both hands.
                action_data = np.concatenate((left_q_target, right_q_target))

                if (
                    dual_hand_state_array is not None
                    and dual_hand_action_array is not None
                ):
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self.ctrl_dual_gripper(left_q_target, right_q_target)

            # Maintain the desired loop rate.
            time_elapsed = time.time() - start_time
            sleep_time = max(0, (1 / self.fps) - time_elapsed)
            time.sleep(sleep_time)

        print("Dex3_1_Controller has been closed.")
        
    def get_current_dual_hand_q(self):
        print("[Dex1_1_Controller] Return dummy values except for the 7th and 14th elements...")
        handstate = np.zeros(14, dtype=np.float64)
        handstate[6] = self.left_gripper_state_value.value
        handstate[13] = self.right_gripper_state_value.value
        return handstate
    
    def get_current_dual_hand_pressure(self):
        print("[Dex1_1_Controller] Return dummy values...")
        return np.zeros((18, 12), dtype=np.float64)

    def shutdown(self):
        print("[Dex1_1_Controller] Shutting down...")
        self.stop_event.set()
        if self.hand_control_process.is_alive():
            self.hand_control_process.join(timeout=2.0)

    def reset(self, max_wait_sec=5.0):
        print("[Dex1_1_Controller] Resetting...")
        pass