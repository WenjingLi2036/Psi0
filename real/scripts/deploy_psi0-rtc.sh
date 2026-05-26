#!/bin/bash

PORT=8014
# TASK="Spray the bowl and wipe it and stack it up."
# TASK="Pick bottle and turn and pour into cup."
# TASK="Pick toys into box and lift and turn and put on the chair new"
# TASK="g1/When the seated human reaches out their hand, extend your arm to meet theirs. Upon physical contact, provide stable, upward physical support to assist them in transitioning from a sitting to a standing position"
# TASK="Move the white box from the yellow rectangular area and place it accurately inside the red rectangular area"
# TASK="When the human approaches and extends their hand, reach forward to meet it and perform a handshake."
TASK="When the person walks toward you, pick up the tape on the table and then hand over the tape toward the person."

cd "$(dirname "$0")/../teleop"

python ../deploy/psi-inference_rtc.py \
    --port "$PORT" \
    --task "$TASK" \
    --keep_standing false
