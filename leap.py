
try:
    import os, sys, inspect, thread, time
    sys.path.insert(0, "C:/Users/cmg22/Documents/ART_ML/full_demo/LeapSDK/lib/x64")
    import Leap
    print("Leap Found")
    LEAP = True
except:
    print("No Leap")
    LEAP = False


if LEAP:
    time.sleep(0.5)
    leapController = Leap.Controller()
    time.sleep(0.5)
    LEAP = leapController.is_connected
    if not LEAP:
        print("no leap controller connected")


###########
## Video
###########

def getVideoIndicesFromLeap(controller):
    frame = controller.frame()
    for hand in frame.hands:
        if hand.is_right:
            pos = hand.palm_position
            return abs(int(pos.x)), int(pos.y)
    return None


###########
## Music
###########

def bound_and_scale(v): 
    num_sigmas = 5.0
    return (min(200.0, max(0.0, v)) / 200.0 - 0.5) * num_sigmas * 2.0

def transform_coords(position):
    x = -bound_and_scale(position.x + 100.0)
    y = -bound_and_scale(position.y)
    z = bound_and_scale(position.z + 100.0)
    return x,y,z

def getMusicPCAFromLeap(controller):
    num_params = 15
    params = [0 for _ in range(num_params)]
    needs_update = False
    frame = controller.frame()
    for hand in frame.hands:
        if hand.is_left:
            # slider updates
            ix = 0
            for finger in hand.fingers:
                x,y,z = transform_coords(finger.tip_position)
                params[ix] = round(y/2, 0)
                params[ix+1] = round(x/2, 0)
                params[ix+2] = round(z/2, 0)
                ix += 3
                needs_update = True
    return needs_update, params