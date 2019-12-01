#!/usr/bin/env python

import sys
import rospy
from walker_planner.srv import Prediction, PredictionRequest, PredictionResponse

def add_two_ints_client(x, y):
    rospy.wait_for_service('walker_planner/prediction')
    try:
        add_two_ints = rospy.ServiceProxy('walker_planner/prediction', AddTwoInts)
        resp1 = add_two_ints(PredictionRequest(x, y))
        return resp1.prediction
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

def usage():
    return "%s [x y]"%sys.argv[0]

if __name__ == "__main__":
    if len(sys.argv) == 3:
        x = float(sys.argv[1])
        y = float(sys.argv[2])
    else:
        print usage()
        sys.exit(1)
    print "Requesting %s+%s"%(x, y)
    print "prediction at %s %s = %s"%(x, y, add_two_ints_client(x, y))