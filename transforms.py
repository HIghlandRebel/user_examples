import numpy as np
# from np import linalg
import math
import cmath
import copy
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d


v_init=[-0.0012, 3.1162, 0.03889]
v=[-0.06, 0.13, -0.04]

def length(v):
    return math.sqrt(pow(v[0],2)+pow(v[1],2)+pow(v[2],2))

def norm(v):
    l=length(v)
    norm=[v[0]/l, v[1]/l, v[2]/l]
    return norm

def _polyscope(x,y,z):

    if ( (abs(x) >= 0.001 and x < 0.0) or (abs(x) < 0.001 and abs(y) >= 0.001 and y < 0.0) or (abs(x) < 0.001 and abs(y) < 0.001 and z < 0.0) ):
        scale = 1 - ((2*3.14) / length([x,y,z]))
        ret = [x/scale, y/scale, z/scale]
        print ("PolyScope SCALED value: ", ret)
        return ret
    else:
        ret = [x,y,z]
        print ("PolyScope value: ", ret)
        return ret

def polyscope(v):

    return _polyscope(v[0], v[1], v[2])


####
####
# rotate the orientation with respect to the base frame
##
#### input arguments
# pose: the TCP pose
# angle: the rotational angle
# axis: the axis in the base frame
#		1: X axis
#		2: Y axis
#		3: Z axis (by default)
##
#### return values
# target: the target TCP pose
##
#### mathematic equation
# target = pose * rotation
##
####
def rotate_base(pose, angle, axis=3):

	rotvec = [ pose[3], pose[4], pose[5] ]
	if (axis == 1):
		axis_tool = frame_base2tool(rotvec, [1,0,0])
	elif (axis == 2): 
		axis_tool = frame_base2tool(rotvec, [0,1,0])
	else:
		axis_tool = frame_base2tool(rotvec, [0,0,1])
	# end
		
	pose_rot = [ 0, 0, 0, angle*axis_tool[0], angle*axis_tool[1], angle*axis_tool[2] ]
		
	target = PoseTrans(pose, pose_rot)

	return target
# end
####

####
####
# represent X, Y, Z tool coordinates in the tool frame
##
#### input arguments
# p_base: X, Y, and Z coordinates in the base frame
# rotvec: the tool orientation
##
#### return values
# p_tool: X, Y, and Z coordinates in the tool frame
##
#### mathematic equation
# p_base = rotmat * p_tool 
# => p_tool = inv(rotmat) * p_base
##
####
def frame_base2tool(rotvec, p_base):

	pose_rotvec = [ 0, 0, 0, rotvec[0], rotvec[1], rotvec[2] ]
	pose_p_base = [ p_base[0], p_base[1], p_base[2], 0, 0, 0 ]
		
	p_tool = PoseTrans(PoseInv(pose_rotvec), pose_p_base)

	return p_tool
####

#================================================
# find rotation vector required to reach the desired picking direction while minimizing the twisting of the gripper
#================================================

def find_rot_vect(proposedPose, desiredPose):
    #==================================================================
    # find vector representing ROBOT_POSE_ABOVE_BIN pose direction
    #==================================================================
    Rx_rad_default = desiredPose[3] # constants.aboveBin[3]
    Ry_rad_default = desiredPose[4] #constants.aboveBin[4]
    Rz_rad_default = desiredPose[5] #constants.aboveBin[5]

    v_default = np.array([0.0,0.0,1.0])

    # construct rotation from rotation vector
    r_default = Rot.from_rotvec(np.array([Rx_rad_default, Ry_rad_default, Rz_rad_default]))

    # convert to rotation matrix
    rotMat_default = r_default.as_matrix()

    # rotate the vecvor
    v_default = np.dot(rotMat_default, v_default)

    #================================================================================================================
    # stage 1 rotation matrix - the rotation matrix from the gripper base frame to the ROBOT_POSE_ABOVE_BIN pose
    #================================================================================================================
    r_1 = r_default.as_matrix()

    #=======================================================
    # find vector representing desired picking direction
    #=======================================================
    v_desired = np.array([0.0,0.0,1.0])

    # form robot pose
    Xcoord = proposedPose[0]
    Ycoord = proposedPose[1]
    Zcoord = proposedPose[2]
    # angle = proposedPose[3]
    angle = math.sqrt(proposedPose[3]*proposedPose[3] + proposedPose[4]*proposedPose[4]  + proposedPose[5]*proposedPose[5] )
    axis_zero = proposedPose[3]/angle
    axis_one = proposedPose[4]/angle
    axis_two = proposedPose[5]/angle

    rotMat_desired = axis_to_rotMat(angle, axis_zero, axis_one, axis_two)
    # rotate the vector to align with the desired picking direction
    v_desired = np.dot(rotMat_desired, v_desired)

    #==============================================================================================
    # find angle between the two vectors (above bin pose and desired picking direction)
    #==============================================================================================
    dot_product = np.dot(v_default, v_desired)
    angle_rad = np.arccos(dot_product)

    #==============================================================================================
    # find rotation axis that rotates the above bin pose vector to the desired picking direction
    #==============================================================================================
    # cross product
    axis_rot = np.array([v_default[1]*v_desired[2] - v_default[2]*v_desired[1],
                        v_default[2]*v_desired[0] - v_default[0]*v_desired[2],
                        v_default[0]*v_desired[1] - v_default[1]*v_desired[0]])
    # normalize
    axis_rot = axis_rot / np.sqrt(np.sum(axis_rot**2))

    #=============================================================================================================================================
    # form stage 2 rotation vector - the rotation vector from the ROBOT_POSE_ABOVE_BIN pose to the desired pick direction
    #=============================================================================================================================================
    r_vect = axis_rot * angle_rad

    #================================================
    # stage 2 rotation matrix
    #================================================
    # construct rotation matrix from rotation vector
    r_2 = Rot.from_rotvec(r_vect)
    r_2 = r_2.as_matrix()

    #================================================
    # total rotation matrix and rotation vector
    #================================================
    # r_total = r_2 * r_1
    r_total = np.dot(r_2, r_1)

    r_total_mat = Rot.from_matrix(r_total)
    
    r_total_vect = r_total_mat.as_rotvec()
    proposedPose = [Xcoord, Ycoord, Zcoord, r_total_vect[0],  r_total_vect[1],  r_total_vect[2]]
    return proposedPose

def rotateAlign(currentPose, desiredPose):
    cp = [currentPose[3], currentPose[4], currentPose[5]]
    dp = [desiredPose[3], desiredPose[4], desiredPose[5]]
    cp_unit = cp / np.linalg.norm(cp)
    dp_unit = dp / np.linalg.norm(dp)
    axis = np.cross(cp_unit, dp_unit)
    # axis = axis / np.linalg.norm(axis)
    # print("\nAxis:", axis)
    cosA = np.dot(cp_unit, dp_unit)
    # print("\ncosA:", cosA)
    
    k = 1.0/(1.0 + cosA)
    # print("\nk:", k)
    rotMat = [[(axis[0] * axis[0] * k) + cosA, (axis[1] * axis[0] * k) - axis[2], (axis[2] * axis[0] * k) + axis[1]],
                 [(axis[0] * axis[1] * k) + axis[2],  (axis[1] * axis[1] * k) + cosA, (axis[2] * axis[1] * k) - axis[0]],
                 [(axis[0] * axis[2] * k) - axis[1], (axis[1] * axis[2] * k) + axis[0], (axis[2] * axis[2] * k) + cosA]]
    # print("\nRotation Matrix:", rotMat)
    rotVec = np.matmul(np.array(rotMat),np.array(cp))
    result = [desiredPose[0], desiredPose[1], desiredPose[2], rotVec[0], rotVec[1], rotVec[2]]
    return result

#================================================
#  Calculate a new robot pose based on the desired change in pose
# NOTE: desiredPoseRPY is the desired offset from the currentPose
#================================================
def changePose(currentPose, desiredPoseRPY):
    currentP = copy.copy(currentPose)
    desiredRPY = copy.copy(desiredPoseRPY)
    if desiredRPY[0] == 180:
        desiredRPY[0] = 179.8
    elif desiredRPY[0] == -180:
        desiredRPY[0] = -179.8   
    # print("\nCurrent Pose:", currentPose)
    a_roll, a_pitch, a_yaw =  rotvec2rpy(currentP[3], currentP[4], currentP[5])

    # change yaw so that angle is between 0 and 360 degrees
    if a_yaw < 0:
        a_yaw = a_yaw + 360

    if desiredRPY[2] <0:
        desiredRPY[2] = desiredRPY[2] + 360

    if a_roll < 0:
        diffRx = abs(desiredRPY[0] + a_roll)
    else:
        diffRx = abs(desiredRPY[0] - a_roll)

    if a_pitch < 0:
        diffRy = abs(desiredRPY[1] - a_pitch)
    else:
        diffRy = abs(desiredRPY[1] - a_pitch)

    diffRz = abs(desiredRPY[2] - a_yaw)

    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("\nDesired relative change in position:")
    print("\tDiff Rx: ", diffRx)
    print("\tDiff Ry: ", diffRy)
    print("\tDiff Rz: ", diffRz)
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # diffRx = abs(desiredRPY[0] - a_roll)
    if a_roll > 0 :
        newRx = (round(diffRx,2) - .05)
    else:
        newRx = - (round(diffRx,2) - 0.05)

    if a_pitch > 0:
        newRy =  - (round(diffRy,2) - 0.05)
    else:
        newRy =  (round(diffRy,2) - 0.05)

    newRz = -diffRz
    # if newRz < 0:
    #     newRz = newRz + 360
    # elif diffRz > 360:
    #     newRz = newRz - 360
    print("\nIn changePose():")
    print("\nDiff Rx:", newRx)
    print("\nDiff Ry:", newRy)
    print("\nDiff Rz:", newRz)

    desiredPose = rpy2rotvec(newRx, newRy, newRz)


    newPose = []
    newPose.append(0.0)
    newPose.append(0.0)
    newPose.append(0.0)
    newPose.append(desiredPose[0])
    newPose.append(desiredPose[1])
    newPose.append(desiredPose[2])

    # create the rotation matrix for the object
    # newPose2 = PoseTrans(currentP, newPose)
    newPose2 = PoseAdd(currentP, newPose)
    # b_roll, b_pitch, b_yaw =  rotvec2rpy(newPose2[3], newPose2[4], newPose2[5])
    # print("\nAfter transforming pose:")
    # print("\nDiff Rx:", diffRx)
    # print("\nDiff Ry:", diffRy)
    # print("\nDiff Rz:", diffRz)
    # print("Calculated new Pose:" , newPose2)
    return newPose2

#================================================
#  Calculate a new robot pose based on the desired change in pose
#================================================
def addPose(currentPose, desiredPoseRPY):
    currentP = copy.copy(currentPose)
    desiredRPY = copy.copy(desiredPoseRPY)

    # print("\nCurrent Pose:", currentPose)
    a_roll, a_pitch, a_yaw =  rotvec2rpy(currentP[3], currentP[4], currentP[5])
    # print("\nCurrent RPY:")
    # print("Roll:", a_roll)
    # print("Pitch:", a_pitch)
    # print("Yaw:", a_yaw)


    # print("\npickPose:", desiredRPY)
    b_roll, b_pitch, b_yaw =  desiredRPY[0], desiredRPY[1], desiredRPY[2] #180, 0, 0 #transform.rotvec2rpy(180, 0, 0)
    # print("\ndesiredRPY:")
    # print("Roll:", desiredRPY[0])
    # print("Pitch:", desiredRPY[1])
    # print("Yaw:", desiredRPY[2])


    # print("\nDiff Rx:", diffRx)
    # print("\nDiff Ry:", diffRy)
    # print("\nDiff Rz:", diffRz)

    desiredPose = rpy2rotvec(b_roll, b_pitch, b_yaw)


    newPose = []
    newPose.append(0.0)
    newPose.append(0.0)
    newPose.append(0.0)
    newPose.append(desiredPose[0])
    newPose.append(desiredPose[1])
    newPose.append(desiredPose[2])

    

    # create the rotation matrix for the object
    newPose2 = PoseAdd(currentP,newPose)
    c_roll, c_pitch, c_yaw =  rotvec2rpy(newPose2[3], newPose2[4], newPose2[5]) #180, 0, 0 #transform.rotvec2rpy(180, 0, 0)
    # print("\nCalculated new pose:")
    # print("Roll:", c_roll)
    # print("Pitch:", c_pitch)
    # print("Yaw:", c_yaw)
    return newPose2
#================================================
#  Convert Euler angles to rotation vector
#================================================
def EulerToVector(roll, pitch, yaw):
    alpha, beta, gamma = yaw, pitch, roll
    ca, cb, cg, sa, sb, sg = math.cos(alpha), math.cos(beta), math.cos(gamma), math.sin(alpha), math.sin(beta), math.sin(gamma)
    r11, r12, r13 = ca*cb, ca*sb*sg-sa*cg, ca*sb*cg+sa*sg
    r21, r22, r23 = sa*cb, sa*sb*sg+ca*cg, sa*sb*cg-ca*sg
    r31, r32, r33 = -sb, cb*sg, cb*cg
    val = (r11+r22+r33-1)/2
    if val > 1:
        val = 1
    elif val <-1:
        val = -1
    theta = math.acos(val)
    sth = math.sin(theta)
    kx, ky, kz = (r32-r23)/(2*sth), (r13-r31)/(2*sth), (r21-r12)/(2*sth)
    return [(theta*kx),(theta*ky),(theta*kz)]

#================================================
#  Convert rotation vector to Euler angles
#================================================
def VectorToEuler(rx,ry,rz):
    theta = math.sqrt(rx*rx + ry*ry + rz*rz)
    kx, ky, kz = rx/theta, ry/theta, rz/theta
    cth, sth, vth = math.cos(theta), math.sin(theta), 1-math.cos(theta)
    r11, r12, r13 = kx*kx*vth + cth, kx*ky*vth - kz*sth, kx*kz*vth + ky*sth
    r21, r22, r23 = kx*ky*vth + kz*sth, ky*ky*vth + cth, ky*kz*vth - kx*sth
    r31, r32, r33 = kx*kz*vth - ky*sth, ky*kz*vth + kx*sth, kz*kz*vth + cth
    beta = math.atan2(-r31,math.sqrt(r11*r11+r21*r21))
    if beta > math.radians(89.99):
        beta = math.radians(89.99)
        alpha = 0
        gamma = math.atan2(r12,r22)
    elif beta < -math.radians(89.99):
        beta = -math.radians(89.99)
        alpha = 0
        gamma = -math.atan2(r12,r22)
    else:
        cb = math.cos(beta)
        alpha = math.atan2(r21/cb,r11/cb)
        gamma = math.atan2(r32/cb,r33/cb)
    return [gamma,beta,alpha]

#================================================
#  Get rotation matrix from ta pose
#================================================
def GetRotationMatrix(Pose):
    X, Y, Z, Rx, Ry, Rz = Pose[0],Pose[1],Pose[2],Pose[3],Pose[4],Pose[5]
    # Rr = VectorToEuler(Rx, Ry, Rz)
    Rr = rotvec2rpy(Rx, Ry, Rz)
    Rx, Ry, Rz = Rr[0], Rr[1], Rr[2]
    M11 = math.cos(Ry)*math.cos(Rz)
    M12 = (math.sin(Rx)*math.sin(Ry)*math.cos(Rz))-(math.cos(Rx)*math.sin(Rz))
    M13 = (math.cos(Rx)*math.sin(Ry)*math.cos(Rz))+(math.sin(Rx)*math.sin(Rz))
    M21 = math.cos(Ry)*math.sin(Rz)
    M22 = (math.sin(Rx)*math.sin(Ry)*math.sin(Rz))+(math.cos(Rx)*math.cos(Rz))
    M23 = (math.cos(Rx)*math.sin(Ry)*math.sin(Rz))-(math.sin(Rx)*math.cos(Rz))
    M31 = -math.sin(Ry)
    M32 = math.sin(Rx)*math.cos(Ry)
    M33 = math.cos(Rx)*math.cos(Ry)
    return np.stack([[M11,M12,M13],[M21,M22,M23],[M31,M32,M33]])


#================================================
#  Get rotation vector from rotation matrix
#================================================
def GetRotation(RM):
    Ry = math.atan2(-RM[2][0],math.sqrt(RM[0][0]**2+RM[1][0]**2))
    Rz = math.atan2(RM[1][0]/math.cos(Ry),RM[0][0]/math.cos(Ry))
    Rx = math.atan2(RM[2][1]/math.cos(Ry),RM[2][2]/math.cos(Ry))
    Rr = rpy2rotvec(Rx, Ry, Rz)
    # Rr = EulerToVector(Rx, Ry, Rz)
    Rx, Ry, Rz = Rr[0], Rr[1], Rr[2]
    return [Rx, Ry, Rz]


#================================================
#  Return X, Y & Z in a column vector
#================================================
def GetPointMatrix(Pose):
    X, Y, Z, Rx, Ry, Rz = Pose[0],Pose[1],Pose[2],Pose[3],Pose[4],Pose[5]
    return np.stack([X,Y,Z])


#================================================
# Pose translation function. Similar to UR Script pose_trans(). Can be used to:
# 1) translate and rotate Pose2 by the parameters of Pose1
# 2) Get the resulting pose when first making a move of Pose1 and then, from that point in space, a relative move of Pose2
# NOTE: the resulting pose is **relative to Pose1**
#================================================
# def PoseTrans(Pose1, Pose2):
#     P1 = GetPointMatrix(Pose1)
#     P2 = GetPointMatrix(Pose2)
#     R1 = GetRotationMatrix(Pose1)
#     R2 = GetRotationMatrix(Pose2)
#     P = np.add(P1, np.matmul(R1, P2))
#     R = GetRotation(np.matmul(R1, R2))
#     return [P[0],P[1],P[2],R[0],R[1],R[2]]

# #================================================
# #  Similar to UR Script pose_add(). Add Pose2 to Pose1 NOTE: Pose2 is added to Pose1 **relative to robot base**
# #================================================
# def PoseAdd(Pose1, Pose2):
#     P1 = GetPointMatrix(Pose1)
#     P2 = GetPointMatrix(Pose2)
#     R1 = GetRotationMatrix(Pose1)
#     R2 = GetRotationMatrix(Pose2)
#     P = np.add(P1, P2)
#     R = GetRotation(np.matmul(R1, R2))
#     return [P[0],P[1],P[2],R[0],R[1],R[2]]

# #================================================
# #  Similar to UR Script pose_add(). Add Pose2 to Pose1 NOTE: Pose2 is added to Pose1 **relative to robot base**
# #================================================
# def PoseInv(Pose1):
#     P1 = GetPointMatrix(Pose1)
#     R1 = GetRotationMatrix(Pose1)
#     R2 = rotVec_to_rotMat_affine(Pose1[0], Pose1[1], Pose1[2], Pose1[3], Pose1[4], Pose1[5])
    
#     print(R1)
#     print(R2)
#     invR1 = np.linalg.inv(R1)
#     # print("")
#     # print(invR1)
#     P = np.matmul(invR1, P1)
#     R = GetRotation(invR1)
#     return [-P[0],-P[1],-P[2],R[0],R[1],R[2]]

def PoseTrans(Pose1, Pose2):
    # pose1 = [Pose1[3], Pose1[4], Pose1[5]]
    # pose2 = [Pose2[3], Pose2[4], Pose2[5]]
    P1 = GetPointMatrix(Pose1)
    P2 = GetPointMatrix(Pose2)
    # R1 = GetRotationMatrix(Pose1)
    # R2 = GetRotationMatrix(Pose2)
    R1 = rotVec_to_rotMat(Pose1)
    # print("R1:")
    # print(R1)
    R2 = rotVec_to_rotMat(Pose2)
    # print("R1:")
    # print(R2)
    P = np.add(P1, np.matmul(R1, P2))
    R = rotmat2rotvec(np.matmul(R1, R2))
    # R = GetRotation(np.matmul(R1, R2))
    return [P[0],P[1],P[2],-R[0],-R[1],-R[2]]

#================================================
#  Similar to UR Script pose_add(). Add Pose2 to Pose1 NOTE: Pose2 is added to Pose1 **relative to robot base**
#================================================
def PoseAdd(Pose1, Pose2):
    # pose1 = [Pose1[3], Pose1[4], Pose1[5]]
    # pose2 = [Pose2[3], Pose2[4], Pose2[5]]
    P1 = GetPointMatrix(Pose1)
    P2 = GetPointMatrix(Pose2)
    # R1 = GetRotationMatrix(Pose1)
    # R2 = GetRotationMatrix(Pose2)
    R1 = rotVec_to_rotMat(Pose1)
    R2 = rotVec_to_rotMat(Pose2)
    P = np.add(P1, P2)
    R = rotmat2rotvec(np.matmul(R1, R2))
    # R = GetRotation(np.matmul(R1, R2))
    return [P[0],P[1],P[2],R[0],R[1],R[2]]

#================================================
#  Similar to UR Script pose_add(). Add Pose2 to Pose1 NOTE: Pose2 is added to Pose1 **relative to robot base**
#================================================
def PoseInv(Pose1):
    # pose = [Pose1[3], Pose1[4], Pose1[5]]
    P1 = GetPointMatrix(Pose1)
    # R1 = rotvec2rotmat(Pose1)
    
    # print("")
    # print(R1)
    R1 = rotVec_to_rotMat(Pose1)
    # print("")
    # print(R1)
    # R1 = GetRotationMatrix(Pose1)
    # print("")
    # print(R1)
    # R1 = rotVec_to_rotMat_affine(Pose1[0], Pose1[1], Pose1[2], Pose1[3], Pose1[4], Pose1[5])
    # print(R1)
    invR1 = np.linalg.inv(R1)
    # print("")
    # print(invR1)
    P = np.matmul(invR1, P1)
    # theta = calc_theta(invR1)
    # axis_zero, axis_one, axis_two = calculate_axis(invR1, theta)
    R = rotmat2rotvec(invR1)
    # R = rotMat_to_rotVec(invR1)
    # R = GetRotation(invR1)
    # return [-P[0],-P[1],-P[2], axis_zero, axis_one, axis_two]#R[0],R[1],R[2]]
    return [-P[0],-P[1],-P[2], -R[0],-R[1],-R[2]]

#================================================
# convert degrees to Radian
#================================================
def d2r(theta):
    return theta*math.pi/180

#================================================
# convert Radians to degrees
#================================================
def r2d(theta):
    return theta*180/math.pi

#================================================
# convert RPY to rotation vector for UR
#================================================
def rpy2rotvec(roll,pitch,yaw):
  
    alpha = d2r(yaw) #*math.pi/180
    beta = d2r(pitch) #*math.pi/180
    gamma = d2r(roll) #*math.pi/180
    
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    sa = math.sin(alpha)
    sb = math.sin(beta)
    sg = math.sin(gamma)
    
    r11 = ca*cb
    r12 = ca*sb*sg-sa*cg
    r13 = ca*sb*cg+sa*sg
    r21 = sa*cb
    r22 = sa*sb*sg+ca*cg
    r23 = sa*sb*cg-ca*sg
    r31 = -sb
    r32 = cb*sg
    r33 = cb*cg
    
    val = (r11+r22+r33-1)/2
    if val > 1:
        val = 1
    elif val <-1:
        val = -1
    theta = math.acos(val)
    if theta == 0:
        kx = 0
        ky = 0
        kz = 0
    else:
        sth = math.sin(theta)
        kx = (r32-r23)/(2*sth)
        ky = (r13-r31)/(2*sth)
        kz = (r21-r12)/(2*sth)
    
    return [(theta*kx),(theta*ky),(theta*kz)]

#================================================
# convert rotation vector to RPY for UR
#================================================
def rotvec2rpy(rx,ry,rz):
    # if rx == 0 and ry == 0 and rz == 0:
    #     rx = 0.000000000001
    #     ry = 0.000000000001
    #     rz = 0.000000000001
    theta = math.sqrt(rx*rx + ry*ry + rz*rz)
    if theta == 0:
        kx = 0
        ky = 0
        kz = 0
    else:
        kx = rx/theta
        ky = ry/theta
        kz = rz/theta
    cth = math.cos(theta)
    sth = math.sin(theta)
    vth = 1-math.cos(theta)
    
    r11 = kx*kx*vth + cth
    r12 = kx*ky*vth - kz*sth
    r13 = kx*kz*vth + ky*sth
    r21 = kx*ky*vth + kz*sth
    r22 = ky*ky*vth + cth
    r23 = ky*kz*vth - kx*sth
    r31 = kx*kz*vth - ky*sth
    r32 = ky*kz*vth + kx*sth
    r33 = kz*kz*vth + cth
    
    beta = math.atan2(-r31,math.sqrt(r11*r11+r21*r21))
    
    if beta > d2r(89.99): #*math.pi/180:
        beta = d2r(89.99) #*math.pi/180
        alpha = 0
        gamma = math.atan2(r12,r22)
    elif beta < -d2r(89.99): #*math.pi/180:
        beta = -d2r(89.99) #*math.pi/180
        alpha = 0
        gamma = -math.atan2(r12,r22)
    else:
        cb = math.cos(beta)
        alpha = math.atan2(r21/cb,r11/cb)
        gamma = math.atan2(r32/cb,r33/cb)
    
    return [r2d(gamma), r2d(beta) , r2d(alpha)]

#================================================
# convert axis-angles to rotation Matrix
#================================================
def axis_to_rotMat(angle, axis_zero, axis_one, axis_two):
    x = copy.copy(axis_zero)
    y = copy.copy(axis_one)
    z = copy.copy(axis_two)
    s = math.sin(angle)
    c = math.cos(angle)
    t = 1.0-c
    magnitude = math.sqrt(x*x + y*y + z*z)
    if magnitude == 0:
        # print("!Error! Magnitude = 0")
        magnitude = 0.0000000001
    else:
        x /= magnitude
        y /= magnitude
        z /= magnitude
    # calulate rotation matrix elements
    m00 = c + x*x*t
    m11 = c + y*y*t
    m22 = c + z*z*t
    tmp1 = x*y*t
    tmp2 = z*s
    m10 = tmp1 + tmp2
    m01 = tmp1 - tmp2
    tmp1 = x*z*t
    tmp2 = y*s
    m20 = tmp1 - tmp2
    m02 = tmp1 + tmp2    
    tmp1 = y*z*t
    tmp2 = x*s
    m21 = tmp1 + tmp2
    m12 = tmp1 - tmp2
    matrix = [ [m00, m01, m02], [m10, m11, m12], [m20, m21, m22] ]
    matrix = np.array(matrix)
    return matrix   

#================================================
# convert rotVect to affine rotation Matrix
#================================================
def rotVec_to_rotMat_affine(pose):
    xCoord, yCoord, zCoord, Rx, Ry, Rz = pose
    angle = math.sqrt(Rx*Rx + Ry*Ry + Rz*Rz)
    if angle == 0:
        angle = 0.0000000001
    axis_zero = Rx/angle
    axis_one = Ry/angle
    axis_two = Rz/angle
    matrix = axis_to_rotMat_affine(xCoord, yCoord, zCoord, angle, axis_zero, axis_one, axis_two)
    return matrix

#================================================
# convert rotVect to affine rotation Matrix
#================================================
def rotVec_to_rotMat(pose):
    xCoord, yCoord, zCoord, Rx, Ry, Rz = pose
    angle = math.sqrt(Rx*Rx + Ry*Ry + Rz*Rz)
    if angle == 0:
        axis_zero = 0
        axis_one = 0
        axis_two = 0
    else:
        axis_zero = Rx/angle
        axis_one = Ry/angle
        axis_two = Rz/angle
    matrix = axis_to_rotMat(angle, axis_zero, axis_one, axis_two)
    return matrix

#================================================
# convert axis-angles to affine rotation Matrix
#================================================
def axis_to_rotMat_affine(xCoord, yCoord, zCoord, angle, axis_zero, axis_one, axis_two):
    x = copy.copy(axis_zero)
    y = copy.copy(axis_one)
    z = copy.copy(axis_two)
    s = math.sin(angle)
    c = math.cos(angle)
    t = 1.0-c
    magnitude = math.sqrt(x*x + y*y + z*z)
    if magnitude == 0:
        # print("!Error! Magnitude = 0")
        magnitude = 0.0000000001
    else:
        x /= magnitude
        y /= magnitude
        z /= magnitude
    # calulate rotation matrix elements
    m00 = c + x*x*t
    m11 = c + y*y*t
    m22 = c + z*z*t
    tmp1 = x*y*t
    tmp2 = z*s
    m10 = tmp1 + tmp2
    m01 = tmp1 - tmp2
    tmp1 = x*z*t
    tmp2 = y*s
    m20 = tmp1 - tmp2
    m02 = tmp1 + tmp2    
    tmp1 = y*z*t
    tmp2 = x*s
    m21 = tmp1 + tmp2
    m12 = tmp1 - tmp2
    matrix = [ [m00, m01, m02, xCoord], [m10, m11, m12, yCoord], [m20, m21, m22, zCoord],[0.0, 0.0, 0.0, 1.0] ]
    # matrix = np.array(matrix)
    return matrix

def quaternion_rotation_matrix(xCoord, yCoord, zCoord, w, x, y, z):

    xx = x*x
    xy = x*y
    xz = x*z
    xw = x*w

    yy = y*y
    yz = y*z
    yw = y*w

    zz = z*z
    zw = z*w

    # First row of the rotation matrix
    r00  = 1 - 2 * ( yy + zz )
    r01  =     2 * ( xy - zw )
    r02 =     2 * ( xz + yw )

    # Second row of the rotation matrix
    r10  =     2 * ( xy + zw )
    r11  = 1 - 2 * ( xx + zz )
    r12  =     2 * ( yz - xw )

    # Third row of the rotation matrix
    r20  =     2 * ( xz - yw )
    r21  =     2 * ( yz + xw )
    r22 = 1 - 2 * ( xx + yy )

    
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02, xCoord],
                           [r10, r11, r12, yCoord],
                           [r20, r21, r22, zCoord],
                           [0.0, 0.0, 0.0, 1.0]])
                            
    return rot_matrix

#================================================
# calculate rotation about the axis
#================================================
def calc_theta(Rmat):
    val = ((Rmat[0, 0] + Rmat[1, 1] + Rmat[2, 2]) - 1) / 2
    if val > 1:
        val = 1
    elif val <-1:
        val = -1
    return math.acos(val)

#================================================
# calculate axis of rotation
#================================================
def calculate_axis(rot_mat, theta):
    const = 1 / (2 * math.sin(theta))
    axis_zero = (rot_mat[2][1] - rot_mat[1][2]) * const
    axis_one = (rot_mat[0][2] - rot_mat[2][0]) * const
    axis_two = (rot_mat[1][0] - rot_mat[0][1]) * const
    return axis_zero, axis_one, axis_two

# convert from Euler angles to axis-angle:
def EulerToRotVec(Rx, Ry, Rz):

    rot_matrix = rotvec_to_rotMat(Rx, Ry, Rz)
    #print("rot_matrix:", rot_matrix)
    theta = calc_theta(rot_matrix)
    #print("theta:", theta)
    axis_zero, axis_one, axis_two = calculate_axis(rot_matrix, theta)
    return axis_zero, axis_one, axis_two

# convert Euler angles to rotation matrix
def rotvec_to_rotMat(yaw, pitch, roll):
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]])
    Ry_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]])
    # R = RzRyRx
    rotMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    return rotMat



####
# convert from rotation vector to rotation matrix
####
def rotvec2rotmat(rotvec):

	rx = rotvec[0]
	ry = rotvec[1]
	rz = rotvec[2]

	# rotation vector to angle and unit vector
	theta = math.sqrt(rx*rx + ry*ry + rz*rz)
	ux = rx/theta
	uy = ry/theta
	uz = rz/theta
	cth = math.cos(theta)
	sth = math.sin(theta)
	vth = 1-math.cos(theta)

	# column 1
	r11 = ux*ux*vth + cth
	r21 = ux*uy*vth + uz*sth
	r31 = ux*uz*vth - uy*sth
	# column 2
	r12 = ux*uy*vth - uz*sth
	r22 = uy*uy*vth + cth
	r32 = uy*uz*vth + ux*sth
	# column 3
	r13 = ux*uz*vth + uy*sth
	r23 = uy*uz*vth - ux*sth
	r33 = uz*uz*vth + cth

	# elements are represented as an array
	rotmat = [ [r11, r21, r31], [ r12, r22, r32], [r13, r23, r33] ]
	
	return rotmat

####
####
# convert from rotation matrix to rotation vector
####
def rotmat2rotvec(rotmat):

    # array to matrix
    r11 = rotmat[0][0]
    r21 = rotmat[0][1]
    r31 = rotmat[0][2]
    r12 = rotmat[1][0]
    r22 = rotmat[1][1]
    r32 = rotmat[1][2]
    r13 = rotmat[2][0]
    r23 = rotmat[2][1]
    r33 = rotmat[2][2]

    # print("\n rotmat:")
    # print(rotmat)

    # rotation matrix to rotation vector
    val = (r11+r22+r33-1)/2
    if val > 1:
        val = 1
    elif val <-1:
        val = -1
    theta = math.acos(val)
    sth = math.sin(theta)
    # print("Theta:", theta, "\ttheta = 179.99:", d2r(179.99), "\ttheta = -179.99:", d2r(-179.99),"\ttheta = 180:", d2r(180.0), "\tval:", val)
    if ( (theta > d2r(179.99)) or (theta < d2r(-179.99)) ):
        theta = d2r(180)
        # avoid math domain error when r11, r22 and r33 are less than 0
        if r11 < -1:
            r11 = -1
        if r22 < -1:
            r22 = -1
        if r33 < -1:
            r33 = -1
        if (r21 < 0):

            if (r31 < 0):
                ux = math.sqrt((r11+1)/2)
                uy = -math.sqrt((r22+1)/2)
                uz = -math.sqrt((r33+1)/2)
            else:
                ux = math.sqrt((r11+1)/2)
                uy = -math.sqrt((r22+1)/2)
                uz = math.sqrt((r33+1)/2)

        else:
            if (r31 < 0):
                ux = math.sqrt((r11+1)/2)
                uy = math.sqrt((r22+1)/2)
                uz = -math.sqrt((r33+1)/2)
            else:
                ux = math.sqrt((r11+1)/2)
                uy = math.sqrt((r22+1)/2)
                uz = math.sqrt((r33+1)/2)


    else:
        if theta == 0:
            ux = 0
            uy = 0
            uz = 0
        else:
            ux = (r32-r23)/(2*sth)
            uy = (r13-r31)/(2*sth)
            uz = (r21-r12)/(2*sth)


    rotvec = [(theta*ux),(theta*uy),(theta*uz)]

    return rotvec

####
# transform force from base to TCP
####
def transformForceToTCP(actualPose, TCPForce):
	# Transform the TCP Force into TCP coordinates
	force_B = [TCPForce[0], TCPForce[1], TCPForce[2], 0, 0, 0]
	torque_B = [TCPForce[3], TCPForce[4], TCPForce[5], 0, 0, 0]
	tcp_B = [actualPose[0], actualPose[1], actualPose[2],0, 0, 0]
	invTCP = PoseInv(actualPose)
	poseTrans1 = PoseTrans(tcp_B , force_B )
	# Force in TCP frame
	force_T = PoseTrans(invTCP, poseTrans1 )
	poseTrans2 = PoseTrans(tcp_B , torque_B )
	# Torque in TCP frame
	torque_T = PoseTrans(invTCP, poseTrans2 )
	# transposed wrench 
	wrench_t = [force_T[0], force_T[1], force_T[2], torque_T[0], torque_T[1], torque_T[2]]
	return wrench_t

####
# get DH parameters
##
#### input arguments
# gen: the generation of the robot
#		3: CB3
#		5: e-Series
# model: the robot model
#		3: UR3/UR3e
#		5: UR5/UR5e
#		10: UR10/UR10e
##
#### return values
# a: translational offset in x axis of n frame (returned in the pose variable format)
# d: translational offset in z axis of n-1 frame (returned in the pose variable format)
# alpha: rotatinal offset in x axis of n frame (returned in the pose variable format)
##
####
def get_dh_parameter(gen,model):

	a_pose = [ 0, 0, 0, 0, 0, 0 ]
	d_pose = [ 0, 0, 0, 0, 0, 0 ]
	alpha_pose = [ d2r(90), 0, 0, d2r(90), -d2r(90), 0 ]

	if (gen == 3):
		if (model == 3): 
			a_pose = [ 0, -0.24365, -0.21325, 0, 0, 0 ]
			d_pose = [ 0.1519, 0, 0, 0.11235, 0.08535, 0.0819 ]
		elif (model == 5):
			a_pose = [ 0, -0.425, -0.39225, 0, 0, 0 ]
			d_pose = [ 0.089159, 0, 0, 0.10915, 0.09465, 0.0823 ]
		elif (model == 10):
			a_pose = [ 0, -0.612, -0.5723, 0, 0, 0 ]
			d_pose = [ 0.1273, 0, 0, 0.163941, 0.1157, 0.0922 ]
		# end
	elif (gen == 5 ):
		if (model == 3): 
			a_pose = [ 0, -0.24355, -0.2132, 0, 0, 0 ]
			d_pose = [ 0.15185, 0, 0, 0.13105, 0.08535, 0.0921 ]
		elif (model == 5):
			a_pose = [ 0, -0.425, -0.3922, 0, 0, 0 ]
			d_pose = [ 0.1625, 0, 0, 0.1333, 0.0997, 0.0996 ]
		elif (model == 10):
			a_pose = [ 0, -0.6127, -0.57155, 0, 0, 0 ]
			d_pose = [ 0.1807, 0, 0, 0.17415, 0.11985, 0.11655 ]
		# end
	# end
	
	dh_parameter = [ a_pose, d_pose, alpha_pose ]

	return dh_parameter
# end
####

####
# get the transformation pose from n-1 to n frame
##
#### input arguments
# gen: the generation of the robot
#		3: CB3
#		5: e-Series
# model: the robot model
#		3: UR3/UR3e
#		5: UR5/UR5e
#		10: UR10/UR10e
# n: frame number
# theta: joint positions
##
#### return values
# pose: transformation pose (identical to the transformation matrix)
##
#### mathematic equation
# T = trans_z(d) * rot_z(theta) * trans_x(a) * rot_x(alpha)
##
####
def get_transformation_pose_frame(gen, model, n, theta):

	# load the DH parameters
	dh_parameter = get_dh_parameter(gen, model)
	a_pose = dh_parameter[0]
	d_pose = dh_parameter[1]
	alpha_pose = dh_parameter[2]
	
	trans_z = [0,0,d_pose[n-1],0,0,0]
	rot_z = [0,0,0,0,0,theta[n-1]]
	trans_x = [a_pose[n-1],0,0,0,0,0]
	rot_x = [0,0,0,alpha_pose[n-1],0,0]
	
	pose = PoseTrans(trans_z,PoseTrans(rot_z,(PoseTrans(trans_x,rot_x))))

	return pose
# end
####

####
# get the origin position of the wrist
##
#### input arguments
# gen: the generation of the robot
#		3: CB3
#		5: e-Series
# model: the robot model
#		3: UR3/UR3e
#		5: UR5/UR5e
#		10: UR10/UR10e
# theta: joint positions
##
#### return values
# position: X, Y, Z coordinates of wrist 1 origin
##
####
def calc_joint_positions(gen, model, theta, TCP_pose, TCP_offset):
    # TCP_position = [TCP_pose[0], TCP_pose[1], TCP_pose[2]]
    pose_1 = get_transformation_pose_frame(gen, model, 1, theta)
    # print("\nPose 1:", pose_1)
    pose_2 = get_transformation_pose_frame(gen, model, 2, theta)
    # print("Pose 2:", pose_2)
    pose_3 = get_transformation_pose_frame(gen, model, 3, theta)
    # print("Pose 3:", pose_3)
    pose_4 = get_transformation_pose_frame(gen, model, 4, theta)
    # print("Pose 4:", pose_4)
    pose_5 = get_transformation_pose_frame(gen, model, 5, theta)
    # print("Pose 5:", pose_5)
    pose_6 = get_transformation_pose_frame(gen, model, 6, theta)
    # print("Pose 6:", pose_6)

    # fomat is PoseTrans(pose_from, pose_from_to)
    
    pose_56 = PoseTrans(pose_5, pose_6)
    pose_46 = PoseTrans(pose_4, PoseTrans(pose_5, pose_6))
    pose_36 = PoseTrans(pose_3, pose_46)
    pose_26 = PoseTrans(pose_2, pose_36)
    pose_16 = PoseTrans(pose_1, pose_26)
    pose_TCP_flange = PoseTrans(TCP_pose, PoseInv(TCP_offset))
    flange_pose = pose_TCP_flange
    base_pose = PoseTrans(flange_pose, PoseInv(pose_16))
    shoulder_pose = PoseTrans(flange_pose, PoseInv(pose_26))
    elbow_pose = PoseTrans(flange_pose, PoseInv(pose_36))
    wrist1_pose = PoseTrans(flange_pose, PoseInv(pose_46))
    wrist2_pose = PoseTrans(flange_pose, PoseInv(pose_56))
    wrist3_pose = PoseTrans(flange_pose, PoseInv(pose_6))

    jointPoses = []
    jointPoses.append(base_pose)
    jointPoses.append(shoulder_pose)
    jointPoses.append(elbow_pose)
    jointPoses.append(wrist1_pose)
    jointPoses.append(wrist2_pose)
    jointPoses.append(wrist3_pose)
    jointPoses.append(flange_pose)
    jointPoses.append(TCP_pose)
    
    # wrist_position = [ wrist_pose[0], wrist_pose[1], wrist_pose[2] ]

    return jointPoses #wrist_pose
    # return wrist1_pose


def collisionDectection(jointPoses):
    base_pose = jointPoses[0]
    shoulder_pose = jointPoses[1]
    elbow_pose = jointPoses[2]
    wrist1_pose = jointPoses[3]
    wrist2_pose = jointPoses[4]
    wrist3_pose = jointPoses[5]
    flange_pose = jointPoses[6]
    TCP_pose = jointPoses[7]
    TCP_offset = [0.0, 0.0, 0.02, 0.0, 0.0, 0.0]
    TCP_pose_offset = PoseTrans(TCP_pose, PoseInv(TCP_offset))
    collisionLinks = []

    # base_position = [base_pose[0], base_pose[1], base_pose[2]]
    shoulder_position = [shoulder_pose[0], shoulder_pose[1], shoulder_pose[2]]
    shoulder_elbow_vector = [shoulder_pose[0] - elbow_pose[0], shoulder_pose[1] - elbow_pose[1], shoulder_pose[2] - elbow_pose[2]]
    elbow_wrist1_vector = [wrist1_pose[0] - elbow_pose[0], wrist1_pose[1] - elbow_pose[1], wrist1_pose[2] - elbow_pose[2]]
    shoulder_elbow_vector = shoulder_elbow_vector / np.linalg.norm(shoulder_elbow_vector)
    elbow_wrist1_vector = elbow_wrist1_vector / np.linalg.norm(elbow_wrist1_vector)



    # print("\nshoulder vector", shoulder_elbow_vector)
    # print("Elbow vector", elbow_wrist1_vector)
    perp_vector = np.cross(shoulder_elbow_vector, elbow_wrist1_vector)
    mag_perp_vector = math.sqrt(perp_vector[0]**2 + perp_vector[1]**2  + perp_vector[2]**2  )
    perp_vector[0] = perp_vector[0]* (0.174/mag_perp_vector)
    perp_vector[1] = perp_vector[1]* (0.174/mag_perp_vector)
    perp_vector[2] = perp_vector[2]* (0.174/mag_perp_vector)
    shoulder_offset = [perp_vector[0] + shoulder_position[0], perp_vector[1] + shoulder_position[1], perp_vector[2] + shoulder_position[2] ]
    elbow_position = [elbow_pose[0], elbow_pose[1], elbow_pose[2]]
    elbow_offset = [perp_vector[0] + elbow_position[0], perp_vector[1] + elbow_position[1], perp_vector[2] + elbow_position[2] ]
    wrist1_position = [ wrist1_pose[0], wrist1_pose[1], wrist1_pose[2] ]
    # wrist2_position = [ wrist2_pose[0], wrist2_pose[1], wrist2_pose[2] ]
    # wrist3_position = [ wrist3_pose[0], wrist3_pose[1], wrist3_pose[2] ]
    flange_position = [ flange_pose[0], flange_pose[1], flange_pose[2] ]
    # TCP_position = [TCP_pose[0], TCP_pose[1], TCP_pose[2]]
    
    # lengthBase = 0.1807 #0.270
    # lengthShoulder = 0.190 #0.335
    # lengthShoulderElbow = 0.615
    lengthElbowOffset = 0.174
    # lengthElbowWrist1 = 0.560
    lengthWrist1Wrist2 = 0.140
    lengthWrist2Wrist3 = 0.130
    # lengthWrist3Flange = 0.130
    # lengthGripper = 0.428
    # link 1: base to sholder
    link1_radius = 0.150/2 
    # link 2: shoulder offset
    link2_radius = 0.150/2 
    # link 3: shoulderOffset to elbowOffset
    link3_radius = 0.110/2 
    # link 4: elbowOffset to elbow
    link4_radius = 0.115/2
    # link 5: elbow to wrist 1
    link5_radius = 0.090/2
    # link 6: wrist 1 to wrist2
    # link 7: wrist 2 to wrist3
    # link 8: wrist 3 to flange
    link678_radius = 0.090/2
    # gripper: from flange to TCP
    gripper_radius = 0.090/2
    # stand: from base to z = -1m
    stand_radius = 0.195
    # saftey limit to avoid collisions: units in m
    safetyDelta = 0.01

    # stand_width = 0.260
    # stand_depth = 0.260
    # stand_height = 1.0

    # standTopCorner1 = [stand_width/2, stand_depth/2, 0.0]
    # standTopCorner2 = [stand_width/2, -stand_depth/2, 0.0]
    # standTopCorner3 = [-stand_width/2, -stand_depth/2, 0.0]
    # standTopCorner4 = [-stand_width/2, stand_depth/2, 0.0]
    # standBotCorner1 = [stand_width/2, stand_depth/2, -stand_height]
    # standBotCorner2 = [stand_width/2, -stand_depth/2, -stand_height]
    # standBotCorner3 = [-stand_width/2, -stand_depth/2, -stand_height]
    # standBotCorner4 = [-stand_width/2, stand_depth/2, -stand_height]


    # calculate end points for each link:
    # Link 1: Base
    link1a = base_pose
    link1a_position = [link1a[0], link1a[1], link1a[2]]
    offset = [0, 0 , -(shoulder_pose[2] + link1_radius + 0.02), 0, 0, 0]
    link1b = PoseTrans(base_pose, PoseInv(offset))
    link1b_position = [link1b[0], link1b[1], link1b[2]]
    # print("\nlink1a position:", link1a)
    # print("link1b position:", link1b)

    # link 2: shoulder (horizontal)
    offset = [0, 0 , (link2_radius), 0, 0, 0]
    link2a = PoseTrans(shoulder_pose, PoseInv(offset))
    link2a_position = [link2a[0], link2a[1], link2a[2]]
    offset = [0, 0 , -(shoulder_pose[2] + link1_radius), 0, 0, 0]
    link2b = PoseTrans(shoulder_pose, PoseInv(offset))
    link2b_position = [link2b[0], link2b[1], link2b[2]]
    # print("\nlink2a position:", link2a)
    # print("link2b position:", link2b)

    #link 5: elbow to wrist 1 - pose isn't given so need a different method to find the end points of link5:
    r0 = shoulder_offset
    r1 = elbow_offset
    # print("\nr0:", r0)
    # print("r1:", r1)
    v = [r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]]
    v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    # v is now a unit vector
    v_unit = [v[0]/v_mag, v[1]/v_mag, v[2]/v_mag]
    # t value to calculate r1
    t = (r1[0] - r0[0])/v_unit[0]
    # t1 = (r1[1] - r0[1])/v_unit[1]
    # t2 = (r1[2] - r0[2])/v_unit[2]
    link3a_position = [0,0,0]
    link3b_position = [0,0,0]
    link3c_position = [0,0,0]
    t1 = link2_radius
    link3a_position[0] = r0[0] - (t1)*v_unit[0]
    link3a_position[1] = r0[1] - (t1)*v_unit[1]
    link3a_position[2] = r0[2] - (t1)*v_unit[2]
    t2 = v_mag + link4_radius
    link3b_position[0] = r0[0] + (t2)*v_unit[0]
    link3b_position[1] = r0[1] + (t2)*v_unit[1]
    link3b_position[2] = r0[2] + (t2)*v_unit[2]
    t1 = -link2_radius
    link3c_position[0] = r0[0] - (t1)*v_unit[0]
    link3c_position[1] = r0[1] - (t1)*v_unit[1]
    link3c_position[2] = r0[2] - (t1)*v_unit[2]
     # offset = [0, 0 , (link5_radius), 0, 0, 0]
    # link5a = PoseTrans(elbow_pose, PoseInv(offset))
    link3a_position = [link3a_position[0], link3a_position[1], link3a_position[2]]
    # offset = [0, 0 , -(link5_radius + lengthElbowOffset + link3_radius +  0.02), 0, 0, 0]
    # link5b = PoseTrans(elbow_pose, PoseInv(offset))
    link3b_position = [link3b_position[0], link3b_position[1], link3b_position[2]]
    link3c_position = [link3c_position[0], link3c_position[1], link3c_position[2]]

    dist = math.sqrt((link3a_position[0] - link3b_position[0])**2 + (link3a_position[1] - link3b_position[1])**2 + (link3a_position[2] - link3b_position[2])**2)

    Rx_roll = r2d(math.pi + math.asin((link3a_position[0] - link3b_position[0])/dist))
    Ry_pitch = r2d(math.asin((link3a_position[1] - link3b_position[1])/dist))
    Ry_yaw = 0
    [Rx, Ry, Rz] = rpy2rotvec(Rx_roll, Ry_pitch, Ry_yaw)
    link3a = [link3a_position[0],link3a_position[1], link3a_position[2], Rx, Ry, Rz ]
    link3b = [link3b_position[0],link3b_position[1], link3b_position[2], Rx, Ry, Rz ]

    # print("\nlink3a position:", link3a)
    # print("link3b position:", link3b)

    # link 4: elbow to elbow Offset
    offset = [0, 0 , -(link5_radius + lengthElbowOffset), 0, 0, 0]
    link4a = PoseTrans(elbow_pose, PoseInv(offset))
    link4a_position = [link4a[0], link4a[1], link4a[2]]
    offset = [0, 0 , (link3_radius), 0, 0, 0]
    link4b = PoseTrans(elbow_pose, PoseInv(offset))
    link4b_position = [link4b[0], link4b[1], link4b[2]]
    # print("\nlink3a position:", link4a)
    # print("link3b position:", link4b)

    #link 5: elbow to wrist 1 - pose isn't given so need a different method to find the end points of link5:
    r0 = elbow_position
    r1 = wrist1_position
    # print("\nr0:", r0)
    # print("r1:", r1)
    v = [r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]]
    v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    # v is now a unit vector
    v_unit = [v[0]/v_mag, v[1]/v_mag, v[2]/v_mag]
    # t value to calculate r1
    t = (r1[0] - r0[0])/v_unit[0]
    # t1 = (r1[1] - r0[1])/v_unit[1]
    # t2 = (r1[2] - r0[2])/v_unit[2]
    link5a_position = [0,0,0]
    link5b_position = [0,0,0]
    link5c_position = [0,0,0]
    t1 = link678_radius
    link5a_position[0] = r0[0] - (t1)*v_unit[0]
    link5a_position[1] = r0[1] - (t1)*v_unit[1]
    link5a_position[2] = r0[2] - (t1)*v_unit[2]
    t2 = t + link678_radius
    link5b_position[0] = r0[0] + (t2)*v_unit[0]
    link5b_position[1] = r0[1] + (t2)*v_unit[1]
    link5b_position[2] = r0[2] + (t2)*v_unit[2]

    t3 = t
    link5c_position[0] = r0[0] + (t3)*v_unit[0]
    link5c_position[1] = r0[1] + (t3)*v_unit[1]
    link5c_position[2] = r0[2] + (t3)*v_unit[2]

    # offset = [0, 0 , (link5_radius), 0, 0, 0]
    # link5a = PoseTrans(elbow_pose, PoseInv(offset))
    # link5a_position = [link5a[0], link5a[1], link5a[2]]
    # offset = [0, 0 , -(link5_radius + lengthElbowOffset + link3_radius +  0.02), 0, 0, 0]
    # link5b = PoseTrans(elbow_pose, PoseInv(offset))
    # link5b_position = [link5b[0], link5b[1], link5b[2]]
    # print("\nlink5a position:", link5a)
    # print("link5b position:", link5b)

    dist = math.sqrt((link5a_position[0] - link5b_position[0])**2 + (link5a_position[1] - link5b_position[1])**2 + (link5a_position[2] - link5b_position[2])**2)

    Rx_roll = r2d(math.pi + math.asin((link5a_position[0] - link5b_position[0])/dist))
    Ry_pitch = r2d(math.asin((link5a_position[1] - link5b_position[1])/dist))
    Ry_yaw = 0
    [Rx, Ry, Rz] = rpy2rotvec(Rx_roll, Ry_pitch, Ry_yaw)
    link5a = [link5a_position[0],link5a_position[1], link5a_position[2], Rx, Ry, Rz ]
    link5b = [link5b_position[0],link5b_position[1], link5b_position[2], Rx, Ry, Rz ]
    link5c = [link5c_position[0],link5c_position[1], link5c_position[2], Rx, Ry, Rz ]

    #link 6: wrist 1 to wrist 2
    offset = [0, 0 , (link678_radius), 0, 0, 0]
    link6a = PoseTrans(wrist1_pose, PoseInv(offset))
    link6a_position = [link6a[0], link6a[1], link6a[2]]
    offset = [0, 0 , -(link678_radius + lengthWrist1Wrist2  +  0.02), 0, 0, 0]
    link6b = PoseTrans(wrist1_pose, PoseInv(offset))
    link6b_position = [link6b[0], link6b[1], link6b[2]]
    # print("\nlink6a position:", link6a)
    # print("link6b position:", link6b)

    #link 7: wrist 2 to wrist 3
    offset = [0, 0 , (link678_radius), 0, 0, 0]
    link7a = PoseTrans(wrist2_pose, PoseInv(offset))
    link7a_position = [link7a[0], link7a[1], link7a[2]]
    offset = [0, 0 , -(link678_radius + lengthWrist2Wrist3), 0, 0, 0]
    link7b = PoseTrans(wrist2_pose, PoseInv(offset))
    link7b_position = [link7b[0], link7b[1], link7b[2]]
    # print("\nlink7a position:", link7a)
    # print("link7b position:", link7b)

    #link 7: wrist 3 to flange
    offset = [0, 0 , (link678_radius), 0, 0, 0]
    link8a = PoseTrans(wrist3_pose, PoseInv(offset))
    link8a_position = [link8a[0], link8a[1], link8a[2]]
    # offset = [0, 0 , -(link678_radius + lengthWrist2Wrist3), 0, 0, 0]
    # link8b = PoseTrans(wrist3_pose, PoseInv(offset))
    link8b = flange_pose
    # link8b = flange_pose
    link8b_position = flange_position
    # print("\nlink8a position:", link8a)
    # print("link8b position:", link8b)

    grippera = flange_pose
    grippera_position = flange_position
    gripperb = TCP_pose
    gripperb_position = [TCP_pose_offset[0], TCP_pose_offset[1], TCP_pose_offset[2]]

    standa = [0.0,0.0,0.0, 2.149, -2.291, -0.00]
    standb = [0.0,0.0,-1.0, 2.149, -2.291, -0.00]
    standa_position = [0.0,0.0,0.0]
    standb_position = [0.0,0.0,-1.0]


    # base_position = [base_pose[0], base_pose[1], base_pose[2]]
    shoulder_position = [shoulder_pose[0], shoulder_pose[1], shoulder_pose[2]]
    shoulder_elbow_vector = [shoulder_pose[0] - elbow_pose[0], shoulder_pose[1] - elbow_pose[1], shoulder_pose[2] - elbow_pose[2]]
    elbow_wrist1_vector = [wrist1_pose[0] - elbow_pose[0], wrist1_pose[1] - elbow_pose[1], wrist1_pose[2] - elbow_pose[2]]
    shoulder_elbow_vector = shoulder_elbow_vector / np.linalg.norm(shoulder_elbow_vector)
    elbow_wrist1_vector = elbow_wrist1_vector / np.linalg.norm(elbow_wrist1_vector)



    # print("\nshoulder vector", shoulder_elbow_vector)
    # print("Elbow vector", elbow_wrist1_vector)
    perp_vector = np.cross(shoulder_elbow_vector, elbow_wrist1_vector)
    mag_perp_vector = math.sqrt(perp_vector[0]**2 + perp_vector[1]**2  + perp_vector[2]**2  )
    perp_vector[0] = perp_vector[0]* (0.174/mag_perp_vector)
    perp_vector[1] = perp_vector[1]* (0.174/mag_perp_vector)
    perp_vector[2] = perp_vector[2]* (0.174/mag_perp_vector)
    shoulder_offset = [perp_vector[0] + shoulder_position[0], perp_vector[1] + shoulder_position[1], perp_vector[2] + shoulder_position[2] ]
    elbow_position = [elbow_pose[0], elbow_pose[1], elbow_pose[2]]
    elbow_offset = [perp_vector[0] + elbow_position[0], perp_vector[1] + elbow_position[1], perp_vector[2] + elbow_position[2] ]
    wrist1_position = [ wrist1_pose[0], wrist1_pose[1], wrist1_pose[2] ]
    # wrist2_position = [ wrist2_pose[0], wrist2_pose[1], wrist2_pose[2] ]
    # wrist3_position = [ wrist3_pose[0], wrist3_pose[1], wrist3_pose[2] ]
    flange_position = [ flange_pose[0], flange_pose[1], flange_pose[2] ]
    # TCP_position = [TCP_pose[0], TCP_pose[1], TCP_pose[2]]
    
    ###############################################################
    # Check for collisions between Gripper & Links 5, 3, 2, 1, pedastal
    ###############################################################
    gripper_collisionFlag = False

    # check for collisions between the gripper cylinder and link5 cylinder
    distance = calcCylinderPoints(grippera, gripperb, gripper_radius, link5a, link5c, link5_radius)
    minDistance = safetyDelta + link5_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 5 & Gripper (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        gripper_collisionFlag= True

    # check for collision between gripper and link 5
    distance = findSegmentIntersection(grippera_position, gripperb_position, link5a_position, link5b_position)
    minDistance = gripper_radius + link5_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 5 & Gripper !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        gripper_collisionFlag= True
    # else:
    #     print("\n\tNo collision detected: Link 5 & Gripper")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between the gripper cylinder and link3 cylinder
    distance = calcCylinderPoints(grippera, gripperb, gripper_radius, link3a, link3b, link3_radius)
    minDistance = safetyDelta + link3_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 3 & Gripper (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        gripper_collisionFlag= True

    # check for collision between gripper and link 3
    distance = findSegmentIntersection(grippera_position, gripperb_position, link3a_position, link3b_position)
    minDistance = gripper_radius + link3_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 3 & Gripper !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        gripper_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 3 & Gripper")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between the gripper cylinder and link2 cylinder
    distance = calcCylinderPoints(grippera, gripperb, gripper_radius, link2a, link2b, link2_radius)
    minDistance = safetyDelta + link2_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 2 & Gripper (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        gripper_collisionFlag= True

    # check for collision between gripper and link 2
    distance = findSegmentIntersection(grippera_position, gripperb_position, link2a_position, link2b_position)
    minDistance = gripper_radius + link2_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 2 & Gripper !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        gripper_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 2 & Gripper")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between the gripper cylinder and link1 cylinder
    distance = calcCylinderPoints(grippera, gripperb, gripper_radius, link1a, link1b, link1_radius)
    minDistance = safetyDelta + link1_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 1 & Gripper (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        gripper_collisionFlag= True

    # check for collision between gripper and link 1
    distance = findSegmentIntersection(grippera_position, gripperb_position, link1a_position, link1b_position)
    minDistance = gripper_radius + link1_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 1 & Gripper !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        gripper_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 1 & Gripper")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between the gripper cylinder and link5 cylinder
    distance = calcCylinderPoints(grippera, gripperb, gripper_radius, standa, standb, stand_radius)
    minDistance = safetyDelta + stand_radius
    if distance < minDistance and ((grippera[2] <= (standa[2])) or (gripperb[2] <= (standa[2]+ gripper_radius))):
        print("\n\t!!! Collision detected: Stand & Gripper (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        gripper_collisionFlag= True

    # check for collision between gripper and pedestal (modeled as a 1m tall cylinder)
    distance = findSegmentIntersection(grippera_position, gripperb_position, standa, standb)
    minDistance = gripper_radius + stand_radius + safetyDelta
    if (distance < minDistance) and ((grippera[2] <= (standa[2])) or (gripperb[2] <= (standa[2]+ gripper_radius))):
        print("\n\t!!! Collision detected: Stand & Gripper !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        gripper_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Stand & Gripper")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    if gripper_collisionFlag == True:
        collisionLinks.append('Gripper')

    ###############################################################
    # Check for collisions between Link 8 & Links 5, 3, 2, 1, pedastal
    ###############################################################

    # check for collisions between link 8 and link5 cylinder
    distance = calcCylinderPoints(link8a, link8b, link678_radius, link5a, link5b, link5_radius)
    minDistance = safetyDelta + link5_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 5 & Link 8 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link8_collisionFlag= True

    # check for collision between link 8 and link 5
    link8_collisionFlag = False
    distance = findSegmentIntersection(link8a_position, link8b_position, link5a_position, link5b_position)
    minDistance = link678_radius + link5_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 5 & Link 8 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link8_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 5 & Link 8")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between link 8 and link5 cylinder
    distance = calcCylinderPoints(link8a, link8b, link678_radius, link3a, link3b, link3_radius)
    minDistance = safetyDelta + link3_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 3 & Link 8 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link8_collisionFlag= True

    # check for collision between link8 and link 3
    distance = findSegmentIntersection(link8a_position, link8b_position, link3a_position, link3b_position)
    minDistance = link678_radius + link3_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 3 & Link 8 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link8_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 3 & Link 8")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)


    # check for collisions between link 8 and link2 cylinder
    distance = calcCylinderPoints(link8a, link8b, link678_radius, link2a, link2b, link2_radius)
    minDistance = safetyDelta + link2_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 2 & Link 8 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link8_collisionFlag= True

    # check for collision between link8 and link 2
    distance = findSegmentIntersection(link8a_position, link8b_position, link2a_position, link2b_position)
    minDistance = link678_radius + link2_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 2 & Link 8 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link8_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 2 & Link 8")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between link 8 and link5 cylinder
    distance = calcCylinderPoints(link8a, link8b, link678_radius, link1a, link1b, link1_radius)
    minDistance = safetyDelta + link1_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 1 & Link 8 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link8_collisionFlag= True

    # check for collision between link8 and link 1
    distance = findSegmentIntersection(link8a_position, link8b_position, link1a_position, link1b_position)
    minDistance = link678_radius + link1_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 1 & Link 8 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link8_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 1 & Link 8")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)
    
    # check for collisions between link 8 and link5 cylinder
    distance = calcCylinderPoints(link8a, link8b, link678_radius, standa, standb, stand_radius)
    minDistance = safetyDelta + stand_radius
    if distance < minDistance and ((link8a_position[2] <= (standa[0] + link678_radius)) or (link8b_position[2] <= (standa[0] + link678_radius))):
        print("\n\t!!! Collision detected: Stand & Link 8 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link8_collisionFlag= True

    # check for collision between link8 and pedestal (modeled as a 1m tall cylinder)
    distance = findSegmentIntersection(link8a_position, link8b_position, standa, standb)
    minDistance = link678_radius + stand_radius + safetyDelta
    if distance < minDistance and ((link8a_position[2] <= (standa[0] + link678_radius)) or (link8b_position[2] <= (standa[0] + link678_radius))):
        print("\n\t!!! Collision detected: Stand & Link 8 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link8_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Stand & Link 8")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)
    
    if link8_collisionFlag == True:
        collisionLinks.append('Link8')

    ###############################################################
    # Check for collisions between Link 7 & Links 3, 2, 1, pedastal
    ###############################################################
    link7_collisionFlag = False

    # check for collisions between link 7 and link5 cylinder
    distance = calcCylinderPoints(link7a, link7b, link678_radius, link3a, link3b, link3_radius)
    minDistance = safetyDelta + link3_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 3 & Link 7 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link7_collisionFlag= True

    # check for collision between link7 and link 3
    distance = findSegmentIntersection(link7a_position, link7b_position, link3a_position, link3b_position)
    minDistance = link678_radius + link3_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 3 & Link 7 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link7_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 3 & Link 7")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between link 7 and link3 cylinder
    distance = calcCylinderPoints(link7a, link7b, link678_radius, link2a, link2b, link2_radius)
    minDistance = safetyDelta + link2_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 2 & Link 7 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link7_collisionFlag= True

    # check for collision between link7 and link 2
    distance = findSegmentIntersection(link7a_position, link7b_position, link2a_position, link2b_position)
    minDistance = link678_radius + link2_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 2 & Link 7 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link7_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 2 & Link 7")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between link 7 and link1 cylinder
    distance = calcCylinderPoints(link7a, link7b, link678_radius, link1a, link1b, link1_radius)
    minDistance = safetyDelta + link1_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 1 & Link 7 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link7_collisionFlag= True

    # check for collision between link7 and link 1
    distance = findSegmentIntersection(link7a_position, link7b_position, link1a_position, link1b_position)
    minDistance = link678_radius + link1_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 1 & Link 7 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link7_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 1 & Link 7")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between link 7 and link1 cylinder
    distance = calcCylinderPoints(link7a, link7b, link678_radius, standa, standb, stand_radius)
    minDistance = safetyDelta + stand_radius
    if distance < minDistance and ((link7a_position[2] <= (standa[0] + link678_radius)) or (link7b_position[2] <= (standa[0] + link678_radius))):
        print("\n\t!!! Collision detected: Stand & Link 7 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link7_collisionFlag= True

    # check for collision between link7 and pedestal (modeled as a 1m tall cylinder)
    distance = findSegmentIntersection(link7a_position, link7b_position, standa, standb)
    minDistance = link678_radius + stand_radius + safetyDelta
    if distance < minDistance and ((link7a_position[2] <= (standa[0] + link678_radius)) or (link7b_position[2] <= (standa[0] + link678_radius))):
        print("\n\t!!! Collision detected: Stand & Link 7 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link8_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Stand & Link 7")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    if link7_collisionFlag == True:
        collisionLinks.append('Link7')

    ###############################################################
    # Check for collisions between Link 6 & Links 3, 2, 1, pedastal
    ###############################################################
    link6_collisionFlag = False

    # check for collisions between link 6 and link5 cylinder
    distance = calcCylinderPoints(link6a, link6b, link678_radius, link3a, link3b, link3_radius)
    minDistance = safetyDelta + link5_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 5 & Link 6 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link6_collisionFlag= True

    # check for collision between link6 and link 3
    distance = findSegmentIntersection(link6a_position, link6b_position, link3a_position, link3b_position)
    minDistance = link678_radius + link3_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 3 & Link 6 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link6_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 3 & Link 6")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between link 6 and link2 cylinder
    distance = calcCylinderPoints(link6a, link6b, link678_radius, link2a, link2b, link2_radius)
    minDistance = safetyDelta + link2_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 2 & Link 6 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link6_collisionFlag= True

    # check for collision between link6 and link 2
    distance = findSegmentIntersection(link6a_position, link6b_position, link2a_position, link2b_position)
    minDistance = link678_radius + link2_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 2 & Link 6 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link6_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 2 & Link 6")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between link 6 and link1 cylinder
    distance = calcCylinderPoints(link6a, link6b, link678_radius, link1a, link1b, link1_radius)
    minDistance = safetyDelta + link1_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 1 & Link 6 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link6_collisionFlag= True

    # check for collision between link6 and link 1
    distance = findSegmentIntersection(link6a_position, link6b_position, link1a_position, link1b_position)
    minDistance = link678_radius + link1_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 1 & Link 6 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link6_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 1 & Link 6")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between link 6 and link5 cylinder
    distance = calcCylinderPoints(link6a, link6b, link678_radius, standa, standb, stand_radius)
    minDistance = safetyDelta + stand_radius
    if distance < minDistance and ((link6a_position[2] <= (standa[0] + link678_radius)) or (link6b_position[2] <= (standa[0] + link678_radius))):
        print("\n\t!!! Collision detected: stand & Link 6 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link6_collisionFlag= True

    # check for collision between link6 and pedestal (modeled as a 1m tall cylinder)
    distance = findSegmentIntersection(link6a_position, link6b_position, standa, standb)
    minDistance = link678_radius + stand_radius + safetyDelta
    if distance < minDistance and ((link6a_position[2] <= (standa[0] + link678_radius)) or (link6b_position[2] <= (standa[0] + link678_radius))):
        print("\n\t!!! Collision detected: Stand & Link 6 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link6_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Stand & Link 6")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    if link6_collisionFlag == True:
        collisionLinks.append('Link6')

    ###############################################################
    # Check for collisions between Link 5 & Links 2, 1, pedastal
    ###############################################################
    link5_collisionFlag = False

    # check for collisions between link 5 and link2 cylinder
    distance = calcCylinderPoints(link5a, link5b, link5_radius, link2a, link2b, link2_radius)
    minDistance = safetyDelta + link2_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 2 & Link 5 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link5_collisionFlag= True

    # check for collision between link5 and link 2
    distance = findSegmentIntersection(link5a_position, link5b_position, link2a_position, link2b_position)
    minDistance = link5_radius + link2_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 2 & Link 5 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link5_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 2 & Link 5")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between link 5 and link1 cylinder
    distance = calcCylinderPoints(link5a, link5b, link5_radius, link1a, link1b, link1_radius)
    minDistance = safetyDelta + link1_radius
    dist = math.sqrt((link5b[0] - link1b[0])**2 +  (link5b[1] - link1b[1])**2 +  (link5b[2] - link1b[2])**2)
    if (distance < minDistance) and (link5b[2] > (link1b[2] + link5_radius ) and (dist < minDistance)):
        print("\n\t!!! Collision detected: Link 1 & Link 5 (cylinder method 1) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link5_collisionFlag= True
    elif (dist < minDistance) and (link5b[2] < (link1b[2] - link5_radius)):
        print("\n\t!!! Collision detected: Link 1 & Link 5 (cylinder method 2) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link5_collisionFlag= True
    elif (dist < minDistance) and ((link5b[2] < (link1b[2] + link5_radius)) and (link5b[2] > (link1b[2] - link5_radius)) ):
        print("\n\t!!! Collision detected: Link 1 & Link 5 (cylinder method 3) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link5_collisionFlag= True

    # check for collision between link5 and link 1
    distance = findSegmentIntersection(link6a_position, link6b_position, link1a_position, link1b_position)
    minDistance = link5_radius + link1_radius + safetyDelta
    if distance < minDistance:
        print("\n\t!!! Collision detected: Link 1 & Link 5 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link5_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Link 1 & Link 5")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    # check for collisions between link 5 and link2 cylinder
    distance = calcCylinderPoints(link5a, link5b, link5_radius, standa, standb, stand_radius)
    minDistance = safetyDelta + stand_radius
    if distance < minDistance and ((link5a_position[2] <= (standa[0] + link5_radius)) or (link5b_position[2] <= (standa[0] + link5_radius))):
        print("\n\t!!! Collision detected: Stand & Link 5 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link5_collisionFlag= True

    # check for collision between link6 and pedestal (modeled as a 1m tall cylinder)
    distance = findSegmentIntersection(link5a_position, link5b_position, standa, standb)
    minDistance = link5_radius + stand_radius + safetyDelta
    if distance < minDistance and ((link5a_position[2] <= (standa[0] + link5_radius)) or (link5b_position[2] <= (standa[0] + link5_radius))):
        print("\n\t!!! Collision detected: Stand & Link 5 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link5_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Stand & Link 5")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    if link5_collisionFlag == True:
        collisionLinks.append('Link5')
    ###############################################################
    # Check for collisions between Link 4 & pedastal
    ###############################################################
    link4_collisionFlag = False
    # check for collision between link4 and pedestal (modeled as a 1m tall cylinder)

    # check for collisions between link 4 and link2 cylinder
    distance = calcCylinderPoints(link4a, link4b, link4_radius, standa, standb, stand_radius)
    minDistance = safetyDelta + stand_radius
    if distance < minDistance and ((link4a_position[2] <= (standa[0] + link4_radius)) or (link4b_position[2] <= (standa[0] + link4_radius))):
        print("\n\t!!! Collision detected: Stand & Link 4 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link4_collisionFlag= True

    distance = findSegmentIntersection(link4a_position, link4b_position, standa, standb)
    minDistance = link4_radius + stand_radius + safetyDelta
    if distance < minDistance and ((link4a_position[2] <= (standa[0] + link4_radius)) or (link4b_position[2] <= (standa[0] + link4_radius))):
        print("\n\t!!! Collision detected: Stand & Link 4 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link4_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Stand & Link 4")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    if link4_collisionFlag == True:
        collisionLinks.append('Link4')

    ###############################################################
    # Check for collisions between Link 3 & pedastal
    ###############################################################
    link3_collisionFlag = False

    # check for collisions between link 5 and link2 cylinder
    distance = calcCylinderPoints(link3a, link3b, link3_radius, standa, standb, stand_radius)
    minDistance = safetyDelta #+ stand_radius
    if distance < minDistance:
        print("\n\t!!! Collision detected: Stand & Link 3 (cylinder method) !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link3_collisionFlag= True

    # check for collision between link3 and pedestal (modeled as a 1m tall cylinder)
    distance = findSegmentIntersection(link3c_position, link3b_position, standa, standb)
    minDistance = link3_radius + stand_radius + safetyDelta
    if distance < minDistance and ((link3a_position[2] <= (standa[0] + link3_radius)) or (link3b_position[2] <= (standa[0] + link3_radius))):
        print("\n\t!!! Collision detected: Stand & Link 3 !!!")
        print("\tDistance is: ", distance)
        print("\tMinimum distance: ", minDistance)
        link3_collisionFlag = True
    # else:
    #     print("\n\tNo collision detected: Stand & Link 3")
    #     print("\tDistance is: ", distance)
    #     print("\tMinimum distance: ", minDistance)

    if link3_collisionFlag == True:
        collisionLinks.append('Link3')

    # if (link3_collisionFlag == True or link4_collisionFlag == True or link5_collisionFlag == True or link6_collisionFlag == True or  link7_collisionFlag == True or link8_collisionFlag == True or gripper_collisionFlag == True):
    #     print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     print( "\n !!!!!!         COLLISION DETECTED           !!!!!!")
    #     print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # else:
    #     print("\n----------------------------------------------")
    #     print("\n------     No collisions detected      -------")
    #     print("\n----------------------------------------------")

    return collisionLinks
    # distance = calculateSegmentDistance(shoulder_position, wrist1_position, wrist2_position)
    # print("Distance btwn point and line segment:", distance)

def calcCylinderPoints(poseAOnLine1, poseBOnLine1, radius1, poseAOnLine2, poseBOnLine2, radius2): # points1 & 2 are centreline endpoints of the cylinder
    pointAOnLine1 = [poseAOnLine1[0], poseAOnLine1[1], poseAOnLine1[2] ]
    pointBOnLine1 = [poseBOnLine1[0], poseBOnLine1[1], poseBOnLine1[2] ]
    pointAOnLine2 = [poseAOnLine2[0], poseAOnLine2[1], poseAOnLine2[2] ]
    pointBOnLine2 = [poseBOnLine2[0], poseBOnLine2[1], poseBOnLine2[2] ]


    dir_vector = [pointBOnLine1[0] - pointAOnLine1[0], pointBOnLine1[1] - pointAOnLine1[1], pointBOnLine1[2] - pointAOnLine1[2]]
    mag = math.sqrt(dir_vector[0]**2 + dir_vector[1]**2 + dir_vector[2]**2)
    norm_dir_vector = [dir_vector[0]/mag, dir_vector[1]/mag, dir_vector[2]/mag]
    x_dir = [1,0,0]
    y_dir = [0,1,0]
    z_dir = [0,0,1]

    # find a vector perpendicular to the centerline of the cylinder
    res = np.dot(x_dir,norm_dir_vector)
    comp = np.dot(res, norm_dir_vector)
    perp_vector = [x_dir[0] - comp[0], x_dir[1] - res*comp[1], x_dir[2] - res*comp[2]]
    dot_result = np.dot(norm_dir_vector, perp_vector)
    # perp_vector = [x_dir[0] - res*norm_dir_vector[0], x_dir[1] - res*norm_dir_vector[1], x_dir[2] - res*norm_dir_vector[2]]
    # print("\nCalc1: perpvector*dirvector", dot_result)
    # make sure dot product isn't close to 0
    if dot_result < -0.00000000001 or dot_result > 0.00000000001:
        res = np.dot(y_dir,norm_dir_vector)
        comp = np.dot(res, norm_dir_vector)
        perp_vector = [y_dir[0] - comp[0], y_dir[1] - comp[1], y_dir[2] - comp[2]]
        dot_result = np.dot(norm_dir_vector, perp_vector)
        # print("\nCalc2: perpvector*dirvector", dot_result)
        if dot_result < -0.00000000001 or dot_result > 0.00000000001:
            res = np.dot(z_dir,norm_dir_vector)
            perp_vector = [z_dir[0] - comp[0], z_dir[1] - comp[1], z_dir[2] - comp[2]]
            dot_result = np.dot(norm_dir_vector, perp_vector)
            # print("\nCalc2: perpvector*dirvector", dot_result)
    
    # if res > -0.3 and res < 0.3:
    #     res = np.dot(y_dir,norm_dir_vector)
    #     comp = np.dot(res, norm_dir_vector)
    #     perp_vector = [y_dir[0] - comp[0], y_dir[1] - comp[1], y_dir[2] - comp[2]]
    #     print("\nCalc2: perpvector*dirvector", np.dot(norm_dir_vector, perp_vector))
    #     if res > -0.3 and res < 0.3:
    #         res = np.dot(z_dir,norm_dir_vector)
    #         perp_vector = [z_dir[0] - comp[0], z_dir[1] - comp[1], z_dir[2] - comp[2]]
    #         print("\nCalc3: perpvector*dirvector", np.dot(norm_dir_vector, perp_vector))
    # print("\nNorm dir vector:", norm_dir_vector)
    # print("res:", res)
    # print("comp:", comp)
    # print("\nPerpindicular test:")
    # print("perpvector:",perp_vector)
    
    # first perpendicular unit vector on cylinder 1
    mag = math.sqrt(perp_vector[0]**2 + perp_vector[1]**2 + perp_vector[2]**2)
    norm_perp_vector = [perp_vector[0]/mag, perp_vector[1]/mag,  perp_vector[2]/mag]
    # print("\tperpvector*dirvector", np.dot(norm_dir_vector, norm_perp_vector))
    # calculate second perpendicular unit vector on cylinder 1
    norm_perp_vector2 = np.cross(norm_dir_vector, norm_perp_vector)
    # print("\tperpvector*dirvector", np.dot(norm_dir_vector, norm_perp_vector2))
    # find end of cylinder 1 closest to cylinder 2:
    dir_vector2 = [pointBOnLine2[0] - pointAOnLine2[0], pointBOnLine2[1] - pointAOnLine2[1], pointBOnLine2[2] - pointAOnLine2[2]]
    dista = calculateDistance(pointAOnLine1, pointAOnLine2, dir_vector2)
    distb = calculateDistance(pointBOnLine1, pointAOnLine2, dir_vector2)
    if dista <= distb:
        smallestDist = dista
        # print("\nside A is closer!", smallestDist)
    else:
        smallestDist = distb
        # print("\nside B is closer!", smallestDist)
    smallestAngle = None
    points = []
    for theta in range(0,360):
        if dista <= distb:
            newPoint_x = radius1*math.cos(d2r(theta))*norm_perp_vector[0] + radius1*math.sin(d2r(theta))*norm_perp_vector2[0] + pointAOnLine1[0]
            newPoint_y = radius1*math.cos(d2r(theta))*norm_perp_vector[1] + radius1*math.sin(d2r(theta))*norm_perp_vector2[1] + pointAOnLine1[1]
            newPoint_z = radius1*math.cos(d2r(theta))*norm_perp_vector[2] + radius1*math.sin(d2r(theta))*norm_perp_vector2[2] + pointAOnLine1[2]
            newPoint = [newPoint_x, newPoint_y, newPoint_z]
        else:
            newPoint_x = radius1*math.cos(d2r(theta))*norm_perp_vector[0] + radius1*math.sin(d2r(theta))*norm_perp_vector2[0] + pointBOnLine1[0]
            newPoint_y = radius1*math.cos(d2r(theta))*norm_perp_vector[1] + radius1*math.sin(d2r(theta))*norm_perp_vector2[1] + pointBOnLine1[1]
            newPoint_z = radius1*math.cos(d2r(theta))*norm_perp_vector[2] + radius1*math.sin(d2r(theta))*norm_perp_vector2[2] + pointBOnLine1[2]
            newPoint = [newPoint_x, newPoint_y, newPoint_z]
        newDist = calculateDistance(newPoint, pointAOnLine2, dir_vector2)
        points.append(newPoint)
        if newDist <= smallestDist:
            smallestDist = newDist
            smallestAngle = theta

    # create a cylinder to represent the robot base
    lengthCyl1 = math.sqrt((pointAOnLine1[0] - pointBOnLine1[0])**2 + (pointAOnLine1[1] - pointBOnLine1[1])**2 + (pointAOnLine1[2] - pointBOnLine1[2])**2)
    # print("LengthBase:"), lengthBase
    cylinder1 = o3d.geometry.TriangleMesh.create_cylinder(radius = radius1, height = lengthCyl1) #height= heightBase)
    rotMat = rotVec_to_rotMat(poseAOnLine1)
    cylinder1.rotate(rotMat, [0,0,0])
    cylinder1.translate([pointAOnLine1[0] + (pointBOnLine1[0] - pointAOnLine1[0] )/2, pointAOnLine1[1] + (pointBOnLine1[1] - pointAOnLine1[1])/2, pointAOnLine1[2] + (pointBOnLine1[2] - pointAOnLine1[2] )/2  ])
    # base_cylinder.translate([base_pose[0] + (shoulder_pose[0] - base_pose[0] )/2, base_pose[1] + (shoulder_pose[1] - base_pose[1])/2, base_pose[2] + (shoulder_pose[2] - base_pose[2] + 0.0893 )/2  ])
    # robot += base_cylinder
    
    link1geo = cylinder1

    # create a cylinder to represent the robot base
    lengthCyl2 = math.sqrt((pointAOnLine2[0] - pointBOnLine2[0])**2 + (pointAOnLine2[1] - pointBOnLine2[1])**2 + (pointAOnLine2[2] - pointBOnLine2[2])**2)
    # print("LengthBase:"), lengthBase
    cylinder2 = o3d.geometry.TriangleMesh.create_cylinder(radius = radius2, height = lengthCyl2) #height= heightBase)
    rotMat = rotVec_to_rotMat(poseAOnLine2)
    cylinder2.rotate(rotMat, [0,0,0])
    cylinder2.translate([pointAOnLine2[0] + (pointBOnLine2[0] - pointAOnLine2[0] )/2, pointAOnLine2[1] + (pointBOnLine2[1] - pointAOnLine2[1])/2, pointAOnLine2[2] + (pointBOnLine2[2] - pointAOnLine2[2] )/2  ])
    # base_cylinder.translate([base_pose[0] + (shoulder_pose[0] - base_pose[0] )/2, base_pose[1] + (shoulder_pose[1] - base_pose[1])/2, base_pose[2] + (shoulder_pose[2] - base_pose[2] + 0.0893 )/2  ])
    # robot += base_cylinder
    
    link2geo = cylinder2

    geometries = o3d.geometry.TriangleMesh()

    # smaller robot joints
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01) #create a small sphere to represent point
        
        # rotMat = rotVec_to_rotMat(poseAOnLine2)
        # sphere.rotate(rotMat, [0,0,0])
        sphere.translate(point) #translate this sphere to point
        geometries += sphere
    
    # geometries.paint_uniform_color([1.0, 0.0, 0.0])
    # link1geo.paint_uniform_color([15/255, 99/255, 175/255])
    # link2geo.paint_uniform_color([10/255, 173/255, 254/255])

    # # o3d.visualization.draw_geometries([geometries, line_set])
    # visualizer = o3d.visualization.Visualizer() 
    # visualizer.create_window()
    # visualizer.get_render_option().show_coordinate_frame = True
    # visualizer.add_geometry(geometries)
    # visualizer.get_render_option().point_size = 1.5
    # # visualizer.add_geometry(box)
    # # visualizer.add_geometry(robot)
    # visualizer.add_geometry(link1geo)
    # visualizer.add_geometry(link2geo)

    # visualizer.get_view_control().set_front([1, 0, 0])
    # visualizer.get_view_control().set_up([0, 0, 1])
    # visualizer.run()
    # visualizer.destroy_window()

    # print("Smallest Dist:", smallestDist)
    # print("Smallest Angle:", smallestAngle)
    return smallestDist


def calcCylinderPoints2(poseAOnLine1, poseBOnLine1, radius1, poseAOnLine2, poseBOnLine2, radius2): # points1 & 2 are centreline endpoints of the cylinder
    pointAOnLine1 = [poseAOnLine1[0], poseAOnLine1[1], poseAOnLine1[2] ]
    pointBOnLine1 = [poseBOnLine1[0], poseBOnLine1[1], poseBOnLine1[2] ]
    pointAOnLine2 = [poseAOnLine2[0], poseAOnLine2[1], poseAOnLine2[2] ]
    pointBOnLine2 = [poseBOnLine2[0], poseBOnLine2[1], poseBOnLine2[2] ]


    dir_vector = [pointBOnLine1[0] - pointAOnLine1[0], pointBOnLine1[1] - pointAOnLine1[1], pointBOnLine1[2] - pointAOnLine1[2]]
    mag = math.sqrt(dir_vector[0]**2 + dir_vector[1]**2 + dir_vector[2]**2)
    norm_dir_vector = [dir_vector[0]/mag, dir_vector[1]/mag, dir_vector[2]/mag]
    x_dir = [1,0,0]
    y_dir = [0,1,0]
    z_dir = [0,0,1]

    # find a vector perpendicular to the centerline of the cylinder
    res = np.dot(x_dir,norm_dir_vector)
    comp = np.dot(res, norm_dir_vector)
    perp_vector = [x_dir[0] - comp[0], x_dir[1] - res*comp[1], x_dir[2] - res*comp[2]]
    dot_result = np.dot(norm_dir_vector, perp_vector)
    # perp_vector = [x_dir[0] - res*norm_dir_vector[0], x_dir[1] - res*norm_dir_vector[1], x_dir[2] - res*norm_dir_vector[2]]
    # print("\nCalc1: perpvector*dirvector", dot_result)
    # make sure dot product isn't close to 0
    if dot_result < -0.00000000001 or dot_result > 0.00000000001:
        res = np.dot(y_dir,norm_dir_vector)
        comp = np.dot(res, norm_dir_vector)
        perp_vector = [y_dir[0] - comp[0], y_dir[1] - comp[1], y_dir[2] - comp[2]]
        dot_result = np.dot(norm_dir_vector, perp_vector)
        # print("\nCalc2: perpvector*dirvector", dot_result)
        if dot_result < -0.00000000001 or dot_result > 0.00000000001:
            res = np.dot(z_dir,norm_dir_vector)
            perp_vector = [z_dir[0] - comp[0], z_dir[1] - comp[1], z_dir[2] - comp[2]]
            dot_result = np.dot(norm_dir_vector, perp_vector)
            # print("\nCalc2: perpvector*dirvector", dot_result)
    
    # if res > -0.3 and res < 0.3:
    #     res = np.dot(y_dir,norm_dir_vector)
    #     comp = np.dot(res, norm_dir_vector)
    #     perp_vector = [y_dir[0] - comp[0], y_dir[1] - comp[1], y_dir[2] - comp[2]]
    #     print("\nCalc2: perpvector*dirvector", np.dot(norm_dir_vector, perp_vector))
    #     if res > -0.3 and res < 0.3:
    #         res = np.dot(z_dir,norm_dir_vector)
    #         perp_vector = [z_dir[0] - comp[0], z_dir[1] - comp[1], z_dir[2] - comp[2]]
    #         print("\nCalc3: perpvector*dirvector", np.dot(norm_dir_vector, perp_vector))
    # print("\nNorm dir vector:", norm_dir_vector)
    # print("res:", res)
    # print("comp:", comp)
    # print("perpvector:",perp_vector)
    # print("\nPerpindicular test:")
    # first perpendicular unit vector on cylinder 1
    mag = math.sqrt(perp_vector[0]**2 + perp_vector[1]**2 + perp_vector[2]**2)
    norm_perp_vector = [perp_vector[0]/mag, perp_vector[1]/mag,  perp_vector[2]/mag]
    # print("\tperpvector*dirvector", np.dot(norm_dir_vector, norm_perp_vector))
    # calculate second perpendicular unit vector on cylinder 1
    norm_perp_vector2 = np.cross(norm_dir_vector, norm_perp_vector)
    dot_result = np.dot(norm_dir_vector, norm_perp_vector2)
    # print("\tperpvector2*dirvector", dot_result)
    # find end of cylinder 1 closest to cylinder 2:
    dista = calculateDistance(pointAOnLine1, pointAOnLine2, pointBOnLine2)
    distb = calculateDistance(pointBOnLine1, pointAOnLine2, pointBOnLine2)
    if dista <= distb:
        smallestDist = dista
        # print("\nside A is closer!", smallestDist)
    else:
        smallestDist = distb
        # print("\nside B is closer!", smallestDist)
    smallestAngle = None
    points = []
    for theta in range(0,360):
        if dista <= distb:
            newPoint_x = radius1*math.cos(d2r(theta))*norm_perp_vector[0] + radius1*math.sin(d2r(theta))*norm_perp_vector2[0] + pointAOnLine1[0]
            newPoint_y = radius1*math.cos(d2r(theta))*norm_perp_vector[1] + radius1*math.sin(d2r(theta))*norm_perp_vector2[1] + pointAOnLine1[1]
            newPoint_z = radius1*math.cos(d2r(theta))*norm_perp_vector[2] + radius1*math.sin(d2r(theta))*norm_perp_vector2[2] + pointAOnLine1[2]
            newPoint = [newPoint_x, newPoint_y, newPoint_z]
        else:
            newPoint_x = radius1*math.cos(d2r(theta))*norm_perp_vector[0] + radius1*math.sin(d2r(theta))*norm_perp_vector2[0] + pointBOnLine1[0]
            newPoint_y = radius1*math.cos(d2r(theta))*norm_perp_vector[1] + radius1*math.sin(d2r(theta))*norm_perp_vector2[1] + pointBOnLine1[1]
            newPoint_z = radius1*math.cos(d2r(theta))*norm_perp_vector[2] + radius1*math.sin(d2r(theta))*norm_perp_vector2[2] + pointBOnLine1[2]
            newPoint = [newPoint_x, newPoint_y, newPoint_z]
        newDist = calculateDistance(newPoint, pointAOnLine2, pointBOnLine2)
        points.append(newPoint)
        if newDist <= smallestDist:
            smallestDist = newDist
            smallestAngle = theta

    # create a cylinder to represent the robot base
    lengthCyl1 = math.sqrt((pointAOnLine1[0] - pointBOnLine1[0])**2 + (pointAOnLine1[1] - pointBOnLine1[1])**2 + (pointAOnLine1[2] - pointBOnLine1[2])**2)
    # print("LengthBase:"), lengthBase
    cylinder1 = o3d.geometry.TriangleMesh.create_cylinder(radius = radius1, height = lengthCyl1) #height= heightBase)
    rotMat = rotVec_to_rotMat(poseAOnLine1)
    cylinder1.rotate(rotMat, [0,0,0])
    cylinder1.translate([pointAOnLine1[0] + (pointBOnLine1[0] - pointAOnLine1[0] )/2, pointAOnLine1[1] + (pointBOnLine1[1] - pointAOnLine1[1])/2, pointAOnLine1[2] + (pointBOnLine1[2] - pointAOnLine1[2] )/2  ])
    # base_cylinder.translate([base_pose[0] + (shoulder_pose[0] - base_pose[0] )/2, base_pose[1] + (shoulder_pose[1] - base_pose[1])/2, base_pose[2] + (shoulder_pose[2] - base_pose[2] + 0.0893 )/2  ])
    # robot += base_cylinder
    
    link1geo = cylinder1

    # create a cylinder to represent the robot base
    lengthCyl2 = math.sqrt((pointAOnLine2[0] - pointBOnLine2[0])**2 + (pointAOnLine2[1] - pointBOnLine2[1])**2 + (pointAOnLine2[2] - pointBOnLine2[2])**2)
    # print("LengthBase:"), lengthBase
    cylinder2 = o3d.geometry.TriangleMesh.create_cylinder(radius = radius2, height = lengthCyl2) #height= heightBase)
    rotMat = rotVec_to_rotMat(poseAOnLine2)
    cylinder2.rotate(rotMat, [0,0,0])
    cylinder2.translate([pointAOnLine2[0] + (pointBOnLine2[0] - pointAOnLine2[0] )/2, pointAOnLine2[1] + (pointBOnLine2[1] - pointAOnLine2[1])/2, pointAOnLine2[2] + (pointBOnLine2[2] - pointAOnLine2[2] )/2  ])
    # base_cylinder.translate([base_pose[0] + (shoulder_pose[0] - base_pose[0] )/2, base_pose[1] + (shoulder_pose[1] - base_pose[1])/2, base_pose[2] + (shoulder_pose[2] - base_pose[2] + 0.0893 )/2  ])
    # robot += base_cylinder
    
    link2geo = cylinder2

    geometries = o3d.geometry.TriangleMesh()

    # smaller robot joints
    for point in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01) #create a small sphere to represent point
        
        # rotMat = rotVec_to_rotMat(poseAOnLine2)
        # sphere.rotate(rotMat, [0,0,0])
        sphere.translate(point) #translate this sphere to point
        geometries += sphere
    
    geometries.paint_uniform_color([1.0, 0.0, 0.0])
    link1geo.paint_uniform_color([15/255, 99/255, 175/255])
    link2geo.paint_uniform_color([10/255, 173/255, 254/255])

    # # o3d.visualization.draw_geometries([geometries, line_set])
    # visualizer = o3d.visualization.Visualizer() 
    # visualizer.create_window()
    # visualizer.get_render_option().show_coordinate_frame = True
    # visualizer.add_geometry(geometries)
    # visualizer.get_render_option().point_size = 1.5
    # # visualizer.add_geometry(box)
    # # visualizer.add_geometry(robot)
    # visualizer.add_geometry(link1geo)
    # visualizer.add_geometry(link2geo)

    # visualizer.get_view_control().set_front([1, 0, 0])
    # visualizer.get_view_control().set_up([0, 0, 1])
    # visualizer.run()
    # visualizer.destroy_window()

    # print("Smallest Dist:", smallestDist)
    # print("Smallest Angle:", smallestAngle)
    return geometries #smallestDist


# calculate the distance between a vector and a point given a point, a point on the vector and the vector
def calculateDistance(point, pointOnLink, link):
    # Calculate the distance between link2 (shoulder offset) and wrist3_position
    AP_vec = np.array([pointOnLink[0] - point[0], pointOnLink[1] - point[1], pointOnLink[2] - point[2] ])
    d_vec = link
    # print("\nAP_vec:", AP_vec)
    mag_d_vec = math.sqrt(d_vec[0]**2 + d_vec[1]**2 + d_vec[2]**2)
    res_vec= np.cross(AP_vec, d_vec)
    # print("res_vec", res_vec)
    mag_APxd = math.sqrt(res_vec[0]**2 + res_vec[1]**2 + res_vec[2]**2)
    # print("mag_APxd", mag_APxd)
    distance = mag_APxd/mag_d_vec
    return distance

def distance2points(p1, p2):
    line_length = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
    return line_length

def calculatePointPlaneDistance(p, a, b, c):
     # find normal to plane1:
    v1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    v2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]]
    # calculte normal to the plane
    norm = np.cross(v1, v2)
    # print("\nNorm:", norm)
    mag_norm = math.sqrt(norm[0]**2 + norm[1]**2 + norm[2]**2)
    # calculate unit vector of the norm
    # unit_norm = norm/mag_norm
    # calculte a vector from point to the plane
    pq = [p[0] - a[0], p[1] - a[1], p[2] - a[2]]
    distance = (np.dot(pq, norm))/mag_norm
    return distance

def calculateSegmentDistance(point, point1OnLine, point2OnLine):
    # print("Point: ", point )
    # print("Point1 on line: ", point1OnLine)
    # print("Point2 on line: ", point2OnLine)
    # Line equation is P = P1 + u(P2-P1)
    line_length = distance2points(point1OnLine, point2OnLine)
    # print("Line length: ", line_length)
    if line_length == 0:
        return distance2points(point, point1OnLine)
    t = ((point[0] - point1OnLine[0])*(point2OnLine[0] - point1OnLine[0]) + (point[1] - point1OnLine[1])*(point2OnLine[1] - point1OnLine[1])+ (point[2] - point1OnLine[2])*(point2OnLine[2] - point1OnLine[2]))/line_length
    # print("t:", t)
    if t <0:
        t = 0
    elif t > 1:
        t = 1
    
    pointOnLine = [point1OnLine[0]+ t*(point2OnLine[0] - point1OnLine[0]), point1OnLine[1]+ t*(point2OnLine[1] - point1OnLine[1]), point1OnLine[2]+ t*(point2OnLine[2] - point1OnLine[2])]
    distance = distance2points(point, pointOnLine)
    # distanceP_P1 = distance2points(pointOnLine, point1OnLine)
    # distanceP_P2 = distance2points(pointOnLine, point2OnLine)
    # distanceP1_P2 = distance2points(point1OnLine, point2OnLine)
    # print("\nDistance btwn calcP and P1:", distanceP_P1)
    # print("Distance btwn calcP and P2:", distanceP_P2)
    # print("Distance btwn P1 and P2:", distanceP1_P2)
    return distance


def orient(a, b, c):
    v1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
    v2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]]
    return np.cross(v1, v2)

def properInter(a,b,c,d):
    oa = orient(c,d,a)
    ob = orient(c,d,b)
    oc = orient(a,b,c)
    od = orient(a,b,d)
    # print("oa", oa)
    # print("ob", ob)
    # print("oc", oc)
    # print("od", od)

    if (np.dot(oa,ob) < 0) and (np.dot(oc,od) < 0):
        out = (np.dot(a,ob) - np.dot(b,oa))/(ob - oa)
        # print("Out:" , out)
        return True, out
    else:
        return False, None

def findSegmentIntersection(a,b,c,d):
    # check if colliniar
    res, data = properInter(a,b,c,d)
    if res == True:
        return 0
    else:
        r1 = calculateSegmentDistance(c,a,b)
        r2 = calculateSegmentDistance(d,a,b)
        r3 = calculateSegmentDistance(a,c,d)
        r4 = calculateSegmentDistance(b,c,d)
        # print("r1:", r1)
        # print("r2:", r2)
        # print("r3:", r3)
        # print("r4:", r4)

        return min([r1, r2, r3, r4])



# en
####
#### developed by Universal Robots Global Applications Management

def displayRobotPosition(jointPoses, collisionLinks):
    base_pose = jointPoses[0]
    shoulder_pose = jointPoses[1]
    elbow_pose = jointPoses[2]
    wrist1_pose = jointPoses[3]
    wrist2_pose = jointPoses[4]
    wrist3_pose = jointPoses[5]
    flange_pose = jointPoses[6]
    TCP_pose = jointPoses[7]
    TCP_offset = [0.0, 0.0, 0.02, 0.0, 0.0, 0.0]
    TCP_sphere_offset = [0.0, 0.0, 0.045, 0.0, 0.0, 0.0]
    TCP_pose_offset = PoseTrans(TCP_pose, PoseInv(TCP_offset))
    TCP_pose_sphere_offset = PoseTrans(TCP_pose, PoseInv(TCP_sphere_offset))

    lengthBase = 0.1807 #0.270
    lengthShoulder = 0.190 #0.335
    lengthShoulderElbow = 0.615
    lengthElbowOffset = 0.174
    lengthElbowWrist1 = 0.560
    lengthWrist1Wrist2 = 0.140
    lengthWrist2Wrist3 = 0.130
    lengthWrist3Flange = 0.130
    lengthGripper = 0.428
    # link 1: base to sholder
    link1_radius = 0.150/2 
    # link 2: shoulder offset
    link2_radius = 0.150/2 
    # link 3: shoulderOffset to elbowOffset
    link3_radius = 0.110/2 
    # link 4: elbowOffset to elbow
    link4_radius = 0.115/2
    # link 5: elbow to wrist 1
    link5_radius = 0.090/2
    # link 6: wrist 1 to wrist2
    # link 7: wrist 2 to wrist3
    # link 8: wrist 3 to flange
    link678_radius = 0.090/2
    # gripper: from flange to TCP
    gripper_radius = 0.090/2
    # saftey limit to avoid collisions: units in m
    safetyDelta = 0.025

    base_position = [base_pose[0], base_pose[1], base_pose[2]]
    shoulder_position = [shoulder_pose[0], shoulder_pose[1], shoulder_pose[2]]
    shoulder_elbow_vector = [shoulder_pose[0] - elbow_pose[0], shoulder_pose[1] - elbow_pose[1], shoulder_pose[2] - elbow_pose[2]]
    elbow_wrist1_vector = [wrist1_pose[0] - elbow_pose[0], wrist1_pose[1] - elbow_pose[1], wrist1_pose[2] - elbow_pose[2]]
    shoulder_elbow_vector = shoulder_elbow_vector / np.linalg.norm(shoulder_elbow_vector)
    elbow_wrist1_vector = elbow_wrist1_vector / np.linalg.norm(elbow_wrist1_vector)

    



    # print("\nshoulder vector", shoulder_elbow_vector)
    # print("Elbow vector", elbow_wrist1_vector)
    perp_vector = np.cross(shoulder_elbow_vector, elbow_wrist1_vector)
    mag_perp_vector = math.sqrt(perp_vector[0]**2 + perp_vector[1]**2  + perp_vector[2]**2  )
    perp_vector[0] = perp_vector[0]* (0.174/mag_perp_vector)
    perp_vector[1] = perp_vector[1]* (0.174/mag_perp_vector)
    perp_vector[2] = perp_vector[2]* (0.174/mag_perp_vector)

    # print("Perp vector:", perp_vector)
    shoulder_offset = [perp_vector[0] + shoulder_position[0], perp_vector[1] + shoulder_position[1], perp_vector[2] + shoulder_position[2] ]
    shoulder_offset_pose = [shoulder_offset[0], shoulder_offset[1], shoulder_offset[2], shoulder_pose[3], shoulder_pose[4] , shoulder_pose[4]]
    
    elbow_position = [elbow_pose[0], elbow_pose[1], elbow_pose[2]]
    elbow_offset = [perp_vector[0] + elbow_position[0], perp_vector[1] + elbow_position[1], perp_vector[2] + elbow_position[2] ]
    wrist1_position = [ wrist1_pose[0], wrist1_pose[1], wrist1_pose[2] ]
    wrist2_position = [ wrist2_pose[0], wrist2_pose[1], wrist2_pose[2] ]
    wrist3_position = [ wrist3_pose[0], wrist3_pose[1], wrist3_pose[2] ]
    flange_position = [ flange_pose[0], flange_pose[1], flange_pose[2] ]
    TCP_position = [TCP_pose[0], TCP_pose[1], TCP_pose[2]]
    TCP_sphere_position = [TCP_pose_sphere_offset[0], TCP_pose_sphere_offset[1], TCP_pose_sphere_offset[2]]
    # print("\nShoulder pose:\t", shoulder_pose)
    # print("Shoulder offset:", shoulder_offset)
    # print("Elbow offset:\t", elbow_offset)
    # print("Elbow:\t\t", elbow_pose)

    # calculate end points for each link:
    # Link 1: Base
    link1a = base_pose
    link1a_position = [link1a[0], link1a[1], link1a[2]]
    offset = [0, 0 , -(shoulder_pose[2] + link1_radius + 0.02), 0, 0, 0]
    link1b = PoseTrans(base_pose, PoseInv(offset))
    link1b_position = [link1b[0], link1b[1], link1b[2]]
    # print("\nlink1a position:", link1a)
    # print("link1b position:", link1b)

    # link 2: shoulder (horizontal)
    offset = [0, 0 , (link2_radius), 0, 0, 0]
    link2a = PoseTrans(shoulder_pose, PoseInv(offset))
    # link2a = shoulder_pose
    link2a_position = [link2a[0], link2a[1], link2a[2]]
    offset = [0, 0 , -(shoulder_pose[2] + link1_radius), 0, 0, 0]
    link2b = PoseTrans(shoulder_pose, PoseInv(offset))
    link2b_position = [link2b[0], link2b[1], link2b[2]]
    # print("\nlink2a position:", link2a)
    # print("link2b position:", link2b)

    # #link 5: elbow to wrist 1 - pose isn't given so need a different method to find the end points of link5:
    # r0 = shoulder_offset
    # r1 = elbow_offset
    # # print("\nr0:", r0)
    # # print("r1:", r1)
    # v = [r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]]
    # v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    # # v is now a unit vector
    # v_unit = [v[0]/v_mag, v[1]/v_mag, v[2]/v_mag]
    # # t value to calculate r1
    # t = (r1[0] - r0[0])/v_unit[0]
    # # t1 = (r1[1] - r0[1])/v_unit[1]
    # # t2 = (r1[2] - r0[2])/v_unit[2]
    # link3a = [0,0,0]
    # link3b = [0,0,0]
    # t1 = link2_radius
    # link3a[0] = r0[0] - (t1)*v_unit[0]
    # link3a[1] = r0[1] - (t1)*v_unit[1]
    # link3a[2] = r0[2] - (t1)*v_unit[2]
    # t2 = v_mag + link4_radius
    # link3b[0] = r0[0] + (t2)*v_unit[0]
    # link3b[1] = r0[1] + (t2)*v_unit[1]
    # link3b[2] = r0[2] + (t2)*v_unit[2]
    #  # offset = [0, 0 , (link5_radius), 0, 0, 0]
    # # link5a = PoseTrans(elbow_pose, PoseInv(offset))
    # link3a_position = [link3a[0], link3a[1], link3a[2]]
    # # offset = [0, 0 , -(link5_radius + lengthElbowOffset + link3_radius +  0.02), 0, 0, 0]
    # # link5b = PoseTrans(elbow_pose, PoseInv(offset))
    # link3b_position = [link3b[0], link3b[1], link3b[2]]
    # # print("\nlink3a position:", link3a)
    # # print("link3b position:", link3b)

    #link 5: elbow to wrist 1 - pose isn't given so need a different method to find the end points of link5:
    r0 = shoulder_offset
    r1 = elbow_offset
    # print("\nr0:", r0)
    # print("r1:", r1)
    v = [r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]]
    v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    # v is now a unit vector
    v_unit = [v[0]/v_mag, v[1]/v_mag, v[2]/v_mag]
    # t value to calculate r1
    t = (r1[0] - r0[0])/v_unit[0]
    # t1 = (r1[1] - r0[1])/v_unit[1]
    # t2 = (r1[2] - r0[2])/v_unit[2]
    link3a_position = [0,0,0]
    link3b_position = [0,0,0]
    link3c_position = [0,0,0]
    t1 = link2_radius
    link3a_position[0] = r0[0] - (t1)*v_unit[0]
    link3a_position[1] = r0[1] - (t1)*v_unit[1]
    link3a_position[2] = r0[2] - (t1)*v_unit[2]
    t2 = v_mag + link4_radius
    link3b_position[0] = r0[0] + (t2)*v_unit[0]
    link3b_position[1] = r0[1] + (t2)*v_unit[1]
    link3b_position[2] = r0[2] + (t2)*v_unit[2]
    t1 = -link2_radius
    link3c_position[0] = r0[0] - (t1)*v_unit[0]
    link3c_position[1] = r0[1] - (t1)*v_unit[1]
    link3c_position[2] = r0[2] - (t1)*v_unit[2]
     # offset = [0, 0 , (link5_radius), 0, 0, 0]
    # link5a = PoseTrans(elbow_pose, PoseInv(offset))
    link3a_position = [link3a_position[0], link3a_position[1], link3a_position[2]]
    # offset = [0, 0 , -(link5_radius + lengthElbowOffset + link3_radius +  0.02), 0, 0, 0]
    # link5b = PoseTrans(elbow_pose, PoseInv(offset))
    link3b_position = [link3b_position[0], link3b_position[1], link3b_position[2]]
    link3c_position = [link3c_position[0], link3c_position[1], link3c_position[2]]

    dist = math.sqrt((link3a_position[0] - link3b_position[0])**2 + (link3a_position[1] - link3b_position[1])**2 + (link3a_position[2] - link3b_position[2])**2)

    Rx_roll = r2d(math.pi + math.asin((link3a_position[0] - link3b_position[0])/dist))
    Ry_pitch = r2d(math.asin((link3a_position[1] - link3b_position[1])/dist))
    Ry_yaw = 0
    [Rx, Ry, Rz] = rpy2rotvec(Rx_roll, Ry_pitch, Ry_yaw)
    link3a = [link3a_position[0],link3a_position[1], link3a_position[2], Rx, Ry, Rz ]
    link3b = [link3b_position[0],link3b_position[1], link3b_position[2], Rx, Ry, Rz ]

    # link 4: elbow to elbow Offset
    offset = [0, 0 , -(link5_radius + lengthElbowOffset), 0, 0, 0]
    link4a = PoseTrans(elbow_pose, PoseInv(offset))
    link4a_position = [link4a[0], link4a[1], link4a[2]]
    offset = [0, 0 , (link3_radius), 0, 0, 0]
    link4b = PoseTrans(elbow_pose, PoseInv(offset))
    link4b_position = [link4b[0], link4b[1], link4b[2]]
    # print("\nlink3a position:", link4a)
    # print("link3b position:", link4b)

    # #link 5: elbow to wrist 1 - pose isn't given so need a different method to find the end points of link5:
    # r0 = elbow_position
    # r1 = wrist1_position
    # # print("\nr0:", r0)
    # # print("r1:", r1)
    # v = [r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]]
    # v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    # # v is now a unit vector
    # v_unit = [v[0]/v_mag, v[1]/v_mag, v[2]/v_mag]
    # # t value to calculate r1
    # t = (r1[0] - r0[0])/v_unit[0]
    # # t1 = (r1[1] - r0[1])/v_unit[1]
    # # t2 = (r1[2] - r0[2])/v_unit[2]
    # link5a = [0,0,0]
    # link5b = [0,0,0]
    # t1 = link678_radius
    # link5a[0] = r0[0] - (t1)*v_unit[0]
    # link5a[1] = r0[1] - (t1)*v_unit[1]
    # link5a[2] = r0[2] - (t1)*v_unit[2]
    # t2 = t + link678_radius
    # link5b[0] = r0[0] + (t2)*v_unit[0]
    # link5b[1] = r0[1] + (t2)*v_unit[1]
    # link5b[2] = r0[2] + (t2)*v_unit[2]

    # # offset = [0, 0 , (link5_radius), 0, 0, 0]
    # # link5a = PoseTrans(elbow_pose, PoseInv(offset))
    # link5a_position = [link5a[0], link5a[1], link5a[2]]
    # # offset = [0, 0 , -(link5_radius + lengthElbowOffset + link3_radius +  0.02), 0, 0, 0]
    # # link5b = PoseTrans(elbow_pose, PoseInv(offset))
    # link5b_position = [link5b[0], link5b[1], link5b[2]]
    # # print("\nlink5a position:", link5a)
    # # print("link5b position:", link5b)

     #link 5: elbow to wrist 1 - pose isn't given so need a different method to find the end points of link5:
    r0 = elbow_position
    r1 = wrist1_position
    # print("\nr0:", r0)
    # print("r1:", r1)
    v = [r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]]
    v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    # v is now a unit vector
    v_unit = [v[0]/v_mag, v[1]/v_mag, v[2]/v_mag]
    # t value to calculate r1
    t = (r1[0] - r0[0])/v_unit[0]
    # t1 = (r1[1] - r0[1])/v_unit[1]
    # t2 = (r1[2] - r0[2])/v_unit[2]
    link5a_position = [0,0,0]
    link5b_position = [0,0,0]
    t1 = link678_radius
    link5a_position[0] = r0[0] - (t1)*v_unit[0]
    link5a_position[1] = r0[1] - (t1)*v_unit[1]
    link5a_position[2] = r0[2] - (t1)*v_unit[2]
    t2 = t + link678_radius
    link5b_position[0] = r0[0] + (t2)*v_unit[0]
    link5b_position[1] = r0[1] + (t2)*v_unit[1]
    link5b_position[2] = r0[2] + (t2)*v_unit[2]

    # offset = [0, 0 , (link5_radius), 0, 0, 0]
    # link5a = PoseTrans(elbow_pose, PoseInv(offset))
    # link5a_position = [link5a[0], link5a[1], link5a[2]]
    # offset = [0, 0 , -(link5_radius + lengthElbowOffset + link3_radius +  0.02), 0, 0, 0]
    # link5b = PoseTrans(elbow_pose, PoseInv(offset))
    # link5b_position = [link5b[0], link5b[1], link5b[2]]
    # print("\nlink5a position:", link5a)
    # print("link5b position:", link5b)

    dist = math.sqrt((link5a_position[0] - link5b_position[0])**2 + (link5a_position[1] - link5b_position[1])**2 + (link5a_position[2] - link5b_position[2])**2)

    Rx_roll = r2d(math.pi + math.asin((link5a_position[0] - link5b_position[0])/dist))
    Ry_pitch = r2d(math.asin((link5a_position[1] - link5b_position[1])/dist))
    Ry_yaw = 0
    [Rx, Ry, Rz] = rpy2rotvec(Rx_roll, Ry_pitch, Ry_yaw)
    link5a = [link5a_position[0],link5a_position[1], link5a_position[2], Rx, Ry, Rz ]
    link5b = [link5b_position[0],link5b_position[1], link5b_position[2], Rx, Ry, Rz ]

    #link 6: wrist 1 to wrist 2
    offset = [0, 0 , (link678_radius), 0, 0, 0]
    link6a = PoseTrans(wrist1_pose, PoseInv(offset))
    link6a_position = [link6a[0], link6a[1], link6a[2]]
    offset = [0, 0 , -(link678_radius + lengthWrist1Wrist2  +  0.02), 0, 0, 0]
    link6b = PoseTrans(wrist1_pose, PoseInv(offset))
    link6b_position = [link6b[0], link6b[1], link6b[2]]
    # print("\nlink6a position:", link6a)
    # print("link6b position:", link6b)

    #link 7: wrist 2 to wrist 3
    offset = [0, 0 , (link678_radius), 0, 0, 0]
    link7a = PoseTrans(wrist2_pose, PoseInv(offset))
    link7a_position = [link7a[0], link7a[1], link7a[2]]
    offset = [0, 0 , -(link678_radius + lengthWrist2Wrist3), 0, 0, 0]
    link7b = PoseTrans(wrist2_pose, PoseInv(offset))
    link7b_position = [link7b[0], link7b[1], link7b[2]]
    # print("\nlink7a position:", link7a)
    # print("link7b position:", link7b)

    #link 7: wrist 3 to flange
    offset = [0, 0 , (link678_radius), 0, 0, 0]
    link8a = PoseTrans(wrist3_pose, PoseInv(offset))
    link8a_position = [link8a[0], link8a[1], link8a[2]]
    # offset = [0, 0 , -(link678_radius + lengthWrist2Wrist3), 0, 0, 0]
    # link8b = PoseTrans(wrist3_pose, PoseInv(offset))
    # link8b_position = [link8b[0], link8b[1], link8b[2]]
    link8b = flange_pose
    link8b_position = flange_position
    # print("\nlink8a position:", link8a)
    # print("link8b position:", link8b)

    grippera = flange_pose
    grippera_position = flange_position
    gripperb = TCP_pose_offset
    gripperb_position = [TCP_pose_offset[0], TCP_pose_offset[1], TCP_pose_offset[2]]

    # display the calculated points
    points = [base_position, shoulder_position, shoulder_offset, elbow_offset, elbow_position, wrist1_position, wrist2_position, wrist3_position, flange_position, TCP_sphere_position] #, TCP_position ]
    points2 = [base_position, shoulder_position, shoulder_offset, elbow_offset]
    points3 = [elbow_position, wrist1_position]
    points4 = [link1a_position, link1b_position, link2a_position, link2b_position, link3a_position, link3b_position,link4a_position, link4b_position, link5a_position, link5b_position, link6a_position, link6b_position , link7a_position, link7b_position, link8a_position, link8b_position]
    # points4 = [link1a_position, link1b_position, link2a_position, link2b_position, link4a_position, link4b_position, link6a_position, link6b_position , link7a_position, link7b_position, link8a_position, link8b_position]
    # points4 = [link8a_position, link8b_position]
    lines = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8], [8,9]]


    
    geometries = o3d.geometry.TriangleMesh()

    # # smaller robot joints
    # for point in points:
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.045) #create a small sphere to represent point
    #     sphere.translate(point) #translate this sphere to point
    #     geometries += sphere
    
    # # oversize joints
    # for point in points2:
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.075) #create a small sphere to represent point
    #     sphere.translate(point) #translate this sphere to point
    #     geometries += sphere

    # oversize joints
    # for point in points3:
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.055) #create a small sphere to represent point
    #     sphere.translate(point) #translate this sphere to point
    #     geometries += sphere
        # oversize joints
    # for point in points4:
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025) #create a small sphere to represent point
    #     sphere.translate(point) #translate this sphere to point
    #     newGeometries += sphere
    
    lengthBase = 0.1807 #0.270
    lengthShoulder = 0.190 #0.335
    lengthShoulderElbow = 0.615
    lengthElbowOffset = 0.174
    lengthElbowWrist1 = 0.560
    lengthWrist2Wrist3 = 0.130
    lengthWrist3Flange = 0.130
    lengthGripper = 0.428
    # base_radius = 0.150/2 
    # shoulderOffset_radius = 0.150/2 
    # shoulderElbow_radius = 0.110/2 
    # elbowOffset_radius = 0.120/2
    # elbowWrist1_radius = 0.090/2
    # wrist_radius = 0.090/2



    # # stand size is 0.260m x 0.260m x 0.320m high above table or 1.000m above floor
    # stand_width = 0.260
    # stand_depth = 0.260
    # stand_height = 1.0

    # standTopCorner1 = [stand_width/2, stand_depth/2, 0.0]
    # standTopCorner2 = [stand_width/2, -stand_depth/2, 0.0]
    # standTopCorner3 = [-stand_width/2, -stand_depth/2, 0.0]
    # standTopCorner4 = [-stand_width/2, stand_depth/2, 0.0]
    # standBotCorner1 = [stand_width/2, stand_depth/2, -1.0]
    # standBotCorner2 = [stand_width/2, -stand_depth/2, -1.0]
    # standBotCorner3 = [-stand_width/2, -stand_depth/2, -1.0]
    # standBotCorner4 = [-stand_width/2, stand_depth/2, -1.0]

    # # side A of stand goes from standTopCorner1, standTopCorner2, standBotCorner1, standBotCorner2
    # # find normal to plane1:
    # v1 = [standTopCorner1[0] - standTopCorner2[0], standTopCorner1[1] - standTopCorner2[1], standTopCorner1[2] - standTopCorner2[2]]
    # v2 = [standTopCorner1[0] - standBotCorner1[0], standTopCorner1[1] - standBotCorner1[1], standTopCorner1[2] - standBotCorner1[2]]
    # np1 = np.cross(v1, v2)
    
    newGeometries = o3d.geometry.TriangleMesh()
    stand = o3d.geometry.TriangleMesh()
    link1geo = o3d.geometry.TriangleMesh()
    link2geo = o3d.geometry.TriangleMesh()
    link3geo = o3d.geometry.TriangleMesh()
    link4geo = o3d.geometry.TriangleMesh()
    link5geo = o3d.geometry.TriangleMesh()
    link6geo = o3d.geometry.TriangleMesh()
    link7geo = o3d.geometry.TriangleMesh()
    link8geo = o3d.geometry.TriangleMesh()
    robot = o3d.geometry.TriangleMesh()
    # create a cylinder to represent the robot stand
    stand = o3d.geometry.TriangleMesh.create_cylinder(radius = 0.260/2, height = 1.0) #height= heightBase)
    stand.translate([0, 0, -1.0/2])
    # stand = o3d.geometry.TriangleMesh.create_box(width = 0.260, height = 0.260, depth = 1.0) #height= heightBase)
    # stand.translate([-0.260/2, -0.260/2, -1.0])
    # robot += stand

    # create a cylinder to represent the robot base
    lengthBase = math.sqrt((link1a[0] - link1b[0])**2 + (link1a[1] - link1b[1])**2 + (link1a[2] - link1b[2])**2)
    # print("LengthBase:"), lengthBase
    base_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius = link1_radius, height = lengthBase) #height= heightBase)
    base_cylinder.translate([link1a[0] + (link1b[0] - link1a[0] )/2, link1a[1] + (link1b[1] - link1a[1])/2, link1a[2] + (link1b[2] - link1a[2] )/2  ])
    # base_cylinder.translate([base_pose[0] + (shoulder_pose[0] - base_pose[0] )/2, base_pose[1] + (shoulder_pose[1] - base_pose[1])/2, base_pose[2] + (shoulder_pose[2] - base_pose[2] + 0.0893 )/2  ])
    # robot += base_cylinder
    link1geo = base_cylinder

    # create a cylinder to represent the robot shoulder
    lengthShoulder = math.sqrt((shoulder_pose[0] - link2b[0])**2 + (shoulder_pose[1] - link2b[1])**2 + (shoulder_pose[2] - link2b[2])**2)
    # print("\nlength shoulder:", lengthShoulder)
    shoulder_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius = link2_radius, height = lengthShoulder)
    rotMat = rotVec_to_rotMat(shoulder_pose)
    shoulder_cylinder.rotate(rotMat, [0,0,0])

    shoulder_cylinder.translate([link2b[0] + (shoulder_pose[0] - link2b[0] )/2, link2b[1] + (shoulder_pose[1] - link2b[1] )/2, link2b[2] + (shoulder_pose[2] - link2b[2] )/2 ])
    # robot += shoulder_cylinder
    link2geo = shoulder_cylinder

    # create a cylinder to represent the robot shoulder-elbow
    lengthShoulderElbow = math.sqrt((link3a[0] - link3b[0])**2 + (link3a[1] - link3b[1])**2 + (link3a[2] - link3b[2])**2)
    elbow_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius = link3_radius, height = lengthShoulderElbow)
    elbow_pose2 = PoseTrans(elbow_pose, [0,0,0,0, d2r(90), 0])
    rotMat = rotVec_to_rotMat(elbow_pose2)
    elbow_cylinder.rotate(rotMat, [0,0,0])
    elbow_cylinder.translate([link3a[0] + (link3b[0] - link3a[0] )/2, link3a[1] + (link3b[1] - link3a[1] )/2, link3a[2] + (link3b[2] - link3a[2] )/2 ])
    # robot += elbow_cylinder
    link3geo = elbow_cylinder

    # create a cylinder to represent the robot elbow offset
    lengthElbowOffset = math.sqrt((link4a[0] - link4b[0])**2 + (link4a[1] - link4b[1])**2 + (link4a[2] - link4b[2])**2)
    elbow_offset_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius = link4_radius, height = lengthElbowOffset)
    wrist1_pose2 = PoseTrans(wrist1_pose, [0,0,0,0, d2r(0), 0])
    rotMat = rotVec_to_rotMat(wrist1_pose2)
    elbow_offset_cylinder.rotate(rotMat, [0,0,0])
    elbow_offset_cylinder.translate([link4a[0] + (link4b[0] - link4a[0] )/2, link4a[1] + (link4b[1] - link4a[1] )/2, link4a[2] + (link4b[2] - link4a[2] )/2 ])
    # robot += elbow_offset_cylinder
    link4geo = elbow_offset_cylinder

    # create a cylinder to represent the robot elbow-wrist1
    lengthElbowWrist1 = math.sqrt((link5a[0] - link5b[0])**2 + (link5a[1] - link5b[1])**2 + (link5a[2] - link5b[2])**2)
    wrist1_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius = link5_radius, height = lengthElbowWrist1)
    wrist1_pose2 = PoseTrans(wrist1_pose, [0,0,0,0, d2r(90), 0])
    rotMat = rotVec_to_rotMat(wrist1_pose2)
    wrist1_cylinder.rotate(rotMat, [0,0,0])
    wrist1_cylinder.translate([link5a[0] + (link5b[0] - link5a[0] )/2, link5a[1] + (link5b[1] - link5a[1] )/2, link5a[2] + (link5b[2] - link5a[2] )/2 ])
    # robot += wrist1_cylinder
    link5geo = wrist1_cylinder
    
    # # create a cylinder to represent the robot wrist1-wrist2
    lengthElbowOffset = math.sqrt((link6a[0] - link6b[0])**2 + (link6a[1] - link6b[1])**2 + (link6a[2] - link6b[2])**2)
    wrist1_wrist2_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius = link678_radius, height = lengthElbowOffset)
    wrist2_pose2 = PoseTrans(wrist2_pose, [0,0,0,d2r(90), 0, 0])
    rotMat = rotVec_to_rotMat(wrist2_pose2)
    wrist1_wrist2_cylinder.rotate(rotMat, [0,0,0])
    wrist1_wrist2_cylinder.translate([link6a[0] + (link6b[0] - link6a[0]  )/2, link6a[1] + (link6b[1] - link6a[1] )/2, link6a[2] + (link6b[2] - link6a[2] )/2 ])
    # robot += wrist1_wrist2_cylinder
    link6geo = wrist1_wrist2_cylinder

    # create a cylinder to represent the robot wrist2-wrist3
    lengthWrist2Wrist3 = math.sqrt((link7a[0] - link7b[0])**2 + (link7a[1] - link7b[1])**2 + (link7a[2] - link7b[2])**2)
    wrist2_wrist3_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius = link678_radius, height = lengthWrist2Wrist3)
    wrist3_pose2 = PoseTrans(wrist3_pose, [0,0,0,d2r(90), 0, 0])
    rotMat = rotVec_to_rotMat(wrist3_pose2)
    wrist2_wrist3_cylinder.rotate(rotMat, [0,0,0])
    wrist2_wrist3_cylinder.translate([link7a[0] + (link7b[0] - link7a[0]  )/2, link7a[1] + (link7b[1] - link7a[1] )/2, link7a[2] + (link7b[2] - link7a[2] )/2 ])
    # robot += wrist2_wrist3_cylinder
    link7geo = wrist2_wrist3_cylinder

    # create a cylinder to represent the robot wrist3-Flange
    lengthWrist3Flange = math.sqrt((link8a[0] - link8b[0])**2 + (link8a[1] - link8b[1])**2 + (link8a[2] - link8b[2])**2)
    wrist3_flange_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius = link678_radius, height = lengthWrist3Flange)
    flange_pose2 = PoseTrans(flange_pose, [0,0,0,d2r(0), 0, 0])
    rotMat = rotVec_to_rotMat(flange_pose2)
    # TCP_pose2 = PoseTrans(TCP_pose, [0,0,0,d2r(0), 0, 0])
    # rotMat = rotVec_to_rotMat(TCP_pose2)
    wrist3_flange_cylinder.rotate(rotMat, [0,0,0])
    wrist3_flange_cylinder.translate([link8a[0] + (link8b[0] - link8a[0]  )/2, link8a[1] + (link8b[1] - link8a[1] )/2, link8a[2] + (link8b[2] - link8a[2] )/2 ])
    # robot += wrist3_flange_cylinder
    link8geo = wrist3_flange_cylinder

    # create a cylinder to represent the gripper
    gripper = o3d.geometry.TriangleMesh()
    gripper_cylinder = o3d.geometry.TriangleMesh()
    lengthGripper = math.sqrt((grippera[0] - gripperb[0])**2 + (grippera[1] - gripperb[1])**2 + (grippera[2] - gripperb[2])**2)
    gripper_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius = link678_radius, height = lengthGripper)
    TCP_pose2 = PoseTrans(TCP_pose, [0,0,0,d2r(0), 0, 0])
    rotMat = rotVec_to_rotMat(TCP_pose2)
    gripper_cylinder.rotate(rotMat, [0,0,0])
    gripper_cylinder.translate([grippera[0] + (gripperb[0] - grippera[0]  )/2, grippera[1] + (gripperb[1] - grippera[1]  )/2, grippera[2] + (gripperb[2] - grippera[2]  )/2 ])
    gripper += gripper_cylinder

    # create a cylinder to represent the gripper end
    gripperTCP = o3d.geometry.TriangleMesh()
    gripper_cylinder = o3d.geometry.TriangleMesh()
    gripper_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius = 0.0085, height = 0.04)
    TCP_pose2 = PoseTrans(TCP_pose, [0,0,0,d2r(0), 0, 0])
    rotMat = rotVec_to_rotMat(TCP_pose2)
    gripper_cylinder.rotate(rotMat, [0,0,0])
    # gripper_cylinder.translate([TCP_pose[0] + (flange_pose[0] - TCP_pose[0]  )/32, TCP_pose[1] + (flange_pose[0] - TCP_pose[1])/32, TCP_pose[2] + (flange_pose[0] - TCP_pose[2] )/32 ])
    gripper_cylinder.translate([TCP_pose_offset[0], TCP_pose_offset[1], TCP_pose_offset[2]])
    gripperTCP += gripper_cylinder

    # cylinder += o3d.geometry.TriangleMesh.create_cylinder(radius = 0.01, height= pose_2[2])
    # cylinder.translate(base_position)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # line_set.colors = o3d.utility.Vector3dVector([0,0,1])


    geometries.paint_uniform_color([1.0, 0.0, 0.0])
    newGeometries.paint_uniform_color([0.0, 0.0, 0.0])
    stand.paint_uniform_color([16/255, 152/255, 194/255])
    link1geo.paint_uniform_color([15/255, 99/255, 175/255])
    link2geo.paint_uniform_color([10/255, 173/255, 254/255])
    link3geo.paint_uniform_color([20/255, 207/255, 244/255])
    link4geo.paint_uniform_color([28/255, 186/255, 236/255])
    link5geo.paint_uniform_color([50/255, 193/255, 238/255])
    link6geo.paint_uniform_color([67/255, 198/255, 239/255])
    link7geo.paint_uniform_color([89/255, 205/255, 241/255])
    link8geo.paint_uniform_color([120/255, 214/255, 244/255])

    # robot.paint_uniform_color([0.0, 0.0, 1.0])
    gripper.paint_uniform_color([13/255, 117/255, 149/255])
    gripperTCP.paint_uniform_color([0.0, 0.0, 0.0])
    # if collisionLinks[0] != None:
    # for i in range(0, len(collisionLinks)):
        # links = collisionLinks[i]
    collisionColor = [255/255, 114/255, 27/255]
    for links in collisionLinks:
        if links == "Gripper":
            gripper.paint_uniform_color(collisionColor)
            gripperTCP.paint_uniform_color(collisionColor)
        elif links == "Link1":
            link1geo.paint_uniform_color(collisionColor)
        elif links == "Link2":
            link2geo.paint_uniform_color(collisionColor)
        elif links == "Link3":
            link3geo.paint_uniform_color(collisionColor)
        elif links == "Link4":
            link4geo.paint_uniform_color(collisionColor)
        elif links == "Link5":
            link5geo.paint_uniform_color(collisionColor)
        elif links == "Link6":
            link6geo.paint_uniform_color(collisionColor)
        elif links == "Link7":
            link7geo.paint_uniform_color(collisionColor)
        elif links == "Link8":
            link8geo.paint_uniform_color(collisionColor)

    
    # o3d.visualization.draw_geometries([geometries, line_set])
    visualizer = o3d.visualization.Visualizer() 
    visualizer.create_window()
    visualizer.get_render_option().show_coordinate_frame = True


    
    geos = calcCylinderPoints2(link8a, link8b, link678_radius, link1a, link1b, link1_radius)  
    geos += calcCylinderPoints2(link7a, link7b, link678_radius, link1a, link1b, link1_radius) 
    geos += calcCylinderPoints2(link6a, link6b, link678_radius, link1a, link1b, link1_radius) 
    geos += calcCylinderPoints2(link5a, link5b, link5_radius, link1a, link1b, link1_radius) 
    geos += calcCylinderPoints2(link4a, link4b, link4_radius, link1a, link1b, link1_radius) 
    geos += calcCylinderPoints2(link3a, link3b, link3_radius, link1a, link1b, link1_radius) 
    geos += calcCylinderPoints2(link2a, link2b, link2_radius, link1a, link1b, link1_radius) 
    visualizer.add_geometry(geometries)
    visualizer.add_geometry(geos)
    visualizer.add_geometry(newGeometries)
    visualizer.get_render_option().point_size = 1.5
    visualizer.add_geometry(line_set)
    # visualizer.add_geometry(box)
    # visualizer.add_geometry(robot)
    visualizer.add_geometry(stand)
    visualizer.add_geometry(link1geo)
    visualizer.add_geometry(link2geo)
    visualizer.add_geometry(link3geo)
    visualizer.add_geometry(link4geo)
    visualizer.add_geometry(link5geo)
    visualizer.add_geometry(link6geo)
    visualizer.add_geometry(link7geo)
    visualizer.add_geometry(link8geo)

    visualizer.add_geometry(gripper)
    visualizer.add_geometry(gripperTCP)
    visualizer.get_view_control().set_front([1, 0, 0])
    visualizer.get_view_control().set_up([0, 0, 1])
    visualizer.run()
    visualizer.destroy_window()

def calc_joint_angles(desired_TCP_pose, TCP_offset):
    # perfrom inverse kinematic calculations to find joint angles:

    flange_pose = PoseTrans(desired_TCP_pose, PoseInv(TCP_offset))
    # print("Flange pose: ", flange_pose)
    flange_affine = rotVec_to_rotMat_affine(flange_pose)
    jointAngles = invKine(flange_affine)
    # print("\nINV joint angles:")
    # print(jointAngles )
    # for i in range(0,len(jointAngles)):
        # print("[", jointAngles.item(i,0), ",", jointAngles.item(i,1), ",", jointAngles.item(i,2), ",", jointAngles.item(i,3), ",", jointAngles.item(i,4), ",", jointAngles.item(i,5), ",", jointAngles.item(i,6), ",", jointAngles.item(i,7), "]")
    calc_joint_angles = [jointAngles.item(0,5), jointAngles.item(1,5), jointAngles.item(2,5), jointAngles.item(3,5), jointAngles.item(4,5), jointAngles.item(5,5)]

    # print("\nActual joint angles:\t", theta)
    # print("\nCalculated joint angles:", calc_joint_angles)

    return calc_joint_angles

# ************************************************** FORWARD KINEMATICS

def AH( n,th,c  ):
    dh_parameter = get_dh_parameter(gen = 5, model = 10)
    a_pose = dh_parameter[0]
    d_pose = dh_parameter[1]
    alpha_pose = dh_parameter[2]
    T_a = np.matrix(np.identity(4), copy=False)
    T_a[0,3] = a_pose[n-1]
    T_d = np.matrix(np.identity(4), copy=False)
    T_d[2,3] = d_pose[n-1]

    Rzt = np.matrix([[math.cos(th[n-1,c]), -math.sin(th[n-1,c]), 0 ,0],
                [math.sin(th[n-1,c]),  math.cos(th[n-1,c]), 0, 0],
                [0,               0,              1, 0],
                [0,               0,              0, 1]],copy=False)
        

    Rxa = np.matrix([[1, 0,                 0,                  0],
                [0, math.cos(alpha_pose[n-1]), -math.sin(alpha_pose[n-1]),   0],
                [0, math.sin(alpha_pose[n-1]),  math.cos(alpha_pose[n-1]),   0],
                [0, 0,                 0,                  1]],copy=False)

    A_i = T_d * Rzt * T_a * Rxa
	    

    return A_i

# ************************************************** INVERSE KINEMATICS 

def invKine(desired_pos):# T60
    dh_parameter = get_dh_parameter(gen = 5, model = 10)
    a_pose = dh_parameter[0]
    d_pose = dh_parameter[1]
    alpha_pose = dh_parameter[2]
    mat = np.matrix
    th = mat(np.zeros((6, 8)))

    # P_05 = (desired_pos * mat([0,0, -d6, 1]).T-mat([0,0,0,1 ]).T)
    P_05 = (desired_pos * mat([0,0, -d_pose[5], 1]).T-mat([0,0,0,1 ]).T)

    # **** theta1 ****

    psi = math.atan2(P_05[2-1,0], P_05[1-1,0])
    phi = math.acos(d_pose[3] /math.sqrt(P_05[2-1,0]*P_05[2-1,0] + P_05[1-1,0]*P_05[1-1,0]))
    #The two solutions for theta1 correspond to the shoulder
    #being either left or right
    th[0, 0:4] = math.pi/2 + psi + phi
    th[0, 4:8] = math.pi/2 + psi - phi
    th = th.real

    # **** theta5 ****

    cl = [0, 4]# wrist up or down
    for i in range(0,len(cl)):
            c = cl[i]
            T_10 = np.linalg.inv(AH(1,th,c))
            T_16 = T_10 * desired_pos
            th[4, c:c+2] = + math.acos((T_16[2,3]-d_pose[3])/d_pose[5])
            th[4, c+2:c+4] = - math.acos((T_16[2,3]-d_pose[3])/d_pose[5])

    th = th.real

    # **** theta6 ****
    # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.

    cl = [0, 2, 4, 6]
    for i in range(0,len(cl)):
            c = cl[i]
            T_10 = np.linalg.inv(AH(1,th,c))
            T_16 = np.linalg.inv( T_10 * desired_pos )
            th[5, c:c+2] = math.atan2((-T_16[1,2]/math.sin(th[4, c])),(T_16[0,2]/math.sin(th[4, c])))
            
    th = th.real

    # **** theta3 ****
    cl = [0, 2, 4, 6]
    for i in range(0,len(cl)):
            c = cl[i]
            T_10 = np.linalg.inv(AH(1,th,c))
            T_65 = AH( 6,th,c)
            T_54 = AH( 5,th,c)
            T_14 = ( T_10 * desired_pos) * np.linalg.inv(T_54 * T_65)
            P_13 = T_14 * mat([0, -d_pose[3], 0, 1]).T - mat([0,0,0,1]).T
            t3 = cmath.acos((np.linalg.norm(P_13)**2 - a_pose[1]**2 - a_pose[2]**2 )/(2 * a_pose[1] * a_pose[2])) # norm ?
            th[2, c] = t3.real
            th[2, c+1] = -t3.real

    # **** theta2 and theta 4 ****

    cl = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(0,len(cl)):
            c = cl[i]
            T_10 = np.linalg.inv(AH( 1,th,c ))
            T_65 = np.linalg.inv(AH( 6,th,c))
            T_54 = np.linalg.inv(AH( 5,th,c))
            T_14 = (T_10 * desired_pos) * T_65 * T_54
            P_13 = T_14 * mat([0, -d_pose[3], 0, 1]).T - mat([0,0,0,1]).T
            
            # theta 2
            th[1, c] = -math.atan2(P_13[1], -P_13[0]) + math.asin(a_pose[2]* math.sin(th[2,c])/np.linalg.norm(P_13))
            # theta 4
            T_32 = np.linalg.inv(AH( 3,th,c))
            T_21 = np.linalg.inv(AH( 2,th,c))
            T_34 = T_32 * T_21 * T_14
            th[3, c] = math.atan2(T_34[1,0], T_34[0,0])
    th = th.real

    return th

def length(v):
    return math.sqrt(pow(v[0],2)+pow(v[1],2)+pow(v[2],2))

def norm(v):
    l=length(v)
    norm=[v[0]/l, v[1]/l, v[2]/l]
    return norm

def _polyscope(x,y,z):
    if ( (abs(x) >= 0.001 and x < 0.0) or (abs(x) < 0.001 and abs(y) >= 0.001 and y < 0.0) or (abs(x) < 0.001 and abs(y) < 0.001 and z < 0.0) ):
        scale = 1 - 2*math.pi / length([x,y,z])
        ret = [scale*x , scale*y , scale*z ]
        print("PolyScope SCALED value: ", ret)
        return ret
    else:
        ret = [x,y,z]
        print ("PolyScope value: ", ret)
        return ret

def polyscope(v):
    return _polyscope(v[3], v[4], v[5])

def main(args):
    # initPose = [0.9648439361362923, 0.03173050928400711, -0.023478384712796174, -2.048020391572826, 1.6573095883345166, 0.07659439768896442]
    initPose = [0.9648695516626178, 0.031763587024930706, -0.023409559736040364, -2.2318167486772045, 1.0969172048412485, 0.6670341708643618]
    # initPose = [0.9648640317459857, 0.031787688637914355, -0.023440022820671425, -2.355769189957532, 1.93134498201992, -0.03266587933722321]
    print("Initial Pose:", initPose)
    [rot_roll, rot_pitch, rot_yaw] = rotvec2rpy( initPose[3],  initPose[4],  initPose[5])
    # if rot_yaw < 0:
    #     rot_yaw = rot_yaw + 360
    # print("\nRotated Gripper Pose:", rotatedVec)
    print("\tRx:", rot_roll)
    print("\tRy:",  rot_pitch)
    print("\tRz:", rot_yaw, "\n")

    if rot_roll < -100:
        deg = -(180 + rot_roll)
    elif rot_roll > 100:
        deg = -(180 - rot_roll)
    nextPose = [0.0, 0.0, 0.0, d2r(deg), d2r(rot_pitch), d2r(0)]
    firstPose = PoseTrans(initPose, nextPose)
    # firstPose = PoseAdd(initPose, nextPose)
    
    # secondPose = PoseAdd(firstPose, nextPose)
    print("\nFirst pose:", firstPose)
    [rot_roll, rot_pitch, rot_yaw] = rotvec2rpy( firstPose[3],  firstPose[4],  firstPose[5])
    # if rot_yaw < 0:
    #     rot_yaw = rot_yaw + 360
    # print("\nRotated Gripper Pose:", rotatedVec)
    print("\tRx:", rot_roll)
    print("\tRy:",  rot_pitch)
    print("\tRz:", rot_yaw, "\n")

    if rot_yaw < 0:
        deg = rot_yaw
    else:
        deg = -rot_yaw

    nextPose = [0.0, 0.0, 0.0, d2r(0), d2r(0), d2r(deg)]
    secondPose = PoseTrans(firstPose, nextPose)

    print("\nSecond pose:", secondPose)
    [rot_roll, rot_pitch, rot_yaw] = rotvec2rpy( secondPose[3],  secondPose[4],  secondPose[5])
    # if rot_yaw < 0:
    #     rot_yaw = rot_yaw + 360
    # print("\nRotated Gripper Pose:", rotatedVec)
    print("\tRx:", rot_roll)
    print("\tRy:",  rot_pitch)
    print("\tRz:", rot_yaw, "\n")
    
    if rot_pitch > 1 or rot_pitch < -1:
        nextPose = [0.0, 0.0, 0.0, d2r(0), d2r(rot_pitch), d2r(0)]
        lastPose = PoseTrans(secondPose, nextPose)
        [rot_roll, rot_pitch, rot_yaw] = rotvec2rpy( lastPose[3],  lastPose[4],  lastPose[5])
        # if rot_yaw < 0:
        #     rot_yaw = rot_yaw + 360
        # print("\nRotated Gripper Pose:", rotatedVec)
        print("\nThrid pose:", lastPose)
        print("\tRx:", rot_roll)
        print("\tRy:",  rot_pitch)
        print("\tRz:", rot_yaw, "\n")

    if rot_roll > -178.5 and rot_roll < 178.5:
        if rot_roll < -100:
            deg = 180 + rot_roll
        elif rot_roll > 100:
            deg = 180 - rot_roll
        print("Deg:", deg)
        nextPose = [0.0, 0.0, 0.0, d2r(deg), d2r(0), d2r(0)]
        lastPose = PoseTrans(lastPose, nextPose)
        [rot_roll, rot_pitch, rot_yaw] = rotvec2rpy( lastPose[3],  lastPose[4],  lastPose[5])
        # if rot_yaw < 0:
        #     rot_yaw = rot_yaw + 360
        # print("\nRotated Gripper Pose:", rotatedVec)
        print("\nThrid pose:", lastPose)
        print("\tRx:", rot_roll)
        print("\tRy:",  rot_pitch)
        print("\tRz:", rot_yaw, "\n")

    if rot_yaw > 1 or rot_yaw < -1:
        if rot_yaw < 0:
            deg = rot_yaw
        else:
            deg = -rot_yaw
        nextPose = [0.0, 0.0, 0.0, d2r(0), d2r(0), d2r(deg)]
        lastPose = PoseTrans(lastPose, nextPose)
        [rot_roll, rot_pitch, rot_yaw] = rotvec2rpy( lastPose[3],  lastPose[4],  lastPose[5])
        # if rot_yaw < 0:
        #     rot_yaw = rot_yaw + 360
        # print("\nRotated Gripper Pose:", rotatedVec)
        print("\nThrid pose:", lastPose)
        print("\tRx:", rot_roll)
        print("\tRy:",  rot_pitch)
        print("\tRz:", rot_yaw, "\n")




if __name__ == "__main__":
    import sys

    main(sys.argv[1:])