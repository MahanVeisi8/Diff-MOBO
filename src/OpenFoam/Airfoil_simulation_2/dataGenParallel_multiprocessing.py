################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Generate training data via OpenFOAM
#
################

import os, math, uuid, sys, random
import numpy as np
import utils
import multiprocessing

samples = 4  # no. of datasets to produce
freestream_angle = math.pi / 8.  # -angle ... angle
freestream_length = 10.  # len * (1. ... factor)
freestream_length_factor = 10.  # length factor

# airfoil_database  = "./airfoil_database/"
# os.chdir("/home/foam_airfoil/data/")
# airfoil_database = "/home/foam_airfoil/data/My_airfoil/"
# output_dir = "/home/foam_airfoil/data/train/"

seed = random.randint(0, 2 ** 32 - 1)
np.random.seed(seed)
print("Seed: {}".format(seed))

utils.makeDirs(["./data_pictures", "./train", "./OpenFOAM/constant/polyMesh/sets", "./OpenFOAM/constant/polyMesh"])
xs_train_fname = '/home/foam_airfoil/data_ahmad/xs_test.npy'
airfoil_sample_train = np.load(xs_train_fname)
print(airfoil_sample_train.shape[0])
# Test
# airfoil_sample_train = airfoil_sample_train[1:3,:,:]

def genMesh(ar):
    # ar = np.loadtxt(airfoilFile, skiprows=1)

    # removing duplicate end point
    if np.max(np.abs(ar[0] - ar[(ar.shape[0] - 1)])) < 1e-6:
        ar = ar[:-1]

    output = ""
    pointIndex = 1000
    for n in range(ar.shape[0]):
        output += "Point({}) = {{ {}, {}, 0.00000000, 0.005}};\n".format(pointIndex, ar[n][0], ar[n][1])
        pointIndex += 1

    with open("airfoil_template.geo", "rt") as inFile:
        with open("airfoil.geo", "wt") as outFile:
            for line in inFile:
                line = line.replace("POINTS", "{}".format(output))
                line = line.replace("LAST_POINT_INDEX", "{}".format(pointIndex - 1))
                outFile.write(line)

    if os.system("/home/Download/gmsh.info/bin/Linux/gmsh-3.0.6-Linux64/bin/gmsh airfoil.geo -3 -o airfoil.msh > /dev/null") != 0:
        print("error during mesh creation!")
        return (-1)

    if os.system("/opt/openfoam5/platforms/linux64GccDPInt32Opt/bin/gmshToFoam airfoil.msh > /dev/null") != 0:
        print("error during conversion to OpenFoam mesh!")
        return (-1)

    with open("constant/polyMesh/boundary", "rt") as inFile:
        with open("constant/polyMesh/boundaryTemp", "wt") as outFile:
            inBlock = False
            inAerofoil = False
            for line in inFile:
                if "front" in line or "back" in line:
                    inBlock = True
                elif "aerofoil" in line:
                    inAerofoil = True
                if inBlock and "type" in line:
                    line = line.replace("patch", "empty")
                    inBlock = False
                if inAerofoil and "type" in line:
                    line = line.replace("patch", "wall")
                    inAerofoil = False
                outFile.write(line)
    os.rename("constant/polyMesh/boundaryTemp", "constant/polyMesh/boundary")

    return (0)


def runSim(freestreamX, freestreamY):
    with open("U_template", "rt") as inFile:
        with open("0/U", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", "{}".format(freestreamX))
                line = line.replace("VEL_Y", "{}".format(freestreamY))
                outFile.write(line)
    # To avoid permission denied error
    os.system("sudo chmod +x Allclean")
    os.system("./Allclean && /opt/openfoam5/platforms/linux64GccDPInt32Opt/bin/simpleFoam > foam.log")
    return (0)




def readClCd():

    res = np.loadtxt('./postProcessing/forceCoeffs1/0/forceCoeffs_1.dat', skiprows=508)
    if len(res) == 6:
        CL = res[2]
        CD = res[3]
    else:
        CL = -np.inf
        CD = np.inf
    return CL, CD


# files = os.listdir(airfoil_database)
# files.sort()
# if len(files) == 0:
#     print("error - no airfoils found in %s" % airfoil_database)
#     exit(1)




# main
cl_all= []
cd_all = []
inputs = range(samples) 
def calculate_cd_cl(n, num_cores):
    # For each core assign TotalSampls/NumberOfCores
    devision_factor = np.int(np.ceil(airfoil_sample_train.shape[0]/num_cores))
    # print(devision_factor)
    for sample_num in range(n*devision_factor,(n+1)*devision_factor):
        print("core {}:".format(n))
        print("sample {}:".format(sample_num))

        length = 40
        angle = 0
        fsX = math.cos(angle) * length
        fsY = -math.sin(angle) * length

        print("\tUsing len %5.3f angle %+5.3f " % (length, angle))
        print("\tResulting freestream vel x,y: {},{}".format(fsX, fsY))

        os.chdir("./OpenFOAM_%d/" %(n))
        if genMesh(airfoil_sample_train[sample_num, :, :]) != 0:
            print("\tmesh generation failed, aborting")
            CL = -np.inf
            CD = np.inf
            os.chdir("..")
            a_file = open("results/%d.txt" %(sample_num), "w")
            np.savetxt(a_file, np.array([CL, CD]), fmt="%f")
            a_file.close()
            print(CL)
            print(CD)
            continue
        if runSim(fsX, fsY) != 0:
            print("\simulation failed, aborting")
            CL = -np.inf
            CD = np.inf
        else:
            os.chdir("..")
            os.chdir("./OpenFOAM_%d/" %(n))
            CL, CD = readClCd()
            
        
        os.chdir("..")
        a_file = open("results/%d.txt" %(sample_num), "w")
        np.savetxt(a_file, np.array([CL, CD]), fmt="%f")
        a_file.close()
        print(CL)
        print(CD)

    

if __name__ == '__main__':
    jobs = []
    num_cores = multiprocessing.cpu_count()
    print(num_cores)
    
    #for n in range(0,255):
    for n in range(0,255):
        p = multiprocessing.Process(target=calculate_cd_cl, args=(n,num_cores,))
        jobs.append(p)
        p.start()
