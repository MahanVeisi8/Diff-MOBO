################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Generate training data via OpenFOAM
#
################

import os, math, uuid, sys, random
import numpy as np
import Airfoil_simulation.utils
import multiprocessing
import os, os.path

# airfoil_database  = "./airfoil_database/"
# os.chdir("/home/foam_airfoil/data/")
# airfoil_database = "/home/foam_airfoil/data/My_airfoil/"
# output_dir = "/home/foam_airfoil/data/train/"

# seed = random.randint(0, 2 ** 32 - 1)
# np.random.seed(seed)
# print("Seed: {}".format(seed))
#
# utils.makeDirs(["./data_pictures", "./train", "./OpenFOAM/constant/polyMesh/sets", "./OpenFOAM/constant/polyMesh"])
# xs_train_fname = '/home/foam_airfoil/data_ahmad/xs_test.npy'
# airfoil_sample_train = np.load(xs_train_fname)
# print(airfoil_sample_train.shape[0])
samples = 4  # no. of datasets to produce
freestream_angle = math.pi / 8.  # -angle ... angle
freestream_length = 10.  # len * (1. ... factor)
freestream_length_factor = 10.  # length factor

# Test
# airfoil_sample_train = airfoil_sample_train[1:3,:,:]

def genMesh(ar, n):
    # os.chdir("./Airfoil_simulation/OpenFOAM_%d/" % (n))
    # ar = np.loadtxt(airfoilFile, skiprows=1)

    # removing duplicate end point
    operating_folder = "./Airfoil_simulation/OpenFOAM_%d/" % (n)
    if np.max(np.abs(ar[0] - ar[(ar.shape[0] - 1)])) < 1e-6:
        ar = ar[:-1]

    output = ""
    pointIndex = 1000
    for n in range(ar.shape[0]):
        output += "Point({}) = {{ {}, {}, 0.00000000, 0.005}};\n".format(pointIndex, ar[n][0], ar[n][1])
        pointIndex += 1

    with open(operating_folder + "airfoil_template.geo", "rt") as inFile:
        with open(operating_folder + "airfoil.geo", "wt") as outFile:
            for line in inFile:
                line = line.replace("POINTS", "{}".format(output))
                line = line.replace("LAST_POINT_INDEX", "{}".format(pointIndex - 1))
                outFile.write(line)

    system_command = "/home/Download/gmsh.info/bin/Linux/gmsh-3.0.6-Linux64/bin/gmsh airfoil.geo -3 -o " + operating_folder + "airfoil.msh > /dev/null"
    if os.system(system_command) != 0:
        print("error during mesh creation!")
        return (-1)

    system_command = "/opt/openfoam5/platforms/linux64GccDPInt32Opt/bin/gmshToFoam" + operating_folder + "airfoil.msh > /dev/null"
    if os.system("/opt/openfoam5/platforms/linux64GccDPInt32Opt/bin/gmshToFoam airfoil.msh > /dev/null") != 0:
        print("error during conversion to OpenFoam mesh!")
        return (-1)

    with open(operating_folder + "constant/polyMesh/boundary", "rt") as inFile:
        with open(operating_folder+ "constant/polyMesh/boundaryTemp", "wt") as outFile:
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
    os.rename(operating_folder + "constant/polyMesh/boundaryTemp", operating_folder + "constant/polyMesh/boundary")

    return (0)


def runSim(freestreamX, freestreamY, n):
    operating_folder = "./Airfoil_simulation/OpenFOAM_%d/" % (n)
    with open(operating_folder + "U_template", "rt") as inFile:
        with open(operating_folder + "0/U", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", "{}".format(freestreamX))
                line = line.replace("VEL_Y", "{}".format(freestreamY))
                outFile.write(line)
    # To avoid permission denied error
    system_command = "sudo chmod +x " + operating_folder + "Allclean"
    os.system("sudo chmod +x Allclean")
    system_command = "./Allclean"
    os.system("./Allclean && /opt/openfoam5/platforms/linux64GccDPInt32Opt/bin/simpleFoam > foam.log")
    return (0)


def readClCd(n):
    operating_folder = "./Airfoil_simulation/OpenFOAM_%d/" % (n)
    res = np.loadtxt(operating_folder + 'postProcessing/forceCoeffs1/0/forceCoeffs_1.dat', skiprows=508)
    if len(res) == 6:
        CL = res[3]
        CD = res[2]
    else:
        CL = -1000
        CD = 1000
    return CL, CD


# files = os.listdir(airfoil_database)
# files.sort()
# if len(files) == 0:
#     print("error - no airfoils found in %s" % airfoil_database)
#     exit(1)


# main
cl_all = []
cd_all = []
inputs = range(samples)


def calculate_cd_cl(n, num_cores, airfoil_sample_train):
    # For each core assign TotalSampls/NumberOfCores
    devision_factor = np.int(np.floor(airfoil_sample_train.shape[0] / num_cores))
    # print(devision_factor)
    for sample_num in range(n * devision_factor, (n + 1) * devision_factor):
        print("core {}:".format(n))
        print("sample {}:".format(sample_num))

        length = 40
        angle = 0
        fsX = math.cos(angle) * length
        fsY = -math.sin(angle) * length

        print("\tUsing len %5.3f angle %+5.3f " % (length, angle))
        print("\tResulting freestream vel x,y: {},{}".format(fsX, fsY))

        # os.chdir("./Airfoil_simulation/OpenFOAM_%d/" % (n))
        operating_folder = "./Airfoil_simulation/OpenFOAM_%d/" % (n)
        if genMesh(airfoil_sample_train[sample_num, :, :], n) != 0:

            print("\tmesh generation failed, aborting")
            CL = -1000
            CD = 1000
            # os.chdir("../..")
            a_file = open("Airfoil_simulation/results/%d.txt" % (sample_num), "w")
            np.savetxt(a_file, np.array([CL, CD]), fmt="%f")
            a_file.close()
            print(CL)
            print(CD)
            continue
        if runSim(fsX, fsY, n) != 0:
            print("\simulation failed, aborting")
            CL = -1000
            CD = 1000
            # os.chdir("../..")
            a_file = open("./Airfoil_simulation/results/%d.txt" % (sample_num), "w")
            np.savetxt(a_file, np.array([CL, CD]), fmt="%f")
            a_file.close()
            print(CL)
            print(CD)
            continue
        else:
            # os.chdir("../..")
            CL, CD = readClCd(n)

        # os.chdir("..")
        a_file = open("./Airfoil_simulation/" + "results/%d.txt" % (sample_num), "w")
        np.savetxt(a_file, np.array([CL, CD]), fmt="%f")
        a_file.close()
        print(CL)
        print(CD)
        


def calculate_cd_cl_res(n, sample_num, airfoil_sample_train):
    # For each core assign TotalSampls/NumberOfCores
    # print(devision_factor)
    # for sample_num in range(n * devision_factor, (n + 1) * devision_factor):
    print("core {}:".format(n))
    print("sample {}:".format(sample_num))

    length = 40
    angle = 0
    fsX = math.cos(angle) * length
    fsY = -math.sin(angle) * length

    print("\tUsing len %5.3f angle %+5.3f " % (length, angle))
    print("\tResulting freestream vel x,y: {},{}".format(fsX, fsY))

    # os.chdir("./Airfoil_simulation/OpenFOAM_%d/" % (n))
    operating_folder = "./Airfoil_simulation/OpenFOAM_%d/" % (n)
    if genMesh(airfoil_sample_train[sample_num, :, :], n) != 0:

        print("\tmesh generation failed, aborting")
        CL = -1000
        CD = 1000
        # os.chdir("../..")
        a_file = open("Airfoil_simulation/results/%d.txt" % (sample_num), "w")
        np.savetxt(a_file, np.array([CL, CD]), fmt="%f")
        a_file.close()
        print(CL)
        print(CD)
        
    if runSim(fsX, fsY, n) != 0:
        print("\simulation failed, aborting")
        CL = -1000
        CD = 1000
        # os.chdir("../..")
        a_file = open("./Airfoil_simulation/results/%d.txt" % (sample_num), "w")
        np.savetxt(a_file, np.array([CL, CD]), fmt="%f")
        a_file.close()
        print(CL)
        print(CD)
        
    else:
        # os.chdir("../..")
        CL, CD = readClCd(n)

    # os.chdir("..")
    a_file = open("./Airfoil_simulation/" + "results/%d.txt" % (sample_num), "w")
    np.savetxt(a_file, np.array([CL, CD]), fmt="%f")
    a_file.close()
    print(CL)
    print(CD)
        



def shape_to_performance(airfoil_sample_train):
    jobs = []
    num_cores = multiprocessing.cpu_count()
    print(num_cores)
    devision_factor = np.int(np.floor(airfoil_sample_train.shape[0] / num_cores))
    processes = []
    # for n in range(0,255):
    if airfoil_sample_train.shape[0] > num_cores:
        for n in range(0, num_cores):
            p = multiprocessing.Process(target=calculate_cd_cl, args=(n, num_cores, airfoil_sample_train))
            jobs.append(p)
            p.start()
            processes.append(p)
    for p in processes:
        p.join()
        
    # handels either the data that are less than the number of cores or handels what has been left of the batch.
    if devision_factor*num_cores < airfoil_sample_train.shape[0]:
        for n in range(0, airfoil_sample_train[devision_factor*num_cores:,:,:].shape[0]):
            p = multiprocessing.Process(target=calculate_cd_cl_res, args=(n, devision_factor*num_cores + n, airfoil_sample_train))
            jobs.append(p)
            p.start()
            processes.append(p)
        
    for p in processes:
        p.join()

    Data_path = "./Airfoil_simulation/results/"
    num_files = len([name for name in os.listdir(Data_path) if os.path.isfile(os.path.join(Data_path, name))])
    cd = []
    cl = []

    for i in range(num_files):
        print(i)
        with open(Data_path + '%d.txt' % (i)) as f:
            cl = np.append(cl, float(next(f).split()[0]))
            cd = np.append(cd, float(next(f).split()[0]))


    cl = np.array(cl)
    cd = np.array(cd)
    # savemat("dataset_Cd_Cl_test.mat", {"cl": cl, "cd": cd})

    final = np.append(np.expand_dims(cl, axis=1), np.expand_dims(cd, axis=1), axis=1)
    return final


# shape_to_performance(np.random.rand(1,192))