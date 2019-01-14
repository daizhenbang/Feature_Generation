import numpy as np
import operations as op
import matplotlib.pyplot as plt

def determineAxis(vector):
    axis = 0;
    sign = 0;
    if abs(vector[0]) > abs(vector[1]):
        if abs(vector[0]) > abs(vector[2]):
            sign = vector[0];
            axis = 1;
        else:
            sign = vector[2];
            axis = 3;
    else:
        if abs(vector[1]) > abs(vector[2]):
            sign = vector[1];
            axis = 2;
        else:
            sign = vector[2];
            axis = 3;
    
    if sign < 0:
        axis = -1*axis;
    
    return axis;

def findNeighbor(coords,lattice):
    original_n = coords;#Just being lazy
    
    NN = np.zeros((24,));#Store the distance
    tempNN = np.zeros((48,1));#because of the double counting
    count = 0;
    for i in range(8):
        tempN = np.zeros((4,2));
        axis = np.zeros((3,1));
        op.gothrough(original_n[i,:],original_n,tempN);
        for j in range(3):
            tempNN[count] = tempN[j+1,1];
            count += 1;
            nnVec = original_n[int(tempN[j+1,0]),:] - original_n[i,:];
            axis[j] = determineAxis(nnVec);
            if axis[j] > 0:
                anotherN = original_n[int(tempN[j+1,0]),:] - lattice[int(axis[j]-1),:];
                tempNN[count] = np.linalg.norm(anotherN - original_n[i,:]);
                count += 1;
            else:
                anotherN = original_n[int(tempN[j+1,0]),:] + lattice[-int(axis[j]+1),:];
                tempNN[count] = np.linalg.norm(anotherN - original_n[i,:]);
                count += 1;
    
    '''Remove Double Counting'''
    count = 0;
    NN[0] = tempNN[0];
    for i in range(48):
        count2 = 0;
        for j in range(count):
            if tempNN[i] == NN[j]:
                count2 += 1;
                break;
        
        if count2 == 0:
            NN[count] = tempNN[i];
            count += 1;
            if count == 24:
                break;
        else:
            continue;
    
    return NN;

def findChargeCenter(name):
    '''Atoms and Charge should be one-to-one correspondance'''
    #import the coordinates
    lattice = np.zeros((3,3));
    original_pb = np.zeros((8,3));
    original_cl = np.zeros((24,3));
    original_c = np.zeros((8,3));
    original_n = np.zeros((8,3));
    original_h = np.zeros((48,3));

    op.in_coord(name, lattice, original_pb, original_cl, original_c, original_n, original_h);
    #enlarge the unit cell by 8 times
    pb = np.zeros((64,3));
    cl = np.zeros((192,3));
    c = np.zeros((64,3));
    n = np.zeros((64,3));
    h = np.zeros((384,3));

    op.enlarge(original_pb,pb,lattice,8);
    op.enlarge(original_cl,cl,lattice,24);
    op.enlarge(original_c,c,lattice,8);
    op.enlarge(original_n,n,lattice,8);
    op.enlarge(original_h,h,lattice,48);

    the_n = np.zeros((8,3));
    index_n = np.zeros((8,1));
    the_c = np.zeros((8,3));
    index_c = np.zeros((8,1));
    the_cl = np.zeros((8,12,3));
    the_pb = np.zeros((8,8,3));

    distance_o_cl = np.zeros((8,12));
    distance_h_cl = np.zeros((8,3));
    
#    mapToOrigin = np.zeros((8,2));# Tell which original molecule they actually correspond to
#    mapToOrigin[:,0] = np.arange(0,8,1);
    mapToOrigin = np.zeros((8,1));
    #first get one small cell
    count = 0;
    for i in range(64):
        temp_n = n[i];
        temp_center = np.zeros((3,1));
        temp_cn = np.zeros((64,2));
        op.gothrough(temp_n,c,temp_cn);
        temp_c = c[int(temp_cn[0][0])];

        op.midpoint(temp_c,temp_n,temp_center);#determine the center coordinate
        #determine the neighbor Cl atoms
        temp_o_cl = np.zeros((12,2));
        op.gothrough(temp_center, cl, temp_o_cl);
        #determine the neighbor Pb atoms
        temp_o_pb = np.zeros((8,2));
        op.gothrough(temp_center,pb,temp_o_pb);
        if temp_o_cl[11][1] > 5.5:
            continue;
        else:
            #determine the H(N) atoms in this unit cell
            temp_h_n = np.zeros((3,2));
            op.gothrough(temp_n, h, temp_h_n);
            temp_nh_cl = np.zeros((3,2));
            for j in range(3):
                op.gothrough(h[int(temp_h_n[j][0])],cl,temp_nh_cl[j,:]);

            #decide whether we choose this unit cell or not, trying to avoid overlap
            count2 = 0;
            for j in range(count+1):
                indicator1 = np.linalg.norm(temp_o_cl[:,1] - distance_o_cl[j,:]);
                indicator2 = np.linalg.norm(temp_nh_cl[:,1] - distance_h_cl[j,:]);
                if indicator1 > 0.0000001 and indicator2 > 0.0000001:
                    count2 = count2 + 1;
                    continue;
                else:
                    break;

            if count2 == count+1:
                distance_o_cl[count,:] = temp_o_cl[:,1];
                distance_h_cl[count,:] = temp_nh_cl[:,1];

                the_n[count] = temp_n;
                the_c[count] = temp_c;
                index_n[count] = i;
                mapToOrigin[count] = i%8;
                index_c[count] = temp_cn[0][0];
                for k in range(12):
                    the_cl[count][k] = cl[int(temp_o_cl[k][0])];
                for k in range(8):
                    the_pb[count][k] = pb[int(temp_o_pb[k][0])];

                count = count + 1;
    
    #rearrange the chlorine atoms
    for i in range(8):
        op.rearrange(the_cl[i]);


    #Find the hydrogen atoms in each of the cell
    the_nh = np.zeros((8,3,3));
    the_ch = np.zeros((8,3,3));
    for i in range(8):
        temp_h_n = np.zeros((3,2));
        temp_h_c = np.zeros((3,2));
        op.gothrough(the_n[i],h,temp_h_n);
        op.gothrough(the_c[i],h,temp_h_c);
        for j in range(3):
            the_nh[i][j] = h[int(temp_h_n[j][0])];
            the_ch[i][j] = h[int(temp_h_c[j][0])];
    
    '''Charges. Could be replaced with Bader Charge'''
    qCl = -1;
    qPb = 2;
    qC = -2;
    qN = -3;
    qH = 1;
    
    orgoCenter = np.zeros((8,3));
    inorgoCenter = np.zeros((8,3));
    for i in range(8):
        tempCenter = np.zeros((1,3));
        for j in range(8):
            tempCenter += abs(qPb)*the_pb[i][j][:];
        for j in range(12):
            tempCenter += abs(qCl)*the_cl[i][j][:];
        inorgoCenter[i,:] = tempCenter/(8*abs(qPb)+12*abs(qCl));
        pullBackFactor = np.round(np.matmul(np.linalg.inv(lattice), n[int(index_n[i])] - n[int(mapToOrigin[i])]));
        inorgoCenter[i,:] = inorgoCenter[i,:] - np.matmul(lattice,pullBackFactor);

    for i in range(8):
        tempCenter = np.zeros((1,3));
        for j in range(3):
            tempCenter += abs(qH)*the_nh[i][j][:];
        for j in range(3):
            tempCenter += abs(qH)*the_ch[i][j][:];
        tempCenter += abs(qC)*the_c[i,:] + abs(qN)*the_n[i,:];
        
        orgoCenter[i,:] = tempCenter/(6*abs(qH)+abs(qC)+abs(qN));
        pullBackFactor = np.round(np.matmul(np.linalg.inv(lattice), n[int(index_n[i])] - n[int(mapToOrigin[i])]));
        orgoCenter[i,:] = orgoCenter[i,:] - np.matmul(lattice,pullBackFactor);

    
    
    mapToOrigin = np.argsort(mapToOrigin,axis=0);

    orgo = orgoCenter[np.int64(mapToOrigin[:,0]).reshape([8,1]),[0,1,2]];
    inorgo = inorgoCenter[np.int64(mapToOrigin[:,0]).reshape([8,1]),[0,1,2]];
    
    return orgo,inorgo;

def findNN(name):
    #import the coordinates
    lattice = np.zeros((3,3));
    original_pb = np.zeros((8,3));
    original_cl = np.zeros((24,3));
    original_c = np.zeros((8,3));
    original_n = np.zeros((8,3));
    original_h = np.zeros((48,3));

    op.in_coord(name, lattice, original_pb, original_cl, original_c, original_n, original_h);
    
    '''For the current cell, find it's neighbors'''
    NN = np.zeros((24,));#Store the distance
    NN = findNeighbor(original_n,lattice);
    
    return NN,original_n;

def dipoleCoupling(allDipoles):
    numDipoles = len(allDipoles[:,0]);#The dipoles are stored as nx3
    coupling = 0;
    for i in range(numDipoles):
        tempNeigh = np.zeros((4,2));
        op.gothrough(allDipoles[0,:],allDipoles,tempNeigh);
        for j in range(3):
            coupling += np.dot(allDipoles[0,:],allDipoles[int(tempNeigh[j+1,0])]);
        
    return coupling/24;


def findMolecules(name):
   #import the coordinates
    lattice = np.zeros((3,3));
    original_pb = np.zeros((8,3));
    original_cl = np.zeros((24,3));
    original_c = np.zeros((8,3));
    original_n = np.zeros((8,3));
    original_h = np.zeros((48,3));

    op.in_coord(name, lattice, original_pb, original_cl, original_c, original_n, original_h);
    
    molecules = np.zeros((8,3));
    for i in range(8):
        tempN = np.zeros((3,2));
        op.gothrough(original_c[i,:],original_n,tempN);
        molecules[i] = original_n[int(tempN[0,0]),:] - original_c[i,:];
    
    return molecules;
    
    
if __name__ == '__main__':
    dft = np.zeros((200,1));
    dftFile = open('dft.txt','r');
    temp = dftFile.readlines();
    dftFile.close();
    
    for i in range(len(temp)):
        dft[i] = float(temp[i])*13.6*1000;
    dft = dft - np.min(dft);
    
    lattice = np.zeros((3,3));
    lattice[0][0] = 11.2820;
    lattice[1][1] = 11.1747;
    lattice[2][2] = 11.3552;
    
    highSym = np.zeros((8,3));
    count = 0;
    
    
    NN = np.zeros((200,24));
    aveNN = np.zeros((200,1));
    orgo = np.zeros((200,8,3));
    inorgo = np.zeros((200,8,3));
    dipoleOI = np.zeros((200,8,3));
    dipoleNI = np.zeros((288,8,3));
    dipoleNHighSymCouple = np.zeros((200,8,3));
    molecules = np.zeros((200,8,3));
    
    
    OIcouple = np.zeros((200,1));
    moleCouple = np.zeros((200,1));
    NIcouple = np.zeros((200,1));
    NHighSymCouple = np.zeros((200,1));
    
    for i in range(200):
        name = 'rand_' + str(i+51) +'.xsf';
        NN[i,:],original_N = findNN(name);
        #Find the nearest high symmetry position
        for j in range(8):
            tempHighSym = np.zeros((3,2));
            op.gothrough(original_N[j,:],highSym,tempHighSym);
            dipoleNHighSymCouple[i,j,:] = original_N - highSym[int(tempHighSym[0,0])];
        
#        aveNN[i] = np.mean(NN[i,:]);
#        orgo[i], inorgo[i] =  findChargeCenter(name);
        molecules[i] = findMolecules(name);
#        dipoleNI[i] = orignal_N - inorgo[i];
#        dipoleOI[i] = orgo[i] - inorgo[i];
#        OIcouple[i] = dipoleCoupling(dipoleOI[i]);
        moleCouple[i] = dipoleCoupling(molecules[i]);
#        NIcouple[i] = dipoleCoupling(dipoleNI[i]);
        
        print(NIcouple[i,0])
    
#    flattenNN = np.zeros((200,24));
#    flattenNN = NN;
#    flattenNN = NN.flatten();
#    plt.scatter(dft,aveNN);
#    plt.hist(aveNN,30)
    
