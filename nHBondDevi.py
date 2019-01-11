import numpy as np
import operations as op
import math as ma
import sys

#define the formula to calculate the number of hydrogen bonds
def countHBond(r0,dij):
    return (r0/dij)**6;

def countFD(r0,B,dij):
    return 2/(ma.exp((dij-r0)/B) + 1);

#import the coordinates
def findDeviation(name):
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

    #find the center of C-N
    CNCenter = np.zeros((64,3));
    for i in range(64):
        temp = np.zeros((3,2));
        op.gothrough(n[i],c,temp);
        CNCenter[i] = 1/2*(n[i] + c[int(temp[0,0])]);


    #Find the 4 molecules that surround the Cl atoms. Take N atoms as indicator. Store the index
    cl_n = np.zeros((24,4));

    for i in range(24):
        temp = np.zeros((4,2));
        op.gothrough(original_cl[i],CNCenter,temp);
        if temp[-1,1] > 6:
            print("There is something wrong my friend.")
            sys.exit();
        cl_n[i,:] = temp[:,0];

    #Assume the ideal ideal hydrogen bond number is 1 for each Cl atom
    NumHBond = np.zeros((24,1));
    r0 = 2.2;
    B = 0.3;
    for i in range(24):
        for j in range(4):
            tempH = np.zeros((3,2));
            indexN = int(cl_n[i,j]);
            op.gothrough(n[indexN],h,tempH);
            for k in range(3):
                dij = np.linalg.norm(original_cl[i,:]-h[int(tempH[k,0])]);
                NumHBond[i,0] += countFD(r0,B,dij);

    deviation = np.zeros((8,1));
    deviation = np.square(NumHBond - 1);
    totDevi = np.sum(deviation);
    return totDevi;

#Starting of the main function
totDevi = np.zeros((200,1));
for i in range(200):
    name = 'rand_' + str(i+51) +'.xsf';
    totDevi[i,0] = findDeviation(name);
    print(totDevi[i,0]);
