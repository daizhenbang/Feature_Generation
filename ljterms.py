import numpy as np
import operations as op
import math as ma

#define the formula to calculate the number of hydrogen bonds
def countHBond(r0,dij):
    return (r0/dij)**6;

def countFD(r0,B,dij):
    return 2/(ma.exp((dij-r0)/B) + 1);

def count6LJ(dij):
    return 1/(dij**6);

def count12LJ(dij):
    return 1/(dij**12);


def findNHBond(name):
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
    #            print(indicator1)
    #            print(indicator2)
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
                index_c[count] = temp_cn[0][0];
                for k in range(12):
                    the_cl[count][k] = cl[int(temp_o_cl[k][0])];
                for k in range(8):
                    the_pb[count][k] = pb[int(temp_o_pb[k][0])];

                count = count + 1;

    #rearrange the chlorine atoms
    for i in range(8):
        op.rearrange(the_cl[i]);


    #After spliting into small cell, let's calculate the hydrogen bonds length.
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
            
            

    #Now starting to calculate the LJ terms
    #For all Cl atoms in one small cell
    HCl_6LJ = np.zeros((8,1));
    HCl_12LJ = np.zeros((8,1));
    
    CHCl_6LJ = np.zeros((8,1));
    CHCl_12LJ = np.zeros((8,1));
    
    NCl_6LJ = np.zeros((8,1));
    NCl_12LJ = np.zeros((8,1));
    
    CCl_6LJ = np.zeros((8,1));
    CCl_12LJ = np.zeros((8,1));
    
    for i in range(8):
        for j in range(12):
            dij_NCl = np.linalg.norm(the_n[i] - the_cl[i,j]);
            dij_CCl = np.linalg.norm(the_c[i] - the_cl[i,j]);
            NCl_6LJ[i,0] += count6LJ(dij_NCl);
            NCl_12LJ[i,0] += count12LJ(dij_NCl);
            CCl_6LJ[i,0] += count6LJ(dij_CCl);
            CCl_12LJ[i,0] += count12LJ(dij_CCl);
        
        for j in range(3):
            for k in range(12):
                dij = np.linalg.norm(the_nh[i][j] - the_cl[i][k]);
                dij_CH = np.linalg.norm(the_ch[i][j] - the_cl[i][k]);
                
                HCl_6LJ[i,0] += count6LJ(dij);
                HCl_12LJ[i,0] += count12LJ(dij);
                CHCl_6LJ[i,0] += count6LJ(dij_CH);
                CHCl_12LJ[i,0] += count12LJ(dij_CH);
                
    #For the nearest 3 Cl atoms for H
#    for i in range(8):
#        for j in range(12):
#            dij_NCl = np.linalg.norm(the_n[i] - the_cl[i,j]);
#            NCl_6LJ[i,0] += count6LJ(dij_NCl);
#            NCl_12LJ[i,0] += count12LJ(dij_NCl);
#        
#        for j in range(3):
#            tempCl = np.zeros((3,2));
#            op.gothrough(the_nh[i,j],the_cl[i],tempCl);
#            for k in range(3):
#                dij = tempCl[k,1];
#                HCl_6LJ[i,0] += count6LJ(dij);
#                HCl_12LJ[i,0] += count12LJ(dij);
                
    return np.sum(HCl_6LJ), np.sum(HCl_12LJ), np.sum(CHCl_6LJ),\
    np.sum(CHCl_12LJ),np.sum(NCl_6LJ), np.sum(NCl_12LJ),np.sum(CCl_6LJ), np.sum(CCl_12LJ);

#Start of the main function
LJ_terms = np.zeros((200,8)); #6HCl, 12HCl, 6CHCl, 12CHCl,6LJ, 12LJ
for i in range(200):
    name = 'rand_' + str(i+51) + '.xsf';
    LJ_terms[i,:] = findNHBond(name);
    print(str(LJ_terms[i,0]) + ' ' + str(LJ_terms[i,1]) + ' ' +str(LJ_terms[i,2]) + ' '\
          +str(LJ_terms[i,3])+' ' +str(LJ_terms[i,4]) + ' ' +str(LJ_terms[i,5]) + ' '\
          ' ' +str(LJ_terms[i,6]) + ' ' +str(LJ_terms[i,7]));

#    print(str(LJ_terms[i,0]) + ' ' +str(LJ_terms[i,2]) + ' '+str(LJ_terms[i,4]));
    
#    print(str(LJ_terms[i,1]) + ' ' +str(LJ_terms[i,3]) + ' '+str(LJ_terms[i,5]));
