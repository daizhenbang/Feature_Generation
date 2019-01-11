import numpy as np
import operations as op
import math as ma

def euler(index_pb,original_pb,cl):
    pb = original_pb[int(index_pb),:]; 
    temp_cl = np.zeros((6,2));
    the_cl = np.zeros((6,3));
    op.gothrough(pb,cl,temp_cl);
    for i in range(6):
        the_cl[i,:] = cl[int(temp_cl[i][0]),:];
    
    #Determine the new x,y,z principal axis
    axis_index = np.full((3,2),-1);#x-x, y-y, z-z
    max_xyz = np.zeros((3,1));
    for i in range(6):
        for j in range(i,6,1):
            for k in range(3):
                distance = abs(the_cl[i][k] - the_cl[j][k]);
                if distance > max_xyz[k]:
                    max_xyz[k] = distance;
                    axis_index[k][0] = i;
                    axis_index[k][1] = j;
    
    principal = np.zeros((3,3)); #x,y,z new principal axis
    for i in range(3):
        principal[i,:] = the_cl[int(axis_index[i][0])] - the_cl[int(axis_index[i][1])];
        if principal[i][i] < 0:
            principal[i,:] = -1*principal[i,:];
        
        principal[i,:] = principal[i,:]/np.linalg.norm(principal[i,:]);

    ##===z followed by y followed by x===####
    alpha = ma.asin(-principal[2][1]/ma.sqrt(1-principal[2][0]**2));
    beta = ma.asin(principal[2][0]);
    gamma = ma.asin(-principal[1][0]/ma.sqrt(1-principal[2][0]**2));

    
    return alpha*180/ma.pi, beta*180/ma.pi, gamma*180/ma.pi;

if __name__ == '__main__':
    for i in range(51,251,1):
        name = 'rand_' + str(i) + '.xsf';
        
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
        
        #According to our own system, we can just choose the original Pb atoms
        #We take the Pb atom that has the least x,y,z coordinate as our central Pb atoms, i.e. the reference octahedron
        re_pb = np.zeros((8,3));
        op.rearrange_pb(original_pb,re_pb);#reference Pb is the first Pb atom
        
        EightSets = np.zeros((8,3));
        
        for i in range(8):
            EightSets[i,:] = euler(i,re_pb,cl);
        
#        print(np.sum(np.square(EightSets)));

        print(np.sum(EightSets));