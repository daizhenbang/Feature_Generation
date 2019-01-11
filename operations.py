import numpy as np
import math as ma
import itertools
import cmath
import os


def in_coord(name, lattice, original_pb, original_cl, original_c, original_n, original_h): #Lattice is given since it's 'relax' calculation
    
    coord_file = open(name,'r');
    tempLines = coord_file.readlines();
    coord_file.close();
    
    index = 0;
    for i in range(len(tempLines)):
        if tempLines[i].startswith('PRIMCOORD'):
            index = i+2;
            break;
    
    lines = tempLines[index:index+96];
    
    lattice[0][0] = 11.2820;
    lattice[1][1] = 11.1747;
    lattice[2][2] = 11.3552;
            
    for i in range(8):
        temp1 = lines[i].split();
        temp2 = lines[i+32].split();
        temp3 = lines[i+40].split();
        for j in range(3):
            original_pb[i][j] = temp1[j+1];
            original_c[i][j] = temp2[j+1];
            original_n[i][j] =  temp3[j+1];
    
    for i in range(24):
        temp4 = lines[i+8].split();
        for j in range(3):
            original_cl[i][j] = temp4[j+1];
    
    for i in range(48):
        temp5 = lines[i+48].split();
        for j in range(3):
            original_h[i][j] = temp5[j+1];
    

def enlarge(original, enlarged, lattice, quantity):
    #this function can enlarge the unit cell by twice according to the lattice parameter
    #'quantity' here means the number of atoms that in the original cell
    for i in range(quantity):
        for j in range(3):
            enlarged[i][j] = original[i][j];
    for i in range(quantity):
        for j in range(3):
            enlarged[i+quantity*1][j] = enlarged[i][j] + lattice[0][j];#enlarge in x direction
    for i in range(quantity):
        for j in range(3):
            enlarged[i+quantity*2][j] = enlarged[i][j] + lattice[1][j];
            enlarged[i+quantity*3][j] = enlarged[i+quantity*1][j] + lattice[1][j];#the last two lines enlarge in y direction
    for i in range(quantity):
        for j in range(3):
            enlarged[i+quantity*4][j] = enlarged[i][j] + lattice[2][j];
            enlarged[i+quantity*5][j] = enlarged[i+quantity*1][j] + lattice[2][j];
            enlarged[i+quantity*6][j] = enlarged[i+quantity*2][j] + lattice[2][j];
            enlarged[i+quantity*7][j] = enlarged[i+quantity*3][j] + lattice[2][j];#The last four lines enlarge in z direction

def rearrange(the_cl):
    front = np.zeros((4,3));
    middle = np.zeros((4,3));
    back = np.zeros((4,3));
    original = the_cl;
    new = original[original[:,0].argsort()];
    for i in range(4):
        front[i] = new[i+8];
        middle[i] = new[i+4];
        back[i] = new[i];
    
    front = front[front[:,1].argsort()];
    middle = middle[middle[:,1].argsort()];
    back = back[back[:,1].argsort()];
    
    if front[1][2] > front[2][2]:
        for j in range(3):
            temp = front[2][j];
            front[2][j] = front[1][j];
            front[1][j] = temp;


    if back[1][2] > back[2][2]:
        for j in range(3):
            temp = back[2][j];
            back[2][j] = back[1][j];
            back[1][j] = temp;    
    
    if middle[0][2] < middle[1][2]:
        for j in range(3):  
            temp = middle[1][j];
            middle[1][j] = middle[0][j];
            middle[0][j] = temp;

    if middle[2][2] > middle[3][2]:
        for j in range(3):     
            temp = middle[3][j];
            middle[3][j] = middle[2][j];
            middle[2][j] = temp;
    
    for i in range(4):
        the_cl[i] = front[i];
        the_cl[i+4] = middle[i];
        the_cl[i+8] = back[i];
        
def rearrange_pb(pb, final):
    front = np.zeros((4,3));
    back = np.zeros((4,3));
    
    original = pb;
    new = original[original[:,0].argsort()];
    
    for i in range(4):
        front[i] = new[i+4];
        back[i] = new[i];
    
    front = front[front[:,1].argsort()];
    back = back[back[:,1].argsort()];
    
    for i in range(2):
        if front[2*i][2] > front[2*i+1][2]:
            for j in range(3):
                temp = front[2*i+1][j];
                front[2*i+1][j] = front[2*i][j];
                front[2*i][j] = temp;
    
    for i in range(2):
        if back[2*i][2] > back[2*i+1][2]:
            for j in range(3):
                temp = back[2*i+1][j];
                back[2*i+1][j] = back[2*i][j];
                back[2*i][j] = temp;

    final[0] = back[0];
    final[1] = front[0];
    final[2] = back[2];
    final[3] = back[1];
    final[4] = front[2];
    final[5] = back[3];
    final[6] = front[1];
    final[7] = front[3];
    
def gothrough(central_coord, target, temp_length):
    #this function can go through all the atoms specified around a given central atom and give the distances 
    #between the central atoms and the neighbor atoms and their corresponding indices.(It has to be larger than 3)
    #The number of distances is determined by the length of the array 'temp_length'
    size = len(target[:,0]);
    size2 = len(temp_length);
    if size2 == 2:
        size2 = 1;
    length = np.zeros((size,2));
    for i in range(size):
        temp_distance = 0;
        for j in range(3):
            temp_distance = temp_distance + (central_coord[j] - target[i][j])**2;
        length[i][0] = i;
        length[i][1] = ma.sqrt((temp_distance));
    length = length[length[:,1].argsort()];
    if size2 == 1:
        temp_length[0] = length[i][0];
        temp_length[1] = length[i][1];
    else:
        for i in range(size2):
            temp_length[i] = length[i];

def twodatoms(central_coord,target,plane,atoms):
    size = len(target);
    if plane == 'bc':
        count = 0;
        for i in range(size):
            temp_a = abs(target[i][0] - central_coord[0]);
            if temp_a < 1:
                atoms[count] = i;
                count = count + 1;
    elif plane == 'ca':
        count = 0;
        for i in range(size):
            temp_a = abs(target[i][1] - central_coord[1]);
            if temp_a < 1:
                atoms[count] = i;
                count = count + 1;
    elif plane == 'ab':
        count = 0;
        for i in range(size):
            temp_a = abs(target[i][2] - central_coord[2]);
            if temp_a < 1:
                atoms[count] = i;
                count = count + 1;

def midcl(pb1,pb2,cl):      
    #This function will give a Cl atom that is in the middle of two given neighbor Pb atoms          
    dist = np.zeros((192,2));
    for i in range(192):
        dist1 = (cl[i][0]-pb1[0])**2 + (cl[i][1]-pb1[1])**2 + (cl[i][2]-pb1[2])**2;
        dist2 = (cl[i][0]-pb2[0])**2 + (cl[i][1]-pb2[1])**2 + (cl[i][2]-pb2[2])**2;
        dist1 = ma.sqrt(dist1);
        dist2 = ma.sqrt(dist2);
        dist[i][0] = i;
        dist[i][1] = dist1 + dist2;
    
    dist = dist[dist[:,1].argsort()];
    mid_cl = int(dist[0][0]);
    return mid_cl;

def neighbor_pb(central_pb,all_pb,index_of_pb, neigh_pb, plane):
    #This function can find the four Pb neigbors of a Pb atom and put them in a counterclock way
    dist = np.zeros((16,2));
    for i in range(16):
        index = int(index_of_pb[i]);
        dist[i][0] = index;
        dist[i][1] = (all_pb[index][0]-central_pb[0])**2 + (all_pb[index][1]-central_pb[1])**2 + (all_pb[index][2]-central_pb[2])**2;
        dist[i][1] = ma.sqrt(dist[i][1]);
    
    dist = dist[dist[:,1].argsort()];
    
    if plane == 'bc':
        coord_b = np.zeros((4,2));
        coord_c = np.zeros((4,2));
        for i in range(4):
            coord_b[i][0] = dist[i+1][0];
            coord_b[i][1] = all_pb[int(dist[i+1][0])][1];
            coord_c[i][0] = dist[i+1][0];
            coord_c[i][1] = all_pb[int(dist[i+1][0])][2]; 
        
        coord_b = coord_b[coord_b[:,1].argsort()];
        coord_c = coord_c[coord_c[:,1].argsort()];
        neigh_pb[0] = coord_b[3][0];
        neigh_pb[2] = coord_b[0][0];
        neigh_pb[1] = coord_c[3][0];
        neigh_pb[3] = coord_c[0][0];
    elif plane == 'ca':
        coord_a = np.zeros((4,2));
        coord_c = np.zeros((4,2));
        for i in range(4):
            coord_a[i][0] = dist[i+1][0];
            coord_a[i][1] = all_pb[int(dist[i+1][0])][0];
            coord_c[i][0] = dist[i+1][0];
            coord_c[i][1] = all_pb[int(dist[i+1][0])][2]; 
        
        coord_a = coord_a[coord_a[:,1].argsort()];
        coord_c = coord_c[coord_c[:,1].argsort()];
        neigh_pb[0] = coord_c[3][0];
        neigh_pb[2] = coord_c[0][0];
        neigh_pb[1] = coord_a[3][0];
        neigh_pb[3] = coord_a[0][0];
    elif plane == 'ab':
        coord_a = np.zeros((4,2));
        coord_b = np.zeros((4,2));
        for i in range(4):
            coord_a[i][0] = dist[i+1][0];
            coord_a[i][1] = all_pb[int(dist[i+1][0])][0];
            coord_b[i][0] = dist[i+1][0];
            coord_b[i][1] = all_pb[int(dist[i+1][0])][1]; 
        
        coord_a = coord_a[coord_a[:,1].argsort()];
        coord_b = coord_b[coord_b[:,1].argsort()];
        neigh_pb[0] = coord_a[3][0];
        neigh_pb[2] = coord_a[0][0];
        neigh_pb[1] = coord_b[3][0];
        neigh_pb[3] = coord_b[0][0];

def midpoint(pb1,pb2,midpb):
    for i in range(3):
        midpb[i] = 0.5*(pb1[i] + pb2[i]);


def distortion(plane,pb1,midpb,midcl):
    vector1 = np.zeros((3,1));
    vector2 = np.zeros((3,1));
    if plane == 'bc':
        for i in range(3):
            vector1[0] = 0;
            vector1[1] = midpb[1] - pb1[1];
            vector1[2] = midpb[2] - pb1[2];
            vector2[0] = 0;
            vector2[1] = midcl[1] - midpb[1];
            vector2[2] = midcl[2] - midpb[2];
            distortion = vector1[1]*vector2[2] - vector1[2]*vector2[1];#cross product
    if plane == 'ca':
        for i in range(3):
            vector1[0] = midpb[0] - pb1[0];
            vector1[1] = 0;
            vector1[2] = midpb[2] - pb1[2];
            vector2[0] = midcl[0] - midpb[0];
            vector2[1] = 0;
            vector2[2] = midcl[2] - midpb[2];
            distortion = vector1[2]*vector2[0] - vector1[0]*vector2[2];
    if plane == 'ab':
        for i in range(3):
            vector1[0] = midpb[0] - pb1[0];
            vector1[1] = midpb[1] - pb1[1];
            vector1[2] = 0;
            vector2[0] = midcl[0] - midpb[0];
            vector2[1] = midcl[1] - midpb[1];
            vector2[2] = 0;
            distortion = vector1[0]*vector2[1] - vector1[1]*vector2[0];
    
    return distortion;

def planemodes(distortion_plane, plane_mode):
    for i in range(8):
        count = 0;
        for j in range(4):
            if distortion_plane[i][j] > 0:
                count = count + 1;
        if count == 4:
            plane_mode[i] = 1; #here 1 represents pure rotation
        elif count == 0:
            plane_mode[i] = 3;
        elif count == 2:
            if distortion_plane[i][1]*distortion_plane[i][3] > 0:
                plane_mode[i] = 4;
            else:
                plane_mode[i] = 2;

def overallmodes(plane_modes,overall_modes):
    database = np.zeros((20,3));
    for i in range(4):
        database[i] = [i+1, i+1, i+1];
    
    for i in range(3):
        database[i+4] = [1,1,i+2];
        database[i+13] = [4,4,i+1];
    
    database[7] = [2,2,1];
    database[8] = [2,2,3];
    database[9] = [2,2,4];    
        
    database[10] = [3,3,1];
    database[11] = [3,3,2];
    database[12] = [3,3,4];
    
    database[16] = [1,2,3];
    database[17] = [1,2,4];
    database[18] = [1,3,4];
    database[19] = [2,3,4];
        
    for i in range(8):
        for j in range(20):
            a = list(itertools.permutations(database[j]));
            count = 0;
            for k in range(len(a)):
                count = count + 1;
                temp = list(a[k]);
                if np.linalg.norm(plane_modes[i] - temp) < 0.01:
                    overall_modes[i] = j+1;
                    break;
                else:
                    continue;
            if count == len(a):
                continue;
            else:
                break;

def det_orien(vector):
    x0 = [1, 0, 0];
    y0 = [0, 1, 0];
    z0 = [0, 0, 1];
    indicator = np.zeros((3,1))
    indicator[0] = np.dot(vector,x0);
    indicator[1] = np.dot(vector,y0);
    indicator[2] = np.dot(vector,z0);
    orien_temp = indicator[0];
    symbol = 1; #refer to the note to know the meaning of the symbol
    #determine the orientation of the C-N
    if abs(indicator[1]) > abs(indicator[0]):
        orien_temp = indicator[1];
        symbol = 2;
        if abs(indicator[2]) > abs(orien_temp):
            orien_temp = indicator[2];
            symbol = 3;
    elif abs(indicator[2]) > abs(orien_temp):
        orien_temp = indicator[2];
        symbol = 3;
    else:
        symbol = 1;
    if orien_temp > 0:
        orientation = symbol;
    else:
        orientation = symbol + 3;
    
    return orientation;

def ewald(charges,vq,lattice,sig,rcut,gcut):
    na = len(charges);
    short = 0;
    long = 0;
    selfe = 0;
    pi = 3.14159265359;
    V = abs(np.linalg.det(lattice));
    gvecs = 2*ma.pi*np.linalg.inv(lattice);
    for lx in range(-rcut,rcut+1,1):
        for ly in range(-rcut,rcut+1,1):
            for lz in range(-rcut,rcut+1,1):
                for i in range(na):
                    for j in range(na):
                        if i==j and lx==0 and ly ==0 and lz==0:
                            continue;
                        else:
                            deno = np.linalg.norm(vq[i] - vq[j] + np.matmul([lx,ly,lz],lattice));
                            short = short + 0.5*charges[i][0]*charges[j][0]*ma.erfc(deno/ma.sqrt(2)/sig)/deno;
                        
    for i in range(na):
        selfe = selfe + 1/ma.sqrt(2*pi)/sig*charges[i][0]*charges[i][0];
    
    for gx in range(-gcut,gcut+1,1):
        for gy in range(-gcut,gcut+1,1):
            for gz in range(-gcut,gcut+1,1):
                if gx == 0 and gy == 0 and gz ==0:
                    continue;
                else:

                    sk = complex(0,0);
                    gvec = np.matmul([gx,gy,gz],gvecs);
                    for i in range(na):
                        charge = complex(charges[i][0],0);
                        sk = sk + charge*cmath.exp(1j*np.dot(gvec,vq[i]));
                    
                    k = np.linalg.norm(gvec);
                    long = long + 2*pi*ma.exp(-sig*sig*k*k/2)*abs(sk)*abs(sk)/(V*k*k);
    
#    print(type(short));
#    print(type(long));
#    print(type(selfe));
    return short+long-selfe;

def cart_sample(vec_cn,cart_angle):
    for i in range(8):
        x0 = vec_cn[i][0]; y0 = vec_cn[i][1]; z0 = vec_cn[i][2];
        cart_angle[i][1] = z0;
        cart_angle[i][0] = ma.acos(x0/ma.sqrt(x0*x0 + y0*y0));
        if y0 < 0:
            cart_angle[i][0] = 2*ma.pi - cart_angle[i][0];
                            

def det_mode(mode_name, index_h_cl1, index_h_cl2, whether_mode):
    #import the mode file
    num_mode = 0;
    mode_file = open(mode_name,'r');
    modelines = mode_file.readlines();
    num_pattern = len(modelines);
    num_atoms = len(modelines[0].split());
    mode = np.zeros((num_pattern,num_atoms));
    for i in range(num_pattern):
        modeline = modelines[i].split();
        mode[i] = modeline;
    mode_file.close();
    
    for i in range(8):
        if index_h_cl2[i] != 0 and num_atoms == 4:
            temp_pattern = np.zeros((1,4));
            for j in range(3):
                temp_pattern[0][j] = index_h_cl1[i][j];
                
            temp_pattern[0][3] = index_h_cl2[i];
            
            for indi_pattern in mode:
                for permu in list(itertools.permutations(indi_pattern)):
                    if np.linalg.norm(temp_pattern - permu) < 0.0001:
                        whether_mode[i] = 1;
                        num_mode = num_mode + 1;
                        break;

                
        elif index_h_cl2[i] == 0 and num_atoms == 3:
            temp_pattern = index_h_cl1[i];
            
            for indi_pattern in mode:
                for permu in list(itertools.permutations(indi_pattern)):
                    if np.linalg.norm(temp_pattern - permu) < 0.0001:
                        whether_mode[i] = 1;
                        num_mode = num_mode + 1;
                        break;
        else:
            continue;
    
    return num_mode;          

#Determine the orienation
def orientation(vec):
    vec = vec/np.linalg.norm(vec);
    unit_100 = [[1,0,0],[0,1,0],[0,0,1]];
    unit_110 = [[1,1,0],[1,0,1],[0,1,1],[1,-1,0],[1,0,-1],[0,1,-1]];
    unit_111 = [[1,1,1],[1,-1,-1],[1,1,-1],[1,-1,1]];
    unit_100 = np.asarray(unit_100); unit_110 = np.asarray(unit_110); unit_111 = np.asarray(unit_111);
    temp_indicator = np.concatenate((unit_100,-1*unit_100,unit_110,-1*unit_110,unit_111,-1*unit_111),axis=0)
    indicator = np.zeros((len(temp_indicator),3));
    #normalize the indicator
    length = len(temp_indicator);
    for i in range(length):
        norm = (np.linalg.norm(temp_indicator[i]))
        for j in range(3):
            indicator[i][j] = (temp_indicator[i][j]+0.0)/norm;

    dotprod = np.zeros((len(indicator),1));
    for i in range(len(indicator)):
        dotprod[i] = np.dot(vec,indicator[i]);

    temp_index = np.argsort(dotprod,axis=0);
    index = temp_index[-1];
    if index+1 < 7:
        simplified = 1;
    elif index+1 < 19:
        simplified = 2;
    else:
        simplified = 3;
    return simplified

#Determine and assign a simplified orientation
#Determine the orienation
def assign_orient(vec):
    vec = vec/np.linalg.norm(vec);
    unit_100 = [[1,0,0],[0,1,0],[0,0,1]];
    unit_110 = [[1,1,0],[1,0,1],[0,1,1],[1,-1,0],[1,0,-1],[0,1,-1]];
    unit_111 = [[1,1,1],[1,-1,-1],[1,1,-1],[1,-1,1]];
    unit_100 = np.asarray(unit_100); unit_110 = np.asarray(unit_110); unit_111 = np.asarray(unit_111);
    temp_indicator = np.concatenate((unit_100,-1*unit_100,unit_110,-1*unit_110,unit_111,-1*unit_111),axis=0)
    indicator = np.zeros((len(temp_indicator),3));
    #normalize the indicator
    length = len(temp_indicator);
    for i in range(length):
        norm = (np.linalg.norm(temp_indicator[i]))
        for j in range(3):
            indicator[i][j] = (temp_indicator[i][j]+0.0)/norm;

    dotprod = np.zeros((len(indicator),1));
    for i in range(len(indicator)):
        dotprod[i] = np.dot(vec,indicator[i]);

    temp_index = np.argsort(dotprod,axis=0);
    index = temp_index[-1];
    simplified = indicator[int(index)];
    return simplified
        
#Determine the short-range mode (old one)
def det_sr(p1,p2,d):
    prod0 = np.dot(p1,p2)
    prod1 = np.dot(p1,d);
    prod2 = np.dot(p2,d);
    temp_pattern = 0;
    if abs(prod0) >  0.0001: #which means that they are parallel
        if abs(prod1) > 0.0001: #which means that they are both aligned to the principal axis
            if prod0 < 0: #which means that they are anti-parallel
                temp_pattern = 4;
            else:
                temp_pattern = 3;
        else:
            if prod0 < 0:
                temp_pattern = 2;
            else:
                temp_pattern = 1;
    else:
        if abs(prod1) > 0.0001 and abs(prod2) < 0.0001: #which means only one is aligned to the principal axis
            if prod1 > 0:
                temp_pattern = 5;
            else:
                temp_pattern = 5;
        elif abs(prod1) < 0.0001 and abs(prod2) > 0.0001:
            if prod2 < 0:
                temp_pattern = 5;
            else:
                temp_pattern = 5;
        else:
            temp_pattern = 6;
      
    return temp_pattern;
