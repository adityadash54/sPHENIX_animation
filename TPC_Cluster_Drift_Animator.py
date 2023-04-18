import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import awkward as ak
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Wedge
#from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import array

#/*************************************************************/
#/*              TPC Cluster Drift Animator                  */
#/*         Aditya Prasad Dash, Thomas Marshall              */
#/*      aditya55@physics.ucla.edu, rosstom@ucla.edu         */
#/*************************************************************/
#Code in python
#Input:
# root or json file containing TPC clusters
#Output:
# Animation of drifting of TPC clusters with user defined speed and option to save in .mp4 format

def TPC_surface(inner_radius,outer_radius, length_z):
    ngridpoints=30
    z = np.linspace(-length_z, length_z, ngridpoints)
    phi = np.linspace(0, 2*np.pi, ngridpoints)
    phi_grid, z_grid=np.meshgrid(phi, z)
    x_grid_inner = inner_radius*np.cos(phi_grid)
    y_grid_inner = inner_radius*np.sin(phi_grid)
    x_grid_outer = outer_radius*np.cos(phi_grid)
    y_grid_outer = outer_radius*np.sin(phi_grid)
    
    return x_grid_inner,y_grid_inner,x_grid_outer,y_grid_outer,z_grid

def TPC_endcap(inner_radius,outer_radius, length_z):
    ngridpoints=30
    radius = np.linspace(inner_radius, outer_radius, 5)
    phi = np.linspace(0, 2*np.pi, ngridpoints)
    phi_grid, r_grid=np.meshgrid(phi, radius)
    x_grid=[]
    y_grid=[]
    for r in radius:
        x_grid_radius = r*np.cos(phi_grid)
        y_grid_radius = r*np.sin(phi_grid)
        x_grid=np.append(x_grid,x_grid_radius)
        y_grid=np.append(y_grid,y_grid_radius)
    
    z_grid=x_grid
    return x_grid,y_grid,z_grid


def raddist_cluster(cluster_pos):
    radius=np.sqrt(cluster_pos[:,0]*cluster_pos[:,0]+cluster_pos[:,1]*cluster_pos[:,1])
    return radius
    
def animate_scatters(iteration, data, scatters,fig_text,time_scale,iteration_time):
    for i in range(data[0].shape[0]):
        if(i<data[iteration].shape[0]):
            if(iteration>=data[iteration][i,4]):
                scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:3])
                color=['r','g','b','c','m','y']
                scatters[i].set_color(color[int(data[iteration][i,3]%6)])
                scatters[i].set(alpha=1.0)
                scatters[i].set_sizes([10])
                scatters[i].set_zorder((iteration))
                
            else:
                scatters[i]._offsets3d = ([100], [-100], [100])  #clusters from event not yet taken place
                color=['r','g','b','c','m','y']
                scatters[i].set_color('black')
                scatters[i].set_sizes([10]) #= [0.1]
                
        else:
            scatters[i]._offsets3d = ([100], [-100], [100]) #to plot all points outside TPC at one point
            scatters[i].set_color('black')
    fig_text.set_text(str(round(iteration*iteration_time/time_scale*(10**3),3))+"$\mu$s")
        
    return scatters,fig_text


def animate_clusters(data, save=False):
#    Input:
#        data (list): List of the cluster positions at each iteration.
#        save (bool): Option to save the recording of the animation. (Default is False).
#    Output:
#        Animates the clusters, plots it and saves it if save=True

    # Attaching 3D axis to the figure
    fig = plt.figure(figsize=[7.5,7.5],layout='constrained')
    ax = fig.add_subplot(111, projection='3d',facecolor='black',alpha=0.0)
    ax.grid(False)
    ax.margins(0.0)
    ax.xaxis.set_pane_color((1,1,1,0))
    ax.xaxis.line.set_color('w')
    ax.spines['top'].set_color('w')
    ax.spines['bottom'].set_color('w')
    ax.spines['left'].set_color('w')
    ax.spines['right'].set_color('w')
    
    ax.yaxis.set_pane_color((1,1,1,0))
    ax.yaxis.line.set_color('w')
    ax.zaxis.set_pane_color((1,1,1,0))
    ax.zaxis.line.set_color('w')
    
    #Drawing TPC
    endcap1=Wedge((0, 0), 80, 0, 360, color='blue',alpha=0.7,zorder=1)
    endcap2=Wedge((0, 0), 80, 0, 360, color='blue',alpha=0.7,zorder=2)
    endcap1.set_width(60)
    endcap2.set_width(60)
    
    ax.add_artist(endcap1)
    ax.add_artist(endcap2)
    
    art3d.pathpatch_2d_to_3d(endcap1, z=105, zdir="z")
    art3d.pathpatch_2d_to_3d(endcap2, z=-105, zdir="z")
    
    Xc_in,Yc_in,Xc_out,Yc_out,Zc = TPC_surface(20,80,105)
    ax.plot_surface(Xc_in, Yc_in, Zc, alpha=0.5,zorder=3)
    ax.plot_surface(Xc_out, Yc_out, Zc, alpha=0.5,zorder=4)
    
    # Setting the axes properties
    ax.set_xlim3d([-100, 100])
    ax.set_xlabel('X',color='white',fontsize=7)
    ax.xaxis.set_tick_params(colors='white',labelsize=7)
    
    ax.set_ylim3d([-100, 100])
    ax.set_ylabel('Y',color='white',fontsize=7)
    ax.yaxis.set_tick_params(colors='white',labelsize=7)

    ax.set_zlim3d([-120, 120])
    ax.set_zlabel('Z',color='white',fontsize=7)
    ax.zaxis.set_tick_params(colors='white',labelsize=7)

    ax.set_title('Clusters drifting in TPC') #(time scaled by $2*10^{5}$)')
    fig_text=ax.text(-100,90,100,  'time', size=10,color='w',alpha=0.9)
    fig_text_sPhenix=ax.text(-100,130,100,  'sPHENIX', size=10,fontweight='bold',style='italic',color='w',alpha=0.9)
    fig_text_TPC=ax.text(-60,135,70,  'TPC simulation', size=10,style='italic',color='w',alpha=0.9)
    
    fig_text_sPhenix=ax.text(-100,110,100,  'p+p, $\sqrt{s_{NN}}$ = 200 GeV, 4MHz', size=10,color='w',alpha=0.9)
    
    
    # Provide starting angle for the view.
    ax.view_init(10,70,0,'y')
    
    # Initialize scatters
    scatters = [ ax.scatter([100],[-100],[100],zorder=5) for i in range(data[0].shape[0])]
    print("Plot initialized")
    # Number of iterations
    iterations = len(data)

    ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters,fig_text,time_scale,iteration_time),
                                       interval=20, blit=False, repeat=True) #interval is in milliseconds and is the time between each frame
    #ani.set_sizes(np.ones(scatters.shape))
    if save:
        print("Saving animation as Animated_clusters_TPC.mp4")
        ani.save('Animated_clusters_TPC.mp4',writer='ffmpeg')
        
        print("Animation saved")
    plt.show()

# Main Program starts from here
    
#User defined values
time_scale=5.0*(10.0**6) #inverse of speed scale
iteration_time=100.0 #20ms
TPC_drift_speed=8.0*(10.0**3) #Actual TPC drift speed =8cm/microsecond=8*10^3cm/millisecond
drift_speed_posz=np.array([0.0,0.0,TPC_drift_speed/time_scale*iteration_time,0.0,0.0])
drift_speed_negz=np.array([0.0,0.0,-TPC_drift_speed/time_scale*iteration_time,0.0,0.0])
print("drift_speed_posz=")
print(drift_speed_posz)
print("drift_speed_negz=")
print(drift_speed_negz)

def read_cluster_pos(inFile):
    if(inFile.lower().endswith('.json')):
        print("Reading data from json file")
        file=open(inFile)
        data_json=json.load(file)
        file.close()
        dict_json = data_json.items()
        numpy_array_json = np.array(list(dict_json))
        mom_dict=numpy_array_json[3][1] #only works for this format the 3rd row is TRACKS and and 1 refers to INNERTRACKER
        innertracker_arr=mom_dict['INNERTRACKER'] #stores the tracks information as an array

        print("Reading clusters")
        #Getting the cluster positions of a track
        x_y_z_clusters=np.array([])
        no_tracks=len(innertracker_arr) #set no_tracks=10 for testing
        
        for track_no in range(no_tracks):
            x_y_z_dict_track=innertracker_arr[track_no].copy() #innertracker_arr[i] reads the ith track
            x_y_z_clusters_track=np.array(x_y_z_dict_track['pts']) #2D array the x,y,z positions corresponding to each cluster e.g. x_y_z[0][0] is the x coordinate of the 1st cluster
            x_y_z_clusters_track=x_y_z_clusters_track[raddist_cluster(x_y_z_clusters_track)>30]
    
            if(track_no==0):
                x_y_z_clusters=np.copy(x_y_z_clusters_track)
    
            else:
                x_y_z_clusters=np.append(x_y_z_clusters,x_y_z_clusters_track,axis=0)
        return x_y_z_clusters

    if(inFile.lower().endswith('.root')):
        print("Reading data from root file")
        file = uproot.open(inFile)
        ntp_cluster_tree=file['ntp_cluster']
        branches=ntp_cluster_tree.arrays(["x","y","z","event","gvt"])
        branches=branches[~np.isnan(branches.gvt)]
        branches=branches[((branches.x)**2+(branches.y)**2)>900]
        #branches=branches[branches.event<10]
        #print(((branches.x)**2+(branches.y)**2))
        print("Reading clusters")
        x_y_z_clusters_run=np.array([])
        len_events=len(np.unique(branches.event))
        event_times=[0]
        event_times=np.append(event_times,np.random.poisson(333.33,len_events-1))
        event_times=np.cumsum(event_times,dtype=float)
        for cluster in range(len(branches)):
            gvt_event=event_times[int(branches[cluster]['event'])]
            gvt_event=gvt_event*(10**(-6))*time_scale #Time in milliseconds scaled for animation
            gvt_event=gvt_event/(iteration_time) #20ms is the time per iterations so this is the time in terms of iterations
            x_y_z_clusters_track=np.array([[branches[cluster]['x'], branches[cluster]['y'], branches[cluster]['z'],branches[cluster]['event'],gvt_event]])
            if(cluster==0):
                x_y_z_clusters_run=np.copy(x_y_z_clusters_track)
            else:
                x_y_z_clusters_run=np.append(x_y_z_clusters_run,x_y_z_clusters_track,axis=0)
        return x_y_z_clusters_run
        
print("Generating data for animation")

#ANIMATION
x_y_z_clusters=read_cluster_pos("Data_files/G4sPHENIX_g4svtx_eval_gvt_13April.root")
data = [x_y_z_clusters]

nbr_iterations=100000
for iteration in range(nbr_iterations):
        previous_positions = np.copy(data[-1]) #use np.copy() otherwise the values in data[-1] change when we change values in previous_positions
        new_positions=np.copy(previous_positions) #initialisation
        for jj in range(len(previous_positions)):
            if(previous_positions[jj][2]>0 and iteration>=previous_positions[jj][4]):
                new_positions[jj] = previous_positions[jj] + drift_speed_posz
            elif(previous_positions[jj][2]<0 and iteration>=previous_positions[jj][4]):
                new_positions[jj] = previous_positions[jj] + drift_speed_negz
        new_positions=new_positions[abs(new_positions[:,2])<105] #retaining only the clusters inside TPC
        data.append(new_positions) #the last array should have size 0 for animation to remove all clusters outside TPC
        if(len(new_positions)==0):
            break
        
print("Animation starting!")

#Saving takes a long time so use Save=True only when necessary
#increase drift_speed_posz and drift_speed_negz if desired
animate_clusters(data, save=False)

