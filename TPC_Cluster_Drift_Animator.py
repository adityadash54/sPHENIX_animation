import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import awkward as ak
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import array

#/*************************************************************/
#/*              TPC Cluster Drift Animator                  */
#/*         Aditya Prasad Dash, Thomas Marshall              */
#/*      aditya55@physics.ucla.edu, rosstom@ucla.edu         */
#/*************************************************************/
#Code in python
#Input:
# json file containing TPC clusters
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

def raddist_cluster(cluster_pos):
    radius=np.sqrt(cluster_pos[:,0]*cluster_pos[:,0]+cluster_pos[:,1]*cluster_pos[:,1])
    return radius
    
def animate_scatters(iteration, data, scatters):   #adapted from https://medium.com/@pnpsegonne/animating-a-3d-scatterplot-with-matplotlib-ca4b676d4b55
#    Input:
#        iteration (int): Current iteration of the animation
#        data (list): List of the data positions at each iteration.
#        scatters (list): List of all the scatters (One per element)
#    Returns:
#        list: List of scatters (One per element) with new coordinates

    for i in range(data[0].shape[0]):
        if(i<data[iteration].shape[0]):
            if(iteration>data[iteration][i,4]): #and iteration>data[iteration][i,2:3]<105):
                scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:3])
            else:
                scatters[i]._offsets3d = ([100], [-100], [100])  #clusters from event not yet taken place
            
        else:
            scatters[i]._offsets3d = ([100], [-100], [100]) #to plot all points outside TPC at one point
        
    return scatters


def animate_clusters(data, save=False):
#    Input:
#        data (list): List of the cluster positions at each iteration.
#        save (bool): Option to save the recording of the animation. (Default is False).
#    Output:
#        Animates the clusters, plots it and saves it if save=True

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #Drawing TPC
    Xc_in,Yc_in,Xc_out,Yc_out,Zc = TPC_surface(20,80,105)
    ax.plot_surface(Xc_in, Yc_in, Zc, alpha=0.3)
    ax.plot_surface(Xc_out, Yc_out, Zc, alpha=0.3)
    
    # Initialize scatters
    scatters = [ ax.scatter([100],[-100],[100]) for i in range(data[0].shape[0])]
    #scatters = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:3]) for i in range(data[0].shape[0])]
    print("Plot initialized")
    # Number of iterations
    iterations = len(data)

    # Setting the axes properties
    ax.set_xlim3d([-120, 120])
    ax.set_xlabel('X')

    ax.set_ylim3d([-120, 120])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-120, 120])
    ax.set_zlabel('Z')

    ax.set_title('Clusters drifting in TPC (speed scaled by $5*10^{-6}$)')

    # Provide starting angle for the view.
    ax.view_init(20, 30,0,'y')

    ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                                       interval=20, blit=False, repeat=True) #interval is in milliseconds and is the time between each frame

    if save:
        print("Saving animation as Animated_clusters.mp4")
        ani.save('Animated_clusters.mp4',writer='ffmpeg',fps=50)
        
        print("Animation saved")
    plt.show()

# Main Program starts from here
#print("Reading json file")
    
#User defined values
drift_speed_posz=np.array([0.0,0.0,0.8,0.0,0.0]) #z distance travelled in cm per iteration(in 20ms as fps is 50) in the animation #Actual drift speed=8cm/microsecond, so here it is scaled by 5*10^-6 i.e. in animation speed is 40cm/microsecond
drift_speed_negz=np.array([0.0,0.0,-0.8,0.0,0.0]) #(x,y,z,event,gvt)

def read_cluster_pos(inFile):
    if inFile.lower().endswith('.json'):
        print("Reading data from json file")
        file=open(inFile)
        data_json=json.load(file)
        file.close()
        dict_json = data_json.items()
        numpy_array_json = np.array(list(dict_json))
        mom_dict=numpy_array_json[3][1] #only works for this format the 3rd row is TRACKS and and 1 refers to INNERTRACKER
        innertracker_arr=mom_dict['INNERTRACKER'] #stores the tracks information as an array

        #print(innertracker_arr)
        print("Reading clusters")
        #Getting the cluster positions of a track
        x_y_z_clusters=np.array([])
        no_tracks=len(innertracker_arr) #set no_tracks=10 for testing
        
        for track_no in range(no_tracks):
            x_y_z_dict_track=innertracker_arr[track_no].copy() #innertracker_arr[i] reads the ith track
            print(x_y_z_dict_track['pts'])
            x_y_z_clusters_track=np.array(x_y_z_dict_track['pts']) #2D array the x,y,z positions corresponding to each cluster e.g. x_y_z[0][0] is the x coordinate of the 1st cluster
            x_y_z_clusters_track=x_y_z_clusters_track[raddist_cluster(x_y_z_clusters_track)>30]
    
            if(track_no==0):
                x_y_z_clusters=np.copy(x_y_z_clusters_track)
    
            else:
                x_y_z_clusters=np.append(x_y_z_clusters,x_y_z_clusters_track,axis=0)
        return x_y_z_clusters

    if inFile.lower().endswith('.root'):
        print("Reading data from root file")
        file = uproot.open(inFile)
        ntp_cluster_tree=file['ntp_cluster']
        branches=ntp_cluster_tree.arrays(["x","y","z","event","gvt"])
        branches=branches[~np.isnan(branches.gvt)]
        print("Reading clusters")
        x_y_z_clusters_run=np.array([])
        gvt_clusters_run=np.array([])
        
        for cluster in range(len(branches)):#range(len(branches)):
            #if(branches[cluster]['gvt']==0):
            #    continue
            branches[cluster]['gvt']=branches[cluster]['event']*5000
            x_y_z_clusters_track=np.array([[branches[cluster]['x'], branches[cluster]['y'], branches[cluster]['z'],branches[cluster]['event'],branches[cluster]['gvt']]])
            #gvt_clusters_track=np.array([[branches[cluster]['gvt']]])
            gvt_clusters_track=np.array([[branches[cluster]['event']]])*50000
            #gvt_clusters_track=gvt_clusters_track
            if(cluster==0):
                x_y_z_clusters_run=np.copy(x_y_z_clusters_track)
                gvt_clusters_run=np.copy(gvt_clusters_track)
    
            else:
                x_y_z_clusters_run=np.append(x_y_z_clusters_run,x_y_z_clusters_track,axis=0)
                gvt_clusters_run=np.append(gvt_clusters_run,gvt_clusters_track,axis=0)
        return x_y_z_clusters_run,gvt_clusters_run
        
print("Generating data for animation")

#ANIMATION
x_y_z_clusters,gvt_clusters=read_cluster_pos("Data_files/G4sPHENIX_g4svtx_eval_gvt.root")
gvt_clusters=gvt_clusters/(5*10-6)  #Scaled time in nanoseconds
gvt_clusters=gvt_clusters/(20*10^6) #20ms is the time per iterations so this is the time in terms of iterations
data = [x_y_z_clusters]
print(data)
print(gvt_clusters)

#print(True and gvt_clusters[0]*3>1000)
nbr_iterations=1000
for iteration in range(nbr_iterations):
        previous_positions = np.copy(data[-1]) #use np.copy() otherwise the values in data[-1] change when we change values in previous_positions
        new_positions=np.copy(previous_positions) #initialisation
        
        for jj in range(len(previous_positions)):
            #if jj>len(previous_positions)/2.0:
             #   gvt_clusters[jj]=500
            if(previous_positions[jj][2]>0 and iteration>gvt_clusters[jj]):
                new_positions[jj] = previous_positions[jj] + drift_speed_posz
            elif(previous_positions[jj][2]<0 and iteration>gvt_clusters[jj]):
                new_positions[jj] = previous_positions[jj] + drift_speed_negz
                
        new_positions=new_positions[abs(new_positions[:,2])<105] #retaining only the clusters inside TPC
        data.append(new_positions) #the last array should have size 0 for animation to remove all clusters outside TPC
        if(len(new_positions)==0):
            break
        
        
print("Animation starting!")

#Saving takes a long time so use Save=True only when necessary
#increase drift_speed_posz and drift_speed_negz if desired
animate_clusters(data, save=False)


#ntp_cluster->event and gtrack_id
#go to ntp_gtrack and select track with same event and gtrack_id
#grab gvt from gtrack_id and assign to cluster
