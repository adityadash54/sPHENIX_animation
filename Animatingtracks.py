#Reading the json file
import numpy as np
#import pandas as pd
import json
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import array

f=open('TRACKS2_21March.json') #else use the folder option on the left column and upload the file manually
data_json=json.load(f)
print(data_json)
f.close()

#User defined values
drift_speed_posz=np.array([0.0,0.0,1.0])
drift_speed_negz=np.array([0.0,0.0,-1.0])
no_tracks=20 #number of tracks to animate it will start from track 1



dict_json = data_json.items()
numpy_array_json = np.array(list(dict_json))
mom_dict=numpy_array_json[3][1] #only works for this format the 3rd row is TRACKS and and 1 refers to INNERTRACKER
innertracker_arr=mom_dict['INNERTRACKER'] #stores the tracks information as an array

print("Input Dictionary =",dict_json)
print("The resultant numpy array:\n", numpy_array_json)
print("mom_dict=",mom_dict)
print("innertracker_arr=",innertracker_arr)

#Getting the cluster positions of a track
x_y_z_clusters=[]
for track_no in range(no_tracks):
    x_y_z_dict_track=innertracker_arr[track_no] #innertracker_arr[i] reads the ith track
    x_y_z_clusters_track=np.array(x_y_z_dict_track['pts']) #2D array the x,y,z positions corresponding to each cluster e.g. x_y_z[0][0] is the x coordinate of the 1st cluster
    #for jj in range(len(x_y_z_clusters_track)):
        #x_y_z_clusters.append(x_y_z_clusters_track[jj])
    #print("x_y_z_clusters_track[0]=")
    #print(x_y_z_clusters_track[0])
    
    if(track_no==0):
        x_y_z_clusters=x_y_z_clusters_track
        #print("track_no==0, x_y_z_clusters[0]=")
        #print(x_y_z_clusters[0])
        #lentrack1=len(x_y_z_clusters_track)
    
    else:
        x_y_z_clusters=np.append(x_y_z_clusters,x_y_z_clusters_track,axis=0)
        #print("track_no!=0, x_y_z_clusters["+str(lentrack1)+"]=")
        #print(x_y_z_clusters[lentrack1])

    print("shape of x_y_z clusters=")
    print(x_y_z_clusters.shape)
    
        
        
#cluster_positions=pd.DataFrame.from_dict(x_y_z_clusters) #not required but makes visualisation easier
#cluster_positions.columns=['x','y','z']

#print(innertracker_arr[1])
#print(x_y_z_clusters)
#print(cluster_positions)


#ANIMATION
#based on https://medium.com/@pnpsegonne/animating-a-3d-scatterplot-with-matplotlib-ca4b676d4b55

start_speed_posz = np.zeros(x_y_z_clusters.shape)
start_speed_negz = np.zeros(x_y_z_clusters.shape)


for jj in range(len(start_speed_posz)):
  start_speed_posz[jj]=drift_speed_posz
  start_speed_negz[jj]=drift_speed_negz

#We can do this faster by separating the clusters with z>0 and z<0 and rearranging the cluster with z>0 on top of z<0 then we do not need to check if z>0 for every iteration
#print(start_speed)
data = [x_y_z_clusters]
#print(data)
        
nbr_iterations=100
for iteration in range(nbr_iterations):
        previous_positions = np.copy(data[-1]) #otherwise the values in data[-1] change when we change values in previous_positions be careful
        new_positions=np.copy(previous_positions) #initialisation
        
        for jj in range(len(previous_positions)):
            if(previous_positions[jj][2]>0):
                new_positions[jj] = previous_positions[jj] + drift_speed_posz
            elif(previous_positions[jj][2]<0):
                #print(new_positions[jj])
                new_positions[jj] = previous_positions[jj] + drift_speed_negz
                #print("negz")
                #print(new_positions[jj])
            else:
                print("Track found with z postion exactly equal to 0, it stays there")
        #print("new positions")
        #print(new_positions)
        data.append(new_positions)
        #print(data)

        #if(iteration<5):
        #  print("previous_positions=")
        #  print(previous_positions)
        #  print("new_positions=")
        #  print(new_positions)

        

def animate_scatters(iteration, data, scatters):
#    """
#    Update the data held by the scatter plot and therefore animates it.
#    Args:
#        iteration (int): Current iteration of the animation
#        data (list): List of the data positions at each iteration.
#        scatters (list): List of all the scatters (One per element)
#    Returns:
#        list: List of scatters (One per element) with new coordinates
#    """
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])
        #if(iteration==1):
         #   if(i==1):
         #       print(data[iteration][i,0:1])
         #       print(data[iteration][i,1:2])
         #       print(data[iteration][i,2:])
        
    return scatters

def main(data, save=False):
#    """
#    Creates the 3D figure and animates it with the input data.
#    Args:
#        data (list): List of the data positions at each iteration.
#        save (bool): Whether to save the recording of the animation. (Default to False).
#    """

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax = p3.Axes3D(fig)

    # Initialize scatters
    scatters = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:]) for i in range(data[0].shape[0]) ]

    # Number of iterations
    iterations = len(data)

    # Setting the axes properties
    ax.set_xlim3d([-100, 100])
    ax.set_xlabel('X')

    ax.set_ylim3d([-100, 100])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-100, 100])
    ax.set_zlabel('Z')

    ax.set_title('Track drifting in TPC')

    # Provide starting angle for the view.
    ax.view_init(25, 90,0,'y')

    ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                                       interval=1000/iterations, blit=False, repeat=True)

    if save:
        ani.save('Animated_tracks.mp4',writer='ffmpeg',fps=iterations)
        #ani.save('Animated_tracks.gif',writer='Pillow',fps=iterations)
        
        #Writer = animation.writers['ffmpeg']
        #writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        #ani.save('3d-scatted-animated.mp4', writer=writer)

    plt.show()


#print(data)
main(data, save=True)
