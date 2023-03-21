#Reading the json file
import numpy as np
#import pandas as pd
import json
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

f=open('TRACKS2.json') #else use the folder option on the left column and upload the file manually
data_json=json.load(f)
print(data_json)
f.close()

dict_json = data_json.items()
numpy_array_json = np.array(list(dict_json))
mom_dict=numpy_array_json[3][1] #only works for this format the 3rd row is TRACKS and and 1 refers to INNERTRACKER
innertracker_arr=mom_dict['INNERTRACKER'] #stores the tracks information as an array

print("Input Dictionary =",dict_json)
print("The resultant numpy array:\n", numpy_array_json)
print("mom_dict=",mom_dict)
print("innertracker_arr=",innertracker_arr)

#Getting the cluster positions of a track
x_y_z_dict=innertracker_arr[1] #innertracker_arr[i] reads the ith track
x_y_z_clusters=np.array(x_y_z_dict['pts']) #2D array the x,y,z positions corresponding to each cluster e.g. x_y_z[0][0] is the x coordinate of the 1st cluster
#cluster_positions=pd.DataFrame.from_dict(x_y_z_clusters) #not required but makes visualisation easier
#cluster_positions.columns=['x','y','z']

print(innertracker_arr[1])
print(x_y_z_clusters)
#print(cluster_positions)


#ANIMATION
#based on https://medium.com/@pnpsegonne/animating-a-3d-scatterplot-with-matplotlib-ca4b676d4b55

drift_speed=[0.0,0.0,1.0]
start_speed = np.zeros(x_y_z_clusters.shape)


for jj in range(len(start_speed)):
  start_speed[jj]=drift_speed

print(start_speed)
data = [x_y_z_clusters]
nbr_iterations=5
for iteration in range(nbr_iterations):
        previous_positions = data[-1]
        new_positions = previous_positions + start_speed
        data.append(new_positions)
        #if(iteration<5):
        #  print("previous_positions=")
        #  print(previous_positions)
        #  print("new_positions=")
        #  print(new_positions)

        
#data

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
    ax = p3.Axes3D(fig)

    # Initialize scatters
    scatters = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:]) for i in range(data[0].shape[0]) ]

    # Number of iterations
    iterations = len(data)

    # Setting the axes properties
    ax.set_xlim3d([-50, 50])
    ax.set_xlabel('X')

    ax.set_ylim3d([-50, 50])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-50, 50])
    ax.set_zlabel('Z')

    ax.set_title('3D Animated Scatter Example')

    # Provide starting angle for the view.
    ax.view_init(25, 10)

    ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                                       interval=50, blit=False, repeat=True)

    if save:
        f = r"animations/animation_new.gif"
        writergif = animation.PillowWriter(fps=30)
        ani.save(f, writer=writergif)

    plt.show()


#print(data)
main(data, save=True)
