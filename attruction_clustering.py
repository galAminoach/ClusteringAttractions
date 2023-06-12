import numpy as np
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw
from scipy.cluster import hierarchy
import os

from attractionToVector import create_attraction_vec
attraction_vectors = create_attraction_vec()


# -------------- CONSTANTS -------------- #
ZOOM_STEP = 0.05
ARROWS_MOVEMENT_AMOUNT = 0.1
IMAGE_SIZE = 50
INITIAL_ZOOM = 0.15
CLICKED_IMAGE_ENLARGED_ = 1.5
IMAGES_ZOOM_ENLARGE = 0.5
ZOOM_SCALE = 2
ZOOM_AMOUNT = 0.1
DISTANCE_THRESHOLD = 5  # todo: pick wisely

# ----------- GLOBAL VARIABLES ----------- #
G = nx.Graph()
current_image, fig, ax, canvas, window, start_ylim, start_xlim, pos2 = None, None, None, None, None,None, None, None
x_min, x_max, y_min, y_max = 0, 0, 0, 0
current_zoom_scale = 1
attraction_names = list(attraction_vectors.keys())
attraction_matrix = np.array(list(attraction_vectors.values()))


# ----------- BUTTONS FUNCTIONS ----------- #

def on_button_press(event):
    global current_image

    if event.inaxes is not None:
        for node in G.nodes:
            imagebox = G.nodes[node]['imagebox']
            if imagebox.contains(event)[0]:

                if current_image == node:  # if the clicked image is the same as the current image (the clicked image is already enlarged)
                    G.nodes[current_image]['imagebox'].set_zoom(
                        INITIAL_ZOOM)  # restore the previously clicked image to its initial size
                    if 'annotation' in G.nodes[current_image]:
                        if G.nodes[current_image]['annotation'] is not None:
                            G.nodes[current_image]['annotation'].remove()
                        G.nodes[current_image]['annotation'] = None
                    canvas.draw_idle()
                    current_image = None

                else:  # If a different image is clicked - current_image != node
                    if current_image is not None:
                        G.nodes[current_image]['imagebox'].set_zoom(INITIAL_ZOOM)  # restore the previously clicked image to its initial size
                        if 'annotation' in G.nodes[current_image]:
                            if G.nodes[current_image]['annotation'] is not None:
                                G.nodes[current_image]['annotation'].remove()
                            G.nodes[current_image]['annotation'] = None
                    imagebox.set_zoom(CLICKED_IMAGE_ENLARGED_)  # enlarge the new clicked image
                    # Display the name of the attraction
                    attraction_name = node
                    if ('annotation' not in G.nodes[node]) or (G.nodes[node]['annotation'] is None):
                        G.nodes[node]['annotation'] = ax.annotate(attraction_name, (pos2[node][0], pos2[node][1]+0.03),
                                    xytext=(0, 7), textcoords='offset points',
                                    ha='center', va='bottom', color='black',
                                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

                    canvas.draw_idle()  # Update the plot
                    current_image = node

                fig.canvas.draw_idle()  # Use draw_idle instead of draw for better performance
                #plt.savefig('graph.png', dpi=2000)
                break  # Exit the loop after handling the click event


def reset_zoom_out():
    global current_image, current_zoom_scale

    buffer = ZOOM_AMOUNT * ZOOM_SCALE
    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)
    for node in G.nodes:
        imagebox = G.nodes[node]['imagebox']
        imagebox.set_zoom(INITIAL_ZOOM)  # restore images size on zoomout

    if current_image is not None:
        if 'annotation' in G.nodes[current_image]:
            if G.nodes[current_image]['annotation'] is not None:
                G.nodes[current_image]['annotation'].remove()
    current_image = None
    current_zoom_scale = 0

    canvas.draw_idle()  # Use draw_idle instead of draw for better performance


def zoom_in(event): # Define the function to handle zoom in
    global current_image, current_zoom_scale
    current_zoom_scale = 1

    for node in G.nodes:
        imagebox = G.nodes[node]['imagebox']
        if current_image != node:
            imagebox.set_zoom(IMAGES_ZOOM_ENLARGE)  # enlarge images when zooming

    ax.set_xlim(event.xdata - ZOOM_AMOUNT*ZOOM_SCALE, event.xdata + ZOOM_AMOUNT*ZOOM_SCALE)
    ax.set_ylim(event.ydata - ZOOM_AMOUNT*ZOOM_SCALE, event.ydata + ZOOM_AMOUNT*ZOOM_SCALE)

    canvas.draw()


def zoom_out(event):
    ax.set_xlim(x_min - ZOOM_STEP, x_max + ZOOM_STEP)
    ax.set_ylim(y_min - ZOOM_STEP, y_max + ZOOM_STEP)
    # ax.set_xlim(ax.get_xlim())
    # ax.set_ylim(ax.get_ylim())

    for node in G.nodes:
        imagebox = G.nodes[node]['imagebox']
        if current_image != node:
            imagebox.set_zoom(INITIAL_ZOOM)  # Restore images size on zoom out

    canvas.draw()


# Function to handle key press events
def on_key_press(event):
    if event.key == 'left':
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0] - ARROWS_MOVEMENT_AMOUNT, xlim[1] - ARROWS_MOVEMENT_AMOUNT)
    elif event.key == 'right':
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[0] + ARROWS_MOVEMENT_AMOUNT, xlim[1] + ARROWS_MOVEMENT_AMOUNT)
    elif event.key == 'up':
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] + ARROWS_MOVEMENT_AMOUNT, ylim[1] + ARROWS_MOVEMENT_AMOUNT)
    elif event.key == 'down':
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0] - ARROWS_MOVEMENT_AMOUNT, ylim[1] - ARROWS_MOVEMENT_AMOUNT)

    elif event.key == '=' or event.key == '+':
        manually_zoom_in(event)
    elif event.key == '-':
        manually_zoom_out(event)

    fig.canvas.draw()


def manually_zoom_in(event): # Define the function to handle zoom in
    global current_image, current_zoom_scale

    current_zoom_scale += ZOOM_STEP
    if current_zoom_scale > 1:
        current_zoom_scale = 1
        return
    for node in G.nodes:
        imagebox = G.nodes[node]['imagebox']
        zoom = max(INITIAL_ZOOM, imagebox.get_zoom()*1.1)
        zoom = min(IMAGES_ZOOM_ENLARGE, zoom)
        if (node == current_image):
            zoom = min(CLICKED_IMAGE_ENLARGED_, zoom)
            if 'annotation' in G.nodes[current_image]:
                if G.nodes[current_image]['annotation'] is not None:
                    G.nodes[current_image]['annotation'].remove()
            current_image = None
        imagebox.set_zoom(zoom)

    xlim = ax.get_xlim()
    xlim = (xlim[0] + ZOOM_STEP*2, xlim[1] - ZOOM_STEP*2)
    ylim = ax.get_ylim()
    ylim = (ylim[0] + ZOOM_STEP*2, ylim[1] - ZOOM_STEP*2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    canvas.draw()
    canvas.drawIdle()


def manually_zoom_out(event): # Define the function to handle zoom out
    global current_zoom_scale, current_image

    current_zoom_scale -= ZOOM_STEP
    if current_zoom_scale < 0:
        current_zoom_scale = 0
        return
    for node in G.nodes:
        imagebox = G.nodes[node]['imagebox']
        zoom = max(INITIAL_ZOOM,imagebox.get_zoom()*0.9)
        zoom = min(IMAGES_ZOOM_ENLARGE,zoom)
        if(node == current_image):
            zoom = min(CLICKED_IMAGE_ENLARGED_, zoom)
            if 'annotation' in G.nodes[current_image]:
                if G.nodes[current_image]['annotation'] is not None:
                    G.nodes[current_image]['annotation'].remove()
            current_image = None

        imagebox.set_zoom(zoom)

    xlim = ax.get_xlim()
    xlim = (xlim[0] - ZOOM_STEP*2, xlim[1] + ZOOM_STEP*2)
    ylim = ax.get_ylim()
    ylim = (ylim[0] - ZOOM_STEP*2, ylim[1] + ZOOM_STEP*2)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    canvas.draw()


# ----------- GRAPH CREATION FUNCTIONS ----------- #

def create_attraction_image_map():  # Generate a dictionary to map attraction names to their corresponding image paths
    image_paths = {}

    image_folder = '/Users/galaminoach/Desktop/clusteringProject'  # Specify the folder where the attraction images are located
    image_extension = '.jpg'  # Specify the image file extension

    for attraction_name in attraction_names:
        image_filename = f'{attraction_name}{image_extension}'
        image_path = os.path.join(image_folder, image_filename)
        image_paths[attraction_name] = image_path

    return image_paths


def add_images_to_graph():
    image_paths = create_attraction_image_map()

    for attraction_name in attraction_names:  # Add node attributes for images to the graph
        image_path = image_paths[attraction_name]
        image = Image.open(image_path)
        image = image.convert('RGB')  # Convert image to RGB format

        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)  # Increase the image size to desired dimensions (e.g., 200x200)
        G.nodes[attraction_name]['image'] = image


def perform_Agglomerative_Clustering():
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=DISTANCE_THRESHOLD)
    cluster_labels = clustering.fit_predict(attraction_matrix)
    return cluster_labels


def add_edges_to_graph(cluster_labels):  # Add edges to the graph based on cluster labels
    for i, attraction_name in enumerate(attraction_names):
        parent = attraction_names[cluster_labels[i]]
        G.add_edge(parent, attraction_name)


def adjust_plot_limits(pos):
    global x_min, x_max, y_min, y_max

    # Find the maximum and minimum x and y coordinates of the node positions
    x_max = max(pos.values(), key=lambda x: x[0])[0]
    x_min = min(pos.values(), key=lambda x: x[0])[0]
    y_max = max(pos.values(), key=lambda x: x[1])[1]
    y_min = min(pos.values(), key=lambda x: x[1])[1]

    ax.set_xlim(x_min - ZOOM_STEP, x_max + ZOOM_STEP)
    ax.set_ylim(y_min - ZOOM_STEP, y_max + ZOOM_STEP)


def create_figure_and_axes():
    global fig, ax

    fig, ax = plt.subplots(figsize=(14, 7), dpi=100, frameon=False)
    fig.tight_layout()


def draw_nodes(pos):  # Draw nodes as circles with attraction images
    for node in G.nodes:

        image = G.nodes[node]['image']
        x, y = pos[node]

        # Create a circular mask
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse((0, 0, image.size[0], image.size[1]), fill=255)

        # Apply the circular mask to the image
        image.putalpha(mask)

        # Create annotation box for the image
        imagebox = OffsetImage(image, zoom=INITIAL_ZOOM)
        #imagebox = OffsetImage(image, zoom=0.1)

        imagebox.set_zoom(INITIAL_ZOOM)
        ab = AnnotationBbox(imagebox, (x, y), xycoords='data', frameon=False)
        ax.add_artist(ab)
        # image.putalpha(mask)  # Apply the circular mask to the image

        G.nodes[node]['imagebox'] = imagebox  # Store the imagebox in the node's data

    fig.canvas.mpl_connect('button_press_event',
                           on_button_press)  # Connect the button press event to the callback function
    ax.axis('off')  # Hide axes and labels
    plt.show()  # Show the plot


def draw_edges_to_graph(pos):
    for edge in G.edges:
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        ax.plot([x1, x2], [y1, y2], '-', color='black', linewidth=0.3, alpha=0.5)


# ----------- TKINTER FUNCTIONS ----------- #

def create_figure_canvas():  # Create a FigureCanvasTkAgg object and display it in the tkinter window
    global canvas

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Set the margins to remove the black square
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()

    # Connect the zoom in and zoom out functions to mouse events
    canvas.mpl_connect('button_press_event', zoom_in)
    canvas.mpl_connect('scroll_event', zoom_out)

    # Bind left and right arrow keys to the corresponding functions
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    plt.show()

    canvas.get_tk_widget().pack()

    canvas.get_tk_widget().grid(row=0, column=0, sticky='news')


def create_reset_button():  # Create a tkinter Button for resetting the view
    reset_button = tk.Button(window, text="Reset View", command=reset_zoom_out)
    reset_button.grid(row=3, column=0, pady=10)

    canvas.draw()


def create_tkinter_window():
    global window

    window = tk.Tk()
    window.title('Attraction Tree Spring')
    window.configure(bg='white')
    window.geometry('2880x1880')  # todo: Adjust the dimensions of the window as desired: 2880x1880
    plt.show()  # Show the plot


def tkinter_func():
    create_tkinter_window()
    create_figure_canvas()
    create_reset_button()

    canvas.draw()  # Redraw the canvas with the updated plot limits


# ----------- DENDROGRAM FUNCTION ----------- #

def plot_dendrogram(graph):
    # Convert graph to distance matrix
    dist_matrix = nx.to_numpy_array(graph)

    # Perform hierarchical clustering
    linkage = hierarchy.linkage(dist_matrix)

    # Plot the dendrogram
    plt.figure()
    dendrogram = hierarchy.dendrogram(linkage)

    # Cut the dendrogram at a specific threshold (adjust as needed)
    threshold = 3
    clusters = hierarchy.fcluster(linkage, threshold, criterion='distance')

    # Highlight clusters in the dendrogram plot
    color_palette = plt.cm.get_cmap('tab10', len(set(clusters)))
    ax = plt.gca()
    for i, color in zip(range(1, len(set(clusters)) + 1), color_palette.colors):
        y = np.where(clusters == i)[0]
        x = np.ones_like(y) * ax.get_xlim()[1]
        plt.plot(x, y, 'o', color=color)

    plt.show()


# ----------- MAIN FUNCTIONS ----------- #

def main_func():
    global start_ylim, start_xlim, pos2
    G.add_nodes_from(attraction_names)  # Add nodes to the graph with attraction names as labels

    add_images_to_graph()
    cluster_labels = perform_Agglomerative_Clustering()
    num_clusters = max(cluster_labels)

    for i in range(num_clusters):
        cluster_attractions = [attraction_names[j] for j, cluster_label in enumerate(cluster_labels) if cluster_label == i]
        print(f"Cluster {i+1} Attractions:")
        vector = []
        for attraction in cluster_attractions:
            vector.append(attraction)
        print(vector)
        print()

    add_edges_to_graph(cluster_labels)

    # TODO: pick witch layout best visualize the attractions
    pos = nx.layout.spring_layout(G,scale=3, seed=42)  # Compute force-directed layout
    # pos = kamada_kawai_layout(G)    # Compute kamada_kawai_layout
    # pos = nx.shell_layout(G)  # Use the Level-Spaced Tree Layout
    # pos = nx.fruchterman_reingold_layout(G, seed=42)  # Use the Fruchterman-Reingold Layout
    # pos = nx.shell_layout(G)
    nx.draw_networkx(G, pos=pos, with_labels=False, node_size=0)
    pos2=pos

    create_figure_and_axes()
    adjust_plot_limits(pos)
    draw_nodes(pos)
    draw_edges_to_graph(pos)

    plt.title('Attraction Tree Spring')
    ax.set_title('Attraction Tree Spring')  # Set the title for the plot
    plot_dendrogram(G)  # Plot the dendrogram  # TODO: needed?

    start_xlim = ax.get_xlim()
    start_ylim = ax.get_ylim()

    tkinter_func()


main_func()
tk.mainloop()  # Start the tkinter event loop


