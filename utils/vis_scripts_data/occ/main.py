import json
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 6
})

def filter_data(task, threshold=50002):
    filtered_x = [x for x in task["x"] if x <= threshold]
    filtered_y = [task["y"][i] for i, x in enumerate(task["x"]) if x <= threshold]
    return {"name": task["name"], "x": filtered_x, "y": filtered_y, "type": task["type"], "task": task["task"]}

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def plot_data(ax, task_data, linestyle='-o'):
    for task_name, task_info in task_data.items():
        if "No" in task_name:
            linestyle='--o'
        else:
            linestyle='-o'
        ax.plot(task_info["x"], task_info["y"], linestyle, label=task_name, color=task_info["color"], linewidth = '0.5',markersize=2)

# Load data from the files
file_paths = [
    'val_imagine1_chamfer_distance _ val_imagine1_chamfer_distance.json', 
    'val_imagine1_psnr _ val_imagine1_psnr.json'
]

data_1 = [filter_data(task) for task in load_data(file_paths[0])]
data_2 = [filter_data(task) for task in load_data(file_paths[1])]

# Create a dictionary to store data for each task
task_data_1 = {}
task_data_2 = {}

# Define a list of colors to use for each task
colors = ['#117733', '#44AA99', '#882255', '#AA4499']

# Process filtered data
for i, (filtered_data, task_data) in enumerate(zip([data_1, data_2], [task_data_1, task_data_2])):
    for j, item in enumerate(filtered_data):
        print(i,j)
        task_name = item["task"]
        #task_name = task_names[i]
        color = colors[j % len(colors)]
        if task_name not in task_data:
            task_data[task_name] = {"x": [], "y": [], "name": item["name"], "color": color}

        task_data[task_name]["x"].extend(item["x"])
        task_data[task_name]["y"].extend(item["y"])

# Plot the data for each task in subplots
subplot_width = 1.6125  # 6.875 inches divided by 4 subplots

fig, axs = plt.subplots(1, 2, figsize=(3.25, subplot_width), gridspec_kw={'width_ratios': [subplot_width] * 2})
#fig.set_size_inches(w=6.875, h=1.5)

plot_data(axs[0], task_data_1)
plot_data(axs[1], task_data_2)

# Set titles and labels
axs[0].set_title("Chamfer Distance (Lidar) $\\downarrow$")
axs[1].set_title("PSNR (Camera) $\\uparrow")

for ax in axs:
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

# Only show legend in the top-left subplot
leg = axs[0].legend(fontsize=4,fancybox=True, loc="lower right", mode="expand", bbox_to_anchor=(0.48, 0.55, 0.5, 0.6))


# Save the plot
fig.tight_layout()
plt.savefig('occ_voxel_sensor_data.pgf')

