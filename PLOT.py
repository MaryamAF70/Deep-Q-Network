import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

def draw_box(ax, text, xy, width=2.5, height=1, fc='lightblue', ec='black'):
    """Draws a rectangle with centered text."""
    rect = Rectangle(xy, width, height, facecolor=fc, edgecolor=ec, lw=1.5)
    ax.add_patch(rect)
    ax.text(xy[0] + width/2, xy[1] + height/2, text, 
            ha='center', va='center', fontsize=10, weight='bold')

def draw_arrow(ax, start, end, text=None):
    """Draws an arrow between two points."""
    arrow = FancyArrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                       width=0.02, head_width=0.2, head_length=0.2, length_includes_head=True, color='black')
    ax.add_patch(arrow)
    if text:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2 + 0.2, text,
                ha='center', va='center', fontsize=9)

# Create plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-2, 12)
ax.set_ylim(-5, 8)
ax.axis('off')

# Global Network
draw_box(ax, "Global Network\n(Shared Parameters)", (4, 5), width=4, height=1.5, fc='lightgreen')

# Agents (Workers)
agents_y = [2, 0, -2]
for i, y in enumerate(agents_y):
    draw_box(ax, f"Worker {i+1}\n(Environment + Local Net)", (0, y), width=4, height=1.5, fc='lightyellow')
    draw_arrow(ax, (4, y+0.75), (4, 5), text="Gradients")
    draw_arrow(ax, (8, 5), (4, y+0.75), text="Updated Params")

# Local flow inside a worker
draw_box(ax, "Environment", (-1.5, 2.2), width=3, height=1, fc='lightpink')
draw_box(ax, "Policy & Value\n(Local Actor-Critic)", (2, 2.2), width=3, height=1, fc='lightblue')
draw_arrow(ax, (1.5, 2.7), (2, 2.7))

plt.tight_layout()
plt.show()
