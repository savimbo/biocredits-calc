import pickle
import dash
from dash import dcc
from dash import html


# Load the figure from the pickle file
with open('slider_plot.pkl', 'rb') as file:
    fig = pickle.load(file)
fig.update_layout(transition_duration=0)

# Initialize your Dash app
app = dash.Dash(__name__)

# Use the figure in a Graph component within your app layout
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
