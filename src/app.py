import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash import dcc, html, Input, Output
import dash
import pandas as pd
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html
from dash_bootstrap_components._components.Container import Container
import sympy
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform, html
from dash.dependencies import Input, Output, State

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, "index.css"])
server = app.server

data = {'x': [],
        'y': [],
        'f1': [],
        'f2': [],
        'pareto': []}

# Create DataFrame
df = pd.DataFrame(data)

table = dbc.Table.from_dataframe(
    df, striped=False, bordered=True, hover=True, index=True
)


def turn_into_function(str):
    x, y = sympy.symbols('x y')
    z = sympy.parse_expr(str)
    f = sympy.lambdify((x, y), z, 'numpy')
    return f


def pareto_front_calc():
    global df
    for i in range(len(df)):
        df['pareto'][i] = 1
        for j in range(len(df)):
            if (df['f1'][i] > df['f1'][j] and df['f2'][i] >= df['f2'][j]):
                df['pareto'][i] = 0
    # print(df)

    return


function_1_str = 'x * y+3*sin(x)+cos(y)*4+x*3+y*2'
function_2_str = '(x-2)**2*0.2 + (y+1.5)**2*0.3'

func1 = turn_into_function(function_1_str)
func2 = turn_into_function(function_2_str)


# Define the range of values to plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z1 = func1(X, Y)
Z2 = func2(X, Y)

# Create the contour plot with both functions
fig = go.Figure()
fig.update_layout(showlegend=False, paper_bgcolor='white',
                  margin=dict(l=10, r=10, t=10, b=10), height=598,)


navbar = dbc.Navbar(
    [
        html.P('ParetoPlotter 3.0, created by Simon Wichmann', className="navbar-brand mx-auto",
               style={"color": "black"}),
    ],
    dark=True,
    color='#E8E8E8',
)

function_section = html.Div([
    dbc.Row([
        dbc.Col([
            html.P('Range in x:')
        ], width=2),
        dbc.Col([
            dcc.RangeSlider(-20, 20, 5,
                            value=[-5, 5], id='x-range-slider', className="custom-slider"),
        ], width=10)
    ]),
    dbc.Row([
        dbc.Col([
            html.P('Range in y:')
        ], width=2),
        dbc.Col([
            dcc.RangeSlider(-20, 20, 5,
                            value=[-5, 5], id='y-range-slider', className="custom-slider"),
        ], width=10)
    ]),
    dbc.Row([
        dbc.Col([
            html.P('Function 1:', style={'marginTop': '5px'})
        ], width=2),
        dbc.Col([
            dbc.Input(id='function_1_str',
                      placeholder='x*y+3*sin(x)+cos(y)*4+x*3+y*2', style={'width': '100%'})
        ], width=8),
        dbc.Col([
            dbc.Button('Plot', id='plot_function_1', style={'width': '100%'})
        ], width=2)
    ], style={'marginTop': '15px'}),
    dbc.Row([
        dbc.Col([
            html.P('Function 2:', style={'marginTop': '5px'})
        ], width=2),
        dbc.Col([
            dbc.Input(id='function_2_str',
                      placeholder='(x-2)**2*0.2+(y+1.5)**2*0.3', style={'width': '100%'})
        ], width=8),
        dbc.Col([
            dbc.Button('Plot', id='plot_function_2', style={'width': '100%'})
        ], width=2)
    ], style={'marginTop': '5px'}),
])

card1 = dbc.Card(
    dbc.CardBody([
        function_section
    ]),
    style={"width": "100%", "marginLeft": "0px",
           "marginRight": "px", "marginBottom": "20px"}
)

card1_5 = dbc.Card(
    dbc.CardBody([
        dbc.Tabs([
            dbc.Tab([dcc.Graph(figure=fig, id='2d_plot')], label="Top View",
                    # style={'height': '100vh'}
                    ),
            dbc.Tab([dcc.Graph(figure=fig, id='3d_plot')], label="3D View"),
            dbc.Tab([
                dbc.Row([
                    table,
                ], id='table_tab'),
                dbc.Row([
                    dbc.Button('Save as CSV', id='save_as_csv_button',
                               style={'width': '20%', 'marginRight': '10px'}),
                    dcc.Download(id='download-text')

                ]),
            ], label='Data Table', style={'marginLeft': '10', 'marginRight': '10', 'marginTop': '20', 'marginBottom': '20'})
        ], id='tabs', active_tab="tab-0"),
    ]),
    style={"width": "100%", "marginLeft": "0px",
           "marginRight": "0px", "marginBottom": "20px"}
)

card2 = dbc.Card(
    dbc.CardBody([
        dcc.Graph(figure=fig, id='pareto_plot')
    ]),
    style={"width": "100%", "marginLeft": "0px",
           "marginRight": "0px", "marginBottom": "20px"}
)

row = dbc.Row([
    dbc.Col([card1_5], width=6),
    dbc.Col([card1, card2], width=6)
], style={"marginLeft": "20px", "marginRight": "20px"})

app.layout = html.Div([
    navbar,
    row,
], style={'backgroundColor': '#E8E8E8'})


@app.callback(
    Output("2d_plot", "figure"),
    Output("3d_plot", "figure"),
    [Input("plot_function_1", "n_clicks"),
     Input("plot_function_2", "n_clicks"),
     State("x-range-slider", "value"),
     State("y-range-slider", "value"),
     State("function_1_str", "value"),
     State("function_2_str", "value"),
     ],
)
def plot_functions(func1, func2, x_range, y_range, function_1_str, function_2_str):
    # print(func1, func2, x_range, y_range, function_1_str, function_2_str)
    global df
    df = pd.DataFrame(columns=['x', 'y', 'f1', 'f2'])

    if function_1_str == None:
        function_1_str = 'x * y+3*sin(x)+cos(y)*4+x*3+y*2'

    if function_2_str == None:
        function_2_str = '(x-2)**2*0.2 + (y+1.5)**2*0.3'

    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)

    func1 = turn_into_function(function_1_str)
    func2 = turn_into_function(function_2_str)

    Z1 = func1(X, Y)
    Z2 = func2(X, Y)

    span_z_1 = np.max(Z1)-np.min(Z1)
    span_z_2 = np.max(Z2)-np.min(Z2)
    span_x = x_range[1] - x_range[0]
    span_y = y_range[1] - y_range[0]
    k = 4
    span_mean_1 = ((span_x+span_y)+span_z_1)/30
    span_mean_2 = ((span_x+span_y)+span_z_2)/30

    fig = go.Figure()
    fig.add_trace(go.Contour(x=x, y=y, z=Z1,
                             contours=dict(showlabels=True, start=Z1.min(
                             ), end=Z1.max(), size=span_mean_1, coloring='lines'),
                             line=dict(smoothing=1.3, width=2, color='black'), name='x*y', showscale=False))

    fig.add_trace(go.Contour(x=x, y=y, z=Z2,
                             contours=dict(showlabels=True, start=Z2.min(
                             ), end=Z2.max(), size=span_mean_2, coloring='lines'),
                             line=dict(smoothing=1.3, width=2, color='red'), name='x+y', showscale=False))

    fig.update_layout(showlegend=False, paper_bgcolor='white',
                      margin=dict(l=10, r=10, t=10, b=10), height=800,)

    fig.update_traces(selector=dict(name='x*y'), line_color='black')
    fig.update_traces(selector=dict(name='x+y'), line_color='red')

    fig_2 = go.Figure()

    fig_2.add_trace(go.Surface(x=x, y=y, z=Z1, name='x*y',
                    colorscale='Viridis', showscale=False))
    fig_2.add_trace(go.Surface(x=x, y=y, z=Z2, name='x+y',
                    colorscale='Viridis', showscale=False))

    # Update the layout
    fig_2.update_layout(scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
                        height=800, margin=dict(l=10, r=10, b=10, t=10))

    return fig, fig_2


@app.callback(
    [
        Output("pareto_plot", "figure"),
        Output("table_tab", "children"),

    ],
    [
        Input("2d_plot", "clickData"),
        Input("3d_plot", "clickData"),
        State("tabs", "active_tab"),
        State("function_1_str", "value"),
        State("function_2_str", "value"),
    ],
    prevent_initial_call=True
)
def plot_points(clickData_1, clickData_2, active_tab, function_1_str, function_2_str):
    if clickData_1 == None and clickData_2 == None:
        return fig, ()

    if active_tab == 'tab-1':
        xp = np.round(clickData_2['points'][0]['x'], 3)
        yp = np.round(clickData_2['points'][0]['y'], 3)
    elif active_tab == 'tab-0':
        xp = np.round(clickData_1['points'][0]['x'], 3)
        yp = np.round(clickData_1['points'][0]['y'], 3)

    if function_1_str == None:
        function_1_str = 'x * y+3*sin(x)+cos(y)*4+x*3+y*2'

    if function_2_str == None:
        function_2_str = '(x-2)**2*0.2 + (y+1.5)**2*0.3'

    func1 = turn_into_function(function_1_str)
    func2 = turn_into_function(function_2_str)

    new_row = {'x': xp, 'y': yp, 'f1': np.round(func1(xp, yp), 3),
               'f2': np.round(func2(xp, yp), 3), 'pareto': 0}

    global df
    df = df.append(new_row, ignore_index=True)

    scatter = go.Scatter(
        x=df['f1'],
        y=df['f2'],
        mode='markers',
        marker=dict(color='darkgray')
    )

    # Create figure object
    fig_5 = go.Figure(data=[scatter])
    fig_5.update_layout(showlegend=False, paper_bgcolor='white',
                        margin=dict(l=10, r=10, t=10, b=10), height=598,)

    if len(df) > 1:
        pareto_front_calc()
        pareto_set = df[df['pareto'] == True]
        pareto_set = pareto_set.sort_values(by='f1', ascending=True)
        pareto_set = pareto_set.sort_values(by='f2', ascending=False)

        if len(pareto_set) == 1:
            pareto_trace = go.Scatter(
                x=pareto_set['f1'], y=pareto_set['f2'], mode='markers', line=dict(color='#008B8B'))
            fig_5.add_trace(pareto_trace)
        else:
            pareto_trace = go.Scatter(
                x=pareto_set['f1'], y=pareto_set['f2'], mode='lines+markers', line=dict(color='#008B8B', width=3))
            fig_5.add_trace(pareto_trace)

    table = dbc.Table.from_dataframe(
        df, striped=False, bordered=True, hover=True, index=True
    )

    return [fig_5, table]


@app.callback(
    Output("download-text", "data"),
    Input("save_as_csv_button", "n_clicks"),
    prevent_initial_call=True
)
def save_as_csv(n_clicks):
    return dcc.send_data_frame(df.to_csv, 'pareto_set_download.csv')


if __name__ == '__main__':
    app.run_server(debug=True)
