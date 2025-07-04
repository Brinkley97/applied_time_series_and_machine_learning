# stocks/views.py

import io
from django.shortcuts import render
from django.http import HttpResponse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from collections import namedtuple

# Your imports from the framework are now clean and will work correctly
from tslearn.data_loader import build_stock_uts
# We don't need the other framework imports in this file anymore

def _generate_plot_figure():
    """
    This helper function contains all your logic for creating the plot.
    It now returns the Matplotlib 'figure' object that your view needs.
    """
    start_date, end_date = "2010-01-04", "2020-02-07"
    Stock = namedtuple("Stock", ["symbol", "name"])
    stocks_to_load = [("000001.SS", "Shanghai Composite Index")]
    
    stocks_to_load = [Stock(*s) for s in stocks_to_load]
    
    stocks_data = {
        s.symbol: build_stock_uts(
            s.symbol, s.name, "Close", 
            start_date=start_date, end_date=end_date, frequency='1d'
        ) 
        for s in stocks_to_load
    }
    
    stock_of_interest = stocks_data['000001.SS']
    
    # CRUCIAL: Your '.plot()' method must return a Matplotlib Figure object.
    # If it returns an Axes object (ax), you would use 'fig = ax.figure' instead.
    fig = stock_of_interest.plot(tick_skip=75)
    
    return fig


# --- Your Django Views ---

def stock_chart_view(request):
    """This view renders the HTML page that will display the chart."""
    return render(request, 'stocks/stock_chart.html')


def plot_png_view(request):
    """
    This view now correctly generates the plot and returns it as a PNG image.
    """
    # 1. Call our helper function to get the generated figure.
    fig = _generate_plot_figure()

    # 2. Create an in-memory binary buffer.
    buf = io.BytesIO()
    
    # 3. Save the figure to the buffer in PNG format.
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)

    # 4. Create an HTTP response with the image data and the correct content type.
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    buf.close()
    
    # 5. Return the response. Django sends this back to the browser.
    return response
