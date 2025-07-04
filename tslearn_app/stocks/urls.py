# stocks/urls.py

from django.urls import path
from . import views

# This is important for namespacing URLs in templates
app_name = 'stocks'

urlpatterns = [
    # URL for the page that will SHOW the chart (e.g., /stocks/chart/)
    path('chart/', views.stock_chart_view, name='stock_chart'),
    
    # URL that will GENERATE and return the plot image
    path('plot.png', views.plot_png_view, name='plot_png'),
]