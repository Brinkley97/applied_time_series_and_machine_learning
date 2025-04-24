from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('get_chart/', views.get_chart, name='get_chart'),
]