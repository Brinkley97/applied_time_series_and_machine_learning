from django.urls import path
from . import views

urlpatterns = [
    # get home function from views
    path('', views.home, name='blog-home'), # 127.0.0.1:8000/blog/
    path('about/', views.about, name='blog-about'), # 127.0.0.1:8000/blog/about/
]