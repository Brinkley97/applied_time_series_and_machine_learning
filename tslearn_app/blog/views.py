from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def home(request):
    """templates/blog/home.html"""
    return render(request, 'blog/home.html')

def about(request):
    """templates/blog/about.html"""
    return render(request, 'blog/about.html', {'title': 'About'})
