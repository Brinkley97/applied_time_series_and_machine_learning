from django.shortcuts import render
from django.http import HttpResponse
from .models import Post

# Create your views here.
def home(request):
    """templates/blog/home.html"""
    context = {
        'posts': Post.objects.all()
    }
    return render(request, 'blog/home.html', context)

def about(request):
    """templates/blog/about.html"""
    return render(request, 'blog/about.html', {'title': 'About'})
