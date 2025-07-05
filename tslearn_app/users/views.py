from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import UserRegisterForm

# Create your views here.
def register(request):
    """templates/users/register.html"""
    if request.method == 'POST':
        new_user_form = UserRegisterForm(request.POST) # If it's a POST request, istantiate user creation form with data
        if new_user_form.is_valid():
            new_user_form.save()
            username = new_user_form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}!')
            return redirect('blog-home')
    else:
        new_user_form = UserRegisterForm()
    return render(request, 'users/register.html', {'form': new_user_form}) # http://127.0.0.1:8000/users/register.html