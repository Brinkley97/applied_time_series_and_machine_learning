from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm # Use Django's base form to create a user

# Create your views here.
def register(request):
    new_user_form = UserCreationForm()
    return render(request, 'users/register.html', {'form': new_user_form})