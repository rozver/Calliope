from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from .forms import SignUpForm
from .models import Logo
import shutil
from django.utils import timezone
import os


# View for rendering index page
def index_view(request):
    user = request.user

    # If user has authenticated, redirect him to home page
    if bool(user.is_authenticated):
        return HttpResponseRedirect('home/')
    else:
        return render(request, 'index.html')


# View for signup
def signup_view(request):
    if request.method == 'POST':

        signup_form = SignUpForm(request.POST)

        if signup_form.is_valid():
            user = signup_form.save()
            user.refresh_from_db()

            # Additional email field
            user.email = signup_form.cleaned_data.get('email')
            user.save()

            raw_password = signup_form.cleaned_data.get('password1')
            user = authenticate(username=user.username, password=raw_password)
            login(request, user)

            return HttpResponseRedirect('/home/')

    else:
        signup_form = SignUpForm()

    return render(request, 'signup.html', {'form': signup_form})


# View for login
def login_view(request):
    if request.method == 'POST':

        login_form = AuthenticationForm(data=request.POST)

        if login_form.is_valid():
            username = login_form.cleaned_data.get('username')
            password = login_form.cleaned_data.get('password')

            user = authenticate(username=username, password=password)
            login(request, user)

            return HttpResponseRedirect('/home/')

    else:
        login_form = AuthenticationForm()

    return render(request, 'login.html', {'form': login_form})


# View for logout
@login_required(login_url='/login')
def logout_view(request):
    logout(request)
    return HttpResponseRedirect('/')


# View for showing user's saved logos
@login_required(login_url='/login')
def show_saved_logos(request):
    # Get the user logos by filtering all of them and getting those matching profile
    logos = Logo.objects.filter(profile=request.user.profile).all()
    for logo in logos:
        print(logo.image)
    return render(request, 'saved_logos.html', {'logos': logos})


# View for saving logo
def save_logo(request):
    # If user has not generated logo yet, create it's folder in /logos/
    if not os.path.exists('media/logos/' + str(request.user)):
        os.makedirs('media/logos/' + str(request.user))

    # The logo name is combination of username and current time in the same timezone
    logo_name = str(request.user) + '/' + str(timezone.now())

    # The current generated logo location is in /media/temporary logos
    logo_location = str('media/temporary_logos/' + str(request.user) + '_output.png')

    # Location to save the logo
    saved_logo_location = str('media/logos/' + logo_name + '.png')

    # Copy the logo
    shutil.copy(logo_location, saved_logo_location)

    # Create an object in Logo model
    Logo.objects.create(image=saved_logo_location[6:], profile=request.user.profile, name=logo_name)
    return HttpResponseRedirect('/home/')


# Delete a logo by id
def delete_logo(request, id):
    # Check if logo with that id exists
    if Logo.objects.filter(id=id).exists():
        logo = Logo.objects.get(id=id)
        location = logo.image

        # Check if the current user is the owner of the logo he wants to delete
        if request.user.profile == logo.profile:
            logo.delete()
            os.remove('media/' + str(location))

    return HttpResponseRedirect('/my_logos/')