from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import AuthenticationForm
from .forms import SignUpForm

# Create your views here.


def index_view(request):
    return render(request, 'index.html')


def about_view(request):
    return render(request, 'about_us.html')


def info_view(request):
    return render(request, 'info.html')


def contact_us_view(request):
    return render(request, 'contact_us.html')


def signup_view(request):
    if request.method == 'POST':

        signup_form = SignUpForm(request.POST)

        if signup_form.is_valid():
            user = signup_form.save()
            user.refresh_from_db()
            user.email = signup_form.cleaned_data.get('email')
            user.save()

            raw_password = signup_form.cleaned_data.get('password1')
            user = authenticate(username=user.username, password=raw_password)
            login(request, user)

            return HttpResponseRedirect('/home/')

    else:
        signup_form = SignUpForm()

    return render(request, 'signup.html', {'form': signup_form})


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
