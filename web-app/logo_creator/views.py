from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from .complementary_colors import complement
import tensorflow as tf
import subprocess


def home(request):
    return render(request, 'home.html')


def generate_logo(request):
    subprocess.run(['python', './logo_creator/test_gan.py'], shell=False, timeout=1800)
    return HttpResponseRedirect('/home')


def complement_image(request):
    complement('static/output.png')
    return render(request, 'home.html')
