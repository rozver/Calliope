from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse
from django.contrib.auth.decorators import login_required
from .complementary_colors import complement
import subprocess


# View for rendering home page with updated logo
@login_required(login_url='/login')
def home(request):
    image_path = str('/media/temporary_logos/' + str(request.user) + '_output.png')
    print(image_path)
    return render(request, 'home.html', {'image_path': image_path})


# View for generating logo via subprocess and running external script
@login_required(login_url='/login')
def generate_logo(request):
    subprocess.run(['python', './logo_creator/test_gan.py', str(request.user)], shell=False, timeout=1800)
    return HttpResponseRedirect('/home')


# View for image complementing
@login_required(login_url='/login')
def complement_image(request):
    complement('media/temporary_logos/' + str(request.user) + '_output.png')
    return HttpResponseRedirect('/home')
