"""Calliope URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from logo_creator import views as logo_creator_views
from accounts import views as acccounts_views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', acccounts_views.index_view, name='index'),
    path('signup/', acccounts_views.signup_view, name='signup'),
    path('login/', acccounts_views.login_view, name='login'),
    path('home/', logo_creator_views.home, name='home'),
    path('generate_logo/', logo_creator_views.generate_logo, name='generate_logo'),
    path('complement_image/', logo_creator_views.complement_image, name='complement_image'),
    path('about/', acccounts_views.about_view, name='about'),
    path('info/', acccounts_views.info_view, name='info'),
    path('contact/', acccounts_views.contact_us_view, name='contact'),
]
urlpatterns += staticfiles_urlpatterns()
