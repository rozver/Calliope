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
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.contrib.staticfiles.urls import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', logo_creator_views.index, name='index'),
    path('generate_logo/', logo_creator_views.generate_logo, name='generate_logo'),
    path('complement_image/', logo_creator_views.complement_image, name='complement_image'),
    path('about/', logo_creator_views.about, name='about'),
    path('info/', logo_creator_views.info, name='info'),
    path('contact/', logo_creator_views.contact_us, name='contact'),
]
urlpatterns += staticfiles_urlpatterns()
