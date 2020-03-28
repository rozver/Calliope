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
from .settings import MEDIA_ROOT, MEDIA_URL
from django.contrib import admin
from django.urls import path, re_path
from logo_creator import views as logo_creator_views
from accounts import views as acccounts_views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.contrib.staticfiles.urls import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', acccounts_views.index_view, name='index'),
    path('signup/', acccounts_views.signup_view, name='signup'),
    path('login/', acccounts_views.login_view, name='login'),
    path('logout/', acccounts_views.logout_view, name='logout'),
    path('home/', logo_creator_views.home, name='home'),
    path('generate_logo/', logo_creator_views.generate_logo, name='generate_logo'),
    path('complement_image/', logo_creator_views.complement_image, name='complement_image'),
    path('save_logo/', acccounts_views.save_logo, name='save_logo'),
    path('my_logos/', acccounts_views.show_saved_logos, name='saved_logos'),
    re_path(r'my_logos/delete/id=(?P<id>[0-9]+)$', acccounts_views.delete_logo, name='delete_logo')
]
urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(MEDIA_URL, document_root=MEDIA_ROOT)