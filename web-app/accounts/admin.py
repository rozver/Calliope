from django.contrib import admin
from .models import Profile, Logo

# Add the models to the admin panel
admin.site.register(Profile)
admin.site.register(Logo)
