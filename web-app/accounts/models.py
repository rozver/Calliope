from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


# User profile model
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    def __str__(self):
        return str(self.user)


# Create user from the model
@receiver(post_save, sender=User)
def update_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
    instance.profile.save()


# Save user from the model
@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()


# Logo model - it has a foreign key to the profile it belongs to
class Logo(models.Model):
    name = models.CharField(max_length=100, blank=True, default='MyLogo')
    image = models.ImageField(upload_to='logos/', blank=True, default='default.png')
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE)

    def __str__(self):
        return str(self.name)
