from django.core.exceptions import ValidationError
from django.contrib.auth.models import User


def validate_email(value):
    exists = User.objects.filter(email=value)
    if exists:
        raise ValidationError("This email has already been used!")
